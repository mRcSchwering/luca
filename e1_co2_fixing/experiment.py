from pathlib import Path
import random
import torch
import magicsoup as ms
from .chemistry import ESSENTIAL_MOLS

THIS_DIR = Path(__file__).parent


def _sigm_sample(t: torch.Tensor, k: float, n: int) -> list[int]:
    p = t**n / (t**n + k**n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def _rev_sigm_sample(t: torch.Tensor, k: float, n: int) -> list[int]:
    p = k**n / (t**n + k**n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


class MoleculeDependentCellDeath:
    """
    Sample cell indexes based on molecule abundance.
    Lower abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n: int):
        self.k = k
        self.n = n

    def __call__(self, cellmols: torch.Tensor) -> list[int]:
        return _rev_sigm_sample(cellmols, self.k, self.n)


class GenomeSizeDependentCellDeath:
    """
    Sample cell indexes based on genome size.
    Larger genomes lead to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n: int):
        self.k = k
        self.n = n

    def __call__(self, genomes: list[str]) -> list[int]:
        sizes = torch.tensor([float(len(d)) for d in genomes])
        return _sigm_sample(sizes, self.k, self.n)


class MoleculeDependentCellDivision:
    """
    Sample cell indexes based on molecule abundance.
    Higher abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n: int):
        self.k = k
        self.n = n

    def __call__(self, cellmols: torch.Tensor) -> list[int]:
        return _sigm_sample(cellmols, self.k, self.n)


class LinearGenerationDepedentMutationRate:
    """
    Linearly change mutation rate from `from_p` to `to_p`
    over the course of `n_gens` generations
    """

    def __init__(self, n_gens: float, from_p: float, to_p: float):
        self.n_gens = n_gens
        self.from_p = from_p
        self.to_p = to_p

    def __call__(self, gen_i: float) -> float:
        if gen_i > self.n_gens:
            return self.to_p
        dn = (self.n_gens - gen_i) / self.n_gens
        return dn * self.from_p + (1 - dn) * self.to_p


class IncrementWithLimit:
    """
    Increment molecule abundance by a constant value `val`
    up to a maximum value `limit`.
    """

    def __init__(self, val: float, limit: float):
        self.val = val
        self.limit = limit

    def __call__(self, molmap: torch.Tensor):
        molmap += self.val
        molmap -= (molmap - self.limit).clamp(min=0.0)


class LinearComplexToMinimalMedium:
    """
    Linearly move from complex medium to minimal medium over `n_gens` generations.
    Non-essential molecule species will be reduced to zero, essential ones
    will stay at `mol_init`.
    """

    def __init__(
        self,
        n_gens: int,
        mol_init: float,
        molecules: list[ms.Molecule],
        essentials: list[ms.Molecule],
        molmap: torch.Tensor,
    ):
        self.n_gens = n_gens
        self.mol_init = mol_init
        self.essentials = [i for i, d in enumerate(molecules) if d in essentials]
        self.others = [i for i, d in enumerate(molecules) if d not in essentials]
        self.eps = 1e-3
        self.device = molmap.device
        self.size = molmap.size()

    def __call__(self, gen_i: int) -> torch.Tensor:
        molmap = torch.zeros(self.size, device=self.device)
        if gen_i > self.n_gens:
            molmap[:] = self.eps
            return molmap
        dn = (self.n_gens - gen_i) / self.n_gens
        molmap[self.essentials] = dn * self.mol_init + self.eps
        molmap[self.others] = self.eps
        return molmap


class Experiment:
    def __init__(
        self,
        world: ms.World,
        mol_map_init: float,
        init_genomes: list[str],
        init_cell_cover: float,
        split_ratio: float,
        split_thresh: float,
    ):
        self.world = world
        assert world.n_cells == 0

        self.n_pxls = world.map_size**2
        n_init_cells = int(init_cell_cover * self.n_pxls)
        self.split_ratio = split_ratio
        self.split_at_n = int(split_thresh * self.n_pxls)
        self.split_i = 0
        self.gen_i = 0.0

        molecules = self.world.chemistry.molecules
        self.mol_2_idx = {d.name: i for i, d in enumerate(molecules)}
        self.CO2_I = self.mol_2_idx["CO2"]
        self.X_I = self.mol_2_idx["X"]
        self.Y_I = self.mol_2_idx["Y"]

        self.mutations_by_gen = LinearGenerationDepedentMutationRate(
            n_gens=1000, from_p=1e-3, to_p=1e-6
        )
        self.mutation_rate = self.mutations_by_gen(self.gen_i)

        self.divide_by_mol = MoleculeDependentCellDivision(k=15.0, n=3)
        self.kill_by_mol = MoleculeDependentCellDeath(k=0.5, n=3)
        self.kill_by_genome = GenomeSizeDependentCellDeath(k=4000.0, n=7)

        self.increment_energy = IncrementWithLimit(val=1.0, limit=10.0)
        self.increment_co2 = IncrementWithLimit(val=1.0, limit=10.0)

        self.medium_fact = LinearComplexToMinimalMedium(
            n_gens=1000,
            mol_init=mol_map_init,
            molecules=molecules,
            essentials=ESSENTIAL_MOLS,
            molmap=self.world.molecule_map,
        )

        self._prepare_fresh_plate()
        seqs = random.choices(init_genomes, k=n_init_cells)
        self.world.add_cells(genomes=seqs)

    def step_10s(self):
        self._replicate_cells()
        self._kill_cells()
        self._split_cells()
        self._mutate_cells()

        self.world.increment_cell_survival()
        self.gen_i = self.world.cell_divisions.mean().item()
        self.mutation_rate = self.mutations_by_gen(self.gen_i)

    def step_1s(self):
        self.increment_energy(self.world.molecule_map[self.X_I])
        self.increment_energy(self.world.molecule_map[self.CO2_I])
        for _ in range(10):
            self.world.enzymatic_activity()
        self.world.diffuse_molecules()
        self.world.degrade_molecules()

    def _split_cells(self):
        n_cells = self.world.n_cells
        if n_cells > self.split_at_n:
            kill_n = n_cells - int(n_cells * self.split_ratio)
            idxs = torch.randint(n_cells, (kill_n,)).tolist()
            self.world.kill_cells(cell_idxs=idxs)
            self.split_i += 1
        self._prepare_fresh_plate()
        self.world.reposition_cells()  # TODO: implement

    def _mutate_cells(self):
        mutated = ms.point_mutations(seqs=self.world.genomes, p=self.mutation_rate)
        self.world.update_cells(genome_idx_pairs=mutated)

    def _replicate_cells(self):
        i = self.X_I
        idxs0 = self.divide_by_mol(self.world.cell_molecules[:, i])

        idxs1 = torch.argwhere(self.world.cell_molecules[:, i] > 2.2).flatten().tolist()
        idxs = list(set(idxs0) & set(idxs1))

        successes = self.world.divide_cells(cell_idxs=idxs)
        if len(successes) == 0:
            return

        ps, cs = list(zip(*successes))
        self.world.cell_molecules[ps, i] -= 1.0
        self.world.cell_molecules[cs, i] -= 1.0

        pairs = [(self.world.genomes[p], self.world.genomes[c]) for p, c in successes]
        mutated = ms.recombinations(seq_pairs=pairs, p=self.mutation_rate)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = successes[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs)

    def _kill_cells(self):
        idxs0 = self.kill_by_mol(self.world.cell_molecules[:, self.X_I])
        idxs1 = self.kill_by_genome(self.world.genomes)
        self.world.kill_cells(cell_idxs=set(idxs0 + idxs1))

    def _prepare_fresh_plate(self):
        self.world.molecule_map = self.medium_fact(gen_i=self.gen_i)
        self.increment_energy(self.world.molecule_map[self.X_I])
        self.increment_co2(self.world.molecule_map[self.CO2_I])
