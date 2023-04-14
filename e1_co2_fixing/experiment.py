from pathlib import Path
import random
import math
import torch
import magicsoup as ms
from .chemistry import ESSENTIAL_MOLS
from .util import sigm_sample, rev_sigm_sample

THIS_DIR = Path(__file__).parent


class MoleculeDependentCellDeath:
    """
    Sample cell indexes based on molecule abundance.
    Lower abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n=3):
        self.k = k
        self.n = n

    def __call__(self, cellmols: torch.Tensor) -> list[int]:
        return rev_sigm_sample(cellmols, self.k, self.n)


class GenomeSizeDependentCellDeath:
    """
    Sample cell indexes based on genome size.
    Larger genomes lead to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n=7):
        self.k = k
        self.n = n

    def __call__(self, genomes: list[str]) -> list[int]:
        sizes = torch.tensor([float(len(d)) for d in genomes])
        return sigm_sample(sizes, self.k, self.n)


class MoleculeDependentCellDivision:
    """
    Sample cell indexes based on molecule abundance.
    Higher abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n=3):
        self.k = k
        self.n = n

    def __call__(self, cellmols: torch.Tensor) -> list[int]:
        return sigm_sample(cellmols, self.k, self.n)


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
        if gen_i >= self.n_gens:
            return self.to_p
        dn = (self.n_gens - gen_i) / self.n_gens
        return dn * self.from_p + (1 - dn) * self.to_p


class IncrementWithLimit:
    """
    Increment molecule abundance by a constant value `val`
    up to a maximum value `limit`.
    """

    def __init__(self, val: float, limit=10.0):
        self.val = val
        self.limit = limit

    def __call__(self, molmap: torch.Tensor):
        molmap += self.val
        molmap -= (molmap - self.limit).clamp(min=0.0)


class KillCellsForSplit:
    """
    Sample cell indexes to be killed if the cell map is overgrown.
    Cell map has `n_pxls` pixels and cells are split if more than `thresh`
    of this map is overgrown with cells.
    Afterwards `ratio` of cells will survive.
    """

    def __init__(self, ratio: float, thresh: float, n_pxls: int):
        self.ratio = ratio
        self.split_at_n = int(n_pxls * thresh)

    def __call__(self, n_cells: int) -> list[int]:
        if n_cells > self.split_at_n:
            kill_n = n_cells - int(n_cells * self.ratio)
            return random.sample(range(n_cells), k=kill_n)
        return []


class LinearComplexToMinimalMedium:
    """
    Linearly move from complex medium to minimal medium over `n_gens` generations.
    Non-essential molecule species will be reduced to zero, essential ones
    will stay at `mol_init`.
    """

    def __init__(
        self,
        n_gens: float,
        mol_init: float,
        molecules: list[ms.Molecule],
        essentials: list[ms.Molecule],
        molmap: torch.Tensor,
    ):
        self.eps = 1e-5
        self.n_gens = n_gens
        self.mol_init = mol_init
        self.essentials = [i for i, d in enumerate(molecules) if d in essentials]
        self.others = [i for i, d in enumerate(molecules) if d not in essentials]
        self.molmap = molmap

    def __call__(self, gen_i: float) -> torch.Tensor:
        molmap = torch.zeros_like(self.molmap)
        molmap[self.essentials] = self.mol_init
        if gen_i >= self.n_gens:
            molmap[self.others] = self.eps
            return molmap
        dn = (self.n_gens - gen_i) / self.n_gens
        molmap[self.others] = dn * self.mol_init + self.eps
        return molmap


class Experiment:
    def __init__(
        self,
        world: ms.World,
        n_adaption_gens: float,
        n_final_gens: float,
        split_ratio: float,
        split_thresh: float,
        init_genomes: list[str],
    ):
        self.world = world

        self.n_pxls = world.map_size**2
        self.n_adaption_gens = n_adaption_gens
        self.n_total_gens = n_adaption_gens + n_final_gens
        self.split_i = 0
        self.gen_i = 0.0
        self.score = 0.0

        molecules = self.world.chemistry.molecules
        self.mol_2_idx = {d.name: i for i, d in enumerate(molecules)}
        self.CO2_I = self.mol_2_idx["CO2"]
        self.X_I = self.mol_2_idx["X"]
        self.Y_I = self.mol_2_idx["Y"]

        self.point_mutations_by_gen = LinearGenerationDepedentMutationRate(
            n_gens=n_adaption_gens, from_p=1e-4, to_p=1e-6
        )
        self.recombinations_by_gen = LinearGenerationDepedentMutationRate(
            n_gens=n_adaption_gens, from_p=1e-5, to_p=1e-7
        )
        self.point_mutation_rate = self.point_mutations_by_gen(self.gen_i)
        self.recombination_rate = self.recombinations_by_gen(self.gen_i)

        self.replicate_by_mol = MoleculeDependentCellDivision(k=20.0)  # [15;30]
        self.kill_by_mol = MoleculeDependentCellDeath(k=0.25)  # [0.2;0.4]
        self.kill_by_genome = GenomeSizeDependentCellDeath(k=2_250.0)  # [2000;2500]

        self.add_energy = IncrementWithLimit(val=1.0)
        self.add_co2 = IncrementWithLimit(val=1.0)
        self.kill_for_split = KillCellsForSplit(
            ratio=split_ratio, thresh=split_thresh, n_pxls=self.n_pxls
        )

        self.medium_fact = LinearComplexToMinimalMedium(
            n_gens=n_adaption_gens,
            mol_init=10.0,
            molecules=molecules,
            essentials=ESSENTIAL_MOLS,
            molmap=self.world.molecule_map,
        )

        self._prepare_fresh_plate()

        # TODO: rm
        print(
            "cell_survival",
            self.world.cell_survival.device,
            self.world.cell_survival.dtype,
        )
        print("world device", self.world.device)
        print("trying to expand by 10")
        t = self.world.cell_survival
        size = t.size()
        zeros = torch.zeros(10, *size[1:], dtype=t.dtype).to(self.world.device)
        t1 = torch.cat([t, zeros], dim=0)

        print("trying world add cells")
        self.world.add_cells(genomes=init_genomes)

    def step_10s(self):
        self._replicate_cells()
        self._kill_cells()
        self._passage_cells()
        self._mutate_cells()

        self.world.increment_cell_survival()
        avg = self.world.cell_divisions.float().mean().item()
        self.gen_i = 0.0 if math.isnan(avg) else avg
        self.point_mutation_rate = self.point_mutations_by_gen(self.gen_i)
        self.recombination_rate = self.recombinations_by_gen(self.gen_i)
        self.score = max((self.gen_i - self.n_adaption_gens) / self.n_total_gens, 0.0)

    def step_1s(self):
        self.add_energy(self.world.molecule_map[self.X_I])
        self.add_co2(self.world.molecule_map[self.CO2_I])
        self.world.diffuse_molecules()
        self.world.degrade_molecules()
        for _ in range(10):
            self.world.enzymatic_activity()

    def _passage_cells(self):
        idxs = self.kill_for_split(self.world.n_cells)
        if len(idxs) > 0:
            self.world.kill_cells(cell_idxs=idxs)
            self._prepare_fresh_plate()
            self.world.reposition_cells(cell_idxs=list(range(self.world.n_cells)))
            self.split_i += 1

    def _mutate_cells(self):
        mutated = ms.point_mutations(
            seqs=self.world.genomes, p=self.point_mutation_rate
        )
        self.world.update_cells(genome_idx_pairs=mutated)

    def _replicate_cells(self):
        i = self.X_I
        idxs0 = self.replicate_by_mol(self.world.cell_molecules[:, i])

        idxs1 = torch.argwhere(self.world.cell_molecules[:, i] > 2.2).flatten().tolist()
        idxs = list(set(idxs0) & set(idxs1))

        successes = self.world.divide_cells(cell_idxs=idxs)
        if len(successes) == 0:
            return

        ps, cs = list(zip(*successes))
        self.world.cell_molecules[ps, i] -= 1.0
        self.world.cell_molecules[cs, i] -= 1.0

        pairs = [(self.world.genomes[p], self.world.genomes[c]) for p, c in successes]
        mutated = ms.recombinations(seq_pairs=pairs, p=self.recombination_rate)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = successes[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs)

    def _kill_cells(self):
        idxs0 = self.kill_by_mol(self.world.cell_molecules[:, self.X_I])
        idxs1 = self.kill_by_genome(self.world.genomes)
        self.world.kill_cells(cell_idxs=list(set(idxs0 + idxs1)))

    def _prepare_fresh_plate(self):
        self.world.molecule_map = self.medium_fact(self.gen_i)
        self.add_energy(self.world.molecule_map[self.X_I])
        for _ in range(100):
            self.add_co2(self.world.molecule_map[self.CO2_I])
            self.world.diffuse_molecules()
