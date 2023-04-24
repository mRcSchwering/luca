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

    def __init__(self, k: float, n=1):
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


class LinearChange:
    """
    Linearly change from value `from_d` to value `to_d`
    dependent on value `v` over the course of `n_steps`.
    """

    def __init__(self, n_steps: float, from_d: float, to_d: float):
        self.n_steps = n_steps
        self.from_d = from_d
        self.to_d = to_d

    def __call__(self, v: float) -> float:
        if v >= self.n_steps:
            return self.to_d
        dn = (self.n_steps - v) / self.n_steps
        return dn * self.from_d + (1 - dn) * self.to_d


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
        split_thresh_mols: float,
        split_thresh_cells: float,
        init_genomes: list[str],
    ):
        self.world = world

        n_pxls = world.map_size**2
        self.n_adaption_gens = n_adaption_gens
        self.n_total_gens = n_adaption_gens + n_final_gens
        self.split_i = 0
        self.gen_i = 0.0
        self.score = 0.0

        molecules = self.world.chemistry.molecules
        self.mol_2_idx = {d.name: i for i, d in enumerate(molecules)}
        self.CO2_I = self.mol_2_idx["CO2"]
        self.X_I = self.mol_2_idx["X"]
        self.E_I = self.mol_2_idx["E"]

        self.mutation_rate_by_gen = LinearChange(
            n_steps=n_adaption_gens, from_d=1e-4, to_d=1e-6
        )
        self.mutation_rate = self.mutation_rate_by_gen(self.gen_i)

        self.replicate_by_mol = MoleculeDependentCellDivision(k=20.0)  # [15;30]
        self.kill_by_mol = MoleculeDependentCellDeath(k=0.04)  # [0.01;0.04]
        self.kill_by_genome = GenomeSizeDependentCellDeath(k=2_000.0)  # [2000;2500]

        mol_init = 10.0
        self.medium_fact = LinearComplexToMinimalMedium(
            n_gens=n_adaption_gens,
            mol_init=mol_init,
            molecules=molecules,
            essentials=ESSENTIAL_MOLS,
            molmap=self.world.molecule_map,
        )

        self.mol_thresh = mol_init * n_pxls * split_thresh_mols
        self.cell_thresh = int(n_pxls * split_thresh_cells)
        self.split_n = int(split_ratio * n_pxls)

        self._prepare_fresh_plate()
        self.world.add_cells(genomes=init_genomes)

    def step_1s(self):
        self.world.diffuse_molecules()
        self.world.degrade_molecules()

        for _ in range(10):
            self.world.enzymatic_activity()

        self._kill_cells()
        self.world.increment_cell_survival()
        self._replicate_cells()
        self._passage_cells()
        self._mutate_cells()

        avg = self.world.cell_divisions.float().mean().item()
        self.gen_i = 0.0 if math.isnan(avg) else avg
        self.mutation_rate = self.mutation_rate_by_gen(self.gen_i)

    def _passage_cells(self):
        if any(
            [
                self.world.molecule_map[self.E_I].sum().item() <= self.mol_thresh,
                self.world.molecule_map[self.CO2_I].sum().item() <= self.mol_thresh,
                self.world.n_cells >= self.cell_thresh,
            ]
        ):
            idxs = random.sample(range(self.world.n_cells), k=self.split_n)
            self.world.kill_cells(cell_idxs=list(set(idxs)))
            self._prepare_fresh_plate()
            self.world.reposition_cells(cell_idxs=list(range(self.world.n_cells)))
            self.split_i += 1

            self.score = max(
                (self.gen_i - self.n_adaption_gens) / self.n_total_gens, 0.0
            )

    def _mutate_cells(self):
        mutated = ms.point_mutations(seqs=self.world.genomes, p=self.mutation_rate)
        self.world.update_cells(genome_idx_pairs=mutated)

    def _replicate_cells(self):
        i = self.X_I
        idxs0 = self.replicate_by_mol(self.world.cell_molecules[:, i])
        idxs1 = torch.argwhere(self.world.cell_survival >= 10).flatten().tolist()
        idxs2 = torch.argwhere(self.world.cell_molecules[:, i] > 4.1).flatten().tolist()
        idxs = list(set(idxs0) & set(idxs1) & set(idxs2))

        successes = self.world.divide_cells(cell_idxs=idxs)
        if len(successes) == 0:
            return

        ps, cs = list(zip(*successes))
        self.world.cell_molecules[ps, i] -= 2.0
        self.world.cell_molecules[cs, i] -= 2.0

        pairs = [(self.world.genomes[p], self.world.genomes[c]) for p, c in successes]
        mutated = ms.recombinations(seq_pairs=pairs, p=self.mutation_rate)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = successes[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs)

    def _kill_cells(self):
        idxs0 = self.kill_by_mol(self.world.cell_molecules[:, self.E_I])
        idxs1 = self.kill_by_genome(self.world.genomes)
        idxs2 = torch.argwhere(self.world.cell_survival <= 3).flatten().tolist()
        self.world.kill_cells(cell_idxs=list(set(idxs0 + idxs1) - set(idxs2)))

    def _prepare_fresh_plate(self):
        self.world.molecule_map = self.medium_fact(self.gen_i)
        self.world.diffuse_molecules()
