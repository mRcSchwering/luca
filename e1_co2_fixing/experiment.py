from pathlib import Path
import random
import math
import torch
import magicsoup as ms
from .util import sigm_sample, rev_sigm_sample

THIS_DIR = Path(__file__).parent


class MutationRateFact:
    """
    Factory for returning current mutation rates.
    """

    def __call__(self, exp: "Experiment") -> float:
        """Returns current mutation rates"""
        raise NotImplementedError


class MediumFact:
    """
    Factory for returning new medium.
    """

    additives_init: float  # initial additives concentration
    substrates_init: float  # initial substrates concentration

    def __call__(self, exp: "Experiment") -> torch.Tensor:
        """Returns a new molecule map"""
        raise NotImplementedError


class Passager:
    """
    Class defining when and how passages are done.
    """

    split_leftover: int

    def __call__(self, exp: "Experiment") -> bool:
        """Returns whether to passage cells now"""
        raise NotImplementedError


class CellSampler:
    """
    Class for sampling cell idxs.
    """

    def __call__(self, exp: "Experiment") -> list[int]:
        raise NotImplementedError


class MoleculeDependentCellDeath(CellSampler):
    """
    Sample cell indexes based on molecule abundance.
    Lower abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, mol_i: int, k: float, n: int):
        self.k = k
        self.n = n
        self.mol_i = mol_i

    def __call__(self, exp: "Experiment") -> list[int]:
        mols = exp.world.cell_molecules[:, self.mol_i]
        return rev_sigm_sample(mols, self.k, self.n)


class MoleculeDependentCellDivision(CellSampler):
    """
    Sample cell indexes based on molecule abundance.
    Higher abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, mol_i: int, k: float, n: int):
        self.k = k
        self.n = n
        self.mol_i = mol_i

    def __call__(self, exp: "Experiment") -> list[int]:
        mols = exp.world.cell_molecules[:, self.mol_i]
        return sigm_sample(mols, self.k, self.n)


class GenomeSizeController(CellSampler):
    """
    Sample cell indexes based on genome size.
    Larger genomes lead to higher probability of being sampled.
    """

    def __init__(
        self,
        k: float,
        n: int,
    ):
        self.k = k
        self.n = n

    def __call__(self, exp: "Experiment") -> list[int]:
        genome_lens = [len(d) for d in exp.world.genomes]
        sizes = torch.tensor(genome_lens)
        return sigm_sample(sizes, self.k, self.n)


class PassageByCells(Passager):
    """
    Passage cells if world too full.
    """

    def __init__(
        self,
        split_ratio: float,
        split_thresh: float,
        max_cells: int,
    ):
        self.max_cells = int(max_cells * split_thresh)
        self.split_leftover = int(split_ratio * max_cells)

    def __call__(self, exp: "Experiment") -> bool:
        return exp.world.n_cells >= self.max_cells


class GenomeEditor:
    """
    Editing cell genomes during the experiment at passage time.
    """

    def __call__(self, exp: "Experiment"):
        raise NotImplementedError


class Experiment:
    """
    Common experimental procedure.

    Parameters:
        world: Initialized and loaded world object on device
        lgt_rate: lateral gene transfer rate
        passage: scheme defining how and when to passage cells
        mutation_rate_fact: factory defining mutation rates
        medium_fact: factory for generating media
    """

    def __init__(
        self,
        world: ms.World,
        lgt_rate: float,
        passager: Passager,
        mutation_rate_fact: MutationRateFact,
        medium_fact: MediumFact,
        division_by_x: CellSampler,
        death_by_e: CellSampler,
        genome_size_controller: CellSampler,
        genome_editor: GenomeEditor | None = None,
    ):
        self.world = world
        self.score = 0.0
        self.split_i = 0
        self.step_i = 0
        self.cpd = 0.0
        self.growth_rate = 0.0
        self.progress = 0.0

        molecules = [d.name for d in self.world.chemistry.molecules]
        self.CO2_I = molecules.index("CO2")
        self.X_I = molecules.index("X")
        self.E_I = molecules.index("E")

        self.mutation_rate_fact = mutation_rate_fact
        self.mutation_rate = self.mutation_rate_fact(self)
        self.lgt_rate = lgt_rate

        self.division_by_x = division_by_x
        self.death_by_e = death_by_e
        self.genome_size_controller = genome_size_controller

        self.medium_fact = medium_fact
        self.passager = passager
        self.genome_editor = genome_editor

        self._prepare_fresh_plate()

        self._n0 = self.world.n_cells
        self._s0 = self.step_i

    def run(self, max_steps: int):
        for step_i in range(max_steps):
            self.step_i = step_i
            yield step_i

    def step_1s(self):
        self.world.diffuse_molecules()
        self.world.degrade_molecules()
        self.world.enzymatic_activity()

        self._kill_cells()
        self.world.increment_cell_survival()
        self._replicate_cells()

        # calculate growth
        n_steps = self.step_i - self._s0
        if n_steps > 0 and self._n0 > 0:
            self.growth_rate = math.log(self.world.n_cells / self._n0, 2) / n_steps

        if self.passager(self):
            self._passage_cells()

        self._mutate_cells()
        self._lateral_gene_transfer()

        self.mutation_rate = self.mutation_rate_fact(self)

    def _passage_cells(self):
        n_old = self.world.n_cells

        # calculate cumulative population doubling
        if self._n0 > 0:
            self.cpd += math.log(n_old / self._n0, 2)

        # split cells
        kill_n = max(n_old - self.passager.split_leftover, 0)
        idxs = random.sample(range(n_old), k=kill_n)
        self.world.kill_cells(cell_idxs=idxs)
        self.split_i += 1
        self._prepare_fresh_plate()
        n_new = self.world.n_cells
        self.world.reposition_cells(cell_idxs=list(range(n_new)))

        # edit genomes
        if self.genome_editor is not None:
            self.genome_editor(self)

        # set for passage-averaged growth calculations
        self._n0 = n_new
        self._s0 = self.step_i

    def _mutate_cells(self):
        mutated = ms.point_mutations(seqs=self.world.genomes, p=self.mutation_rate)
        self.world.update_cells(genome_idx_pairs=mutated, batch_size=500)

    def _lateral_gene_transfer(self):
        # sample cells according to LGT rate
        p = torch.full((self.world.n_cells,), self.lgt_rate)
        sample = torch.bernoulli(p).to(self.world.device).bool()

        # cells that havent replicated in a while are open to LGT
        old_cells = self.world.cell_survival >= 20
        idxs = torch.argwhere(old_cells & sample).flatten().tolist()
        nghbr_idxs = torch.argwhere(old_cells).flatten().tolist()

        nghbrs = self.world.get_neighbors(cell_idxs=idxs, nghbr_idxs=nghbr_idxs)
        pairs = [(self.world.genomes[a], self.world.genomes[b]) for a, b in nghbrs]
        mutated = ms.recombinations(seq_pairs=pairs, p=self.mutation_rate)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = nghbrs[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs, batch_size=500)

    def _replicate_cells(self):
        i = self.X_I

        # max mu will be every 10 steps, consumes 4X
        idxs0 = self.division_by_x(self)
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
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs, batch_size=500)

    def _kill_cells(self):
        idxs0 = self.death_by_e(self)
        idxs1 = self.genome_size_controller(self)
        idxs2 = torch.argwhere(self.world.cell_survival <= 3).flatten().tolist()
        self.world.kill_cells(cell_idxs=list(set(idxs0 + idxs1) - set(idxs2)))

    def _prepare_fresh_plate(self):
        self.world.molecule_map = self.medium_fact(self)
        self.world.diffuse_molecules()
