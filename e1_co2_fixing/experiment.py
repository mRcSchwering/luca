from pathlib import Path
import random
import math
import torch
import magicsoup as ms
from .util import (
    Finished,
    sigm_sample,
    rev_sigm_sample,
    batch_update_cells,
)

THIS_DIR = Path(__file__).parent


class MutationRateFact:
    """
    Factory for returning current mutation rates.
    """

    def __call__(self, exp: "Experiment") -> float:
        """Returns current mutation rates"""
        raise NotImplementedError


class GenomeFact:
    """
    Factory for generating genes that will be added every phase
    to the surviving cells
    """

    def __call__(self, exp: "Experiment") -> str:
        """Returns a new gene/genome for each cell"""
        raise NotImplementedError


class MediumFact:
    """
    Factory for returning new medium.
    """

    essentials_max: float  # maximum essentials concentration
    substrates_max: float  # maximum substrates concentration
    substrates: list[str]  # list of substrates
    essentials: list[str]  # list of essentials

    def __call__(self, exp: "Experiment") -> torch.Tensor:
        """Returns a new molecule map"""
        raise NotImplementedError


class Passage:
    """
    Class defining when and how passages are done.
    """

    split_leftover: int

    def __call__(self, exp: "Experiment") -> bool:
        """Returns whether to passage cells now"""
        raise NotImplementedError


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


class Experiment:
    """
    Common experimental procedure.

    Parameters:
        world: Initialized and loaded world object on device
        n_phases: Number of experimental phases
        n_phase_gens: Number of generations cells grow to finish a phase
        mol_divide_k: k for X-dependent cell division (should be [15;30])
        mol_kill_k: k for E-dependent cell death (should be [0.01;0.04])
        genome_kill_k: k for genome-size-dependent cell death (should be [2000;2500])
        lgt_rate: lateral gene transfer rate
        passage: scheme defining how and when to passage cells
        mutation_rate_fact: factory defining mutation rates
        medium_fact: factory for generating media
        genome_fact: factory for editing genomes every phase
    """

    def __init__(
        self,
        world: ms.World,
        n_phases: int,
        n_phase_gens: float,
        mol_divide_k: float,
        mol_kill_k: float,
        genome_kill_k: float,
        lgt_rate: float,
        passage: Passage,
        mutation_rate_fact: MutationRateFact,
        medium_fact: MediumFact,
        genome_fact: GenomeFact | None = None,
    ):
        self.world = world
        self.n_phases = n_phases
        self.n_phase_gens = n_phase_gens
        self.total_gens = self.n_phases * self.n_phase_gens
        self.score = 0.0
        self.phase_i = 0
        self.split_i = 0
        self.gen_i = 0.0
        self.growth_rate = 0.0

        molecules = [d.name for d in self.world.chemistry.molecules]
        self.CO2_I = molecules.index("CO2")
        self.X_I = molecules.index("X")
        self.E_I = molecules.index("E")

        self.mutation_rate_fact = mutation_rate_fact
        self.mutation_rate = self.mutation_rate_fact(self)
        self.lgt_rate = lgt_rate

        self.division_by_x = MoleculeDependentCellDivision(k=mol_divide_k)
        self.death_by_e = MoleculeDependentCellDeath(k=mol_kill_k)
        self.death_by_genome = GenomeSizeDependentCellDeath(k=genome_kill_k)

        self.genome_fact = genome_fact
        self.medium_fact = medium_fact
        self.passage = passage

        self._prepare_fresh_plate()

    def step_1s(self):
        n0 = self.world.n_cells

        self.world.diffuse_molecules()
        self.world.degrade_molecules()
        self.world.enzymatic_activity()

        self._kill_cells()
        self.world.increment_cell_survival()
        self._replicate_cells()

        n1 = self.world.n_cells
        self.growth_rate = math.log(n1 / n0) if n0 > 0 else float("nan")

        self._passage_cells()
        self._mutate_cells()
        self._lateral_gene_transfer()

        avg = self.world.cell_divisions.float().mean().item()
        self.gen_i = 0.0 if math.isnan(avg) else avg
        self.mutation_rate = self.mutation_rate_fact(self)

        # note, in any phase cells might grow for many generations, eventually dying,
        # advancing generations without split
        gen = self.phase_i * self.n_phase_gens + min(self.gen_i, self.n_phase_gens)
        self.score = min(max(gen / self.total_gens, 0.0), 1.0)

    def _next_phase(self) -> bool:
        if self.gen_i >= self.n_phase_gens:
            self.gen_i = 0.0
            self.world.cell_divisions[:] = 0
            self.phase_i += 1
            if self.phase_i > self.n_phases:
                raise Finished
            return True
        return False

    def _passage_cells(self):
        n_cells = self.world.n_cells
        if self.passage(self):
            kill_n = max(n_cells - self.passage.split_leftover, 0)
            idxs = random.sample(range(n_cells), k=kill_n)
            self.world.kill_cells(cell_idxs=idxs)
            if self._next_phase():
                self._edit_genomes()
            self._prepare_fresh_plate()
            self.world.reposition_cells(cell_idxs=list(range(self.world.n_cells)))
            self.split_i += 1

    def _mutate_cells(self):
        mutated = ms.point_mutations(seqs=self.world.genomes, p=self.mutation_rate)
        batch_update_cells(world=self.world, genome_idx_pairs=mutated)

    def _edit_genomes(self):
        if self.genome_fact is not None:
            genome_idx_pairs = []
            for idx, old_genome in enumerate(self.world.genomes):
                genes = self.genome_fact(self)
                genome_idx_pairs.append((old_genome + genes, idx))
            batch_update_cells(world=self.world, genome_idx_pairs=genome_idx_pairs)

    def _lateral_gene_transfer(self):
        # if cell can't replicate for a while it is open to LGT
        p = torch.full((self.world.n_cells,), self.lgt_rate)
        sample = torch.bernoulli(p).to(self.world.device).bool()
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
        batch_update_cells(world=self.world, genome_idx_pairs=genome_idx_pairs)

    def _replicate_cells(self):
        i = self.X_I

        # max mu will be every 10 steps, consumes 4X
        idxs0 = self.division_by_x(self.world.cell_molecules[:, i])
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
        batch_update_cells(world=self.world, genome_idx_pairs=genome_idx_pairs)

    def _kill_cells(self):
        idxs0 = self.death_by_e(self.world.cell_molecules[:, self.E_I])
        idxs1 = self.death_by_genome(self.world.genomes)
        idxs2 = torch.argwhere(self.world.cell_survival <= 3).flatten().tolist()
        self.world.kill_cells(cell_idxs=list(set(idxs0 + idxs1) - set(idxs2)))

    def _prepare_fresh_plate(self):
        self.world.molecule_map = self.medium_fact(self)
        self.world.diffuse_molecules()
