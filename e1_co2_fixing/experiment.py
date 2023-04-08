from pathlib import Path
import random
import torch
import magicsoup as ms
from .chemistry import CHEMISTRY

THIS_DIR = Path(__file__).parent


def _sigm_incr(t: torch.Tensor, k: float, n: int) -> list[int]:
    p = t**n / (t**n + k**n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def _sigm_decr(t: torch.Tensor, k: float, n: int) -> list[int]:
    p = k**n / (t**n + k**n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def _get_high_co2_spots(map_size: int) -> tuple[list[int], list[int]]:
    init_ticks = list(range(32, map_size + 1 - 32, 64))
    ticks = []
    for tick in init_ticks:
        ticks.extend([tick - 1, tick, tick + 1])
    n = len(ticks)
    xs = ticks * n
    ys = [d for d in ticks for _ in range(n)]
    return xs, ys


class LinearMutationRate:
    """
    Linearly change mutation rate from `from_p` to `to_p`
    over the course of `n_gens` generations
    """

    def __init__(self, n_gens=1000, from_p=1e-3, to_p=1e-6):
        self.n_gens = n_gens
        self.from_p = from_p
        self.to_p = to_p

    def __call__(self, gen_i: int) -> float:
        if gen_i > self.n_gens:
            return self.to_p
        dn = self.n_gens - gen_i
        return (dn * self.from_p + (1 - dn) * self.to_p) / 2


class CO2IslandsFact:
    def __init__(self, map_size: int):
        self.mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
        self.co2_i = self.mol_2_idx["CO2"]

        self.co2_xs, self.co2_ys = _get_high_co2_spots(map_size=map_size)
        self.co2_lows = list(range(0, map_size, 64))

    def __call__(self, molmap: torch.Tensor):
        molmap[self.co2_i, self.co2_xs, self.co2_ys] = 100.0
        molmap[self.co2_i, self.co2_lows] = 10.0
        molmap[self.co2_i, :, self.co2_lows] = 10.0


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

        self.init_genomes = init_genomes
        self.mol_map_init = mol_map_init
        self.n_pxls = world.map_size**2
        self.n_init_cells = int(init_cell_cover * self.n_pxls)
        self.split_ratio = split_ratio
        self.split_at_n = int(split_thresh * self.n_pxls)
        self.split_i = 0
        self.gen_i = 0.0

        self.mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
        self.CO2_I = self.mol_2_idx["CO2"]
        self.X_I = self.mol_2_idx["X"]
        self.Y_I = self.mol_2_idx["Y"]

        # TODO: injection
        self.mutation_rate = LinearMutationRate()
        self.co2_fact = CO2IslandsFact(map_size=world.map_size)

    def prep_world(self):
        self.world.kill_cells(cell_idxs=list(range(self.world.n_cells)))

        # fresh molecule map
        self.world.molecule_map[:] = self.mol_map_init
        # TODO: class for this too

        # setup CO2 gradient
        co2_i = self.mol_2_idx["CO2"]
        self.world.molecule_map[co2_i] = 35.0
        for _ in range(500):
            self.co2_fact(self.world.molecule_map)
            self.world.diffuse_molecules()

        # initial cells
        seqs = random.choices(self.init_genomes, k=self.n_init_cells)
        self.world.add_cells(genomes=seqs)

    def step_10s(self):
        self._replicate_cells()
        self._kill_cells()
        self._split_cells()
        self._mutate_cells()
        self.world.increment_cell_survival()

    def step_1s(self):
        self.co2_fact(self.world.molecule_map)
        self._add_energy()
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
        # TODO: reset molecule_map
        # TODO: randomly place remaining cells?

    def _mutate_cells(self):
        p = self.mutation_rate(self.gen_i)
        mutated = ms.point_mutations(seqs=self.world.genomes, p=p)
        self.world.update_cells(genome_idx_pairs=mutated)

    def _add_energy(self):
        i = self.Y_I
        self.world.molecule_map[i] += 1.0
        self.world.molecule_map[i] = self.world.molecule_map[i].clamp(max=10.0)

    def _kill_cells(self):
        ies = _sigm_decr(self.world.cell_molecules[:, self.Y_I], 0.5, 3)
        sizes = torch.tensor([float(len(d)) for d in self.world.genomes])
        iss = _sigm_incr(sizes, 4000.0, 7)
        self.world.kill_cells(cell_idxs=list(set(ies + iss)))

    def _replicate_cells(self):
        i = self.X_I
        ics = _sigm_incr(self.world.cell_molecules[:, i], 15.0, 3)

        # cell divisions will use up 2 X
        its = torch.argwhere(self.world.cell_molecules[:, i] > 2.2).flatten().tolist()
        idxs = list(set(ics) & set(its))
        self.world.cell_molecules[idxs, i] -= 2.0

        successes = self.world.divide_cells(cell_idxs=idxs)
        if len(successes) == 0:
            return

        # add random recombinations
        p = self.mutation_rate(self.gen_i)
        pairs = [(self.world.genomes[p], self.world.genomes[c]) for p, c in successes]
        mutated = ms.recombinations(seq_pairs=pairs, p=p)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = successes[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs)
