from pathlib import Path
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


def _get_co2_spots(world: ms.World) -> tuple[list[int], list[int]]:
    s = world.map_size
    ticks = list(range(0, s, 32))[1:]
    n = len(ticks)
    xs = ticks * n
    ys = [d for d in ticks for _ in range(n)]
    return xs, ys


class Experiment:
    def __init__(
        self,
        map_size: int,
        init_genome_size: int,
        split_ratio: float,
        split_thresh: float,
        max_splits: int,
        device: str,
        n_workers: int,
    ):
        self.world = ms.World(
            chemistry=CHEMISTRY,
            map_size=map_size,
            mol_map_init="zeros",
            device=device,
            workers=n_workers,
        )

        self.init_genome_size = init_genome_size
        self.n_pxls = map_size**2
        self.split_ratio = split_ratio
        self.split_at_n = int(split_thresh * self.n_pxls)
        self.max_splits = max_splits
        self.split_i = 0

        self.mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
        self.CO2_I = self.mol_2_idx["CO2"]
        self.X_I = self.mol_2_idx["X"]
        self.Y_I = self.mol_2_idx["Y"]

        self.co2_xs, self.co2_ys = _get_co2_spots(world=self.world)

    def prep_world(self):
        self.world.kill_cells(cell_idxs=[d.idx for d in self.world.cells])

        self._add_base_mols()

        n_init_cells = int(self.n_pxls * 0.5)
        s = self.init_genome_size
        genomes = [ms.random_genome(s) for _ in range(n_init_cells)]
        self.world.add_random_cells(genomes=genomes)

    def step_10s(self):
        self._replicate_cells()
        self._kill_cells()
        self._split_cells()
        self._mutate_cells()
        self.world.increment_cell_survival()

    def step_1s(self):
        self._add_co2()
        self._add_energy()
        for _ in range(10):
            self.world.enzymatic_activity()
        self.world.diffuse_molecules()
        self.world.degrade_molecules()

    def _split_cells(self):
        if self.split_i >= self.max_splits:
            return
        n_cells = len(self.world.cells)
        if n_cells > self.split_at_n:
            kill_n = n_cells - int(n_cells * self.split_ratio)
            idxs = torch.randint(n_cells, (kill_n,)).tolist()
            self.world.kill_cells(cell_idxs=idxs)
            self._add_base_mols()
            self.split_i += 1

    def _add_base_mols(self):
        # fresh molecule map
        self.world.molecule_map[:] = 10.0

        # setup CO2 gradient
        inner = slice(min(self.co2_xs), max(self.co2_xs))
        outer = slice(min(self.co2_xs) - 15, max(self.co2_xs) + 15)
        self.world.molecule_map[self.CO2_I] = 20.0
        self.world.molecule_map[self.CO2_I, outer, outer] = 40.0
        self.world.molecule_map[self.CO2_I, inner, inner] = 60.0

        # create equilibrium
        for _ in range(500):
            self._add_co2()
            self.world.diffuse_molecules()

    def _mutate_cells(self):
        mutated = ms.point_mutations(seqs=[d.genome for d in self.world.cells])
        self.world.update_cells(genome_idx_pairs=mutated)

    def _add_co2(self):
        self.world.molecule_map[self.CO2_I, self.co2_xs, self.co2_ys] = 100.0
        self.world.molecule_map[self.CO2_I, [0, -1]] = 1.0
        self.world.molecule_map[self.CO2_I, :, [0, -1]] = 1.0

    def _add_energy(self):
        i = self.Y_I
        self.world.molecule_map[i] += 1.0
        self.world.molecule_map[i] = self.world.molecule_map[i].clamp(max=10.0)

    def _kill_cells(self):
        ies = _sigm_decr(self.world.cell_molecules[:, self.Y_I], 0.5, 3)
        sizes = torch.tensor([float(len(d.genome)) for d in self.world.cells])
        iss = _sigm_incr(sizes, 4000.0, 7)
        self.world.kill_cells(cell_idxs=list(set(ies + iss)))

    def _replicate_cells(self):
        i = self.X_I
        ics = _sigm_incr(self.world.cell_molecules[:, i], 15.0, 3)

        # cell divisions will use up 2 X
        its = torch.argwhere(self.world.cell_molecules[:, i] > 2.2).flatten().tolist()
        idxs = list(set(ics) & set(its))
        self.world.cell_molecules[idxs, i] -= 2.0

        replicated = self.world.replicate_cells(parent_idxs=idxs)
        if len(replicated) == 0:
            return

        # add random recombinations
        genome_pairs = [
            (self.world.cells[p].genome, self.world.cells[c].genome)
            for p, c in replicated
        ]
        mutated = ms.recombinations(seq_pairs=genome_pairs)

        genome_idx_pairs = []
        for parent, child, idx in mutated:
            parent_i, child_i = replicated[idx]
            genome_idx_pairs.append((parent, parent_i))
            genome_idx_pairs.append((child, child_i))
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs)
