from pathlib import Path
import torch
import magicsoup as ms
from .chemistry import CHEMISTRY

THIS_DIR = Path(__file__).parent

# TODO: init world mit maps und save
#       damit ich damit später verschiedene sims laufen lassen kann

# TODO: vllt mehrere enzyme steps before ein mutation step kommt

# TODO: labels für cells, die über Replikationen hinweg erhalten werden


def sigm_incr(t: torch.Tensor, k: float, n: int) -> list[int]:
    p = t**n / (t**n + k**n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def sigm_decr(t: torch.Tensor, k: float, n: int) -> list[int]:
    p = k**n / (t**n + k**n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


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

        self.n_pxls = map_size**2
        self.split_ratio = split_ratio
        self.split_at_n = int(split_thresh * self.n_pxls)
        self.max_splits = max_splits
        self.n_splits = 0

        self.mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
        self.CO2_I = self.mol_2_idx["CO2"]
        self.ATP_I = self.mol_2_idx["ATP"]
        self.ADP_I = self.mol_2_idx["ADP"]
        self.NADPH_I = self.mol_2_idx["NADPH"]
        self.NADP_I = self.mol_2_idx["NADP"]
        self.ACA_I = self.mol_2_idx["acetyl-CoA"]
        self.HCA_I = self.mol_2_idx["HS-CoA"]

        self._add_base_mols()

        # most cells will not be viable
        n_init_cells = int(self.n_pxls * 0.7)
        genomes = [ms.random_genome(init_genome_size) for _ in range(n_init_cells)]
        self.world.add_random_cells(genomes=genomes)

    def step(self):
        self._add_co2()
        self._add_energy()
        self._replicate_cells()
        self.world.enzymatic_activity()
        self.world.diffuse_molecules()
        self.world.degrade_molecules()
        self._kill_cells()
        self._split_cells()
        self._mutate_cells()
        self.world.increment_cell_survival()

    def _split_cells(self):
        if self.n_splits >= self.max_splits:
            return
        n_cells = len(self.world.cells)
        if n_cells > self.split_at_n:
            keep_n = int(n_cells * self.split_ratio)
            idxs = torch.randint(n_cells, (keep_n,)).tolist()
            self.world.kill_cells(cell_idxs=idxs)
            self._add_base_mols()
            self.n_splits += 1

    def _add_base_mols(self):
        # fresh molecule map
        self.world.molecule_map = 10.0

        # setup CO2 gradient
        device = self.world.device
        s = self.world.map_size
        n = int(s / 2)
        map_ones = torch.ones((s, s)).to(device)
        gradient = torch.cat(
            [
                torch.linspace(1.0, 100.0, n).to(device),
                torch.linspace(100.0, 1.0, n).to(device),
            ]
        )
        self.world.molecule_map[self.CO2_I] = torch.einsum(
            "xy,x->xy", map_ones, gradient
        )

        # create equilibrium
        for _ in range(10):
            self.world.diffuse_molecules()

    def _mutate_cells(self):
        mutated = ms.point_mutations(seqs=[d.genome for d in self.world.cells])
        self.world.update_cells(genome_idx_pairs=mutated)

    def _add_co2(self):
        n = int(self.world.map_size / 2)
        self.world.molecule_map[self.CO2_I, [n - 1, n]] = 100.0
        self.world.molecule_map[self.CO2_I, [0, -1]] = 1.0

    def _add_energy(self):
        for high, low in [(self.ATP_I, self.ADP_I), (self.NADPH_I, self.NADP_I)]:
            high_avg = self.world.molecule_map[high].mean()
            low_avg = self.world.molecule_map[low].mean()
            if high_avg / (low_avg + 1e-4) < 5.0:
                self.world.molecule_map[high] += self.world.molecule_map[low] * 0.99
                self.world.molecule_map[low] *= 0.01

    def _kill_cells(self):
        idxs0 = sigm_decr(self.world.cell_molecules[:, self.ATP_I], 0.5, 4)
        idxs1 = sigm_decr(self.world.cell_molecules[:, self.NADPH_I], 0.5, 4)
        idxs = list(set(idxs0 + idxs1))
        self.world.kill_cells(cell_idxs=idxs)

    def _replicate_cells(self):
        idxs1 = sigm_incr(self.world.cell_molecules[:, self.ACA_I], 15.0, 5)

        # successful cells will share their n molecules, which will then be reduced by 1.0
        # so a cell must have n >= 2.0 to replicate
        idxs2 = (
            torch.argwhere(self.world.cell_molecules[:, self.ACA_I] > 2.2)
            .flatten()
            .tolist()
        )
        idxs = list(set(idxs1) & set(idxs2))

        replicated = self.world.replicate_cells(parent_idxs=idxs)
        if len(replicated) == 0:
            return

        # these cells have successfully divided and shared their molecules
        parents, children = list(map(list, zip(*replicated)))
        self.world.cell_molecules[parents + children, self.ACA_I] -= 1.0
        self.world.cell_molecules[parents + children, self.HCA_I] += 1.0

        # add random recombinations
        genomes = [
            (self.world.cells[p].genome, self.world.cells[c].genome)
            for p, c in replicated
        ]
        mutated = ms.recombinations(seq_pairs=genomes)

        genome_idx_pairs = []
        for parent, child, idx in mutated:
            parent_i, child_i = replicated[idx]
            genome_idx_pairs.append((parent, parent_i))
            genome_idx_pairs.append((child, child_i))
        self.world.update_cells(genome_idx_pairs=genome_idx_pairs)
