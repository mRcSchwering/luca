import time
import random
import torch
import magicsoup as ms
from .util import sigm, rev_sigm
from .culture import Culture, BatchCulture


class GenomeEditor:
    """At progress give cells genes from genome factory"""

    def __init__(self, at_progress: float, fact: ms.GenomeFact):
        self.is_edited = False
        self.fact = fact
        self.at_progress = at_progress

    def __call__(self, cltr: Culture):
        if not self.is_edited and cltr.progress >= self.at_progress:
            updates = [
                (cltr.world.cell_genomes[d] + self.fact.generate(), d)
                for d in range(cltr.world.n_cells)
            ]
            cltr.world.update_cells(genome_idx_pairs=updates)
            self.is_edited = True


class MediumRefresher:
    """Set substrate and additive concentrations in medium"""

    def __init__(
        self,
        world: ms.World,
        substrates: list[ms.Molecule],
        additives: list[ms.Molecule] | None = None,
        substrates_val=1.0,
        additives_val=1.0,
    ):
        if additives is None:
            additives = []
        self.subs_idxs = [world.chemistry.mol_2_idx[d] for d in substrates]
        self.add_idxs = [world.chemistry.mol_2_idx[d] for d in additives]
        self.subs_val = substrates_val
        self.add_val = additives_val
        self.other_idxs = list(
            set(world.chemistry.mol_2_idx.values())
            - set(self.subs_idxs)
            - set(self.add_idxs)
        )

    def __call__(self, cltr: Culture):
        cltr.world.molecule_map[self.other_idxs] = 0.0
        cltr.world.molecule_map[self.subs_idxs] = self.subs_val
        cltr.world.molecule_map[self.add_idxs] = self.add_val


class Passager:
    """Passage cells between min and max confluency"""

    def __init__(self, world: ms.World, cnfls=(0.2, 0.7)):
        n_max = world.map_size**2
        self.min_cells = int(n_max * min(cnfls))
        self.max_cells = int(n_max * max(cnfls))

    def __call__(self, cltr: BatchCulture) -> bool:
        if cltr.world.n_cells < self.max_cells:
            return False

        n_old = cltr.world.n_cells
        kill_n = max(n_old - self.min_cells, 0)
        idxs = random.sample(range(n_old), k=kill_n)
        cltr.world.kill_cells(cell_idxs=idxs)
        cltr.world.reposition_cells()
        return True


class Progressor:
    """Advance progress by average cell divisions"""

    def __init__(self, n_avg_divisions=100.0):
        self.n_avg_divisions = n_avg_divisions

    def __call__(self, cltr: Culture) -> float:
        mean_divis = cltr.world.cell_divisions.float().mean()
        return min(1.0, mean_divis.item() / self.n_avg_divisions)


class Mutator:
    """Mutate cells and recombinate cells"""

    def __init__(self, snp_p=1e-6, lgt_p=1e-7, lgt_rate=0.1):
        self.snp_p = snp_p
        self.lgt_p = lgt_p
        self.lgt_rate = lgt_rate

    def __call__(self, cltr: Culture):
        cltr.world.mutate_cells(p=self.snp_p)
        n_cells = cltr.world.n_cells
        idxs = random.sample(range(n_cells), k=int(n_cells * self.lgt_rate))
        cltr.world.recombinate_cells(cell_idxs=idxs, p=self.lgt_p)


class Replicator:
    """Replicate cells for high molecule concentration"""

    def __init__(
        self,
        world: ms.World,
        mol: ms.Molecule,
        k_x=30.0,
        n_x=3,
        min_x=4.0,
        min_lifetime=10,
    ):
        self.k_x = k_x
        self.n_x = n_x
        self.mol_i = world.chemistry.mol_2_idx[mol]
        self.min_x = min_x + 0.001
        self.half_x = min_x / 2
        self.min_lifetime = min_lifetime

    def __call__(self, cltr: Culture):
        t = cltr.world.cell_molecules[:, self.mol_i]
        x_sample = sigm(t=t, k=self.k_x, n=self.n_x)
        has_x = t >= self.min_x
        is_old = cltr.world.cell_lifetimes >= self.min_lifetime
        idxs = torch.argwhere(has_x & is_old & x_sample).flatten().tolist()
        successes = cltr.world.divide_cells(cell_idxs=idxs)
        if len(successes) == 0:
            return

        ps, cs = list(zip(*successes))
        cltr.world.cell_molecules[ps, self.mol_i] -= self.half_x
        cltr.world.cell_molecules[cs, self.mol_i] -= self.half_x


class Killer:
    """Kill cells for low molecule concentration and high genome size"""

    def __init__(
        self,
        world: ms.World,
        mol: ms.Molecule,
        k_x=0.04,
        n_x=1,
        k_g=2_000.0,
        n_g=7,
        spare_age=3,
        max_g_size=4_000,
    ):
        self.k_x = k_x
        self.n_x = n_x
        self.mol_i = world.chemistry.mol_2_idx[mol]
        self.k_g = k_g
        self.n_g = n_g
        self.spare_age = spare_age
        self.max_g_size = max_g_size

    def __call__(self, cltr: Culture):
        device = cltr.world.device
        x = cltr.world.cell_molecules[:, self.mol_i]
        g = torch.tensor([len(d) for d in cltr.world.cell_genomes], device=device)
        x_sample = rev_sigm(t=x, k=self.k_x, n=self.n_x)
        g_sample = sigm(t=g.float(), k=self.k_g, n=self.n_g)
        is_old = cltr.world.cell_lifetimes <= self.spare_age
        is_too_big = g > self.max_g_size
        mask = (x_sample & g_sample & is_old) | is_too_big
        idxs = torch.argwhere(mask).flatten().tolist()
        cltr.world.kill_cells(cell_idxs=idxs)


class Stopper:
    """Stop iteration on different conditions"""

    def __init__(
        self, max_steps=100_000, max_time_m=180, max_progress=1.0, min_cells=100
    ):
        self.max_steps = max_steps
        self.max_time_s = max_time_m * 60
        self.start_time = time.time()
        self.max_progress = max_progress
        self.min_cells = min_cells

    def __call__(self, cltr: Culture):
        if cltr.step_i >= self.max_steps:
            print(f"Maximum steps {self.max_steps:,} reached")
            raise StopIteration
        if time.time() - self.start_time >= self.max_time_s:
            print(f"Maximum time of {int(self.max_time_s/60):,}m reached")
            raise StopIteration
        if cltr.progress >= self.max_progress:
            print(f"Maximum progress of {self.max_progress:.2f} reached")
            raise StopIteration
        if cltr.world.n_cells <= self.min_cells:
            print(f"Minimum number of cells {self.min_cells:,} reached")
            raise StopIteration
