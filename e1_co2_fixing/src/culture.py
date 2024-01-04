import time
import random
import math
import torch
from .util import sigm, rev_sigm
import magicsoup as ms


class Culture:
    """Baseclass for culturing cells"""

    def __init__(
        self,
        world: ms.World,
        medium_refresher: "MediumRefresher",
        killer: "Killer",
        replicator: "Replicator",
        mutator: "Mutator",
        progressor: "Progressor",
        stopper: "Stopper",
    ):
        self.world = world
        self.step_i = 0
        self.progress = 0.0
        self.medium_refresher = medium_refresher
        self.progressor = progressor
        self.stopper = stopper
        self.killer = killer
        self.replicator = replicator
        self.mutator = mutator
        self.medium_refresher(self)

    def __iter__(self):
        return self

    def post_replication(self):
        pass

    def __next__(self):
        self.step_i = self.step_i + 1
        self.stopper(self)

        self.world.diffuse_molecules()
        self.world.degrade_molecules()
        self.world.enzymatic_activity()
        self.killer(self)
        self.mutator(self)
        self.replicator(self)
        self.post_replication()
        self.world.increment_cell_lifetimes()

        self.progress = self.progressor(self)
        return self.step_i


class ChemoStat(Culture):
    """Grow cells in ChemoStat"""

    def post_replication(self):
        self.medium_refresher(self)


class BatchCulture(Culture):
    """Grow cells in batch culture"""

    def __init__(
        self,
        passager: "Passager",
        genome_editor: "GenomeEditor" | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.passager = passager
        self.genome_editor = genome_editor
        self.growth_rate = 0.0
        self.cpd = 0.0
        self.split_i = 0
        self.split_start_step = 0
        self.split_start_cells = self.world.n_cells

    def post_replication(self):
        n_steps = self.step_i - self.split_start_step
        doubling = math.log(self.world.n_cells / self.split_start_cells, 2)
        if n_steps > 0:
            self.growth_rate = doubling / n_steps

        if self.passager(self):
            self.cpd += doubling
            self.medium_refresher(self)
            self.split_i += 1
            self.split_start_step = self.step_i
            self.split_start_cells = self.world.n_cells

        if self.genome_editor is not None:
            self.genome_editor(self)


class GenomeEditor:
    """At progress give cells genes from genome factory"""

    def __init__(self, at_progress: float, fact: ms.GenomeFact):
        self.is_edited = False
        self.fact = fact
        self.at_progress = at_progress

    def __call__(self, cltr: Culture):
        if not self.is_edited and cltr.progress > self.at_progress:
            updates = [
                (cltr.world.cell_genomes[d] + self.fact.generate(), d)
                for d in range(cltr.world.n_cells)
            ]
            cltr.world.update_cells(genome_idx_pairs=updates)
            self.is_edited = True


class MediumRefresher:
    """Set molecule species concentrations in medium to value"""

    def __init__(self, world: ms.World, val: float, molecules: list[ms.Molecule]):
        mol_2_idx = world.chemistry.mol_2_idx
        self.mol_idxs = [mol_2_idx[d] for d in molecules]
        self.val = val

    def __call__(self, cltr: Culture):
        cltr.world.molecule_map[self.mol_idxs] = self.val


class Passager:
    """Passage cells between min and max confluency"""

    def __init__(self, world: ms.World, min_confl=0.2, max_confl=0.7):
        n_max = world.map_size**2
        self.min_cells = int(n_max * min_confl)
        self.max_cells = int(n_max * max_confl)

    def __call__(self, cltr: BatchCulture) -> bool:
        if cltr.world.n_cells < self.max_cells:
            return False

        n_old = cltr.world.n_cells
        kill_n = max(n_old - self.max_cells, 0)
        idxs = random.sample(range(n_old), k=kill_n)
        cltr.world.kill_cells(cell_idxs=idxs)

        n_new = cltr.world.n_cells
        cltr.world.reposition_cells(cell_idxs=list(range(n_new)))
        return True


class Progressor:
    """Advance progress by average cell divisions"""

    def __init__(self, n_avg_divisions=100.0):
        self.n_avg_divisions = n_avg_divisions

    def __call__(self, cltr: Culture) -> float:
        mean_divis = cltr.world.cell_divisions.float().mean()
        return min(1.0, mean_divis.item() / self.n_avg_divisions)


class Mutator:
    """Mutate cells and recombinate old cells"""

    def __init__(self, snp_p=1e-6, lgt_p=1e-7, lgt_age=10):
        self.snp_p = snp_p
        self.lgt_p = lgt_p
        self.lgt_age = lgt_age

    def __call__(self, cltr: Culture):
        cltr.world.mutate_cells(p=self.snp_p)
        is_old = cltr.world.cell_lifetimes > self.lgt_age
        idxs = torch.argwhere(is_old).flatten().tolist()
        cltr.world.recombinate_cells(cell_idxs=idxs, p=self.lgt_p)


class Replicator:
    """Replicate cells for high molecule concentration"""

    def __init__(self, mol_i: int, k_x=30.0, n_x=3, min_x=4.0, min_lifetime=10):
        self.k_x = k_x
        self.n_x = n_x
        self.mol_i = mol_i
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

    def __init__(self, mol_i: int, k_x=0.04, n_x=1, k_g=3_000.0, n_g=7):
        self.k_x = k_x
        self.n_x = n_x
        self.mol_i = mol_i
        self.k_g = k_g
        self.n_g = n_g

    def __call__(self, cltr: Culture):
        x = cltr.world.cell_molecules[:, self.mol_i]
        g = torch.tensor([len(d) for d in cltr.world.cell_genomes])
        x_sample = rev_sigm(t=x, k=self.k_x, n=self.n_x)
        g_sample = sigm(t=g.float(), k=self.k_g, n=self.n_g)
        is_old = cltr.world.cell_lifetimes <= 3
        idxs = torch.argwhere(x_sample & g_sample & is_old).flatten().tolist()
        cltr.world.kill_cells(cell_idxs=idxs)


class Stopper:
    """Stop iteration on different conditions"""

    def __init__(self, max_steps=100_000, max_time_m=180, max_progress=1.0):
        self.max_steps = max_steps
        self.max_time_s = max_time_m * 60
        self.start_time = time.time()
        self.max_progress = max_progress

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
