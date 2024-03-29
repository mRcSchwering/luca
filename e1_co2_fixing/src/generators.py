import time
import random
import torch
import magicsoup as ms
from .util import sigm, Config
from .culture import Culture, BatchCulture


class GenomeEditor:
    """At progress give cells genes from genome factory"""

    def __init__(
        self,
        at_progress: float,
        fact: ms.GenomeFact,
        accuracy=0.3,
        efficiency=0.7,
    ):
        self.is_edited = False
        self.fact = fact
        self.at_progress = at_progress
        self.accuracy = accuracy
        self.efficiency = efficiency

    def __call__(self, cltr: Culture):
        if not self.is_edited and cltr.progress >= self.at_progress:
            n = int(cltr.world.n_cells * self.efficiency)
            updates: list[tuple[str, int]] = []
            for idx in random.sample(range(cltr.world.n_cells), k=n):
                genome = cltr.world.cell_genomes[idx]
                genes = self.fact.generate()
                if random.random() < self.accuracy:
                    updates.append((genome + genes, idx))
                else:
                    s = random.randint(0, len(genome) - 1)
                    updates.append((genome[:s] + genes + genome[s:], idx))
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
        non_essentials_val=0.0,
    ):
        if additives is None:
            additives = []
        self.subs_idxs = [world.chemistry.mol_2_idx[d] for d in substrates]
        self.add_idxs = [world.chemistry.mol_2_idx[d] for d in additives]
        self.subs_val = substrates_val
        self.add_val = additives_val
        self.others_val = non_essentials_val
        self.other_idxs = list(
            set(world.chemistry.mol_2_idx.values())
            - set(self.subs_idxs)
            - set(self.add_idxs)
        )

    def __call__(self, cltr: Culture):
        cltr.world.molecule_map[self.other_idxs] = self.others_val
        cltr.world.molecule_map[self.subs_idxs] = self.subs_val
        cltr.world.molecule_map[self.add_idxs] = self.add_val


class Passager:
    """Passage cells between min and max confluency"""

    def __init__(self, world: ms.World, cnfls=(0.2, 0.7), max_steps=10_000):
        n_max = world.map_size**2
        self.min_cells = int(n_max * min(cnfls))
        self.max_cells = int(n_max * max(cnfls))
        self.max_steps = max_steps

    def __call__(self, cltr: BatchCulture) -> bool:
        if cltr.world.n_cells < self.max_cells or cltr.step_i > self.max_steps:
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

    def __init__(
        self, snp_p=1e-6, lgt_p=1e-7, lgt_rate=0.1, spike_p=0.0, spike_size=50
    ):
        self.snp_p = snp_p
        self.lgt_p = lgt_p
        self.lgt_rate = lgt_rate
        self.spike_p = spike_p
        self.spike_size = spike_size

    def mutate(self, cltr: Culture, snp_p: float, lgt_p: float, spike_p: float):
        cltr.world.mutate_cells(p=snp_p)
        n_cells = cltr.world.n_cells
        idxs = random.sample(range(n_cells), k=int(n_cells * self.lgt_rate))
        cltr.world.recombinate_cells(cell_idxs=idxs, p=lgt_p)
        updates: list[tuple[str, int]] = []
        for idx in range(cltr.world.n_cells):
            if random.random() <= spike_p:
                genome = cltr.world.cell_genomes[idx]
                spike = ms.random_genome(s=self.spike_size)
                s = random.randint(0, len(genome) - 1)
                updates.append((genome[:s] + spike + genome[s:], idx))
        cltr.world.update_cells(genome_idx_pairs=updates)

    def __call__(self, cltr: Culture):
        self.mutate(
            cltr=cltr,
            snp_p=self.snp_p,
            lgt_p=self.lgt_p,
            spike_p=self.spike_p,
        )


class Replicator:
    """Replicate cells for high molecule concentration"""

    # TODO: need more X?

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

    # TODO: X dependent killing?

    def __init__(
        self,
        world: ms.World,
        mol: ms.Molecule,
        k_e=0.5,
        n_e=-2,
        k_g=2_000.0,
        n_g=7,
        max_g_size=3_000,
    ):
        self.k_e = k_e
        self.n_e = n_e
        self.mol_i = world.chemistry.mol_2_idx[mol]
        self.k_g = k_g
        self.n_g = n_g
        self.max_g_size = max_g_size

    def __call__(self, cltr: Culture):
        device = cltr.world.device
        x = cltr.world.cell_molecules[:, self.mol_i]
        g = torch.tensor([len(d) for d in cltr.world.cell_genomes], device=device)
        x_sample = sigm(t=x + 0.1, k=self.k_e, n=self.n_e)
        g_sample = sigm(t=g.float(), k=self.k_g, n=self.n_g)
        is_too_big = g > self.max_g_size  # avoid wasting memory
        mask = x_sample | g_sample | is_too_big
        idxs = torch.argwhere(mask).flatten().tolist()
        cltr.world.kill_cells(cell_idxs=idxs)


class Stopper:
    """Stop iteration on different conditions"""

    def __init__(
        self,
        max_steps=100_000,
        max_time_m=180,
        max_progress=1.0,
        min_cells=100,
        max_steps_without_progress=1000,
    ):
        self.max_steps = max_steps
        self.max_time_s = max_time_m * 60
        self.start_time = time.time()
        self.max_progress = max_progress
        self.min_cells = min_cells
        self.steps_wo_progress = max_steps_without_progress
        self.last_progress = 0.0
        self.last_progress_step = 0

    def __call__(self, cltr: Culture):
        if cltr.progress > self.last_progress:
            self.last_progress_step = cltr.step_i
            self.last_progress = cltr.progress
        if cltr.step_i - self.last_progress_step > self.steps_wo_progress:
            print(f"Maximum steps without progress {self.steps_wo_progress:,} reached")
            raise StopIteration
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

    @classmethod
    def from_config(cls, cnfg: Config, world: ms.World) -> "Stopper":
        return cls(
            max_steps=cnfg.max_steps,
            max_time_m=cnfg.max_time_m,
            max_steps_without_progress=cnfg.max_steps_without_progress,
            min_cells=int(cnfg.min_confluency * world.map_size**2),
        )


class BatchCultureStopper(Stopper):
    """Stop batch culture iterration at different conditions"""

    def __init__(self, max_steps_without_split: int, **kwargs):
        super().__init__(**kwargs)
        self.steps_wo_split = max_steps_without_split
        self.last_split = 0
        self.last_split_step = 0

    def __call__(self, cltr: BatchCulture):  # type: ignore
        super().__call__(cltr)
        if cltr.split_i > self.last_split:
            self.last_split = cltr.split_i
            self.last_split_step = cltr.step_i
        if cltr.step_i - self.last_split_step > self.steps_wo_split:
            print(f"Maximum steps without split {self.steps_wo_split:,} reached")
            raise StopIteration

    @classmethod
    def from_config(cls, cnfg: Config, world: ms.World) -> "Stopper":
        return cls(
            max_steps=cnfg.max_steps,
            max_time_m=cnfg.max_time_m,
            max_steps_without_progress=cnfg.max_steps_without_progress,
            min_cells=int(cnfg.min_confluency * world.map_size**2),
            max_steps_without_split=cnfg.max_steps_without_split,
        )
