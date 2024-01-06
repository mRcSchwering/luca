import time
import torch
import magicsoup as ms
from .util import Config, load_cells, sigm, rev_sigm
from .checkpointing import BatchCultureCheckpointer
from .chemistry import ADDITIVES, SUBSTRATES, _X, _E
from .culture import Culture, BatchCulture
from .generators import (
    Replicator,
    Stopper,
    Passager,
    MediumRefresher,
)


class Progressor:
    """Advance progress by splits if growth rate high enough"""

    def __init__(self, n_splits: int, min_gr: float):
        self.min_gr = min_gr
        self.n_valid_total_splits = n_splits
        self.valid_split_i = 0

    def __call__(self, cltr: BatchCulture) -> float:
        if cltr.growth_rate >= self.min_gr:
            self.valid_split_i += 1
        return min(1.0, self.valid_split_i / self.n_valid_total_splits)


class Mutator:
    """Increase mutation rates during progress interval"""

    def __init__(
        self,
        progress_range: tuple[float, float],
        by: float,
        snp_p=1e-6,
        lgt_p=1e-7,
        lgt_age=10,
    ):
        self.start = min(progress_range)
        self.end = max(progress_range)
        self.by = by
        self.snp_p = snp_p
        self.lgt_p = lgt_p
        self.lgt_age = lgt_age

    def __call__(self, cltr: Culture):
        snp_p = self.snp_p
        lgt_p = self.lgt_p
        if self.start < cltr.progress < self.end:
            snp_p *= self.by
            lgt_p *= self.by
        cltr.world.mutate_cells(p=snp_p)
        is_old = cltr.world.cell_lifetimes > self.lgt_age
        idxs = torch.argwhere(is_old).flatten().tolist()
        cltr.world.recombinate_cells(cell_idxs=idxs, p=lgt_p)


class Killer:
    """Adjust genome-size-controller's k depending on progress"""

    def __init__(
        self,
        world: ms.World,
        mol: ms.Molecule,
        progress_range: tuple[float, float],
        k_g_range: tuple[float, float],
        k_x=0.04,
        n_x=1,
        n_g=7,
    ):
        self.mol_i = world.chemistry.mol_2_idx[mol]
        self.k_x = k_x
        self.n_x = n_x
        self.n_g = n_g
        self.k_g_start = k_g_range[0]
        self.k_g_end = k_g_range[1]
        self.k_g_slope = self.k_g_end - self.k_g_start
        self.start = min(progress_range)
        self.end = max(progress_range)
        self.interval = self.end - self.start

    def _get_k_g(self, progress: float) -> float:
        if progress <= self.start:
            return self.k_g_start
        if progress >= self.end:
            return self.k_g_end
        x = (progress - self.start) / self.interval
        return self.k_g_start + x * self.k_g_slope

    def __call__(self, cltr: Culture):
        k_g = self._get_k_g(cltr.progress)
        x = cltr.world.cell_molecules[:, self.mol_i]
        g = torch.tensor([len(d) for d in cltr.world.cell_genomes])
        x_sample = rev_sigm(t=x, k=self.k_x, n=self.n_x)
        g_sample = sigm(t=g.float(), k=k_g, n=self.n_g)
        is_old = cltr.world.cell_lifetimes <= 3
        idxs = torch.argwhere(x_sample & g_sample & is_old).flatten().tolist()
        cltr.world.kill_cells(cell_idxs=idxs)


def run_trial(run_name: str, config: Config, hparams: dict):
    n_init_splits = hparams["n_init_splits"]
    n_init_adapt_splits = n_init_splits + hparams["n_adapt_splits"]
    n_total_splits = n_init_adapt_splits + hparams["n_final_splits"]
    adaption_start = n_init_splits / n_total_splits
    adaption_end = n_init_adapt_splits / n_total_splits
    print("Genome size k:")
    print(f"  k shrinks from {hparams['from_k']} to {hparams['to_k']}")
    print(f"  during progress {adaption_start:.2f} to {adaption_end:.2f}")

    trial_dir = config.runs_dir / run_name
    world = ms.World.from_file(rundir=config.runs_dir, device=config.device)

    stopper = Stopper(max_steps=config.max_steps, max_time_m=config.max_time_m)
    replicator = Replicator(world=world, mol=_X)
    progressor = Progressor(n_splits=n_total_splits, min_gr=hparams["min_gr"])
    passager = Passager(world=world, cnfls=(hparams["min_confl"], hparams["max_confl"]))

    medium_refresher = MediumRefresher(
        world=world,
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        additives_val=hparams["additives_init"],
        substrates_val=hparams["substrates_init"],
    )

    killer = Killer(
        world=world,
        mol=_E,
        progress_range=(adaption_start, adaption_end),
        k_g_range=(hparams["from_k"], hparams["to_k"]),
    )

    mutator = Mutator(
        progress_range=(adaption_start, adaption_end),
        by=hparams["mutation_rate_mult"],
    )

    cltr = BatchCulture(
        world=world,
        medium_refresher=medium_refresher,
        killer=killer,
        replicator=replicator,
        mutator=mutator,
        progressor=progressor,
        stopper=stopper,
        passager=passager,
    )

    # load previous cells
    load_cells(world=world, label=hparams["init-label"], runsdir=config.runs_dir)

    manager = BatchCultureCheckpointer(
        trial_dir=trial_dir,
        hparams=hparams,
        cltr=cltr,
        watch_mols=[_X, _E],
        scalar_freq=5,
        img_freq=50,
        save_freq=100,
    )

    with manager:
        t0 = time.time()
        for step in cltr:
            t1 = time.time()
            manager.throttled_log_scalars(step, {"Other/TimePerStep[s]": t1 - t0})
            manager.throttled_log_imgs(step)
            manager.throttled_save_state(step)
            t0 = t1
