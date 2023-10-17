from pathlib import Path
import time
import torch
import magicsoup as ms
from .chemistry import SUBSTRATES, ADDITIVES
from .util import load_cells
from .experiment import (
    Experiment,
    ChemoStat,
    ProgressController,
    MediumFact,
)
from .logging import ChemoStatLogger


class AdvanceByCellDivisions(ProgressController):
    """Increment progress by average cell divisions up to `n_divisions`"""

    def __init__(self, n_divisions: float):
        self.n_divisions = n_divisions

    def __call__(self, exp: ChemoStat) -> float:  # type: ignore[override]
        mean_divis = exp.world.cell_divisions.float().mean()
        return min(1.0, mean_divis.item() / self.n_divisions)


class AdjustGenomeSizeK:
    """Adjust genome-size-controller's k depending on progress"""

    def __init__(
        self,
        n_divisions: float,
        init_divisions=10,
        final_divisions=10,
        high=3000.0,
        low=1000.0,
        alpha=5,
    ):
        self.n = n_divisions
        self.pl = init_divisions / self.n
        self.pr = (n_divisions - final_divisions) / self.n
        self.a = high - low
        self.c = low
        self.alpha = alpha

    def _get_x(self, progress: float) -> float:
        if progress <= self.pl:
            return 0.0
        if progress >= self.pr:
            return 1.0
        return (progress - self.pl) / (self.pr - self.pl)

    def __call__(self, progress: float) -> float:
        x = self._get_x(progress=progress)
        return self.c + self.a * 0.5 ** (self.alpha * x)


class XGradient(MediumFact):
    """
    A 1D gradient is created over the X axis of the map.
    In the middle there is an area where fresh medium is added,
    at the borders there is an area where medium is removed.
    The total area where medium is added and where medium is removed
    is 10% each. Fresh medium contains substrates at `substrates_init`
    and additives at `additives_init`.
    """

    def __init__(
        self,
        substrates: list[ms.Molecule],
        additives: list[ms.Molecule],
        substrates_init: float,
        additives_init: float,
        world: ms.World,
    ):
        mol_2_idx = {d.name: i for i, d in enumerate(world.chemistry.molecules)}

        self.substrates_init = substrates_init
        self.additives_init = additives_init
        self.subs_idxs = [mol_2_idx[d.name] for d in substrates]
        self.add_idxs = [mol_2_idx[d.name] for d in additives]
        self.molmap = world.molecule_map

        s = world.map_size
        m = int(s / 2)
        w = int(s * 0.05)

        self.subs_mask = torch.zeros_like(world.molecule_map).bool()
        for idx in self.subs_idxs:
            self.subs_mask[idx, list(range(m - w, m + w))] = True

        self.add_mask = torch.zeros_like(world.molecule_map).bool()
        for idx in self.add_idxs:
            self.add_mask[idx, list(range(m - w, m + w))] = True

        self.extract_mask = torch.zeros_like(world.molecule_map).bool()
        self.extract_mask[:, list(range(0, w)) + list(range(s - w, s))] = True

    def __call__(self, exp: Experiment) -> torch.Tensor:
        self.molmap[self.subs_mask] = self.substrates_init
        self.molmap[self.add_mask] = self.additives_init
        self.molmap[self.extract_mask] = 0.0
        return self.molmap


def run_trial(
    device: str,
    n_workers: int,
    runs_dir: Path,
    run_name: str,
    n_steps: int,
    trial_max_time_s: int,
    hparams: dict,
):
    # runs reference
    trial_dir = runs_dir / run_name
    world = ms.World.from_file(rundir=runs_dir, device=device, workers=n_workers)

    # factories
    progress_controller = AdvanceByCellDivisions(n_divisions=hparams["n_divisions"])
    k_controller = AdjustGenomeSizeK(n_divisions=hparams["n_divisions"])

    medium_fact = XGradient(
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        substrates_init=hparams["substrates_init"],
        additives_init=hparams["additives_init"],
        world=world,
    )

    # init experiment with fresh medium
    exp = ChemoStat(
        world=world,
        progress_controller=progress_controller,
        medium_fact=medium_fact,
    )

    # load initial cells
    load_cells(world=world, label=hparams["init-label"], runsdir=runs_dir)

    trial_t0 = time.time()
    print(f"Starting trial {run_name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")

    # start logging
    with ChemoStatLogger(
        trial_dir=trial_dir,
        hparams=hparams,
        exp=exp,
        watch_mols=list(set(SUBSTRATES + ADDITIVES)),
    ) as logger:
        exp.world.save_state(statedir=trial_dir / "step=0")

        # start steps
        min_cells = int(exp.world.map_size**2 * 0.01)
        for step_i in exp.run(max_steps=n_steps):
            step_t0 = time.time()

            exp.step_1s()
            dtime = time.time() - step_t0

            if exp.progress >= 1.0:
                print(f"target reached after {step_i + 1} steps")
                exp.world.save_state(statedir=trial_dir / f"step={step_i}")
                logger.log_scalars(
                    step=step_i,
                    dtime=dtime,
                    kwargs={"Other/GenomeSizeK": exp.genome_size_controller.k},
                )
                break

            # shrink genome size k
            exp.genome_size_controller.k = k_controller(exp.progress)

            if step_i % 5 == 0:
                logger.log_scalars(
                    step=step_i,
                    dtime=dtime,
                    kwargs={"Other/GenomeSizeK": exp.genome_size_controller.k},
                )

            if step_i % 50 == 0:
                exp.world.save_state(statedir=trial_dir / f"step={step_i}")
                logger.log_imgs(step=step_i)

            if exp.world.n_cells < min_cells:
                print(f"after {step_i} steps less than {min_cells} cells left")
                break

            if (time.time() - trial_t0) > trial_max_time_s:
                print(f"{trial_max_time_s} hours have passed")
                break

    print(f"Finishing trial {run_name}")
