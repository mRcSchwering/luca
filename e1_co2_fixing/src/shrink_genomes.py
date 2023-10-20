from pathlib import Path
import time
import torch
import magicsoup as ms
from .chemistry import SUBSTRATES, ADDITIVES
from .util import load_cells
from .experiment import (
    Experiment,
    BatchCulture,
    PassageByCells,
    MediumFact,
    ProgressController,
    MutationRateSteps,
)
from .logging import BatchCultureLogger


class AdvanceBySplitsAndGrowthRate(ProgressController):
    """
    Increment progress by passages up to `n_total_splits`.
    But only passages with an average growth rate of at least
    `min_gr` are counted.
    """

    def __init__(self, n_total_splits: float, min_gr: float):
        self.min_gr = min_gr
        self.n_valid_total_splits = n_total_splits
        self.valid_split_i = 0

    def __call__(self, exp: BatchCulture) -> float:  # type: ignore[override]
        if exp.growth_rate >= self.min_gr:
            self.valid_split_i += 1

        return min(1.0, self.valid_split_i / self.n_valid_total_splits)


class DefinedMedium(MediumFact):
    """
    Medium is replaced during passage with fresh medium
    containing substrates at `substrate_init` and additives at `additives_init`.
    """

    def __init__(
        self,
        substrates: list[ms.Molecule],
        additives: list[ms.Molecule],
        substrates_init: float,
        additives_init: float,
        molmap: torch.Tensor,
        mol_2_idx: dict[str, int],
    ):
        self.substrates_init = substrates_init
        self.additives_init = additives_init
        self.subs_idxs = [mol_2_idx[d.name] for d in substrates]
        self.add_idxs = [mol_2_idx[d.name] for d in additives]
        self.molmap = molmap

    def __call__(self, exp: Experiment) -> torch.Tensor:
        t = torch.zeros_like(self.molmap)
        t[self.subs_idxs] = self.substrates_init
        t[self.add_idxs] = self.additives_init
        return t


class AdjustGenomeSizeK:
    """Adjust genome-size-controller's k depending on progress"""

    def __init__(
        self,
        at_progress: float,
        to_progress: float,
        high: float,
        low: float,
    ):
        self.at_progress = at_progress
        self.to_progress = to_progress
        self.a = low - high
        self.c = high

    def _get_x(self, progress: float) -> float:
        if progress <= self.at_progress:
            return 0.0
        if progress >= self.to_progress:
            return 1.0
        return (progress - self.at_progress) / (self.to_progress - self.at_progress)

    def __call__(self, progress: float) -> float:
        x = self._get_x(progress=progress)
        return self.c + self.a * x


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
    mol_2_idx = {d.name: i for i, d in enumerate(world.chemistry.molecules)}
    n_pxls = world.map_size**2

    # factories
    n_init_splits = hparams["n_init_splits"]
    n_init_adapt_splits = n_init_splits + hparams["n_adapt_splits"]
    n_total_splits = n_init_adapt_splits + hparams["n_final_splits"]
    adaption_start = n_init_splits / n_total_splits
    adaption_end = n_init_adapt_splits / n_total_splits

    base_rate = MutationRateSteps.base_rate
    mutation_rate_fact = MutationRateSteps(
        progress_rate_pairs=[
            (0.0, base_rate),
            (adaption_start, base_rate * hparams["mutation_rate_mult"]),
            (adaption_end, base_rate),
        ]
    )

    progress_cntrlr = AdvanceBySplitsAndGrowthRate(
        n_total_splits=n_total_splits, min_gr=hparams["min_gr"]
    )

    k_cntrlr = AdjustGenomeSizeK(
        at_progress=adaption_start,
        to_progress=adaption_end,
        high=hparams["from_k"],
        low=hparams["to_k"],
    )

    medium_fact = DefinedMedium(
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        molmap=world.molecule_map,
        mol_2_idx=mol_2_idx,
        additives_init=hparams["additives_init"],
        substrates_init=hparams["substrates_init"],
    )

    passager = PassageByCells(
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        max_cells=n_pxls,
    )

    print("Medium:")
    print(f"  substrates: {', '.join(d.name for d in SUBSTRATES)}")
    print(f"  additives: {', '.join(d.name for d in ADDITIVES)}")
    print("Genome size k:")
    print(f"  k shrinks from {k_cntrlr.c} to {k_cntrlr.c + k_cntrlr.a:.2f}")
    print(f"  during progress {adaption_start:.2f} to {adaption_end:.2f}")

    # init experiment with fresh medium
    exp = BatchCulture(
        world=world,
        passager=passager,
        progress_controller=progress_cntrlr,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
    )

    # load initial cells
    load_cells(world=world, label=hparams["init-label"], runsdir=runs_dir)

    trial_t0 = time.time()
    print(f"Starting trial {run_name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")

    # start logging
    with BatchCultureLogger(
        trial_dir=trial_dir,
        hparams=hparams,
        exp=exp,
        watch_mols=list(set(ADDITIVES + SUBSTRATES)),
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
            exp.genome_size_controller.k = k_cntrlr(exp.progress)

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
