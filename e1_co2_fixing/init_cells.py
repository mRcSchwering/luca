from pathlib import Path
import time
import torch
import magicsoup as ms
from .chemistry import WL_STAGES_MAP, _X, _E
from .experiment import (
    Experiment,
    BatchCulture,
    ProgressController,
    PassageByCells,
    ConstantRate,
    MediumFact,
    GenomeSizeController,
    MoleculeDependentCellDivision,
    MoleculeDependentCellDeath,
)
from .logging import BatchCultureLogger


THIS_DIR = Path(__file__).parent


class AdvanceBySplit(ProgressController):
    """Increment progress by each passage up to `n_splits`"""

    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def __call__(self, exp: BatchCulture) -> float:  # type: ignore[override]
        return min(1.0, exp.split_i / self.n_splits)


class DefinedMedium(MediumFact):
    """
    Medium is replaced during passage with fresh medium
    containing substrates at `substrate_init`.
    """

    def __init__(
        self,
        substrates: list[ms.Molecule],
        substrates_init: float,
        molmap: torch.Tensor,
        mol_2_idx: dict[str, int],
    ):
        self.substrates_init = substrates_init
        self.subs_idxs = [mol_2_idx[d.name] for d in substrates]
        self.molmap = molmap

    def __call__(self, exp: Experiment) -> torch.Tensor:
        t = torch.zeros_like(self.molmap)
        t[self.subs_idxs] = self.substrates_init
        return t


def run_trial(
    device: str,
    n_workers: int,
    run_name: str,
    n_steps: int,
    trial_max_time_s: int,
    hparams: dict,
):
    # runs reference
    runsdir = THIS_DIR / "runs"
    trial_dir = runsdir / run_name
    world = ms.World.from_file(rundir=runsdir, device=device, workers=n_workers)
    mol_2_idx = {d.name: i for i, d in enumerate(world.chemistry.molecules)}
    n_pxls = world.map_size**2

    # factories
    progress_controller = AdvanceBySplit(n_splits=hparams["n_splits"])

    medium_fact = DefinedMedium(
        substrates=WL_STAGES_MAP["WL-0"][1],
        molmap=world.molecule_map,
        mol_2_idx=mol_2_idx,
        substrates_init=hparams["substrates_init"],
    )

    mutation_rate_fact = ConstantRate(rate=hparams["mutation_rate"])

    passager = PassageByCells(
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        max_cells=n_pxls,
    )

    division_by_x = MoleculeDependentCellDivision(
        mol_i=mol_2_idx["X"], k=hparams["mol_divide_k"], n=3
    )
    death_by_e = MoleculeDependentCellDeath(
        mol_i=mol_2_idx["E"], k=hparams["mol_kill_k"], n=1
    )
    genome_size_controller = GenomeSizeController(k=hparams["genome_kill_k"], n=7)

    # init experiment with fresh medium
    exp = BatchCulture(
        world=world,
        lgt_rate=hparams["lgt_rate"],
        passager=passager,
        progress_controller=progress_controller,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
        division_by_x=division_by_x,
        death_by_e=death_by_e,
        genome_size_controller=genome_size_controller,
    )

    # add initial cells
    xt = ms.ProteinFact(ms.TransporterDomainFact(_X))
    et = ms.ProteinFact(ms.TransporterDomainFact(_E))
    init_genomes = [
        world.generate_genome(proteome=[xt, et], size=hparams["genome_size"])
        for _ in range(int(n_pxls * hparams["init_cell_cover"]))
    ]
    exp.world.add_cells(genomes=init_genomes)

    trial_t0 = time.time()
    print(f"Starting trial {run_name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")

    # start logging
    with BatchCultureLogger(
        trial_dir=trial_dir,
        hparams=hparams,
        exp=exp,
        watch_mols=[_X, _E],
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
                logger.log_scalars(step=step_i, dtime=dtime)
                break

            if step_i % 5 == 0:
                logger.log_scalars(step=step_i, dtime=dtime)

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
