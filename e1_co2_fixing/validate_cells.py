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
    ConstantRate,
    MediumFact,
    GenomeSizeController,
    MoleculeDependentCellDivision,
    MoleculeDependentCellDeath,
)
from .logging import ChemoStatLogger


THIS_DIR = Path(__file__).parent


class AdvanceByCellDivisions(ProgressController):
    """Increment progress by average cell divisions up to `n_divisions`"""

    def __init__(self, n_divisions: float):
        self.n_divisions = n_divisions

    def __call__(self, exp: ChemoStat) -> float:  # type: ignore[override]
        mean_divis = exp.world.cell_divisions.float().mean()
        return min(1.0, mean_divis.item() / self.n_divisions)


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

    # factories
    progress_controller = AdvanceByCellDivisions(n_divisions=hparams["n_divisions"])

    medium_fact = XGradient(
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        substrates_init=hparams["substrates_init"],
        additives_init=hparams["additives_init"],
        world=world,
    )

    mutation_rate_fact = ConstantRate(rate=hparams["mutation_rate"])

    division_by_x = MoleculeDependentCellDivision(
        mol_i=mol_2_idx["X"], k=hparams["mol_divide_k"], n=3
    )
    death_by_e = MoleculeDependentCellDeath(
        mol_i=mol_2_idx["E"], k=hparams["mol_kill_k"], n=1
    )
    genome_size_controller = GenomeSizeController(k=hparams["genome_kill_k"], n=7)

    # init experiment with fresh medium
    exp = ChemoStat(
        world=world,
        lgt_rate=hparams["lgt_rate"],
        progress_controller=progress_controller,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
        division_by_x=division_by_x,
        death_by_e=death_by_e,
        genome_size_controller=genome_size_controller,
    )

    # load initial cells
    load_cells(world=world, label=hparams["init_label"], runsdir=runsdir)
    world.cell_divisions[:] = 0.0
    world.labels = [ms.randstr(n=12) for _ in range(world.n_cells)]

    avg_genome_len = sum(len(d) for d in world.genomes) / world.n_cells
    print(f"{exp.world.n_cells} cells were added")
    print(f"   Average genome size is {avg_genome_len:.0f}")

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
                logger.log_scalars(step=step_i, dtime=dtime)
                break

            logger.log_scalars(step=step_i, dtime=dtime)

            if step_i % 5 == 0:
                exp.world.save_state(statedir=trial_dir / f"step={step_i}")
                logger.log_imgs(step=step_i)

            if exp.world.n_cells < min_cells:
                print(f"after {step_i} steps less than {min_cells} cells left")
                break

            if (time.time() - trial_t0) > trial_max_time_s:
                print(f"{trial_max_time_s} hours have passed")
                break

    print(f"Finishing trial {run_name}")
