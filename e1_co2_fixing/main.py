"""
Simulation to teach cells to fix CO2.

  python -m e1_co2_fixing.main --help

"""
from argparse import ArgumentParser
import time
import datetime as dt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from .experiment import Experiment

THIS_DIR = Path(__file__).parent


def _log_scalars(
    exp: Experiment,
    writer: SummaryWriter,
    step: int,
    dtime: float,
    mol_name_idx_list: list[tuple[str, int]],
):
    n_cells = len(exp.world.cells)
    molecule_map = exp.world.molecule_map
    cell_molecules = exp.world.cell_molecules
    molecules = {f"Molecules/{s}": i for s, i in mol_name_idx_list}

    if n_cells == 0:
        for scalar, idx in molecules.items():
            writer.add_scalar(scalar, molecule_map[idx].mean().item(), step)
    else:
        writer.add_scalar("Cells/total[n]", n_cells, step)
        cell_surv = exp.world.cell_survival.float()
        writer.add_scalar("Cells/Survival[avg]", cell_surv.mean(), step)
        cell_divis = exp.world.cell_divisions.float()
        writer.add_scalar("Cells/Divisions[avg]", cell_divis.mean(), step)
        for scalar, idx in molecules.items():
            mm = molecule_map[idx].sum().item()
            cm = cell_molecules[:, idx].sum().item()
            writer.add_scalar(scalar, (mm + cm) / exp.n_pxls, step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)
    writer.add_scalar("Other/Split", exp.split_i, step)
    writer.add_scalar("Other/MaxStep", step)


def _log_imgs(exp: Experiment, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


def trial(hparams: dict):
    name = hparams.pop("name")
    trial_dir = THIS_DIR / "runs" / name
    writer = SummaryWriter(log_dir=trial_dir)
    writer.add_hparams(
        hparam_dict=hparams, metric_dict={"Other/MaxStep": 0}, run_name=name
    )

    exp = Experiment(
        map_size=hparams["map_size"],
        mol_map_init=hparams["mol_map_init"],
        init_genome_size=hparams["init_genome_size"],
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        max_splits=hparams["max_splits"],
        device=hparams["device"],
        n_workers=hparams["n_workers"],
    )

    exp.world = exp.world.from_file(rundir=trial_dir.parent)
    assert exp.world.map_size == hparams["map_size"]
    assert exp.world.device == hparams["device"]
    assert exp.world.workers == hparams["n_workers"]
    exp.prep_world()

    watch_molecules = [
        ("acetyl-CoA", exp.mol_2_idx["acetyl-CoA"]),
        ("G3P", exp.mol_2_idx["G3P"]),
        ("pyruvate", exp.mol_2_idx["pyruvate"]),
        ("CO2", exp.mol_2_idx["CO2"]),
    ]

    print(f"Starting trial {name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    trial_max_time_s = hparams["trial_max_time_h"] * 60 * 60
    trial_t0 = time.time()

    _log_scalars(
        exp=exp,
        writer=writer,
        step=0,
        dtime=0,
        mol_name_idx_list=watch_molecules,
    )
    _log_imgs(exp=exp, writer=writer, step=0)

    for step_i in range(1, hparams["n_steps"] + 1):
        step_t0 = time.time()

        exp.step_1s()

        if step_i % 10 == 0:
            exp.step_10s()
            _log_scalars(
                exp=exp,
                writer=writer,
                step=step_i,
                dtime=time.time() - step_t0,
                mol_name_idx_list=watch_molecules,
            )

        if step_i % 100 == 0:
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            _log_imgs(exp=exp, writer=writer, step=step_i)

        if len(exp.world.cells) == 0:
            print(f"after {step_i} steps 0 cells left")
            break

        if (time.time() - trial_t0) > trial_max_time_s:
            print(f"{hparams['trial_max_time_h']} hours have passed")
            break

    print(f"Finishing trial {name}")
    exp.world.save_state(statedir=trial_dir / f"step={step_i}")
    writer.close()


def init_exp(hparams: dict):
    exp = Experiment(
        map_size=hparams["map_size"],
        mol_map_init=hparams["mol_map_init"],
        init_genome_size=hparams["init_genome_size"],
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        max_splits=hparams["max_splits"],
        device=hparams["device"],
        n_workers=hparams["n_workers"],
    )
    rundir = THIS_DIR / "runs"
    exp.world.save(rundir=rundir)


def main(kwargs: dict):
    is_init = kwargs.pop("init")
    if is_init:
        init_exp(hparams=kwargs)
        return

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    n_trials = kwargs.pop("n_trials")
    for trial_i in range(n_trials):
        trial(hparams={"name": f"{ts}_{trial_i}", **kwargs})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--n_trials", default=3, type=int)
    parser.add_argument("--n_steps", default=10_000, type=int)
    parser.add_argument("--map_size", default=128, type=int)
    parser.add_argument("--mol_map_init", default=5.0, type=float)
    parser.add_argument("--init_genome_size", default=500, type=int)
    parser.add_argument("--split_ratio", default=0.2, type=float)
    parser.add_argument("--split_thresh", default=0.8, type=float)
    parser.add_argument("--max_splits", default=5, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--trial_max_time_h", default=12, type=int)
    args = parser.parse_args()

    main(vars(args))
    print("done")
