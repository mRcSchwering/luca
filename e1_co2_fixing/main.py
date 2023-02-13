"""
Simulation to teach cells to fix CO2.

  python -m experiments.e1_co2_fixing.main --help

"""
from argparse import ArgumentParser
import time
import datetime as dt
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from .experiment import Experiment

THIS_DIR = Path(__file__).parent


def _log_tensorboard(
    exp: Experiment,
    writer: SummaryWriter,
    step: int,
    dtime: float,
    mol_name_idx_list: list[tuple[str, int]],
):
    if step % 10 != 0:
        return

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
    writer.add_scalar("Other/NSplits[s]", exp.n_splits, step)

    if step % 100 == 0:
        writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


def trial(hparams: dict):
    exp = Experiment(
        map_size=hparams["map_size"],
        init_genome_size=hparams["init_genome_size"],
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        max_splits=hparams["max_splits"],
        device=hparams["device"],
        n_workers=hparams["n_workers"],
    )

    rundir = THIS_DIR / "runs"
    exp.world = exp.world.from_file(rundir=rundir)
    assert exp.world.map_size == hparams["map_size"]
    assert exp.world.device == hparams["device"]
    assert exp.world.workers == hparams["n_workers"]

    trial_dir = rundir / hparams["name"]
    writer = SummaryWriter(log_dir=trial_dir)

    watch_molecules = [
        ("acetyl-CoA", exp.mol_2_idx["acetyl-CoA"]),
        ("HS-CoA", exp.mol_2_idx["HS-CoA"]),
        ("formiat", exp.mol_2_idx["formiat"]),
        ("FH4", exp.mol_2_idx["FH4"]),
        ("formyl-FH4", exp.mol_2_idx["formyl-FH4"]),
        ("methyl-FH4", exp.mol_2_idx["methyl-FH4"]),
        ("methylen-FH4", exp.mol_2_idx["methylen-FH4"]),
        ("Ni-ACS", exp.mol_2_idx["Ni-ACS"]),
        ("methyl-Ni-ACS", exp.mol_2_idx["methyl-Ni-ACS"]),
    ]

    print(f"Starting trial {hparams['name']}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    trial_t0 = time.time()
    trial_max_time_h = hparams["trial_max_time_h"]

    for step_i in range(hparams["n_steps"]):
        step_t0 = time.time()

        exp.step_1s()

        if step_i % 10 == 0:
            exp.step_10s()

        _log_tensorboard(
            exp=exp,
            writer=writer,
            step=step_i,
            dtime=time.time() - step_t0,
            mol_name_idx_list=watch_molecules,
        )

        if step_i % 100 == 0:
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")

        if len(exp.world.cells) == 0:
            print(f"after {step_i} steps 0 cells left")
            break

        if (time.time() - trial_t0) * 60 * 60 > trial_max_time_h:
            print(f"{trial_max_time_h}h have passed")
            break

    print(f"Finishing trial {hparams['name']}")
    exp.world.save_state(statedir=trial_dir / f"step={step_i}")
    writer.close()


def init_exp(hparams: dict):
    exp = Experiment(
        map_size=hparams["map_size"],
        init_genome_size=hparams["init_genome_size"],
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        max_splits=hparams["max_splits"],
        device=hparams["device"],
        n_workers=hparams["n_workers"],
    )

    rundir = THIS_DIR / "runs"
    rundir.mkdir(exist_ok=True)
    exp.world.save(rundir=rundir)
    with open(rundir / "hparams.json", "w", encoding="utf-8") as fh:
        json.dump(hparams, fh)


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
    parser.add_argument("--init", default=False, type=bool)
    parser.add_argument("--n_trials", default=3, type=int)
    parser.add_argument("--n_steps", default=10_000, type=int)
    parser.add_argument("--map_size", default=128, type=int)
    parser.add_argument("--init_genome_size", default=500, type=int)
    parser.add_argument("--split_ratio", default=0.2, type=float)
    parser.add_argument("--split_thresh", default=0.7, type=float)
    parser.add_argument("--max_splits", default=3, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--trial_max_time_h", default=12, type=int)
    args = parser.parse_args()

    main(**vars(args))
    print("done")
