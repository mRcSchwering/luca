"""
Simulation to teach cells to fix CO2.

  python -m e1_co2_fixing.main --help

"""
from argparse import ArgumentParser
import time
import datetime as dt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
from .world import init_world, load_world
from .genomes import GENOMES, generate_genomes
from .experiment import Experiment

THIS_DIR = Path(__file__).parent


def _log_scalars(
    exp: Experiment,
    writer: SummaryWriter,
    step: int,
    dtime: float,
):
    mol_name_idx_list = [
        ("acetyl-CoA", exp.mol_2_idx["acetyl-CoA"]),
        ("G3P", exp.mol_2_idx["G3P"]),
        ("pyruvate", exp.mol_2_idx["pyruvate"]),
        ("CO2", exp.mol_2_idx["CO2"]),
    ]

    n_cells = exp.world.n_cells
    molecule_map = exp.world.molecule_map
    cell_molecules = exp.world.cell_molecules
    molecules = {f"Molecules/{s}": i for s, i in mol_name_idx_list}

    if n_cells == 0:
        for scalar, idx in molecules.items():
            writer.add_scalar(scalar, molecule_map[idx].mean().item(), step)
    else:
        writer.add_scalar("Cells/total[n]", n_cells, step)
        mean_surv = exp.world.cell_survival.mean()
        writer.add_scalar("Cells/Survival[avg]", mean_surv, step)
        writer.add_scalar("Cells/Generation", exp.gen_i, step)
        for scalar, idx in molecules.items():
            mm = molecule_map[idx].sum().item()
            cm = cell_molecules[:, idx].sum().item()
            writer.add_scalar(scalar, (mm + cm) / exp.n_pxls, step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)
    writer.add_scalar("Other/Split", exp.split_i, step)
    writer.add_scalar("Other/Step", step, step)


def _log_imgs(exp: Experiment, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


# TODO: test params shown nicely
# TODO: params for kill/replicate sampling
# TODO: splits = on/off (only to prevent overcrouding)


def trial(
    device: str,
    n_workers: int,
    name: str,
    init_genomes: list[str],
    trial_max_time_s: int,
    hparams: dict,
):
    rundir = THIS_DIR / "runs"
    world = load_world(rundir=rundir, device=device, n_workers=n_workers)

    trial_dir = rundir / name
    writer = SummaryWriter(log_dir=trial_dir)
    exp, ssi, sei = get_summary(hparam_dict=hparams, metric_dict={"Other/Step": 0})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    exp = Experiment(
        world=world,
        mol_map_init=hparams["mol_map_init"],
        init_genomes=init_genomes,
        init_cell_cover=hparams["init_cell_cover"],
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
    )

    assert exp.world.map_size == hparams["map_size"]
    assert exp.world.device == hparams["device"]
    assert exp.world.workers == hparams["n_workers"]

    print(f"Starting trial {name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    trial_t0 = time.time()

    _log_scalars(exp=exp, writer=writer, step=0, dtime=0)
    _log_imgs(exp=exp, writer=writer, step=0)

    for step_i in range(1, hparams["n_steps"] + 1):
        step_t0 = time.time()

        exp.step_1s()

        if step_i % 10 == 0:
            exp.step_10s()
            _log_scalars(
                exp=exp, writer=writer, step=step_i, dtime=time.time() - step_t0
            )

        if step_i % 100 == 0:
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            _log_imgs(exp=exp, writer=writer, step=step_i)

        if exp.world.n_cells == 0:
            print(f"after {step_i} steps 0 cells left")
            break

        if (time.time() - trial_t0) > trial_max_time_s:
            print(f"{trial_max_time_s} hours have passed")
            break

    print(f"Finishing trial {name}")
    exp.world.save_state(statedir=trial_dir / f"step={step_i}")
    writer.close()


def trials(kwargs: dict):
    init_genome = kwargs.pop("init_genome")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60

    genomes = generate_genomes(
        rundir=THIS_DIR / "runs",
        genome=init_genome,
        genome_size=kwargs["init_genome_size"],
        n_genomes=1000,
    )

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
    for trial_i in range(n_trials):
        trial(
            device=device,
            n_workers=n_workers,
            name=f"{ts}_{init_genome}_{trial_i}",
            init_genomes=genomes,
            trial_max_time_s=trial_max_time_s,
            hparams=kwargs,
        )


def init(kwargs: dict):
    rundir = THIS_DIR / "runs"
    map_size = kwargs["map_size"]
    print(f"Initialing world with map_size={map_size}")
    init_world(rundir=rundir, map_size=map_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    init_parser = subparsers.add_parser("init", help="initialize world object")
    init_parser.set_defaults(func=init)
    init_parser.add_argument("--map_size", default=256, type=int)

    trial_parser = subparsers.add_parser("trials", help="run trials")
    trial_parser.set_defaults(func=trials)
    trial_parser.add_argument("init_genome", choices=GENOMES, type=str)
    parser.add_argument("--n_trials", default=3, type=int)
    parser.add_argument("--n_steps", default=10_000, type=int)
    parser.add_argument("--mol_map_init", default=5.0, type=float)
    parser.add_argument("--init_genome_size", default=500, type=int)
    parser.add_argument("--init_cell_cover", default=0.1, type=float)
    parser.add_argument("--split_ratio", default=0.15, type=float)
    parser.add_argument("--split_thresh", default=0.75, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--n_workers", default=4, type=int)
    parser.add_argument("--trial_max_time_h", default=12, type=int)

    args = parser.parse_args()
    args.func(vars(args))

    print("done")
