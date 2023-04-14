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
import magicsoup as ms
from .chemistry import GENOMES
from .util import init_world, generate_genomes
from .experiment import Experiment

THIS_DIR = Path(__file__).parent

# TODO: 30% split ratio? (mehr Zellen bleiben am leben)
# TODO: Zellen werden vor der neuen Platte gekillt?
# TODO: 1000 adaption generations?
# TODO: Zellen Transporter für essentials, CO2, X geben
# TODO: splits = immer 20% von Platte füllen, immer splitten wenn Energie low
#       oder splitten wenn platte voll (zB 60%)
# TODO: Energie wie Platte, wird immer mit zB 100 geliefert


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
        mean_surv = exp.world.cell_survival.float().mean()
        writer.add_scalar("Cells/Survival[avg]", mean_surv, step)
        writer.add_scalar("Cells/Generation", exp.gen_i, step)
        for scalar, idx in molecules.items():
            mm = molecule_map[idx].sum().item()
            cm = cell_molecules[:, idx].sum().item()
            writer.add_scalar(scalar, (mm + cm) / exp.n_pxls, step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)
    writer.add_scalar("Other/Split", exp.split_i, step)
    writer.add_scalar("Other/Score", exp.score, step)

    co2_max = exp.n_pxls * exp.add_co2.val
    co2_act = exp.world.molecule_map[exp.CO2_I].sum().item()
    energy_max = exp.n_pxls * exp.add_energy.val
    energy_act = exp.world.molecule_map[exp.X_I].sum().item()
    writer.add_scalar("Other/dCO2", co2_max - co2_act, step)
    writer.add_scalar("Other/dEnergy", energy_max - energy_act, step)


def _log_imgs(exp: Experiment, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


def trial(
    device: str,
    n_workers: int,
    name: str,
    init_genome: str,
    init_cell_cover: float,
    n_final_gens: int,
    n_steps: int,
    trial_max_time_s: int,
    hparams: dict,
):
    rundir = THIS_DIR / "runs"
    world = ms.World.from_file(rundir=rundir, device=device, workers=n_workers)

    trial_dir = rundir / name
    writer = SummaryWriter(log_dir=trial_dir)
    exp, ssi, sei = get_summary(hparam_dict=hparams, metric_dict={"Other/Score": 0.0})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    n_init_cells = int(world.map_size**2 * init_cell_cover)
    genomes = generate_genomes(
        rundir=THIS_DIR / "runs",
        name=init_genome,
        genome_size=600,  # min=600 to accomodate proteomes
        n_genomes=n_init_cells,
    )

    exp = Experiment(
        world=world,
        n_adaption_gens=hparams["n_adaption_gens"],
        n_final_gens=n_final_gens,
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        init_genomes=genomes,
    )

    assert exp.world.device == device
    assert exp.world.workers == n_workers

    print(f"Starting trial {name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    trial_t0 = time.time()

    _log_scalars(exp=exp, writer=writer, step=0, dtime=0)
    _log_imgs(exp=exp, writer=writer, step=0)

    for step_i in range(1, n_steps + 1):
        step_t0 = time.time()

        exp.step_1s()

        if step_i % 10 == 0:
            exp.step_10s()
            dtime = time.time() - step_t0
            _log_scalars(exp=exp, writer=writer, step=step_i, dtime=dtime)

        if step_i % 100 == 0:
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            _log_imgs(exp=exp, writer=writer, step=step_i)

        if exp.world.n_cells < 500:
            print(f"after {step_i} stepsless than 500 cells left")
            break

        if exp.gen_i > exp.n_total_gens:
            print(f"target generation {exp.n_total_gens} reached after {step_i} steps")
            break

        if (time.time() - trial_t0) > trial_max_time_s:
            print(f"{trial_max_time_s} hours have passed")
            break

    print(f"Finishing trial {name}")
    exp.world.save_state(statedir=trial_dir / f"step={step_i}")
    writer.close()


def trials(kwargs: dict):
    kwargs.pop("func")
    init_genome = kwargs.pop("init_genome")
    init_cell_cover = kwargs.pop("init_cell_cover")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_final_gens = kwargs.pop("n_final_gens")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        trial(
            device=device,
            n_workers=n_workers,
            name=f"{ts}_{init_genome}_{trial_i}",
            init_genome=init_genome,
            init_cell_cover=init_cell_cover,
            n_final_gens=n_final_gens,
            n_steps=n_steps,
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
    init_parser.add_argument(
        "--map_size",
        default=256,
        type=int,
        help="Number of pixels of 2D map in each direction (default %(default)s)",
    )

    trial_parser = subparsers.add_parser("trials", help="run trials")
    trial_parser.set_defaults(func=trials)
    trial_parser.add_argument("init_genome", choices=GENOMES, type=str)
    trial_parser.add_argument(
        "--n_final_gens",
        default=100.0,
        type=float,
        help="For how many generatins to run the experiment after reaching minimal medium (default %(default)s)",
    )
    trial_parser.add_argument(
        "--n_adaption_gens",
        default=1_000.0,
        type=float,
        help="Over how many generations to reduce complex medium to minimal medium (default %(default)s)",
    )
    trial_parser.add_argument(
        "--init_cell_cover",
        default=0.2,
        type=float,
        help="Ratio of map initially covered by cells (default %(default)s)",
    )
    trial_parser.add_argument(
        "--split_ratio",
        default=0.2,
        type=float,
        help="Ratio of cells carried over during passage (theoretically 0.13-0.2 is best, default %(default)s)",
    )
    trial_parser.add_argument(
        "--split_thresh",
        default=0.6,
        type=float,
        help="Ratio of map covered in cells that will trigger passage (should be below 0.8, default %(default)s)",
    )
    trial_parser.add_argument(
        "--n_trials",
        default=3,
        type=int,
        help="How many times to run the full experiment/trial (default %(default)s)",
    )
    trial_parser.add_argument(
        "--n_steps",
        default=1_000_000,
        type=int,
        help="For how many steps (=virtual seconds) to run each trial (default %(default)s)",
    )
    trial_parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for tensors ('cpu', 'cuda', ..., default %(default)s)",
    )
    trial_parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="How many processes to use for transcription and translation (default %(default)s)",
    )
    trial_parser.add_argument(
        "--trial_max_time_h",
        default=6,
        type=int,
        help="Interrupt and stop trial after that many hours (default %(default)s)",
    )

    args = parser.parse_args()
    args.func(vars(args))

    print("done")
