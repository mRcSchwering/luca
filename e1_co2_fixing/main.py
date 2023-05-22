"""
Entrypoint for simulation. Run with:

  python -m e1_co2_fixing.main --help

"""
import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
import magicsoup as ms
from .chemistry import CHEMISTRY, WL_STAGES_MAP
from .train_pathway import run_trial

THIS_DIR = Path(__file__).parent


def init_world_cmd(kwargs: dict):
    rundir = THIS_DIR / "runs"
    map_size = kwargs["map_size"]
    print(f"Initialing world with map_size={map_size}")
    world = ms.World(
        chemistry=CHEMISTRY,
        map_size=map_size,
        mol_map_init="zeros",
    )
    world.save(rundir=rundir)


def train_pathway_cmd(kwargs: dict):
    kwargs.pop("func")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        run_trial(
            device=device,
            n_workers=n_workers,
            run_name=f"{ts}_{trial_i}",
            n_steps=n_steps,
            trial_max_time_s=trial_max_time_s,
            hparams=kwargs,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # init world
    init_parser = subparsers.add_parser("init_world", help="initialize world object")
    init_parser.set_defaults(func=init_world_cmd)
    init_parser.add_argument(
        "--map_size",
        default=128,
        type=int,
        help="Number of pixels of 2D map in each direction (default %(default)s)",
    )

    # train pathway
    train_parser = subparsers.add_parser(
        "train_pathway", help="train a CO2 fixing pathway in trials"
    )
    train_parser.set_defaults(func=train_pathway_cmd)
    train_parser.add_argument(
        "pathway_label",
        type=str,
        choices=WL_STAGES_MAP,
        help="Which part of which pathway to train",
    )
    train_parser.add_argument(
        "init_label",
        type=str,
        help="""Describes from where initial genomes are loaded. E.g. 
        `2023-05-09_14-08_0:-1` to load genomes from run '2023-05-09_14-08_0' last 
        saved state, or `2023-05-09_14-08_0/step=150` to load step 150. `random`
         to initialize random genomes (default %(default)s)""",
    )
    train_parser.add_argument(
        "--init_cell_cover",
        default=0.2,
        type=float,
        help="Ratio of map initially covered by cells (default %(default)s)",
    )
    train_parser.add_argument(
        "--gene_size",
        type=int,
        default=200,
        help="Sequence size in which genes will be added (default %(default)s).",
    )
    train_parser.add_argument(
        "--n_init_splits",
        default=2.0,
        type=float,
        help="How many passages to let cells grow before starting adaption"
        " (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_adapt_splits",
        default=2.0,
        type=float,
        help="How many passages to let cells adapt to new substrates"
        " (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_final_splits",
        default=4.0,
        type=float,
        help="How many passages to grow cells after adaption finished"
        " (default %(default)s)",
    )
    train_parser.add_argument(
        "--min_gr",
        default=0.05,
        type=float,
        help="Minimum growth rate to use for advancing training phases"
        " (max. theoretically possible is 0.1, default %(default)s)",
    )
    train_parser.add_argument(
        "--substrates_init",
        default=100.0,
        type=float,
        help="Substrate concentration in medium (default %(default)s)",
    )
    train_parser.add_argument(
        "--additives_init",
        default=10.0,
        type=float,
        help="Additives concentration in medium (default %(default)s)",
    )
    train_parser.add_argument(
        "--mutation_rate_high",
        default=1e-4,
        type=float,
        help="High mutation rate during adaption (default %(default)s)",
    )
    train_parser.add_argument(
        "--mutation_rate_low",
        default=1e-6,
        type=float,
        help="Low mutation rate during normal growth (default %(default)s)",
    )
    train_parser.add_argument(
        "--mol_divide_k",
        default=30.0,
        type=float,
        help="Affinity k for X-dependent cell division ([15;30], default %(default)s)",
    )
    train_parser.add_argument(
        "--mol_kill_k",
        default=0.04,
        type=float,
        help="Affinity k for E-dependent cell death ([0.01;0.04], default %(default)s)",
    )
    train_parser.add_argument(
        "--genome_kill_k",
        default=3_000.0,
        type=float,
        help="Affinity k for genome-size-dependent cell death"
        " ([2000;4000], default %(default)s)",
    )
    train_parser.add_argument(
        "--lgt_rate",
        default=1e-3,
        type=float,
        help="Lateral gene transfer rate (default %(default)s)",
    )
    train_parser.add_argument(
        "--split_ratio",
        default=0.2,
        type=float,
        help="Fraction of cells (to fully covered map) carried over during passage"
        " (theoretically 0.13-0.2 is best, default %(default)s)",
    )
    train_parser.add_argument(
        "--split_thresh",
        default=0.7,
        type=float,
        help="Ratio of map covered in cells that will trigger passage"
        " (should be below 0.8, default %(default)s)",
    )
    train_parser.add_argument(
        "--n_trials",
        default=3,
        type=int,
        help="How many times to try the training (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_steps",
        default=1_000_000,
        type=int,
        help="Maxmimum number of steps (=virtual seconds) for each trial"
        " (default %(default)s)",
    )
    train_parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for tensors ('cpu', 'cuda', ..., default %(default)s)",
    )
    train_parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="How many processes to use for transcription and translation"
        " (default %(default)s)",
    )
    train_parser.add_argument(
        "--trial_max_time_h",
        default=6,
        type=int,
        help="Interrupt and stop trial after that many hours (default %(default)s)",
    )

    args = parser.parse_args()
    args.func(vars(args))
    print("done")
