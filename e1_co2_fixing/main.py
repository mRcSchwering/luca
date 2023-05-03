"""
Entrypoint for simulation. Run with:

  python -m e1_co2_fixing.main --help

"""
from argparse import ArgumentParser
from pathlib import Path
from .chemistry import PATHWAY_PHASES_MAP
from .util import init_world
from .experiment import run_trials

THIS_DIR = Path(__file__).parent


def init(kwargs: dict):
    rundir = THIS_DIR / "runs"
    map_size = kwargs["map_size"]
    print(f"Initialing world with map_size={map_size}")
    init_world(rundir=rundir, map_size=map_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # init
    init_parser = subparsers.add_parser("init", help="initialize world object")
    init_parser.set_defaults(func=init)
    init_parser.add_argument(
        "--map_size",
        default=128,
        type=int,
        help="Number of pixels of 2D map in each direction (default %(default)s)",
    )

    # trials
    trial_parser = subparsers.add_parser("trials", help="run trials")
    trial_parser.set_defaults(func=run_trials)
    trial_parser.add_argument(
        "--pathway",
        default="WL",
        type=str,
        choices=PATHWAY_PHASES_MAP,
        help="Which pathway to train (default %(default)s)",
    )
    trial_parser.add_argument(
        "--n_adapt_gens",
        default=100.0,
        type=float,
        help="Over how many generations to reduce complex medium to minimal medium (default %(default)s)",
    )
    trial_parser.add_argument(
        "--n_static_gens",
        default=100.0,
        type=float,
        help="Over how many generations to reduce complex medium to minimal medium (default %(default)s)",
    )
    trial_parser.add_argument(
        "--mut_scheme",
        default="linear",
        type=str,
        choices=["linear", "step", "none"],
        help="Mutation rate scheme used during adaption phase (default %(default)s)",
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
        help="Fraction of cells (to fully covered map) carried over during passage (theoretically 0.13-0.2 is best, default %(default)s)",
    )
    trial_parser.add_argument(
        "--split_thresh_cells",
        default=0.7,
        type=float,
        help="Ratio of map covered in cells that will trigger passage (should be below 0.8, default %(default)s)",
    )
    trial_parser.add_argument(
        "--split_thresh_mols",
        default=0.2,
        type=float,
        help="Trigger passage if relative CO2 or E levels in medium go below this (default %(default)s)",
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
