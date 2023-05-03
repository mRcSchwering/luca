"""
Entrypoint for simulation. Run with:

  python -m e1_co2_fixing.main --help

"""
import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
from .chemistry import PATHWAY_PHASES_MAP
from .util import init_world
from .train_pathway import run_trial

THIS_DIR = Path(__file__).parent


def init_world_cmd(kwargs: dict):
    rundir = THIS_DIR / "runs"
    map_size = kwargs["map_size"]
    print(f"Initialing world with map_size={map_size}")
    init_world(rundir=rundir, map_size=map_size)


def train_pathway_cmd(kwargs: dict):
    kwargs.pop("func")
    pathway = kwargs.pop("pathway")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        run_trial(
            pathway=pathway,
            device=device,
            n_workers=n_workers,
            name=f"{pathway}_{ts}_{trial_i}",
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
        "--pathway",
        default="WL",
        type=str,
        choices=PATHWAY_PHASES_MAP,
        help="Which pathway to train. Each training will occur in adaption phases"
        " (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_trials",
        default=3,
        type=int,
        help="How many times to try the training (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_adapt_gens",
        default=100.0,
        type=float,
        help="For how many generations to let cells adapt in a new phase"
        " (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_static_gens",
        default=100.0,
        type=float,
        help="For how many generations to let cells continue growing"
        " before starting the next adaption phase (default %(default)s)",
    )
    train_parser.add_argument(
        "--mut_scheme",
        default="linear",
        type=str,
        choices=["linear", "step", "none"],
        help="Mutation rate scheme used during adaption phase (default %(default)s)",
    )
    train_parser.add_argument(
        "--init_cell_cover",
        default=0.2,
        type=float,
        help="Ratio of map initially covered by cells (default %(default)s)",
    )
    train_parser.add_argument(
        "--split_ratio",
        default=0.2,
        type=float,
        help="Fraction of cells (to fully covered map) carried over during passage"
        " (theoretically 0.13-0.2 is best, default %(default)s)",
    )
    train_parser.add_argument(
        "--split_thresh_cells",
        default=0.7,
        type=float,
        help="Ratio of map covered in cells that will trigger passage"
        " (should be below 0.8, default %(default)s)",
    )
    train_parser.add_argument(
        "--split_thresh_mols",
        default=0.2,
        type=float,
        help="Trigger passage if CO2 or E levels (relative to initial levels)"
        " in medium go below this (default %(default)s)",
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
