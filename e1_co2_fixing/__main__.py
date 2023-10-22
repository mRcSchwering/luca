"""
Entrypoint for simulation. Run with:

  python -m e1_co2_fixing --help

"""
import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
import magicsoup as ms
from .src.chemistry import CHEMISTRY, WL_STAGES_MAP
from .src.init_cells import run_trial as init_cells_trial
from .src.train_pathway import run_trial as train_pathway_trial
from .src.validate_cells import run_trial as validate_cells_trial
from .src.shrink_genomes import run_trial as shrink_genomes_trial

_RUNS_DIR = Path(__file__).parent / "runs"


def _init_world_cmd(kwargs: dict):
    map_size = kwargs["map_size"]
    print(f"Initialing world with map_size={map_size}")
    world = ms.World(
        chemistry=CHEMISTRY,
        map_size=map_size,
        mol_map_init="zeros",
    )
    world.save(rundir=_RUNS_DIR)


def _init_cells_cmd(kwargs: dict):
    kwargs.pop("func")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        init_cells_trial(
            device=device,
            n_workers=n_workers,
            runs_dir=_RUNS_DIR,
            run_name=f"{ts}_{trial_i}",
            n_steps=n_steps,
            trial_max_time_s=trial_max_time_s,
            hparams=kwargs,
        )


def _train_pathway_cmd(kwargs: dict):
    kwargs.pop("func")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        train_pathway_trial(
            device=device,
            n_workers=n_workers,
            runs_dir=_RUNS_DIR,
            run_name=f"{ts}_{trial_i}",
            n_steps=n_steps,
            trial_max_time_s=trial_max_time_s,
            hparams=kwargs,
        )


def _validate_cells_cmd(kwargs: dict):
    kwargs.pop("func")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        validate_cells_trial(
            device=device,
            n_workers=n_workers,
            runs_dir=_RUNS_DIR,
            run_name=f"{ts}_{trial_i}",
            n_steps=n_steps,
            trial_max_time_s=trial_max_time_s,
            hparams=kwargs,
        )


def _shrink_genomes_cmd(kwargs: dict):
    kwargs.pop("func")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        shrink_genomes_trial(
            device=device,
            n_workers=n_workers,
            runs_dir=_RUNS_DIR,
            run_name=f"{ts}_{trial_i}",
            n_steps=n_steps,
            trial_max_time_s=trial_max_time_s,
            hparams=kwargs,
        )


def _add_batch_culture_args(subparser: ArgumentParser):
    subparser.add_argument(
        "--split-ratio",
        default=0.2,
        type=float,
        help="Fraction of cells (to fully covered map) carried over during passage"
        " (theoretically 0.13-0.2 should be best, default %(default)s)",
    )
    subparser.add_argument(
        "--split-thresh",
        default=0.7,
        type=float,
        help="Ratio of map covered in cells that will trigger passage"
        " (should be below 0.8, default %(default)s)",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for tensors ('cpu', 'cuda', ..., default %(default)s)",
    )
    parser.add_argument(
        "--n-workers",
        default=4,
        type=int,
        help="How many processes to use for transcription and translation"
        " (default %(default)s)",
    )
    parser.add_argument(
        "--n-steps",
        default=100_000,
        type=int,
        help="Maxmimum number of steps (=virtual seconds) for each trial"
        " (default %(default)s)",
    )
    parser.add_argument(
        "--trial-max-time-h",
        default=3,
        type=int,
        help="Interrupt and stop trial after that many hours (default %(default)s)",
    )
    parser.add_argument(
        "--n-trials",
        default=1,
        type=int,
        help="How many times to try experiment (default %(default)s)",
    )
    parser.add_argument(
        "--substrates-init",
        default=100.0,
        type=float,
        help="Substrate concentration in medium (default %(default)s)",
    )
    parser.add_argument(
        "--additives-init",
        default=10.0,
        type=float,
        help="Additives concentration in medium (default %(default)s)",
    )

    subparsers = parser.add_subparsers()

    # init world
    world_parser = subparsers.add_parser(
        "init-world",
        help="Initialize new world object."
        " This object will be used as a basis for all other runs.",
    )
    world_parser.set_defaults(func=_init_world_cmd)
    world_parser.add_argument(
        "--map-size",
        default=128,
        type=int,
        help="Number of pixels of 2D map in each direction (default %(default)s)",
    )

    # init cells
    cells_parser = subparsers.add_parser(
        "init-cells",
        help="Initialize mostly random cells"
        " that are able to grow on X in batch culture."
        " These cells will have transporters for X and E and"
        " are cultivated in X- and E-rich medium.",
    )
    cells_parser.set_defaults(func=_init_cells_cmd)
    _add_batch_culture_args(subparser=cells_parser)
    cells_parser.add_argument(
        "--init-cell-cover",
        default=0.2,
        type=float,
        help="Ratio of map initially covered by cells (default %(default)s)",
    )
    cells_parser.add_argument(
        "--genome-size",
        type=int,
        default=500,
        help="Initial genome size (default %(default)s).",
    )
    cells_parser.add_argument(
        "--n-splits",
        default=5.0,
        type=float,
        help="How many passages to let cells grow" " (default %(default)s)",
    )

    # train pathway
    train_parser = subparsers.add_parser(
        "train-pathway",
        help="Train a CO2 fixing pathway in batch culture."
        " Training is done in multiple stages with different phases for each stage."
        " Each stage incrementally trains 1 part of the pathway with the remaining"
        " cells of the previous. Each stage has 3 phases: init, adapt, final."
        " Init grows cells in previous medium, adapt changes to target medium and"
        " increases mutation rate, final grows cells in target medium at base rate.",
    )
    train_parser.set_defaults(func=_train_pathway_cmd)
    train_parser.add_argument(
        "pathway-label",
        type=str,
        choices=WL_STAGES_MAP,
        help="Label for the stage that should be trained."
        " Each stage starts with an initial phase in which cells grow in medium A."
        " In the adaption phase genomes are edited and medium is changed to B."
        " In the final phase cells grow in medium B.",
    )
    train_parser.add_argument(
        "init-label",
        type=str,
        help="Describes from where initial genomes are loaded."
        " E.g.  '2023-05-09_14-08_0:-1' to load genomes from run '2023-05-09_14-08_0'"
        " last saved state, or '2023-05-09_14-08_0/step=150' to load step 150.",
    )
    _add_batch_culture_args(subparser=train_parser)
    train_parser.add_argument(
        "--gene-size",
        type=int,
        default=200,
        help="Size of nucleotide sequence in which new genes will be added"
        " (default %(default)s).",
    )
    train_parser.add_argument(
        "--n-init-splits",
        default=5.0,
        type=float,
        help="Number of passages in initial phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    train_parser.add_argument(
        "--n-adapt-splits",
        default=5.0,
        type=float,
        help="Number of passages in adaption phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    train_parser.add_argument(
        "--n-final-splits",
        default=5.0,
        type=float,
        help="Number of passages in final phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    train_parser.add_argument(
        "--min-gr",
        default=0.05,
        type=float,
        help="Minimum average growth rate during passage for it to be considered"
        " successful (max. possible is 0.1, default %(default)s).",
    )
    train_parser.add_argument(
        "--mutation-rate-mult",
        default=10.0,
        type=float,
        help="By how much to multiply mutation rate during adaption phase"
        " (default %(default)s)",
    )

    # validate cells
    val_parser = subparsers.add_parser(
        "validate-cells",
        help="Validate cell viability by growing them in ChemoStat with E and CO2."
        " The ChemoStat will create a horizontal gradient with high E- and CO2-levels"
        " in the middle and 0 E and CO2 at the edges.",
    )
    val_parser.set_defaults(func=_validate_cells_cmd)
    val_parser.add_argument(
        "init-label",
        type=str,
        help="Describes from where initial genomes are loaded."
        " E.g.  '2023-05-09_14-08_0:-1' to load genomes from run '2023-05-09_14-08_0'"
        " last saved state, or '2023-05-09_14-08_0/step=150' to load step 150.",
    )
    val_parser.add_argument(
        "--n-divisions",
        default=100.0,
        type=float,
        help="How many average cell divisions to let cells grow"
        " (default %(default)s)",
    )
    val_parser.add_argument(
        "--genome-size-k",
        default=3000.0,
        type=float,
        help="M.M. constant that penalizes genome size (default %(default)s)"
        " (default %(default)s)",
    )

    # shrink genomes
    shr_parser = subparsers.add_parser(
        "shrink-genomes",
        help="Grow cells in batch culture with E and CO2 while reducing genome-size-controlling k."
        " There are 3 phases: Init, with low mutation rate and high k;"
        " Adapt, with high mutation rate and decreasing k; Final, with low k and low mutation rate.",
    )
    shr_parser.set_defaults(func=_shrink_genomes_cmd)
    shr_parser.add_argument(
        "init-label",
        type=str,
        help="Describes from where initial genomes are loaded."
        " E.g.  '2023-05-09_14-08_0:-1' to load genomes from run '2023-05-09_14-08_0'"
        " last saved state, or '2023-05-09_14-08_0/step=150' to load step 150.",
    )
    _add_batch_culture_args(subparser=shr_parser)
    shr_parser.add_argument(
        "--n-init-splits",
        default=10.0,
        type=float,
        help="Number of passages in initial phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    shr_parser.add_argument(
        "--n-adapt-splits",
        default=50.0,
        type=float,
        help="Number of passages in adaption phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    shr_parser.add_argument(
        "--n-final-splits",
        default=10.0,
        type=float,
        help="Number of passages in final phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    shr_parser.add_argument(
        "--min-gr",
        default=0.05,
        type=float,
        help="Minimum average growth rate during passage for it to be considered"
        " successful (max. possible is 0.1, default %(default)s).",
    )
    shr_parser.add_argument(
        "--mutation-rate-mult",
        default=10.0,
        type=float,
        help="By how much to multiply mutation rate during adaption phase"
        " (default %(default)s)",
    )
    shr_parser.add_argument(
        "--from-k",
        default=3000.0,
        type=float,
        help="Starting value of genome-size-reducing k (default %(default)s)",
    )
    shr_parser.add_argument(
        "--to-k",
        default=1500.0,
        type=float,
        help="Final value of genome-size-reducing k default %(default)s)",
    )

    args = parser.parse_args()
    args.func(vars(args))
    print("done")
