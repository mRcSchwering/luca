"""
Entrypoint for simulation. Run with:

  python -m e1_co2_fixing.main --help

"""
import datetime as dt
from argparse import ArgumentParser
from pathlib import Path
import magicsoup as ms
from .src.chemistry import CHEMISTRY, WL_STAGES_MAP
from .src.init_cells import run_trial as init_cells_trial
from .src.train_pathway import run_trial as train_pathway_trial
from .src.validate_cells import run_trial as validate_cells_trial

_RUNS_DIR = Path(__file__).parent / "runs"


def init_world_cmd(kwargs: dict):
    map_size = kwargs["map_size"]
    print(f"Initialing world with map_size={map_size}")
    world = ms.World(
        chemistry=CHEMISTRY,
        map_size=map_size,
        mol_map_init="zeros",
    )
    world.save(rundir=_RUNS_DIR)


def init_cells_cmd(kwargs: dict):
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


def train_pathway_cmd(kwargs: dict):
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


def validate_cells_cmd(kwargs: dict):
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


if __name__ == "__main__":
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # init world
    world_parser = subparsers.add_parser(
        "init_world",
        help="Initialize new world object."
        " This object will be used as a basis for all other runs.",
    )
    world_parser.set_defaults(func=init_world_cmd)
    world_parser.add_argument(
        "--map_size",
        default=128,
        type=int,
        help="Number of pixels of 2D map in each direction (default %(default)s)",
    )

    # init cells
    cells_parser = subparsers.add_parser(
        "init_cells",
        help="Initialize mostly random cells"
        " that are able to grow on X in batch culture",
    )
    cells_parser.set_defaults(func=init_cells_cmd)
    cells_parser.add_argument(
        "--init_cell_cover",
        default=0.2,
        type=float,
        help="Ratio of map initially covered by cells (default %(default)s)",
    )
    cells_parser.add_argument(
        "--genome_size",
        type=int,
        default=500,
        help="Initial genome size (default %(default)s).",
    )
    cells_parser.add_argument(
        "--substrates_init",
        default=100.0,
        type=float,
        help="Substrate concentration in medium (default %(default)s)",
    )
    cells_parser.add_argument(
        "--n_splits",
        default=5.0,
        type=float,
        help="How many passages to let cells grow" " (default %(default)s)",
    )
    cells_parser.add_argument(
        "--mol_divide_k",
        default=30.0,
        type=float,
        help="Affinity k for X-dependent cell division ([15;30], default %(default)s)",
    )
    cells_parser.add_argument(
        "--mol_kill_k",
        default=0.04,
        type=float,
        help="Affinity k for E-dependent cell death ([0.01;0.04], default %(default)s)",
    )
    cells_parser.add_argument(
        "--genome_kill_k",
        default=3_000.0,
        type=float,
        help="Affinity k for genome-size-dependent cell death"
        " ([2000;4000], default %(default)s)",
    )
    cells_parser.add_argument(
        "--mutation_rate",
        default=1e-6,
        type=float,
        help="Mutation rate (default %(default)s)",
    )
    cells_parser.add_argument(
        "--lgt_rate",
        default=1e-3,
        type=float,
        help="Lateral gene transfer rate (default %(default)s)",
    )
    cells_parser.add_argument(
        "--split_ratio",
        default=0.2,
        type=float,
        help="Fraction of cells (to fully covered map) carried over during passage"
        " (theoretically 0.13-0.2 is best, default %(default)s)",
    )
    cells_parser.add_argument(
        "--split_thresh",
        default=0.7,
        type=float,
        help="Ratio of map covered in cells that will trigger passage"
        " (should be below 0.8, default %(default)s)",
    )
    cells_parser.add_argument(
        "--n_trials",
        default=1,
        type=int,
        help="How many times to try the training (default %(default)s)",
    )
    cells_parser.add_argument(
        "--n_steps",
        default=1_000,
        type=int,
        help="Maxmimum number of steps (=virtual seconds) for each trial"
        " (default %(default)s)",
    )
    cells_parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for tensors ('cpu', 'cuda', ..., default %(default)s)",
    )
    cells_parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="How many processes to use for transcription and translation"
        " (default %(default)s)",
    )
    cells_parser.add_argument(
        "--trial_max_time_h",
        default=1,
        type=int,
        help="Interrupt and stop trial after that many hours (default %(default)s)",
    )

    # train pathway
    train_parser = subparsers.add_parser(
        "train_pathway",
        help="Train a CO2 fixing pathway in batch culture."
        " Training is done in multiple stages with different phases for each stage.",
    )
    train_parser.set_defaults(func=train_pathway_cmd)
    train_parser.add_argument(
        "pathway_label",
        type=str,
        choices=WL_STAGES_MAP,
        help="Label for the stage that should be trained."
        " Each stage starts with an initial phase in which cells grow in medium A."
        " In the adaption phase genomes are edited and medium is changed to B."
        " In the final phase cells grow in medium B.",
    )
    train_parser.add_argument(
        "init_label",
        type=str,
        help="Describes from where initial genomes are loaded."
        " E.g.  '2023-05-09_14-08_0:-1' to load genomes from run '2023-05-09_14-08_0'"
        " last saved state, or '2023-05-09_14-08_0/step=150' to load step 150.",
    )
    train_parser.add_argument(
        "--gene_size",
        type=int,
        default=200,
        help="Size of nucleotide sequence in which new genes will be added"
        " (default %(default)s).",
    )
    train_parser.add_argument(
        "--n_init_splits",
        default=5.0,
        type=float,
        help="Number of passages in initial phase" " (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_adapt_splits",
        default=5.0,
        type=float,
        help="Number of passages in adaption phase" " (default %(default)s)",
    )
    train_parser.add_argument(
        "--n_final_splits",
        default=5.0,
        type=float,
        help="Number of passages in final phase" " (default %(default)s)",
    )
    train_parser.add_argument(
        "--min_gr",
        default=0.05,
        type=float,
        help="Minimum growth rate to use for advancing training phases"
        " (max. possible is 0.1, default %(default)s)",
    )
    train_parser.add_argument(
        "--substrates_init",
        default=100.0,
        type=float,
        help="Substrate concentration in fresh medium (default %(default)s)",
    )
    train_parser.add_argument(
        "--additives_init",
        default=10.0,
        type=float,
        help="Additives concentration in fresh medium (default %(default)s)",
    )
    train_parser.add_argument(
        "--mutation_rate_high",
        default=1e-4,
        type=float,
        help="High mutation rate during adaption phase (default %(default)s)",
    )
    train_parser.add_argument(
        "--mutation_rate_low",
        default=1e-6,
        type=float,
        help="Low mutation rate during other phases (default %(default)s)",
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
        " (theoretically 0.13-0.2 should be best, default %(default)s)",
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
        help="How many times to try train this stage (default %(default)s)",
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

    # validate cells
    val_parser = subparsers.add_parser(
        "validate_cells",
        help="Validate cell viability by growing them in ChemoStat with E and CO2.",
    )
    val_parser.set_defaults(func=validate_cells_cmd)
    val_parser.add_argument(
        "init_label",
        type=str,
        help="Describes from where initial genomes are loaded."
        " E.g.  '2023-05-09_14-08_0:-1' to load genomes from run '2023-05-09_14-08_0'"
        " last saved state, or '2023-05-09_14-08_0/step=150' to load step 150.",
    )
    val_parser.add_argument(
        "--substrates_init",
        default=100.0,
        type=float,
        help="Substrate concentration in feed medium (default %(default)s)",
    )
    val_parser.add_argument(
        "--additives_init",
        default=10.0,
        type=float,
        help="Additives concentration in feed medium (default %(default)s)",
    )
    val_parser.add_argument(
        "--n_divisions",
        default=100.0,
        type=float,
        help="How many average cell divisions to let cells grow"
        " (default %(default)s)",
    )
    val_parser.add_argument(
        "--mol_divide_k",
        default=30.0,
        type=float,
        help="Affinity k for X-dependent cell division ([15;30], default %(default)s)",
    )
    val_parser.add_argument(
        "--mol_kill_k",
        default=0.04,
        type=float,
        help="Affinity k for E-dependent cell death ([0.01;0.04], default %(default)s)",
    )
    val_parser.add_argument(
        "--genome_kill_k",
        default=3_000.0,
        type=float,
        help="Affinity k for genome-size-dependent cell death"
        " ([2000;4000], default %(default)s)",
    )
    val_parser.add_argument(
        "--mutation_rate",
        default=1e-6,
        type=float,
        help="Mutation rate (default %(default)s)",
    )
    val_parser.add_argument(
        "--lgt_rate",
        default=1e-3,
        type=float,
        help="Lateral gene transfer rate (default %(default)s)",
    )
    val_parser.add_argument(
        "--n_trials",
        default=1,
        type=int,
        help="How many times to repeat this experiment (default %(default)s)",
    )
    val_parser.add_argument(
        "--n_steps",
        default=1_000,
        type=int,
        help="Maxmimum number of steps (=virtual seconds) for each trial"
        " (default %(default)s)",
    )
    val_parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for tensors ('cpu', 'cuda', ..., default %(default)s)",
    )
    val_parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="How many processes to use for transcription and translation"
        " (default %(default)s)",
    )
    val_parser.add_argument(
        "--trial_max_time_h",
        default=1,
        type=int,
        help="Interrupt and stop trial after that many hours (default %(default)s)",
    )

    args = parser.parse_args()
    args.func(vars(args))
    print("done")
