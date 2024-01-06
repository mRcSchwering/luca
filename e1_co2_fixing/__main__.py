"""
Entrypoint for simulation. Run with:

  python -m e1_co2_fixing --help

"""
from typing import Callable
from pathlib import Path
import magicsoup as ms
from .src.chemistry import CHEMISTRY, WL_STAGES_MAP
from .src.init_cells import run_trial as init_cells_trial
from .src.train_pathway import run_trial as train_pathway_trial
from .src.validate_cells import run_trial as validate_cells_trial
from .src.shrink_genomes import run_trial as shrink_genomes_trial
from .src.util import Config
from .src import cli

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


def _run_trials_cmd(
    trialfun: Callable[[str, Config, dict], None], cmd: str, kwargs: dict
):
    config = Config.pop_from(kwargs)
    for trial_i in range(config.n_trials):
        run_name = f"{cmd}_{config.timestamp}_{trial_i}"
        print(f"Starting trial {run_name} on {config.device}")
        trialfun(run_name, config, kwargs)


_MAP: dict[str, Callable[[str, Config, dict], None]] = {
    "shrink-genomes": shrink_genomes_trial,
    "validate-cells": validate_cells_trial,
    "train-pathway": train_pathway_trial,
    "init-cells": init_cells_trial,
}


def main(kwargs: dict):
    cmd = kwargs.pop("cmd")
    if cmd == "init-world":
        _init_world_cmd(kwargs)
    trialfun = _MAP[cmd]
    _run_trials_cmd(trialfun=trialfun, cmd=cmd, kwargs=kwargs)
    print("done")


if __name__ == "__main__":
    parser = cli.get_argparser()
    subparsers = parser.add_subparsers(dest="cmd")

    # init world
    world_parser = subparsers.add_parser(
        "init-world",
        help="Initialize new world object."
        " This object will be used as a basis for all other runs.",
    )
    cli.add_mapsize_arg(parser=world_parser)

    # init cells
    cells_parser = subparsers.add_parser(
        "init-cells",
        help="Initialize mostly random cells"
        " that are able to grow on X in batch culture."
        " These cells will have transporters for X and E and"
        " are cultivated in X- and E-rich medium.",
    )
    cli.add_batch_culture_args(parser=cells_parser)
    cli.add_n_splits_arg(parser=cells_parser)

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
    cli.add_pathway_label_arg(parser=train_parser, choices=WL_STAGES_MAP)
    cli.add_init_label_arg(parser=train_parser)
    cli.add_batch_culture_args(parser=train_parser)
    cli.add_batch_culture_training_args(parser=train_parser)

    # validate cells
    val_parser = subparsers.add_parser(
        "validate-cells",
        help="Validate cell viability by growing them in ChemoStat with E and CO2."
        " The ChemoStat will create a horizontal gradient with high E- and CO2-levels"
        " in the middle and 0 E and CO2 at the edges.",
    )
    cli.add_init_label_arg(parser=val_parser)
    cli.add_n_divisions_arg(parser=val_parser)

    # shrink genomes
    shr_parser = subparsers.add_parser(
        "shrink-genomes",
        help="Grow cells in batch culture with E and CO2 while reducing genome-size-controlling k."
        " There are 3 phases: Init, with low mutation rate and high k;"
        " Adapt, with high mutation rate and decreasing k; Final, with low k and low mutation rate.",
    )
    cli.add_init_label_arg(parser=shr_parser)
    cli.add_batch_culture_args(parser=shr_parser)
    cli.add_batch_culture_training_args(parser=shr_parser)
    cli.add_shrink_genome_args(parser=shr_parser)

    args = parser.parse_args()
    main(vars(args))
