"""
Entrypoint for simulation. Run with:

  python -m e1_co2_fixing --help

"""
from typing import Callable
from pathlib import Path
import magicsoup as ms
from .src.chemistry import CHEMISTRY, WL_STAGES_MAP
from .src.train_pathway import run_trial as train_pathway_trial
from .src.grow_batch import run_trial as grow_batch_trial
from .src.grow_chemostat import run_trial as grow_chemostat_trial
from .src.shrink_genomes import run_trial as shrink_genomes_trial
from .src.util import Config
from .src import cli

_RUNS_DIR = Path(__file__).parent / "runs"


def _init_world_cmd(kwargs: dict):
    map_size = kwargs["map_size"]
    print(f"Initialing world with map_size={map_size}")
    world = ms.World(chemistry=CHEMISTRY, map_size=map_size)
    world.save(rundir=_RUNS_DIR)


def _run_trials_cmd(
    trialfun: Callable[[str, Config, dict], None], cmd: str, kwargs: dict
):
    kwargs["runs_dir"] = _RUNS_DIR
    config = Config.pop_from(kwargs)
    for trial_i in range(config.n_trials):
        run_name = f"{cmd}_{config.timestamp}_{trial_i}"
        print(f"Starting trial {run_name} on {config.device}")
        trialfun(run_name, config, kwargs)


_MAP: dict[str, Callable[[str, Config, dict], None]] = {
    "train-pathway": train_pathway_trial,
    "shrink-genomes": shrink_genomes_trial,
    "grow-chemostat": grow_chemostat_trial,
    "grow-batch": grow_batch_trial,
}


def main(kwargs: dict):
    cmd = kwargs.pop("cmd")
    if cmd == "init-world":
        _init_world_cmd(kwargs)
    else:
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
    cli.add_init_label_arg(parser=train_parser, extra="Use 'init' to spawn new cells")
    cli.add_batch_culture_args(parser=train_parser)
    cli.add_batch_culture_training_args(parser=train_parser)

    # grow cells in chemostat
    chemo_parser = subparsers.add_parser(
        "grow-chemostat",
        help="Grow cells in ChemoStat with E and CO2."
        " The ChemoStat will create a horizontal gradient with high E- and CO2-levels"
        " in the middle and 0 E and CO2 at the edges.",
    )
    cli.add_init_label_arg(parser=chemo_parser)
    cli.add_n_divisions_arg(parser=chemo_parser)

    # grow cells in batch culture
    batch_parser = subparsers.add_parser(
        "grow-batch",
        help="Grow cells in batch culture with E and CO2.",
    )
    cli.add_init_label_arg(parser=batch_parser)
    cli.add_n_divisions_arg(parser=batch_parser)
    cli.add_batch_culture_args(parser=batch_parser)

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
