from .src import cli
from .src import plots
from .src.describe_setup import describe_setup
from .src.describe_state import describe_state
from .src.describe_cell import describe_cell
from .src.describe_runs import describe_run, describe_pathway_training

# TODO: clustering: want to see actual cell proteomes
#       somehow within each cluster find representative cells
#       might be multiple
#       maybe cooperation already ongoing


# TODO: find things that are fundamentally different betwen

_CMDS = {
    "setup": describe_setup,
    "state": describe_state,
    "run": describe_run,
    "pathway-training": describe_pathway_training,
    "cells": describe_cell,
}


def main(kwargs: dict):
    plots.set_theme()
    cmd = kwargs.pop("cmd")
    cmd_fun = _CMDS[cmd]
    cmd_fun(kwargs)


if __name__ == "__main__":
    parser = cli.get_analysis_argparser()
    subparsers = parser.add_subparsers(dest="cmd")

    # describe setup
    cltr_parser = subparsers.add_parser(
        "setup",
        help="Describe experimental setup with chemistry",
    )

    # describe run
    run_parser = subparsers.add_parser(
        "run",
        help="Describe previous run",
    )
    cli.add_run_arg(parser=run_parser)

    # describe state
    state_parser = subparsers.add_parser(
        "state",
        help="Describe saved state with cells and molecules",
    )
    cli.add_state_arg(parser=state_parser)
    cli.add_molecules_arg(parser=state_parser)
    cli.add_grping_flags(parser=state_parser)

    # describe pathway training
    pathtrain_parser = subparsers.add_parser(
        "pathway-training",
        help="Describe previous pathway training run",
    )
    cli.add_runslist_arg(parser=pathtrain_parser)

    # describe cells
    cells_parser = subparsers.add_parser(
        "cells",
        help="Describe saved state with cells and molecules",
    )
    cli.add_state_arg(parser=cells_parser)
    cli.add_cell_arg(parser=cells_parser)
    cli.add_molecules_arg(parser=cells_parser)

    main(vars(parser.parse_args()))
    print("done")
