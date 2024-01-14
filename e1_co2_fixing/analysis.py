from .src import cli
from .src import plots
from .src.describe_culturing import create as describe_culturing
from .src.describe_chemistry import create as describe_chemistry
from .src.describe_state import create as describe_state


_CMDS = {
    "describe-culturing": describe_culturing,
    "describe-chemistry": describe_chemistry,
    "describe-state": describe_state,
}


def main(kwargs: dict):
    plots.set_theme()
    cmd = kwargs.pop("cmd")
    cmd_fun = _CMDS[cmd]
    cmd_fun(kwargs)


if __name__ == "__main__":
    parser = cli.get_analysis_argparser()
    subparsers = parser.add_subparsers(dest="cmd")

    # describe chemistry
    chem_parser = subparsers.add_parser(
        "describe-chemistry",
        help="Describe chemistry",
    )

    # describe culturing
    cltr_parser = subparsers.add_parser(
        "describe-culturing",
        help="Describe culturing",
    )

    # describe cells
    state_parser = subparsers.add_parser(
        "describe-state",
        help="Describe cells in state",
    )
    cli.add_state_arg(parser=state_parser)
    cli.add_state_flags(parser=state_parser)

    main(vars(parser.parse_args()))
    print("done")
