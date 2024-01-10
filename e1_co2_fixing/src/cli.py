from typing import Iterable
from argparse import ArgumentParser


# run


def get_run_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="Device for tensors ('cpu', 'cuda', ..., default %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        default=100_000,
        type=int,
        help="Maxmimum number of steps (=virtual seconds) for each trial"
        " (default %(default)s)",
    )
    parser.add_argument(
        "--max-steps-without-progress",
        default=1000,
        type=int,
        help="Maxmimum number of steps (=virtual seconds) without any progress advancement"
        " (default %(default)s)",
    )
    parser.add_argument(
        "--max-time-m",
        default=180,
        type=int,
        help="Interrupt and stop trial after that many minutes (default %(default)s)",
    )
    parser.add_argument(
        "--n-trials",
        default=1,
        type=int,
        help="How many times to culture cells (default %(default)s)",
    )
    parser.add_argument(
        "--substrates-init",
        default=100.0,
        type=float,
        help="Substrate concentration in fresh medium (default %(default)s)",
    )
    parser.add_argument(
        "--additives-init",
        default=10.0,
        type=float,
        help="Additives concentration in fresh medium (default %(default)s)",
    )
    return parser


def add_batch_culture_args(parser: ArgumentParser):
    parser.add_argument(
        "--min-confl",
        default=0.2,
        type=float,
        help="Minimum confluency. Cells are passaged to this minimum confluency"
        " (theoretically 0.13-0.2 should be best, default %(default)s)",
    )
    parser.add_argument(
        "--max_confl",
        default=0.7,
        type=float,
        help="Maximum confluency. Cells are passaged after reaching this confluency"
        " (should be below 0.8, default %(default)s)",
    )


def add_batch_culture_training_args(parser: ArgumentParser):
    parser.add_argument(
        "--n-init-splits",
        default=5.0,
        type=float,
        help="Number of passages in initial phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    parser.add_argument(
        "--n-adapt-splits",
        default=5.0,
        type=float,
        help="Number of passages in adaption phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    parser.add_argument(
        "--n-final-splits",
        default=5.0,
        type=float,
        help="Number of passages in final phase (default %(default)s)."
        " Only passages with high enough growth rate are counted (see min_gr).",
    )
    parser.add_argument(
        "--min-gr",
        default=0.05,
        type=float,
        help="Minimum average growth rate during passage for it to be considered"
        " successful (max. possible is 0.1, default %(default)s).",
    )
    parser.add_argument(
        "--mutation-rate-mult",
        default=100.0,
        type=float,
        help="By how much to multiply mutation and recombination rate during adaption phase"
        " (default %(default)s)",
    )


def add_shrink_genome_args(parser: ArgumentParser):
    parser.add_argument(
        "--from-k",
        default=3000.0,
        type=float,
        help="Starting value of genome-size-reducing k (default %(default)s)",
    )
    parser.add_argument(
        "--to-k",
        default=1500.0,
        type=float,
        help="Final value of genome-size-reducing k default %(default)s)",
    )


def add_init_label_arg(
    parser: ArgumentParser, extra="Use 'random' to spawn random cells"
):
    parser.add_argument(
        "init-label",
        type=str,
        help="Describes from where initial genomes are loaded."
        " E.g.  '2023-05-09_14-08_0:-1' to load genomes from run '2023-05-09_14-08_0'"
        " last saved state, or '2023-05-09_14-08_0/step=150' to load step 150."
        f" {extra}.",
    )


def add_mapsize_arg(parser: ArgumentParser):
    parser.add_argument(
        "--map-size",
        default=256,
        type=int,
        help="Number of pixels of 2D map in each direction (default %(default)s)",
    )


def add_pathway_label_arg(parser: ArgumentParser, choices: Iterable):
    parser.add_argument(
        "pathway-label",
        type=str,
        choices=choices,
        help="Label for the stage that should be trained."
        " Each stage starts with an initial phase in which cells grow in medium A."
        " In the adaption phase genomes are edited and medium is changed to B."
        " In the final phase cells grow in medium B.",
    )


def add_n_divisions_arg(parser: ArgumentParser):
    parser.add_argument(
        "--n-divisions",
        default=100.0,
        type=float,
        help="How many average cell divisions to let cells grow (default %(default)s)",
    )


# analysis


def get_analysis_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    return parser


def add_state_arg(parser: ArgumentParser):
    parser.add_argument(
        "state",
        type=str,
        help="Which state should be loaded?"
        " E.g.  '2023-05-09_14-08_0:-1' to load the last step from run '2023-05-09_14-08_0',"
        " or '2023-05-09_14-08_0/step=150' to load step 150.",
    )


def add_genomic_clustering_arg(parser: ArgumentParser):
    parser.add_argument(
        "--genomic-clustering",
        action="store_true",
        help="Whether to also do a genomic-clustering (default %(default)s)",
    )
