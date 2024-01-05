from argparse import ArgumentParser


def get_argparser() -> ArgumentParser:
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
        "--trial-max-time-m",
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


def add_init_label_args(parser: ArgumentParser):
    parser.add_argument(
        "init-label",
        type=str,
        help="Describes from where initial genomes are loaded."
        " E.g.  '2023-05-09_14-08_0:-1' to load genomes from run '2023-05-09_14-08_0'"
        " last saved state, or '2023-05-09_14-08_0/step=150' to load step 150.",
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
        default=10.0,
        type=float,
        help="By how much to multiply mutation and recombination rate during adaption phase"
        " (default %(default)s)",
    )
