import multiprocessing as mp
from functools import partial
import numpy as np
from Levenshtein import distance as ls_dist
import magicsoup as ms
from .src import cli
from .src.util import load_cells, RUNS_DIR


label = "train-pathway_2024-01-08_21-59_2:-1"
world = ms.World.from_file(rundir=RUNS_DIR, device="cpu")
load_cells(world=world, label=label, runsdir=RUNS_DIR, reset_cells=False)


def _genome_dists_row(i: int, genomes: list[str], minlen=1) -> np.ndarray:
    n = len(genomes)
    gi = genomes[i]
    ni = len(gi)
    if ni < minlen:
        return np.full((n,), float("nan"))
    Di = np.zeros((n,))
    for j in range(i + 1, world.n_cells):
        gj = world.cell_genomes[j]
        nj = len(gj)
        if nj > minlen:
            Di[j] = ls_dist(gi, gj) / max(ni, nj)
        else:
            Di[j] = float("nan")
    return Di


def genome_distances(world: ms.World) -> np.ndarray:
    args = [(d, world.cell_genomes) for d in range(world.n_cells)]
    with mp.Pool() as pool:
        arrs = pool.starmap(_genome_dists_row, args)
    return np.stack(arrs)


def _calc_genomic_distance_matrix(world: ms.World):
    D = genome_distances(world=world)
    i_min = D.sum(axis=0).argmin()
    D[i_min]
    cell = world.get_cell(by_idx=1500)
    for protein in cell.proteome:
        print(str(protein))


def _describe_cells_cmd(kwargs: dict):
    label = kwargs["state"]
    world = ms.World.from_file(rundir=RUNS_DIR, device="cpu")
    load_cells(world=world, label=label, runsdir=RUNS_DIR, reset_cells=False)


_CMDS = {
    "describe-cells": _describe_cells_cmd,
}


def main(kwargs: dict):
    cmd = kwargs.pop("cmd")
    cmd_fun = _CMDS[cmd]
    cmd_fun(kwargs)


if __name__ == "__main__":
    parser = cli.get_analysis_argparser()
    subparsers = parser.add_subparsers(dest="cmd")

    # 1
    cells_parser = subparsers.add_parser(
        "describe-cells",
        help="Describe cells in state",
    )
    cli.add_state_arg(parser=cells_parser)

    args = parser.parse_args()
    main(vars(args))
    print("done")
