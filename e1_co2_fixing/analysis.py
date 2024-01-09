import pandas as pd
import magicsoup as ms
from .src import cli
from .src import plots
from .src import tables
from .src.util import load_cells, RUNS_DIR, save_img, genome_distances, cluster_cells


def _describe_state(kwargs: dict):
    label = kwargs["state"]
    world = ms.World.from_file(rundir=RUNS_DIR, device="cpu")
    statedir = load_cells(world=world, label=label, runsdir=RUNS_DIR, reset_cells=False)
    title = f"{statedir.parent.name}_{statedir.name}"

    # cell statistics
    img = plots.state_cell_stats(
        divisions=world.cell_divisions.tolist(),
        lifetimes=world.cell_lifetimes.tolist(),
        genome_sizes=[len(d) for d in world.cell_genomes],
    )
    save_img(img=img, name=f"{title}_cell_stats.png")

    # cell labels
    labels_df = pd.DataFrame(
        {
            "label": world.cell_labels,
            "x": world.cell_positions[:, 0].tolist(),
            "y": world.cell_positions[:, 1].tolist(),
        }
    )
    img = plots.marked_cellmap(df=labels_df, top_n=10, map_size=world.map_size)
    save_img(img=img, name=f"{title}_cell_labels.png")

    # genome clustering
    kill_idxs = [i for i, d in enumerate(world.cell_genomes) if len(d) < 100]
    world.kill_cells(cell_idxs=kill_idxs)
    D = genome_distances(world=world)
    labels = cluster_cells(D=D)
    clusters_df = pd.DataFrame(
        {
            "label": [f"c{d}" if d >= 0 else "other" for d in labels],
            "x": world.cell_positions[:, 0].tolist(),
            "y": world.cell_positions[:, 1].tolist(),
        }
    )
    tables.write_table(df=clusters_df, name=f"{title}_genome_clusters.csv")
    img = plots.marked_cellmap(df=clusters_df, map_size=world.map_size)
    save_img(img=img, name=f"{title}_genome_clusters.png")


_CMDS = {
    "describe-state": _describe_state,
}


def main(kwargs: dict):
    plots.set_theme()
    cmd = kwargs.pop("cmd")
    cmd_fun = _CMDS[cmd]
    cmd_fun(kwargs)


if __name__ == "__main__":
    parser = cli.get_analysis_argparser()
    subparsers = parser.add_subparsers(dest="cmd")

    # 1
    state_parser = subparsers.add_parser(
        "describe-state",
        help="Describe cells in state",
    )
    cli.add_state_arg(parser=state_parser)

    main(vars(parser.parse_args()))
    print("done")
