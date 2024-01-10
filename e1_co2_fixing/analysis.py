import pandas as pd
from collections import Counter
import magicsoup as ms
from .src.chemistry import CHEMISTRY
from .src import cli
from .src import plots
from .src import tables
from .src.util import (
    load_cells,
    RUNS_DIR,
    save_img,
    genome_distances,
    cluster_cells,
    save_doc,
)


def _describe_chemistry(_: dict):
    records = [
        {
            "index": i,
            "name": d.name,
            "energy[kJ]": int(d.energy / 1000),
            "diffusivity": d.diffusivity,
            "permeability": d.permeability,
        }
        for i, d in enumerate(CHEMISTRY.molecules)
    ]
    mols = pd.DataFrame.from_records(records)
    tables.write_table(df=mols, name="all_molecules.csv")
    molstab = tables.to_markdown(
        df=mols,
        name="1.1 Molecules",
        descr="Definition of all molecules."
        " Energy affects reaction equilibriums,"
        " diffusivity allows diffusion,"
        " permeability allows permeating cell membranes.",
    )

    records = []
    for subs, prods in CHEMISTRY.reactions:
        subs_n = Counter(str(d) for d in subs)
        prods_n = Counter([str(d) for d in prods])
        subs_s = " + ".join([f"{d} {k}" for k, d in subs_n.items()])
        prods_s = " + ".join([f"{d} {k}" for k, d in prods_n.items()])
        energy = sum(d.energy for d in prods) - sum(d.energy for d in subs)
        react = f"{subs_s} $\\rightleftharpoons$ {prods_s}"
        records.append({"reaction": react, "energy [kJ]": int(energy / 1000)})

    reacts = pd.DataFrame.from_records(records)
    tables.write_table(df=reacts, name="all_reactions.csv")
    reactstab = tables.to_markdown(
        df=reacts,
        name="1.2 Reactions",
        descr="Definition of all reactions. Energy affects reaction equilibriums.",
    )

    summary = ["\n## Chemistry\n\n"]
    summary.append("\n### Molecules\n\n")
    summary.append(molstab + "\n")
    summary.append("\n### Reactions\n\n")
    summary.append(reactstab + "\n")
    save_doc(content=summary, name="chemistry.md")


def _describe_state(kwargs: dict):
    world = ms.World.from_file(rundir=RUNS_DIR, device="cpu")
    statedir = load_cells(
        world=world, label=kwargs["state"], runsdir=RUNS_DIR, reset_cells=False
    )
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

    if kwargs["genomic_clustering"]:
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
        # TODO: most common proteins and map with cluster highlted


_CMDS = {
    "describe-chemistry": _describe_chemistry,
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

    # describe cells
    chem_parser = subparsers.add_parser(
        "describe-chemistry",
        help="Describe chemistry",
    )

    # describe cells
    state_parser = subparsers.add_parser(
        "describe-state",
        help="Describe cells in state",
    )
    cli.add_state_arg(parser=state_parser)
    cli.add_genomic_clustering_arg(parser=state_parser)

    main(vars(parser.parse_args()))
    print("done")
