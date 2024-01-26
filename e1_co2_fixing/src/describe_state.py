import pandas as pd
from collections import Counter
import magicsoup as ms
from . import plots
from .util import (
    get_statedir,
    RUNS_DIR,
    save_img,
    genome_distances,
    dbscan_cells,
    hcat_imgs,
    vcat_imgs,
    write_table,
    proteome_distances,
)


def _all_cells_plot(
    world: ms.World,
    img_name: str,
    molecules: list[ms.Molecule],
    proteins: list[str],
    proteomes: list[list[str]],
):
    colors = {"all": plots.PRIM_COL}
    grouping = {"all": list(range(world.n_cells))}
    cellstats_img = plots.cellhists(world=world, proteomes=proteomes)
    mols_img = plots.molecule_boxes(
        world=world,
        molecules=molecules,
        grp2idxs=grouping,
        grp2col=colors,
        figsize=(8, len(molecules) * 0.5),
    )
    protcnts_img = plots.protein_counts(
        proteins=proteins,
        proteomes=proteomes,
        grp2idxs=grouping,
        grp2col=colors,
        figsize=(4, 0.2 * len(proteins)),
    )
    img = vcat_imgs(cellstats_img, mols_img)
    img = vcat_imgs(img, protcnts_img)
    save_img(img=img, name=img_name)


def _grouped_cells_plot(
    world: ms.World,
    img_name: str,
    grouping: dict[str, list[int]],
    colors: dict[str, str],
    molecules: list[ms.Molecule],
    proteins: list[str],
    proteomes: list[list[str]],
    n_subgrps=3,
):
    cm_img = plots.cellmap(world=world, grp2idxs=grouping, grp2col=colors)
    labs_img = plots.grp_counts(world=world, grp2idxs=grouping, grp2col=colors)
    stats_img = plots.cellboxes(
        world=world,
        proteomes=proteomes,
        grp2idxs={k: grouping[k] for k in list(grouping)[:n_subgrps]},
        grp2col=colors,
    )
    mols_img = plots.molecule_boxes(
        world=world,
        grp2idxs={k: grouping[k] for k in list(grouping)[:n_subgrps]},
        grp2col=colors,
        molecules=molecules,
        figsize=(8, len(molecules) * 0.5),
    )
    counts_img = plots.protein_counts(
        proteins=proteins,
        proteomes=proteomes,
        grp2idxs={k: grouping[k] for k in list(grouping)[:n_subgrps]},
        grp2col=colors,
        figsize=(4, 0.3 * len(proteins)),
    )
    img = hcat_imgs(cm_img, labs_img)
    img = vcat_imgs(img, stats_img)
    img = vcat_imgs(img, mols_img)
    img = vcat_imgs(img, counts_img)
    save_img(img=img, name=img_name)


def describe_state(kwargs: dict):
    world = ms.World.from_file(rundir=RUNS_DIR, device="cpu")
    statedir = get_statedir(label=kwargs["state"])
    world.load_state(statedir=statedir, ignore_cell_params=True)
    title = f"{statedir.parent.name}_{statedir.name}"
    molecules = [ms.Molecule.from_name(d) for d in kwargs["molecules"]]

    # cell statistics
    proteomes = []
    for cell_i in range(world.n_cells):
        cell = world.get_cell(by_idx=cell_i)
        proteomes.append(list(set(str(d) for d in cell.proteome)))

    prot_cnts = Counter(dd for d in proteomes for dd in d).most_common(
        n=kwargs["n_proteins"]
    )
    top_prots = [d[0] for d in prot_cnts]

    if kwargs["all_cells"]:
        _all_cells_plot(
            world=world,
            img_name=f"{title}_all_cells.png",
            proteins=top_prots,
            proteomes=proteomes,
            molecules=molecules,
        )

    if kwargs["by_cell_labels"]:
        lab_cnts = Counter(world.cell_labels).most_common(10)
        top_labs = [d[0] for d in lab_cnts]
        colors = plots.tabcolors(top_labs)
        grouping = {
            k: [i for i, d in enumerate(world.cell_labels) if d == k] for k in top_labs
        }
        _grouped_cells_plot(
            world=world,
            img_name=f"{title}_cell_labels.png",
            grouping=grouping,
            colors=colors,
            proteins=top_prots,
            proteomes=proteomes,
            molecules=molecules,
        )

    if kwargs["by_genomic_clustering"]:
        D = genome_distances(genomes=world.cell_genomes)
        grouping = dbscan_cells(D=D)
        colors = plots.tabcolors(list(grouping))
        records = []
        for clst, cell_idxs in grouping.items():
            center_idx = cell_idxs[D[cell_idxs].sum(axis=1).argmin()]
            for cell_i in cell_idxs:
                center = "Y" if center_idx == cell_i else "N"
                x, y = world.cell_positions[cell_i].tolist()
                records.append(
                    {"cluster": clst, "x": x, "y": y, "cell": cell_i, "center": center}
                )

        clusters_df = pd.DataFrame.from_records(records)
        write_table(df=clusters_df, name=f"{title}_genomic_clustering.csv")
        _grouped_cells_plot(
            world=world,
            img_name=f"{title}_genomic_clustering.png",
            grouping=grouping,
            colors=colors,
            proteins=top_prots,
            proteomes=proteomes,
            molecules=molecules,
        )

    if kwargs["by_proteomic_clustering"]:
        D = proteome_distances(proteomes=proteomes)
        grouping = dbscan_cells(D=D)
        colors = plots.tabcolors(list(grouping))
        records = []
        for clst, cell_idxs in grouping.items():
            center_idx = cell_idxs[D[cell_idxs].sum(axis=1).argmin()]
            for cell_i in cell_idxs:
                center = "Y" if center_idx == cell_i else "N"
                x, y = world.cell_positions[cell_i].tolist()
                records.append(
                    {"cluster": clst, "x": x, "y": y, "cell": cell_i, "center": center}
                )

        clusters_df = pd.DataFrame.from_records(records)
        write_table(df=clusters_df, name=f"{title}_proteomic_clustering.csv")
        _grouped_cells_plot(
            world=world,
            img_name=f"{title}_proteomic_clustering.png",
            grouping=grouping,
            colors=colors,
            proteins=top_prots,
            proteomes=proteomes,
            molecules=molecules,
        )
