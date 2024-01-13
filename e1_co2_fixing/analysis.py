import pandas as pd
from collections import Counter
import magicsoup as ms
from .src.chemistry import CHEMISTRY
from .src import cli
from .src import plots
from .src.util import (
    get_statedir,
    RUNS_DIR,
    save_img,
    genome_distances,
    cluster_cells,
    save_doc,
    hcat_imgs,
    vcat_imgs,
    write_table,
    table_to_markdown,
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
    write_table(df=mols, name="all_molecules.csv")
    molstab = table_to_markdown(
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
    write_table(df=reacts, name="all_reactions.csv")
    reactstab = table_to_markdown(
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
    statedir = get_statedir(label=kwargs["state"], runsdir=RUNS_DIR)
    world.load_state(statedir=statedir, ignore_cell_params=True)
    title = f"{statedir.parent.name}_{statedir.name}"

    # cell statistics
    proteomes = []
    for cell_i in range(world.n_cells):
        cell = world.get_cell(by_idx=cell_i)
        proteomes.append(list(set(str(d) for d in cell.proteome)))

    prot_cnts = Counter(dd for d in proteomes for dd in d).most_common(n=30)
    top_prots = [d[0] for d in prot_cnts]

    if kwargs["all_cells"]:
        colors = plots.tabcolors([])
        grouping = {"other": list(range(world.n_cells))}
        cellstats_img = plots.cellhists(world=world)
        mols_img = plots.molecule_concentrations(
            world=world,
            molnames=["CO2", "NADPH", "NADP", "ATP", "ADP", "acetyl-CoA", "HS-CoA"],
            grp2idxs=grouping,
            grp2col=colors,
            figsize=(7, 2),
        )
        protcnts_img = plots.protein_counts(
            proteins=top_prots,
            proteomes=proteomes,
            grp2idxs=grouping,
            grp2col=colors,
        )
        img = vcat_imgs(cellstats_img, mols_img)
        img = vcat_imgs(img, protcnts_img)
        save_img(img=img, name=f"{title}_all_cells.png")

    if kwargs["by_cell_labels"]:
        lab_cnts = Counter(world.cell_labels).most_common(10)
        top_labs = [d[0] for d in lab_cnts]
        colors = plots.tabcolors(top_labs)
        grouping = {
            k: [i for i, d in enumerate(world.cell_labels) if d == k] for k in top_labs
        }

        cm_img = plots.cellmap(world=world, grp2idxs=grouping, grp2col=colors)
        labs_img = plots.grp_counts(world=world, grp2idxs=grouping, grp2col=colors)
        stats_img = plots.cellboxes(
            world=world,
            grp2idxs={k: grouping[k] for k in top_labs[:3]},
            grp2col=colors,
        )
        mols_img = plots.molecule_concentrations(
            world=world,
            grp2idxs={k: grouping[k] for k in top_labs[:3]},
            grp2col=colors,
            molnames=["CO2", "NADPH", "NADP", "ATP", "ADP", "acetyl-CoA"],
        )
        counts_img = plots.protein_counts(
            proteins=top_prots,
            proteomes=proteomes,
            grp2idxs={k: grouping[k] for k in top_labs[:3]},
            grp2col=colors,
        )
        img = hcat_imgs(cm_img, labs_img)
        img = vcat_imgs(img, stats_img)
        img = vcat_imgs(img, mols_img)
        img = vcat_imgs(img, counts_img)
        save_img(img=img, name=f"{title}_cell_labels.png")

    if kwargs["by_genomic_clustering"]:
        D = genome_distances(genomes=world.cell_genomes)
        grouping = cluster_cells(D=D)
        colors = plots.tabcolors(list(grouping))
        records = []
        for clst, cell_idxs in grouping.items():
            for cell_i in cell_idxs:
                x, y = world.cell_positions[cell_i].tolist()
                records.append({"cluster": clst, "x": x, "y": y})

        clusters_df = pd.DataFrame.from_records(records)
        write_table(df=clusters_df, name=f"{title}_genomic_clustering.csv")

        cm_img = plots.cellmap(world=world, grp2idxs=grouping, grp2col=colors)
        labs_img = plots.grp_counts(world=world, grp2idxs=grouping, grp2col=colors)
        stats_img = plots.cellboxes(
            world=world,
            grp2idxs={k: grouping[k] for k in list(grouping)[:3]},
            grp2col=colors,
        )
        mols_img = plots.molecule_concentrations(
            world=world,
            grp2idxs={k: grouping[k] for k in list(grouping)[:3]},
            grp2col=colors,
            molnames=["CO2", "NADPH", "NADP", "ATP", "ADP", "acetyl-CoA"],
        )
        counts_img = plots.protein_counts(
            proteins=top_prots,
            proteomes=proteomes,
            grp2idxs={k: grouping[k] for k in list(grouping)[:3]},
            grp2col=colors,
        )
        img = hcat_imgs(cm_img, labs_img)
        img = vcat_imgs(img, stats_img)
        img = vcat_imgs(img, mols_img)
        img = vcat_imgs(img, counts_img)
        save_img(img=img, name=f"{title}_genomic_clustering.png")


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
    cli.add_state_flags(parser=state_parser)

    main(vars(parser.parse_args()))
    print("done")
