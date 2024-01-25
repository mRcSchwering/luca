import pandas as pd
import magicsoup as ms
from . import plots
from .util import (
    get_statedir,
    RUNS_DIR,
    save_img,
    vcat_imgs,
    write_table,
    write_table_to_md,
)


def describe_cell(kwargs: dict):
    world = ms.World.from_file(rundir=RUNS_DIR, device="cpu")
    statedir = get_statedir(label=kwargs["state"])
    world.load_state(statedir=statedir)
    title = f"{statedir.parent.name}_{statedir.name}"
    molecules = [ms.Molecule.from_name(d) for d in kwargs["molecules"]]

    if "," in kwargs["cell"]:
        x, y = kwargs["cell"].split(",")
        idx = world.get_cell(by_position=(int(x), int(y))).idx
    else:
        idx = int(kwargs["cell"])

    world.kill_cells(cell_idxs=list(set(range(world.n_cells)) - {idx}))

    cell = world.get_cell(by_idx=0)
    prots_df = pd.DataFrame.from_records(
        [{"CDS": i, "protein": str(d)} for i, d in enumerate(cell.proteome)]
    )
    write_table(df=prots_df, name=f"{title}_cell{idx}_proteins.csv")
    write_table_to_md(df=prots_df, name=f"{title}_cell{idx}_proteins.md")

    trnscrpt_img = plots.plot_genome_transcripts(cell=cell)

    records = []
    for t in range(20):
        cell = world.get_cell(by_idx=0)
        for m in molecules:
            i = world.chemistry.mol_2_idx[m]
            records.append(
                {
                    "t": t,
                    "m": m.name,
                    "n": cell.int_molecules[i].item(),
                    "l": "extracellular",
                }
            )
            records.append(
                {
                    "t": t,
                    "m": m.name,
                    "n": cell.ext_molecules[i].item(),
                    "l": "intracellular",
                }
            )
        world.enzymatic_activity()
        world.diffuse_molecules()
        world.degrade_molecules()

    mols_df = pd.DataFrame.from_records(records)
    colors = plots.tabcolors(["intracellular", "extracellular"])
    mols_img = plots.timeseries(df=mols_df, grp2col=colors, figsize=(8, len(molecules)))
    img = vcat_imgs(mols_img, trnscrpt_img)
    # img = hcat_imgs(cm_img, mols_img)
    save_img(img=img, name=f"{title}_cell{idx}.png")
