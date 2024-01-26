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

    other_idxs = list(set(range(world.n_cells)) - {idx})
    world.kill_cells(cell_idxs=other_idxs)
    cell = world.get_cell(by_idx=0)
    ext_env = cell.ext_molecules.tolist()

    prots_df = pd.DataFrame.from_records(
        [{"CDS": i, "protein": str(d)} for i, d in enumerate(cell.proteome)]
    )
    write_table(df=prots_df, name=f"{title}_cell{idx}_proteins.csv")
    write_table_to_md(df=prots_df, name=f"{title}_cell{idx}_proteins.md")

    records = []
    for t in range(kwargs["n_steps"]):
        for i, d in enumerate(ext_env):
            world.molecule_map[i] = d
        cell = world.get_cell(by_idx=0)
        for m in molecules:
            i = world.chemistry.mol_2_idx[m]
            records.append({"t": t, "m": m.name, "n": cell.int_molecules[i].item()})
        world.enzymatic_activity()
        world.diffuse_molecules()
        world.degrade_molecules()

    mols_df = pd.DataFrame.from_records(records)
    mols_img = plots.timeseries(
        df=mols_df, figsize=(8, len(molecules)), strip_angle=-80
    )
    trnscrpt_img = plots.plot_genome_transcripts(cell=cell)
    img = vcat_imgs(mols_img, trnscrpt_img)
    save_img(img=img, name=f"{title}_cell{idx}.png")
