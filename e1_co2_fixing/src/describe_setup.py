from collections import Counter
from .chemistry import CHEMISTRY
import pandas as pd
from .util import (
    save_img,
    hcat_imgs,
    vcat_imgs,
    write_table,
    write_table_to_md,
)
from . import plots


def _describe_chemistry():
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
    write_table_to_md(df=mols, name="molecules.md")

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
    write_table_to_md(df=reacts, name="1.2 Reactions", descr="reactions.md")


def _plot_cell_sampling(params: dict):
    imgs = []
    for key, (k, n, xlims) in params.items():
        sigm_img = plots.sigm(k=k, n=n, var=key, varlims=xlims)
        p_img = plots.sampling(k=k, n=n, var=key, varlims=xlims)
        imgs.append(hcat_imgs(sigm_img, p_img))
    img = imgs.pop(0)
    while len(imgs) > 0:
        img = vcat_imgs(img, imgs.pop(0))
    save_img(img=img, name="cell_sampling.png")


def _describe_cell_sampling():
    params = {
        "[X]": (30.0, 3, (0, 40)),
        "[E]": (0.5, -2, (0, 10)),
        "genome size": (2000, 7, (0, 3000)),
    }
    _plot_cell_sampling(params=params)

    records = [{"variable": k, "k": d[0], "n": d[1]} for k, d in params.items()]
    df = pd.DataFrame.from_records(records)
    write_table(df=df, name="cell_sampling.csv")
    write_table_to_md(df=df, name="cell-sampling.md")


def describe_setup(_: dict):
    _describe_chemistry()
    _describe_cell_sampling()
