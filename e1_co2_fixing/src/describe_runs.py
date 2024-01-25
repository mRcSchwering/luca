import json
from pathlib import Path
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from .chemistry import WL_STAGES_MAP
from .util import (
    RUNS_DIR,
    save_img,
    write_table,
    write_table_to_md,
    hcat_imgs,
    vcat_imgs,
)
from . import plots


def _load_hparams(rundirs: list[Path]) -> pd.DataFrame:
    records = []
    for rundir in rundirs:
        with open(rundir / "hparams.json", "r", encoding="utf-8") as fh:
            data = json.load(fh)
        records.append({**data, "runname": rundir.name, "path": rundir})

    df = pd.DataFrame.from_records(records)
    df["name"] = df["runname"].str.rsplit("_", n=1).str[0]
    df["trial"] = df["runname"].str.rsplit("_", n=1).str[1]
    df["trial"] = pd.to_numeric(df["trial"], downcast="integer")
    return df


def _load_scalars(scalars: list[tuple[str, str]], rundirs: list[Path]) -> pd.DataFrame:
    records = []
    for rundir in rundirs:
        tf_ea = event_accumulator.EventAccumulator(str(rundir))
        tf_ea.Reload()
        for scalar, variable in scalars:
            for obj in tf_ea.Scalars(scalar):
                records.append(
                    {
                        "runname": rundir.name,
                        "variable": variable,
                        "value": obj.value,
                        "s": obj.step,
                    }
                )

    df = pd.DataFrame.from_records(records)
    cats = [d[1] for d in scalars]
    df["variable"] = pd.Categorical(df["variable"], categories=cats, ordered=True)
    return df


def _load_imgs(image: str, rundir: Path, n_steps=10) -> list:
    ea_kwargs = {"size_guidance": {event_accumulator.IMAGES: 0}}
    tf_ea = event_accumulator.EventAccumulator(str(rundir), **ea_kwargs)
    tf_ea.Reload()

    avail = [d.step for d in tf_ea.Images(image)]
    idxs = np.round(np.linspace(0, len(avail) - 1, n_steps)).astype(int)
    steps = [avail[d] for d in idxs]
    tf_ea.Reload()

    imgs = []
    for obj in tf_ea.Images(image):
        if obj.step in steps:
            imgs.append(plots.tensorboard_cellmap(obj.encoded_image_string))
    return imgs


def describe_run(kwargs: dict):
    rundir = RUNS_DIR / kwargs["run"]
    title = rundir.name
    is_chemostat = "chemostat" in rundir.name

    if is_chemostat:
        scalars = [
            ("Cells/Total", "cells"),
            ("Cells/Divisions", "generation"),
            ("Cells/Survival", "age[s]"),
            ("Cells/GenomeSize", "genome-size"),
            ("Other/Progress", "progress"),
        ]
    else:
        scalars = [
            ("Cells/Total", "cells"),
            ("Cells/Divisions", "generation"),
            ("Cells/GrowthRate", "growth-rate"),
            ("Cells/Survival", "age[s]"),
            ("Cells/GenomeSize", "genome-size"),
            ("Other/Split", "passage"),
            ("Cells/cPD", "cPD"),
            ("Other/Progress", "progress"),
        ]

    figsize = (10, len(scalars) * 0.85)
    df = _load_scalars(scalars=scalars, rundirs=[rundir])
    cell_imgs = _load_imgs(image="Maps/Cells", rundir=rundir)
    cells_img = cell_imgs.pop(0)
    for img in cell_imgs:
        cells_img = hcat_imgs(cells_img, img)

    plot_img = plots.timeseries(
        df=df, x="s", y="value", row="variable", figsize=figsize
    )
    img = vcat_imgs(cells_img, plot_img)
    save_img(img=img, name=f"{title}_run-by-step.png")


def describe_pathway_training(kwargs: dict):
    rundirs = [RUNS_DIR / d for d in kwargs["runs"]]
    hparams_df = _load_hparams(rundirs=rundirs)
    hparams_df.rename(columns={"pathway-label": "stage"}, inplace=True)

    write_table(
        df=hparams_df[[d for d in hparams_df.columns if d != "path"]],
        name="pathway-training-hparams.csv",
    )
    write_table_to_md(
        df=hparams_df[["runname", "stage", "init-label"]],
        name="pathway-training-strategy.md",
    )

    records = []
    for stage, (prots, subs_a, subs_b, add) in WL_STAGES_MAP.items():
        records.append(
            {
                "stage": stage,
                "substrates A": ", ".join(d.name for d in subs_a),
                "substrates B": ", ".join(d.name for d in subs_b),
                "additives": ", ".join(d.name for d in add),
                "genes": ",\n".join([" | ".join([str(dd) for dd in d]) for d in prots]),
            }
        )

    stages_df = pd.DataFrame.from_records(records)
    stages_df["genes"] = stages_df["genes"].str.replace("<->", "$\\rightleftharpoons$")
    write_table(df=stages_df, name="pathway-training-stages.csv")
    write_table_to_md(df=stages_df, name="pathway-training-stages.md")

    cols = [
        "runname",
        "n_init_splits",
        "n_adapt_splits",
        "n_final_splits",
        "min_gr",
        "mutation_rate_mult",
    ]
    write_table_to_md(df=hparams_df[cols], name="pathway-training-hparams.md")

    phases_map = {}
    for _, row in hparams_df.iterrows():
        n_init_splits = row["n_init_splits"]
        n_init_adapt_splits = n_init_splits + row["n_adapt_splits"]
        n_total_splits = n_init_adapt_splits + row["n_final_splits"]
        adaption_start = n_init_splits / n_total_splits
        adaption_end = n_init_adapt_splits / n_total_splits
        phases_map[row["runname"]] = (adaption_start, adaption_end)

    scalars = [
        ("Cells/Total", "cells"),
        ("Cells/Divisions", "generation"),
        ("Cells/GrowthRate", "growth-rate"),
        ("Cells/Survival", "age[s]"),
        ("Cells/GenomeSize", "genome-size"),
        ("Other/Split", "passage"),
        ("Cells/cPD", "cPD"),
        ("Other/Progress", "progress"),
    ]
    scalars_df = _load_scalars(scalars=scalars, rundirs=rundirs)

    df = pd.merge(scalars_df, hparams_df[["runname", "stage", "trial"]], on="runname")
    img = plots.pathway_training(df=df, grp2progress=phases_map)
    save_img(img=img, name="pathway-training-by-step.png")
