import json
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from .chemistry import WL_STAGES_MAP
from .util import RUNS_DIR, save_img, write_table, replace_doc_tab, table_to_markdown
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
                        "step": obj.step,
                    }
                )

    df = pd.DataFrame.from_records(records)
    cats = [d[1] for d in scalars]
    df["variable"] = pd.Categorical(df["variable"], categories=cats, ordered=True)
    return df


def describe_pathway_training(kwargs: dict):
    rundirs = [RUNS_DIR / d for d in kwargs["runs"]]
    hparams_df = _load_hparams(rundirs=rundirs)
    hparams_df.rename(
        columns={"pathway-label": "stage", "init-label": "prev-stage"}, inplace=True
    )

    write_table(
        df=hparams_df[[d for d in hparams_df.columns if d != "path"]],
        name="pathway-training-hparams.csv",
    )

    hparams_tab = table_to_markdown(
        df=hparams_df[["runname", "stage", "prev-stage"]],
        name="3.1 Training strategy",
        descr="Training WL pathway in multiple stages, start each stage with successful cells of the previous stage.",
    )
    replace_doc_tab(
        filename="pathway_training.md",
        tabname="3.1 Training strategy",
        tab=hparams_tab,
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
    write_table(df=stages_df, name="training-pathway-stages.csv")

    stages_tab = table_to_markdown(
        df=stages_df,
        name="3.2 WL training stages",
        descr="Training WL pathway in multiple stages, start each stage with successful cells of the previous stage.",
    )
    stages_tab = stages_tab.replace("<->", "$\\rightleftharpoons$")
    replace_doc_tab(
        filename="pathway_training.md",
        tabname="3.2 WL training stages",
        tab=stages_tab,
    )

    cols = [
        "runname",
        "n_init_splits",
        "n_adapt_splits",
        "n_final_splits",
        "min_gr",
        "mutation_rate_mult",
    ]
    hparams_tab = table_to_markdown(
        df=hparams_df[cols],
        name="3.2 WL training hyperparameters",
        descr="Hyperparameters for WL pathway training simulation runs.",
    )
    replace_doc_tab(
        filename="pathway_training.md",
        tabname="3.2 WL training hyperparameters",
        tab=hparams_tab,
    )

    phases_map = {}
    for _, row in hparams_df.iterrows():
        n_init_splits = row["n_init_splits"]
        n_init_adapt_splits = n_init_splits + row["n_adapt_splits"]
        n_total_splits = n_init_adapt_splits + row["n_final_splits"]
        adaption_start = n_init_splits / n_total_splits
        adaption_end = n_init_adapt_splits / n_total_splits
        phases_map[row["runname"]] = (adaption_start, adaption_end)

    scalars = [
        ("Cells/Total", "Cells"),
        ("Cells/Divisions", "Divisions"),
        ("Cells/GrowthRate", "GrowthRate"),
        ("Cells/Survival", "Survival"),
        ("Cells/GenomeSize", "GenomeSize"),
        ("Other/Split", "Passage"),
        ("Cells/cPD", "cPD"),
        ("Other/Progress", "Progress"),
        ("Other/TimePerStep[s]", "s/step"),
    ]
    scalars_df = _load_scalars(scalars=scalars, rundirs=rundirs)

    df = pd.merge(scalars_df, hparams_df[["runname", "stage", "trial"]], on="runname")
    img = plots.pathway_training(df=df, grp2progress=phases_map)
    save_img(img=img, name="pathway-training-by-step.png")

    divisions_df = df.loc[df["variable"] == "Divisions", ["runname", "step", "value"]]
    divisions_df.columns = ["runname", "step", "Generation"]
    keep = [
        "Cells",
        "GenomeSize",
        "GrowthRate",
        "Survival",
        "cPD",
        "Passage",
        "Progress",
    ]
    df = pd.merge(df[df["variable"].isin(keep)], divisions_df, on=["runname", "step"])
    img = plots.pathway_training(df=df, grp2progress=phases_map, x="Generation")
    save_img(img=img, name="pathway-training-by-generation.png")
