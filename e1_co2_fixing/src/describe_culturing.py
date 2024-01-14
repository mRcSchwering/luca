import pandas as pd
from .util import (
    save_img,
    hcat_imgs,
    vcat_imgs,
    write_table,
    table_to_markdown,
    save_doc,
    read_doc,
)
from . import plots


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


def create(_: dict):
    params = {
        "[X]": (30.0, 3, (0, 40)),
        "[E]": (0.5, -2, (0, 10)),
        "genome size": (2000, 7, (0, 3000)),
    }
    _plot_cell_sampling(params=params)

    records = [{"variable": k, "k": d[0], "n": d[1]} for k, d in params.items()]
    df = pd.DataFrame.from_records(records)
    write_table(df=df, name="cell_sampling.csv")

    lines = read_doc(name="culturing.md")
    starts = [i for i, d in enumerate(lines) if "### Cell Sampling" in d]
    start = starts[0] if len(starts) > 0 else len(lines)
    ends = [i for i, d in enumerate(lines) if i > start and d == "[//]: # (end)"]
    end = ends[0] + 1 if len(ends) > 0 else len(lines)

    tab = table_to_markdown(
        df=df,
        name="2.1. Cell sampling",
        descr="Cells are sampled each step to be killed or replicated."
        " Replication probability depends on X concentration, killing probability on genome size and E concentration."
        " Probability p is calculated as $p_{k,n}(x) = x^n / (x^n + k^n)$.",
    )
    smplng = f"### Cell Sampling\n{tab}\n[//]: # (end)"
    lines = lines[:start] + [smplng] + lines[end:]
    save_doc(content=lines, name="culturing.md")
