# ruff: noqa: F405
# type: ignore
import io
from itertools import product
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib as mpl
import magicsoup as ms
from plotnine import *

MS_COLORS = {
    "genome": "dimgray",
    "CDS": "lightgray",
    "catal": "#fe218b",
    "trnsp": "#21b0fe",
    "reg": "#fed700",
}
NA_COL = "#595959"
NA_COL_RGB = (89, 89, 89)


def set_theme():
    theme_set(theme_minimal())


def _plot_2_img(
    plot: ggplot, figsize: tuple[float, float], dpi=200, add_bkg=True
) -> Image:
    plot = plot + theme(figure_size=figsize)
    buf = io.BytesIO()
    plot.save(buf, width=figsize[0], height=figsize[1], dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    if add_bkg:
        w, h = img.size
        bkg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        bkg.paste(img, (0, 0), img)
        img = bkg
    return img


def tabcolors(vals: list[str], dark=True, NA="other") -> dict[str, str]:
    s = 0 if dark else 1
    colormap = mpl.colormaps["tab20"]
    vals = list(dict.fromkeys(vals))
    cols = {}
    for i, d in enumerate(vals):
        cols[d] = mpl.colors.rgb2hex(colormap(i * 2 + s))
    cols[NA] = NA_COL
    return cols


def tensorboard_cellmap(imgstr: str, cellcol=NA_COL_RGB, border=1, margin=10) -> Image:
    img = Image.open(io.BytesIO(imgstr))
    arr = np.array(img.convert("RGBA"))
    red, green, blue, _ = arr.T
    is_white = (red == 255) & (blue == 255) & (green == 255)
    is_black = (red == 0) & (blue == 0) & (green == 0)
    arr[..., :-1][is_white.T] = cellcol
    arr[..., :-1][is_black.T] = (255, 255, 255)
    img = Image.fromarray(arr)
    img = ImageOps.expand(image=img, border=border, fill="black")
    img = ImageOps.expand(image=img, border=margin, fill="white")
    return img


def cellhists(
    world: ms.World,
    bins=20,
    pad_x=0.2,
    rel_y=0.5,
    text_size=10,
    figsize=(7, 2),
) -> Image:
    dfs = [
        pd.DataFrame({"v": world.cell_divisions, "k": "divisions"}),
        pd.DataFrame({"v": world.cell_lifetimes, "k": "lifetime"}),
        pd.DataFrame({"v": [len(d) for d in world.cell_genomes], "k": "genome-size"}),
    ]
    df = pd.concat(dfs, ignore_index=True)

    records = []
    for key, grpdf in df.groupby("k"):
        avg = grpdf["v"].mean()
        x = avg + avg * 2 * pad_x
        lab = f"{avg:.2f}"
        records.append({"k": key, "m": avg, "x": x, "l": lab})

    avgs = pd.DataFrame.from_records(records)
    avgs["y"] = len(df) / len(df["k"].unique()) * rel_y

    # fmt: off
    g = (ggplot(df)
        + geom_histogram(aes(x="v"), bins=bins)
        + geom_vline(aes(xintercept="m"), linetype="dashed", alpha=0.5, data=avgs)
        + geom_text(aes(x="x", y="y", label="l"), size=text_size, data=avgs)
        + facet_grid(". ~ k", scales="free")
        + theme(axis_title=element_blank()))
    # fmt: on
    return _plot_2_img(g, figsize=figsize)


def protein_counts(
    proteins: list[str],
    proteomes: list[list[str]],
    grp2idxs: dict[str, list[int]],
    grp2col: dict[str, str],
    figsize=(4, 6),
    NA="other",
) -> Image:
    prot_cats = [NA] + list(reversed([d for d in proteins if d != NA]))
    grp_cats = [NA] + list(reversed([d for d in grp2idxs if d != NA]))

    records = []
    for grp, cell_idxs in grp2idxs.items():
        grp_proteomes = [proteomes[d] for d in cell_idxs]
        prots = [dd for d in grp_proteomes for dd in d]
        for prot in proteins:
            n = prots.count(prot) / len(grp_proteomes)
            records.append({"k": prot, "n": n, "l": grp})

    df = pd.DataFrame.from_records(records)
    df.loc[df["k"] == NA, "l"] = NA
    df["k"] = pd.Categorical(df["k"], categories=prot_cats, ordered=True)
    df["l"] = pd.Categorical(df["l"], categories=grp_cats, ordered=True)

    # fmt: off
    g = (ggplot(df, aes(x="k", y="n"))
        + geom_col(aes(fill="l"), position="dodge")
        + scale_fill_manual(values=grp2col)
        + theme(legend_position="none")
        + theme(axis_title=element_blank())
        + coord_flip())
    # fmt: on

    return _plot_2_img(g, figsize=figsize)


def cellmap(
    world: ms.World,
    grp2idxs: dict[str, list[int]],
    grp2col: dict[str, str],
    figsize=(2.5, 2.5),
    NA="other",
) -> Image:
    df = pd.DataFrame(world.cell_positions, columns=["x", "y"])
    df["l"] = NA
    for k, d in grp2idxs.items():
        df.loc[d, "l"] = k

    grp_cats = [NA] + list(reversed([d for d in grp2idxs if d != NA]))
    df["l"] = pd.Categorical(df["l"], categories=grp_cats, ordered=True)

    # fmt: off
    g = (ggplot(df, aes(x="x", y="y"))
        + geom_point(aes(color="l"), size=.1)
        + scale_color_manual(values=grp2col)
        + coord_fixed(ratio=1, xlim=(0, world.map_size), ylim=(0, world.map_size))
        + theme(legend_position="none")
        + theme(strip_background=element_blank(), strip_text=element_blank())
        + theme(plot_margin=0, panel_spacing=0)
        + theme(panel_background=element_blank(), panel_border=element_rect(colour="black", size=0.5))
        + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
        + theme(axis_title=element_blank(), axis_text=element_blank()))
    # fmt: on
    return _plot_2_img(g, figsize=figsize)


def grp_counts(
    world: ms.World,
    grp2idxs: dict[str, list[int]],
    grp2col: dict[str, str],
    figsize=(2.5, 2.5),
    NA="other",
) -> Image:
    records = [{"k": k, "n": len(d)} for k, d in grp2idxs.items()]
    records.append({"k": NA, "n": world.n_cells - sum(d["n"] for d in records)})
    df = pd.DataFrame.from_records(records)

    counts = sorted([(len(d), k) for k, d in grp2idxs.items()], reverse=True)
    cats = [NA] + list(reversed([d[1] for d in counts]))
    df["k"] = pd.Categorical(df["k"], categories=cats, ordered=True)

    # fmt: off
    g = (ggplot(df, aes(x="k", y="n"))
        + geom_col(aes(fill="k"))
        + scale_fill_manual(values=grp2col)
        + theme(legend_position="none")
        + theme(axis_title=element_blank())
        + coord_flip())
    # fmt: on
    return _plot_2_img(g, figsize=figsize)


def molecule_concentrations(
    world: ms.World,
    molecules: list[ms.Molecule],
    grp2idxs: dict[str, list[int]],
    grp2col: dict[str, str],
    figsize=(7, 3),
    NA="other",
    max_q=0.9,
):
    molnames = [d.name for d in molecules]
    mol_cats = [NA] + list(reversed([d for d in molnames if d != NA]))
    grp_cats = [NA] + list(reversed([d for d in grp2idxs if d != NA]))

    idx2grp = {dd: k for k, d in grp2idxs.items() for dd in d}
    records = []
    for cell_i in range(world.n_cells):
        if cell_i in idx2grp:
            label = idx2grp[cell_i]
            cell = world.get_cell(by_idx=cell_i)
            for mol in molecules:
                mol_i = world.chemistry.mol_2_idx[mol]
                records.append(
                    {
                        "c": cell_i,
                        "m": mol.name,
                        "l": "intracellular",
                        "n": cell.int_molecules[mol_i].item(),
                        "k": label,
                    }
                )
                records.append(
                    {
                        "c": cell_i,
                        "m": mol.name,
                        "l": "extracellular",
                        "n": cell.ext_molecules[mol_i].item(),
                        "k": label,
                    }
                )

    df = pd.DataFrame.from_records(records)
    df["m"] = pd.Categorical(df["m"], categories=mol_cats, ordered=True)
    df["k"] = pd.Categorical(df["k"], categories=grp_cats, ordered=True)

    if max_q < 1.0:
        int_mask = df["l"] == "intracellular"
        ext_mask = df["l"] == "extracellular"
        t_int = df.loc[int_mask, "n"].quantile(max_q)
        t_ext = df.loc[ext_mask, "n"].quantile(max_q)
        df = df.loc[~((df["n"] > t_int) & int_mask)]
        df = df.loc[~((df["n"] > t_ext) & ext_mask)]

    # fmt: off
    g = (ggplot(df, aes(x="m", y="n"))
        + geom_boxplot(aes(color="k"))
        + scale_color_manual(values=grp2col)
        + facet_grid(". ~ l", scales="free")
        + theme(legend_position="none")
        + theme(axis_title=element_blank())
        + coord_flip())
    # fmt: on

    return _plot_2_img(g, figsize=figsize)


def cellboxes(
    world: ms.World,
    grp2idxs: dict[str, list[int]],
    grp2col: dict[str, str],
    figsize=(7, 1),
    NA="other",
):
    idx2grp = {dd: k for k, d in grp2idxs.items() for dd in d}
    records = []
    for cell_i in range(world.n_cells):
        if cell_i in idx2grp:
            label = idx2grp[cell_i]
            cell = world.get_cell(by_idx=cell_i)
            records.append(
                {"c": cell_i, "l": label, "k": "genome-size[bp]", "n": len(cell.genome)}
            )
            records.append(
                {"c": cell_i, "l": label, "k": "age[step]", "n": cell.n_steps_alive}
            )
            records.append(
                {"c": cell_i, "l": label, "k": "divisions", "n": cell.n_divisions}
            )

    df = pd.DataFrame.from_records(records)
    clst_cats = [NA] + list(reversed(grp2idxs))
    var_cats = [NA] + ["genome-size[bp]", "age[step]", "divisions"]
    df["l"] = pd.Categorical(df["l"], categories=clst_cats, ordered=True)
    df["k"] = pd.Categorical(df["k"], categories=var_cats, ordered=True)

    # fmt: off
    g = (ggplot(df, aes(x="k", y="n"))
        + geom_boxplot(aes(color="l"))
        + scale_color_manual(values=grp2col)
        + facet_wrap("~ k", scales="free", nrow=1)
        + theme(legend_position="none")
        + theme(subplots_adjust={"wspace": 0.25})
        + theme(axis_title=element_blank(), axis_text_y=element_blank())
        + coord_flip())
    # fmt: on

    return _plot_2_img(g, figsize=figsize)


def sigm(
    k: float, n: int, var: str, varlims=tuple[float, float], figsize=(2, 2)
) -> Image:
    def f(x: np.ndarray, k: float, n: int) -> np.ndarray:
        return x**n / (k**n + x**n)

    records = []
    X = np.arange(start=varlims[0], stop=varlims[1], step=0.1)
    Y = f(x=X, k=k, n=n)
    for x, y in zip(X, Y):
        records.append({var: x, "p": y})

    df = pd.DataFrame.from_records(records)
    m = f(x=k, k=k, n=n)

    # fmt: off
    g = (ggplot(df)
        + geom_line(aes(x=var, y="p"), color=NA_COL)
        + annotate("segment", x=k, y=0, xend=k, yend=m, linetype="dotted")
        + annotate("segment", x=0, y=m, xend=k, yend=m, linetype="dotted")
        + ylim(0.0, 1.0)
        + theme(legend_position="none"))
    # fmt: on
    return _plot_2_img(g, figsize=figsize, dpi=100)


def sampling(
    k: float,
    n: int,
    var: str,
    varlims=tuple[float, float],
    n_ks=8,
    n_steps=100,
    figsize=(3, 2),
) -> Image:
    def f(x: np.ndarray, k: float, n: int, s: int) -> np.ndarray:
        p = x**n / (k**n + x**n)
        return 1 - (1 - p) ** s

    records = []
    X = np.linspace(start=varlims[0], stop=varlims[1], num=n_ks)
    for step in range(n_steps):
        Y = f(x=X, k=k, n=n, s=step)
        for x, y in zip(X, Y):
            records.append({var: x, "p": y, "step": step})

    df = pd.DataFrame.from_records(records)

    # fmt: off
    g = (ggplot(df)
        + geom_line(aes(x="step", y="p", color=var, group=var))
        + theme(axis_title_y=element_blank(), axis_text_y=element_blank())
        + ylim(0.0, 1.0))
    # fmt: on
    return _plot_2_img(g, figsize=figsize, dpi=100)


def run_scalars(
    df: pd.DataFrame,
    x="step",
    y="value",
    var="variable",
    figsize=(10, 8),
    color=NA_COL,
) -> Image:
    # fmt: off
    g = (ggplot(df)
        + geom_line(aes(x=x, y=y), color=color)
        + facet_grid(f"{var} ~ .", scales="free")
        + theme(axis_title_y=element_blank())
        + theme(legend_position="none"))
    # fmt: on
    return _plot_2_img(g, figsize=figsize)


def pathway_training(
    df: pd.DataFrame,
    grp2progress: dict[str, tuple[float, float]],
    x="step",
    y="value",
    lab="stage",
    var="variable",
    trial="trial",
    grp="runname",
    progress="Progress",
    figsize=(12, 10),
) -> Image:
    df = df.copy()
    utrials = df[trial].unique()
    uvars = df[var].unique()
    df[trial] = pd.Categorical(df[trial], categories=utrials)
    darkcols = tabcolors(vals=utrials)
    lightcols = tabcolors(vals=[f"{d}-phase" for d in utrials], dark=False)

    records = []
    for run, rows in df.groupby(grp):
        vars_ = rows[var].unique()
        trials = rows[trial].unique()
        grps = rows[grp].unique()
        labs_ = rows[lab].unique()
        for target in grp2progress[run]:
            is_progress = rows[var] == progress
            is_ge = rows[y] >= target
            progress_delta = rows.loc[is_progress & is_ge, y] - target
            is_min = progress_delta == progress_delta.min()
            steps = rows.loc[is_progress & is_min, x]
            s = sorted(steps.tolist())[0]
            records.extend(
                [
                    {x: s, grp: r, var: v, lab: l_, trial: t, "phase": f"{t}-phase"}
                    for r, v, t, l_ in product(grps, vars_, trials, labs_)
                ]
            )

    df = pd.concat([df, pd.DataFrame.from_records(records)], ignore_index=True)
    df[var] = pd.Categorical(df[var], categories=uvars)
    df[trial] = pd.Categorical(df[trial], categories=utrials)

    # fmt: off
    g = (ggplot(df)
        + geom_vline(aes(xintercept=x, color="phase"), data=df[~df["phase"].isna()], linetype="dotted")
        + geom_line(aes(x=x, y=y, group=grp, color=trial), data=df[df["phase"].isna()])
        + scale_color_manual({**darkcols, **lightcols})
        + facet_grid(f"{var} ~ {lab}", scales="free")
        + theme(axis_title_y=element_blank())
        + theme(legend_position="none"))
    # fmt: on
    return _plot_2_img(g, figsize=figsize)
