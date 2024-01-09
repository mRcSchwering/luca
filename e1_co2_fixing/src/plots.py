# ruff: noqa: F405
# type: ignore
import io
from collections import Counter
from PIL import Image
import pandas as pd
from plotnine import *


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


def state_cell_stats(
    divisions: list[int],
    lifetimes: list[int],
    genome_sizes: list[int],
    bins=20,
    pad_x=0.2,
    rel_y=0.5,
    text_size=10,
) -> Image:
    dfs = [
        pd.DataFrame({"v": divisions, "k": "divisions"}),
        pd.DataFrame({"v": lifetimes, "k": "lifetime"}),
        pd.DataFrame({"v": genome_sizes, "k": "genome-size"}),
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
    return _plot_2_img(g, figsize=(7, 2))


def marked_cellmap(
    df: pd.DataFrame, map_size: int, top_n=15, x="x", y="y", label="label", NA="other"
) -> Image:
    counts = Counter(df.loc[df[label] != NA, label]).most_common(n=top_n)
    labels = [d[0] for d in counts]

    df.loc[~df[label].isin(labels), label] = NA
    counts = Counter(df.loc[df[label] != NA, label]).most_common(n=top_n)
    records = [{"n": c, label: v} for v, c in counts]
    records.append({"n": len(df) - sum(d[1] for d in counts), label: NA})
    cnts = pd.DataFrame.from_records(records)

    cats = list(reversed([d for d in labels if d != NA] + [NA]))
    df[label] = pd.Categorical(df[label], categories=cats, ordered=True)
    cnts[label] = pd.Categorical(cnts[label], categories=cats, ordered=True)

    # fmt: off
    mapplot = (ggplot(df, aes(x=x, y=y))
        + geom_point(color="dimgray", size=.1, data=df.loc[df[label] == NA])
        + geom_point(aes(color=label), size=.1, data=df.loc[df[label] != NA])
        + coord_fixed(ratio=1, xlim=(0, map_size), ylim=(0, map_size))
        + theme(legend_position="none")
        + theme(strip_background=element_blank(), strip_text=element_blank())
        + theme(plot_margin=0, panel_spacing=0)
        + theme(panel_background=element_blank(), panel_border=element_rect(colour="black", size=0.5))
        + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
        + theme(axis_title=element_blank(), axis_text=element_blank()))
    
    barplot = (ggplot(cnts, aes(x=label, y="n"))
        + geom_col(color="dimgray", data=cnts.loc[cnts[label] == NA])
        + geom_col(aes(fill=label), data=cnts.loc[cnts[label] != NA])
        + theme(legend_position="none")
        + theme(axis_title=element_blank())
        + coord_flip())
    # fmt: on

    mapimg = _plot_2_img(mapplot, figsize=(2.5, 2.5))
    barimg = _plot_2_img(barplot, figsize=(2.5, 2.5))
    map_w, map_h = mapimg.size
    bar_w, bar_h = barimg.size
    target_w = map_w + bar_w
    target_h = max(map_h, bar_h)
    bkg = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
    bkg.paste(mapimg, (0, 0), mapimg)
    bkg.paste(barimg, (map_w, target_h - bar_h), barimg)
    return bkg
