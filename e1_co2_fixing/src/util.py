from pathlib import Path
import datetime as dt
import random
from collections import Counter
from itertools import product
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from Levenshtein import distance as ls_dist
from sklearn.cluster import DBSCAN
import magicsoup as ms

EXP_DIR = Path(__file__).parent.parent
RUNS_DIR = EXP_DIR / "runs"
IMGS_DIR = EXP_DIR / "imgs"
TABLES_DIR = EXP_DIR / "tables"
DOCS_DIR = EXP_DIR / "docs"


def sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """Sample bernoulli with $t^n / (t^n + k^n)$ return bool tensor"""
    p = t**n / (t**n + k**n)
    return torch.bernoulli(p).bool()


def cell_work(world: ms.World, eps=1e-8) -> torch.Tensor:
    """A measure for each cell's manipulation of the environment"""
    xs = world.cell_positions[:, 0]
    ys = world.cell_positions[:, 1]
    lxi = torch.log(world.cell_molecules + eps)
    lxe = torch.log(world.molecule_map[:, xs, ys].T + eps)
    return (lxi - lxe).abs().mean(dim=1)


def find_steps(rundir: Path) -> list[int]:
    """Get all sorted steps of rundir"""
    names = [d.name for d in rundir.glob("step=*")]
    return sorted(int(d.split("step=")[-1]) for d in names)


def get_statedir(label: str) -> Path:
    """
    Get statedir from label:
        - "<rundir>/step=<i>" to load step <i> of <rundir>
          e.g. "2023-05-09_14-32/step=100" to load step 100
        - "<rundir>:<i>" to load the <i>th step of <rundir>
          e.g. "2023-05-09_14-32:-1" to load the last step of <rundir>
    """
    if "/" in label:
        return RUNS_DIR / label
    elif ":" in label:
        runname, step_i = label.split(":")
        steps = find_steps(rundir=RUNS_DIR / runname)
        return RUNS_DIR / runname / f"step={steps[int(step_i)]}"
    raise ValueError(f"Label {label} not recognized")


def load_cells(world: ms.World, label: str, target_confl=1.0):
    """
    Load cells from previous state, randomly distribute them on
    map and assign random labels. Reduce to target confluency if
    loaded confluency is higher.

    Use label to load a world's genomes:
        - "<rundir>/step=<i>" to load step <i> of <rundir>
          e.g. "2023-05-09_14-32/step=100" to load step 100
        - "<rundir>:<i>" to load the <i>th step of <rundir>
          e.g. "2023-05-09_14-32:-1" to load the last step of <rundir>
    """
    statedir = get_statedir(label=label)
    world.load_state(statedir=statedir)
    world.reposition_cells(cell_idxs=list(range(world.n_cells)))
    world.cell_divisions[:] = 0
    world.cell_labels = [ms.randstr(n=12) for _ in range(world.n_cells)]
    kill_n = world.n_cells - int(target_confl * world.map_size**2)
    if kill_n > 0:
        idxs = random.sample(range(world.n_cells), k=kill_n)
        world.kill_cells(cell_idxs=idxs)


def _genome_dists_row(i: int, genomes: list[str], minlen=1) -> np.ndarray:
    n = len(genomes)
    gi = genomes[i]
    ni = len(gi)
    Di = np.full((n,), float("nan"))
    if ni < minlen:
        return Di
    for j in range(n):
        gj = genomes[j]
        nj = len(gj)
        if nj > minlen:
            Di[j] = ls_dist(gi, gj) / max(ni, nj)
    return Di


def genome_distances(genomes: list[str], minlen=1) -> np.ndarray:
    """Calculate genomic distance matrix based on Levenshtein"""
    args = [(d, genomes, minlen) for d in range(len(genomes))]
    with mp.Pool() as pool:
        arrs = pool.starmap(_genome_dists_row, args)
    return np.stack(arrs)


def _proteome_dists_row(i: int, proteomes: list[list[str]], minlen=1) -> np.ndarray:
    n = len(proteomes)
    pi = set(proteomes[i])
    ni = len(pi)
    Di = np.full((n,), float("nan"))
    if ni < minlen:
        return Di
    for j in range(n):
        pj = set(proteomes[j])
        nj = len(pj)
        if nj >= minlen:
            Di[j] = 1 - len(pi & pj) / len(pi | pj)
    return Di


def proteome_distances(proteomes: list[list[str]], minlen=1) -> np.ndarray:
    """Calculate proteome distance matrix based on common proteins"""
    args = [(d, proteomes, minlen) for d in range(len(proteomes))]
    with mp.Pool() as pool:
        arrs = pool.starmap(_proteome_dists_row, args)
    return np.stack(arrs)


def dbscan_cells(D: np.ndarray, n_clsts=10, max_d=1.0) -> dict[str, list[int]]:
    """Cluster cells using DBSCAN to get best coverage from top n clusters"""
    D[np.isnan(D)] = max_d
    eps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    min_samples = [10, 20, 30, 40, 50]
    results = []
    for e, m in product(eps, min_samples):
        model = DBSCAN(eps=e, min_samples=m, metric="precomputed")
        labels = model.fit(D).labels_
        top_clsts = Counter(labels[labels != -1]).most_common(n_clsts)
        if len(top_clsts) > 0:
            n = sum(d[1] for d in top_clsts)
            results.append((n, e, m))
    if len(results) == 1:
        raise ValueError("Not a single cluster found")
    _, e, m = sorted(results)[-1]
    model = DBSCAN(eps=e, min_samples=m, metric="euclidean")
    labels = model.fit(D).labels_
    labcnts = Counter([d for d in labels if d >= 0]).most_common(n_clsts)
    clsts = {k: [i for i, d in enumerate(labels) if d == k] for k, _ in labcnts}
    return {f"c{k}": d for k, d in clsts.items()}


def save_img(img: Image, name: str, add_bkg=True):
    """Save image, adding a white background"""
    if add_bkg:
        w, h = img.size
        bkg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        bkg.paste(img, (0, 0), img)
        img = bkg
    img.save(fp=str(IMGS_DIR / name))
    print(f"wrote {IMGS_DIR / name}")


def crop_img(img: Image, pad: tuple[int, ...]) -> Image:
    """Crop image with 1-, 2-, or 4-tuple padding"""
    w, h = img.size
    if len(pad) == 1:
        lft = pad[0]
        up = pad[0]
        rgt = w - pad[0]
        lo = h - pad[0]
    elif len(pad) == 2:
        lft = pad[0]
        up = pad[1]
        rgt = w - pad[0]
        lo = h - pad[1]
    elif len(pad) == 4:
        lft = pad[0]
        up = pad[1]
        rgt = w - pad[2]
        lo = h - pad[3]
    return img.crop((lft, up, rgt, lo))


def vcat_imgs(img0: Image, img1: Image, bkg=(255, 255, 255, 255)) -> Image:
    """Vertically concatenate 2 images"""
    w0, h0 = img0.size
    w1, h1 = img1.size
    w = max(w0, w1)
    h = h0 + h1
    comp = Image.new("RGBA", (w, h), bkg)
    comp.paste(img0, ((w - w0) // 2, 0), img0)
    comp.paste(img1, ((w - w1) // 2, h0), img1)
    return comp


def hcat_imgs(img0: Image, img1: Image, bkg=(255, 255, 255, 255)) -> Image:
    """Horizontally concatenate 2 images"""
    w0, h0 = img0.size
    w1, h1 = img1.size
    w = w0 + w1
    h = max(h0, h1)
    comp = Image.new("RGBA", (w, h), bkg)
    comp.paste(img0, (0, h - h0), img0)
    comp.paste(img1, (w0, h - h1), img1)
    return comp


def write_table(df: pd.DataFrame, name: str, index=False, header=True):
    """Save table as csv with dtypes in column names"""
    df = df.copy()
    df.columns = [f"{c}[{t.name}]" for c, t in zip(df.columns, df.dtypes)]
    df.to_csv(TABLES_DIR / name, header=header, index=index)
    print(f"wrote {TABLES_DIR / name}")


def read_table(name: str, index_col=False, header=0) -> pd.DataFrame:
    """Read table from csv with dtypes in column names"""
    df = pd.read_csv(TABLES_DIR / name, header=header, index_col=index_col)
    colnames = []
    for col in df.columns:
        colname, dtypestr = col.split("[")
        colnames.append(colname)
        df[col] = df[col].astype(dtypestr[:-1])
    df.columns = colnames
    return df


def write_doc(content: list[str], name: str):
    """Save utf-8 text file to docs dir"""
    with open(DOCS_DIR / name, "w", encoding="utf-8") as fh:
        fh.write("\n".join(content))
    print(f"wrote {DOCS_DIR / name}")


def read_doc(name: str) -> list[str]:
    """Read utf-8 file from docs dir"""
    with open(DOCS_DIR / name, "r", encoding="utf-8") as fh:
        return fh.read().split("\n")


def write_table_to_md(df: pd.DataFrame, name: str, descr="", index=False):
    """Convert and write table with description to markdown file"""
    header = f"**{name}**"
    if len(descr) > 0:
        header += f" {descr}"
    header = "_" + header + "_"
    tab = f"{header}\n{df.to_markdown(index=index)}"
    write_doc(content=["", tab, ""], name=name)


class Config:
    """Config container"""

    def __init__(
        self,
        device: str,
        runs_dir: Path | str,
        max_steps: int,
        max_steps_without_progress: int,
        max_steps_without_split: int,
        max_time_m: int,
        max_trials: int,
        max_successful_trials: int,
        min_confluency: float,
    ):
        self.device = device
        self.runs_dir = runs_dir if isinstance(runs_dir, Path) else Path(runs_dir)
        self.max_steps = max_steps
        self.max_steps_without_progress = max_steps_without_progress
        self.max_steps_without_split = max_steps_without_split
        self.max_time_m = max_time_m
        self.max_trials = max_trials
        self.max_successful_trials = max_successful_trials
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.min_confluency = min_confluency

    def reset(self):
        self.timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    @classmethod
    def pop_from(cls, kwargs: dict) -> "Config":
        """Pop config keys from kwargs and return Config"""
        return cls(
            device=kwargs.pop("device"),
            runs_dir=kwargs.pop("runs_dir"),
            max_steps=kwargs.pop("max_steps"),
            max_steps_without_progress=kwargs.pop("max_steps_without_progress"),
            max_time_m=kwargs.pop("max_time_m"),
            max_trials=kwargs.pop("max_trials"),
            max_successful_trials=kwargs.pop("max_successful_trials"),
            min_confluency=kwargs.pop("min_confluency"),
            max_steps_without_split=kwargs.pop("max_steps_without_split"),
        )
