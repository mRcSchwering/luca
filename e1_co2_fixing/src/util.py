from pathlib import Path
import datetime as dt
from collections import Counter
from itertools import product
import multiprocessing as mp
import numpy as np
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


def rev_sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """Sample bernoulli with $k^n / (t^n + k^n)$ return bool tensor"""
    p = k**n / (t**n + k**n)
    return torch.bernoulli(p).bool()


def find_steps(rundir: Path) -> list[int]:
    """Get all sorted steps of rundir"""
    names = [d.name for d in rundir.glob("step=*")]
    return sorted(int(d.split("step=")[-1]) for d in names)


def load_cells(
    world: ms.World,
    label: str,
    runsdir: Path,
    reset_cells: bool = True,
) -> Path:
    """
    Use label to load a world's genomes:
        - "<rundir>/step=<i>" to load step <i> of <rundir>
          e.g. "2023-05-09_14-32/step=100" to load step 100
        - "<rundir>:<i>" to load the <i>th step of <rundir>
          e.g. "2023-05-09_14-32:-1" to load the last step of <rundir>
    """
    if "/" in label:
        statedir = runsdir / label
    elif ":" in label:
        runname, step_i = label.split(":")
        steps = find_steps(rundir=runsdir / runname)
        statedir = runsdir / runname / f"step={steps[int(step_i)]}"
    else:
        raise ValueError(f"Label {label} not recognized")

    world.load_state(statedir=statedir)
    if reset_cells:
        world.reposition_cells(cell_idxs=list(range(world.n_cells)))
        world.cell_divisions[:] = 0
        world.cell_labels = [ms.randstr(n=12) for _ in range(world.n_cells)]

    return statedir


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


def genome_distances(world: ms.World) -> np.ndarray:
    """Calculate genomic distance matrix based on Levenshtein"""
    args = [(d, world.cell_genomes) for d in range(world.n_cells)]
    with mp.Pool() as pool:
        arrs = pool.starmap(_genome_dists_row, args)
    return np.stack(arrs)


def cluster_cells(D: np.ndarray, n_clsts=10) -> np.ndarray:
    """Cluster cells using DBSCAN to get best coverage from top n clusters"""
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
    return model.fit(D).labels_


def save_doc(content: list[str], name: str):
    """Save utf-8 text file to docs dir"""
    with open(DOCS_DIR / name, "w", encoding="utf-8") as fh:
        fh.write("\n".join(content))


def save_img(img: Image, name: str, add_bkg=True):
    """Save image, adding a white background"""
    if add_bkg:
        w, h = img.size
        bkg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        bkg.paste(img, (0, 0), img)
        img = bkg
    img.save(fp=str(IMGS_DIR / name))


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


class Config:
    """Config container"""

    def __init__(
        self,
        device: str,
        runs_dir: Path | str,
        max_steps: int,
        max_steps_without_progress: int,
        max_time_m: int,
        n_trials: int,
    ):
        self.device = device
        self.runs_dir = runs_dir if isinstance(runs_dir, Path) else Path(runs_dir)
        self.max_steps = max_steps
        self.max_steps_without_progress = max_steps_without_progress
        self.max_time_m = max_time_m
        self.n_trials = n_trials
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
            n_trials=kwargs.pop("n_trials"),
        )
