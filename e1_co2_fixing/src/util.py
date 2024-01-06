from pathlib import Path
import datetime as dt
import torch
import magicsoup as ms


def sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """$t^n / (t^n + k^n)$"""
    return t**n / (t**n + k**n)


def rev_sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """$k^n / (t^n + k^n)$"""
    return k**n / (t**n + k**n)


def find_steps(rundir: Path) -> list[int]:
    """Get all sorted steps of rundir"""
    names = [d.name for d in rundir.glob("step=*")]
    return sorted(int(d.split("step=")[-1]) for d in names)


def load_cells(
    world: ms.World, label: str, runsdir: Path, reposition_cells: bool = True
):
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
    if reposition_cells:
        world.reposition_cells(cell_idxs=list(range(world.n_cells)))
    world.cell_divisions[:] = 0
    world.cell_labels = [ms.randstr(n=12) for _ in range(world.n_cells)]


class Config:
    """Config container"""

    def __init__(
        self,
        device: str,
        runs_dir: Path | str,
        max_steps: int,
        max_time_m: int,
        n_trials: int,
    ):
        self.device = device
        self.runs_dir = runs_dir if isinstance(runs_dir, Path) else Path(runs_dir)
        self.max_steps = max_steps
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
            max_time_m=kwargs.pop("max_time_m"),
            n_trials=kwargs.pop("n_trials"),
        )
