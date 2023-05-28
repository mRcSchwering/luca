import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
import magicsoup as ms


class Finished(Exception):
    """Raise to finish experiment"""


def sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """$t^n / (t^n + k^n)$"""
    return t**n / (t**n + k**n)


def rev_sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """$k^n / (t^n + k^n)$"""
    return k**n / (t**n + k**n)


def sigm_sample(t: torch.Tensor, k: float, n: int) -> list[int]:
    """Sample with probability $t^n / (t^n + k^n)$"""
    p = sigm(t=t.float(), k=k, n=n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def rev_sigm_sample(t: torch.Tensor, k: float, n: int) -> list[int]:
    """Sample with probability $k^n / (t^n + k^n)$"""
    p = rev_sigm(t=t.float(), k=k, n=n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def init_writer(logdir: Path, hparams: dict, score="Other/Progress") -> SummaryWriter:
    """Write initial hparams to tensorboard and as JSON"""
    writer = SummaryWriter(log_dir=logdir)
    exp, ssi, sei = get_summary(hparam_dict=hparams, metric_dict={score: 0.0})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    with open(logdir / "hparams.json", "w", encoding="utf-8") as fh:
        json.dump(hparams, fh)

    return writer


def find_steps(rundir: Path) -> list[int]:
    """Get all sorted steps of rundir"""
    names = [d.name for d in rundir.glob("step=*")]
    return sorted(int(d.split("step=")[-1]) for d in names)


def load_cells(world: ms.World, label: str, runsdir: Path):
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

    world.load_state(statedir=statedir, batch_size=500)
