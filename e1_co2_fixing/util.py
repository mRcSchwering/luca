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


def batch_add_cells(world: ms.World, genomes: list[str], d=500):
    """Add cells in batches of `d` to avoid OOM"""
    for a in range(0, len(genomes), d):
        b = a + d
        world.add_cells(genomes=genomes[a:b])


def batch_update_cells(world: ms.World, genome_idx_pairs: list[tuple[str, int]], d=500):
    """Update cells in batches of `d` to avoid OOM"""
    for a in range(0, len(genome_idx_pairs), d):
        b = a + d
        world.update_cells(genome_idx_pairs=genome_idx_pairs[a:b])


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


def load_genomes(label: str, runsdir: Path) -> list[str]:
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

    with open(statedir / "cells.fasta", "r", encoding="utf-8") as fh:
        text: str = fh.read()

    entries = [d.strip() for d in text.split(">") if len(d.strip()) > 0]
    return [d.split("\n")[1] for d in entries]
