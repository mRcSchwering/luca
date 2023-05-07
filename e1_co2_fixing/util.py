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
    p = sigm(t=t, k=k, n=n)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def rev_sigm_sample(t: torch.Tensor, k: float, n: int) -> list[int]:
    """Sample with probability $k^n / (t^n + k^n)$"""
    p = rev_sigm(t=t, k=k, n=n)
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


def init_writer(logdir: Path, hparams: dict, score="Other/Score") -> SummaryWriter:
    """Write initial hparams to tensorboard and as JSON"""
    writer = SummaryWriter(log_dir=logdir)
    exp, ssi, sei = get_summary(hparam_dict=hparams, metric_dict={score: 0.0})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    with open(logdir / "hparams.json", "w", encoding="utf-8") as fh:
        json.dump(hparams, fh)

    return writer
