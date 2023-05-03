from pathlib import Path
import torch
import magicsoup as ms
from .chemistry import CHEMISTRY


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


def init_world(map_size: int, rundir: Path):
    """Initialize world and save it to rundir"""
    world = ms.World(
        chemistry=CHEMISTRY,
        map_size=map_size,
        mol_map_init="zeros",
    )
    world.save(rundir=rundir)


def batch_add_cells(world: ms.World, genomes: list[str], d=1000):
    """Add cells in batches of `d` to avoid OOM"""
    for a in range(0, len(genomes), d):
        b = a + d
        world.add_cells(genomes=genomes[a:b])


def batch_update_cells(
    world: ms.World, genome_idx_pairs: list[tuple[str, int]], d=1000
):
    """Update cells in batches of `d` to avoid OOM"""
    for a in range(0, len(genome_idx_pairs), d):
        b = a + d
        world.update_cells(genome_idx_pairs=genome_idx_pairs[a:b])
