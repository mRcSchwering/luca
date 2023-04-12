from pathlib import Path
import torch
import magicsoup as ms
from .chemistry import CHEMISTRY, get_proteome_fact


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


def load_world(rundir: Path, device: str, n_workers: int) -> ms.World:
    """Load world with device and workers from rundir"""
    world = ms.World.from_file(rundir=rundir, device=device)
    world.workers = n_workers
    world.genetics.workers = n_workers
    return world


def generate_genomes(
    rundir: Path, name: str, genome_size: int, n_genomes: int
) -> list[str]:
    """Generate genomes of a certain size with defined proteomes"""
    world = load_world(rundir=rundir, device="cpu", n_workers=0)
    proteome_fact = get_proteome_fact(proteome_name=name)
    seqs = [
        world.generate_genome(proteome=proteome_fact, size=genome_size)
        for _ in range(n_genomes)
    ]
    return seqs
