from pathlib import Path
import torch
import magicsoup as ms
from .chemistry import CHEMISTRY, get_proteome_facts


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


def generate_genomes(
    rundir: Path, genome_size: int, n_genomes: int, add_enzymes: bool
) -> list[str]:
    """Generate genomes of a certain size with defined proteomes"""
    world = ms.World.from_file(rundir=rundir, device="cpu", workers=0)
    proteomes = get_proteome_facts(n=n_genomes, add_enzymes=add_enzymes)

    seqs = [world.generate_genome(proteome=p, size=genome_size) for p in proteomes]

    return seqs
