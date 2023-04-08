from pathlib import Path
import magicsoup as ms
from .world import load_world
from .chemistry import get_proteome_fact, calvin_reacts

GENOMES = {"calvin": calvin_reacts}


def _generate_genomes(
    world: ms.World,
    s: int,
    n: int,
    reacts: list[tuple[list[ms.Molecule], list[ms.Molecule]]],
) -> list[str]:
    proteome_fact = get_proteome_fact(reacts=reacts)
    return [world.generate_genome(proteome=proteome_fact, size=s) for _ in range(n)]


def generate_genomes(
    rundir: Path, genome: str, genome_size: int, n_genomes: int
) -> list[str]:
    world = load_world(rundir=rundir, device="cpu", n_workers=0)
    reacts = GENOMES[genome]
    seqs = _generate_genomes(world=world, s=genome_size, n=n_genomes, reacts=reacts)
    return seqs
