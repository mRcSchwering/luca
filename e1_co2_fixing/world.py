from pathlib import Path
import magicsoup as ms
from e1_co2_fixing.chemistry import CHEMISTRY


def init_world(map_size: int, rundir: Path):
    world = ms.World(
        chemistry=CHEMISTRY,
        map_size=map_size,
        mol_map_init="zeros",
    )
    world.save(rundir=rundir)


def load_world(rundir: Path, device: str, n_workers: int) -> ms.World:
    world = ms.World.from_file(rundir=rundir, device=device)
    world.workers = n_workers
    world.genetics.workers = n_workers
    return world
