import time
import magicsoup as ms
from .util import Config
from .checkpointing import BatchCultureCheckpointer
from .chemistry import WL_STAGES_MAP, _X, _E
from .culture import BatchCulture
from .generators import (
    MediumRefresher,
    Killer,
    Replicator,
    Mutator,
    Stopper,
    Passager,
)


class Progressor:
    """Advance progress by splits"""

    def __init__(self, n_splits: int):
        self.n_splits = n_splits

    def __call__(self, cltr: BatchCulture) -> float:
        return min(1.0, cltr.split_i / self.n_splits)


def run_trial(run_name: str, config: Config, hparams: dict):
    trial_dir = config.runs_dir / run_name
    world = ms.World.from_file(rundir=config.runs_dir, device=config.device)

    mutator = Mutator()
    stopper = Stopper(max_steps=config.max_steps, max_time_m=config.max_time_m)
    killer = Killer(world=world, mol=_E)
    replicator = Replicator(world=world, mol=_X)
    progressor = Progressor(n_splits=hparams["n_splits"])
    passager = Passager(world=world, cnfls=(hparams["min_confl"], hparams["max_confl"]))

    medium_refresher = MediumRefresher(
        world=world,
        substrates_val=hparams["substrates_init"],
        substrates=WL_STAGES_MAP["WL-0"][1],
    )

    cltr = BatchCulture(
        world=world,
        medium_refresher=medium_refresher,
        killer=killer,
        replicator=replicator,
        mutator=mutator,
        progressor=progressor,
        stopper=stopper,
        passager=passager,
    )

    # add initial cells
    ggen = ms.GenomeFact(
        world=world,
        proteome=[[ms.TransporterDomainFact(_X)], [ms.TransporterDomainFact(_E)]],
        target_size=500,
    )
    genomes = [ggen.generate() for _ in range(passager.min_cells)]
    cltr.world.spawn_cells(genomes=genomes)

    manager = BatchCultureCheckpointer(
        trial_dir=trial_dir,
        hparams=hparams,
        cltr=cltr,
        watch_mols=[_X, _E],
        scalar_freq=5,
        img_freq=50,
        save_freq=100,
    )

    with manager:
        manager.save_state()
        t0 = time.time()
        for _ in cltr:
            t1 = time.time()
            manager.log_scalars(dtime=t1 - t0)
            manager.log_imgs()
            manager.save_state()
            t0 = t1
