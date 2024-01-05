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

    medium_refresher = MediumRefresher(
        world=world,
        val=hparams["substrates_init"],
        molecules=WL_STAGES_MAP["WL-0"][1],
    )

    passager = Passager(
        world=world, min_confl=hparams["min_confl"], max_confl=hparams["max_confl"]
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

    logger = BatchCultureCheckpointer(
        trial_dir=trial_dir,
        hparams=hparams,
        cltr=cltr,
        watch_mols=[_X, _E],
        scalar_freq=5,
        img_freq=50,
        save_freq=100,
    )

    with logger as log:
        log.save_state()
        t0 = time.time()
        for _ in cltr:
            t1 = time.time()
            log.log_scalars(dtime=t1 - t0)
            log.log_imgs()
            log.save_state()
            t0 = t1
