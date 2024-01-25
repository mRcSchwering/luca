import time
import magicsoup as ms
from .util import Config, load_cells
from .managing import BatchCultureManager
from .chemistry import SUBSTRATES, ADDITIVES, _E, _X
from .culture import BatchCulture
from .generators import (
    MediumRefresher,
    Killer,
    Replicator,
    Mutator,
    Stopper,
    Passager,
    Progressor,
)


def run_trial(run_name: str, config: Config, hparams: dict) -> float:
    trial_dir = config.runs_dir / run_name
    world = ms.World.from_file(rundir=config.runs_dir, device=config.device)

    init_confl = hparams["init_confl"]
    if hparams["init-label"] == "random":
        target_n = int(init_confl * world.map_size**2)
        genomes = [ms.random_genome() for _ in range(target_n)]
        world.spawn_cells(genomes=genomes)
    else:
        load_cells(world=world, label=hparams["init-label"], target_confl=init_confl)

    mutator = Mutator()
    stopper = Stopper.from_config(cnfg=config, world=world)
    killer = Killer(world=world, mol=_E)
    replicator = Replicator(world=world, mol=_X)
    progressor = Progressor(n_avg_divisions=hparams["n_divisions"])
    passager = Passager(
        world=world,
        cnfls=(hparams["min_confl"], hparams["max_confl"]),
        max_steps=hparams["max_steps"],
    )

    medium_refresher = MediumRefresher(
        world=world,
        substrates_val=hparams["substrates_init"],
        additives_val=hparams["additives_init"],
        non_essentials_val=hparams["non_essentials_init"],
        substrates=SUBSTRATES,
        additives=ADDITIVES,
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

    manager = BatchCultureManager(
        trial_dir=trial_dir,
        hparams=hparams,
        cltr=cltr,
        watch_mols=list(set(SUBSTRATES + ADDITIVES)),
    )

    with manager:
        t0 = time.time()
        for step in cltr:
            t1 = time.time()
            manager.throttled_light_log(step, {"Other/TimePerStep[s]": t1 - t0})
            manager.throttled_fat_log(step)
            manager.throttled_save_state(step)
            t0 = t1

    return cltr.progress
