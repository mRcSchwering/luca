import time
import torch
import magicsoup as ms
from .util import Config, load_cells
from .checkpointing import ChemoStatCheckpointer
from .chemistry import SUBSTRATES, ADDITIVES, _E, _X
from .culture import Culture, ChemoStat
from .generators import (
    Progressor,
    Killer,
    Replicator,
    Mutator,
    Stopper,
)


class MediumRefresher:
    """1D gradient by adding molecules in middle and removing at edges"""

    def __init__(
        self,
        world: ms.World,
        substrates: list[ms.Molecule],
        additives: list[ms.Molecule],
        substrates_val: float,
        additives_val: float,
    ):
        self.substrates_val = substrates_val
        self.additives_val = additives_val
        self.subs_idxs = [world.chemistry.mol_2_idx[d] for d in substrates]
        self.add_idxs = [world.chemistry.mol_2_idx[d] for d in additives]

        s = world.map_size
        m = int(s / 2)
        w = int(s * 0.05)

        self.set_mask = torch.zeros_like(world.molecule_map).bool()
        for idx in self.subs_idxs:
            self.set_mask[idx, list(range(m - w, m + w))] = True

        self.rm_mask = torch.zeros_like(world.molecule_map).bool()
        self.rm_mask[:, list(range(0, w)) + list(range(s - w, s))] = True

    def __call__(self, cltr: Culture):
        cltr.world.molecule_map[self.set_mask] = self.substrates_val
        cltr.world.molecule_map[self.set_mask] = self.additives_val
        cltr.world.molecule_map[self.rm_mask] = 0.0


def run_trial(run_name: str, config: Config, hparams: dict):
    trial_dir = config.runs_dir / run_name
    world = ms.World.from_file(rundir=config.runs_dir, device=config.device)

    mutator = Mutator()
    stopper = Stopper(max_steps=config.max_steps, max_time_m=config.max_time_m)
    killer = Killer(world=world, mol=_E)
    replicator = Replicator(world=world, mol=_X)
    progressor = Progressor(n_avg_divisions=hparams["n_divisions"])

    medium_refresher = MediumRefresher(
        world=world,
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        substrates_val=hparams["substrates_init"],
        additives_val=hparams["additives_init"],
    )

    cltr = ChemoStat(
        world=world,
        medium_refresher=medium_refresher,
        killer=killer,
        replicator=replicator,
        mutator=mutator,
        progressor=progressor,
        stopper=stopper,
    )

    # load previous cells
    load_cells(world=world, label=hparams["init-label"], runsdir=config.runs_dir)

    manager = ChemoStatCheckpointer(
        trial_dir=trial_dir,
        hparams=hparams,
        cltr=cltr,
        watch_mols=list(set(SUBSTRATES + ADDITIVES)),
        scalar_freq=1,
        img_freq=5,
        save_freq=10,
    )

    with manager:
        t0 = time.time()
        for step in cltr:
            t1 = time.time()
            manager.throttled_log_scalars(step, {"Other/TimePerStep[s]": t1 - t0})
            manager.throttled_log_imgs(step)
            manager.throttled_save_state(step)
            t0 = t1
