import time
import torch
import magicsoup as ms
from .util import Config, load_cells
from .managing import ChemoStatManager
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
        non_essentials_val: float,
        width=0.05,
    ):
        self.substrates_val = substrates_val
        self.additives_val = additives_val
        self.others_val = non_essentials_val
        self.subs_idxs = [world.chemistry.mol_2_idx[d] for d in substrates]
        self.add_idxs = [world.chemistry.mol_2_idx[d] for d in additives]
        self.other_idxs = list(
            set(world.chemistry.mol_2_idx.values())
            - set(self.subs_idxs)
            - set(self.add_idxs)
        )

        s = world.map_size
        m = int(s / 2)
        w = int(s * width)

        self.subs_mask = torch.zeros_like(world.molecule_map).bool()
        for idx in self.subs_idxs:
            self.subs_mask[idx, list(range(m - w, m + w))] = True

        self.add_mask = torch.zeros_like(world.molecule_map).bool()
        for idx in self.add_idxs:
            self.add_mask[idx, list(range(m - w, m + w))] = True

        self.rm_mask = torch.zeros_like(world.molecule_map).bool()
        self.rm_mask[:, list(range(0, w)) + list(range(s - w, s))] = True

    def __call__(self, cltr: Culture):
        cltr.world.molecule_map[self.subs_mask] = self.substrates_val
        cltr.world.molecule_map[self.add_mask] = self.additives_val
        cltr.world.molecule_map[self.other_idxs] = self.others_val
        cltr.world.molecule_map[self.rm_mask] = 0.0


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

    medium_refresher = MediumRefresher(
        world=world,
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        substrates_val=hparams["substrates_init"],
        additives_val=hparams["additives_init"],
        non_essentials_val=hparams["non_essentials_init"],
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

    # prepare gradient
    cltr.world.molecule_map[:] = 0.0
    for _ in range(100):
        cltr.medium_refresher(cltr)
        cltr.world.diffuse_molecules()

    manager = ChemoStatManager(
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
