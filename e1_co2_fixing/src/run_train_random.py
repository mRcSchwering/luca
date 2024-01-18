import time
import random
import shutil
from pathlib import Path
import magicsoup as ms
from .util import Config, load_cells, RUNS_DIR
from .managing import BatchCultureManager
from .chemistry import _X, _E, _co2, ADDITIVES, SUBSTRATES
from .culture import Culture, BatchCulture
from .generators import Killer, Replicator, Stopper, Passager


class Progressor:
    """Advance progress by splits if growth rate high enough"""

    def __init__(
        self,
        n_init_splits: int,
        n_adapt_splits: int,
        n_final_splits: int,
        min_grs: list[float],
    ):
        self.n_valid_total_splits = (
            n_init_splits + n_adapt_splits * len(min_grs) + n_final_splits
        )
        adapt = [
            min_grs[d // n_adapt_splits] for d in range(n_adapt_splits * len(min_grs))
        ]
        self.min_grs = (
            [max(min_grs)] * n_init_splits + adapt + [max(min_grs)] * n_final_splits
        )
        self.valid_split_i = 0

    def __call__(self, cltr: BatchCulture) -> float:
        if self.valid_split_i >= len(self.min_grs):
            return 1.0
        if cltr.growth_rate >= self.min_grs[self.valid_split_i]:
            self.valid_split_i += 1
        return min(1.0, self.valid_split_i / self.n_valid_total_splits)


class GenomeEditor:
    """Give cells random base pairs if they cannot progress"""

    def __init__(self, max_steps: float, size=100):
        self.size = size
        self.prev_state = (0.0, 0)
        self.max_steps = max_steps

    def __call__(self, cltr: Culture):
        n_steps = cltr.step_i - self.prev_state[1]
        if cltr.progress <= self.prev_state[0] and n_steps > self.max_steps:
            updates = [
                (cltr.world.cell_genomes[d] + ms.random_genome(self.size), d)
                for d in range(cltr.world.n_cells)
            ]
            cltr.world.update_cells(genome_idx_pairs=updates)
            self.prev_state = (cltr.progress, cltr.step_i)
        elif cltr.progress > self.prev_state[0]:
            self.prev_state = (cltr.progress, cltr.step_i)


class MediumRefresher:
    """Reduce unnecessary medium parts from A to B"""

    def __init__(
        self,
        world: ms.World,
        additives: list[ms.Molecule],
        substrates: list[ms.Molecule],
        substrates_val: float,
        additives_val: float,
        other_val_a: float,
        other_val_b: float,
        at_progress: float,
    ):
        self.at_progress = at_progress
        self.substrates_val = substrates_val
        self.additives_val = additives_val
        self.other_val_a = other_val_a
        self.other_val_b = other_val_b
        self.subs_idxs = [world.chemistry.mol_2_idx[d] for d in substrates]
        self.add_idxs = [world.chemistry.mol_2_idx[d] for d in additives]
        self.other_idxs = list(
            set(world.chemistry.mol_2_idx.values())
            - set(self.subs_idxs)
            - set(self.add_idxs)
        )

    def __call__(self, cltr: Culture):
        if cltr.progress < self.at_progress:
            other_val = self.other_val_a
        else:
            other_val = self.other_val_b
        cltr.world.molecule_map[self.other_idxs] = other_val
        cltr.world.molecule_map[self.add_idxs] = self.additives_val
        cltr.world.molecule_map[self.subs_idxs] = self.substrates_val


class Mutator:
    """Increase mutation rates if cells dont progress"""

    def __init__(
        self,
        n_splits: int,
        by: float,
        snp_p=1e-6,
        lgt_p=1e-7,
        lgt_rate=0.1,
    ):
        self.prev_split = 0
        self.prev_progress = 0.0
        self.n_splits = n_splits
        self.by = by
        self.snp_p = snp_p
        self.lgt_p = lgt_p
        self.lgt_rate = lgt_rate
        self.current_factor = 1.0

    def _update_step(self, cltr: BatchCulture):
        if cltr.progress > self.prev_progress:
            self.prev_progress = cltr.progress
            self.prev_split = cltr.split_i

    def __call__(self, cltr: BatchCulture):
        self._update_step(cltr)
        snp_p = self.snp_p
        lgt_p = self.lgt_p
        self.current_factor = 1.0
        if cltr.split_i - self.prev_split > self.n_splits:
            snp_p *= self.by
            lgt_p *= self.by
            self.current_factor = self.by
        cltr.world.mutate_cells(p=snp_p)
        n_cells = cltr.world.n_cells
        idxs = random.sample(range(n_cells), k=int(n_cells * self.lgt_rate))
        cltr.world.recombinate_cells(cell_idxs=idxs, p=self.lgt_p)


def run_trial(trial_dir: Path, config: Config, hparams: dict) -> float:
    substrates_val = hparams["substrates_init"]
    additives_val = hparams["additives_init"]
    from_val = hparams["non_essential_init_a"]
    to_val = hparams["non_essential_init_b"]
    n_init_splits = hparams["n_init_splits"]
    n_adapt_splits = hparams["n_adapt_splits"]
    n_final_splits = hparams["n_final_splits"]
    min_grs = hparams["min_grs"]
    n_total_splits = n_init_splits + n_adapt_splits * len(min_grs) + n_final_splits
    adaption_start = n_init_splits / n_total_splits
    print(f"Starting trial {trial_dir.name}")
    print(f"Non-essential molecules are reduced from {from_val:.2f} to {to_val:.2f}")

    world = ms.World.from_file(rundir=config.runs_dir, device=config.device)

    init_confl = hparams["init_confl"]
    if hparams["init-label"] == "init":
        ggen = ms.GenomeFact(
            world=world,
            proteome=[[ms.TransporterDomainFact(_X)], [ms.TransporterDomainFact(_E)]],
        )
        target_n = int(init_confl * world.map_size**2)
        genomes = [ggen.generate() for _ in range(target_n)]
        world.spawn_cells(genomes=genomes)
    else:
        load_cells(world=world, label=hparams["init-label"], target_confl=init_confl)

    stopper = Stopper.from_config(cnfg=config, world=world)
    killer = Killer(world=world, mol=_E)
    replicator = Replicator(world=world, mol=_X)
    passager = Passager(world=world, cnfls=(hparams["min_confl"], hparams["max_confl"]))

    progressor = Progressor(
        n_init_splits=n_init_splits,
        n_adapt_splits=n_adapt_splits,
        n_final_splits=n_final_splits,
        min_grs=min_grs,
    )

    medium_refresher = MediumRefresher(
        world=world,
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        at_progress=adaption_start,
        additives_val=additives_val,
        substrates_val=substrates_val,
        other_val_a=from_val,
        other_val_b=to_val,
    )

    mutator = Mutator(
        n_splits=hparams["mutation_rate_splits"], by=hparams["mutation_rate_mult"]
    )

    genome_editor = GenomeEditor(
        max_steps=hparams["genome_editing_steps"], size=hparams["genome_editing_size"]
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
        genome_editor=genome_editor,
    )

    manager = BatchCultureManager(
        trial_dir=trial_dir,
        hparams=hparams,
        cltr=cltr,
        watch_mols=list(set(SUBSTRATES + ADDITIVES + [_X, _E, _co2])),
    )

    with manager:
        t0 = time.time()
        for step in cltr:
            t1 = time.time()
            log = {
                "Other/TimePerStep[s]": t1 - t0,
                "Other/MutationRateMult": cltr.mutator.current_factor,
            }
            manager.throttled_light_log(step, log)
            manager.throttled_fat_log(step)
            manager.throttled_save_state(step)
            t0 = t1

    return cltr.progress


def run_trials(cmd: str, kwargs: dict):
    config = Config.pop_from(kwargs)
    kwargs["runs_dir"] = RUNS_DIR
    orig_from_val = kwargs["non_essentials_init"]
    print(f"Starting free-training trials on {config.device}")

    default_stage_size = 0.5
    stage_size = default_stage_size
    from_val = orig_from_val
    successful_trials = []

    while True:
        to_val = from_val - from_val * stage_size
        for trial_i in range(config.max_trials):
            trial_dir = RUNS_DIR / f"{cmd}_{config.timestamp}_{trial_i}"
            kwargs["non_essential_init_a"] = from_val
            kwargs["non_essential_init_b"] = to_val
            progress = run_trial(trial_dir=trial_dir, config=config, hparams=kwargs)
            if progress == 1.0:
                successful_trials.append(trial_dir.name)
            else:
                shutil.rmtree(trial_dir)
            if len(successful_trials) >= config.max_successful_trials:
                break
        if len(successful_trials) > 0:
            stage_size = default_stage_size
            from_val = to_val
            kwargs["init-label"] = f"{successful_trials[0]}:-1"
            successful_trials = []
        else:
            stage_size /= 2
        print(f"Finished {len(successful_trials)} trials successfully")
        config.reset()
