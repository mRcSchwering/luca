import time
import random
import shutil
from pathlib import Path
import magicsoup as ms
from .util import Config, load_cells, RUNS_DIR
from .managing import BatchCultureManager
from .chemistry import _X, _E, _co2, ADDITIVES, SUBSTRATES
from .culture import Culture, BatchCulture
from .generators import Killer, Replicator, BatchCultureStopper, Passager


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

    def __init__(self, max_splits: float, efficiency: float, size=100):
        self.size = size
        self.progress0 = 0.0
        self.split0 = 0
        self.max_splits = max_splits
        self.efficiency = efficiency

    def __call__(self, cltr: BatchCulture):
        n_splits = cltr.split_i - self.split0
        if cltr.progress <= self.progress0 and n_splits > self.max_splits:
            n_cells = int(cltr.world.n_cells * self.efficiency)
            updates = [
                (cltr.world.cell_genomes[d] + ms.random_genome(self.size), d)
                for d in random.sample(list(range(cltr.world.n_cells)), k=n_cells)
            ]
            cltr.world.update_cells(genome_idx_pairs=updates)
            self.progress0 = cltr.progress
            self.split0 = cltr.split_i
        elif cltr.progress > self.progress0:
            self.progress0 = cltr.progress
            self.split0 = cltr.split_i


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
    """Increase mutation rates during progress interval"""

    def __init__(
        self,
        progress_range: tuple[float, float],
        by: float,
        snp_p=1e-6,
        lgt_p=1e-7,
        lgt_rate=0.1,
    ):
        self.start = min(progress_range)
        self.end = max(progress_range)
        self.by = by
        self.snp_p = snp_p
        self.lgt_p = lgt_p
        self.lgt_rate = lgt_rate

    def __call__(self, cltr: Culture):
        snp_p = self.snp_p
        lgt_p = self.lgt_p
        if self.start < cltr.progress < self.end:
            snp_p *= self.by
            lgt_p *= self.by
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
    adaption_end = (n_total_splits - n_final_splits) / n_total_splits
    print(f"Starting trial {trial_dir.name}")
    print(f"Non-essential molecules are reduced from {from_val:.2f} to {to_val:.2f}")
    print(f"Adaption lasts from progress {adaption_start:.2f} to {adaption_end:.2f}")

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

    stopper = BatchCultureStopper.from_config(cnfg=config, world=world)
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
        progress_range=(adaption_start, adaption_end),
        by=hparams["mutation_rate_mult"],
    )

    genome_editor = GenomeEditor(
        max_splits=hparams["genome_editing_splits"],
        efficiency=hparams["relative_transformation_efficiency"],
        size=hparams["genome_editing_size"],
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
            manager.throttled_light_log(step, {"Other/TimePerStep[s]": t1 - t0})
            manager.throttled_fat_log(step)
            manager.throttled_save_state(step)
            t0 = t1

    return cltr.progress


def run_trials(cmd: str, kwargs: dict):
    kwargs["runs_dir"] = RUNS_DIR
    config = Config.pop_from(kwargs)
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
