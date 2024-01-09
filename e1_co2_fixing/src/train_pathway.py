import time
import random
import magicsoup as ms
from .util import Config, load_cells
from .managing import BatchCultureManager
from .chemistry import WL_STAGES_MAP, _X, _E
from .culture import Culture, BatchCulture
from .generators import Killer, Replicator, Stopper, Passager, GenomeEditor


class Progressor:
    """Advance progress by splits if growth rate high enough"""

    def __init__(self, n_splits: int, min_gr: float):
        self.min_gr = min_gr
        self.n_valid_total_splits = n_splits
        self.valid_split_i = 0

    def __call__(self, cltr: BatchCulture) -> float:
        if cltr.growth_rate >= self.min_gr:
            self.valid_split_i += 1
        return min(1.0, self.valid_split_i / self.n_valid_total_splits)


class MediumRefresher:
    """Changes substrates from A to B when progress reached"""

    def __init__(
        self,
        world: ms.World,
        additives: list[ms.Molecule],
        substrates_a: list[ms.Molecule],
        substrates_b: list[ms.Molecule],
        at_progress: float,
        substrates_val: float,
        additives_val: float,
    ):
        self.at_progress = at_progress
        self.substrates_val = substrates_val
        self.additives_val = additives_val
        self.subs_a_idxs = [world.chemistry.mol_2_idx[d] for d in substrates_a]
        self.subs_b_idxs = [world.chemistry.mol_2_idx[d] for d in substrates_b]
        self.add_idxs = [world.chemistry.mol_2_idx[d] for d in additives]
        self.other_a_idxs = list(
            set(world.chemistry.mol_2_idx.values())
            - set(self.subs_a_idxs)
            - set(self.add_idxs)
        )
        self.other_b_idxs = list(
            set(world.chemistry.mol_2_idx.values())
            - set(self.subs_b_idxs)
            - set(self.add_idxs)
        )

    def __call__(self, cltr: Culture):
        if cltr.progress < self.at_progress:
            subs_idxs = self.subs_a_idxs
            other_idxs = self.other_a_idxs
        else:
            subs_idxs = self.subs_b_idxs
            other_idxs = self.other_b_idxs
        cltr.world.molecule_map[other_idxs] = 0.0
        cltr.world.molecule_map[self.add_idxs] = self.additives_val
        cltr.world.molecule_map[subs_idxs] = self.substrates_val


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


def run_trial(run_name: str, config: Config, hparams: dict):
    genes, subs_a, subs_b, add = WL_STAGES_MAP[hparams["pathway-label"]]
    n_init_splits = hparams["n_init_splits"]
    n_init_adapt_splits = n_init_splits + hparams["n_adapt_splits"]
    n_total_splits = n_init_adapt_splits + hparams["n_final_splits"]
    adaption_start = n_init_splits / n_total_splits
    adaption_end = n_init_adapt_splits / n_total_splits
    print(f"Adaption lasts from progress {adaption_start:.2f} to {adaption_end:.2f}")
    print("Medium will change from substrates a to substrates b")
    print(f"  substrates a: {', '.join(d.name for d in subs_a)}")
    print(f"  substrates b: {', '.join(d.name for d in subs_b)}")
    print(f"  additives: {', '.join(d.name for d in add)}")

    trial_dir = config.runs_dir / run_name
    world = ms.World.from_file(rundir=config.runs_dir, device=config.device)

    if hparams["init-label"] == "init":
        ggen = ms.GenomeFact(
            world=world,
            proteome=[[ms.TransporterDomainFact(_X)], [ms.TransporterDomainFact(_E)]],
        )
        genomes = [ggen.generate() for _ in range(int(0.5 * world.map_size**2))]
        world.spawn_cells(genomes=genomes)
    else:
        load_cells(world=world, label=hparams["init-label"], runsdir=config.runs_dir)

    stopper = Stopper(vars(config))
    killer = Killer(world=world, mol=_E)
    replicator = Replicator(world=world, mol=_X)
    progressor = Progressor(n_splits=n_total_splits, min_gr=hparams["min_gr"])
    passager = Passager(world=world, cnfls=(hparams["min_confl"], hparams["max_confl"]))

    medium_refresher = MediumRefresher(
        world=world,
        substrates_a=subs_a,
        substrates_b=subs_b,
        additives=add,
        at_progress=adaption_start,
        additives_val=hparams["additives_init"],
        substrates_val=hparams["substrates_init"],
    )

    mutator = Mutator(
        progress_range=(adaption_start, adaption_end),
        by=hparams["mutation_rate_mult"],
    )

    ggen = ms.GenomeFact(world=world, proteome=genes)
    genome_editor = GenomeEditor(at_progress=adaption_start, fact=ggen)

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
        watch_mols=list(set(subs_a + subs_b + add)),
    )

    with manager:
        t0 = time.time()
        for step in cltr:
            t1 = time.time()
            manager.throttled_light_log(step, {"Other/TimePerStep[s]": t1 - t0})
            manager.throttled_fat_log(step)
            manager.throttled_save_state(step)
            t0 = t1
