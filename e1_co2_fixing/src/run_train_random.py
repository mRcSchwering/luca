import time
import random
from itertools import cycle
import magicsoup as ms
from .util import Config, load_cells
from .managing import BatchCultureManager
from .chemistry import _X, _E, _co2, FREE_STAGES_MAP, ADDITIVES, SUBSTRATES
from .culture import Culture, BatchCulture
from .generators import Killer, Replicator, Stopper


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


class ComplexPassager:
    """Passage cells with varying prioritizations"""

    def __init__(
        self,
        world: ms.World,
        mol: ms.Molecule,
        n_by_mol=0,
        n_by_size=0,
        n_random=1,
        cnfls=(0.2, 0.7),
    ):
        n_max = world.map_size**2
        self.mol_i = world.chemistry.mol_2_idx[mol]
        self.min_cells = int(n_max * min(cnfls))
        self.max_cells = int(n_max * max(cnfls))
        self.modes = (
            ["random"] * n_random
            + ["genome-size"] * n_by_size
            + ["molecule"] * n_by_mol
        )
        self.cycle_modes = cycle(self.modes)
        self.idx_fun_map = {
            "random": self._get_random_idxs,
            "genome-size": self._get_genome_size_idxs,
            "molecule": self._get_molecule_idxs,
        }

    def _get_random_idxs(self, world: ms.World, kill_n: int) -> list[int]:
        return random.sample(range(world.n_cells), k=kill_n)

    def _get_genome_size_idxs(self, world: ms.World, kill_n: int) -> list[int]:
        glens = [len(d) for d in world.cell_genomes]
        ordered = sorted([(d, i) for i, d in enumerate(glens)])
        return [i for _, i in ordered[:kill_n]]

    def _get_molecule_idxs(self, world: ms.World, kill_n: int) -> list[int]:
        x = world.cell_molecules[:, self.mol_i]
        ordered = sorted([(d, i) for i, d in enumerate(x.tolist())], reverse=True)
        return [i for _, i in ordered[:kill_n]]

    def __call__(self, cltr: BatchCulture) -> bool:
        if cltr.world.n_cells < self.max_cells:
            return False

        mode = next(self.cycle_modes)
        n_old = cltr.world.n_cells
        kill_n = max(n_old - self.min_cells, 0)
        idxs = self.idx_fun_map[mode](world=cltr.world, kill_n=kill_n)
        cltr.world.kill_cells(cell_idxs=idxs)
        cltr.world.reposition_cells()
        return True


class MediumRefresher:
    """Changes reduce unnecessary medium parts from A to B when progress reached"""

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


def run_trial(run_name: str, config: Config, hparams: dict) -> float:
    from_f, to_f = FREE_STAGES_MAP[hparams["stage"]]
    from_val = from_f * hparams["substrates_init"]
    to_val = to_f * hparams["substrates_init"]
    n_init_splits = hparams["n_init_splits"]
    n_init_adapt_splits = n_init_splits + hparams["n_adapt_splits"]
    n_total_splits = n_init_adapt_splits + hparams["n_final_splits"]
    adaption_start = n_init_splits / n_total_splits
    adaption_end = n_init_adapt_splits / n_total_splits
    print(f"Adaption lasts from progress {adaption_start:.2f} to {adaption_end:.2f}")
    print(f"Non-essential molecules are reduced from {from_val:.2f} to {to_val:.2f}")

    trial_dir = config.runs_dir / run_name
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

    stopper = Stopper(**vars(config))
    killer = Killer(world=world, mol=_E)
    replicator = Replicator(world=world, mol=_X)
    progressor = Progressor(n_splits=n_total_splits, min_gr=hparams["min_gr"])

    passager = ComplexPassager(
        world=world,
        mol=_co2,
        n_by_mol=hparams["passage_by_co2"],
        n_by_size=hparams["passage_by_genome_size"],
        n_random=hparams["passage_random"],
        cnfls=(hparams["min_confl"], hparams["max_confl"]),
    )
    print("Passaging modes: " + ", ".join(passager.modes))

    medium_refresher = MediumRefresher(
        world=world,
        substrates=SUBSTRATES,
        additives=ADDITIVES,
        at_progress=adaption_start,
        additives_val=hparams["additives_init"],
        substrates_val=hparams["substrates_init"],
        other_val_a=from_val,
        other_val_b=to_val,
    )

    mutator = Mutator(
        progress_range=(adaption_start, adaption_end),
        by=hparams["mutation_rate_mult"],
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
            manager.throttled_light_log(step, {"Other/TimePerStep[s]": t1 - t0})
            manager.throttled_fat_log(step)
            manager.throttled_save_state(step)
            t0 = t1

    return cltr.progress
