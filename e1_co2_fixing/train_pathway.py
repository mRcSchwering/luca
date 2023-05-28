from pathlib import Path
from typing import Callable
import time
import torch
import magicsoup as ms
from .chemistry import WL_STAGES_MAP
from .util import load_cells, sigm_sample
from .experiment import (
    Experiment,
    BatchCulture,
    BatchCultureLogger,
    PassageByCells,
    MutationRateFact,
    MediumFact,
    BatchCultureProgress,
    CellSampler,
    MoleculeDependentCellDivision,
    MoleculeDependentCellDeath,
    GenomeEditor,
)


THIS_DIR = Path(__file__).parent


class MutationRateSteps(MutationRateFact):
    def __init__(self, progress_rate_pairs: list[tuple[float, float]]):
        self.progress_rate_pairs = sorted(progress_rate_pairs, reverse=True)

    def __call__(self, exp: Experiment) -> float:
        for progress, rate in self.progress_rate_pairs:
            if exp.progress >= progress:
                return rate
        return 0.0


class GenomeSizeController(CellSampler):
    def __init__(
        self,
        progress_k_pairs: list[tuple[float, float]],
        n: int,
    ):
        self.n = n
        self.progress_k_pairs = sorted(progress_k_pairs, reverse=True)

    def __call__(self, exp: Experiment) -> list[int]:
        genome_lens = [len(d) for d in exp.world.genomes]
        sizes = torch.tensor(genome_lens)
        for progress, k in self.progress_k_pairs:
            if exp.progress >= progress:
                return sigm_sample(sizes, k, self.n)
        return []


class AdvanceBySplitsAndGrowthRate(BatchCultureProgress):
    def __init__(self, n_total_splits: float, min_gr: float):
        self.min_gr = min_gr
        self.n_valid_total_splits = n_total_splits
        self.valid_split_i = 0

    def __call__(self, exp: BatchCulture) -> float:
        if exp.growth_rate >= self.min_gr:
            self.valid_split_i += 1

        return min(1.0, self.valid_split_i / self.n_valid_total_splits)


class InstantMediumChange(MediumFact):
    def __init__(
        self,
        additives: list[ms.Molecule],
        substrates_a: list[ms.Molecule],
        substrates_b: list[ms.Molecule],
        from_progress: float,
        substrates_init: float,
        additives_init: float,
        molmap: torch.Tensor,
        mol_2_idx: dict[str, int],
    ):
        self.substrates_init = substrates_init
        self.additives_init = additives_init
        self.subs_a_idxs = [mol_2_idx[d.name] for d in substrates_a]
        self.subs_b_idxs = [mol_2_idx[d.name] for d in substrates_b]
        self.add_idxs = [mol_2_idx[d.name] for d in additives]
        self.molmap = molmap

        self.from_progress = from_progress  # n_init_splits / n_total_splits

    def __call__(self, exp: Experiment) -> torch.Tensor:
        if exp.progress < self.from_progress:
            subs_idxs = self.subs_a_idxs
        else:
            subs_idxs = self.subs_b_idxs

        t = torch.zeros_like(self.molmap)
        t[self.add_idxs] = self.additives_init
        t[subs_idxs] = self.substrates_init
        return t


class EditAfterInit(GenomeEditor):
    def __init__(self, at_progress: float, genfun: Callable[[], str]):
        self.genfun = genfun

        # fails if size too small
        _ = self.genfun()

        self.at_progress = at_progress
        self.edited = False

    def __call__(self, exp: Experiment):
        if self.edited:
            return

        if exp.progress < self.at_progress:
            return

        pairs: list[tuple[str, int]] = []
        for cell_i, genome in enumerate(exp.world.genomes):
            pairs.append((genome + self.genfun(), cell_i))

        exp.world.update_cells(genome_idx_pairs=pairs)
        self.edited = True


def run_trial(
    device: str,
    n_workers: int,
    run_name: str,
    n_steps: int,
    trial_max_time_s: int,
    hparams: dict,
):
    # runs reference
    runsdir = THIS_DIR / "runs"
    trial_dir = runsdir / run_name
    world = ms.World.from_file(rundir=runsdir, device=device, workers=n_workers)
    mol_2_idx = {d.name: i for i, d in enumerate(world.chemistry.molecules)}
    n_pxls = world.map_size**2

    # stage: (new genes, complex substrates, minimal substrates, essentials)
    genes, subs_a, subs_b, add = WL_STAGES_MAP[hparams["pathway_label"]]
    print("Medium will change from substrates a to substrates b")
    print(f"  substrates a: {', '.join(d.name for d in subs_a)}")
    print(f"  substrates b: {', '.join(d.name for d in subs_b)}")
    print(f"  additives: {', '.join(d.name for d in add)}")

    # factories
    n_init_splits = hparams["n_init_splits"]
    n_init_adapt_splits = n_init_splits + hparams["n_adapt_splits"]
    n_total_splits = n_init_adapt_splits + hparams["n_final_splits"]

    progress_controller = AdvanceBySplitsAndGrowthRate(
        n_total_splits=n_total_splits, min_gr=hparams["min_gr"]
    )

    medium_fact = InstantMediumChange(
        substrates_a=subs_a,
        substrates_b=subs_b,
        additives=add,
        molmap=world.molecule_map,
        mol_2_idx=mol_2_idx,
        from_progress=n_init_splits / n_total_splits,
        additives_init=hparams["additives_init"],
        substrates_init=hparams["substrates_init"],
    )

    genome_editor = EditAfterInit(
        at_progress=n_init_splits / n_total_splits,
        genfun=lambda: world.generate_genome(proteome=genes, size=hparams["gene_size"]),
    )

    mutation_rate_fact = MutationRateSteps(
        progress_rate_pairs=[
            (0.0, hparams["mutation_rate_low"]),
            (n_init_splits / n_total_splits, hparams["mutation_rate_high"]),
            (n_init_adapt_splits / n_total_splits, hparams["mutation_rate_low"]),
        ]
    )

    passager = PassageByCells(
        split_ratio=hparams["split_ratio"],
        split_thresh=hparams["split_thresh"],
        max_cells=n_pxls,
    )

    division_by_x = MoleculeDependentCellDivision(
        mol_i=mol_2_idx["X"], k=hparams["mol_divide_k"], n=3
    )
    death_by_e = MoleculeDependentCellDeath(
        mol_i=mol_2_idx["E"], k=hparams["mol_kill_k"], n=1
    )
    genome_size_controller = GenomeSizeController(
        progress_k_pairs=[
            (0.0, hparams["genome_kill_k"]),
            (n_init_splits / n_total_splits, hparams["genome_kill_k"] + 4000),
            (n_init_adapt_splits / n_total_splits, hparams["genome_kill_k"]),
        ],
        n=7,
    )

    # load initial cells
    load_cells(world=world, label=hparams["init_label"], runsdir=runsdir)
    world.cell_divisions[:] = 0.0
    world.labels = [ms.randstr(n=12) for _ in range(world.n_cells)]

    # init experiment with fresh medium
    exp = BatchCulture(
        world=world,
        lgt_rate=hparams["lgt_rate"],
        passager=passager,
        progress_controller=progress_controller,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
        division_by_x=division_by_x,
        death_by_e=death_by_e,
        genome_size_controller=genome_size_controller,
        genome_editor=genome_editor,
    )

    avg_genome_len = sum(len(d) for d in world.genomes) / world.n_cells
    print(f"In total {len(genes)} genes are added")
    print(f"   Average genome size is {avg_genome_len:.0f}")

    trial_t0 = time.time()
    print(f"Starting trial {run_name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")

    # start logging
    with BatchCultureLogger(
        trial_dir=trial_dir,
        hparams=hparams,
        exp=exp,
        watch_mols=list(set(add + subs_a + subs_b)),
    ) as logger:
        exp.world.save_state(statedir=trial_dir / "step=0")

        # start steps
        min_cells = int(exp.world.map_size**2 * 0.01)
        for step_i in exp.run(max_steps=n_steps):
            step_t0 = time.time()

            exp.step_1s()
            dtime = time.time() - step_t0

            if exp.progress >= 1.0:
                print(f"target reached after {step_i + 1} steps")
                exp.world.save_state(statedir=trial_dir / f"step={step_i}")
                logger.log_scalars(step=step_i, dtime=dtime)
                break

            if step_i % 5 == 0:
                logger.log_scalars(step=step_i, dtime=dtime)

            if step_i % 50 == 0:
                exp.world.save_state(statedir=trial_dir / f"step={step_i}")
                logger.log_imgs(step=step_i)

            if exp.world.n_cells < min_cells:
                print(f"after {step_i} steps less than {min_cells} cells left")
                break

            if (time.time() - trial_t0) > trial_max_time_s:
                print(f"{trial_max_time_s} hours have passed")
                break

    print(f"Finishing trial {run_name}")
