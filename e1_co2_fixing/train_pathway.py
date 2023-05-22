from pathlib import Path
from typing import Callable
import time
import math
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import SUBSTRATE_MOLS, WL_STAGES
from .util import init_writer, load_genomes
from .experiment import (
    Experiment,
    PassageByCellAndSubstrates,
    MutationRateFact,
    MediumFact,
    GenomeSizeController,
    MoleculeDependentCellDivision,
    MoleculeDependentCellDeath,
)


THIS_DIR = Path(__file__).parent


def _log_scalars(
    exp: Experiment,
    writer: SummaryWriter,
    step: int,
    dtime: float,
    mols: list[tuple[str, int]],
):
    n_cells = exp.world.n_cells
    molecule_map = exp.world.molecule_map
    cell_molecules = exp.world.cell_molecules
    molecules = {f"Molecules/{s}": i for s, i in mols}

    for scalar, idx in molecules.items():
        tag = f"{scalar}[ext]"
        writer.add_scalar(tag, molecule_map[idx].mean().item(), step)

    if n_cells > 0:
        writer.add_scalar("Cells/Total", n_cells, step)
        mean_surv = exp.world.cell_survival.float().mean()
        writer.add_scalar("Cells/Survival", mean_surv, step)
        writer.add_scalar("Cells/Generation", exp.gen_i, step)
        writer.add_scalar("Cells/GrowthRate", exp.growth_rate, step)
        writer.add_scalar(
            "Cells/GenomeSize", sum(len(d) for d in exp.world.genomes) / n_cells, step
        )
        for scalar, idx in molecules.items():
            tag = f"{scalar}[int]"
            writer.add_scalar(tag, cell_molecules[:, idx].mean().item(), step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)
    writer.add_scalar("Other/Split", exp.split_i, step)
    writer.add_scalar("Other/Progress", exp.progress, step)
    writer.add_scalar("Other/MutationRate", exp.mutation_rate, step)


def _log_imgs(exp: Experiment, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


class MutationRateSteps(MutationRateFact):
    def __init__(self, progress_rate_pairs: list[tuple[float, float]]):
        self.progress_rate_pairs = progress_rate_pairs

    def __call__(self, exp: Experiment) -> float:
        for progress, rate in self.progress_rate_pairs:
            if exp.progress >= progress:
                return rate
        return 0.0


# TODO: MediumFact braucht init_done usw auf superclass
#       oder ich muss vorsichtig nach dem progress gucken
class DynamicAdaption(MediumFact):
    def __init__(
        self,
        essentials: list[str],
        cmplx_substrates: list[str],
        mini_substrates: list[str],
        n_init_splits: float,
        n_adapt_splits: float,
        n_final_splits: float,
        min_gr: float,
        substrates_max: float,
        essentials_max: float,
        molmap: torch.Tensor,
        mol_2_idx: dict[str, int],
    ):
        self.substrates = list(set(cmplx_substrates + mini_substrates))
        self.essentials = essentials
        self.substrates_max = substrates_max
        self.essentials_max = essentials_max
        self.cmplx_subs_idxs = [mol_2_idx[d] for d in cmplx_substrates]
        self.mini_subs_idxs = [mol_2_idx[d] for d in mini_substrates]
        self.ess_idxs = [mol_2_idx[d] for d in essentials]
        self.molmap = molmap

        self.min_gr = min_gr

        self.init_done = False
        self.adapt_done = False
        self.final_done = False
        self.init_split_i = 0
        self.adapt_split_i = 0
        self.final_split_i = 0
        self.n_init_splits = n_init_splits
        self.n_adapt_splits = n_adapt_splits
        self.n_final_splits = n_final_splits

        self.n_total_splits = (
            self.n_init_splits + self.n_adapt_splits + self.n_final_splits
        )

    def __call__(self, exp: Experiment) -> torch.Tensor:
        if exp.growth_rate >= self.min_gr:
            if not self.init_done:
                self.init_split_i += 1
            elif not self.adapt_done:
                self.adapt_split_i += 1
            elif not self.final_done:
                self.final_split_i += 1

        if self.init_split_i >= self.n_init_splits:
            self.init_done = True
        if self.adapt_split_i >= self.n_adapt_splits:
            self.adapt_done = True
        if self.final_split_i >= self.n_final_splits:
            self.final_done = True

        total_splits = self.init_split_i + self.adapt_split_i + self.final_split_i
        exp.progress = min(1.0, total_splits / self.n_total_splits)

        if not self.init_done:
            subs_idxs = self.cmplx_subs_idxs
        else:
            subs_idxs = self.mini_subs_idxs

        t = torch.zeros_like(self.molmap)
        t[self.ess_idxs] = self.essentials_max
        t[subs_idxs] = self.substrates_max
        return t


# TODO: mit einbinden
class GenomeEditor:
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
    pathway_label = hparams["pathway_label"]
    runsdir = THIS_DIR / "runs"
    trial_dir = runsdir / run_name
    world = ms.World.from_file(rundir=runsdir, device=device, workers=n_workers)
    mol_2_idx = {d.name: i for i, d in enumerate(world.chemistry.molecules)}
    n_pxls = world.map_size**2

    # stage: (new genes, complex substrates, minimal substrates, essentials)
    genes, cmplx, mini, ess = WL_STAGES[pathway_label]
    substrates = sorted(d.name for d in SUBSTRATE_MOLS)
    print("Medium will change from complex to minimal")
    print(f"  complex: {', '.join(cmplx)}")
    print(f"  minimal: {', '.join(mini)}")
    print(f"  essentials: {', '.join(ess)}")
    print(f"  (disregarding substrates {', '.join(substrates)})")

    n_init_splits = hparams["n_init_splits"]
    n_init_adapt_splits = n_init_splits + hparams["n_adapt_splits"]
    n_total_splits = n_init_adapt_splits + hparams["n_final_splits"]

    medium_fact = DynamicAdaption(
        cmplx_substrates=cmplx,
        mini_substrates=mini,
        essentials=ess,
        molmap=world.molecule_map,
        mol_2_idx=mol_2_idx,
        n_init_splits=n_init_splits,
        n_adapt_splits=hparams["n_adapt_splits"],
        n_final_splits=hparams["n_final_splits"],
        essentials_max=10.0,
        substrates_max=100.0,
        min_gr=0.02,
    )

    genome_editor = GenomeEditor(
        at_progress=n_init_splits / n_total_splits,
        genfun=world.generate_genome(proteome=genes, size=hparams["gene_size"]),
    )

    mutation_rate_fact = MutationRateSteps(
        progress_rate_pairs=[
            (0.0, 1e-6),
            (n_init_splits / n_total_splits, 1e-4),
            (n_init_adapt_splits / n_total_splits, 1e-6),
        ]
    )

    passager = PassageByCellAndSubstrates(
        split_ratio=hparams["split_ratio"],
        split_thresh_subs=hparams["split_thresh_subs"],
        split_thresh_cells=hparams["split_thresh_cells"],
        max_subs=medium_fact.substrates_max * n_pxls,
        max_cells=n_pxls,
    )

    division_by_x = MoleculeDependentCellDivision(
        mol_i=mol_2_idx["X"], k=hparams["mol_divide_k"], n=3
    )
    death_by_e = MoleculeDependentCellDeath(
        mol_i=mol_2_idx["E"], k=hparams["mol_kill_k"], n=1
    )
    genome_size_controller = GenomeSizeController(k=hparams["genome_kill_k"], n=7)

    exp = Experiment(
        world=world,
        lgt_rate=hparams["lgt_rate"],
        passager=passager,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
        division_by_x=division_by_x,
        death_by_e=death_by_e,
        genome_size_controller=genome_size_controller,
    )

    # load initial genomes
    init_label = hparams["init_label"]
    genome_size = hparams["genome_size"]
    n_cells = int(n_pxls * hparams["init_cell_cover"])
    if init_label == "random":
        genomes = [ms.random_genome(s=genome_size) for _ in range(n_cells)]
    else:
        pop = load_genomes(label=init_label, runsdir=runsdir)
        genomes = random.choices(pop, k=n_cells)

    exp.init_cells(genomes=genomes)

    avg_genome_len = sum(len(d) for d in world.genomes) / world.n_cells
    print(f"In total {len(genes)} genes were added")
    print(f"   Average genome size is {avg_genome_len:,}")

    writer = init_writer(
        logdir=trial_dir,
        hparams={
            **hparams,
            "mut_rates": f"{1e-4:.0e}-{1e-6:.0e}",
            "cell_split": f"{passager.split_ratio:.1f}-{passager.split_thresh_cells:.1f}",
            "substrates_min": passager.split_thresh_subs,
            "substrates_max": medium_fact.substrates_max,
            "essentials_max": medium_fact.essentials_max,
        },
    )
    exp.world.save_state(statedir=trial_dir / "step=0")

    trial_t0 = time.time()
    watch_substrates = [(d, mol_2_idx[d]) for d in medium_fact.substrates]
    watch_essentials = [(d, mol_2_idx[d]) for d in medium_fact.essentials]
    watch = watch_substrates + watch_essentials
    print(f"Starting trial {run_name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    _log_scalars(exp=exp, writer=writer, step=0, dtime=0, mols=watch)
    _log_imgs(exp=exp, writer=writer, step=0)

    min_cells = int(exp.world.map_size**2 * 0.01)
    for step_i in exp.run(max_steps=n_steps):
        step_t0 = time.time()

        exp.step_1s()
        dtime = time.time() - step_t0

        if exp.progress >= 1.0:
            print(f"target reached after {step_i + 1} steps")
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            _log_scalars(exp=exp, writer=writer, step=step_i, dtime=dtime, mols=watch)
            break

        if step_i % 5 == 0:
            _log_scalars(exp=exp, writer=writer, step=step_i, dtime=dtime, mols=watch)

        if step_i % 50 == 0:
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            _log_imgs(exp=exp, writer=writer, step=step_i)

        if exp.world.n_cells < min_cells:
            print(f"after {step_i} steps less than {min_cells} cells left")
            break

        if (time.time() - trial_t0) > trial_max_time_s:
            print(f"{trial_max_time_s} hours have passed")
            break

    print(f"Finishing trial {run_name}")
    writer.close()
