from pathlib import Path
from typing import Callable
import time
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import WL_STAGES_MAP, _X
from .util import init_writer, load_genomes
from .experiment import (
    Experiment,
    PassageByCells,
    MutationRateFact,
    MediumFact,
    GenomeSizeController,
    MoleculeDependentCellDivision,
    MoleculeDependentCellDeath,
    GenomeEditor,
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
        writer.add_scalar(tag, molecule_map[idx].mean(), step)

    if n_cells > 0:
        writer.add_scalar("Cells/Total", n_cells, step)
        mean_surv = exp.world.cell_survival.float().mean()
        mean_divis = exp.world.cell_divisions.float().mean()
        writer.add_scalar("Cells/Survival", mean_surv, step)
        writer.add_scalar("Cells/Divisions", mean_divis, step)
        writer.add_scalar("Cells/cPD", exp.cpd, step)
        writer.add_scalar("Cells/GrowthRate", exp.growth_rate, step)
        writer.add_scalar(
            "Cells/GenomeSize", sum(len(d) for d in exp.world.genomes) / n_cells, step
        )
        for scalar, idx in molecules.items():
            tag = f"{scalar}[int]"
            writer.add_scalar(tag, cell_molecules[:, idx].mean(), step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)
    writer.add_scalar("Other/Split", exp.split_i, step)
    writer.add_scalar("Other/Progress", exp.progress, step)
    writer.add_scalar("Other/MutationRate", exp.mutation_rate, step)


def _log_imgs(exp: Experiment, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


class MutationRateSteps(MutationRateFact):
    def __init__(self, progress_rate_pairs: list[tuple[float, float]]):
        self.progress_rate_pairs = sorted(progress_rate_pairs, reverse=True)

    def __call__(self, exp: Experiment) -> float:
        for progress, rate in self.progress_rate_pairs:
            if exp.progress >= progress:
                return rate
        return 0.0


class DynamicAdaption(MediumFact):
    def __init__(
        self,
        additives: list[ms.Molecule],
        substrates_a: list[ms.Molecule],
        substrates_b: list[ms.Molecule],
        n_init_splits: float,
        n_total_splits: float,
        min_gr: float,
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

        self.min_gr = min_gr

        self.n_valid_init_splits = n_init_splits
        self.n_valid_total_splits = n_total_splits
        self.valid_split_i = 0

    def __call__(self, exp: Experiment) -> torch.Tensor:
        if exp.growth_rate >= self.min_gr:
            self.valid_split_i += 1

        exp.progress = min(1.0, self.valid_split_i / self.n_valid_total_splits)

        if self.valid_split_i < self.n_valid_init_splits:
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

        print(f"Editing genome at step {exp.step_i}, split {exp.split_i}")  # TODO
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

    # load initial genomes
    n_cells = int(n_pxls * hparams["init_cell_cover"])
    if hparams["init_label"] == "random":
        x_trnsptr = ms.ProteinFact(ms.TransporterDomainFact(_X))
        init_genomes = [
            world.generate_genome(proteome=[x_trnsptr], size=hparams["gene_size"])
            for _ in range(n_cells)
        ]
    else:
        pop = load_genomes(label=hparams["init_label"], runsdir=runsdir)
        init_genomes = random.choices(pop, k=n_cells)

    n_init_splits = hparams["n_init_splits"]
    n_init_adapt_splits = n_init_splits + hparams["n_adapt_splits"]
    n_total_splits = n_init_adapt_splits + hparams["n_final_splits"]

    medium_fact = DynamicAdaption(
        substrates_a=subs_a,
        substrates_b=subs_b,
        additives=add,
        molmap=world.molecule_map,
        mol_2_idx=mol_2_idx,
        n_init_splits=n_init_splits,
        n_total_splits=n_total_splits,
        additives_init=hparams["additives_init"],
        substrates_init=hparams["substrates_init"],
        min_gr=hparams["min_gr"],
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
    genome_size_controller = GenomeSizeController(k=hparams["genome_kill_k"], n=7)

    exp = Experiment(
        world=world,
        init_genomes=init_genomes,
        lgt_rate=hparams["lgt_rate"],
        passager=passager,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
        division_by_x=division_by_x,
        death_by_e=death_by_e,
        genome_size_controller=genome_size_controller,
        genome_editor=genome_editor,
    )

    avg_genome_len = sum(len(d) for d in world.genomes) / world.n_cells
    print(f"In total {len(genes)} genes are added")
    print(f"   Average genome size is {avg_genome_len:,}")

    writer = init_writer(logdir=trial_dir, hparams=hparams)
    exp.world.save_state(statedir=trial_dir / "step=0")

    trial_t0 = time.time()
    watchmols = list(set(add + subs_a + subs_b))
    watch = [(d, mol_2_idx[d.name]) for d in watchmols]
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
