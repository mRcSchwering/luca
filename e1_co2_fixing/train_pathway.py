from pathlib import Path
import time
import math
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import SUBSTRATE_MOLS, TRAIN_WL_GENE_MAP, TRAIN_WL_MOL_MAP
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


class ProgressDependentMutationRate(MutationRateFact):
    """
    Switch rate from from one probability to another after progress reaches threshold
    """

    def __init__(self, thresh: float, from_p: float, to_p: float):
        self.thresh = thresh
        self.from_p = from_p
        self.to_p = to_p

    def __call__(self, exp: Experiment) -> float:
        if exp.progress >= self.thresh:
            return self.to_p
        return self.from_p


class GeneraionDependentLinearAdaption(MediumFact):
    """
    Medium factory for medium with essential molecules
    at `essentials_max` and substrates at `substrates_max`
    concentrations.
    Concentrations of molecules that are to be removed are linearly
    decreased from `essentials_max` to 0 over the course of `n_adapt_gens`.
    After these molecules were reduced to 0, cells should grow a final
    `n_fix_gens` before progress is set to 100%.
    """

    def __init__(
        self,
        rms: list[str],
        essentials: list[str],
        substrates: list[str],
        substrates_max: float,
        essentials_max: float,
        molmap: torch.Tensor,
        mol_2_idx: dict[str, int],
        n_adapt_gens: float,
        n_final_gens: float,
    ):
        self.rms = rms
        self.substrates = substrates
        self.essentials = essentials
        self.substrates_max = substrates_max
        self.essentials_max = essentials_max
        self.rm_idxs = [mol_2_idx[d] for d in rms]
        self.essential_idxs = [mol_2_idx[d] for d in essentials]
        self.substrate_idxs = [mol_2_idx[d] for d in substrates]

        self.n_adapt_gens = n_adapt_gens
        self.n_total_gens = n_final_gens + n_adapt_gens
        self.molmap = molmap

    def __call__(self, exp: Experiment) -> torch.Tensor:
        n = self.n_adapt_gens
        i = exp.gen_i
        decay = max((n - i) / n, 0.0)
        exp.progress = min(1.0, i / self.n_total_gens)

        t = torch.zeros_like(self.molmap)
        t[self.essential_idxs] = self.essentials_max
        t[self.substrate_idxs] = self.substrates_max
        t[self.rm_idxs] = self.essentials_max * decay
        return t


class GrowthDependentExponentialAdaption(MediumFact):
    """
    Medium factory for medium with essential molecules
    at `essentials_max` and substrates at `substrates_max`
    concentrations.
    Concentrations of molecules that are to be removed are exponentially
    decreased from `essentials_max` to 0 over the course of `n_steps`.
    with a decay rate of `decay_lambda` (i.e. lambda=ln(2) for n(t)=(1/2)^t).
    In order to advance 1 step, the previous passage's growth rate must exceed
    `min_gr` and the cells must have grown at least `n_min_gens` generations .
    Progress is increased linearly over `n_steps + 1` steps to 100%.
    For the last step, removed molecules are at exactly 0.
    """

    def __init__(
        self,
        rms: list[str],
        essentials: list[str],
        substrates: list[str],
        substrates_max: float,
        essentials_max: float,
        molmap: torch.Tensor,
        mol_2_idx: dict[str, int],
        n_min_gens: float,
        min_gr: float,
        decay_lambda: float,
        n_steps: int,
    ):
        self.rms = rms
        self.substrates = substrates
        self.essentials = essentials
        self.substrates_max = substrates_max
        self.essentials_max = essentials_max
        self.rm_idxs = [mol_2_idx[d] for d in rms]
        self.essential_idxs = [mol_2_idx[d] for d in essentials]
        self.substrate_idxs = [mol_2_idx[d] for d in substrates]

        self.n_min_gens = n_min_gens
        self.min_gr = min_gr
        self.molmap = molmap

        self.gen_thresh = self.n_min_gens
        self.step_i = 0
        self.n_steps = n_steps
        self.decays = [math.exp(-decay_lambda * i) for i in range(n_steps)]

        # n_steps + 1 adjusting [1;0], final 10% for static phase
        self.incr_progress = 1 / (n_steps + 1)

    def __call__(self, exp: Experiment) -> torch.Tensor:
        if exp.gen_i > self.gen_thresh and exp.growth_rate < self.min_gr:
            self.step_i += 1
            exp.progress = min(1.0, exp.progress + self.incr_progress)
            self.gen_thresh = exp.gen_i + self.n_min_gens

        decay = self.decays[self.step_i] if self.step_i < self.n_steps else 0.0
        t = torch.zeros_like(self.molmap)
        t[self.essential_idxs] = self.essentials_max
        t[self.substrate_idxs] = self.substrates_max
        t[self.rm_idxs] = self.essentials_max * decay
        return t


def run_trial(
    device: str,
    n_workers: int,
    run_name: str,
    n_steps: int,
    trial_max_time_s: int,
    hparams: dict,
):
    train_label = hparams["train_label"]
    runsdir = THIS_DIR / "runs"
    trial_dir = runsdir / run_name
    world = ms.World.from_file(rundir=runsdir, device=device, workers=n_workers)
    mol_2_idx = {d.name: i for i, d in enumerate(world.chemistry.molecules)}
    n_pxls = world.map_size**2

    essentials, rms = TRAIN_WL_MOL_MAP[train_label]
    substrates = sorted(d.name for d in SUBSTRATE_MOLS)
    print("Medium will change from complex to minimal")
    print(f"  complex: {', '.join(essentials)}")
    print(f"  minimal {', '.join(sorted(list(set(essentials) - set(rms))))}")
    print(f"  (disregarding substrates {', '.join(substrates)})")

    medium_fact = GeneraionDependentLinearAdaption(
        rms=rms,
        essentials=essentials,
        substrates=substrates,
        essentials_max=20.0,
        substrates_max=100.0,
        molmap=world.molecule_map,
        mol_2_idx=mol_2_idx,
        n_adapt_gens=hparams["n_adapt_gens"],
        n_final_gens=hparams["n_final_gens"],
    )

    mutation_rate_fact = ProgressDependentMutationRate(
        thresh=hparams["n_adapt_gens"]
        / (hparams["n_adapt_gens"] + hparams["n_final_gens"]),
        from_p=1e-4,
        to_p=1e-6,
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
        pop = load_genomes(world=world, label=init_label, runsdir=runsdir)
        genomes = random.choices(pop, k=n_cells)

    # will fail if gene size too small
    prots = TRAIN_WL_GENE_MAP[train_label]
    size = hparams["gene_size"]
    exp.init_cells(
        genomes=[d + world.generate_genome(proteome=prots, size=size) for d in genomes]
    )

    avg_genome_len = sum(len(d) for d in world.genomes) / world.n_cells
    print(f"In total {len(prots)} genes were added")
    print(f"   Average genome size is {avg_genome_len:,}")

    writer = init_writer(
        logdir=trial_dir,
        hparams={
            **hparams,
            "mut_rates": f"{mutation_rate_fact.from_p:.0e}-{mutation_rate_fact.to_p:.0e}",
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

        if exp.progress >= 1.0:
            print(f"target reached after {step_i + 1} steps")
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            break

        if step_i % 5 == 0:
            dtime = time.time() - step_t0
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


# TODO: vllt lieber Welt neu initialisieren nachdem alte Zellen geladen wurden?
# TODO: load state bruacht auch ein batch argument, weil da ja ein update cells kommt
