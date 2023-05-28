from pathlib import Path
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import WL_STAGES_MAP, _X, _E
from .util import init_writer
from .experiment import (
    Experiment,
    PassageByCells,
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


class ConstantRate(MutationRateFact):
    def __init__(self, rate: float):
        self.rate = rate

    def __call__(self, exp: Experiment) -> float:
        return self.rate


class DefinedMedium(MediumFact):
    def __init__(
        self,
        substrates: list[ms.Molecule],
        n_splits: float,
        substrates_init: float,
        molmap: torch.Tensor,
        mol_2_idx: dict[str, int],
    ):
        self.substrates_init = substrates_init
        self.subs_idxs = [mol_2_idx[d.name] for d in substrates]
        self.molmap = molmap
        self.n_splits = n_splits

    def __call__(self, exp: Experiment) -> torch.Tensor:
        exp.progress = min(1.0, exp.split_i / self.n_splits)
        t = torch.zeros_like(self.molmap)
        t[self.subs_idxs] = self.substrates_init
        return t


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

    # factories
    medium_fact = DefinedMedium(
        substrates=WL_STAGES_MAP["WL-0"][1],
        molmap=world.molecule_map,
        mol_2_idx=mol_2_idx,
        n_splits=hparams["n_splits"],
        substrates_init=hparams["substrates_init"],
    )

    mutation_rate_fact = ConstantRate(rate=hparams["mutation_rate"])

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

    # init experiment with fresh medium
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

    # add initial genomes
    xt = ms.ProteinFact(ms.TransporterDomainFact(_X))
    et = ms.ProteinFact(ms.TransporterDomainFact(_E))
    init_genomes = [
        world.generate_genome(proteome=[xt, et], size=hparams["genome_size"])
        for _ in range(int(n_pxls * hparams["init_cell_cover"]))
    ]
    exp.world.add_cells(genomes=init_genomes)

    avg_genome_len = sum(len(d) for d in world.genomes) / world.n_cells
    print(f"{exp.world.n_cells} cells were added")
    print(f"   Average genome size is {avg_genome_len:.0f}")

    # start logging
    writer = init_writer(logdir=trial_dir, hparams=hparams)
    exp.world.save_state(statedir=trial_dir / "step=0")

    trial_t0 = time.time()
    watch = [(d, mol_2_idx[d.name]) for d in [_X, _E]]
    print(f"Starting trial {run_name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    _log_scalars(exp=exp, writer=writer, step=0, dtime=0, mols=watch)
    _log_imgs(exp=exp, writer=writer, step=0)

    # start steps
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
