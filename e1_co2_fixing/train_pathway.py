from pathlib import Path
from typing import Callable
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import PATHWAY_PHASES_MAP
from .experiment import Experiment, Passage, MutationRateFact, MediumFact, GenomeFact
from .util import Finished, init_writer

THIS_DIR = Path(__file__).parent


class StepAdapt(MediumFact):
    """
    Change medium from complex to minimal depending on the phase of the training
    process. `substrates` will always be added with `substrates_init` to the medium,
    essential molecules as defined by `phases` always with `essentials_init`.
    """

    def __init__(
        self,
        phases: list[tuple[list[ms.ProteinFact], list[ms.Molecule]]],
        essentials_max: float,
        substrates_max: float,
        mol_2_idx: dict[str, int],
        molmap: torch.Tensor,
        substrates: tuple[str, ...] = ("CO2", "E"),
    ):
        self.molmap = molmap
        self.essentials_max = essentials_max
        self.substrates_max = substrates_max
        self.subs_idxs = [mol_2_idx[d] for d in substrates]

        ess_molnames: list[str] = []
        for prot_facts, rm_mols in phases:
            for prot in prot_facts:
                for dom in prot.domain_facts:
                    if isinstance(dom, ms.TransporterDomainFact):
                        ess_molnames.append(dom.molecule.name)
            ess_molnames.extend([d.name for d in rm_mols])

        init_molnames = list(set(ess_molnames) - set(substrates))
        phase_molnames: list[list[str]] = [init_molnames]
        for _, rm_mols in phases[1:]:
            prev_names = set(phase_molnames[-1])
            rm_names = set(d.name for d in rm_mols)
            phase_molnames.append(list(prev_names - rm_names))

        print(f"Medium will change from complex to minimal in {len(phases)} phases")
        print(f"  complex: {', '.join(phase_molnames[0])}")
        print(f"  minimal: {', '.join(phase_molnames[-1])}")
        print(f"  (disregarding {', '.join(substrates)})")
        self.essentials = phase_molnames[0]
        self.substrates = substrates

        self.phase_idxs = [[mol_2_idx[dd] for dd in d] for d in phase_molnames]

    def __call__(self, exp: Experiment) -> torch.Tensor:
        idxs = self.phase_idxs[exp.phase_i]
        t = torch.zeros_like(self.molmap)
        t[idxs] = self.essentials_max
        t[self.subs_idxs] = self.substrates_max
        return t

    def hparams(self) -> dict[str, str | float]:
        return {
            "essentials_max": self.essentials_max,
            "substrates_max": self.substrates_max,
        }


class GainPathway(GenomeFact):
    """
    Generate genomes for each phase according to `phases`.
    """

    def __init__(
        self,
        phases: list[tuple[list[ms.ProteinFact], list[ms.Molecule]]],
        size: int,
        genfun: Callable[[list[ms.ProteinFact], int], str],
    ):
        self.prot_facts_phases: list[list[ms.ProteinFact]] = [d[0] for d in phases]
        self.genfun = genfun
        self.size = size

        # test protein facts are all valid
        for prot_facts in self.prot_facts_phases:
            _ = genfun(prot_facts, size)

        n_genes = len([dd for d in self.prot_facts_phases for dd in d])
        print(f"In total {n_genes} genes will be added in {len(phases)} phases")
        print(f" Inital genome size is {size:,}, final will be {size * len(phases):,}")

    def __call__(self, phase: int) -> str:
        return self.genfun(self.prot_facts_phases[phase], self.size)

    def hparams(self) -> dict[str, str | float]:
        return {"genome_sizes": self.size}


class PassageByCellEnergy(Passage):
    def __init__(
        self,
        split_ratio: float,
        split_thresh_subs: float,
        split_thresh_cells: float,
        max_subs: float,
        max_cells: int,
    ):
        self.split_thresh_subs = split_thresh_subs
        self.split_thresh_cells = split_thresh_cells
        self.min_subs = max_subs * split_thresh_subs
        self.max_cells = int(max_cells * split_thresh_cells)
        self.split_ratio = split_ratio
        self.split_leftover = int(split_ratio * max_cells)

    def __call__(self, exp: "Experiment") -> bool:
        return any(
            [
                exp.world.molecule_map[exp.E_I].sum().item() <= self.min_subs,
                exp.world.molecule_map[exp.CO2_I].sum().item() <= self.min_subs,
                exp.world.n_cells >= self.max_cells,
            ]
        )

    def hparams(self) -> dict[str, str | float]:
        return {
            "cell_split": f"{self.split_ratio:.1f}-{self.split_thresh_cells:.1f}",
            "min_subs": self.split_thresh_subs,
        }


class LinearChange(MutationRateFact):
    """
    Linearly change from value `from_d` to value `to_d` over
    `n` generations starting from 0.
    """

    def __init__(self, n: float, from_d=1e-4, to_d=1e-6):
        self.n = n
        self.from_d = from_d
        self.to_d = to_d

    def __call__(self, exp: Experiment) -> float:
        v = exp.gen_i
        if v >= self.n:
            return self.to_d
        dn = (self.n - v) / self.n
        return dn * self.from_d + (1 - dn) * self.to_d

    def hparams(self) -> dict[str, str | float]:
        return {
            "mut_scheme": f"linear-{self.n}-gens",
            "mut_rates": f"{self.from_d:0e}-{self.to_d:0e}",
        }


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
        writer.add_scalar("Cells/total", n_cells, step)
        mean_surv = exp.world.cell_survival.float().mean()
        writer.add_scalar("Cells/Survival", mean_surv, step)
        writer.add_scalar("Cells/Generation", exp.gen_i, step)
        for scalar, idx in molecules.items():
            tag = f"{scalar}[int]"
            writer.add_scalar(tag, cell_molecules[:, idx].mean().item(), step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)
    writer.add_scalar("Other/Split", exp.split_i, step)
    writer.add_scalar("Other/Phase", exp.phase_i, step)
    writer.add_scalar("Other/Score", exp.score, step)
    writer.add_scalar("Other/MutationRate", exp.mutation_rate, step)


def _log_imgs(exp: Experiment, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


def run_trial(
    device: str,
    n_workers: int,
    name: str,
    n_steps: int,
    trial_max_time_s: int,
    hparams: dict,
):
    rundir = THIS_DIR / "runs"
    trial_dir = rundir / name
    world = ms.World.from_file(rundir=rundir, device=device, workers=n_workers)
    mol_2_idx = {d.name: i for i, d in enumerate(world.chemistry.molecules)}
    n_pxls = world.map_size**2

    pathway_phases = PATHWAY_PHASES_MAP[hparams["pathway"]]

    genome_fact = GainPathway(
        phases=pathway_phases,
        size=200,
        genfun=lambda p, s: world.generate_genome(p, size=s),
    )

    medium_fact = StepAdapt(
        phases=pathway_phases,
        essentials_max=20.0,
        substrates_max=100.0,
        mol_2_idx=mol_2_idx,
        molmap=world.molecule_map,
    )

    passage = PassageByCellEnergy(
        split_ratio=hparams["split_ratio"],
        split_thresh_subs=hparams["split_thresh_subs"],
        split_thresh_cells=hparams["split_thresh_cells"],
        max_subs=medium_fact.substrates_max * n_pxls,
        max_cells=n_pxls,
    )

    mutation_rate_fact = LinearChange(n=hparams["n_adapt_gens"])

    exp = Experiment(
        world=world,
        n_phases=len(pathway_phases),
        n_phase_gens=hparams["n_adapt_gens"] + hparams["n_static_gens"],
        init_cell_cover=hparams["init_cell_cover"],
        mol_divide_k=hparams["mol_divide_k"],
        mol_kill_k=hparams["mol_kill_k"],
        genome_kill_k=hparams["genome_kill_k"],
        lgt_rate=hparams["hparams"],
        passage=passage,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
        genome_fact=genome_fact,
    )

    writer = init_writer(
        logdir=trial_dir,
        hparams={
            **hparams,
            **passage.hparams(),
            **medium_fact.hparams(),
            **mutation_rate_fact.hparams(),
            **genome_fact.hparams(),
        },
    )
    exp.world.save_state(statedir=trial_dir / "step=0")

    trial_t0 = time.time()
    watch_substrates = [(d, mol_2_idx[d]) for d in medium_fact.substrates]
    watch_essentials = [(d, mol_2_idx[d]) for d in medium_fact.essentials]
    watch = watch_substrates + watch_essentials
    print(f"Starting trial {name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    _log_scalars(exp=exp, writer=writer, step=0, dtime=0, mols=watch)
    _log_imgs(exp=exp, writer=writer, step=0)

    min_cells = int(exp.world.map_size**2 * 0.01)
    for step_i in range(1, n_steps + 1):
        step_t0 = time.time()

        try:
            exp.step_1s()
        except Finished:
            print(f"target phase {exp.n_phases} reached after {step_i} steps")
            break

        if step_i % 5 == 0:
            dtime = time.time() - step_t0
            _log_scalars(exp=exp, writer=writer, step=step_i, dtime=dtime, mols=watch)

        if step_i % 50 == 0:
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            _log_imgs(exp=exp, writer=writer, step=step_i)

        if exp.world.n_cells < min_cells:
            print(f"after {step_i} stepsless than {min_cells} cells left")
            break

        if (time.time() - trial_t0) > trial_max_time_s:
            print(f"{trial_max_time_s} hours have passed")
            break

    print(f"Finishing trial {name}")
    exp.world.save_state(statedir=trial_dir / f"step={step_i}")
    writer.close()
