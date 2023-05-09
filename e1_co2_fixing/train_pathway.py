from pathlib import Path
from typing import Callable
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from .chemistry import PATHWAY_PHASES_MAP
from .experiment import (
    Experiment,
    Passage,
    MutationRateFact,
    MediumFact,
    GenomeFact,
    CellSampler,
)
from .util import Finished, init_writer, batch_add_cells, sigm_sample, rev_sigm_sample

THIS_DIR = Path(__file__).parent


class MoleculeDependentCellDeath(CellSampler):
    """
    Sample cell indexes based on molecule abundance.
    Lower abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, mol_i: int, k: float, n: int):
        self.k = k
        self.n = n
        self.mol_i = mol_i

    def __call__(self, exp: "Experiment") -> list[int]:
        mols = exp.world.cell_molecules[:, self.mol_i]
        return rev_sigm_sample(mols, self.k, self.n)


class MoleculeDependentCellDivision(CellSampler):
    """
    Sample cell indexes based on molecule abundance.
    Higher abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, mol_i: int, k: float, n: int):
        self.k = k
        self.n = n
        self.mol_i = mol_i

    def __call__(self, exp: "Experiment") -> list[int]:
        mols = exp.world.cell_molecules[:, self.mol_i]
        return sigm_sample(mols, self.k, self.n)


class GenomeSizeController(CellSampler):
    """
    Sample cell indexes based on genome size.
    Larger genomes lead to higher probability of being sampled.
    `k` and `n` define the final parameters that are used during the
    last phase. But in earlier phases the genomes are much smaller
    so `k` is reduced accordingly during these phases.
    Otherwise cells grow huge genomes in early phases,
    then fail in later phases when they reach `k`.
    """

    def __init__(
        self,
        k: float,
        n: int,
        genome_size: int,
        n_phases: int,
    ):
        init_size = genome_size / n_phases
        self.ks = [(i + 1) * init_size / genome_size * k for i in range(n_phases)]
        self.n = n

    def __call__(self, exp: "Experiment") -> list[int]:
        k = self.ks[exp.phase_i]
        genome_lens = [len(d) for d in exp.world.genomes]
        sizes = torch.tensor(genome_lens)
        return sigm_sample(sizes, k, self.n)


class LinearMediumAdaption(MediumFact):
    """
    Change medium linearly from ``essentials_max` to zero over `n_gens`
    from complex to minimal for each phase.
    Which molecules are essential in each phase is derived from `phases`.
    `substrates` will always be added with `substrates_max` to the medium.
    """

    def __init__(
        self,
        phases: list[tuple[list[ms.ProteinFact], list[ms.Molecule]]],
        n_gens: int,
        essentials_max: float,
        substrates_max: float,
        mol_2_idx: dict[str, int],
        molmap: torch.Tensor,
        substrates: tuple[str, ...] = ("CO2", "E"),
    ):
        self.molmap = molmap
        self.n_gens = n_gens
        self.essentials_max = essentials_max
        self.substrates_max = substrates_max
        self.subs_idxs = [mol_2_idx[d] for d in substrates]

        essentials: list[str] = []
        for prot_facts, rm_mols in phases:
            for prot in prot_facts:
                for dom in prot.domain_facts:
                    if isinstance(dom, ms.TransporterDomainFact):
                        essentials.append(dom.molecule.name)
            essentials.extend([d.name for d in rm_mols])

        self.essentials = list(set(essentials) - set(substrates))
        self.substrates = list(substrates)

        rng_essentials = set(self.essentials)
        phase_essentials: list[list[str]] = []
        phase_removals: list[list[str]] = []
        for _, rm_mols in phases:
            rm_names = set(d.name for d in rm_mols)
            rng_essentials = rng_essentials - rm_names
            phase_essentials.append(list(rng_essentials))
            phase_removals.append(list(rm_names))

        print(f"Medium will change from complex to minimal in {len(phases)} phases")
        print(f"  complex: {', '.join(self.essentials)}")
        for phase_i, essentials in enumerate(phase_essentials):
            print(f"  phase {phase_i}: {', '.join(essentials)}")
        print(f"  (disregarding {', '.join(self.substrates)})")

        self.phase_essentials = [[mol_2_idx[dd] for dd in d] for d in phase_essentials]
        self.phase_removals = [[mol_2_idx[dd] for dd in d] for d in phase_removals]

    def __call__(self, exp: Experiment) -> torch.Tensor:
        i = exp.gen_i
        n = self.n_gens
        ess_idxs = self.phase_essentials[exp.phase_i]
        rm_idxs = self.phase_removals[exp.phase_i]
        t = torch.zeros_like(self.molmap)
        t[rm_idxs] = self.essentials_max * max((n - i) / n, 0.0)
        t[ess_idxs] = self.essentials_max
        t[self.subs_idxs] = self.substrates_max
        return t


class GainPathway(GenomeFact):
    """
    Generate genes for each phase according to `phases`.
    The generator for the first phase is used as initial genomes.
    """

    def __init__(
        self,
        phases: list[tuple[list[ms.ProteinFact], list[ms.Molecule]]],
        genome_size: int,
        genfun: Callable[[list[ms.ProteinFact], int], str],
    ):
        self.prot_facts_phases: list[list[ms.ProteinFact]] = [d[0] for d in phases]
        self.genfun = genfun
        self.size = int(genome_size / len(phases))

        # test protein facts are all valid
        for prot_facts in self.prot_facts_phases:
            _ = genfun(prot_facts, self.size)

        n_genes = len([dd for d in self.prot_facts_phases for dd in d])
        n_phases = len(phases)
        print(f"In total {n_genes} genes will be added in {n_phases} phases")
        print(f"   Inital genome size is {self.size:,}, final will be {genome_size:,}")

    def __call__(self, exp: "Experiment") -> str:
        return self.genfun(self.prot_facts_phases[exp.phase_i], self.size)


class PassageByCellAndSubstrates(Passage):
    """
    Passage cells if world too full or if substrates run low.
    """

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


class StepWiseRateAdaption(MutationRateFact):
    """
    Switch rate from from one probability to another after n generations
    """

    def __init__(self, n: float, from_p: float, to_p: float):
        self.n = n
        self.from_p = from_p
        self.to_p = to_p

    def __call__(self, exp: Experiment) -> float:
        if exp.gen_i >= self.n:
            return self.to_p
        return self.from_p


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
        writer.add_scalar("Cells/GrowhRate", exp.growth_rate, step)
        writer.add_scalar(
            "Cells/GenomeSize", sum(len(d) for d in exp.world.genomes) / n_cells, step
        )
        writer.add_scalar("Cells/AvgProteins", exp.world.kinetics.Km.size(1))
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
        genome_size=hparams["genome_size"],
        genfun=lambda p, s: world.generate_genome(p, size=s),
    )

    medium_fact = LinearMediumAdaption(
        phases=pathway_phases,
        n_gens=hparams["n_adapt_gens"],
        essentials_max=20.0,
        substrates_max=100.0,
        mol_2_idx=mol_2_idx,
        molmap=world.molecule_map,
    )

    passage = PassageByCellAndSubstrates(
        split_ratio=hparams["split_ratio"],
        split_thresh_subs=hparams["split_thresh_subs"],
        split_thresh_cells=hparams["split_thresh_cells"],
        max_subs=medium_fact.substrates_max * n_pxls,
        max_cells=n_pxls,
    )

    mutation_rate_fact = StepWiseRateAdaption(
        n=hparams["n_adapt_gens"],
        from_p=1e-4,
        to_p=1e-6,
    )

    division_by_x_fact = MoleculeDependentCellDivision(
        mol_i=mol_2_idx["X"], k=hparams["mol_divide_k"], n=3
    )
    death_by_e_fact = MoleculeDependentCellDeath(
        mol_i=mol_2_idx["E"], k=hparams["mol_kill_k"], n=1
    )
    death_by_genome_fact = GenomeSizeController(
        k=hparams["genome_kill_k"],
        n=7,
        genome_size=hparams["genome_size"],
        n_phases=len(pathway_phases),
    )

    exp = Experiment(
        world=world,
        n_phases=len(pathway_phases),
        n_phase_gens=hparams["n_adapt_gens"] + hparams["n_static_gens"],
        lgt_rate=hparams["lgt_rate"],
        passage=passage,
        medium_fact=medium_fact,
        mutation_rate_fact=mutation_rate_fact,
        genome_fact=genome_fact,
        division_by_x_fact=division_by_x_fact,
        death_by_e_fact=death_by_e_fact,
        death_by_genome_fact=death_by_genome_fact,
    )

    # initial cells
    n_cells = int(n_pxls * hparams["init_cell_cover"])
    init_genomes = [genome_fact(exp) for _ in range(n_cells)]
    batch_add_cells(world=exp.world, genomes=init_genomes)

    writer = init_writer(
        logdir=trial_dir,
        hparams={
            **hparams,
            "mut_rates": f"{mutation_rate_fact.from_p:.0e}-{mutation_rate_fact.to_p:.0e}",
            "cell_split": f"{passage.split_ratio:.1f}-{passage.split_thresh_cells:.1f}",
            "substrates_min": passage.split_thresh_subs,
            "substrates_max": medium_fact.substrates_max,
            "essentials_max": medium_fact.essentials_max,
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
    for step_i in range(n_steps):
        step_t0 = time.time()

        try:
            exp.step_1s()
        except Finished:
            print(f"target phase {exp.n_phases} finished after {step_i} steps")
            break
        except RuntimeError:
            print("OOM")
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

    exp.world.save_state(statedir=trial_dir / f"step={step_i}")
    print(f"Finishing trial {name}")
    writer.close()
