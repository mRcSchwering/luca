from pathlib import Path
from typing import Callable
import datetime as dt
import random
import math
import time
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
import magicsoup as ms
from .chemistry import PATHWAY_PHASES_MAP
from .util import sigm_sample, rev_sigm_sample

THIS_DIR = Path(__file__).parent


def _batch_add_cells(world: ms.World, genomes: list[str], d=1000):
    # to avoid OOM
    for a in range(0, len(genomes), d):
        b = a + d
        world.add_cells(genomes=genomes[a:b])


def _batch_update_cells(
    world: ms.World, genome_idx_pairs: list[tuple[str, int]], d=1000
):
    # to avoid OOM
    for a in range(0, len(genome_idx_pairs), d):
        b = a + d
        world.update_cells(genome_idx_pairs=genome_idx_pairs[a:b])


class Finished(Exception):
    """Raise to finish experiment"""


class MoleculeDependentCellDeath:
    """
    Sample cell indexes based on molecule abundance.
    Lower abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n=1):
        self.k = k
        self.n = n

    def __call__(self, cellmols: torch.Tensor) -> list[int]:
        return rev_sigm_sample(cellmols, self.k, self.n)


class GenomeSizeDependentCellDeath:
    """
    Sample cell indexes based on genome size.
    Larger genomes lead to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n=7):
        self.k = k
        self.n = n

    def __call__(self, genomes: list[str]) -> list[int]:
        sizes = torch.tensor([float(len(d)) for d in genomes])
        return sigm_sample(sizes, self.k, self.n)


class MoleculeDependentCellDivision:
    """
    Sample cell indexes based on molecule abundance.
    Higher abundance leads to higher probability of being sampled.
    `k` defines sensitivity, `n` cooperativity (M.M.: `n=1`, sigmoid: `n>1`).
    """

    def __init__(self, k: float, n=3):
        self.k = k
        self.n = n

    def __call__(self, cellmols: torch.Tensor) -> list[int]:
        return sigm_sample(cellmols, self.k, self.n)


class LinearChange:
    """
    Linearly change from value `from_d` to value `to_d` over a length of `n`.
    """

    def __init__(self, n: float, from_d=1e-4, to_d=1e-6):
        self.n = n
        self.from_d = from_d
        self.to_d = to_d

    def __call__(self, v: float) -> float:
        if v >= self.n:
            return self.to_d
        dn = (self.n - v) / self.n
        return dn * self.from_d + (1 - dn) * self.to_d


class StepChange:
    """
    Change from value `from_d` to value `to_d` after `n`.
    """

    def __init__(self, n: float, from_d=1e-4, to_d=1e-6):
        self.n = n
        self.from_d = from_d
        self.to_d = to_d

    def __call__(self, v: float) -> float:
        if v <= self.n:
            return self.from_d
        return self.to_d


MUT_RATE_FACTS: dict[str, Callable[[float, float], LinearChange | StepChange]] = {
    "linear": lambda adapt, static: LinearChange(n=adapt),
    "step": lambda adapt, static: StepChange(n=adapt + static),
    "none": lambda adapt, static: StepChange(n=0),
}


class GenomeFact:
    """
    Generate genomes for each phase according to `phases`.
    """

    def __init__(
        self,
        phases: list[tuple[list[ms.ProteinFact], list[ms.Molecule]]],
        genfun: Callable[[list[ms.ProteinFact]], str],
    ):
        self.prot_facts_phases: list[list[ms.ProteinFact]] = [d[0] for d in phases]
        self.genfun = genfun

        # test protein facts are all valid
        for prot_facts in self.prot_facts_phases:
            _ = genfun(prot_facts)

        n_genes = len([dd for d in self.prot_facts_phases for dd in d])
        print(f"In total {n_genes} will be added in {len(phases)} phases")

    def __call__(self, phase: int) -> str:
        return self.genfun(self.prot_facts_phases[phase])


class MediumFact:
    """
    Change medium from complex to minimal depending on the phase of the training process.
    `substrates` will always be added with `substrates_init` to the medium,
    essential molecules as defined by `phases` always with `essentials_init`.
    """

    def __init__(
        self,
        phases: list[tuple[list[ms.ProteinFact], list[ms.Molecule]]],
        essentials_init: float,
        substrates_init: float,
        mol_2_idx: dict[str, int],
        molmap: torch.Tensor,
        substrates: tuple[str, ...] = ("CO2", "E"),
    ):
        self.essentials_init = essentials_init
        self.substrates_init = substrates_init
        self.molmap = molmap
        self.subs_idxs = [mol_2_idx[d] for d in substrates]

        ess_mols: list[str] = []
        for prot_facts, rm_mols in phases:
            for prot in prot_facts:
                for dom in prot.domain_facts:
                    if isinstance(dom, ms.TransporterDomainFact):
                        ess_mols.append(dom.molecule.name)
            ess_mols.extend([d.name for d in rm_mols])

        init_mols = list(set(ess_mols) - set(substrates))
        phase_mols: list[list[str]] = [init_mols]
        for _, rm_mols in phases:
            mols = list(set(phase_mols[-1]) - set(rm_mols))
            phase_mols.append(mols.copy())

        print(f"Medium will change from complex to minimal in {len(phases)} phases")
        print(f"  complex: {', '.join(phase_mols[0])}")
        print(f"  minimal: {', '.join(phase_mols[-1])}")
        print(f"  (disregarding {', '.join(substrates)})")

        self.phase_idxs = [[mol_2_idx[dd] for dd in d] for d in phase_mols]

    def __call__(self, phase_i: int) -> torch.Tensor:
        idxs = self.phase_idxs[phase_i]
        t = torch.zeros_like(self.molmap)
        t[idxs] = self.essentials_init
        t[self.subs_idxs] = self.substrates_init
        return t


class _Experiment:
    """State and methods of the experiment"""

    def __init__(
        self,
        world: ms.World,
        pathway: str,
        n_adapt_gens: float,
        n_static_gens: float,
        mut_scheme: str,
        split_ratio: float,
        split_thresh_mols: float,
        split_thresh_cells: float,
        init_cell_cover: float,
    ):
        self.world = world
        n_pxls = world.map_size**2

        pathway_phases = PATHWAY_PHASES_MAP[pathway]
        self.n_phases = len(pathway_phases)
        self.n_gens_per_phase = n_adapt_gens + n_static_gens
        self.total_gens = self.n_phases * self.n_gens_per_phase
        self.score = 0.0
        self.phase_i = 0
        self.split_i = 0
        self.gen_i = 0.0

        molecules = self.world.chemistry.molecules
        self.mol_2_idx = {d.name: i for i, d in enumerate(molecules)}
        self.CO2_I = self.mol_2_idx["CO2"]
        self.X_I = self.mol_2_idx["X"]
        self.E_I = self.mol_2_idx["E"]

        mut_rate_fact = MUT_RATE_FACTS[mut_scheme]
        self.mutation_rate_fact = mut_rate_fact(n_adapt_gens, n_static_gens)
        self.mutation_rate = self.mutation_rate_fact(self.gen_i)

        self.replicate_by_mol = MoleculeDependentCellDivision(k=20.0)  # [15;30]
        self.kill_by_mol = MoleculeDependentCellDeath(k=0.04)  # [0.01;0.04]
        self.kill_by_genome = GenomeSizeDependentCellDeath(k=2_000.0)  # [2000;2500]

        self.medium_fact = MediumFact(
            phases=pathway_phases,
            essentials_init=10.0,
            substrates_init=100.0,
            mol_2_idx=self.mol_2_idx,
            molmap=self.world.molecule_map,
        )

        self.genome_fact = GenomeFact(
            phases=pathway_phases,
            genfun=lambda d: self.world.generate_genome(d, size=100),
        )

        self.mol_thresh = self.medium_fact.substrates_init * n_pxls * split_thresh_mols
        self.cell_thresh = int(n_pxls * split_thresh_cells)
        self.split_leftover = int(split_ratio * n_pxls)

        self._prepare_fresh_plate()

        n_cells = int(init_cell_cover * n_pxls)
        init_genomes = [self.genome_fact(0) for _ in range(n_cells)]
        _batch_add_cells(world=self.world, genomes=init_genomes)

    def step_1s(self):
        self.world.diffuse_molecules()
        self.world.degrade_molecules()

        for _ in range(10):
            self.world.enzymatic_activity()

        self._kill_cells()
        self.world.increment_cell_survival()
        self._replicate_cells()
        self._passage_cells()
        self._mutate_cells()

        avg = self.world.cell_divisions.float().mean().item()
        self.gen_i = 0.0 if math.isnan(avg) else avg
        self.mutation_rate = self.mutation_rate_fact(self.gen_i)

        self.score = min(
            max(self.gen_i * (self.phase_i + 1) / self.total_gens, 0.0), 1.0
        )

    def _next_phase(self) -> bool:
        if self.gen_i >= self.n_gens_per_phase:
            self.gen_i = 0.0
            self.world.cell_divisions[:] = 0
            self.phase_i += 1
            if self.phase_i > self.n_phases:
                raise Finished()
            return True
        return False

    def _passage_cells(self):
        n_cells = self.world.n_cells
        if any(
            [
                self.world.molecule_map[self.E_I].sum().item() <= self.mol_thresh,
                self.world.molecule_map[self.CO2_I].sum().item() <= self.mol_thresh,
                n_cells >= self.cell_thresh,
            ]
        ):
            kill_n = max(n_cells - self.split_leftover, 0)
            idxs = random.sample(range(n_cells), k=kill_n)
            self.world.kill_cells(cell_idxs=list(set(idxs)))
            n_cells = self.world.n_cells
            if self._next_phase():
                genome_idx_pairs: list[tuple[str, int]] = []
                for idx, old_genome in enumerate(self.world.genomes):
                    genes = self.genome_fact(self.phase_i)
                    genome_idx_pairs.append((old_genome + genes, idx))
                _batch_update_cells(world=self.world, genome_idx_pairs=genome_idx_pairs)
            self._prepare_fresh_plate()
            self.world.reposition_cells(cell_idxs=list(range(n_cells)))
            self.split_i += 1

    def _mutate_cells(self):
        mutated = ms.point_mutations(seqs=self.world.genomes, p=self.mutation_rate)
        _batch_update_cells(world=self.world, genome_idx_pairs=mutated)
        # TODO: add HGT

    def _replicate_cells(self):
        i = self.X_I
        idxs0 = self.replicate_by_mol(self.world.cell_molecules[:, i])
        idxs1 = torch.argwhere(self.world.cell_survival >= 10).flatten().tolist()
        idxs2 = torch.argwhere(self.world.cell_molecules[:, i] > 4.1).flatten().tolist()
        idxs = list(set(idxs0) & set(idxs1) & set(idxs2))

        successes = self.world.divide_cells(cell_idxs=idxs)
        if len(successes) == 0:
            return

        ps, cs = list(zip(*successes))
        self.world.cell_molecules[ps, i] -= 2.0
        self.world.cell_molecules[cs, i] -= 2.0

        pairs = [(self.world.genomes[p], self.world.genomes[c]) for p, c in successes]
        mutated = ms.recombinations(seq_pairs=pairs, p=self.mutation_rate)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = successes[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))
        _batch_update_cells(world=self.world, genome_idx_pairs=mutated)

    def _kill_cells(self):
        idxs0 = self.kill_by_mol(self.world.cell_molecules[:, self.E_I])
        idxs1 = self.kill_by_genome(self.world.genomes)
        idxs2 = torch.argwhere(self.world.cell_survival <= 3).flatten().tolist()
        self.world.kill_cells(cell_idxs=list(set(idxs0 + idxs1) - set(idxs2)))

    def _prepare_fresh_plate(self):
        self.world.molecule_map = self.medium_fact(self.phase_i)
        self.world.diffuse_molecules()


def _init_writer(logdir: Path, hparams: dict) -> SummaryWriter:
    writer = SummaryWriter(log_dir=logdir)
    exp, ssi, sei = get_summary(hparam_dict=hparams, metric_dict={"Other/Score": 0.0})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    with open(logdir / "hparams.json", "w", encoding="utf-8") as fh:
        json.dump(hparams, fh)

    return writer


def _log_scalars(
    exp: _Experiment,
    writer: SummaryWriter,
    step: int,
    dtime: float,
):
    mol_name_idx_list = [
        ("CO2", exp.CO2_I),
        ("X", exp.X_I),
        ("E", exp.E_I),
    ]

    n_cells = exp.world.n_cells
    molecule_map = exp.world.molecule_map
    cell_molecules = exp.world.cell_molecules
    molecules = {f"Molecules/{s}": i for s, i in mol_name_idx_list}

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
    writer.add_scalar("Other/Score", exp.score, step)


def _log_imgs(exp: _Experiment, writer: SummaryWriter, step: int):
    writer.add_image("Maps/Cells", exp.world.cell_map, step, dataformats="WH")


def _run_trial(
    device: str,
    n_workers: int,
    name: str,
    pathway: str,
    n_steps: int,
    trial_max_time_s: int,
    hparams: dict,
):
    rundir = THIS_DIR / "runs"
    world = ms.World.from_file(rundir=rundir, device=device, workers=n_workers)

    trial_dir = rundir / name
    writer = _init_writer(logdir=trial_dir, hparams=hparams)

    exp = _Experiment(
        world=world,
        pathway=pathway,
        n_adapt_gens=hparams["n_adapt_gens"],
        n_static_gens=hparams["n_static_gens"],
        mut_scheme=hparams["mut_scheme"],
        split_ratio=hparams["split_ratio"],
        split_thresh_mols=hparams["split_thresh_mols"],
        split_thresh_cells=hparams["split_thresh_cells"],
        init_cell_cover=hparams["init_cell_cover"],
    )

    assert exp.world.device == device
    assert exp.world.workers == n_workers
    exp.world.save_state(statedir=trial_dir / "step=0")

    print(f"Starting trial {name}")
    print(f"on {exp.world.device} with {exp.world.workers} workers")
    trial_t0 = time.time()

    _log_scalars(exp=exp, writer=writer, step=0, dtime=0)
    _log_imgs(exp=exp, writer=writer, step=0)

    for step_i in range(1, n_steps + 1):
        step_t0 = time.time()

        try:
            exp.step_1s()
        except Finished:
            print(f"target phase {exp.n_phases} reached after {step_i} steps")
            break

        if step_i % 5 == 0:
            dtime = time.time() - step_t0
            _log_scalars(exp=exp, writer=writer, step=step_i, dtime=dtime)

        if step_i % 50 == 0:
            exp.world.save_state(statedir=trial_dir / f"step={step_i}")
            _log_imgs(exp=exp, writer=writer, step=step_i)

        if exp.world.n_cells < 500:
            print(f"after {step_i} stepsless than 500 cells left")
            break

        if (time.time() - trial_t0) > trial_max_time_s:
            print(f"{trial_max_time_s} hours have passed")
            break

    print(f"Finishing trial {name}")
    exp.world.save_state(statedir=trial_dir / f"step={step_i}")
    writer.close()


def run_trials(kwargs: dict):
    kwargs.pop("func")
    pathway = kwargs.pop("pathway")
    device = kwargs.pop("device")
    n_workers = kwargs.pop("n_workers")
    n_trials = kwargs.pop("n_trials")
    n_steps = kwargs.pop("n_steps")
    trial_max_time_s = kwargs.pop("trial_max_time_h") * 60 * 60
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for trial_i in range(n_trials):
        _run_trial(
            pathway=pathway,
            device=device,
            n_workers=n_workers,
            name=f"{pathway}_{ts}_{trial_i}",
            n_steps=n_steps,
            trial_max_time_s=trial_max_time_s,
            hparams=kwargs,
        )
