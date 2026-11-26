from collections.abc import Callable
from typing import TypedDict

import magicsoup as ms
import torch

from src.chemistry import cofactors, substrates
from src.util.mixins import StateMixin

_energy_mins = {d: 0.1 for d in ("ATP", "NADPH", "QH2", "Fe-red", "FADH2", "Fd-red")}


class EnergyKiller(StateMixin):
    """Kill cells if energy sources fall below minimum"""

    def __init__(self, chem: ms.Chemistry):
        m2i = chem.molname_2_idx
        self.i2v = {m2i[m]: v for m, v in _energy_mins.items()}

    def __call__(self, exp: "Experiment") -> None:
        masks = [exp.world.cells.molecules[:, i] < v for i, v in self.i2v.items()]
        mask = torch.stack(masks).all(dim=0)
        exp.world.kill_cells(mask=mask)


class EnergyTaxer(StateMixin):
    """Tax cells ATP per protein"""

    def __init__(self, chem: ms.Chemistry) -> None:
        m2i = chem.molname_2_idx
        self.i_atp = m2i["ATP"]
        self.per_prot_cost = 0.01  # mM ATP per protein per 10s (step)

    def __call__(self, exp: "Experiment") -> None:
        prot_bill = exp.world.cells.proteins * self.per_prot_cost
        exp.world.cells.molecules[:, self.i_atp] -= prot_bill
        exp.world.cells.molecules[:, self.i_atp].clamp_(0)


class XReplicator(StateMixin):
    """Replicate cells based on X"""

    def __init__(self, chem: ms.Chemistry) -> None:
        m2i = chem.molname_2_idx
        self.i_x = m2i["X"]
        self.i2v = {m2i[m]: v for m, v in _energy_mins.items()}
        self.x_cost = 5  # division X cost in mM

    def __call__(self, exp: "Experiment") -> torch.Tensor:
        cell_molecules = exp.world.cells.molecules

        masks = [cell_molecules[:, i] >= d for i, d in self.i2v.items()]
        energy_mask = torch.stack(masks).all(dim=0)

        x_mask = cell_molecules[:, self.i_x] > self.x_cost

        mask = energy_mask & x_mask
        divided = exp.world.divide_cells(mask=mask)

        cell_molecules[divided, self.i_x] -= self.x_cost / 2
        cell_molecules[divided, self.i_x].clamp_(0)

        return divided


class MutatorKwargs(TypedDict, total=False):
    progress_range: tuple[float, float]
    factor: float


class Mutator(StateMixin):
    """Increase default mutation supply within a progress range"""

    def __init__(self, progress_range: tuple[float, float] = (0, 0), factor: float = 1) -> None:
        self.progress_range = progress_range
        self.mutator = ms.Mutator(
            sub_rate=1e-5 * factor, indel_rate=1e-6 * factor, break_rate=1e-4 * factor
        )

    def __call__(self, exp: "Experiment", divided: torch.Tensor) -> None:
        in_range = self.progress_range[0] < exp.progress <= self.progress_range[1]
        mutator = self.mutator if in_range else None

        mask = exp.world.idxzeros(exp.world.cells.c).to(torch.bool)
        mask[divided] = True
        exp.world.mutate_cells(mask=mask, mutator=mutator)


class PassagerKwargs(TypedDict, total=False):
    confl_range: tuple[float, float]


class Passager(StateMixin):
    """Passage cells if confluency is too high"""

    def __init__(self, confl_range: tuple[float, float] = (0.2, 0.7)) -> None:
        self.confl_range = confl_range
        self.passage = 0

    def __call__(self, exp: "Experiment") -> bool:
        n = exp.world.cells.get_n_cells()
        if n <= self.confl_range[1] * exp.world.culture.map_size**2:
            return False

        mask = exp.world.frand(exp.world.cells.c) > self.confl_range[0]
        exp.world.kill_cells(mask=mask)
        exp.world.resuspend_cells()
        self.passage += 1
        return True


class MediaWellKwargs(TypedDict, total=False):
    subs: list[str]
    cofs: list[str]


class MediaWell(StateMixin):
    """Replenish medium with individual substrate, co-factor, and other concentrations"""

    def __init__(
        self, chem: ms.Chemistry, subs: list[str] | None = None, cofs: list[str] | None = None
    ) -> None:
        if subs is None:
            subs = [d.name for d in substrates]

        if cofs is None:
            cofs = [d.name for d in cofactors]

        m2i = chem.molname_2_idx
        self.subs = [m2i[d] for d in subs]
        self.cofs = [m2i[d] for d in cofs]
        self.x_subs = 10.0
        self.x_cofs = 1.0
        self.x_trace = 0.1

    def __call__(self, exp: "Experiment") -> None:
        molmap = exp.world.culture.molecule_map
        molmap[:] = self.x_trace

        for idx in self.cofs:
            molmap[idx] = self.x_cofs

        for idx in self.subs:
            molmap[idx] = self.x_subs


class GenomeEditorKwargs(TypedDict, total=False):
    proteins: list[ms.ProteinSynthesizer]
    progress_gt: float


class GenomeEditor(StateMixin):
    """Edit genomes once progress surpasses threshold"""

    def __init__(
        self,
        world: ms.World,
        proteins: list[ms.ProteinSynthesizer] | None = None,
        progress_gt: float = 1.0,
    ) -> None:
        self.edited = False
        self.progress_gt = progress_gt
        self.synthesizer = ms.GenomeSynthesizer(world=world, proteins=proteins or [])

    def __call__(self, exp: "Experiment") -> None:
        if self.edited or exp.progress <= self.progress_gt:
            return

        alive = exp.world.cells.reconcile_host()
        idxs = alive.nonzero()[0].tolist()

        genomes = exp.world.cells.genomes.tolist()
        for idx, genome in enumerate(genomes):
            genomes[idx] = genome + self.synthesizer.generate()

        exp.world.cells.update_genomes(
            genomes=ms.StrArray.from_strings([genomes[d] for d in idxs]),
            idxs=exp.world.idxtensor(idxs),
        )
        self.edited = True


class ProgressorKwargs(TypedDict, total=False):
    n_passages: int
    min_gr: float


class PassageProgressor(StateMixin):
    """Progress linearly through n passages with minimum growth rate"""

    def __init__(self, n_passages: int = 10, min_gr: float = 0.0) -> None:
        self.n_passages = n_passages
        self.min_gr = min_gr
        self.valid_passages = 0

    def __call__(self, exp: "Experiment") -> float:
        if exp.growth_rate >= self.min_gr:
            self.valid_passages += 1
        return min(1.0, self.valid_passages / self.n_passages)


class GrowthRate(StateMixin):
    """Calculate growth rate based on cell generations in generations per second"""

    def __init__(self, world: ms.World, t0: float) -> None:
        self.g0 = float(world.cells.generations[world.cells.alive].mean())
        self.t0 = t0
        self.value = 0.0

    def update(self, world: ms.World, t: float) -> float:
        g1 = float(world.cells.generations[world.cells.alive].mean())
        self.value = (g1 / self.g0) / (t - self.t0)
        self.g0 = g1
        self.t0 = t
        return self.value


class Experiment(StateMixin):
    def __init__(
        self,
        world: ms.World,
        time_step: int = 10,
        mutator_kwargs: MutatorKwargs | None = None,
        passager_kwargs: PassagerKwargs | None = None,
        progressor_kwargs: ProgressorKwargs | None = None,
        mediawell_kwargs: MediaWellKwargs | None = None,
        genomeeditor_kwargs: GenomeEditorKwargs | None = None,
        mutator: Callable[["Experiment", torch.Tensor], None] | None = None,
        passager: Callable[["Experiment"], bool] | None = None,
        mediawell: Callable[["Experiment"], None] | None = None,
        progressor: Callable[["Experiment"], float] | None = None,
        genomeeditor: Callable[["Experiment"], None] | None = None,
    ) -> None:
        self.time = 0
        self.progress = 0.0
        self.time_step = time_step
        self.world = world
        self.killers = [EnergyKiller(chem=world.chemistry)]
        self.replicators = [XReplicator(chem=world.chemistry)]
        self.taxers = [EnergyTaxer(chem=world.chemistry)]
        self.mutator = mutator or Mutator(**(mutator_kwargs or {}))
        self.passager = passager or Passager(**(passager_kwargs or {}))
        self.mediawell = mediawell or MediaWell(chem=world.chemistry, **(mediawell_kwargs or {}))
        self.progressor = progressor or PassageProgressor(**(progressor_kwargs or {}))
        self.genomeeditor = genomeeditor or GenomeEditor(world=world, **(genomeeditor_kwargs or {}))
        self.gr = GrowthRate(world=world, t0=self.time)

    def advance_time(self) -> None:
        self.world.advance_time(n_steps=self.time_step)

        for taxer in self.taxers:
            taxer(self)

        for killer in self.killers:
            killer(self)

        for replicator in self.replicators:
            divided = replicator(self)
            self.mutator(self, divided)

        self.time += self.time_step

    def passage_cells(self) -> bool:
        self.gr.update(world=self.world, t=self.time)
        passaged = self.passager(self)
        if passaged:
            self.mediawell(self)
            self.progress = self.progressor(self)
            self.genomeeditor(self)

        return passaged
