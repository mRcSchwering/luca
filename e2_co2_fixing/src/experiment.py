import math
from collections.abc import Callable
from typing import TypedDict

import magicsoup as ms
import torch

from src.chemistry import chemistry

_energy_thresholds = {d: 0.1 for d in ("ATP", "NADPH", "QH2", "Fe-red", "FADH2", "Fd-red")}


class EnergyKiller:
    def __init__(self, chemistry: ms.Chemistry) -> None:
        m2i = chemistry.molname_2_idx
        self.i2t = {m2i[m]: t for m, t in _energy_thresholds.items()}

    def __call__(self, exp: "Experiment") -> None:
        masks = [exp.world.cells.molecules[:, i] < t for i, t in self.i2t.items()]
        mask = torch.stack(masks).all(dim=0)
        exp.world.kill_cells(mask=mask)


class EnergyTaxer:
    per_protein_cost = 0.01  # mM ATP per protein

    def __init__(self, chemistry: ms.Chemistry) -> None:
        m2i = chemistry.molname_2_idx
        self.i_atp = m2i["ATP"]

    def __call__(self, exp: "Experiment") -> None:
        protein_bill = exp.world.cells.proteins * self.per_protein_cost
        exp.world.cells.molecules[:, self.i_atp] -= protein_bill
        exp.world.cells.molecules[:, self.i_atp].clamp_(0)


class XReplicator:
    x_cost = 5

    def __init__(self, chemistry: ms.Chemistry) -> None:
        m2i = chemistry.molname_2_idx
        self.i_x = m2i["X"]
        self.energies = {m2i[m]: t for m, t in _energy_thresholds.items()}

    def __call__(self, exp: "Experiment") -> torch.Tensor:
        cell_molecules = exp.world.cells.molecules

        masks = [cell_molecules[:, i] >= d for i, d in self.energies.items()]
        energy_mask = torch.stack(masks).all(dim=0)

        x_mask = cell_molecules[:, self.i_x] > self.x_cost

        mask = energy_mask & x_mask
        divided = exp.world.divide_cells(mask=mask)

        cell_molecules[divided, self.i_x] -= self.x_cost
        cell_molecules[divided, self.i_x].clamp_(0)

        return divided


class MutatorKwargs(TypedDict, total=False):
    progress_range: tuple[float, float]
    factor: float


class Mutator:
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


class Passager:
    def __init__(self, confl_range: tuple[float, float] = (0.2, 0.7)) -> None:
        self.confl_range = confl_range
        self.passage = 0

    def __call__(self, exp: "Experiment", n1: int) -> bool:
        if n1 <= self.confl_range[1] * exp.world.culture.map_size**2:
            return False

        mask = exp.world.frand(exp.world.cells.c) > self.confl_range[0]
        exp.world.kill_cells(mask=mask)

        exp.world.culture.molecule_map[:] = 10.0

        self.passage += 1
        return True


class ProgressorKwargs(TypedDict, total=False):
    n_generations: int
    min_gr: float


class Progressor:
    def __init__(self, n_generations: int = 10, min_gr: float = 0.05) -> None:
        self.n_generations = n_generations
        self.min_gr = min_gr
        self.valid_generations = 0

    def __call__(self, exp: "Experiment") -> float:
        if exp.growth_rate >= self.min_gr:
            self.valid_generations += 1
        return min(1.0, self.valid_generations / self.n_generations)


class GrowthRate:
    def __init__(self, n0: int, t0: float) -> None:
        self.n0 = n0
        self.t0 = t0
        self.value = 0.0

    def update(self, n1: int, t1: float) -> float:
        self.value = math.log2(n1 / self.n0) / (t1 - self.t0)
        self.n0 = n1
        self.t0 = t1
        return self.value


class Experiment:
    def __init__(
        self,
        world: ms.World,
        time_step: int = 10,
        mutator_kwargs: MutatorKwargs | None = None,
        passager_kwargs: PassagerKwargs | None = None,
        progressor_kwargs: ProgressorKwargs | None = None,
        mutator: Callable[["Experiment", torch.Tensor], None] | None = None,
        passager: Callable[["Experiment", int], bool] | None = None,
        progressor: Callable[["Experiment"], float] | None = None,
    ) -> None:
        self.time = 0
        self.progress = 0.0
        self.time_step = time_step
        self.world = world
        self.killers = [EnergyKiller(chemistry=world.chemistry)]
        self.replicators = [XReplicator(chemistry=world.chemistry)]
        self.taxers = [EnergyTaxer(chemistry=world.chemistry)]
        self.mutator = mutator or Mutator(**(mutator_kwargs or {}))
        self.passager = passager or Passager(**(passager_kwargs or {}))
        self.progressor = progressor or Progressor(**(progressor_kwargs or {}))
        self.gr = GrowthRate(n0=world.cells.get_n_cells(), t0=self.time)

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
        n_cells = self.world.cells.get_n_cells()
        passaged = self.passager(self, n_cells)
        if passaged:
            self.gr.update(n1=n_cells, t1=self.time)

        return passaged


def main() -> None:
    world = ms.World(chemistry=chemistry)

    exp = Experiment(world=world)

    while True:
        exp.advance_time()

        if exp.time % 100 == 0:
            exp.passage_cells()
