import math
from collections.abc import Callable

import magicsoup as ms
import torch


class Killer:
    def __call__(self, exp: "Experiment") -> None:
        mask = exp.world.cells.molecules[:, 3] < 0.01
        exp.world.kill_cells(mask=mask)


class Replicator:
    def __call__(self, exp: "Experiment") -> torch.Tensor:
        mask = exp.world.cells.molecules[:, 3] > 1
        idxs = exp.world.divide_cells(mask=mask)
        exp.world.cells.molecules[idxs, 3] -= 1
        exp.world.cells.molecules.clamp_(0)
        return idxs


class Mutator:
    def __call__(self, exp: "Experiment", divided: torch.Tensor) -> None:
        mask = exp.world.idxzeros(exp.world.cells.c).to(torch.bool)
        mask[divided] = True
        exp.world.mutate_cells(mask=mask)


class Passager:
    def __init__(self, lo: float = 0.2, hi: float = 0.8) -> None:
        self.hi = hi
        self.lo = lo
        self.n0 = 0
        self.t0 = 0

    def __call__(self, exp: "Experiment") -> float | None:
        t1 = exp.time
        n1 = exp.world.cells.get_n_cells()
        self.n0 = n1 if self.n0 == 0 else self.n0
        if n1 <= self.hi * exp.world.culture.map_size**2:
            return False

        mask = exp.world.frand(exp.world.cells.c) > self.lo
        exp.world.kill_cells(mask=mask)

        exp.world.culture.molecule_map[:] = 10.0

        rate = math.log2(n1 / self.n0) / (t1 - self.t0)
        self.n0 = exp.world.cells.get_n_cells()
        self.t0 = t1
        return rate


class Experiment:
    def __init__(
        self,
        world: ms.World,
        time_step: int = 10,
        killer: Callable[["Experiment"], None] | None = None,
        replicator: Callable[["Experiment"], torch.Tensor] | None = None,
        mutator: Callable[["Experiment", torch.Tensor], None] | None = None,
        passager: Callable[["Experiment"], float | None] | None = None,
    ) -> None:
        self.time_step = time_step
        self.world = world
        self.killer = killer or Killer()
        self.replicator = replicator or Replicator()
        self.mutator = mutator or Mutator()
        self.passager = passager or Passager()
        self.time = 0
        self.passage = 0
        self.growth_rate = 0.0

    def progress(self) -> None:
        self.world.advance_time(n_steps=self.time_step)
        self.killer(self)
        divided = self.replicator(self)
        self.mutator(self, divided)
        self.time += self.time_step

    def passage_cells(self) -> bool:
        res = self.passager(self)
        if res is None:
            return False

        self.growth_rate = res
        self.passage += 1
        return True
