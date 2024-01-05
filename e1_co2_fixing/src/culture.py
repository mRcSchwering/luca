import math
from typing import Protocol
import magicsoup as ms


class CallCltrType(Protocol):
    def __call__(self, cltr: "Culture"):
        ...


class CallCltrReturnFloatType(Protocol):
    def __call__(self, cltr: "Culture") -> float:
        ...


class CallBatchCltrReturnBoolType(Protocol):
    def __call__(self, cltr: "BatchCulture") -> float:
        ...


class Culture:
    """Baseclass for culturing cells"""

    def __init__(
        self,
        world: ms.World,
        medium_refresher: CallCltrType,
        killer: CallCltrType,
        replicator: CallCltrType,
        mutator: CallCltrType,
        progressor: CallCltrReturnFloatType,
        stopper: CallCltrType,
    ):
        self.world = world
        self.step_i = 0
        self.progress = 0.0
        self.medium_refresher = medium_refresher
        self.progressor = progressor
        self.stopper = stopper
        self.killer = killer
        self.replicator = replicator
        self.mutator = mutator
        self.medium_refresher(self)

    def __iter__(self):
        return self

    def post_replication(self):
        pass

    def __next__(self):
        self.step_i = self.step_i + 1
        self.stopper(self)

        self.world.diffuse_molecules()
        self.world.degrade_molecules()
        self.world.enzymatic_activity()
        self.killer(self)
        self.mutator(self)
        self.replicator(self)
        self.post_replication()
        self.world.increment_cell_lifetimes()

        self.progress = self.progressor(self)
        return self.step_i


class ChemoStat(Culture):
    """Grow cells in ChemoStat"""

    def post_replication(self):
        self.medium_refresher(self)


class BatchCulture(Culture):
    """Grow cells in batch culture"""

    def __init__(
        self,
        passager: CallBatchCltrReturnBoolType,
        genome_editor: CallCltrType | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.passager = passager
        self.genome_editor = genome_editor
        self.growth_rate = 0.0
        self.cpd = 0.0
        self.split_i = 0
        self.split_start_step = 0
        self.split_start_cells = self.world.n_cells

    def post_replication(self):
        n_steps = self.step_i - self.split_start_step
        doubling = math.log(self.world.n_cells / self.split_start_cells, 2)
        if n_steps > 0:
            self.growth_rate = doubling / n_steps

        if self.passager(self):
            self.cpd += doubling
            self.medium_refresher(self)
            self.split_i += 1
            self.split_start_step = self.step_i
            self.split_start_cells = self.world.n_cells

        if self.genome_editor is not None:
            self.genome_editor(self)
