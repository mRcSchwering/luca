import json
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, Self, TypedDict

import magicsoup as ms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
from torch.utils.tensorboard.writer import Event, SessionLog

from src.experiment import Experiment

log = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, rundir: Path) -> None:
        self.rundir = rundir

    def save_state(self, exp: Experiment) -> None:
        statedir = self.rundir / f"t={exp.time}"
        exp.save_state(statedir=statedir)

    def load_state(self, exp: Experiment, state: str) -> None:
        statedir = self.rundir / state
        exp.load_state(statedir=statedir)
        log.info("Loaded last state %s", statedir)

    def get_last_state(self) -> str | None:
        states = self._get_sorted_states()
        if len(states) == 0:
            log.warning("No states left to load")
            return None

        return states[-1]

    def rm_last_state(self) -> None:
        states = self._get_sorted_states()
        if len(states) > 0:
            shutil.rmtree(self.rundir / states[-1])

    def _get_sorted_states(self) -> list[str]:
        statedirs = [d for d in self.rundir.glob("t=*") if d.is_dir()]
        state_times = [(d.name, int(d.name.removeprefix("t="))) for d in statedirs]
        state_times.sort(key=lambda d: d[1])
        return [d[0] for d in state_times]


class MonitorKwargs(TypedDict, total=False):
    mols: Iterable[ms.Molecule]


class Monitor:
    def __init__(self, chem: ms.Chemistry, rundir: Path, mols: Iterable[ms.Molecule] = ()):
        m2i = chem.mol_2_idx
        self.m2i = {d.name: m2i[d] for d in mols}
        self.writer = SummaryWriter(log_dir=rundir)
        self._emit_session_start(step=0)

    def log_summary(self, hparams: dict) -> None:
        metrics = {"Other/Progress": 0.0}
        flat_hparams = {
            k: ", ".join([str(dd) for dd in d]) if isinstance(d, (list, tuple)) else d
            for k, d in hparams.items()
        }
        exp, ssi, sei = get_summary(hparam_dict=flat_hparams, metric_dict=metrics)
        if self.writer.file_writer is None:
            return

        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)

    def log_scalars(self, exp: Experiment) -> None:
        time = exp.time
        n_cells = exp.world.cells.get_n_cells()
        molmap = exp.world.culture.molecule_map
        for mol, idx in self.m2i.items():
            x = float(molmap[idx].mean())
            self.writer.add_scalar(f"{mol}[ext]", x, time)

        if n_cells > 0:
            alive = exp.world.cells.alive
            age = float(exp.world.cells.ages[alive].mean())
            gen = float(exp.world.cells.generations[alive].mean())
            prot = float(exp.world.cells.proteins[alive].mean())
            self.writer.add_scalar("Cells/Total", n_cells, time)
            self.writer.add_scalar("Cells/Age", age, time)
            self.writer.add_scalar("Cells/Generation", gen, time)
            self.writer.add_scalar("Cells/Proteins", prot, time)
            cellmols = exp.world.cells.molecules
            for mol, idx in self.m2i.items():
                x = float(cellmols[alive, idx].mean())
                self.writer.add_scalar(f"{mol}[int]", x, time)

        self.writer.add_scalar("Cells/GrowthRate", exp.growth_rate, time)
        self.writer.add_scalar("Other/Passage", exp.passage, time)
        self.writer.add_scalar("Other/Progress", exp.progress, time)

    def revert_logs(self, exp: Experiment) -> None:
        if self.writer.file_writer is None:
            return

        self.writer.flush()
        purge_step = int(exp.time) + 1
        self._emit_session_start(step=purge_step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()

    def _emit_session_start(self, step: int) -> None:
        file_writer = self.writer.file_writer
        if file_writer is None:
            return
        file_writer.add_event(Event(step=step, session_log=SessionLog(status=SessionLog.START)))


class Stopper(Protocol):
    def __call__(self, exp: Experiment, /) -> bool: ...

    def reset(self, exp: Experiment, /) -> None: ...


class PassageEarlyStop:
    def __init__(self, max_passages: int = 10) -> None:
        self.max_passages = max_passages
        self.last_prog_pass = 0
        self.progress = 0.0

    def __call__(self, exp: Experiment) -> bool:
        if exp.progress > self.progress:
            self.progress = exp.progress
            self.last_prog_pass = exp.passage
            return False

        if (exp.passage - self.last_prog_pass) > self.max_passages:
            log.warning("%s passages without progress", self.max_passages)
            return True

        return False

    def reset(self, exp: Experiment) -> None:
        self.progress = exp.progress
        self.last_prog_pass = exp.passage


class CellAgeEarlyStop:
    def __init__(self, max_age: float = 1000.0) -> None:
        self.max_age = max_age

    def __call__(self, exp: Experiment) -> bool:
        if exp.world.cells.get_n_cells() == 0:
            log.warning("All cells died")
            return True

        age = float(exp.world.cells.ages[exp.world.cells.alive].median())
        if age > self.max_age:
            log.warning("Cells reached max age %s", self.max_age)
            return True

        return False

    def reset(self, _exp: Experiment) -> None:
        return None


class TimeEarlyStop:
    def __init__(self, max_time: int = 10_000) -> None:
        self.max_time = max_time
        self.last_prog_time = 0
        self.progress = 0.0

    def __call__(self, exp: Experiment) -> bool:
        if exp.progress > self.progress:
            self.progress = exp.progress
            self.last_prog_time = exp.time
            return False

        if (exp.time - self.last_prog_time) > self.max_time:
            log.warning("%s s without progress", self.max_time)
            return True

        return False

    def reset(self, exp: Experiment) -> None:
        self.progress = exp.progress
        self.last_prog_time = exp.time


class Manager:
    def __init__(
        self,
        rundir: Path,
        hparams: dict,
        chem: ms.Chemistry,
        max_trials: int = 5,
        check_interval: int = 10,
        checkpoint_interval: int = 1000,
        monitor_kwargs: MonitorKwargs | None = None,
        checkpointer: Checkpointer | None = None,
        monitor: Monitor | None = None,
        stoppers: Iterable[Stopper] | None = None,
    ) -> None:
        self.rundir = rundir
        self.hparams = hparams
        self.trials = 0
        self.max_trials = max_trials
        self.steps = 0
        self.check_interval = check_interval
        self.checkpoint_interval = checkpoint_interval

        self.rundir.mkdir(parents=True, exist_ok=True)
        (self.rundir / "hparams.json").write_text(json.dumps(hparams, indent=2))

        self.stoppers: list[Stopper] = (
            list(stoppers)
            if stoppers is not None
            else [PassageEarlyStop(), CellAgeEarlyStop(), TimeEarlyStop()]
        )
        self.checkpointer = checkpointer or Checkpointer(rundir=self.rundir)
        self.monitor = monitor or Monitor(chem=chem, rundir=self.rundir, **(monitor_kwargs or {}))

    def __enter__(self) -> Self:
        self.monitor.log_summary(hparams=self.hparams)
        return self

    def __exit__(self, *_: object) -> None:
        self.monitor.close()

    def rollback(self, exp: Experiment) -> bool:
        log.info("Rolling back trial")
        self.trials += 1
        if self.trials >= self.max_trials:
            log.info("Reached %s trials. Deleting previous checkpoint", self.trials)
            self.checkpointer.rm_last_state()
            self.trials = 0

        last_state = self.checkpointer.get_last_state()
        if last_state is None:
            return False

        self.checkpointer.load_state(exp=exp, state=last_state)

        self.monitor.revert_logs(exp=exp)
        for stopper in self.stoppers:
            stopper.reset(exp)
        return True

    def checkpoint(self, exp: Experiment) -> None:
        self.checkpointer.save_state(exp=exp)
        self.trials = 0

    def early_stopping(self, exp: Experiment) -> bool:
        for stopper in self.stoppers:
            if stopper(exp):
                return True
        return False

    def run(self, exp: Experiment) -> tuple[bool, str | None]:
        self.steps = 0
        self.trials = 0
        self.monitor.log_scalars(exp=exp)
        self.checkpoint(exp=exp)
        for stopper in self.stoppers:
            stopper.reset(exp)

        while exp.progress < 1.0:
            exp.advance_time()
            self.steps += 1

            if self.steps % self.check_interval == 0:
                if self.early_stopping(exp=exp):
                    if not self.rollback(exp=exp):
                        msg = "No states left to load"
                        log.warning("Run failed: %s", msg)
                        return False, msg
                    continue

                self.monitor.log_scalars(exp=exp)

            if self.steps % self.checkpoint_interval == 0:
                self.checkpoint(exp=exp)

        log.info("Run complete")
        return True, None
