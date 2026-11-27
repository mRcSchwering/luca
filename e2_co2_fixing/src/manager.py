import json
import logging
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

import magicsoup as ms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
from torch.utils.tensorboard.writer import Event, SessionLog

from src.chemistry import chemistry
from src.experiment import Experiment

log = logging.getLogger(__name__)


class RunFailedErr(Exception):
    pass


class TrialFailedErr(Exception):
    pass


class Checkpointer:
    def __init__(self, rundir: Path) -> None:
        self.rundir = rundir

    def save_state(self, exp: Experiment) -> None:
        statedir = self.rundir / f"t={exp.time}"
        exp.save_state(statedir=statedir)

    def load_last_state(self, exp: Experiment) -> None:
        states = self._get_sorted_states()
        if len(states) == 0:
            log.warning("No states left to load")
            raise RunFailedErr("No states left")

        statedir = self.rundir / states[-1]
        exp.load_state(statedir=statedir)
        log.info("Loaded last state %s", statedir)

    def rm_last_state(self) -> None:
        states = self._get_sorted_states()
        if len(states) > 0:
            shutil.rmtree(self.rundir / states[-1])

    def _get_sorted_states(self) -> list[str]:
        statedirs = [d for d in self.rundir.glob("t=*") if d.is_dir()]
        state_times = [(d.name, int(d.removeprefix("t="))) for d in statedirs]
        state_times.sort(key=lambda d: d[1])
        return [d[0] for d in state_times]


class Monitor:
    def __init__(self, chem: ms.Chemistry, rundir: Path, mols: Iterable[ms.Molecule] = ()):
        m2i = chem.mol_2_idx
        self.m2i = {d.name: m2i[d] for d in mols}
        self.writer = SummaryWriter(log_dir=rundir)
        self._session_started = False

    def _emit_session_start(self, step: int) -> None:
        file_writer = self.writer.file_writer
        if file_writer is None:
            return
        file_writer.add_event(Event(step=step, session_log=SessionLog(status=SessionLog.START)))
        self._session_started = True

    def init_summary(self, hparams: dict) -> None:
        self._emit_session_start(step=0)
        metrics = {"Other/Progress": 0.0}
        flat_hparams = {
            k: ", ".join([str(dd) for dd in d]) if isinstance(d, (list, tuple)) else d
            for k, d in hparams.items()
        }
        exp, ssi, sei = get_summary(hparam_dict=flat_hparams, metric_dict=metrics)
        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)

    def log(self, exp: Experiment) -> None:
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
            self.writer.add_scalar("Cells/GrowthRate", exp.gr.value, time)
            self.writer.add_scalar("Cells/Proteins", prot, time)
            cellmols = exp.world.cells.molecules
            for mol, idx in self.m2i.items():
                x = float(cellmols[alive, idx].mean())
                self.writer.add_scalar(f"{mol}[int]", x, time)

        self.writer.add_scalar("Other/Passage", exp.passager.passage, time)
        self.writer.add_scalar("Other/Progress", exp.progress, time)

    def revert_logs(self, exp: Experiment) -> None:
        if self.writer.file_writer is None:
            return

        self.writer.flush()
        purge_step = int(exp.time) + 1
        if not self._session_started:
            # TensorBoard's plugin_event_accumulator treats the first SessionLog.START
            # as an initial session marker and only purges on subsequent restarts.
            self._emit_session_start(step=purge_step)
        self._emit_session_start(step=purge_step)
        self.writer.flush()


class PassageEarlyStop:
    def __init__(self, max_passages: int = 10) -> None:
        self.max_passages = max_passages
        self.last_prog_pass = 0
        self.progress = 0.0

    def __call__(self, exp: "Experiment") -> None:
        if exp.progress > self.progress:
            self.progress = exp.progress
            self.last_prog_pass = exp.passager.passage
            return

        if (exp.passager.passage - self.last_prog_pass) > self.max_passages:
            log.warning("%s passages without progress", self.max_passages)
            raise TrialFailedErr("Too many passages without progress")


class CellAgeEarlyStop:
    def __init__(self, max_age: float = 1000.0) -> None:
        self.max_age = max_age

    def __call__(self, exp: Experiment) -> None:
        age = float(exp.world.cells.ages[exp.world.cells.alive].median())
        if age > self.max_age:
            log.warning("Cells reached max age %s", self.max_age)
            raise TrialFailedErr("Cells got too old")


class TimeEarlyStop:
    def __init__(self, max_time: int = 10_000) -> None:
        self.max_time = max_time
        self.last_prog_time = 0
        self.progress = 0.0

    def __call__(self, exp: Experiment) -> None:
        if exp.progress > self.progress:
            self.progress = exp.progress
            self.last_prog_time = exp.time
            return

        if (exp.time - self.last_prog_time) > self.max_time:
            log.warning("%s s without progress", self.max_time)
            raise TrialFailedErr("Too much time spent without progress")


class Manager:
    def __init__(
        self,
        rundir: Path,
        hparams: dict,
        chem: ms.Chemistry,
        stoppers: Iterable[Callable[[Experiment], None]],
    ) -> None:
        self.rundir = rundir
        self.hparams = hparams
        self.trials = 0
        self.max_trials = 5

        self.stoppers = stoppers
        self.checkpointer = Checkpointer(rundir=self.rundir)
        self.monitor = Monitor(chem=chem, rundir=self.rundir)

    def setup(self) -> None:
        self.rundir.mkdir(parents=True, exist_ok=True)
        (self.rundir / "hparams.json").write_text(json.dumps(self.hparams, indent=2))
        self.monitor.init_summary(hparams=self.hparams)

    def rollback(self, exp: Experiment) -> None:
        log.info("Rolling back trial")
        self.trials += 1
        if self.trials >= self.max_trials:
            log.info("Reached %s trials. Deleting previous checkpoint", self.trials)
            self.checkpointer.rm_last_state()
            self.trials = 0

        self.checkpointer.load_last_state(exp=exp)
        self.monitor.revert_logs(exp=exp)

    def checkpoint(self, exp: Experiment) -> None:
        self.checkpointer.save_state(exp=exp)

    def early_stopping(self, exp: Experiment) -> None:
        try:
            for stopper in self.stoppers:
                stopper(exp)
        except TrialFailedErr:
            self.rollback(exp=exp)


def main(hparams: dict, runsdir: Path) -> None:
    world = ms.World(chemistry=chemistry)
    exp = Experiment(world=world)
    mngr = Manager(
        rundir=runsdir / "asd",
        hparams=hparams,
        chem=chemistry,
        stoppers=[PassageEarlyStop(), CellAgeEarlyStop(), TimeEarlyStop()],
    )
    mngr.setup()
    mngr.monitor.log(exp=exp)

    try:
        while True:
            exp.advance_time()

            if exp.time % 100 == 0:
                passaged = exp.passage_cells()
                if passaged:
                    exp.world.compact()

                mngr.early_stopping(exp=exp)
                mngr.monitor.log(exp=exp)

            if exp.time % 10_000 == 0:
                mngr.checkpoint(exp=exp)

            if exp.progress == 1.0:
                log.info("Run complete")
                break
    except RunFailedErr as err:
        log.warning("Run failed: %s", err)
