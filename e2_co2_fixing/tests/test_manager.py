import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import magicsoup as ms
import pytest
from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore[import-untyped]
    EventAccumulator,
)
from tensorboard.backend.event_processing.event_file_loader import (  # type: ignore[import-untyped]
    RawEventFileLoader,
)
from tensorboard.backend.event_processing.plugin_event_accumulator import (  # type: ignore[import-untyped]
    EventAccumulator as PluginEventAccumulator,
)
from torch.utils.tensorboard.writer import Event, SessionLog

from src.chemistry import chemistry
from src.experiment import Experiment
from src.manager import (
    CellAgeEarlyStop,
    Checkpointer,
    Manager,
    Monitor,
    PassageEarlyStop,
    TimeEarlyStop,
)


class _StubExperiment:
    def __init__(self, world: ms.World, time_step: int = 7) -> None:
        self.time = 0
        self.progress = 0.0
        self.passage = 0
        self.growth_rate = 0.0
        self.time_step = time_step
        self.world = world
        self.passager = SimpleNamespace(passage=0)
        self.gr = SimpleNamespace(value=0.0)
        self.advance_calls = 0

    def advance_time(self) -> None:
        self.advance_calls += 1
        self.time += self.time_step

    def save_state(self, statedir: Path) -> None:
        statedir.mkdir(parents=True, exist_ok=True)
        payload = {
            "time": self.time,
            "progress": self.progress,
            "passage": self.passager.passage,
        }
        (statedir / "state.json").write_text(json.dumps(payload))

    def load_state(self, statedir: Path) -> None:
        payload = json.loads((statedir / "state.json").read_text())
        self.time = payload["time"]
        self.progress = payload["progress"]
        self.passager.passage = payload["passage"]


@pytest.fixture
def experiment() -> Experiment:
    world = ms.World(chemistry=chemistry)
    return cast(Experiment, _StubExperiment(world=world))


def _read_steps(logdir: Path, tag: str) -> tuple[list[int], list[int]]:
    legacy = EventAccumulator(str(logdir))
    legacy.Reload()
    plugin = PluginEventAccumulator(str(logdir))
    plugin.Reload()
    return (
        [event.step for event in legacy.Scalars(tag)],
        [event.step for event in plugin.Tensors(tag)],
    )


def _read_session_start_steps(logdir: Path) -> list[int]:
    steps: list[int] = []
    for path in sorted(logdir.glob("events.out.tfevents.*")):
        for record in RawEventFileLoader(str(path)).Load():
            event = Event.FromString(record)
            if event.HasField("session_log") and event.session_log.status == SessionLog.START:
                steps.append(event.step)
    return steps


def test_revert_logs_clears_scalars_after_rollback_time(
    tmp_path: Path, experiment: Experiment
) -> None:
    monitor = Monitor(chem=chemistry, rundir=tmp_path, mols=chemistry.molecules[:2])
    monitor.log_summary(hparams={"lr": 0.1, "tags": ["a", "b"]})

    live_legacy = EventAccumulator(str(tmp_path))
    live_plugin = PluginEventAccumulator(str(tmp_path))

    for step in range(0, 1001, 100):
        experiment.time = step
        experiment.progress = step / 1000.0
        monitor.log_scalars(exp=experiment)

    monitor.writer.flush()
    live_legacy.Reload()
    live_plugin.Reload()
    assert [e.step for e in live_legacy.Scalars("Other/Progress")] == list(range(0, 1001, 100))
    assert [e.step for e in live_plugin.Tensors("Other/Progress")] == list(range(0, 1001, 100))

    experiment.time = 500
    experiment.progress = 0.5
    monitor.revert_logs(exp=experiment)

    assert _read_session_start_steps(tmp_path) == [0, 501]

    expected = [0, 100, 200, 300, 400, 500]
    live_legacy.Reload()
    live_plugin.Reload()
    assert [e.step for e in live_legacy.Scalars("Other/Progress")] == expected
    assert [e.step for e in live_plugin.Tensors("Other/Progress")] == expected

    fresh_legacy, fresh_plugin = _read_steps(tmp_path, "Other/Progress")
    assert fresh_legacy == expected
    assert fresh_plugin == expected

    mol_tag = f"{chemistry.molecules[0].name}[ext]"
    mol_legacy, mol_plugin = _read_steps(tmp_path, mol_tag)
    assert mol_legacy == expected
    assert mol_plugin == expected

    monitor.close()


def test_revert_logs_multiple_rollbacks_and_without_init_summary(
    tmp_path: Path, experiment: Experiment
) -> None:
    monitor = Monitor(chem=chemistry, rundir=tmp_path)

    for step in [0, 200, 400, 600]:
        experiment.time = step
        monitor.log_scalars(exp=experiment)

    experiment.time = 200
    monitor.revert_logs(exp=experiment)
    assert _read_session_start_steps(tmp_path) == [201, 201]
    assert _read_steps(tmp_path, "Other/Progress") == ([0, 200], [0, 200])

    for step in [300, 500]:
        experiment.time = step
        monitor.log_scalars(exp=experiment)
    monitor.writer.flush()
    assert _read_steps(tmp_path, "Other/Progress") == ([0, 200, 300, 500], [0, 200, 300, 500])

    experiment.time = 0
    monitor.revert_logs(exp=experiment)
    assert _read_session_start_steps(tmp_path) == [201, 201, 1]
    assert _read_steps(tmp_path, "Other/Progress") == ([0], [0])

    monitor.close()


def test_checkpointer_saves_sorts_loads_and_removes(tmp_path: Path, experiment: Experiment) -> None:
    ckpt = Checkpointer(rundir=tmp_path)
    assert ckpt.get_last_state() is None

    for t in [0, 50, 1000, 100]:
        experiment.time = t
        ckpt.save_state(exp=experiment)

    experiment.time = 9999
    last_state = ckpt.get_last_state()
    assert last_state is not None
    ckpt.load_state(exp=experiment, state=last_state)
    assert experiment.time == 1000

    ckpt.rm_last_state()
    assert not (tmp_path / "t=1000").exists()
    last_state = ckpt.get_last_state()
    assert last_state is not None
    ckpt.load_state(exp=experiment, state=last_state)
    assert experiment.time == 100


def test_early_stoppers_return_bool_and_reset(experiment: Experiment) -> None:
    age_stop = CellAgeEarlyStop(max_age=10.0)
    assert experiment.world.cells.get_n_cells() == 0
    assert age_stop(experiment) is True
    age_stop.reset(experiment)

    pass_stop = PassageEarlyStop(max_passages=2)
    time_stop = TimeEarlyStop(max_time=50)

    experiment.passager.passage = 3  # type: ignore[attr-defined]
    experiment.time = 60
    assert pass_stop(experiment) is True
    assert time_stop(experiment) is True

    # Rolling back to an earlier checkpoint and resetting prevents instant re-triggering.
    experiment.progress = 0.2
    experiment.passager.passage = 1  # type: ignore[attr-defined]
    experiment.time = 20
    pass_stop.reset(experiment)
    time_stop.reset(experiment)

    assert pass_stop(experiment) is False
    assert time_stop(experiment) is False


def test_manager_run_completes_with_step_counter_and_initial_checkpoint(tmp_path: Path) -> None:
    world = ms.World(chemistry=chemistry)
    stub = _StubExperiment(world=world, time_step=7)

    original_advance = stub.advance_time

    def advancing() -> None:
        original_advance()
        if stub.advance_calls >= 6:
            stub.progress = 1.0

    stub.advance_time = advancing  # type: ignore[method-assign]
    exp = cast(Experiment, stub)

    with Manager(
        rundir=tmp_path,
        hparams={"lr": 0.1},
        chem=chemistry,
        check_interval=2,
        checkpoint_interval=4,
        monitor_kwargs={"mols": chemistry.molecules[:1]},
        stoppers=[],
    ) as mngr:
        ok, err = mngr.run(exp=exp)

    # check_interval=2 so logs at steps 2,4,6; step 6 sets progress=1.0 then also logs (t=42).
    # Logged at: initial t=0, then t=14, t=28, t=42.
    assert (ok, err) == (True, None)
    assert (tmp_path / "hparams.json").exists()
    assert (tmp_path / "t=0").is_dir()
    assert (tmp_path / f"t={4 * 7}").is_dir()
    assert mngr.steps == 6
    assert _read_steps(tmp_path, "Other/Progress") == ([0, 14, 28, 42], [0, 14, 28, 42])
    assert (
        f"{chemistry.molecules[0].name}[ext]"
        in EventAccumulator(str(tmp_path)).Reload().Tags()["scalars"]
    )


def test_manager_run_rolls_back_resets_stoppers_and_recovers(tmp_path: Path) -> None:
    # Scenario: first trial stalls (no progress for >max_time steps); manager rolls back
    # to the initial t=0 checkpoint, resets the stopper, and the second trial completes.
    # We use max_time=5 so that the stopper fires at t=10 (>5) in the first trial.
    # After rollback, exp.time resets to 0; the second trial sets progress=1.0 immediately.
    world = ms.World(chemistry=chemistry)
    stub = _StubExperiment(world=world, time_step=10)

    original_advance = stub.advance_time

    def advancing() -> None:
        original_advance()
        # exp.time resets to 0 after rollback; advance_calls never resets.
        # So on the second time we see t=10, advance_calls will be > 1.
        if stub.time == 10 and stub.advance_calls > 1:
            stub.progress = 1.0

    stub.advance_time = advancing  # type: ignore[method-assign]
    exp = cast(Experiment, stub)

    stopper = TimeEarlyStop(max_time=5)
    with Manager(
        rundir=tmp_path,
        hparams={"lr": 0.1},
        chem=chemistry,
        max_trials=3,
        check_interval=1,
        checkpoint_interval=100,
        stoppers=[stopper],
    ) as mngr:
        ok, err = mngr.run(exp=exp)

    assert (ok, err) == (True, None)
    # advance_calls > 1: first trial had at least 1 call, rollback happened, second trial ran.
    assert stub.advance_calls > 1
    # After rollback, logs were purged to t=0; second trial logged t=10 with progress=1.0.
    steps_logged = _read_steps(tmp_path, "Other/Progress")[0]
    assert 0 in steps_logged
    assert 10 in steps_logged


def test_manager_run_returns_failure_when_checkpoints_exhausted(tmp_path: Path) -> None:
    world = ms.World(chemistry=chemistry)
    exp = cast(Experiment, _StubExperiment(world=world, time_step=10))

    class _AlwaysStop:
        def __init__(self) -> None:
            self.resets = 0

        def __call__(self, _exp: Experiment) -> bool:
            return True

        def reset(self, _exp: Experiment) -> None:
            self.resets += 1

    stopper = _AlwaysStop()
    with Manager(
        rundir=tmp_path,
        hparams={"lr": 0.1},
        chem=chemistry,
        max_trials=2,
        check_interval=1,
        checkpoint_interval=10,
        stoppers=[stopper],
    ) as mngr:
        ok, err = mngr.run(exp=exp)

    assert ok is False
    assert err == "No states left to load"
    assert not (tmp_path / "t=0").exists()
    assert stopper.resets == 2
