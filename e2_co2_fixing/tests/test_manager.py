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
from src.manager import Monitor


@pytest.fixture
def experiment() -> Experiment:
    world = ms.World(chemistry=chemistry)
    stub = SimpleNamespace(
        time=0,
        progress=0.0,
        world=world,
        passager=SimpleNamespace(passage=0),
        gr=SimpleNamespace(value=0.0),
    )
    return cast(Experiment, stub)


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
    monitor.init_summary(hparams={"lr": 0.1, "tags": ["a", "b"]})

    live_legacy = EventAccumulator(str(tmp_path))
    live_plugin = PluginEventAccumulator(str(tmp_path))

    for step in range(0, 1001, 100):
        experiment.time = step
        experiment.progress = step / 1000.0
        monitor.log(exp=experiment)

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

    monitor.writer.close()


def test_revert_logs_multiple_rollbacks_and_without_init_summary(
    tmp_path: Path, experiment: Experiment
) -> None:
    monitor = Monitor(chem=chemistry, rundir=tmp_path)

    for step in [0, 200, 400, 600]:
        experiment.time = step
        monitor.log(exp=experiment)

    experiment.time = 200
    monitor.revert_logs(exp=experiment)
    assert _read_session_start_steps(tmp_path) == [201, 201]
    assert _read_steps(tmp_path, "Other/Progress") == ([0, 200], [0, 200])

    for step in [300, 500]:
        experiment.time = step
        monitor.log(exp=experiment)
    assert _read_steps(tmp_path, "Other/Progress") == ([0, 200, 300, 500], [0, 200, 300, 500])

    experiment.time = 0
    monitor.revert_logs(exp=experiment)
    assert _read_session_start_steps(tmp_path) == [201, 201, 1]
    assert _read_steps(tmp_path, "Other/Progress") == ([0], [0])

    monitor.writer.close()
