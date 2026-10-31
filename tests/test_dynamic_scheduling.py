"""Dynamic queue policy and dRPMD launch/production checkpoint boundaries."""

import copy
import pickle
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.parallel import execution_plan
from pyrpmd.parallel.execution import _dynamic_dispatch, _TASK_TAG
from pyrpmd.workflows import drpmd
from test_drpmd import config, system


class _FakeMPI:
    ANY_SOURCE = -1

    class Status:
        def Get_source(self):
            return self.source


class _Coordinator:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []

    def send(self, value, dest, tag):
        assert tag == _TASK_TAG
        self.sent.append((dest, value))

    def recv(self, source, tag, status):
        leader, packet = next(self.replies)
        status.source = leader
        return pickle.dumps(packet)


def test_queue_refills_first_available_group_and_preserves_order():
    control = _Coordinator([
        (3, (1, "one", None)), (3, (2, "two", None)),
        (1, (0, "zero", None)), (3, (3, "three", None)),
        (1, (4, "four", None)),
    ])
    values, errors = _dynamic_dispatch(control, [1, 3], 5, _FakeMPI)
    assert values == ["zero", "one", "two", "three", "four"]
    assert errors == []
    assert control.sent == [(1, 0), (3, 1), (3, 2), (3, 3),
                            (1, 4), (3, None), (1, None)]


def test_queue_failure_stops_new_dispatch_and_drains_active_groups():
    control = _Coordinator([(3, (1, None, "force failure")), (1, (0, 12, None))])
    values, errors = _dynamic_dispatch(control, [1, 3], 8, _FakeMPI)
    assert values == [12]
    assert errors == ["task 1: force failure"]
    assert control.sent == [(1, 0), (3, 1), (3, None), (1, None)]


def test_queue_empty_and_underfilled_jobs_stop_unused_groups():
    control = _Coordinator([])
    assert _dynamic_dispatch(control, [1, 3], 0, _FakeMPI) == ([], [])
    assert control.sent == [(1, None), (3, None)]
    control = _Coordinator([(1, (0, 7, None))])
    assert _dynamic_dispatch(control, [1, 3], 1, _FakeMPI) == ([7], [])
    assert control.sent == [(1, 0), (3, None), (1, None)]


def test_dynamic_topology_is_explicit_and_static_defaults_unchanged():
    assert execution_plan().scheduling == "static"
    plan = execution_plan({"backend": "mpi", "scheduling": "dynamic", "trajectory_groups": 2})
    assert plan.coordinator_ranks == 1
    for extra in ({"coordinator_ranks": 0}, {"coordinator_ranks": 2},
                  {"backend": "process"}, {"backend": "serial"}):
        with pytest.raises(ValueError):
            execution_plan({"backend": "mpi", "scheduling": "dynamic", **extra})


class _OrderedExecutor:
    """Execute phases locally with intentionally reversed child completion."""
    def map(self, function, tasks):
        tasks = list(tasks)
        if function is drpmd._collision_task:
            tasks.reverse()
        return [function(task) for task in tasks]


def _dynamic_config():
    value = config()
    value["sampling"]["parents"] = 2
    value["parallel"] = {"backend": "mpi", "scheduling": "dynamic",
                         "trajectory_groups": 2, "bead_workers": 1}
    return value


def _physical_records(result):
    return [{key: value for key, value in record.items() if key != "wall_seconds"}
            for record in result["records"]]


def _compare_states(left, right):
    for path in (Path(left) / "states").glob("*.npz"):
        with np.load(path) as a, np.load(Path(right) / "states" / path.name) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    for path in (Path(left) / "trajectories").glob("*.npz"):
        with np.load(path) as a, np.load(Path(right) / "trajectories" / path.name) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


def test_dynamic_children_retain_parent_chains_and_exact_results(tmp_path):
    dynamic = _dynamic_config()
    static = copy.deepcopy(dynamic)
    static["parallel"] = {"backend": "serial"}
    reference = drpmd.run_drpmd(static, system(), tmp_path / "static")
    actual = drpmd.run_drpmd(dynamic, system(), tmp_path / "dynamic", executor=_OrderedExecutor())
    assert _physical_records(actual) == _physical_records(reference)
    assert actual["parent_thermal_steps"] == reference["parent_thermal_steps"] == {"0": 10, "1": 8}
    _compare_states(tmp_path / "static", tmp_path / "dynamic")
    dynamic["resume"] = True
    resumed = drpmd.run_drpmd(dynamic, system(), tmp_path / "dynamic", executor=_OrderedExecutor())
    assert resumed["records"] == actual["records"]


@pytest.mark.parametrize("resume_dynamic", [False, True])
def test_queue_interruption_resumes_without_resampling(tmp_path, resume_dynamic):
    dynamic = _dynamic_config()
    reference = drpmd.run_drpmd(dynamic, system(), tmp_path / "reference", executor=_OrderedExecutor())

    class Interrupted(_OrderedExecutor):
        def map(self, function, tasks):
            if function is drpmd._collision_task:
                function(list(tasks)[2])
                raise KeyboardInterrupt("job stopped with queued collisions remaining")
            return super().map(function, tasks)

    with pytest.raises(KeyboardInterrupt):
        drpmd.run_drpmd(dynamic, system(), tmp_path / "interrupted", executor=Interrupted())
    dynamic["resume"] = True
    if not resume_dynamic:
        dynamic["parallel"] = {"backend": "serial"}
    actual = drpmd.run_drpmd(dynamic, system(), tmp_path / "interrupted", executor=_OrderedExecutor())
    assert _physical_records(actual) == _physical_records(reference)
    _compare_states(tmp_path / "reference", tmp_path / "interrupted")


def test_static_checkpoint_can_resume_with_dynamic_dispatch(tmp_path):
    dynamic = _dynamic_config()
    static = copy.deepcopy(dynamic)
    static["parallel"] = {"backend": "serial"}
    reference = drpmd.run_drpmd(static, system(), tmp_path)
    dynamic["resume"] = True
    result = drpmd.run_drpmd(dynamic, system(), tmp_path, executor=_OrderedExecutor())
    assert result["records"] == reference["records"]
