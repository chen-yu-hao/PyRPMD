"""Public process scheduling, collision granularity and restart invariants."""

import copy
import os
from pathlib import Path

import pytest

from pyrpmd.parallel import execution_plan, run_tasks
from pyrpmd.parallel import execution


def _dynamic(workers=2):
    return {"backend": "process", "mode": "serial", "scheduling": "dynamic",
            "trajectory_workers": workers, "bead_workers": 1, "threads_per_rank": 1}


def _probe(index):
    from threadpoolctl import threadpool_info
    context = execution._CONTEXT
    return (index, context.plan.scheduling, context.comm,
            os.environ["OPENBLAS_NUM_THREADS"],
            [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"])


@pytest.mark.parametrize("backend", ["serial", "process"])
def test_local_dynamic_plan_has_no_coordinator(backend):
    plan = execution_plan({**_dynamic(), "backend": backend})
    assert plan.backend == "process"
    assert plan.coordinator_ranks == 0
    assert execution_plan().scheduling == "static"


@pytest.mark.parametrize("extra", [
    {"coordinator_ranks": 1}, {"bead_workers": 2},
    {"mode": "bead"}, {"mode": "gpu", "devices": [0]},
])
def test_local_dynamic_rejects_unsupported_placement(extra):
    with pytest.raises(ValueError, match="Dynamic local scheduling"):
        execution_plan({**_dynamic(), **extra})


def test_normalized_input_uses_backend_specific_coordinator_default():
    from pyrpmd.input_runtime import resolve_parallel
    for backend, count in (("process", 0), ("mpi", 1)):
        result = resolve_parallel({"parallel": {**_dynamic(), "backend": backend}}, {})
        assert result["coordinator_ranks"] == count


def test_process_workers_keep_context_and_single_thread_blas():
    values = run_tasks(_probe, range(5), _dynamic())
    assert [row[0] for row in values] == list(range(5))
    for _, scheduling, comm, threads, pools in values:
        assert scheduling == "dynamic" and comm is None and threads == "1"
        assert all(count == 1 for count in pools)


def test_single_worker_restores_parent_environment(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    previous = execution._CONTEXT
    assert run_tasks(_probe, [0], _dynamic(1))[0][3] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert execution._CONTEXT is previous
    assert run_tasks(_probe, [], _dynamic()) == []


def test_thermal_dynamic_process_preserves_parent_chains_and_resume(tmp_path):
    from pyrpmd.workflows import drpmd
    from test_drpmd import config, system
    from test_dynamic_scheduling import _compare_states, _physical_records

    static = config()
    static["sampling"]["parents"] = 2
    dynamic = copy.deepcopy(static)
    dynamic["parallel"] = _dynamic()
    first = drpmd.run_drpmd(static, system(), tmp_path / "static")
    second = drpmd.run_drpmd(dynamic, system(), tmp_path / "dynamic")
    assert _physical_records(first) == _physical_records(second)
    assert first["parent_thermal_steps"] == second["parent_thermal_steps"]
    _compare_states(tmp_path / "static", tmp_path / "dynamic")
    dynamic["resume"] = True
    dynamic["parallel"] = {"backend": "serial"}
    resumed = drpmd.run_drpmd(dynamic, system(), tmp_path / "dynamic")
    assert resumed["records"] == second["records"]
    static["resume"], static["parallel"] = True, _dynamic(3)
    assert drpmd.run_drpmd(static, system(), tmp_path / "static")["records"] == first["records"]


@pytest.mark.parametrize("method,beads", [("qct", 1), ("ne_rpmd", 8)])
def test_selected_process_results_and_changed_worker_resume(tmp_path, method, beads):
    from pyrpmd.workflows import drpmd
    from test_state_selected_workflow import config, system, assert_same_arrays

    static = config(method, beads)
    # A single logical group must not cap independent collision concurrency.
    static["sampling"]["parents"] = 1
    dynamic = copy.deepcopy(static)
    dynamic["parallel"] = _dynamic()
    first = drpmd.run_drpmd(static, system(), tmp_path / "static")
    second = drpmd.run_drpmd(dynamic, system(), tmp_path / "dynamic")
    assert first["counts"] == second["counts"]
    assert [row["trajectory_id"] for row in second["records"]] == [0, 1, 2]
    for i in range(3):
        initial = f"states/initial-{i:08d}.npz"
        assert_same_arrays(tmp_path / "static" / initial, tmp_path / "dynamic" / initial,
                           ("q", "p", "prepared_q", "prepared_p"))
        trace = f"trajectories/trajectory-{i:08d}.npz"
        assert_same_arrays(tmp_path / "static" / trace, tmp_path / "dynamic" / trace)
    dynamic["resume"], dynamic["parallel"] = True, _dynamic(3)
    assert drpmd.run_drpmd(dynamic, system(), tmp_path / "dynamic")["records"] == second["records"]
    dynamic["parallel"] = {"backend": "serial"}
    assert drpmd.run_drpmd(dynamic, system(), tmp_path / "dynamic")["records"] == second["records"]


@pytest.mark.parametrize("backend", ["process", "mpi"])
@pytest.mark.parametrize("method,beads", [("qct", 1), ("ne_rpmd", 8)])
def test_selected_dynamic_dispatch_is_per_trajectory(tmp_path, backend, method, beads):
    from pyrpmd.workflows import drpmd
    from test_state_selected_workflow import config, system

    class Inspect:
        def map(self, function, tasks):
            tasks = list(tasks)
            assert function is drpmd._initial_ensemble_task
            assert [task["trajectory_ids"] for task in tasks] == [[0], [1], [2]]
            assert all(task["parent_id"] == 0 for task in tasks)
            return [function(task) for task in reversed(tasks)]

    value = config(method, beads)
    value["sampling"]["parents"] = 1
    value["parallel"] = {**_dynamic(), "backend": backend}
    result = drpmd.run_drpmd(value, system(), tmp_path, executor=Inspect())
    assert [row["trajectory_id"] for row in result["records"]] == [0, 1, 2]


@pytest.mark.parametrize("resume_dynamic", [False, True])
def test_thermal_process_phase_interruption_does_not_resample(tmp_path, resume_dynamic):
    from pyrpmd.workflows import drpmd
    from test_drpmd import config, system
    from test_dynamic_scheduling import _compare_states, _physical_records, _OrderedExecutor

    value = config()
    value["sampling"]["parents"] = 2
    value["parallel"] = _dynamic()
    reference = drpmd.run_drpmd(value, system(), tmp_path / "reference")

    class Interrupted(_OrderedExecutor):
        def map(self, function, tasks):
            if function is drpmd._collision_task:
                function(list(tasks)[2])
                raise KeyboardInterrupt("stop between sampled collisions")
            return super().map(function, tasks)

    with pytest.raises(KeyboardInterrupt):
        drpmd.run_drpmd(value, system(), tmp_path / "interrupted", executor=Interrupted())
    value["resume"] = True
    value["parallel"] = _dynamic(3) if resume_dynamic else {"backend": "serial"}
    actual = drpmd.run_drpmd(value, system(), tmp_path / "interrupted")
    assert _physical_records(actual) == _physical_records(reference)
    _compare_states(tmp_path / "reference", tmp_path / "interrupted")
