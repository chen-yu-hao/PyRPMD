"""Actual dynamic MPI checks: mpiexec -n 5 python tests/mpi_dynamic_integration.py.

One coordinator plus G=2 trajectory groups with B=2 bead ranks. The rendezvous
proves that a free group takes a third task before the slow first task finishes;
it does not rely on noisy wall-clock speed ratios.
"""

import copy
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

from pyrpmd.parallel import make_task_potential, root_call, run_tasks, task_potential
from pyrpmd.potentials import PESResult
from pyrpmd.workflows.drpmd import run_drpmd
from mpi_integration import SYSTEM, config


PARALLEL = {"backend": "mpi", "scheduling": "dynamic", "coordinator_ranks": 1,
            "trajectory_groups": 2, "bead_workers": 2, "mode": "bead"}


class TrackingPotential:
    def __init__(self, directory):
        self.directory = directory

    def evaluate(self, q):
        return PESResult(np.sum(q * q, axis=(1, 2)), -2.0 * q)

    def close(self):
        with (Path(self.directory) / f"closed-{MPI.COMM_WORLD.Get_rank()}").open("a") as out:
            out.write("closed\n")


def tracking_factory(directory):
    return TrackingPotential(directory)


class CachedTrackingPotential(TrackingPotential):
    def __init__(self, directory, stateful=False, fail_close=False):
        super().__init__(directory)
        self.metadata = {"stateful": stateful}
        self.fail_close = fail_close
        self._record("open")

    def _record(self, operation):
        with (Path(self.directory) / f"lifecycle-{MPI.COMM_WORLD.Get_rank()}").open("a") as out:
            out.write(operation + "\n")

    def close(self):
        self._record("close")
        if self.fail_close and MPI.COMM_WORLD.Get_rank() == 2:
            raise RuntimeError("controlled cached close failure")


def cached_tracking_factory(directory, stateful=False, fail_close=False):
    return CachedTrackingPotential(directory, stateful, fail_close)


def cached_task(task):
    from pyrpmd.parallel import execution

    # Any directly constructed provider left by the preceding task must already
    # be closed; only leases in the worker cache may survive a task boundary.
    context = execution._CONTEXT
    cached = {id(potential) for potential in context.potentials.values()}
    assert all(id(potential) in cached for potential in context.providers)
    spec = {"provider": "python", "factory": "mpi_dynamic_integration:cached_tracking_factory",
            "params": {"directory": str(Path(task["directory"]) / "cached"),
                       "stateful": task.get("stateful", False),
                       "fail_close": task.get("fail_close", False)}}
    q = np.full((3, 2, 3), task["index"], dtype=float)
    with task_potential(spec, SYSTEM, reuse=task.get("reuse", True)) as potential:
        result = potential.evaluate(q)
        np.testing.assert_allclose(result.forces, -2 * q)
        if task.get("failure") == "lease" and task["index"] == 0:
            raise ValueError("controlled cached task failure")
    if task.get("leave_owned", False):
        owned_spec = copy.deepcopy(spec)
        owned_spec["params"]["directory"] = str(Path(task["directory"]) / "owned")
        # Match the executor's documented cleanup when a task omits close().
        owned = make_task_potential(owned_spec, SYSTEM)
        owned.evaluate(q)
    if task.get("failure") == "after-lease" and task["index"] == 0:
        raise ValueError("controlled cached task failure")
    return float(result.energy.sum()), MPI.COMM_WORLD.Get_rank()


def check_dynamic_cache(directory):
    for name, options in (("reuse", {}), ("stateful", {"stateful": True}),
                          ("opt-out", {"reuse": False}),
                          ("owned", {"leave_owned": True})):
        path = Path(directory) / name
        root_call(lambda: (path / "cached").mkdir(parents=True))
        root_call(lambda: (path / "owned").mkdir())
        tasks = [dict(index=i, directory=str(path), **options) for i in range(6)]
        results = run_tasks(cached_task, tasks, PARALLEL)

        def verify():
            np.testing.assert_allclose([row[0] for row in results], 18 * np.arange(6.0) ** 2)
            reusable = not options.get("stateful", False) and options.get("reuse", True)
            for rank in range(1, 5):
                leader = rank if rank % 2 else rank - 1
                assigned = sum(row[1] == leader for row in results)
                expected = 1 if reusable else assigned
                events = (path / "cached" / f"lifecycle-{rank}").read_text().splitlines()
                assert events == ["open", "close"] * expected, (name, rank, events)
                if options.get("leave_owned", False):
                    owned = (path / "owned" / f"lifecycle-{rank}").read_text().splitlines()
                    assert owned == ["open", "close"] * assigned

        root_call(verify)

    for failure, options, message in (
        ("lease", {"failure": "lease"}, "controlled cached task failure"),
        ("after-lease", {"failure": "after-lease"}, "controlled cached task failure"),
        ("close", {"fail_close": True}, "controlled cached close failure"),
    ):
        path = Path(directory) / ("failed-" + failure)
        root_call(lambda: (path / "cached").mkdir(parents=True))
        tasks = [dict(index=i, directory=str(path), **options) for i in range(2)]
        try:
            run_tasks(cached_task, tasks, PARALLEL)
        except RuntimeError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError("Cached provider failure was not collective")
        root_call(lambda: check(
            all((path / "cached" / f"lifecycle-{rank}").read_text().splitlines()
                == ["open", "close"] for rank in range(1, 5)),
            "A failed dynamic task must release every bead provider exactly once"))


def evaluate(task):
    index, directory = task
    if index == 0:
        deadline = time.monotonic() + 20
        while not (Path(directory) / "third-task-started").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Idle group did not receive the third task")
            time.sleep(0.005)
    elif index == 2:
        (Path(directory) / "third-task-started").touch()
    potential = make_task_potential({"provider": "harmonic", "k": 2.0}, SYSTEM)
    try:
        q = np.full((3, 2, 3), index, dtype=float)
        result = potential.evaluate(q)
        np.testing.assert_allclose(result.forces, -2 * q)
        return float(result.energy.sum()), MPI.COMM_WORLD.Get_rank()
    finally:
        potential.close()
        from pyrpmd.parallel import execution
        assert not execution._CONTEXT.providers, "Closed models must not accumulate in the task context"


def failing_task(task):
    index, directory = task
    potential = make_task_potential(
        {"provider": "python", "factory": "mpi_dynamic_integration:tracking_factory",
         "params": {"directory": directory}}, SYSTEM)
    make_task_potential(
        {"provider": "python", "factory": "mpi_dynamic_integration:tracking_factory",
         "params": {"directory": directory}}, SYSTEM)
    potential.evaluate(np.zeros((3, 2, 3)))
    # Deliberately omit close(): the executor must clean both bead ranks.
    if index == 0:
        raise ValueError("controlled dynamic failure")
    return index


def unavailable_task(task):
    make_task_potential({"provider": "missing-provider"}, SYSTEM)


def unpicklable_task(task):
    return lambda: task


def mock_gpu_route(task):
    class Provider:
        def __init__(self, device):
            self.metadata = {"execution_device": device}
    with patch("pyrpmd.potentials.make_potential",
               side_effect=lambda spec, system: Provider(spec["device"])):
        return make_task_potential({"provider": "torch"}, SYSTEM).metadata["execution_device"]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_trajectories(left, right):
    for folder in ("states", "trajectories"):
        paths = sorted((Path(left) / folder).glob("*.npz"))
        assert paths
        for path in paths:
            with np.load(path) as a, np.load(Path(right) / folder / path.name) as b:
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


def check_selected_ensembles(directory):
    """Independent preparations must bypass the thermal-parent queue/ledger."""
    from test_state_selected_workflow import config as selected_config, system as selected_system

    for method, beads in (("qct", 1), ("ne_rpmd", 8)):
        c = selected_config(method, beads)
        c["sampling"]["parents"] = 2
        c["parallel"] = PARALLEL
        dynamic_output = Path(directory) / (method + "-dynamic")
        first = run_drpmd(c, selected_system(), dynamic_output)
        assert first["initialization_mode"] == "independent"
        assert first["parent_thermal_steps"] == {"0": 0, "1": 0}
        assert first["counts"]["unresolved"] == c["collision"]["trajectories"]
        c["resume"] = True
        second = run_drpmd(c, selected_system(), dynamic_output)
        assert second["records"] == first["records"]

        static = copy.deepcopy(c)
        static.pop("resume")
        static["parallel"] = {"backend": "mpi", "trajectory_groups": 5, "bead_workers": 1}
        static_output = Path(directory) / (method + "-static")
        reference = run_drpmd(static, selected_system(), static_output)
        keys = set(first["records"][0]) - {"wall_seconds"}
        assert [{k: r[k] for k in keys} for r in first["records"]] == [
            {k: r[k] for k in keys} for r in reference["records"]]
        root_call(compare_trajectories, static_output, dynamic_output)
        # Scheduling groups are not independent thermal parents and may change
        # on restart without changing any prepared initial condition.
        static["sampling"]["parents"] = 3
        static["resume"] = True
        switched = run_drpmd(static, selected_system(), dynamic_output)
        assert switched["records"] == first["records"]


def main():
    directory = root_call(tempfile.mkdtemp, prefix="pyrpmd-dynamic-mpi-")
    try:
        tasks = [(i, directory) for i in range(7)]
        result = run_tasks(evaluate, tasks, PARALLEL)
        root_call(lambda: np.testing.assert_allclose([r[0] for r in result], 18 * np.arange(7.0) ** 2))
        root_call(lambda: check(result[0][1] != result[2][1], "Third task must run on the free group"))

        empty = run_tasks(evaluate, [], PARALLEL)
        root_call(lambda: check(empty == [], "Empty queue must finish"))
        one = run_tasks(evaluate, [(1, directory)], PARALLEL)
        root_call(lambda: check(one[0][0] == 18, "Underfilled queue must finish"))

        for function, message in ((failing_task, "controlled dynamic failure"),
                                  (unavailable_task, "PES initialization failed"),
                                  (unpicklable_task, "result serialization failed")):
            try:
                run_tasks(function, tasks[:2], PARALLEL)
            except RuntimeError as exc:
                assert message.lower() in str(exc).lower(), str(exc)
            else:
                raise AssertionError("Dynamic task error was swallowed")
        root_call(lambda: check(all((Path(directory) / f"closed-{rank}").read_text().splitlines()
                                    == ["closed", "closed"] for rank in range(1, 5)),
                                "All bead resources must close exactly once"))
        routing = run_tasks(mock_gpu_route, range(4), {
            "backend": "mpi", "scheduling": "dynamic", "mode": "gpu",
            "trajectory_groups": 4, "devices": [0, 1], "trajectories_per_gpu": 2,
        })
        root_call(lambda: check(routing == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"],
                                "Coordinator must not consume a GPU slot"))
        check_dynamic_cache(directory)

        c = config()
        c["sampling"]["parents"] = 2
        c["parallel"] = PARALLEL
        first = run_drpmd(c, SYSTEM, Path(directory) / "dynamic")
        assert first["parent_thermal_steps"] == {"0": 10, "1": 8}
        c["resume"] = True
        second = run_drpmd(c, SYSTEM, Path(directory) / "dynamic")
        assert second["records"] == first["records"]
        static = copy.deepcopy(c)
        static.pop("resume")
        static["parallel"] = {"backend": "mpi", "trajectory_groups": 5, "bead_workers": 1}
        reference = run_drpmd(static, SYSTEM, Path(directory) / "static")
        keys = set(first["records"][0]) - {"wall_seconds"}
        assert [{k: r[k] for k in keys} for r in first["records"]] == [
            {k: r[k] for k in keys} for r in reference["records"]]
        root_call(compare_trajectories, Path(directory) / "static", Path(directory) / "dynamic")
        static["resume"] = True
        switched = run_drpmd(static, SYSTEM, Path(directory) / "dynamic")
        assert switched["records"] == first["records"]
        check_selected_ensembles(directory)
        root_call(lambda: print("Dynamic MPI C=1 G=2 B=2: availability dispatch, ordered forces, "
                                "failure drain/cleanup, empty/underfilled queues, serialization failure, "
                                "worker cache reuse/opt-out/statefulness and collective cache cleanup, "
                                "mock GPU placement, exact thermal/selected dRPMD static equivalence "
                                "and resume passed", flush=True))
    finally:
        root_call(shutil.rmtree, directory)


if __name__ == "__main__":
    main()
