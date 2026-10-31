"""Worker lifetime, actual plugin identity and independent-job equivalence."""
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.parallel import run_tasks, task_potential, execution_plan
from pyrpmd.parallel.execution import _context, ProcessBeadPotential
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.system import System


class LoggedPotential(HarmonicPotential):
    def __init__(self, log, stateful=False):
        super().__init__(k=0.2)
        self.log = log
        self.metadata = {"stateful": stateful}
        self._log("open")

    def _log(self, operation):
        with open(self.log, "a") as stream:
            stream.write(json.dumps([operation, os.getpid()]) + "\n")

    def close(self):
        self._log("close")


def _evaluate_job(task):
    spec, value, reuse = task
    with task_potential(spec, None, reuse=reuse) as potential:
        result = potential.evaluate(np.full((4, 1, 3), value))
        return os.getpid(), float(result.energy.sum())


def _logged_spec(path, **params):
    return {"provider": "python", "factory": __file__ + ":LoggedPotential",
            "params": {"log": str(path), **params}}


def test_native_dispatch_receives_actual_provider_and_closes_once():
    provider = HarmonicPotential(k=0.2)
    with patch("pyrpmd.potentials.make_potential", return_value=provider) as create:
        with patch.object(provider, "close") as close:
            with _context(execution_plan(), 0):
                for _ in range(5):
                    with task_potential({"provider": "harmonic", "k": .2}, None) as got:
                        assert got is provider
                        assert type(got) is HarmonicPotential
                    close.assert_not_called()
            assert create.call_count == close.call_count == 1


@pytest.mark.parametrize("workers,beads", [(1, 1), (2, 1), (1, 2)])
def test_reuse_real_spawn_lifecycle_and_results(tmp_path, workers, beads):
    log = tmp_path / "lifecycle.jsonl"
    jobs = [(_logged_spec(log), i / 10, True) for i in range(6)]
    parallel = {"trajectory_workers": workers, "bead_workers": beads}
    results = run_tasks(_evaluate_job, jobs, parallel)
    expected = [float(HarmonicPotential(k=.2).evaluate(
        np.full((4, 1, 3), i / 10)).energy.sum()) for i in range(6)]
    np.testing.assert_allclose([row[1] for row in results], expected, rtol=1e-15)
    events = [json.loads(line) for line in log.read_text().splitlines()]
    opened = [pid for op, pid in events if op == "open"]
    closed = [pid for op, pid in events if op == "close"]
    assert len(opened) == workers * beads
    assert sorted(opened) == sorted(closed)


@pytest.mark.parametrize("stateful,reuse", [(True, True), (False, False)])
def test_opt_out_preserves_task_local_initialization(tmp_path, stateful, reuse):
    log = tmp_path / "lifecycle.jsonl"
    jobs = [(_logged_spec(log, stateful=stateful), 0.2, reuse)] * 3
    run_tasks(_evaluate_job, jobs)
    events = [json.loads(line)[0] for line in log.read_text().splitlines()]
    assert events == ["open", "close"] * 3


def test_bead_workers_honor_actual_plugin_statefulness(tmp_path):
    log = tmp_path / "stateful.jsonl"
    jobs = [(_logged_spec(log, stateful=True), 0.2, True)] * 2
    run_tasks(_evaluate_job, jobs, {"bead_workers": 2})
    events = [json.loads(line)[0] for line in log.read_text().splitlines()]
    assert events.count("open") == events.count("close") == 4


def test_configuration_and_system_are_snapshot_keys():
    system = System(("H",), np.array([1837.]), np.zeros((1, 3)))
    spec = {"provider": "harmonic", "k": .2}
    with _context(execution_plan(), 0):
        with task_potential(spec, system) as first:
            assert type(first) is HarmonicPotential
        with task_potential(dict(spec), System.from_dict(system.to_dict())) as second:
            assert first is second
        system.masses[0] *= 2
        with task_potential(spec, system) as isotope:
            assert isotope is not first
        system.geometry[0, 0] = 1.
        with task_potential(spec, system) as geometry:
            assert geometry is not isotope
        spec["k"] = .3
        with task_potential(spec, system) as changed:
            assert changed is not geometry


def test_nested_leases_and_failed_tasks_do_not_reuse_live_or_failed_provider():
    spec = {"provider": "harmonic"}
    with _context(execution_plan(), 0):
        with task_potential(spec, None) as outer:
            with task_potential(spec, None) as inner:
                assert inner is not outer
        with pytest.raises(RuntimeError, match="failed"):
            with task_potential(spec, None) as failed:
                assert failed is outer
                raise RuntimeError("failed")
        with task_potential(spec, None) as replacement:
            assert replacement is not failed


def test_batch_failure_always_closes_cached_provider(tmp_path):
    log = tmp_path / "lifecycle.jsonl"
    def fail(task):
        with task_potential(_logged_spec(log), None):
            raise RuntimeError("task failure")
    with pytest.raises(RuntimeError, match="task failure"):
        run_tasks(fail, [None, None])
    assert [json.loads(line)[0] for line in log.read_text().splitlines()] == ["open", "close"]


def test_caught_failure_releases_resources_before_replacement(tmp_path):
    log = tmp_path / "caught.jsonl"
    spec = _logged_spec(log)
    with _context(execution_plan(), 0):
        with pytest.raises(RuntimeError):
            with task_potential(spec, None):
                raise RuntimeError("failed")
        assert [json.loads(line)[0] for line in log.read_text().splitlines()] == ["open", "close"]
        with task_potential(spec, None):
            pass
    assert [json.loads(line)[0] for line in log.read_text().splitlines()] == ["open", "close"] * 2


def test_bead_pool_creation_failure_is_cleaned_up():
    provider = ProcessBeadPotential({"provider": "free"}, None, 2)
    provider.close()
    with patch("pyrpmd.parallel.execution.make_task_potential", return_value=provider):
        with _context(execution_plan(), 0):
            with pytest.raises(RuntimeError):
                with task_potential({"provider": "free"}, None):
                    pass
