"""Global hill transactions, block synchronization and restart determinism."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.bias import MetadynamicsBias
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows.metadynamics import run_metadynamics, reanalyze_metadynamics


def system():
    return System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]]))


def config(*, grid=False):
    return {
        "task": "fes", "fes": {"method": "metadynamics"},
        "beta": 2.0, "beads": 3, "dt_au": 0.02, "seed": 321, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 3, "production_steps": 24, "sample_interval": 2,
                     "andersen_interval": 4, "replicas": 2},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "metadynamics": {"mode": "well_tempered", "sigma": [0.3], "height": 0.05,
                         "pace": 3, "bias_factor": 5.0, "multiple_walkers": True,
                         "block_steps": 5,
                         **({"grid": {"ranges": [[-2, 2]], "bins": [80]}} if grid else {})},
        "analysis": {"bins": [6], "bootstrap": 0},
        "output": {"checkpoint_steps": 5},
    }


def assert_same(left, right):
    assert left["hills"] == right["hills"]
    for a, b in zip(left["streams"], right["streams"]):
        for attr in ("q", "p"):
            np.testing.assert_array_equal(getattr(a["state"], attr), getattr(b["state"], attr))
        for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
            np.testing.assert_array_equal(a[key], b[key])
        for key in ("rng", "propagation", "thermostat_state"):
            assert a[key] == b[key]
    np.testing.assert_array_equal(left["data"]["free_energy"], right["data"]["free_energy"])


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("failure", ["step", "hill"])
def test_failed_block_restores_the_whole_ensemble_exactly(tmp_path, monkeypatch, grid, failure):
    cfg = config(grid=grid)
    reference = run_metadynamics(cfg, system(), tmp_path / "reference", return_data=True)
    step_original, deposit_original = Dynamics.step, MetadynamicsBias.deposit

    def fail_step(self, state, bias=None):
        result = step_original(self, state, bias)
        if state.step == 12:
            raise RuntimeError("injected shared step failure")
        return result

    def fail_deposit(self, value, **kwargs):
        result = deposit_original(self, value, **kwargs)
        if kwargs["step"] == 12:
            raise RuntimeError("injected shared hill failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "step", fail_step) if failure == "step" else patch.setattr(MetadynamicsBias, "deposit", fail_deposit)
        with pytest.raises(RuntimeError, match="injected"):
            run_metadynamics(cfg, system(), tmp_path / "resumed")
    checkpoint = tmp_path / "resumed/checkpoints/metadynamics_walkers.npz"
    with load_checkpoint(checkpoint) as saved:
        assert int(saved["elapsed"]) == 10
        assert saved["q"].shape[0] == 2
        assert [(h["step"], h["walker"]) for h in json.loads(str(saved["hills"]))] == [
            (6, 0), (6, 1), (9, 0), (9, 1)]
    assert list((tmp_path / "resumed/checkpoints").glob("*.npz")) == [checkpoint]
    resumed = run_metadynamics(cfg, system(), tmp_path / "resumed", return_data=True)
    assert_same(reference, resumed)


def test_shared_history_is_synchronized_and_not_mistaken_for_independent_errors(tmp_path, monkeypatch):
    cfg = config()
    result = run_metadynamics(cfg, system(), tmp_path, return_data=True)
    hills = result["hills"]
    assert len(hills) == 16
    assert hills[0]["step"] == hills[1]["step"] == 6
    # Same-block updates see only each walker's local history. The first hills
    # have the full initial height; later blocks include all earlier deposits.
    assert hills[0]["height"] == hills[1]["height"] == 0.05
    assert hills[4]["bias_before"] > hills[2]["bias_before"]
    assert result["summary"]["walkers"] == 2
    assert result["summary"]["independent_histories"] == 1
    assert np.isnan(result["data"]["replica_spread"]).all()
    assert result["summary"]["uncertainty_kind"] == "replica_spread"
    assert (tmp_path / "HILLS.csv").read_text().splitlines()[0].endswith(",walker")
    with load_checkpoint(tmp_path / "checkpoints/metadynamics_walkers.npz") as saved:
        assert "hills" in saved
        assert not any(key.startswith("hills_") for key in saved.files)

    def unavailable(*args, **kwargs):
        raise AssertionError("a completed restart must not construct the potential")

    monkeypatch.setattr("pyrpmd.workflows.metadynamics.make_task_potential", unavailable)
    again = reanalyze_metadynamics(cfg, system(), tmp_path, return_data=True)
    assert_same(result, again)
    assert_same(result, run_metadynamics(cfg, system(), tmp_path, return_data=True))


def test_process_walker_order_does_not_change_synchronization(tmp_path):
    cfg = config()
    # Fewer, longer blocks keep this scheduler test cheap while checking two
    # probability-dependent depositions within the same synchronization block.
    cfg["metadynamics"]["block_steps"] = 12
    serial = run_metadynamics(cfg, system(), tmp_path / "serial", return_data=True)
    cfg["parallel"] = {"backend": "process", "trajectory_workers": 2}
    parallel = run_metadynamics(cfg, system(), tmp_path / "process", return_data=True)
    assert_same(serial, parallel)


def test_single_shared_walker_preserves_the_independent_trajectory(tmp_path):
    cfg = config()
    cfg["sampling"]["replicas"] = 1
    shared = run_metadynamics(cfg, system(), tmp_path / "shared", return_data=True)
    cfg["metadynamics"].pop("multiple_walkers")
    cfg["metadynamics"].pop("block_steps")
    independent = run_metadynamics(cfg, system(), tmp_path / "independent", return_data=True)
    for key in ("samples", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(shared["streams"][0][key], independent["streams"][0][key])
    expected = [{key: value for key, value in hill.items() if key != "walker"} for hill in shared["hills"]]
    assert expected == independent["streams"][0]["hills"]


@pytest.mark.parametrize("damage", ["walker", "clock", "samples"])
def test_rechecksummed_global_checkpoint_is_still_validated(tmp_path, damage):
    cfg = config()
    run_metadynamics(cfg, system(), tmp_path)
    path = tmp_path / "checkpoints/metadynamics_walkers.npz"
    with load_checkpoint(path) as saved:
        payload = {key: saved[key].copy() for key in saved.files if key != "_pyrpmd_checkpoint"}
    if damage == "walker":
        hills = json.loads(str(payload["hills"]))
        hills[0]["walker"] = 1
        payload["hills"] = json.dumps(hills)
    elif damage == "clock":
        payload["elapsed"] = 27.5
    else:
        payload["sample_steps"][0] += 1
    atomic_savez(path, **payload)
    with pytest.raises(RuntimeError, match="clock"):
        reanalyze_metadynamics(cfg, system(), tmp_path)


def test_sync_interval_is_physics_significant(tmp_path):
    cfg = config()
    run_metadynamics(cfg, system(), tmp_path)
    changed = copy.deepcopy(cfg)
    changed["metadynamics"]["block_steps"] += 1
    with pytest.raises(RuntimeError, match="physics differs"):
        run_metadynamics(changed, system(), tmp_path)


@pytest.mark.parametrize("thermostat", ["nose-hoover-chain", "pile-l"])
def test_thermostat_state_survives_shared_blocks_and_restarts(tmp_path, monkeypatch, thermostat):
    cfg = config()
    cfg["sampling"].update(thermostat=thermostat,
                           thermostat_params={"tdamp": 0.4, "reset_after_equilibration": True})
    reference = run_metadynamics(cfg, system(), tmp_path / "reference", return_data=True)
    method = "sample_step" if thermostat == "pile-l" else "step"
    original = getattr(Dynamics, method)

    def fail(self, state, bias=None, **kwargs):
        result = original(self, state, bias, **kwargs)
        if state.step == 12:
            raise RuntimeError("injected bath failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, method, fail)
        with pytest.raises(RuntimeError, match="injected"):
            run_metadynamics(cfg, system(), tmp_path / "resumed")
    assert_same(reference, run_metadynamics(cfg, system(), tmp_path / "resumed", return_data=True))


def test_zero_equilibration_fixed_slab_and_completed_walker_failure(tmp_path, monkeypatch):
    from pyrpmd.workflows import metadynamics

    cfg = config(grid=True)
    cfg["potential"] = {"provider": "free"}
    cfg["sampling"].update(equilibration_steps=0, thermostat="nose-hoover-chain",
                           thermostat_params={"tdamp": 0.4, "reset_after_equilibration": True})
    slab = System(("H", "Cu"), np.array([1.0, 30.0]),
                  np.array([[0.1, -0.05, 0.0], [0.0, 0.0, 0.0]]),
                  cell=np.diag([8.0, 8.0, 20.0]), pbc=[True, True, False], fixed_atoms=[1])
    reference = run_metadynamics(cfg, slab, tmp_path / "reference", return_data=True)
    original = metadynamics._sample_replica

    def fail_after_second_walker(task):
        result = original(task)
        if task["replica"] == 1 and task["stop"] == 15:
            raise RuntimeError("injected after both walkers advanced")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(metadynamics, "_sample_replica", fail_after_second_walker)
        with pytest.raises(RuntimeError, match="both walkers"):
            run_metadynamics(cfg, slab, tmp_path / "resumed")
    with load_checkpoint(tmp_path / "resumed/checkpoints/metadynamics_walkers.npz") as saved:
        assert int(saved["elapsed"]) == 10
    resumed = run_metadynamics(cfg, slab, tmp_path / "resumed", return_data=True)
    assert_same(reference, resumed)
    for walker in resumed["streams"]:
        np.testing.assert_array_equal(walker["state"].q[:, 1], 0.0)
        np.testing.assert_array_equal(walker["state"].p[:, 1], 0.0)
    with load_checkpoint(tmp_path / "resumed/checkpoints/metadynamics_walkers.npz") as saved:
        resumed_grid = saved["grid_values"].copy()
    with load_checkpoint(tmp_path / "reference/checkpoints/metadynamics_walkers.npz") as saved:
        np.testing.assert_array_equal(resumed_grid, saved["grid_values"])


@pytest.mark.parametrize("bad", ["duplicate", "negative", "boolean", "time"])
def test_shared_hill_clock_validation_keeps_physical_times(tmp_path, bad):
    cfg = config()
    result = run_metadynamics(cfg, system(), tmp_path, return_data=True)
    hills = copy.deepcopy(result["hills"])
    if bad == "duplicate":
        hills[1]["walker"] = 0
    elif bad == "negative":
        hills[0]["walker"] = -1
    elif bad == "boolean":
        hills[0]["walker"] = True
    else:
        hills[1]["time"] += 1e-6
    from pyrpmd.workflows.metadynamics import resolve_metadynamics
    cv, _, options = resolve_metadynamics(cfg, system())
    with pytest.raises(ValueError, match="walker|time"):
        MetadynamicsBias(cv, options["sigma"], options["height"], cfg["beta"],
                         bias_factor=options["bias_factor"], hills=hills)
