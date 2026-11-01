"""Shared-bias numerical forces, synchronized failure recovery and CLI checks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.bias import MetadynamicsBias
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.cv import Cartesian, CVStack, Distance
from pyrpmd.dynamics import Dynamics
from pyrpmd.shared_metadynamics import SharedMetadynamicsBias
from pyrpmd.system import System
from pyrpmd.workflows.metadynamics import run_metadynamics, reanalyze_metadynamics
import pyrpmd.workflows.shared_metadynamics as shared


def system():
    return System(("H",), np.array([2.]), np.array([[.1, -.05, 0.]]))


def config(mode="well_tempered", thermostat="andersen", store=True):
    result = {
        "task": "metadynamics", "beta": 2., "beads": 3, "dt_au": .02,
        "seed": 31, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .2}},
        "sampling": {"equilibration_steps": 2, "production_steps": 12, "sample_interval": 2,
                     "andersen_interval": 4, "replicas": 3, "thermostat": thermostat},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis} for axis in ("x", "y")],
        "metadynamics": {"shared_walkers": True, "mode": mode, "sigma": [.3, .4],
                         "height": .05, "pace": 2,
                         **({"bias_factor": 5.} if mode == "well_tempered" else {})},
        "analysis": {"bins": [5, 6], "bootstrap": 0},
        "output": {"checkpoint_steps": 5, "store_configurations": store},
    }
    if thermostat == "pile-l":
        result["sampling"]["gamma_centroid"] = .3
    return result


def read(path):
    with load_checkpoint(Path(path)/"checkpoints"/"shared_metadynamics.npz",
                          expected_kind=shared.KIND, require_integrity=True) as archive:
        return {key: archive[key].copy() for key in archive.files}


def assert_identical_checkpoints(first, second):
    a, b = read(first), read(second)
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_well_tempered_batch_uses_same_predeposition_history_without_bead_factor():
    cv = CVStack([Cartesian(0, "x")])
    bias = SharedMetadynamicsBias(cv, [.5], .4, 2., walkers=3, bias_factor=3.)
    first = bias.deposit_walkers([[0.], [0.], [0.]], step=1, time=.1)
    np.testing.assert_array_equal([h["height"] for h in first], [.4]*3)
    np.testing.assert_array_equal([h["bias_before"] for h in first], [0.]*3)
    second = bias.deposit_walkers([[0.], [0.], [0.]], step=2, time=.2)
    expected_height = .4*np.exp(-2.*(3*.4)/(3.-1.))
    np.testing.assert_allclose([h["height"] for h in second], expected_height, rtol=2e-15)
    np.testing.assert_allclose([h["bias_before"] for h in second], 1.2, rtol=2e-15)
    for beads in (1, 8):
        energy, force = bias.energy_force(np.zeros((beads, 1, 3)))
        assert energy/beads == pytest.approx(1.2+3*expected_height)
        np.testing.assert_array_equal(force, 0.)


def test_shared_bias_is_independent_of_walker_iteration_order():
    cv = CVStack([Cartesian(0, "x")])
    a = SharedMetadynamicsBias(cv, [.5], .2, 3., walkers=3, bias_factor=4.)
    b = SharedMetadynamicsBias(cv, [.5], .2, 3., walkers=3, bias_factor=4.)
    centers = np.array([[-.5], [.1], [.8]])
    order = np.array([2, 0, 1])
    for step in range(1, 4):
        a.deposit_walkers(centers+.1*step, step=step, time=.1*step)
        b.deposit_walkers(centers[order]+.1*step, step=step, time=.1*step)
    grid = np.linspace(-1., 1., 19)[:, None]
    for actual, expected in zip(a.physical_energy_gradient(grid), b.physical_energy_gradient(grid)):
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize("beads", [1, 4])
def test_shared_full_ring_force_matches_finite_difference_of_sum_bead_bias(beads):
    cv = CVStack([Distance([0, 1], estimator="bead_average"), Cartesian(1, "y")])
    bias = SharedMetadynamicsBias(cv, [.45, .25], .03, 2., walkers=2, bias_factor=6.)
    bias.deposit_walkers([[1.2, .1], [1.4, -.2]], step=2, time=.1)
    bias.deposit_walkers([[1.1, -.1], [1.5, .2]], step=4, time=.2)
    rng = np.random.default_rng(6)
    q = np.tile([[[-.5, .1, .2], [.8, -.2, .3]]], (beads, 1, 1))
    q += rng.normal(scale=.07, size=q.shape)
    _, force = bias.energy_force(q)
    numerical = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = -(bias.energy_force(plus)[0]-bias.energy_force(minus)[0])/2e-6
    np.testing.assert_allclose(force, numerical, rtol=2e-7, atol=1e-10)


def test_shared_history_requires_complete_ordered_batches_and_legacy_clocks_stay_strict():
    cv = CVStack([Cartesian(0, "x")])
    bias = SharedMetadynamicsBias(cv, [.5], .1, 2., walkers=2)
    bias.deposit_walkers([[0.], [1.]], step=2, time=.2)
    history = bias.hills
    with pytest.raises(ValueError, match="complete walker"):
        bias.truncate_history(1)
    with pytest.raises(ValueError, match="deposit_walkers"):
        bias.deposit([0.], step=3, time=.3)
    with pytest.raises(ValueError, match="complete walker"):
        SharedMetadynamicsBias(cv, [.5], .1, 2., walkers=2, hills=history[:-1])
    swapped = [history[1], history[0]]
    with pytest.raises(ValueError, match="clocks"):
        SharedMetadynamicsBias(cv, [.5], .1, 2., walkers=2, hills=swapped)
    assert bias.hills == history
    ordinary = MetadynamicsBias(cv, [.5], .1, 2.)
    ordinary.deposit([0.], step=2, time=.2)
    with pytest.raises(ValueError, match="strictly increase"):
        ordinary.deposit([1.], step=2, time=.2)
    assert ordinary.hill_count == 1


def test_shared_batch_append_failure_rolls_back_numeric_force_buffers(monkeypatch):
    cv = CVStack([Cartesian(0, "x")])
    bias = SharedMetadynamicsBias(cv, [.5], .1, 2., walkers=3)
    bias.deposit_walkers([[0.], [.2], [-.3]], step=1, time=.1)
    history = bias.hills
    grid = np.linspace(-1., 1., 9)[:, None]
    before = bias.physical_energy_gradient(grid)
    original = bias._append
    def failing(hill):
        original(hill)
        if hill["walker"] == 1:
            raise RuntimeError("partial batch")
    monkeypatch.setattr(bias, "_append", failing)
    with pytest.raises(RuntimeError, match="partial batch"):
        bias.deposit_walkers([[.1], [.3], [.5]], step=2, time=.2)
    assert bias.hills == history
    for actual, expected in zip(bias.physical_energy_gradient(grid), before):
        np.testing.assert_array_equal(actual, expected)


def test_all_walkers_see_frozen_history_until_deposition_barrier(tmp_path, monkeypatch):
    cfg = config()
    original = Dynamics.step
    counts = []
    def recording(self, state, bias=None):
        shared_bias = bias.biases[0]
        counts.append((state.step, shared_bias.hill_count))
        return original(self, state, bias)
    monkeypatch.setattr(Dynamics, "step", recording)
    result = run_metadynamics(cfg, system(), tmp_path, return_data=True)
    for start in range(0, len(counts), 3):
        step_counts = counts[start:start+3]
        assert len(set(step_counts)) == 1
        old_step, hills = step_counts[0]
        assert hills == 3*(max(0, old_step-2)//2)
    assert result["summary"]["hills"] == 18
    assert result["summary"]["independent_histories"] == 1
    assert result["summary"]["uncertainty"] is None
    assert np.isnan(result["data"]["standard_error"]).all()
    assert result["data"]["bias_per_replica"].shape == (1, 5, 6)


@pytest.mark.parametrize("mode", ["standard", "well_tempered"])
@pytest.mark.parametrize("failure", ["walker", "hill"])
@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
def test_synchronized_failure_resumes_bitwise_identically(tmp_path, monkeypatch, mode, failure, thermostat):
    cfg = config(mode, thermostat)
    run_metadynamics(cfg, system(), tmp_path/"reference")
    with monkeypatch.context() as patch:
        if failure == "walker":
            method = "step" if thermostat == "andersen" else "sample_step"
            original = getattr(Dynamics, method)
            calls = 0
            def fail_after_second_walker(self, state, *args, **kwargs):
                nonlocal calls
                calls += 1
                result = original(self, state, *args, **kwargs)
                if state.step == 6 and calls % 3 == 2:
                    raise RuntimeError("second walker failed")
                return result
            patch.setattr(Dynamics, method, fail_after_second_walker)
        else:
            original = SharedMetadynamicsBias._append
            def fail_after_second_hill(self, hill):
                original(self, hill)
                if hill["step"] == 6 and hill["walker"] == 1:
                    raise RuntimeError("second hill failed")
            patch.setattr(SharedMetadynamicsBias, "_append", fail_after_second_hill)
        with pytest.raises(RuntimeError, match="second (walker|hill) failed"):
            run_metadynamics(cfg, system(), tmp_path/"interrupted")
    cp = read(tmp_path/"interrupted")
    assert int(cp["elapsed"]) == 5
    hills = json.loads(str(cp["hills"]))
    assert [h["step"] for h in hills] == [4]*3
    assert [h["walker"] for h in hills] == [0, 1, 2]
    assert cp["q"].shape == (3, 3, 1, 3)
    np.testing.assert_array_equal(cp["sample_steps"], [4])
    partial = reanalyze_metadynamics(cfg, system(), tmp_path/"interrupted")
    assert partial["status"] == "partial_bias_estimate"
    assert partial["samples"] == 3
    run_metadynamics(cfg, system(), tmp_path/"interrupted")
    assert_identical_checkpoints(tmp_path/"reference", tmp_path/"interrupted")


def test_completed_resume_and_offline_analysis_never_deposit_duplicate_hills(tmp_path, monkeypatch):
    cfg = config(store=False)
    reference = run_metadynamics(cfg, system(), tmp_path, return_data=True)
    before = read(tmp_path)
    run_metadynamics(cfg, system(), tmp_path)
    after = read(tmp_path)
    for key in before:
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    def no_potential(*args, **kwargs):
        raise RuntimeError("offline analysis must not construct a PES")
    monkeypatch.setattr(shared, "make_potential", no_potential)
    restored = reanalyze_metadynamics(cfg, system(), tmp_path, return_data=True)
    for key in ("free_energy", "raw_count", "mask", "bias_per_replica"):
        np.testing.assert_array_equal(reference["data"][key], restored["data"][key])
    cfg["analysis"]["bins"] = [3, 4]
    resized = reanalyze_metadynamics(cfg, system(), tmp_path, return_data=True)
    assert resized["data"]["free_energy"].shape == (3, 4)
    assert resized["summary"]["independent_histories"] == 1


def test_saved_sample_bias_excludes_entire_simultaneous_deposition(tmp_path):
    cfg = config()
    run_metadynamics(cfg, system(), tmp_path)
    saved = read(tmp_path)
    hills = json.loads(str(saved["hills"]))
    sigma = np.asarray(cfg["metadynamics"]["sigma"])
    for step, values, actual in zip(saved["sample_steps"], saved["samples"], saved["sample_bias"]):
        expected = np.zeros(3)
        for hill in hills:
            if hill["step"] < step:
                delta = (values-np.asarray(hill["center"]))/sigma
                expected += hill["height"]*np.exp(-.5*np.sum(delta*delta, axis=1))
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-16)
    # Coordinate-dependent observations correspond to each stored walker frame.
    np.testing.assert_allclose(saved["configurations"].mean(axis=2)[:, :, 0, :2], saved["samples"], atol=1e-15)


@pytest.mark.parametrize("field,value,match", [
    ("elapsed", 14.0, "integer scalar"),
    ("elapsed", True, "integer scalar"),
    ("time", 1.3, "time disagrees"),
    ("sample_steps", np.array([4, 6, 8, 10, 12, 13]), "clock"),
    ("sample_bias", np.zeros((6, 2)), "sampled bias"),
])
def test_reanalysis_rejects_integrity_valid_but_inconsistent_checkpoint(tmp_path, field, value, match):
    cfg = config()
    run_metadynamics(cfg, system(), tmp_path)
    arrays = read(tmp_path)
    arrays[field] = np.asarray(value)
    atomic_savez(tmp_path/"checkpoints"/"shared_metadynamics.npz", kind=shared.KIND, **arrays)
    with pytest.raises(ValueError, match=match):
        reanalyze_metadynamics(cfg, system(), tmp_path)


def test_shared_input_and_cli_run_resume_and_offline_analyze(tmp_path, capsys):
    from pyrpmd.cli import main
    from pyrpmd.input import JobSpec

    source = tmp_path/"shared.inp"
    source.write_text('''task = "metadynamics"
temperature = "300 K"
beads = 2
seed = 7
native = false
[system]
symbols = ["H"]
geometry = [[0.1, 0.0, 0.0]]
[potential]
provider = "builtin"
name = "harmonic"
params = {k = 0.2}
[dynamics]
timestep = "0.001 fs"
[sampling]
equilibration_steps = 2
production_steps = 12
sample_interval = 2
andersen_interval = 4
replicas = 2
[[cvs]]
name = "x"
kind = "cartesian"
atom = 0
axis = "x"
[metadynamics]
shared_walkers = true
mode = "well_tempered"
sigma = ["0.3 bohr"]
height = "0.01 kcal/mol"
pace = 2
bias_factor = 5.0
[analysis]
bins = [6]
bootstrap = 0
''')
    cfg, _ = JobSpec.from_file(source).resolve()
    assert cfg["metadynamics"]["shared_walkers"] is True
    target = tmp_path/"cli-run"
    assert main(["run", str(source), "--output", str(target)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["shared_bias"] is True and result["walkers"] == 2
    original = read(target)
    assert main(["resume", str(target)]) == 0
    capsys.readouterr()
    resumed = read(target)
    for key in original:
        np.testing.assert_array_equal(original[key], resumed[key], err_msg=key)
    assert main(["analyze", str(target), "--recompute", "--bins", "4"]) == 0
    analyzed = json.loads(capsys.readouterr().out)
    assert analyzed["independent_histories"] == 1
    assert analyzed["hills"] == 12
    assert analyzed["uncertainty"] is None


# Block-sharing walkers keep locally tempered heights within each block.
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.workflows.metadynamics import run_metadynamics, reanalyze_metadynamics


def multiple_system():
    return System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]]))


def multiple_config(*, grid=False):
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
        assert ("sample_configurations" in a) == ("sample_configurations" in b)
        if "sample_configurations" in a:
            np.testing.assert_array_equal(a["sample_configurations"], b["sample_configurations"])
        for key in ("rng", "propagation", "thermostat_state"):
            assert a[key] == b[key]
        assert a.get("operator_state") == b.get("operator_state")
    np.testing.assert_array_equal(left["data"]["free_energy"], right["data"]["free_energy"])


@pytest.mark.parametrize("grid", [False, True])
@pytest.mark.parametrize("failure", ["step", "hill"])
@pytest.mark.parametrize("store_configurations", [False, True])
def test_failed_block_restores_the_whole_ensemble_exactly(tmp_path, monkeypatch, grid, failure, store_configurations):
    cfg = multiple_config(grid=grid)
    cfg["output"]["store_configurations"] = store_configurations
    reference = run_metadynamics(cfg, multiple_system(), tmp_path / "reference", return_data=True)
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
            run_metadynamics(cfg, multiple_system(), tmp_path / "resumed")
    checkpoint = tmp_path / "resumed/checkpoints/metadynamics_walkers.npz"
    with load_checkpoint(checkpoint) as saved:
        assert int(saved["elapsed"]) == 10
        assert saved["q"].shape[0] == 2
        assert [(h["step"], h["walker"]) for h in json.loads(str(saved["hills"]))] == [
            (6, 0), (6, 1), (9, 0), (9, 1)]
    assert list((tmp_path / "resumed/checkpoints").glob("*.npz")) == [checkpoint]
    resumed = run_metadynamics(cfg, multiple_system(), tmp_path / "resumed", return_data=True)
    assert_same(reference, resumed)


def test_shared_configuration_history_survives_blocks_and_offline_analysis(tmp_path):
    cfg = multiple_config()
    plain = run_metadynamics(cfg, multiple_system(), tmp_path / "plain", return_data=True)
    cfg["output"]["store_configurations"] = True
    stored = run_metadynamics(cfg, multiple_system(), tmp_path / "stored", return_data=True)
    from pyrpmd.workflows.metadynamics import resolve_metadynamics
    cv, _, _ = resolve_metadynamics(cfg, multiple_system())
    for before, after in zip(plain["streams"], stored["streams"]):
        for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
            np.testing.assert_array_equal(before[key], after[key])
        positions = after["sample_configurations"]
        assert positions.shape == (12, 3, 1, 3)
        np.testing.assert_allclose([cv.value(q) for q in positions], after["samples"])
        np.testing.assert_array_equal(positions[-1], after["state"].q)
        assert not np.shares_memory(positions[0], positions[-1])
    assert_same(stored, reanalyze_metadynamics(cfg, multiple_system(), tmp_path / "stored", return_data=True))
    assert_same(stored, run_metadynamics(cfg, multiple_system(), tmp_path / "stored", return_data=True))
    cfg["output"]["store_configurations"] = False
    with pytest.raises(RuntimeError, match="configuration history differs"):
        run_metadynamics(cfg, multiple_system(), tmp_path / "stored")


def test_shared_history_is_synchronized_and_not_mistaken_for_independent_errors(tmp_path, monkeypatch):
    cfg = multiple_config()
    result = run_metadynamics(cfg, multiple_system(), tmp_path, return_data=True)
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
    again = reanalyze_metadynamics(cfg, multiple_system(), tmp_path, return_data=True)
    assert_same(result, again)
    assert_same(result, run_metadynamics(cfg, multiple_system(), tmp_path, return_data=True))


def test_process_walker_order_does_not_change_synchronization(tmp_path):
    cfg = multiple_config()
    # Fewer, longer blocks keep this scheduler test cheap while checking two
    # probability-dependent depositions within the same synchronization block.
    cfg["metadynamics"]["block_steps"] = 12
    serial = run_metadynamics(cfg, multiple_system(), tmp_path / "serial", return_data=True)
    cfg["parallel"] = {"backend": "process", "trajectory_workers": 2}
    parallel = run_metadynamics(cfg, multiple_system(), tmp_path / "process", return_data=True)
    assert_same(serial, parallel)


def test_single_shared_walker_preserves_the_independent_trajectory(tmp_path):
    cfg = multiple_config()
    cfg["sampling"]["replicas"] = 1
    shared = run_metadynamics(cfg, multiple_system(), tmp_path / "shared", return_data=True)
    cfg["metadynamics"].pop("multiple_walkers")
    cfg["metadynamics"].pop("block_steps")
    independent = run_metadynamics(cfg, multiple_system(), tmp_path / "independent", return_data=True)
    for key in ("samples", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(shared["streams"][0][key], independent["streams"][0][key])
    expected = [{key: value for key, value in hill.items() if key != "walker"} for hill in shared["hills"]]
    assert expected == independent["streams"][0]["hills"]


@pytest.mark.parametrize("damage", ["walker", "clock", "samples"])
def test_rechecksummed_global_checkpoint_is_still_validated(tmp_path, damage):
    cfg = multiple_config()
    run_metadynamics(cfg, multiple_system(), tmp_path)
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
        reanalyze_metadynamics(cfg, multiple_system(), tmp_path)


def test_sync_interval_is_physics_significant(tmp_path):
    cfg = multiple_config()
    run_metadynamics(cfg, multiple_system(), tmp_path)
    changed = copy.deepcopy(cfg)
    changed["metadynamics"]["block_steps"] += 1
    with pytest.raises(RuntimeError, match="physics differs"):
        run_metadynamics(changed, multiple_system(), tmp_path)


@pytest.mark.parametrize("thermostat", ["nose-hoover-chain", "pile-l"])
def test_thermostat_state_survives_shared_blocks_and_restarts(tmp_path, monkeypatch, thermostat):
    cfg = multiple_config()
    cfg["sampling"].update(thermostat=thermostat,
                           thermostat_params={"tdamp": 0.4, "reset_after_equilibration": True})
    reference = run_metadynamics(cfg, multiple_system(), tmp_path / "reference", return_data=True)
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
            run_metadynamics(cfg, multiple_system(), tmp_path / "resumed")
    assert_same(reference, run_metadynamics(cfg, multiple_system(), tmp_path / "resumed", return_data=True))


def test_zero_equilibration_fixed_slab_and_completed_walker_failure(tmp_path, monkeypatch):
    from pyrpmd.workflows import metadynamics

    cfg = multiple_config(grid=True)
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
    cfg = multiple_config()
    result = run_metadynamics(cfg, multiple_system(), tmp_path, return_data=True)
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
    cv, _, options = resolve_metadynamics(cfg, multiple_system())
    with pytest.raises(ValueError, match="walker|time"):
        MetadynamicsBias(cv, options["sigma"], options["height"], cfg["beta"],
                         bias_factor=options["bias_factor"], hills=hills)


@pytest.mark.parametrize("mode,parameters", [
    ("nhc", {"tau": 0.4, "chain_length": 3}),
    ("svr", {"tau": 0.4}),
    ("gle", {"gle": {"A": [[0.7, 0.1], [-0.1, 1.2]], "C": [[1.0, 0.0], [0.0, 1.0]]}}),
])
def test_composed_walkers_resume_auxiliary_state_after_failed_deposition(
        tmp_path, monkeypatch, mode, parameters):
    """Every new block must retain each walker's common thermostat engine."""
    pytest.importorskip("pyrpmd._thermal")
    cfg = multiple_config()
    cfg["sampling"].update(thermostat=mode, thermostat_params=parameters)
    reference = run_metadynamics(cfg, multiple_system(), tmp_path / "full", return_data=True)
    original = MetadynamicsBias.deposit

    def fail_after_deposit(self, value, **kwargs):
        result = original(self, value, **kwargs)
        if kwargs["step"] == 12 and kwargs["walker"] == 1:
            raise RuntimeError("injected composed walker deposition failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(MetadynamicsBias, "deposit", fail_after_deposit)
        with pytest.raises(RuntimeError, match="composed walker"):
            run_metadynamics(cfg, multiple_system(), tmp_path / "resumed")
    with load_checkpoint(tmp_path / "resumed/checkpoints/metadynamics_walkers.npz") as saved:
        assert int(saved["elapsed"]) == 10
        operators = json.loads(str(saved["operator_states"]))
        assert len(operators) == 2 and all(state["sampling"] for state in operators)
    resumed = run_metadynamics(cfg, multiple_system(), tmp_path / "resumed", return_data=True)
    assert_same(reference, resumed)
    assert_same(resumed, reanalyze_metadynamics(cfg, multiple_system(), tmp_path / "resumed", return_data=True))


@pytest.mark.parametrize("missing", ["operators", "auxiliary"])
def test_composed_walker_checkpoint_requires_auxiliaries_before_loading_pes(
        tmp_path, monkeypatch, missing):
    pytest.importorskip("pyrpmd._thermal")
    cfg = multiple_config()
    cfg["sampling"].update(thermostat="nhc", thermostat_params={"tau": 0.4})
    run_metadynamics(cfg, multiple_system(), tmp_path)
    path = tmp_path / "checkpoints/metadynamics_walkers.npz"
    with load_checkpoint(path) as saved:
        payload = {key: saved[key].copy() for key in saved.files if key != "_pyrpmd_checkpoint"}
    if missing == "operators":
        payload.pop("operator_states")
    else:
        operators = json.loads(str(payload["operator_states"]))
        operators[1]["sampling"] = {}
        payload["operator_states"] = json.dumps(operators)
    atomic_savez(path, **payload)
    unchanged = path.read_bytes()

    def unavailable(*args, **kwargs):
        pytest.fail("invalid ensemble checkpoints must fail before constructing a PES")

    monkeypatch.setattr("pyrpmd.workflows.metadynamics.make_task_potential", unavailable)
    with pytest.raises(RuntimeError, match="missing.*state"):
        run_metadynamics(cfg, multiple_system(), tmp_path)
    assert path.read_bytes() == unchanged


def test_walker_sharing_modes_cannot_be_combined(tmp_path):
    cfg = multiple_config()
    cfg["metadynamics"]["shared_walkers"] = True
    with pytest.raises(ValueError, match="select only one"):
        run_metadynamics(cfg, multiple_system(), tmp_path / "unused")
    assert not (tmp_path / "unused").exists()
