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
