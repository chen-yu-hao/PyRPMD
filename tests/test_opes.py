"""Probability, force, normalization and restart evidence for native OPES."""

import copy
import json
import numpy as np
import pytest
from scipy.integrate import trapezoid

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.cv import CVStack, Cartesian, Distance
from pyrpmd.dynamics import Dynamics
from pyrpmd.opes import OPESBias, normalized_weights
from pyrpmd.system import System
from pyrpmd.workflows.opes import (
    _sample_replica, _analyze, resolve_opes, run_opes, reanalyze_opes, validate_opes,
)


def cv1():
    return CVStack([Cartesian(0, 0)])


def system():
    return System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]]))


def config(dimension=1):
    return {
        "task": "opes", "beta": 2.0, "beads": 3, "dt_au": 0.02,
        "seed": 321, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 3, "production_steps": 24, "sample_interval": 2,
                     "andersen_interval": 4, "replicas": 1},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                for axis in ("x", "y")[:dimension]],
        "opes": {"sigma": [0.3] * dimension, "barrier": 2.0, "pace": 3, "bias_factor": 5},
        "analysis": {"bins": [6] * dimension, "bootstrap": 0},
        "output": {"checkpoint_steps": 5},
    }


def test_weighted_kde_and_center_normalization_match_independent_formula():
    bias = OPESBias(cv1(), [0.4], 2.0, barrier=1.3, bias_factor=4)
    centers, weights = [], []
    for step, center in enumerate([-0.4, 0.7, 0.2, -1.1, 0.0], 1):
        before = float(bias.physical_energy([center]))
        event = bias.update([center], step=step, time=0.1 * step)
        assert event["log_weight"] == pytest.approx(2 * before)
        centers.append(center)
        weights.append(np.exp(2 * before))
        point = np.linspace(-1.8, 1.8, 71)
        c = np.asarray(centers)
        w = np.asarray(weights)
        kernel = np.exp(-0.5 * ((point[:, None] - c) / 0.4)**2) / (np.sqrt(2 * np.pi) * 0.4)
        probability = kernel @ w / w.sum()
        centers_kernel = np.exp(-0.5 * ((c[:, None] - c) / 0.4)**2) / (np.sqrt(2 * np.pi) * 0.4)
        z = np.mean(centers_kernel @ w / w.sum())
        expected = (1 - 1 / 4) / 2 * np.log(probability / z + np.exp(-2 * 1.3 / (1 - 1 / 4)))
        np.testing.assert_allclose(np.exp(bias.log_probability(point[:, None])), probability, rtol=2e-14)
        np.testing.assert_allclose(bias.physical_energy(point[:, None]), expected, rtol=2e-14, atol=2e-14)
        assert bias.log_z == pytest.approx(np.log(z), abs=5e-15)
    assert np.ptp(weights) > 0.5  # this is not an accidentally unweighted KDE test


@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
@pytest.mark.parametrize("beads", [1, 4])
def test_full_ring_force_and_physical_bias_convention(estimator, beads):
    cv = CVStack([Distance([0, 1], estimator=estimator), Cartesian(1, 1, estimator=estimator)])
    bias = OPESBias(cv, [0.45, 0.3], 5.0, barrier=0.8, bias_factor=7)
    bias.update([1.1, 0.0], step=1, time=0.1)
    bias.update([1.3, 0.1], step=2, time=0.2)
    q = np.tile([[-0.5, 0.1, 0.2], [0.8, -0.2, 0.3]], (beads, 1, 1))
    q += np.random.default_rng(42).normal(scale=0.07, size=q.shape)
    energy, force = bias.energy_force(q)
    assert energy == pytest.approx(beads * float(bias.physical_energy(cv.value(q))))
    numerical = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = -(bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / 2e-6
    np.testing.assert_allclose(force, numerical, atol=3e-10, rtol=3e-7)


@pytest.mark.parametrize("width", [0.2, 1.4, 5.0])
def test_wrapped_probability_integrates_to_one_and_has_a_smooth_force(width):
    axis = Cartesian(0, 0)
    axis.periods = (2 * np.pi,)
    cv = CVStack([axis])
    bias = OPESBias(cv, [width], 1.0, barrier=8)
    bias.update([0.2], step=1, time=0.1)
    bias.update([2.5], step=2, time=0.2)
    points = np.linspace(-np.pi, np.pi, 50001)[:, None]
    integral = trapezoid(np.exp(bias.log_probability(points)), points[:, 0])
    assert integral == pytest.approx(1, abs=2e-12)
    energy, gradient = bias.physical_energy_gradient(points[::1000])
    numerical = (bias.physical_energy(points[::1000] + 1e-6) - bias.physical_energy(points[::1000] - 1e-6)) / 2e-6
    np.testing.assert_allclose(gradient[:, 0], numerical, atol=1e-8, rtol=2e-6)
    other, other_gradient = bias.physical_energy_gradient(points[::1000] + 2 * np.pi)
    np.testing.assert_allclose(energy, other, atol=5e-14)
    np.testing.assert_allclose(gradient, other_gradient, atol=5e-14)
    seam = bias.physical_energy_gradient([[0.2 + np.pi - 1e-8], [0.2 + np.pi + 1e-8]])[1]
    assert abs(seam[0, 0] - seam[1, 0]) < 1e-6


def test_nonperiodic_kde_is_normalized_and_bias_has_a_stable_floor():
    bias = OPESBias(cv1(), [0.3], 1000, barrier=20)
    bias.update([0], step=1, time=0.1)
    points = np.linspace(-4, 4, 20001)
    assert trapezoid(np.exp(bias.log_probability(points[:, None])), points) == pytest.approx(1, abs=1e-13)
    # exp(log_epsilon) underflows, but log-domain evaluation must remain finite.
    assert bias.log_epsilon < -20000
    energy, force = bias.physical_energy_gradient([[10000], [-10000]])
    np.testing.assert_allclose(energy, [-20, -20], atol=1e-13)
    np.testing.assert_array_equal(force, [[0], [0]])


def test_log_weight_normalization_is_offset_invariant_and_overflow_safe():
    expected = np.array([1, np.exp(-2), np.exp(-1000)])
    expected /= expected.sum()
    np.testing.assert_allclose(normalized_weights([1000, 998, 0]), expected)
    np.testing.assert_allclose(normalized_weights([-1000, -1002, -2000]), expected)
    for invalid in ([], [np.nan], [np.inf], [[0]]):
        with pytest.raises(ValueError):
            normalized_weights(invalid)


def test_frozen_histogram_recovers_analytic_gaussian_with_independent_bias_gauges(tmp_path):
    from scipy.integrate import cumulative_trapezoid
    from scipy.special import ndtr

    cfg = config()
    cfg["opes"].update(freeze_after_steps=12)
    cfg["analysis"].update(method="reweighted_histogram", bins=[24], ranges=[[-2, 2]])
    grid = np.linspace(-7, 7, 100001)
    quantiles = (np.arange(100000) + 0.5) / 100000
    streams = []
    for replica, shift in enumerate((-0.8, 0.8)):
        bias = OPESBias(cv1(), [0.3], 2, barrier=2, bias_factor=5)
        for step, center in enumerate([shift, shift + 0.1, shift - 0.2], 1):
            bias.update([center], step=step, time=0.1 * step)
        # Independent deterministic inverse-CDF reference for exp[-beta(U+V)]
        # with U=x^2/2; this never uses the PyRPMD propagator.
        density = np.exp(-2 * (grid**2 / 2 + bias.physical_energy(grid[:, None])))
        cdf = cumulative_trapezoid(density, grid, initial=0)
        samples = np.interp(quantiles, cdf / cdf[-1], grid)[:, None]
        streams.append({"samples": samples, "sample_steps": np.arange(len(samples)) + 100,
            "sample_bias": bias.physical_energy(samples), "bias_state": bias.state_dict(),
            "propagation": {}, "checkpoint": "synthetic-equilibrium-reference"})
    reference = _analyze(streams, cfg, system(), tmp_path / "base", True)
    edges = reference["data"]["edges"][0]
    mass = np.diff(ndtr(edges * np.sqrt(2)))
    exact = -np.log(mass / np.diff(edges)) / 2
    reference_bin = tuple(reference["summary"]["reference_bin"])
    exact -= exact[reference_bin]
    np.testing.assert_allclose(reference["data"]["free_energy"], exact, atol=0.012, rtol=0)
    shifted = copy.deepcopy(streams)
    shifted[0]["sample_bias"] += 1000
    shifted[1]["sample_bias"] -= 1000
    result = _analyze(shifted, cfg, system(), tmp_path / "gauge", True)
    np.testing.assert_allclose(result["data"]["free_energy"], reference["data"]["free_energy"], atol=1e-12, rtol=0)


def test_state_is_detached_roundtrips_exactly_and_checks_cached_normalization():
    bias = OPESBias(cv1(), [0.4], 2, barrier=3)
    for i in range(1, 10):
        bias.update([np.sin(i)], step=i, time=i * 0.1)
    state = bias.state_dict()
    restored = OPESBias(cv1(), [0.4], 2, barrier=3, state=json.loads(json.dumps(state)))
    assert restored.state_dict() == state
    for i in range(10, 14):
        assert bias.update([np.sin(i)], step=i, time=i * 0.1) == restored.update([np.sin(i)], step=i, time=i * 0.1)
    assert restored.state_dict() == bias.state_dict()
    state["centers"][0][0] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        OPESBias(cv1(), [0.4], 2, barrier=3, state=state)
    assert bias.state_dict()["centers"][0] != state["centers"][0]
    invalid = restored.state_dict()
    invalid["log_z"] += 0.1
    with pytest.raises(ValueError, match="normalization"):
        OPESBias(cv1(), [0.4], 2, barrier=3, state=invalid)


def test_invalid_update_leaves_history_and_bias_unchanged():
    bias = OPESBias(cv1(), [0.4], 2, barrier=3)
    bias.update([0], step=2, time=0.2)
    before = bias.state_dict()
    for kwargs in ({"step": 2, "time": 0.3}, {"step": 3, "time": -0.1}, {"step": True, "time": 0.3}):
        with pytest.raises(ValueError):
            bias.update([0.3], **kwargs)
        assert bias.state_dict() == before
    with pytest.raises(ValueError):
        bias.update([np.nan], step=3, time=0.3)
    assert bias.state_dict() == before
    with pytest.raises(ValueError):
        bias.update([0.3], step=2**63 + 1, time=0.3)
    overflow = bias.state_dict()
    overflow["steps"] = [2**63 + 1]
    with pytest.raises(ValueError, match="clocks"):
        OPESBias(cv1(), [0.4], 2, barrier=3, state=overflow)


def test_remote_zero_weight_kernel_does_not_contaminate_finite_force():
    bias = OPESBias(cv1(), [0.3], 2, barrier=3)
    bias.update([1e308], step=1, time=0.1)
    bias.update([-1e308], step=2, time=0.2)
    energy, gradient = bias.physical_energy_gradient([[1e308], [-1e308]])
    assert np.isfinite(energy).all()
    np.testing.assert_array_equal(gradient, [[0], [0]])
    for period in (1e-308, 1e308):
        axis = Cartesian(0, 0)
        axis.periods = (period,)
        with pytest.raises(ValueError, match="representable"):
            OPESBias(CVStack([axis]), [0.3], 2, barrier=3)


@pytest.mark.parametrize("failure", ["step", "update"])
def test_interruption_rolls_back_rng_bias_and_clocks_and_resumes_exactly(tmp_path, monkeypatch, failure):
    cfg = config()
    reference = _sample_replica({"config": cfg, "system": system(), "output": str(tmp_path / "full"), "replica": 0})
    task = {"config": cfg, "system": system(), "output": str(tmp_path / "resumed"), "replica": 0}
    step_original, update_original = Dynamics.step, OPESBias.update
    def fail_step(self, state, bias=None):
        result = step_original(self, state, bias)
        if state.step == 12:
            raise RuntimeError("injected after step")
        return result
    def fail_update(self, values, **kwargs):
        result = update_original(self, values, **kwargs)
        if kwargs["step"] == 12:
            raise RuntimeError("injected after probability update")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "step", fail_step) if failure == "step" else patch.setattr(OPESBias, "update", fail_update)
        with pytest.raises(RuntimeError, match="injected"):
            _sample_replica(task)
    with load_checkpoint(tmp_path / "resumed/checkpoints/opes_0000.npz") as data:
        assert int(data["elapsed"]) == 11
        assert json.loads(str(data["bias_state"]))["steps"] == [6, 9]
    resumed = _sample_replica(task)
    for attr in ("q", "p"):
        np.testing.assert_array_equal(getattr(reference["state"], attr), getattr(resumed["state"], attr))
    for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(reference[key], resumed[key])
    for key in ("rng", "bias_state", "propagation"):
        assert reference[key] == resumed[key]


@pytest.mark.parametrize("method", ["kde", "reweighted_histogram"])
def test_two_dimensional_workflow_freezes_and_reanalyzes_without_a_potential(tmp_path, monkeypatch, method):
    cfg = config(2)
    cfg["sampling"]["replicas"] = 2
    cfg["opes"].update(freeze_after_steps=12, freeze_equilibration_steps=2)
    cfg["analysis"]["method"] = method
    result = run_opes(cfg, system(), tmp_path, return_data=True)
    assert result["summary"]["kernels_per_replica"] == [4, 4]
    assert result["summary"]["samples"] == (24 if method == "kde" else 10)
    assert not result["summary"]["converged"]
    assert np.isnan(result["data"]["standard_error"]).all()
    assert result["streams"][0]["bias_state"] != result["streams"][1]["bias_state"]
    for stream in result["streams"]:
        cv, _, options = resolve_opes(cfg, system())
        bias = OPESBias(cv, options["sigma"], cfg["beta"], barrier=options["barrier"], bias_factor=options["bias_factor"], state=stream["bias_state"])
        frozen = stream["sample_steps"] > 15
        np.testing.assert_allclose(stream["sample_bias"][frozen], bias.physical_energy(stream["samples"][frozen]), rtol=1e-14)
    def unavailable(*args, **kwargs):
        raise RuntimeError("PES must not be constructed")
    monkeypatch.setattr("pyrpmd.workflows.opes.make_task_potential", unavailable)
    again = reanalyze_opes(cfg, system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(result["data"]["free_energy"], again["data"]["free_energy"])
    completed = run_opes(cfg, system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(completed["data"]["free_energy"], result["data"]["free_energy"])
    cfg["analysis"]["bins"] = [4, 5]
    changed = reanalyze_opes(cfg, system(), tmp_path, return_data=True)
    assert changed["data"]["free_energy"].shape == (4, 5)


@pytest.mark.parametrize("table,key,value", [
    ("dynamics", "loop_backend", "native"), ("analysis", "method", "mbar"),
    ("analysis", "method", "reweighted_histogram"), ("analysis", "measure", "rate_flux"),
    ("analysis", "bootstrap", 20), ("opes", "typo", 1), ("opes", "freeze_after_steps", 5),
    ("opes", "freeze_equilibration_steps", 10), ("opes", "bias_factor", 1),
    ("sampling", "thermostat", "unimplemented"), ("opes", "barrier", -1),
])
def test_rejects_unsupported_or_inconsistent_inputs(table, key, value):
    cfg = config()
    cfg.setdefault(table, {})[key] = value
    with pytest.raises(ValueError):
        validate_opes(cfg, system())


def test_explicit_width_and_energy_units():
    cfg = config()
    cfg["cvs"][0]["unit"] = "angstrom"
    cfg["opes"].update(sigma=[0.1], barrier="2 kcal/mol")
    _, _, options = resolve_opes(cfg, system())
    assert options["sigma"][0] == pytest.approx(0.18897261246, rel=1e-8)
    assert options["barrier"] == pytest.approx(2 / 627.509474, rel=1e-7)
    cfg["opes"].update(sigma=["0.1 angstrom"])
    assert resolve_opes(cfg, system())[2]["sigma"] == pytest.approx(options["sigma"])


def test_actual_cartesian_cv_object_can_be_resolved_and_resumed_after_copy(tmp_path):
    cfg = config()
    cfg["cvs"] = [Cartesian(0, "x")]
    cv, selected, options = resolve_opes(cfg, system())
    assert selected[0] is cfg["cvs"][0]
    assert options["sigma"] == [0.3]
    assert cv.dimension == 1
    result = run_opes(cfg, system(), tmp_path, return_data=True)
    assert result["streams"][0]["complete"]
    assert len(result["streams"][0]["samples"]) == 12
    resumed = run_opes(copy.deepcopy(cfg), system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(result["data"]["free_energy"], resumed["data"]["free_energy"])
    cfg["cvs"] = [Cartesian(0, "y")]
    with pytest.raises(ValueError, match="physics differs"):
        _sample_replica({"config": cfg, "system": system(), "output": str(tmp_path), "replica": 0})


def test_custom_object_cv_requires_a_factory_configuration_for_checkpoint_identity():
    class CustomCartesian(Cartesian):
        pass
    cfg = config()
    cfg["cvs"] = [CustomCartesian(0, "x")]
    with pytest.raises(ValueError, match="factory configuration"):
        resolve_opes(cfg, system())


def test_two_process_scheduler_matches_serial_independent_histories(tmp_path):
    cfg = config(2)
    cfg["sampling"]["replicas"] = 2
    serial = run_opes(cfg, system(), tmp_path / "serial", return_data=True)
    cfg["parallel"] = {"backend": "process", "trajectory_workers": 2}
    parallel = run_opes(cfg, system(), tmp_path / "parallel", return_data=True)
    for left, right in zip(serial["streams"], parallel["streams"]):
        for attr in ("q", "p"):
            np.testing.assert_array_equal(getattr(left["state"], attr), getattr(right["state"], attr))
        for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
            np.testing.assert_array_equal(left[key], right[key])
        assert left["rng"] == right["rng"]
        assert left["bias_state"] == right["bias_state"]
    np.testing.assert_array_equal(serial["data"]["free_energy"], parallel["data"]["free_energy"])


def test_public_executor_iterator_is_consumed_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    cfg = config()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = run_opes(cfg, system(), tmp_path, executor=executor, return_data=True)
    assert result["summary"]["samples"] == 12
    assert len(result["streams"]) == 1


def test_corrupted_but_rechecksummed_sample_clock_is_rejected(tmp_path):
    cfg = config()
    _sample_replica({"config": cfg, "system": system(), "output": str(tmp_path), "replica": 0})
    path = tmp_path / "checkpoints/opes_0000.npz"
    with load_checkpoint(path) as data:
        payload = {key: data[key].copy() for key in data.files if key != "_pyrpmd_checkpoint"}
    payload["sample_steps"][0] += 1
    atomic_savez(path, **payload)
    with pytest.raises(RuntimeError, match="sampling clock"):
        reanalyze_opes(cfg, system(), tmp_path)


@pytest.mark.parametrize("key,value", [("elapsed", 27.75), ("step", 27.75), ("step", True), ("elapsed", [27])])
def test_noninteger_rechecksummed_checkpoint_clock_is_rejected(tmp_path, key, value):
    cfg = config()
    _sample_replica({"config": cfg, "system": system(), "output": str(tmp_path), "replica": 0})
    path = tmp_path / "checkpoints/opes_0000.npz"
    with load_checkpoint(path) as data:
        payload = {name: data[name].copy() for name in data.files if name != "_pyrpmd_checkpoint"}
    payload[key] = np.asarray(value)
    atomic_savez(path, **payload)
    with pytest.raises(RuntimeError, match="integer scalar clock"):
        reanalyze_opes(cfg, system(), tmp_path)
