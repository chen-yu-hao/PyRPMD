"""Metadynamics force, measure, event-clock and restart regression checks."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

from pyrpmd.bias import MetadynamicsBias
from pyrpmd.cv import CVStack, Distance, Cartesian
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System
from pyrpmd.analysis.metadynamics import bias_fes
from pyrpmd.workflows.metadynamics import (
    _sample_replica, resolve_metadynamics, run_metadynamics,
    reanalyze_metadynamics, validate_metadynamics,
)
from pyrpmd.checkpoint import load_checkpoint


def system():
    return System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]]))


def config(mode="well_tempered", dimension=1):
    return {
        "task": "metadynamics", "beta": 2.0, "beads": 3, "dt_au": 0.02,
        "seed": 321, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 3, "production_steps": 24, "sample_interval": 2,
                     "andersen_interval": 4, "replicas": 1},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                for axis in ("x", "y")[:dimension]],
        "metadynamics": {"mode": mode, "sigma": [0.3] * dimension,
                         "height": 0.05, "pace": 3,
                         **({"bias_factor": 5.0} if mode == "well_tempered" else {})},
        "analysis": {"bins": [6] * dimension, "bootstrap": 0},
        "output": {"checkpoint_steps": 5},
    }


@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
@pytest.mark.parametrize("beads", [1, 4])
def test_gaussian_full_ring_force_matches_energy_derivative(estimator, beads):
    cv = CVStack([Distance([0, 1], estimator=estimator), Cartesian(1, 1, estimator=estimator)])
    bias = MetadynamicsBias(cv, [0.45, 0.2], 0.03, 5.0, bias_factor=7)
    bias.deposit([1.1, 0.0], step=1, time=0.1)
    bias.deposit([1.3, 0.1], step=2, time=0.2)
    rng = np.random.default_rng(42)
    q = np.tile([[-0.5, 0.1, 0.2], [0.8, -0.2, 0.3]], (beads, 1, 1))
    q += rng.normal(scale=0.07, size=q.shape)
    energy, force = bias.energy_force(q)
    assert energy == pytest.approx(beads * float(bias.physical_energy(cv.value(q))))
    numerical = np.zeros_like(q)
    for idx in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[idx] += 1e-6
        minus[idx] -= 1e-6
        numerical[idx] = -(bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / 2e-6
    np.testing.assert_allclose(force, numerical, atol=2e-10, rtol=2e-7)


def test_tempering_uses_physical_bias_without_an_extra_bead_factor():
    cv = CVStack([Cartesian(0, 0)])
    bias = MetadynamicsBias(cv, [0.5], 0.4, 2.0, bias_factor=3.0)
    first = bias.deposit([0.0], step=1, time=0.01)
    second = bias.deposit([0.0], step=2, time=0.02)
    assert first["height"] == 0.4
    assert second["height"] == pytest.approx(0.4 * np.exp(-2 * 0.4 / (3 - 1)))
    for beads in (1, 8):
        q = np.zeros((beads, 1, 3))
        assert bias.energy_force(q)[0] / beads == pytest.approx(first["height"] + second["height"])
    with pytest.raises(ValueError, match="strictly increase"):
        bias.deposit([0.0], step=2, time=0.03)
    assert len(bias.hills) == 2
    standard = MetadynamicsBias(cv, [0.5], 0.4, 2.0, mode="standard")
    for i in range(3):
        assert standard.deposit([0.0], step=i + 1, time=0.1 * (i + 1))["height"] == 0.4


def test_bias_history_append_restore_and_rollback_keep_exact_gaussian_sum():
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    sigma = np.array([0.13, 0.29])
    bias = MetadynamicsBias(cv, sigma, 0.05, 2.0, mode="standard")
    rng = np.random.default_rng(731)
    points = rng.normal(size=(3, 4, 2))

    def check_sum():
        energy, gradient = bias.physical_energy_gradient(points)
        expected_energy = np.zeros(points.shape[:-1])
        expected_gradient = np.zeros_like(points)
        for hill in bias.hills:
            delta = points - hill["center"]
            contribution = hill["height"] * np.exp(-0.5 * np.sum((delta / sigma)**2, axis=-1))
            expected_energy += contribution
            expected_gradient -= contribution[..., None] * delta / sigma**2
        np.testing.assert_allclose(energy, expected_energy, atol=1e-15)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-15)
        np.testing.assert_array_equal(bias.physical_energy(points), energy)

    check_sum()
    for step in range(1, 8):
        bias.deposit(rng.normal(size=2), step=step, time=step * 0.1)
        check_sum()
    bias.truncate_history(3)
    check_sum()
    bias.deposit([0.3, -0.1], step=4, time=0.4)
    check_sum()
    restored = MetadynamicsBias(cv, sigma, 0.05, 2.0, mode="standard", hills=bias.hills)
    for actual, expected in zip(restored.physical_energy_gradient(points), bias.physical_energy_gradient(points)):
        np.testing.assert_array_equal(actual, expected)
    bias.truncate_history(0)
    check_sum()


@pytest.mark.parametrize("count", [-1, 1, 0.5, True])
def test_invalid_history_truncation_is_rejected(count):
    bias = MetadynamicsBias(CVStack([Cartesian(0, 0)]), [0.5], 0.4, 2.0)
    with pytest.raises(ValueError, match="history count"):
        bias.truncate_history(count)


def test_history_snapshot_and_deposition_record_cannot_mutate_force_cache():
    bias = MetadynamicsBias(CVStack([Cartesian(0, 0)]), [0.5], 0.4, 2.0)
    record = bias.deposit([0.2], step=1, time=0.1)
    expected = bias.physical_energy_gradient([0.1])
    snapshot = bias.hills
    snapshot[0]["center"][0] = 7.0
    snapshot[0]["height"] = 10.0
    record["center"][0] = 9.0
    snapshot.clear()
    assert bias.hill_count == 1
    assert bias.hills[0]["center"] == [0.2]
    for actual, original in zip(bias.physical_energy_gradient([0.1]), expected):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("width", [0.2, 1.4, 5.0])
def test_periodic_gaussians_are_smooth_and_match_finite_difference(width):
    axis = Cartesian(0, 0)
    axis.periods = (2 * np.pi,)
    cv = CVStack([axis])
    bias = MetadynamicsBias(cv, [width], 0.5, 1.0, mode="standard")
    bias.deposit([0.2], step=1, time=0.1)
    points = np.array([[-np.pi], [-1.1], [0.0], [np.pi - 1e-7]])
    energy, grad = bias.physical_energy_gradient(points)
    numerical = (bias.physical_energy(points + 1e-6) - bias.physical_energy(points - 1e-6)) / 2e-6
    np.testing.assert_allclose(grad[:, 0], numerical, atol=1e-9, rtol=2e-6)
    e2, g2 = bias.physical_energy_gradient(points + 2 * np.pi)
    np.testing.assert_allclose(energy, e2, atol=1e-14)
    np.testing.assert_allclose(grad, g2, atol=1e-14)
    assert bias.physical_energy([0.2]) == pytest.approx(0.5)
    # The seam is exactly half a period away from this hill's center.
    _, seam = bias.physical_energy_gradient(np.array([[0.2 + np.pi - 1e-8], [0.2 + np.pi + 1e-8]]))
    assert abs(seam[0, 0] - seam[1, 0]) < 1e-7


@pytest.mark.parametrize("mode", ["standard", "well_tempered"])
@pytest.mark.parametrize("failure_after", ["step", "deposit"])
def test_failure_at_a_deposition_boundary_resumes_exactly_without_duplicate_hills(tmp_path, monkeypatch, mode, failure_after):
    cfg = config(mode)
    ref = _sample_replica({"config": cfg, "system": system(), "output": str(tmp_path / "full"), "replica": 0})
    target = Dynamics if failure_after == "step" else MetadynamicsBias
    original = getattr(target, failure_after)
    def fail_after_a_successful_map(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        step = args[0].step if failure_after == "step" else kwargs["step"]
        if step == 12:
            raise RuntimeError("injected interruption at deposition boundary")
        return result
    task = {"config": cfg, "system": system(), "output": str(tmp_path / "resumed"), "replica": 0}
    with monkeypatch.context() as patch:
        patch.setattr(target, failure_after, fail_after_a_successful_map)
        with pytest.raises(RuntimeError, match="injected interruption"):
            _sample_replica(task)
    with load_checkpoint(tmp_path / "resumed/checkpoints/metadynamics_0000.npz") as cp:
        assert int(cp["step"]) == 11
        assert [h["step"] for h in json.loads(str(cp["hills"]))] == [6, 9]
    hills_path = tmp_path / "resumed/replicas/0000/HILLS.csv"
    hills_path.write_text(hills_path.read_text() + "999, stale derived row\n")
    resumed = _sample_replica(task)
    for attr in ("q", "p"):
        np.testing.assert_array_equal(getattr(ref["state"], attr), getattr(resumed["state"], attr))
    for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(ref[key], resumed[key])
    assert ref["rng"] == resumed["rng"]
    assert ref["hills"] == resumed["hills"]
    assert ref["propagation"] == resumed["propagation"]
    assert [h["step"] for h in resumed["hills"]] == list(range(6, 28, 3))
    assert len(hills_path.read_text().splitlines()) == 9


def test_two_dimensional_workflow_and_saved_reanalysis(tmp_path):
    cfg = config(dimension=2)
    cfg["sampling"]["replicas"] = 2
    result = run_metadynamics(cfg, system(), tmp_path, return_data=True)
    assert result["summary"]["hills_per_replica"] == [8, 8]
    assert result["summary"]["samples"] == 24
    assert result["summary"]["converged"] is False
    assert result["data"]["bias_per_replica"].shape == (2, 6, 6)
    assert result["streams"][0]["hills"] != result["streams"][1]["hills"]
    assert np.isnan(result["data"]["standard_error"]).all()
    again = reanalyze_metadynamics(cfg, system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(result["data"]["free_energy"], again["data"]["free_energy"])
    cfg["analysis"]["bins"] = [4, 5]
    changed = reanalyze_metadynamics(cfg, system(), tmp_path, return_data=True)
    assert changed["data"]["free_energy"].shape == (4, 5)


def test_bias_fes_uses_distinct_standard_and_tempered_factors_and_masks_unsampled_bins():
    cv = CVStack([Cartesian(0, 0)])
    standard = MetadynamicsBias(cv, [0.5], 0.4, 1.0, mode="standard")
    standard.deposit([0.0], step=1, time=0.1)
    tempered = MetadynamicsBias(cv, [0.5], 0.4, 1.0, mode="well_tempered", bias_factor=4,
                               hills=standard.hills)
    samples = [np.array([[-0.9], [-0.4], [0.1]])]
    a = bias_fes([standard], samples, bins=4, ranges=[[-1, 1]])
    b = bias_fes([tempered], samples, bins=4, ranges=[[-1, 1]])
    valid = ~a["mask"]
    np.testing.assert_allclose(b["free_energy"][valid], 4 / 3 * a["free_energy"][valid])
    assert a["mask"][-1] and np.isnan(a["free_energy"][-1])


@pytest.mark.parametrize("difference", ["temperature", "unit", "axis", "estimator"])
def test_bias_fes_rejects_replica_temperature_or_cv_mismatch(difference):
    cv1 = CVStack([Cartesian(0, "x")])
    axis = Cartesian(0, "y" if difference == "axis" else "x",
                     estimator="bead_average" if difference == "estimator" else "centroid")
    if difference == "unit":
        axis.units = ("dimensionless",)
    cv2 = CVStack([axis])
    a = MetadynamicsBias(cv1, [0.5], 0.4, 1.0)
    b = MetadynamicsBias(cv2, [0.5], 0.4, 2.0 if difference == "temperature" else 1.0)
    samples = [np.array([[-0.4], [0.1]])] * 2
    with pytest.raises(ValueError, match="temperature, method and CVs"):
        bias_fes([a, b], samples)


def test_unsupported_analysis_and_native_history_are_rejected_before_sampling():
    for table, key, value in [("dynamics", "loop_backend", "native"),
                              ("analysis", "method", "mbar"),
                              ("analysis", "measure", "rate_flux"),
                              ("analysis", "bootstrap", 20),
                              ("metadynamics", "typo", 1)]:
        cfg = config()
        cfg.setdefault(table, {})[key] = value
        with pytest.raises(ValueError):
            validate_metadynamics(cfg, system())


@pytest.mark.parametrize("extra", [
    {"cell": np.diag([6.0, 6.0, 12.0]).tolist(), "pbc": [True, True, False]},
    {"fixed_atoms": [0]},
])
def test_surface_metadynamics_is_rejected_before_any_trajectory(tmp_path, extra):
    slab = System.from_dict({**system().to_dict(), **extra})
    for operation in (
        lambda: resolve_metadynamics(config(), slab),
        lambda: run_metadynamics(config(), slab, tmp_path),
        lambda: _sample_replica({"config": config(), "system": slab,
                                "output": str(tmp_path), "replica": 0}),
    ):
        with pytest.raises(ValueError, match="periodic systems or fixed atoms"):
            operation()
    assert not list(tmp_path.iterdir())


def test_frontend_resolves_width_units_and_rejects_bad_options():
    from pyrpmd.input import JobSpec, InputError
    job = {"task": "metadynamics", "temperature": "300 K", "beads": 2,
           "system": {"symbols": ["H"], "geometry": [[0, 0, 0]]},
           "potential": {"provider": "builtin", "name": "harmonic"},
           "sampling": {"equilibration_steps": 0, "production_steps": 20, "interval_steps": 2},
           "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x", "unit": "angstrom"}],
           "metadynamics": {"sigma": [0.1], "height": "0.1 kcal/mol", "pace": 5}}
    cfg, sys = JobSpec.from_mapping(job).resolve()
    _, _, options = resolve_metadynamics(cfg, sys)
    assert options["sigma"][0] == pytest.approx(0.18897261246, rel=1e-8)
    assert options["height"] == pytest.approx(0.1 / 627.509474, rel=1e-7)
    job["metadynamics"]["bias_factor"] = 1
    with pytest.raises(InputError, match="bias_factor"):
        JobSpec.from_mapping(job).resolve()
