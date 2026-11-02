"""Frozen-bias PMF measure, stationary-stage and restart evidence."""

import copy
import importlib

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from pyrpmd.analysis.fes_result import FESResult
from pyrpmd.analysis.frozen_bias import frozen_bias_fes, frozen_samples, weighted_histogram
from pyrpmd.bias import FluxMetricBias
from pyrpmd.dynamics import Dynamics
from pyrpmd.system import System


def _input(method, *, rate_flux=False):
    geometry = np.array([[-1.4, 0.0, 0.0], [0.0, 0.0, 0.0], [1.8, 0.15, 0.0]])
    system = (System(("H", "H", "H"), np.array([1.0, 2.0, 3.0]), geometry)
              if rate_flux else System(("H",), np.array([1.0]), np.array([[0.1, -0.05, 0.0]])))
    cfg = {
        "task": method, "beta": 2.0, "beads": 3, "dt_au": 0.002,
        "seed": 983, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 3, "production_steps": 36, "sample_interval": 2,
                     "andersen_interval": 4, "replicas": 1},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        method: {"sigma": [0.3], "pace": 3, "bias_factor": 5.0,
                 "freeze_after_steps": 12, "freeze_equilibration_steps": 4},
        "analysis": {"method": "reweighted_histogram", "bins": [4], "bootstrap": 0},
        "output": {"checkpoint_steps": 5},
    }
    cfg[method].update({"mode": "well_tempered", "height": 0.05}
                       if method == "metadynamics" else {"barrier": 2.0})
    if rate_flux:
        cfg["analysis"]["measure"] = "rate_flux"
        cfg["cvs"] = [{"name": "xi", "kind": "rpmdrate_xi", "estimator": "centroid",
                       "fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                       "breaking_bonds": [[1, 2]], "rinf": 4.0}]
    return cfg, system


def test_stationary_stage_strictly_excludes_freeze_and_relaxation_boundaries():
    stream = {"samples": np.array([[999.], [-999.], [777.], [0.2], [0.8]]),
              "sample_steps": np.array([6, 8, 10, 12, 14]),
              "sample_bias": np.array([800., -800., 600., 0.1, 0.4])}
    values, logs = frozen_samples(stream, 2., equilibration_steps=2,
                                  freeze_after_steps=6, freeze_equilibration_steps=2)
    np.testing.assert_array_equal(values, [[0.2], [0.8]])
    np.testing.assert_array_equal(logs, [0.2, 0.8])
    data = frozen_bias_fes([stream], 2., bins=[2], ranges=[[0., 1.]], periods=[None],
                           widths=[0.1], equilibration_steps=2,
                           freeze_after_steps=6, freeze_equilibration_steps=2)
    assert data["retained_samples_per_replica"] == [2]
    np.testing.assert_array_equal(data["raw_count"], [1, 1])
    assert np.isnan(data["standard_error"]).all()
    with pytest.raises(ValueError, match="no samples"):
        frozen_samples(stream, 2., equilibration_steps=2, freeze_after_steps=12)


class _SquareCoordinate:
    """xi=x_centroid**2, with analytic nonconstant flux metric."""
    estimator = "centroid"

    def centroid_value_gradient_hessian(self, q):
        gradient = np.zeros_like(q)
        gradient[0, 0] = 2 * q[0, 0]
        hessian = np.zeros(q.shape + q.shape)
        hessian[0, 0, 0, 0] = 2.
        return q[0, 0] ** 2, gradient, hessian


def test_nonconstant_flux_metric_and_frozen_reweighting_recover_analytic_target():
    beta, beads, stiffness = 2., 4, 0.7
    coordinate = _SquareCoordinate()
    q = np.zeros((beads, 1, 3))
    q[:, 0, 0] = [0.7, 0.9, 1.1, 1.3]
    energy, force = FluxMetricBias(coordinate, [1.], beta).energy_force(q)
    x = q[:, 0, 0].mean()
    assert energy == pytest.approx(-beads * np.log(2*x / np.sqrt(2*np.pi*beta)) / beta)
    np.testing.assert_allclose(force[:, 0, 0], 1 / (beta*x))
    np.testing.assert_array_equal(force[:, 0, 1:], 0.)

    # Independently integrate p_b(x) ~ exp[-beta*(U(x)+V(x^2))]*f_s(x).
    # With xi=x^2 and f_s proportional to x, the target xi density is exactly
    # exp(-beta*k*xi/2), unlike the ordinary CV density's extra 1/sqrt(xi).
    grid = np.linspace(0.4, 2.5, 100001)
    frozen = lambda s: 0.25*np.sin(1.7*s) + 0.12*s
    probability = np.exp(-beta*(stiffness*grid**2/2 + frozen(grid**2))) * grid
    cdf = cumulative_trapezoid(probability, grid, initial=0)
    samples_x = np.interp((np.arange(100000)+0.5)/100000, cdf/cdf[-1], grid)
    samples = samples_x[:, None]**2
    edges = [np.linspace(0.4**2, 2.5**2, 17)]
    logs = beta*frozen(samples[:, 0])
    actual, ess = weighted_histogram(samples, logs, edges, beta)
    a = beta*stiffness/2
    exact_mass = (np.exp(-a*edges[0][:-1]) - np.exp(-a*edges[0][1:])) / a
    exact = -np.log(exact_mass / np.diff(edges[0])) / beta
    np.testing.assert_allclose(actual-actual[0], exact-exact[0], atol=0.0015, rtol=0)
    assert 0 < ess <= len(samples)
    wrong, _ = weighted_histogram(samples, beads*logs, edges, beta)
    assert np.max(np.abs((wrong-wrong[0])-(exact-exact[0]))) > 0.3
    shifted, _ = weighted_histogram(samples, logs+1000., edges, beta)
    np.testing.assert_allclose(shifted, actual, atol=1e-12, rtol=0)


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_rate_flux_changes_actual_propagation_force_and_exports_reweighted_measure(method, tmp_path, monkeypatch):
    workflow = importlib.import_module("pyrpmd.workflows." + method)
    cfg, system = _input(method, rate_flux=True)
    cv, _, _ = getattr(workflow, "resolve_" + method)(cfg, system)
    recorded = []
    original = Dynamics.step

    def observe(self, state, bias=None):
        if not recorded:
            recorded.append((state.q.copy(), bias.energy_force(state.q)[1]))
        return original(self, state, bias)

    monkeypatch.setattr(Dynamics, "step", observe)
    result = getattr(workflow, "run_" + method)(cfg, system, tmp_path, return_data=True)
    q, actual_force = recorded[0]

    def independent_metric_energy(positions):
        _, gradient, _ = cv.cvs[0].centroid_value_gradient_hessian(positions.mean(axis=0))
        fs = np.sqrt(np.sum(gradient**2/system.masses[:, None]) / (2*np.pi*cfg["beta"]))
        return -len(positions)*np.log(fs)/cfg["beta"]

    numerical = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = -(independent_metric_energy(plus)-independent_metric_energy(minus))/2e-6
    # Initially neither sampler has a nonconstant history bias. This compares
    # the force actually handed to the propagator to an independent derivative.
    assert np.max(np.abs(numerical)) > 1e-3
    np.testing.assert_allclose(actual_force, numerical, rtol=3e-6, atol=3e-9)
    saved = FESResult.read(tmp_path)
    assert saved.measure == "rate_flux"
    assert saved.metadata["reweighted"] is True
    assert saved.metadata["estimator"] == "reweighted_histogram"
    assert result["summary"]["retained_samples_per_replica"] == [10]
    assert result["streams"][0]["state"].q.shape == (3, 3, 3)
    assert (~saved.mask).any()
    assert np.isnan(result["data"]["standard_error"]).all()


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
@pytest.mark.parametrize("interrupt_at", [12, 15, 22])
def test_frozen_bias_restart_before_at_and_after_final_update(method, interrupt_at, tmp_path, monkeypatch):
    workflow = importlib.import_module("pyrpmd.workflows." + method)
    cfg, system = _input(method, rate_flux=True)
    run = getattr(workflow, "run_" + method)
    full = run(cfg, system, tmp_path/"full", return_data=True)
    original = Dynamics.step

    def interrupted(self, state, bias=None):
        if state.step == interrupt_at:
            raise RuntimeError("deliberate frozen-stage interruption")
        return original(self, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "step", interrupted)
        with pytest.raises(RuntimeError, match="deliberate frozen-stage"):
            run(cfg, system, tmp_path/"resume", return_data=True)
    resumed = run(cfg, system, tmp_path/"resume", return_data=True)
    first, second = full["streams"][0], resumed["streams"][0]
    for name in ("samples", "sample_steps", "sample_bias", "sample_energies"):
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_array_equal(first["state"].q, second["state"].q)
    np.testing.assert_array_equal(first["state"].p, second["state"].p)
    assert first["rng"] == second["rng"]
    history = "hills" if method == "metadynamics" else "bias_state"
    assert first[history] == second[history]
    steps = [hill["step"] for hill in first["hills"]] if method == "metadynamics" else first["bias_state"]["steps"]
    assert steps == [6, 9, 12, 15]
    np.testing.assert_array_equal(full["data"]["free_energy"], resumed["data"]["free_energy"])


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_rate_flux_rejects_raw_estimates_missing_freeze_and_unimplemented_bootstrap(method):
    workflow = importlib.import_module("pyrpmd.workflows." + method)
    cfg, system = _input(method, rate_flux=True)
    resolve = getattr(workflow, "resolve_" + method)
    for estimator in ("bias_estimate", "kde"):
        invalid = copy.deepcopy(cfg)
        invalid["analysis"]["method"] = estimator
        with pytest.raises(ValueError):
            resolve(invalid, system)
    invalid = copy.deepcopy(cfg)
    del invalid[method]["freeze_after_steps"]
    with pytest.raises(ValueError, match="freeze"):
        resolve(invalid, system)
    invalid = copy.deepcopy(cfg)
    invalid["analysis"]["bootstrap"] = 20
    with pytest.raises(ValueError, match="bootstrap"):
        resolve(invalid, system)
    invalid = copy.deepcopy(cfg)
    invalid[method]["freeze_equilibration_steps"] = 22
    with pytest.raises(ValueError, match="two saved"):
        resolve(invalid, system)


def test_final_bias_audit_accepts_constant_gauge_but_rejects_changed_shape():
    class Fixed:
        @staticmethod
        def physical_energy(values):
            return values[:, 0]**2
    stream = {"samples": np.array([[0.2], [0.5], [0.8]]), "sample_steps": np.array([8, 10, 12]),
              "sample_bias": np.array([0.04, 0.25, 0.64])+1000.}
    frozen_samples(stream, 2., equilibration_steps=0, freeze_after_steps=6, final_bias=Fixed())
    stream["sample_bias"][1] += 0.01
    with pytest.raises(ValueError, match="final fixed bias"):
        frozen_samples(stream, 2., equilibration_steps=0, freeze_after_steps=6, final_bias=Fixed())


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_checkpoint_rejects_changed_measure_or_freezing_protocol(method, tmp_path):
    workflow = importlib.import_module("pyrpmd.workflows." + method)
    cfg, system = _input(method, rate_flux=True)
    run = getattr(workflow, "run_" + method)
    run(cfg, system, tmp_path)
    density = copy.deepcopy(cfg)
    density["analysis"]["measure"] = "cv_density"
    later_freeze = copy.deepcopy(cfg)
    later_freeze[method]["freeze_after_steps"] = 15
    for changed in (density, later_freeze):
        with pytest.raises((ValueError, RuntimeError), match="physics differs"):
            run(changed, system, tmp_path)
