"""Rate PMF support, log-domain weights, stream health and failure artifacts.

Regression coverage for the H + H2 / BKMP2 rate-workflow report: histogram
center boundaries, ESS collapse, displaced umbrella streams, the far branch
of the RPMDrate coordinate, frozen-production adaptive gates and structured
partial-history failures of history-dependent samplers.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.fes import mbar_rate_pmf
from pyrpmd.analysis.mbar import solve_mbar
from pyrpmd.analysis.rate_support import (
    classify_point,
    kish_ess,
    log_histogram,
    preflight_message,
    uniform_edges,
    window_anomalies,
)
from pyrpmd.analysis.umbrella import _histogram
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.input import InputError, rate_pmf_options
from pyrpmd.potentials import PESResult, Potential, PotentialError
from pyrpmd.system import System
from pyrpmd.workflows.rate_pmf import _adaptive_production_gate, _stream_state_checks

BETA = 1.0 / (200.0 * 3.166811563e-6)  # 200 K in 1/hartree


def _mbar_options(bins, ranges, search):
    return rate_pmf_options({
        "rate": {"pmf_method": "mbar", "pmf_search_range": search},
        "analysis": {"bins": bins, "ranges": [ranges]},
    })


@pytest.mark.parametrize("bins", [240, 480, 960])
def test_search_endpoint_on_last_edge_is_rejected_before_sampling(bins):
    with pytest.raises(InputError) as error:
        _mbar_options(bins, [-0.1, 1.1], [0.8, 1.1])
    message = str(error.value)
    width = 1.2 / bins
    assert "upper endpoint=1.1 lies above the last histogram bin center" in message
    assert f"{1.1 - width / 2:.10g}" in message      # reported last center
    assert "never clipped" in message
    assert f"{1.1 + width / 2:.10g}" in message      # required upper edge


@pytest.mark.parametrize("bins", [240, 480, 960])
def test_search_endpoint_on_center_or_interior_edge_is_accepted(bins):
    edges = uniform_edges(bins, [-0.1, 1.1])
    centers = 0.5 * (edges[1:] + edges[:-1])
    on_center = float(centers[-1])
    options = _mbar_options(bins, [-0.1, 1.1], [0.8, on_center])
    assert options["search_range"] == [0.8, on_center]
    interior_edge = float(edges[-20])
    assert _mbar_options(bins, [-0.1, 1.1], [0.8, interior_edge])["search_range"][1] == interior_edge
    # Extending the range by half a bin admits 1.1 as a center endpoint.
    wide = rate_pmf_options({
        "rate": {"pmf_method": "mbar", "pmf_search_range": [0.8, 1.1]},
        "analysis": {"bins": bins + 1, "ranges": [[-0.1, 1.1 + 1.2 / bins]]},
    })
    assert wide["search_range"] == [0.8, 1.1]


def test_reference_and_lower_endpoint_preflight_and_classification():
    edges = uniform_edges(240, [-0.1, 1.1])
    assert preflight_message("reference", 0.0, edges, option="x") is None
    message = preflight_message("lower", -0.1, edges, option="x")
    assert "below the first histogram bin center" in message
    with pytest.raises(InputError, match="reference xi=0 lies below"):
        rate_pmf_options({"rate": {"pmf_method": "mbar", "dividing_surface": 1.0},
                          "analysis": {"bins": 100, "ranges": [[0.0, 1.2]]}})
    grid = 0.5 * (edges[1:] + edges[:-1])
    mask = np.zeros(len(grid), bool)
    assert classify_point(grid, mask, 1.1)["status"] == "outside_histogram_centers"
    assert classify_point(grid, mask, grid[-1])["status"] == "on_center"
    assert classify_point(grid, mask, edges[-2])["status"] == "between_centers"
    mask[-2] = True
    masked = classify_point(grid, mask, edges[-2])
    assert masked["status"] == "masked" and masked["masked_bins"] == [len(grid) - 2]


def _streams(seed=7, displaced=None, reduced_stiffness=400.0):
    rng = np.random.default_rng(seed)
    centers = np.linspace(-0.05, 1.1, 24)
    kappa = reduced_stiffness / BETA
    sigma = 1 / np.sqrt(BETA * kappa)
    samples = [c + sigma * rng.standard_normal(800) for c in centers]
    if displaced is not None:
        samples[-1] = displaced + 0.05 * rng.standard_normal(800)
    return samples, centers, np.full(len(centers), kappa)


def test_histogram_endpoint_outside_centers_is_reported_not_clipped():
    samples, centers, kappa = _streams()
    result = mbar_rate_pmf(samples, centers, kappa, BETA, bins=240,
                           ranges=[-0.1, 1.1], search_range=[0.8, 1.1], bootstrap=0)
    assert not result["ready"]
    assert result["status"] == "search_upper_outside_histogram_centers"
    support = result["support"]
    assert support["histogram"]["last_center"] == pytest.approx(1.0975)
    assert support["histogram"]["last_edge"] == pytest.approx(1.1)
    assert len(support["histogram"]["edges"]) == 241
    upper = support["search_endpoints"]["upper"]
    assert upper["status"] == "outside_histogram_centers" and upper["side"] == "above"
    assert upper["nearest_center"] == pytest.approx(1.0975)
    assert result["xi_star"] is None


def test_log_histogram_retains_mass_spanning_more_than_1000_log_units():
    x = np.repeat([0.05, 0.15, 0.25], 10)
    lw = np.concatenate([np.zeros(10), np.full(10, -1000.0), np.full(10, -1500.0)])
    lw -= np.log(10)
    edges = np.array([0.0, 0.1, 0.2, 0.3])
    _, _, _, _, linear_mask = _histogram(x[:, None], np.exp(lw), [edges], 1.0, 1)
    assert linear_mask.tolist() == [False, True, True]   # ordinary exp underflow
    log_mass, log_density, count, mask = log_histogram(x, lw, edges)
    assert not mask.any()
    np.testing.assert_allclose(log_mass, [0.0, -1000.0, -1500.0], atol=1e-12)
    np.testing.assert_allclose(log_density, log_mass - np.log(0.1), atol=1e-12)
    assert count.tolist() == [10, 10, 10]
    # The last edge is inclusive and out-of-range samples are dropped.
    _, _, count, _ = log_histogram([0.3, 0.31, -0.01], [0.0, 0.0, 0.0], edges)
    assert count.tolist() == [0, 0, 1]


def test_kish_ess_is_stable_for_extreme_log_weights():
    assert kish_ess(np.zeros(50)) == pytest.approx(50)
    collapsed = np.concatenate([[0.0], np.full(10_000, -1200.0)])
    assert kish_ess(collapsed) == pytest.approx(1.0)
    w = np.random.default_rng(3).random(40)
    assert kish_ess(np.log(w)) == pytest.approx(w.sum() ** 2 / (w @ w))


def test_ess_collapse_rejects_the_rate_but_keeps_a_diagnostic_curve():
    samples, centers, kappa = _streams()

    def collapsed_solver(*args, **kwargs):
        fit = solve_mbar(*args, **kwargs)
        lw = np.full(len(fit.weights), -1200.0)
        lw[np.argmax(fit.weights)] = 0.0
        lw -= np.logaddexp.reduce(lw)
        fit.log_weights = lw
        fit.weights = np.exp(lw)
        return fit

    from pyrpmd.analysis.rate_pmf import mbar_rate_pmf as raw

    result = raw(samples, centers, kappa, BETA, bins=60, ranges=[-0.1, 1.2],
                 xi_star=1.0, bootstrap=0, min_count=1, _solve_mbar=collapsed_solver)
    weights = result["weight_diagnostics"]
    assert weights["global_kish_ess"] == pytest.approx(1.0)
    assert weights["max_normalized_weight"] == pytest.approx(1.0)
    assert weights["log_weight_span"] >= 1000
    assert len(weights["per_stream"]) == len(samples)
    assert not result["ready"]
    assert result["status"] == "weight_ess_collapse"
    assert result["free_energy_barrier_hartree"] is None
    assert np.isfinite(result["diagnostic_free_energy_barrier_hartree"])
    gate = {g["name"]: g for g in result["gates"]}["global_weight_ess"]
    assert gate["passed"] is False and gate["minimum"] == 10.0
    # A healthy fit of the same data passes all gates.
    healthy = mbar_rate_pmf(samples, centers, kappa, BETA, bins=60, ranges=[-0.1, 1.2],
                            xi_star=1.0, bootstrap=0, min_count=1)
    assert healthy["ready"], healthy["status"]
    assert all(g["passed"] for g in healthy["gates"])
    assert healthy["weight_ess"] > 10


def test_stream_far_from_its_umbrella_center_is_a_hard_failure():
    samples, centers, kappa = _streams(displaced=0.53, reduced_stiffness=4000.0)
    records = window_anomalies(samples, centers, kappa, BETA)
    assert [r["stream"] for r in records] == [len(samples) - 1]
    record = records[0]
    assert record["displacement_in_sigma"] > 20
    expected = 0.5 * kappa[-1] * (0.53 - 1.1) ** 2 * 627.509474
    assert record["bias_at_mean_kcal_mol"] == pytest.approx(expected, rel=0.02)
    result = mbar_rate_pmf(samples, centers, kappa, BETA, bins=60, ranges=[-0.1, 1.2],
                           xi_star=1.0, bootstrap=0, min_count=1)
    assert not result["ready"]
    assert result["status"] == "umbrella_stream_anomaly"
    assert result["window_anomalies"][0]["stream"] == len(samples) - 1
    disabled = mbar_rate_pmf(samples, centers, kappa, BETA, bins=60, ranges=[-0.1, 1.2],
                             xi_star=1.0, bootstrap=0, min_count=1,
                             max_center_displacement_sigma=None)
    assert disabled["window_anomalies"] == []


def _h3_coordinate():
    masses = np.full(3, 1837.15)
    ts = np.array([[-1.757, 0, 0], [0, 0, 0], [1.757, 0, 0]])
    return RPMDrateCoordinate([[0], [1, 2]], masses, ts, [[0, 1]], [[1, 2]], 10.0,
                              equivalent_paths=[[0, 2, 1]])


def test_rpmdrate_xi_far_branch_is_detected_from_the_final_ring():
    coordinate = _h3_coordinate()
    regular = np.array([[-1.5, 0, 0], [0, 0, 0], [2.0, 0, 0]])
    far = np.array([[-1.4, 0, 0], [0, 0, 0], [40.0, 0, 0]])
    assert coordinate.branch(regular)["branch"] == "regular"
    far_branch = coordinate.branch(np.repeat(far[None], 4, axis=0))
    assert far_branch["branch"] == "beyond_singular_surface"
    # Past the singular surface xi returns to an innocent-looking value.
    assert 0 < far_branch["xi"] < 1
    assert far_branch["xi"] == pytest.approx(coordinate.value(far[None])[0])
    streams = [
        {"window": 0, "replica": 0, "center": [1.1], "samples": np.array([[1.09], [1.1]]),
         "last_q": np.repeat(regular[None], 4, axis=0)},
        {"window": 1, "replica": 0, "center": [1.1], "samples": np.array([[0.5], [0.4]]),
         "last_q": np.repeat(far[None], 4, axis=0)},
    ]
    checks = _stream_state_checks(streams, coordinate)
    assert [c["passed"] for c in checks] == [True, False]
    assert checks[1]["reason"] == "rpmdrate_xi_beyond_singular_surface"
    assert checks[1]["final_centroid"]["denominator"] < 0


def test_adaptive_pilot_pass_production_fail_is_not_ready():
    summary = {"adaptive_umbrella": {
        "pilot_target_met": True, "production_target_met": False,
        "production_spanning_overlap": 0.0178,
        "production_overlap_connected_at_target": False,
        "production_effective_samples_target_met": True,
        "production_effective_samples_per_window": [40.0, 50.0],
        "production_analysis_overlap_connected": True,
        "production_coverage": {"target_met": True},
    }}
    gate = _adaptive_production_gate(summary)
    assert gate["passed"] is False and gate["pilot_target_met"] is True
    assert gate["spanning_overlap"] == 0.0178
    assert gate["coverage_target_met"] is True
    assert _adaptive_production_gate({"warnings": []}) is None


class _DomainPotential(Potential):
    """Two-atom harmonic bond that reports a BKMP2-like domain failure."""

    def __init__(self, fail_after):
        self.calls = 0
        self.fail_after = fail_after

    def evaluate(self, x):
        x = np.asarray(x, float)
        self.calls += 1
        if self.calls > self.fail_after:
            raise PotentialError("geometry outside domain: every pair distance must be >= 0.2 bohr (status=1)")
        d = x[:, 1] - x[:, 0]
        r = np.linalg.norm(d, axis=-1)
        energy = 0.1 * (r - 1.4) ** 2
        g = (0.2 * (r - 1.4) / r)[:, None] * d
        forces = np.stack([g, -g], axis=1)
        return PESResult(energy, forces)


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_domain_failure_writes_structured_partial_history(tmp_path, monkeypatch, method):
    import importlib

    workflow = importlib.import_module(f"pyrpmd.workflows.{method}")
    monkeypatch.setattr(workflow, "make_task_potential",
                        lambda spec, system: _DomainPotential(fail_after=25))
    system = System(("H", "H"), np.array([1837.15, 1837.15]),
                    np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]))
    config = {
        "task": method, "beta": 50.0, "beads": 3, "dt_au": 2.0, "seed": 5, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 2, "production_steps": 40, "sample_interval": 2,
                     "andersen_interval": 4, "replicas": 1},
        "cvs": [{"name": "r", "kind": "distance", "atoms": [0, 1]}],
        "analysis": {"bins": [6], "bootstrap": 0},
        "output": {"checkpoint_steps": 4},
    }
    if method == "metadynamics":
        config["metadynamics"] = {"mode": "well_tempered", "sigma": [0.1], "height": 0.001,
                                  "pace": 3, "bias_factor": 5.0}
        run = workflow.run_metadynamics
    else:
        config["opes"] = {"sigma": [0.1], "barrier": 0.01, "pace": 3, "bias_factor": 5}
        run = workflow.run_opes
    with pytest.raises(Exception, match="outside domain"):
        run(config, system, tmp_path)
    records = list((tmp_path / "failures").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["status"] == "history_failed"
    assert record["history"] == {"method": method, "replica": 0}
    assert record["error"]["type"] == "PotentialError"
    state = record["state"]
    assert state["role"] == "last_committed_state_before_failure"
    assert 0 < state["elapsed_steps"] < 42 and state["timestep_au"] == 2.0
    geometry = record["geometry"]
    assert geometry["minimum_pair"] == [0, 1] and geometry["minimum_bead_pair_distance"] > 0
    assert record["centroid_cv"] is not None and len(record["centroid_cv"]) == 1
    assert record["bias_state"]["kind"] == method
    assert "python_steps" in record["propagation"]
    checkpoint = Path(record["checkpoint"]["path"])
    assert checkpoint.is_file()
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert int(saved["elapsed"]) == state["elapsed_steps"]
        assert int(saved["step"]) == state["step"]
    assert "must not be selected" in record["partial_history_policy"]
