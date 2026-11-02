"""Analytic/synthetic contracts only: never construct or evaluate CBE/PES."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, multivariate_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.hch4 import efficiency_pilot as pilot


@pytest.fixture
def model():
    radius = 2.2
    q = np.zeros((6, 3)); q[3] = [0, 0, radius]
    q[0] = radius*np.array([math.sqrt(8/9), 0, -1/3])
    for atom, phi in ((2, 2*math.pi/3), (4, -2*math.pi/3)):
        q[atom] = radius*np.array([math.sqrt(8/9)*math.cos(phi), math.sqrt(8/9)*math.sin(phi), -1/3])
    q[5] = q[3]+[0, 0, 1.5]
    return pilot.legacy().Model(np.array([1., 12., 1., 1., 1., 1.]), 2., -1.3, q, energy_offset=0.)


@pytest.fixture
def proposal_models(model):
    specs = pilot.training_specs(model); result = {}
    for name, spec in specs.items():
        result[name] = {"mean_x": spec["x0"], "covariance_x": np.diag(np.asarray(spec["scales"])**2).tolist()}
    return result


@pytest.mark.parametrize("dimension", [9, 11])
@pytest.mark.parametrize("student", [False, True])
def test_elliptical_density_matches_independent_normal_and_Student_oracles(dimension, student):
    mean = np.linspace(-.3, .5, dimension)
    covariance = np.diag(np.linspace(.4, 1.2, dimension))+.02*np.ones((dimension, dimension))
    x = np.linspace(.1, -.4, dimension)
    expected = (multivariate_t.logpdf(x, loc=mean, shape=(12/5)*covariance, df=5) if student else
                multivariate_normal.logpdf(x, mean=mean, cov=covariance))
    assert pilot.log_density(x, mean, covariance, student=student) == pytest.approx(expected, abs=2e-13)
    if student:
        np.testing.assert_allclose(5/3*((12/5)*covariance), 4*covariance, atol=1e-15)


@pytest.mark.parametrize("t", [[0, 0], [.2, -1.1], [3, 2], [-4, .3]])
def test_stereographic_jacobian_against_independent_cartesian_area(t):
    t = np.asarray(t, float); u, J = pilot.sphere(t); h = 1e-6
    derivatives = [(pilot.sphere(t+h*np.eye(2)[i])[0]-pilot.sphere(t-h*np.eye(2)[i])[0])/(2*h) for i in range(2)]
    assert np.linalg.norm(np.cross(*derivatives)) == pytest.approx(J, rel=3e-9)
    recovered, recovered_J = pilot.inverse_sphere(u)
    np.testing.assert_allclose(recovered, t, rtol=2e-14, atol=1e-14)
    assert recovered_J == pytest.approx(J, rel=2e-14)


@pytest.mark.parametrize("student", [False, True])
def test_pushforward_sphere_density_has_unit_integral_without_extra_Jacobian(student):
    def integrand(cosine):
        if cosine == -1:
            return 0.
        t = np.array([math.sqrt((1-cosine)/(1+cosine)), 0.])
        return 2*math.pi*math.exp(pilot.log_density(t, np.zeros(2), np.eye(2)*.3, student=student))/(1+cosine)**2
    integral, error = quad(integrand, -1, 1, epsabs=2e-11, epsrel=2e-11)
    assert integral == pytest.approx(1., abs=2e-11)
    assert error < 1e-9


def test_full_stratified_mixture_recovers_known_truncated_integral_not_component_ratios():
    weights = [1/8, 3/4, 1/8]
    def densities(x):
        return np.array([1/6 if abs(x) <= 3 else 0,
                         math.exp(pilot.log_density([x], [0.], [[1.]])),
                         math.exp(pilot.log_density([x], [0.], [[1.]], student=True))])
    contributions = [weight*quad(lambda x: densities(x)[j]*x*x/(weights@densities(x)), -1, 2,
                                 epsabs=1e-12, epsrel=1e-12)[0] for j, weight in enumerate(weights)]
    assert sum(contributions) == pytest.approx(3., abs=2e-12)
    # The finite target support is retained; no conditional acceptance factor.
    assert sum(contributions) != pytest.approx(3/3)


def test_uniform_floor_is_positive_even_at_south_pole_and_reflection_preserves_mixture(proposal_models):
    z = np.asarray(proposal_models["alpha_0.999"]["mean_x"][:9])
    u = np.array([.2, -.3, math.sqrt(.87)])
    first = pilot.mixture_log_density(z, u, proposal_models)
    mirrored = z.copy(); mirrored[[4, 7]] *= -1; reflected_u = u.copy(); reflected_u[1] *= -1
    assert pilot.mixture_log_density(mirrored, reflected_u, proposal_models) == pytest.approx(first, abs=1e-13)
    volume = (5-1.2)**2*2*10**6
    assert pilot.mixture_log_density(z, [0., 0., -1.], proposal_models) == pytest.approx(-math.log(8*volume*4*math.pi))


def test_new_geometry_keeps_four_active_charts_and_single_entry_analytic_normalization(model):
    old = pilot.legacy(); q = model.reference.copy(); radius = np.linalg.norm(q[3]); rho = radius+model.c
    nodes, weights = np.polynomial.legendre.leggauss(48)
    numerator = 0.
    for h in pilot.CHANNELS:
        z = old.to_chart(q, h); values = []
        for cosine in nodes:
            row = pilot.geometry(z, [math.sqrt(1-cosine*cosine), 0., cosine], h, 1., model)
            assert row["domains"]["original"]
            values.append(math.exp(row["log_geometric_measure"]))
        numerator += 2*math.pi*(weights@values)
    z = old.to_chart(q, 3)
    entry = pilot.geometry(z, [0., 0., 1.], 3, None, model)
    denominator = 4*math.pi*math.exp(entry["log_geometric_measure"])
    A, B = 3+1/12, 2.
    angular = 4*math.pi*((A+B)**1.5-(A-B)**1.5)/(3*B*math.sqrt(2*math.pi*model.beta))
    expected = 4*radius**4*rho**2*angular
    assert numerator == pytest.approx(expected, rel=2e-14)
    assert denominator == pytest.approx(4*math.pi*radius**4*30**2/math.sqrt(2*math.pi*model.beta*model.reduced_mass), rel=2e-14)


def test_nested_domains_do_not_change_original_cutoff_or_discard_outer_diagnostic(model):
    local = replace(model, c=.1)
    z = pilot.legacy().to_chart(model.reference, 3); z[0] = 1.3
    row = pilot.geometry(z, [0., 0., 1.], 3, .999, local)
    assert row["common_zero_reason"] is None
    assert row["domains"] == {"inner": False, "original": False, "outer": True}
    outside = z.copy(); outside[0] = 5.1
    rejected = pilot.geometry(outside, [0., 0., 1.], 3, .999, local)
    assert rejected["common_zero_reason"] == "outside_outer_methane_domain"
    assert not any(rejected["domains"].values())


@pytest.mark.parametrize("dimension", [9, 11])
def test_full_Hessian_single_Newton_back_transform_and_unit_probes_match_exact_quadratic(dimension):
    H = np.diag(np.linspace(1., 2., dimension))+.03*np.ones((dimension, dimension))
    gradient = np.linspace(-.1, .1, dimension)
    values = {tag: float(1.7+gradient@y+.5*y@H@y) for tag, y in pilot.stencil(dimension)}
    scales = np.linspace(.1, .25, dimension); x0 = np.linspace(1., 2., dimension)
    fitted = pilot.fit_quadratic({"x0": x0.tolist(), "scales": scales.tolist()}, values)
    expected_delta = -np.linalg.solve(H, gradient)
    np.testing.assert_allclose(fitted["H_y"], H, atol=5e-12)
    np.testing.assert_allclose(fitted["mean_x"], x0+scales*expected_delta, atol=1e-12)
    expected_covariance = np.linalg.inv(H)*scales[:, None]*scales[None, :]
    np.testing.assert_allclose(fitted["covariance_x"], expected_covariance, atol=1e-13)
    axes = np.asarray(fitted["unit_probe_axes_x"])
    np.testing.assert_allclose(axes.T@np.linalg.inv(expected_covariance)@axes, np.eye(dimension), atol=5e-12)
    assert fitted["eigenvalues_y"] == sorted(fitted["eigenvalues_y"])


@pytest.mark.parametrize("failure", ["negative", "singular", "refinement", "large_shift", "nonfinite"])
def test_failed_Hessian_or_shift_never_clips_or_switches_proposal(failure):
    d = 9; H = np.eye(d); g = np.zeros(d)
    if failure == "negative": H[0, 0] = -1.
    if failure == "singular": H[0, 0] = 1e-8
    if failure == "large_shift": g[0] = 4.
    values = {tag: float(g@y+.5*y@H@y) for tag, y in pilot.stencil(d)}
    if failure == "refinement": values["refine:0:1"] += .001
    if failure == "nonfinite": values["diag:0:1"] = float("nan")
    with pytest.raises(ValueError): pilot.fit_quadratic({"x0": [0.]*d, "scales": [1.]*d}, values)


def test_complete_staged_geometry_counts_and_parameter_independent_ledger(model, proposal_models):
    specs = pilot.training_specs(model)
    initial = pilot.initial_calibration_ledger(specs, model)
    assert len(initial) == 711
    models = {}
    for name, spec in specs.items():
        values = {tag: float(.5*y@y) for tag, y in pilot.stencil(len(spec["x0"]))}
        models[name] = pilot.fit_quadratic(spec, values)
    followup = pilot.followup_calibration_ledger(models, model)
    assert len(followup) == 89 and [row["index"] for row in followup] == list(range(711, 800))
    assert sum("reflected_original_tag" in row for row in followup) == 24
    main, control = pilot.diagnostic_ledger(proposal_models, model)
    assert len(main) == 3072 and len(control) == 128
    mock_results = [{"index": row["index"]} for row in main]
    mock_control = [{"index": row["index"]} for row in control]
    pilot.validate_fresh_membership(main, control, mock_results, mock_control)
    again, repeated_control = pilot.diagnostic_ledger(proposal_models, model)
    assert main == again and control == repeated_control


def synthetic_ledger():
    """Pure algebraic rows; no physical coordinates or PES are evaluated."""
    rows, controls = [], []
    common = {"coordinates_bohr": [[0.]*3 for _ in range(6)], "common_zero_reason": None,
              "domains": {name: True for name in pilot.DOMAINS}, "minimum_pair_bohr": 2.,
              "CH_radii_bohr": [2.]*4, "shape": [2., 2., -.3]+[0.]*6,
              "direction": [0., 0., 1.], "log_proposal_density": 0., "separation_bohr": 30.}
    for offset, seed in enumerate(pilot.SEEDS):
        TS_counts = [("uniform", 4), *[(f"{kind}:{target}:{mirror}", 6 if kind == "normal" else 1)
                     for kind in ("normal", "student5") for target in pilot.TARGETS[:2] for mirror in ("positive", "negative")]]
        for h in pilot.CHANNELS:
            for component, count in TS_counts:
                for draw in range(count):
                    for target in pilot.TARGETS[:2]:
                        rows.append({**common, "index": len(rows), "seed": seed, "channel": h, "target": target,
                                     "latent_id": f"{seed}:{h}:{component}:{draw}", "component": component,
                                     "component_draw": draw, "log_geometric_measure": math.log(2.)})
        entry_counts = [("uniform", 16), *[(f"{kind}:entry:{mirror}", 48 if kind == "normal" else 8)
                        for kind in ("normal", "student5") for mirror in ("positive", "negative")]]
        for component, count in entry_counts:
            for draw in range(count):
                row = {**common, "index": len(rows), "seed": seed, "channel": None, "target": "entry",
                       "latent_id": f"{seed}:entry:{component}:{draw}", "component": component,
                       "component_draw": draw, "log_geometric_measure": math.log(1. if offset < 4 else 9.)}
                rows.append(row)
                limit = 2 if component == "uniform" else 6 if component.startswith("normal:") else 1
                if draw < limit:
                    controls.append({**row, "paired_main_index": row["index"], "separation_bohr": 60.,
                                     "log_geometric_measure": row["log_geometric_measure"]+math.log(4.)})
    for number, row in enumerate(controls): row["index"] = 3072+number
    results = [{"index": row["index"], "energy_hartree": 0., "log_weight": {name: row["log_geometric_measure"] for name in pilot.DOMAINS}} for row in rows]
    control_results = [{"index": row["index"], "energy_hartree": 0., "log_weight": {name: row["log_geometric_measure"] for name in pilot.DOMAINS}} for row in controls]
    return rows, controls, results, control_results


def test_whole_seed_pooled_ratio_four_channels_and_matched16_control_are_analytic():
    rows, controls, results, control_results = synthetic_ledger()
    record = pilot.analyze(rows, controls, results, control_results)
    ratio = record["dimensionless_flux_ratio_diagnostics"]["alpha_0.999"]
    assert math.exp(ratio["log_ratio_of_pooled_means"]) == pytest.approx(8/5)
    assert math.exp(ratio["log_ratio_of_pooled_means"]) != pytest.approx((8+8/9)/2)
    assert ratio["seed_vectors"] == 8 and ratio["degrees_of_freedom"] == 7
    expected_influence = 1-np.array([1.]*4+[9.]*4)/5
    np.testing.assert_allclose(ratio["seed_influence"], expected_influence, atol=1e-15)
    assert ratio["nominal_delta_standard_error"] == pytest.approx(expected_influence.std(ddof=1)/math.sqrt(8))
    assert record["paired_alpha_log_integral_difference"]["nominal_delta_standard_error"] == 0.
    separation = record["entry_only_separation_control"]
    assert separation["log_ratio_of_pooled_means"] == pytest.approx(0., abs=2e-15)
    assert separation["matched_draws_per_seed"] == 16 and separation["primary_entry_denominator_draws_per_seed"] == 128
    assert record["original_domain_efficiency_screen_passed"]


@pytest.mark.parametrize("failure", ["missing", "seed", "stratum", "pair", "control_source", "control_index"])
def test_partial_or_reassigned_fresh_cohorts_are_refused(failure):
    rows, controls, results, control_results = synthetic_ledger()
    if failure == "missing": rows.pop()
    elif failure == "seed": rows[0]["seed"] = 0
    elif failure == "stratum": rows[0]["component"] = "wrong"
    elif failure == "pair": rows[1]["shape"] = [3.]*9
    elif failure == "control_source": controls[0]["paired_main_index"] = 0
    else: controls[0]["component_draw"] = 100
    with pytest.raises(ValueError): pilot.validate_fresh_membership(rows, controls, results, control_results)


def test_unconditional_zeros_and_unresolved_integral_are_not_replaced():
    summary = pilot.concentration([None, math.log(2.)])
    assert summary["log_mean"] == pytest.approx(0.) and summary["draws"] == 2
    assert pilot.paired_log_ratio([None]*8, [0.]*8)["resolved"] is False
    rows, controls, results, control_results = synthetic_ledger()
    results[0]["log_weight"]["original"] = None
    result = pilot.integral_logs(rows, results)
    assert math.exp(result["alpha_0.999"]) == pytest.approx(8-2/256)
