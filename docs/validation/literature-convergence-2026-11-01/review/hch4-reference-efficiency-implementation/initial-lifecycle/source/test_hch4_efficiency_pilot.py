"""Analytic/synthetic contracts only: never construct or evaluate CBE/PES."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

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


def install_analytic_stages(monkeypatch, *, negative=False, invalid=False):
    """Synthetic coordinate encoding of a known quadratic, never a molecule."""
    specs = {name: {"x0": [0.]*(9 if name == "entry" else 11),
                    "scales": [1.]*(9 if name == "entry" else 11),
                    "alpha": None if name == "entry" else float(name.split("_")[1])} for name in pilot.TARGETS}
    monkeypatch.setattr(pilot, "training_specs", lambda model: specs)
    def point(name, tag, x, model):
        x = np.asarray(x); q = np.zeros((6, 3)); q[0, 0] = .5*(x@x)-(x[0]**2 if negative else 0)
        return {"target": name, "tag": tag, "x": x.tolist(), "coordinates_bohr": q.tolist(),
                "domains": {key: not invalid for key in pilot.DOMAINS}, "common_zero_reason": None,
                "log_geometric_measure": 0., "log_training_measure": 0.}
    monkeypatch.setattr(pilot, "calibration_point", point)
    main, control, _, _ = synthetic_ledger()
    monkeypatch.setattr(pilot, "diagnostic_ledger", lambda models, model: (main, control))
    # The algebraic pipeline test needs event semantics, not thousands of real
    # fsync calls. Separate lifecycle tests exercise the actual durable markers.
    def fast_event(path, value):
        with Path(path).open("a") as stream: stream.write(json.dumps(value, sort_keys=True)+"\n")
    monkeypatch.setattr(pilot, "event", fast_event)
    return SimpleNamespace(beta=1., energy_offset=0.)


def test_full4000_synthetic_pipeline_freezes_each_ledger_before_evaluation(tmp_path, monkeypatch):
    model = install_analytic_stages(monkeypatch); seen = []
    guard = tmp_path/"guard.txt"; guard.write_text("immutable synthetic guard")
    def evaluator(q):
        n = len(seen)+1
        if n <= 711:
            assert (tmp_path/"calibration-initial-ledger.json").exists()
            if n == 1: assert len(pilot.read_json(tmp_path/"calibration-initial-ledger.json")["rows"]) == 711
            assert not (tmp_path/"frozen-proposal.json").exists()
        elif n <= 800:
            assert (tmp_path/"calibration-followup-ledger.json").exists()
            if n == 712: assert len(pilot.read_json(tmp_path/"calibration-followup-ledger.json")["rows"]) == 89
            assert not (tmp_path/"frozen-proposal.json").exists()
        else:
            # Read once at admission rather than repeatedly parsing all3200 rows.
            if n == 801:
                frozen = pilot.read_json(tmp_path/"frozen-proposal.json")
                assert frozen["calibration_PES_evaluations"] == 800
                assert frozen["fresh_diagnostic_draws_generated"] == 0
                ledger = pilot.read_json(tmp_path/"diagnostic-ledger.json")
                assert len(ledger["main_rows"]) == 3072 and len(ledger["control_rows"]) == 128
                assert ledger["proposal_sha256"] == pilot.sha(tmp_path/"frozen-proposal.json")
        seen.append(n); return float(q[0, 0])
    report = pilot.run_stages(tmp_path, model, evaluator, registration_sha="excluded-analytic-fixture",
                              guards={str(guard): pilot.sha(guard)}, cpu_clock=lambda: 0.)
    assert report["status"] == "complete_excluded_efficiency_pilot"
    assert len(seen) == report["attempted_PES_geometries"] == report["successful_PES_geometries"] == 4000
    assert report["main_draws_complete"] == 3072 and report["control_draws_complete"] == 128
    assert math.exp(report["analysis"]["dimensionless_flux_ratio_diagnostics"]["alpha_0.999"]["log_ratio_of_pooled_means"]) == pytest.approx(8/5)
    assert set(report) == set(pilot.OUTPUT_FIELDS)
    assert all(pilot.sha(tmp_path/name) == digest for name, digest in report["ledger_sha256"].items())
    assert all(word not in json.dumps(report) for word in ("cm3", "paper_pass", "k_qtst"))
    with pytest.raises(ValueError, match="already exists"):
        pilot.run_stages(tmp_path, model, evaluator, registration_sha="fixture", cpu_clock=lambda: 0.)


@pytest.mark.parametrize("failure", ["negative", "invalid_stencil", "PES_exception", "nonfinite"])
def test_failed_calibration_preserves_calls_and_never_draws_fresh_diagnostics(tmp_path, monkeypatch, failure):
    model = install_analytic_stages(monkeypatch, negative=failure == "negative", invalid=failure == "invalid_stencil")
    monkeypatch.setattr(pilot, "diagnostic_ledger", lambda *args: pytest.fail("a failed calibration generated fresh draws"))
    seen = []
    def evaluator(q):
        seen.append(1)
        if len(seen) == 713 and failure == "PES_exception": raise RuntimeError("excluded synthetic failure")
        if len(seen) == 713 and failure == "nonfinite": return float("nan")
        return float(q[0, 0])
    report = pilot.run_stages(tmp_path, model, evaluator, registration_sha="fixture", cpu_clock=lambda: 0.)
    assert report["status"] == "failed_calibration_or_evaluation"
    assert report["analysis"] is None and report["main_draws_complete"] == 0
    assert not (tmp_path/"frozen-proposal.json").exists() and not (tmp_path/"diagnostic-ledger.json").exists()
    expected = 711 if failure == "negative" else 0 if failure == "invalid_stencil" else 713
    assert report["attempted_PES_geometries"] == len(seen) == expected
    if failure in ("PES_exception", "nonfinite"):
        assert report["successful_PES_geometries"] == 712
        records = [json.loads(line) for line in (tmp_path/"calls.jsonl").read_text().splitlines()]
        assert records[-1]["event"] == "failure" and records[-1]["attempt_number"] == 713


@pytest.mark.parametrize("limit", ["CPU", "wall", "geometry"])
def test_budget_refuses_next_evaluation_before_any_evaluator_call(tmp_path, limit):
    budget = pilot.EvaluationBudget(tmp_path, lambda q: pytest.fail("budget admitted an evaluator"),
                                     cpu_clock=lambda: 50. if limit == "CPU" else 0.,
                                     wall_clock=lambda: 100., deadline=101. if limit == "wall" else 1000.)
    if limit == "geometry": budget.attempted = 4000
    with pytest.raises(pilot.AdmissionStop): budget.evaluate({"index": 0, "target": "fixture"}, "fixture")
    assert not (tmp_path/"calls.jsonl").exists()


def test_partial_diagnostic_progress_and_original_domain_zero_are_retained(tmp_path):
    model = SimpleNamespace(beta=1., energy_offset=0.)
    rows = [{"index": i, "target": "entry", "coordinates_bohr": [[0.]*3 for _ in range(6)],
             "common_zero_reason": "outside_outer" if i == 0 else None,
             "domains": {"inner": False, "original": i != 1, "outer": i != 0},
             "log_geometric_measure": 0., "log_proposal_density": 0.} for i in range(3)]
    calls = []
    def evaluator(q): calls.append(1); return 0.
    budget = pilot.EvaluationBudget(tmp_path, evaluator, cpu_clock=lambda: 0.)
    result = pilot.evaluate_diagnostics(rows, budget, model, "main_diagnostics")
    assert len(calls) == 2 and budget.completed_rows["main_diagnostics"] == 3
    assert result[0]["log_weight"]["outer"] is None
    assert result[1]["log_weight"]["original"] is None and result[1]["log_weight"]["outer"] == 0.


def test_existing_call_journal_refuses_implicit_recovery(tmp_path):
    (tmp_path/"calls.jsonl").write_text('{"event":"attempt"}\n')
    with pytest.raises(ValueError, match="no repeated attempt"):
        pilot.EvaluationBudget(tmp_path, lambda q: 0.)


@pytest.mark.skipif(sys.platform != "linux", reason="actual resource contract is Linux")
@pytest.mark.parametrize("role,expected_soft,expected_hard", [("prepare", 2, 2), ("run", 3, None), ("_worker", 55, 55)])
def test_CLI_limits_precede_scientific_imports_even_when_registration_is_missing(tmp_path, role, expected_soft, expected_hard):
    program = r'''
import atexit,json,os,resource,runpy,sys,time
source,role,missing=sys.argv[1:]
os.environ['PYRPMD_HCH4_EFFICIENCY_WORKER_CAP']='55'
os.environ['PYRPMD_HCH4_EFFICIENCY_DEADLINE']=str(time.monotonic()+30)
def report():
 print(json.dumps({'limits':list(resource.getrlimit(resource.RLIMIT_CPU)),
                   'numpy_imported':'numpy' in sys.modules,'scipy_imported':'scipy' in sys.modules}))
atexit.register(report)
sys.argv=[source,role,'--plan' if role=='prepare' else '--registration',missing]
if role=='prepare':sys.argv += ['--output',missing+'-output']
runpy.run_path(source,run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", program, str(pilot.SELF), role, str(tmp_path/"missing.json")],
                            text=True, capture_output=True, timeout=10, cwd=pilot.ROOT)
    assert result.returncode != 0
    record = json.loads(result.stdout)
    assert record["limits"][0] == expected_soft
    if expected_hard is not None: assert record["limits"][1] == expected_hard
    assert record["numpy_imported"] is False and record["scipy_imported"] is False


@pytest.mark.skipif(sys.platform != "linux", reason="private resource role is Linux")
@pytest.mark.parametrize("fault", ["cap", "deadline"])
def test_private_worker_bad_resource_admission_stops_before_heavy_imports(tmp_path, fault):
    environment = dict(os.environ, PYRPMD_HCH4_EFFICIENCY_WORKER_CAP="54" if fault == "cap" else "55",
                       PYRPMD_HCH4_EFFICIENCY_DEADLINE=str(time.monotonic()-1 if fault == "deadline" else time.monotonic()+30))
    result = subprocess.run([sys.executable, str(pilot.SELF), "_worker", "--registration", str(tmp_path/"absent")],
                            text=True, capture_output=True, env=environment, timeout=10)
    assert result.returncode != 0
    if fault == "deadline": assert "before scientific imports" in result.stderr


def test_private_worker_exact_owner_and_persistent_one_attempt(tmp_path, monkeypatch):
    registration = tmp_path/"registration.json"; pilot.write_new_json(registration, {"synthetic": True})
    deadline = time.monotonic()+30; token = "a"*32
    pilot.write_new_json(tmp_path/"RUN_ONCE.json", {"registration_sha256": pilot.sha(registration), "supervisor_PID": 1234,
                                                  "supervisor_process_token": token, "deadline_monotonic": deadline})
    monkeypatch.setattr(pilot, "ROLE", "_worker"); monkeypatch.setattr(pilot, "RESOURCE_ROLE", True)
    monkeypatch.setenv("PYRPMD_HCH4_EFFICIENCY_SUPERVISOR_TOKEN", token)
    monkeypatch.setenv("PYRPMD_HCH4_EFFICIENCY_DEADLINE", str(deadline))
    monkeypatch.setattr(pilot.os, "getppid", lambda: 0)
    with pytest.raises(ValueError, match="ownership"):
        pilot.claim_worker(registration, {"output": str(tmp_path)})
    monkeypatch.setattr(pilot.os, "getppid", lambda: 1234)
    pilot.claim_worker(registration, {"output": str(tmp_path)})
    with pytest.raises(FileExistsError): pilot.claim_worker(registration, {"output": str(tmp_path)})


def supervisor_fixture(tmp_path, monkeypatch):
    registration = tmp_path/"registration.json"; pilot.write_new_json(registration, {"synthetic": True})
    guard = tmp_path/"guard"; guard.write_text("unchanged fixture source")
    reg = {"output": str(tmp_path), "preparation_process_token": "e"*32}
    monkeypatch.setattr(pilot, "load_registration", lambda path: (reg, {}, {str(guard): pilot.sha(guard)}))
    monkeypatch.setattr(pilot, "ENTRY_MONOTONIC", time.monotonic())
    return registration, reg


@pytest.mark.skipif(sys.platform != "linux", reason="owned-session termination is Linux")
def test_wall_timeout_reaps_only_owned_inert_worker_and_refuses_retry(tmp_path, monkeypatch):
    registration, _ = supervisor_fixture(tmp_path, monkeypatch)
    command = [sys.executable, "-c", "import time; time.sleep(5)"]
    result = pilot.supervise(registration, command=command, enforce_resources=False, wall_seconds=2.15)
    assert result["stop"] == "wall_admission" and result["all_owned_workers_drained"]
    assert not result["complete_excluded_efficiency_pilot"]
    with pytest.raises(ProcessLookupError): os.kill(result["worker_PID"], 0)
    monkeypatch.setattr(pilot, "ENTRY_MONOTONIC", time.monotonic())
    with pytest.raises(FileExistsError):
        pilot.supervise(registration, command=command, enforce_resources=False, wall_seconds=2.15)


@pytest.mark.skipif(sys.platform != "linux", reason="owned-session termination is Linux")
def test_post_Popen_journal_error_still_reaps_the_exact_child(tmp_path, monkeypatch):
    registration, _ = supervisor_fixture(tmp_path, monkeypatch)
    original = pilot.write_new_json
    def failing(path, value):
        if Path(path).name == "child-started.json": raise OSError("synthetic post-spawn journal failure")
        return original(path, value)
    monkeypatch.setattr(pilot, "write_new_json", failing)
    result = pilot.supervise(registration, command=[sys.executable, "-c", "import time; time.sleep(5)"],
                              enforce_resources=False, wall_seconds=3)
    assert result["all_owned_workers_drained"] and result["failure"]["error_type"] == "OSError"
    assert not result["complete_excluded_efficiency_pilot"]
    with pytest.raises(ProcessLookupError): os.kill(result["worker_PID"], 0)


@pytest.mark.skipif(sys.platform != "linux", reason="owned-session termination is Linux")
def test_malformed_worker_report_remains_failed_with_durable_supervisor_receipt(tmp_path, monkeypatch):
    registration, _ = supervisor_fixture(tmp_path, monkeypatch)
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{broken')", str(tmp_path/"report.json")]
    result = pilot.supervise(registration, command=command, enforce_resources=False, wall_seconds=3)
    assert result["worker_exit_code"] == 0 and result["failure"]["report_validation_failed"]
    assert not result["complete_excluded_efficiency_pilot"] and (tmp_path/"supervisor.json").exists()


def test_same_invocation_cannot_charge_a_separate_preparation_budget(tmp_path, monkeypatch):
    registration, reg = supervisor_fixture(tmp_path, monkeypatch)
    reg["preparation_process_token"] = pilot.PROCESS_TOKEN
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("same invocation admitted a child"))
    with pytest.raises(ValueError, match="distinct invocations"):
        pilot.supervise(registration, enforce_resources=False)


def test_invocation_identity_does_not_consume_the_scientific_RNG():
    program = r'''
import json,numpy as np,uuid
np.random.seed(825623)
before=np.random.get_state()
from benchmarks.hch4 import efficiency_pilot as p
after=np.random.get_state()
print(json.dumps({'same':before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:],
                  'uuid_version':uuid.UUID(hex=p.PROCESS_TOKEN).version}))
'''
    result = subprocess.run([sys.executable, "-c", program], cwd=pilot.ROOT, text=True, capture_output=True, check=True, timeout=10)
    assert json.loads(result.stdout) == {"same": True, "uuid_version": 4}


def plan_fixture(tmp_path, monkeypatch):
    library = tmp_path/"synthetic.so"; library.write_bytes(b"INERT FIXTURE: NOT A LIBRARY")
    build = library.with_suffix(".build.json"); pilot.write_new_json(build, {"synthetic": True})
    method = tmp_path/"method.json"; pilot.write_new_json(method, {"synthetic": True})
    original = tmp_path/"legacy.py"; original.write_text("# excluded fixture only\n")
    design = tmp_path/"design.json"
    pilot.write_new_json(design, {"fixed_PES_identity": {"absolute_path": str(library), "sha256": pilot.sha(library)}})
    monkeypatch.setattr(pilot, "DESIGN", design); monkeypatch.setattr(pilot, "DESIGN_SHA", pilot.sha(design))
    monkeypatch.setattr(pilot, "METHOD_REVIEW", method); monkeypatch.setattr(pilot, "METHOD_REVIEW_SHA", pilot.sha(method))
    monkeypatch.setattr(pilot, "LEGACY", original); monkeypatch.setattr(pilot, "LEGACY_SHA", pilot.sha(original))
    source = tmp_path/"source.py"; source.write_text("# excluded source guard fixture\n")
    required = {str(source): pilot.sha(source), str(pilot.SELF): pilot.sha(pilot.SELF)}
    monkeypatch.setattr(pilot, "required_sources", lambda design: required.copy())
    environment = {"synthetic": "no scientific imports"}
    monkeypatch.setattr(pilot, "environment_identity", lambda: environment)
    def identity(path): return {"path": str(path), "sha256": pilot.sha(path), "bytes": path.stat().st_size}
    receipts = {}
    for key in ("synthetic_gate", "implementation_review", "resource_adoption"):
        path = tmp_path/(key+".json")
        value = {"synthetic": True}
        if key == "resource_adoption": value.update(authorize_single_bounded_pilot=True, available_CPU_count=1, maximum_PES_geometries=4000)
        pilot.write_new_json(path, value); receipts[key] = identity(path)
    guards = {**required, str(library): pilot.sha(library), str(build): pilot.sha(build),
              **{item["path"]: item["sha256"] for item in receipts.values()}}
    plan = {"schema": pilot.PLAN_SCHEMA, "parent_adoption": True, "design_sha256": pilot.DESIGN_SHA,
            "method_review_sha256": pilot.METHOD_REVIEW_SHA, "seeds": list(pilot.SEEDS),
            "geometry_counts": {"initial_calibration": 711, "followup_calibration": 89, "main": 3072, "control": 128, "maximum_total": 4000},
            "resources": {"CPUs": 1, "prepare_CPU_seconds": 2, "supervisor_CPU_seconds": 3, "worker_CPU_seconds": 55,
                          "sum_CPU_caps_seconds": 60, "combined_soft_CPU_seconds": 55, "wall_seconds": 120},
            "output_fields": list(pilot.OUTPUT_FIELDS), "automatic_retry": False, "automatic_extension": False,
            "cpu_affinity": min(os.sched_getaffinity(0)), "environment": environment,
            "library": identity(library), "library_build_receipt": identity(build),
            **receipts, "source_sha256": guards, "output": str(tmp_path/"separate-execution")}
    return plan, source


@pytest.mark.skipif(sys.platform != "linux", reason="concrete execution adoption is Linux")
@pytest.mark.parametrize("fault", [None, "adoption", "design", "seeds", "counts", "CPU", "source", "runtime",
                                   "automatic_retry", "receipt", "output", "model"])
def test_actual_adoption_rejects_contract_changes_before_PES_loading(tmp_path, monkeypatch, fault):
    plan, source = plan_fixture(tmp_path, monkeypatch)
    if fault == "adoption": plan["parent_adoption"] = False
    elif fault == "design": plan["design_sha256"] = "0"*64
    elif fault == "seeds": plan["seeds"].pop()
    elif fault == "counts": plan["geometry_counts"]["maximum_total"] = 4001
    elif fault == "CPU": plan["resources"]["worker_CPU_seconds"] = 56
    elif fault == "source": source.write_text("mutated fixture")
    elif fault == "runtime": plan["environment"] = {"changed": True}
    elif fault == "automatic_retry": plan["automatic_retry"] = True
    elif fault == "receipt": Path(plan["implementation_review"]["path"]).write_text("changed")
    elif fault == "output": plan["output"] = str(pilot.ROOT/"forbidden-output")
    elif fault == "model": plan["library"]["sha256"] = "0"*64
    path = tmp_path/"plan.json"; pilot.write_new_json(path, plan)
    if fault is None:
        loaded, guards = pilot.load_plan(path)
        assert loaded == plan and guards == plan["source_sha256"]
    else:
        with pytest.raises(ValueError): pilot.load_plan(path)
