"""Analytic/synthetic checks only: this module never loads the external PES."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.hch4 import reference_pilot as pilot


@pytest.fixture(scope="module")
def model():
    reference = np.asarray(pilot.read_json(pilot.ROOT/"benchmarks/hch4/geometries.json")["official_input"]["coordinates"])*pilot.ANGSTROM_TO_AU
    masses = np.array([1.00782503207, 12., 1.00782503207, 1.00782503207, 1.00782503207, 1.00782503207])*pilot.AMU_TO_AU
    c = np.linalg.norm(reference[5]-reference[3])-np.linalg.norm(reference[3]-reference[1])
    return pilot.Model(masses, 1/(300*pilot.KB_AU), float(c), reference)


@pytest.fixture(scope="module")
def ledger(model):
    return pilot.build_ledger(model)


def test_shape_mixture_is_normalized_full_density_with_uniform_support(model):
    means = pilot.proposal_means(model, 3)
    plus, minus, sigma = means
    rng = np.random.default_rng(11)
    low, high = pilot.box_bounds(model)
    volume = np.prod(high-low)
    for z in [plus, minus, low, high, *rng.uniform(low, high, size=(24, 9)), plus+20*sigma]:
        normal = (.4*multivariate_normal.pdf(z, mean=plus, cov=np.diag(sigma**2))
                  +.4*multivariate_normal.pdf(z, mean=minus, cov=np.diag(sigma**2)))
        uniform = .2/volume if np.all((z >= low)&(z <= high)) else 0.
        assert np.exp(pilot.shape_logpdf(z, model, means)) == pytest.approx(normal+uniform, rel=2e-14)
        if uniform:
            assert pilot.shape_logpdf(z, model, means) >= np.log(.2/volume)-1e-14
    # Independent Cartesian product integration of each normal component.
    normal_integrals = [quad(lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi), -np.inf, np.inf,
                             epsabs=1e-12)[0] for _ in range(9)]
    assert .2+.4*np.prod(normal_integrals)+.4*np.prod(normal_integrals) == pytest.approx(1., abs=2e-14)


def test_direction_mixture_density_and_vmf_inverse_are_normalized():
    nodes, weights = np.polynomial.legendre.leggauss(80)
    for entry in (False, True):
        density = [np.exp(pilot.direction_logpdf(np.array([np.sqrt(1-x*x), 0, x]), entry=entry)) for x in nodes]
        assert 2*np.pi*np.dot(weights, density) == pytest.approx(1., abs=3e-14)
    for value in [0., .001, .1, .5, .99, 1.]:
        cosine = pilot.vmf_inverse(value)
        cdf = (np.exp(8*cosine)-np.exp(-8))/(np.exp(8)-np.exp(-8))
        assert cdf == pytest.approx(value, abs=3e-15)


def test_both_mirrors_are_proposals_not_additional_symmetry_factors(model):
    plus, minus, sigma = pilot.proposal_means(model, 3)
    assert np.array_equal(plus[[4, 7]], -minus[[4, 7]])
    assert np.array_equal(plus[[0, 1, 2, 3, 5, 6, 8]], minus[[0, 1, 2, 3, 5, 6, 8]])
    assert pilot.shape_logpdf(plus, model, (plus, minus, sigma)) == pilot.shape_logpdf(minus, model, (plus, minus, sigma))
    for h in pilot.HYDROGENS:
        means = pilot.proposal_means(model, h)
        q = pilot.from_chart(means[0], h)
        np.testing.assert_allclose(pilot.to_chart(q, h), means[0], rtol=0, atol=1e-15)


def test_rotation_chart_measure_has_independent_analytic_box_integral(model):
    low, high = pilot.box_bounds(model)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    radii = (nodes+1)*(high[0]-low[0])/2+low[0]
    quadrature = weights*(high[0]-low[0])/2
    integral = sum(wr*ws*pilot.shape_jacobian(np.r_[r, s, np.zeros(7)])
                   for r, wr in zip(radii, quadrature) for s, ws in zip(radii, quadrature))
    integral *= 2*(2*model.radius_high)**6
    exact = ((model.radius_high**3-model.radius_low**3)/3)**2*2*(2*model.radius_high)**6
    assert integral == pytest.approx(exact, rel=1e-15)


def test_four_disjoint_max_surfaces_and_single_entry_match_analytic_sphere_integrals(model):
    # This synthetic alpha=1 target has V=0. Its angular flux integral is
    # analytic, so no PES/importance approximation supplies the expected answer.
    means_entry = pilot.proposal_means(model, 3, entry=True)
    qtetra = pilot.from_chart(means_entry[0], 3)
    radius = np.linalg.norm(qtetra[3])
    rho = radius+model.c
    assert 2*rho < min(np.linalg.norm(qtetra[a]-qtetra[b]) for a in pilot.HYDROGENS for b in pilot.HYDROGENS if a < b)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    integrals = []
    for h in pilot.HYDROGENS:
        z = pilot.to_chart(qtetra, h)
        values = []
        for cosine in nodes:
            u = np.array([np.sqrt(1-cosine*cosine), 0, cosine])
            row = pilot.candidate(z, u, h, 1., model, pilot.proposal_means(model, h))
            assert row["zero_reason"] is None and row["active_channel"] == h
            assert row["rho_bohr"] == pytest.approx(rho, abs=1e-14)
            values.append(np.exp(row["log_geometric_measure"]))
        integrals.append(2*np.pi*np.dot(weights, values))
    entry_values = []
    for cosine in nodes:
        u = np.array([np.sqrt(1-cosine*cosine), 0, cosine])
        row = pilot.candidate(means_entry[0], u, 3, None, model, means_entry)
        assert row["zero_reason"] is None
        entry_values.append(np.exp(row["log_geometric_measure"]))
    entry = 2*np.pi*np.dot(weights, entry_values)
    m, carbon = model.masses[0], model.masses[1]
    A, B = 3/m+1/carbon, 2/m
    angular = 4*np.pi*((A+B)**1.5-(A-B)**1.5)/(3*B*np.sqrt(2*np.pi*model.beta))
    expected_TS = 4*radius**4*rho**2*angular
    expected_entry = 4*np.pi*radius**4*model.length**2/np.sqrt(2*np.pi*model.beta*model.reduced_mass)
    assert sum(integrals) == pytest.approx(expected_TS, rel=2e-14)
    assert entry == pytest.approx(expected_entry, rel=3e-15)
    assert sum(integrals)/entry == pytest.approx(expected_TS/expected_entry, rel=3e-14)


@pytest.mark.parametrize("alpha", [.999, 1.007])
def test_metric_matches_independent_cartesian_gradient_and_root_derivative(model, alpha):
    z = pilot.proposal_means(model, 3)[0]
    u = np.array([.2, .3, np.sqrt(.87)])
    q = pilot.from_chart(z, 3)
    rho = pilot.root_radius(q, 3, u, alpha, model); q[5] = q[3]+rho*u
    def surface(points):
        com = np.average(points[:5], weights=model.masses[:5], axis=0)
        return (1-alpha)*(model.length-np.linalg.norm(points[5]-com))+alpha*(model.c+np.linalg.norm(points[3]-points[1])-np.linalg.norm(points[5]-points[3]))
    finite = np.empty((6, 3)); step = 1e-5
    for atom in range(6):
        for axis in range(3):
            plus, minus = q.copy(), q.copy(); plus[atom, axis] += step; minus[atom, axis] -= step
            finite[atom, axis] = (surface(plus)-surface(minus))/(2*step)
    metric, derivative = pilot.inverse_mass_metric(q, 3, alpha, model)
    assert metric == pytest.approx(np.sum(finite*finite/model.masses[:, None]), rel=2e-9)
    assert derivative == pytest.approx(np.dot(finite[5], u), rel=2e-9)
    assert derivative <= -(alpha-abs(1-alpha))+1e-14
    assert abs(surface(q)) < 1e-12


def test_positive_root_absence_and_inactive_channel_are_zero_not_resampled(model):
    means = pilot.proposal_means(model, 3)
    q = pilot.from_chart(means[0], 3); q[3] *= .1
    assert pilot.root_radius(q, 3, np.array([0., 0., 1.]), 1.007, model) is None
    # Inside the finite methane radius domain, but another branch wins.
    q = model.reference-model.reference[1]
    q[0] = [4., 0., 0.]; q[2] = [4., 2., 0.]
    z = pilot.to_chart(q, 0)
    canonical = pilot.from_chart(z, 0)
    u = canonical[2]-canonical[0]; u /= np.linalg.norm(u)
    row = pilot.candidate(z, u, 0, .999, model, pilot.proposal_means(model, 0))
    assert row["zero_reason"] == "inactive_max_channel"
    assert row["minimum_pair_bohr"] >= model.minimum_pair
    assert row["log_importance_prefactor"] is None


@pytest.mark.parametrize("mutation,reason", [(lambda z: z.__setitem__(2, 1.1), "invalid_chart"),
                                            (lambda z: z.__setitem__(0, 5.), "methane_radius_domain")])
def test_outside_target_shape_draws_remain_explicit_zero(model, mutation, reason):
    means = pilot.proposal_means(model, 3); z = means[0].copy(); mutation(z)
    row = pilot.candidate(z, np.array([0., 0., 1.]), 3, .999, model, means)
    assert row["zero_reason"] == reason
    assert np.isfinite(row["shape_log_density"])
    assert row["log_importance_prefactor"] is None


def test_unconditional_means_keep_all_zero_draws_and_all_zero_is_unresolved():
    row = pilot.importance_statistics([-np.inf, np.log(2.)])
    assert row["draws"] == 2 and row["positive_weights"] == 1
    assert row["log_mean"] == pytest.approx(0., abs=1e-15)
    assert row["empirical_ESS"] == 1
    empty = pilot.importance_statistics([-np.inf, -np.inf])
    assert empty["log_mean"] is None and empty["empirical_ESS"] == 0


def test_channel_sum_and_pooled_ratio_are_not_mean_seed_ratios():
    rows, logs = [], []
    for seed, denominator in [(1, 1.), (2, 9.)]:
        rows.append({"target": "entry", "channel": None, "seed": seed}); logs.append(np.log(denominator))
        for target in ("alpha_0.999", "alpha_1.007"):
            for h in pilot.HYDROGENS:
                rows.append({"target": target, "channel": h, "seed": seed}); logs.append(np.log(2.))
    summary, ratio, difference = pilot.combined_integrals(rows, logs)
    assert np.exp(ratio["alpha_0.999"]) == pytest.approx(8/5)
    assert np.exp(ratio["alpha_0.999"]) != pytest.approx((8/1+8/9)/2)
    assert np.exp(summary["alpha_0.999"]["log_surface_integral_without_common_rotation"]) == pytest.approx(8)
    assert difference == 0


def test_full_fixed_ledger_is_deterministic_all_eight_and_paired_before_PES(model, ledger):
    assert ledger == pilot.build_ledger(model)
    assert len(ledger) == 768
    assert {row["seed"] for row in ledger} == set(pilot.SEEDS)
    for seed in pilot.SEEDS:
        subset = [row for row in ledger if row["seed"] == seed]
        assert len(subset) == 96
        assert sum(row["target"] == "entry" for row in subset) == 32
        for h in pilot.HYDROGENS:
            first = [row for row in subset if row["channel"] == h and row["alpha"] == .999]
            second = [row for row in subset if row["channel"] == h and row["alpha"] == 1.007]
            assert len(first) == len(second) == 8
            for a, b in zip(first, second):
                assert a["latent_id"] == b["latent_id"] and a["shape"] == b["shape"] and a["direction"] == b["direction"]
    assert any(row["zero_reason"] is not None for row in ledger)


def test_synthetic_execution_retains_zero_draws_and_never_outputs_converted_rate(model, ledger, tmp_path):
    payload = {"draws": ledger, "PES_geometries_evaluated_before_this_ledger": 0}
    pilot.atomic_json(tmp_path/"ledger.json", payload)
    digest = pilot.sha256(tmp_path/"ledger.json")
    calls = []
    def evaluator(q):
        assert pilot.sha256(tmp_path/"ledger.json") == digest
        assert pilot.read_json(tmp_path/"ledger.json")["PES_geometries_evaluated_before_this_ledger"] == 0
        calls.append(q.copy())
        return -.7
    report = pilot.evaluate_ledger(ledger, evaluator, tmp_path, beta=model.beta, energy_offset=model.energy_offset,
                                   registration_sha="fixture", ledger_sha=digest, clock=lambda: 0.)
    assert report["status"] == "complete_cost_pilot"
    assert len(calls) == sum(row["zero_reason"] is None for row in ledger) <= 768
    assert report["completed_target_draws"] == 768 and report["pending_target_draws"] == 0
    assert len(report["seed_summaries"]) == 8
    assert set(report) == set(pilot.OUTPUT_FIELDS)
    rendered = json.dumps(report)
    assert "cm3" not in rendered and "paper_pass" not in rendered and "k_qtst" not in rendered
    assert report["weight_diagnostics"]["entry"]["draws"] == 256
    with pytest.raises(ValueError, match="already exists"):
        pilot.evaluate_ledger(ledger, evaluator, tmp_path, beta=model.beta, energy_offset=model.energy_offset,
                              registration_sha="fixture", ledger_sha=digest, clock=lambda: 0.)


@pytest.mark.parametrize("behavior", ["raise", "nonfinite"])
def test_PES_failure_is_preserved_and_never_replaced_or_interpreted_as_zero(model, ledger, tmp_path, behavior):
    calls = []
    def evaluator(q):
        calls.append(q.copy())
        if behavior == "raise":
            raise RuntimeError("Synthetic evaluator failure, not a PES call")
        return float("nan")
    report = pilot.evaluate_ledger(ledger, evaluator, tmp_path, beta=model.beta, energy_offset=model.energy_offset,
                                   registration_sha="fixture", ledger_sha="fixture", clock=lambda: 0.)
    assert len(calls) == report["attempted_PES_geometries"] == 1
    assert report["status"] == "failed_PES_or_weight" and report["weight_diagnostics"] is None
    assert report["pending_target_draws"] > 0 and report["successful_PES_geometries"] == 0
    events = [json.loads(line) for line in (tmp_path/"calls.jsonl").read_text().splitlines()]
    assert [row["event"] for row in events] == ["attempt", "failure"]


def test_CPU_admission_stop_is_incomplete_with_no_partial_normalization(model, ledger, tmp_path):
    report = pilot.evaluate_ledger(ledger, lambda q: pytest.fail("CPU ceiling admitted evaluation"), tmp_path,
                                   beta=model.beta, energy_offset=model.energy_offset,
                                   registration_sha="fixture", ledger_sha="fixture", clock=lambda: 55.)
    assert report["status"] == "incomplete_CPU_admission_stop"
    assert report["attempted_PES_geometries"] == 0
    assert report["weight_diagnostics"] is None and report["seed_summaries"] is None


def test_wall_supervisor_retains_ledger_and_refuses_second_attempt(tmp_path, monkeypatch):
    # A bounded synthetic sleeping process only; no native/MD/PES invocation.
    regpath = tmp_path/"registration.json"
    pilot.atomic_json(regpath, {"fixture": True})
    guard = tmp_path/"guard.txt"; guard.write_text("unchanged synthetic source")
    reg = {"output_root": str(tmp_path), "source_sha256": {str(guard): pilot.sha256(guard)},
           "prepare_hard_CPU_seconds": 1}
    # This wall-only fixture must not inherit a long pytest process's CPU age.
    # Keep the real monotonic deadline and owned-child cleanup below unchanged.
    monkeypatch.setattr(pilot.time, "process_time", lambda: .1)
    monkeypatch.setattr(pilot, "load_registration", lambda path: (reg, {}, []))
    outcome = pilot.supervise(regpath, command=[sys.executable, "-c", "import time; time.sleep(5)"], wall_seconds=.1,
                              enforce_resources=False)
    assert outcome["status"] == "wall_timeout" and outcome["complete_cost_pilot"] is False
    assert (tmp_path/"RUN_ONCE.json").exists() and guard.read_text() == "unchanged synthetic source"
    with pytest.raises(FileExistsError):
        pilot.supervise(regpath, command=[sys.executable, "-c", "raise RuntimeError('must not run')"], wall_seconds=.1,
                        enforce_resources=False)


def test_CPU_hard_limit_is_installed_in_an_isolated_fixture_process():
    pytest.importorskip("resource", reason="declared real execution is POSIX")
    import subprocess
    program = ("import json,resource; from benchmarks.hch4.reference_pilot import install_CPU_limit; "
               "value=install_CPU_limit(57); print(json.dumps([value,list(resource.getrlimit(resource.RLIMIT_CPU))]))")
    result = subprocess.run([sys.executable, "-c", program], cwd=pilot.ROOT, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == [57, [57, 57]]


def test_private_worker_requires_supervisor_identity_and_cannot_repeat(tmp_path, monkeypatch):
    registration = tmp_path/"registration.json"
    pilot.atomic_json(registration, {"synthetic": True})
    with pytest.raises(FileNotFoundError):
        pilot.claim_worker(registration, tmp_path)
    token = "a"*32
    pilot.atomic_json(tmp_path/"CPU-budget.json", {"registration_sha256": pilot.sha256(registration), "supervisor_PID": 1234,
                                                  "supervisor_process_token": token})
    pilot.atomic_json(tmp_path/"RUN_ONCE.json", {"registration_sha256": pilot.sha256(registration), "supervisor_PID": 1234,
                                               "supervisor_process_token": token,
                                               "CPU_budget_sha256": pilot.sha256(tmp_path/"CPU-budget.json")})
    monkeypatch.setenv("PYRPMD_HCH4_REFERENCE_SUPERVISOR_TOKEN", token)
    monkeypatch.setattr(pilot.os, "getppid", lambda: 999)
    with pytest.raises(ValueError, match="one child"):
        pilot.claim_worker(registration, tmp_path)
    monkeypatch.setattr(pilot.os, "getppid", lambda: 1234)
    pilot.claim_worker(registration, tmp_path)
    with pytest.raises(FileExistsError):
        pilot.claim_worker(registration, tmp_path)


@pytest.mark.parametrize("change", ["source", "budget", "seeds", "output_fields", "adoption", "receipt"])
def test_changed_execution_contract_is_rejected_before_any_PES(tmp_path, change):
    receipt = tmp_path/"synthetic-validation.json"; pilot.atomic_json(receipt, {"fixture": "no engine"})
    source = str((pilot.ROOT/pilot.SELF).resolve())
    plan = {"schema": pilot.PLAN_SCHEMA, "proposal_sha256": pilot.PROPOSAL_SHA,
            "source_sha256": {source: pilot.sha256(source)}, "diagnostic_seeds": list(pilot.SEEDS),
            "maximum_PES_geometries": 768,
            "resource_ceiling": {"CPUs": 1, "soft_CPU_seconds": 55, "hard_CPU_seconds": 60, "wall_seconds": 120},
            "output_fields": list(pilot.OUTPUT_FIELDS), "parent_adoption": True,
            "synthetic_gate": {"path": str(receipt), "sha256": pilot.sha256(receipt)},
            "implementation_review": {"path": str(receipt), "sha256": pilot.sha256(receipt)}}
    if change == "source": plan["source_sha256"][source] = "0"*64
    elif change == "budget": plan["maximum_PES_geometries"] = 769
    elif change == "seeds": plan["diagnostic_seeds"] = list(pilot.SEEDS[:-1])
    elif change == "output_fields": plan["output_fields"].append("converted_rate")
    elif change == "adoption": plan["parent_adoption"] = False
    else: receipt.write_text("Changed synthetic receipt")
    path = tmp_path/"plan.json"; pilot.atomic_json(path, plan)
    with pytest.raises(ValueError): pilot.load_plan(path)


def test_aggregate_CPU_allocation_charges_both_prior_process_caps(model, ledger, tmp_path):
    budget = pilot.allocate_CPU_budget(3, 1.1)
    assert budget["supervisor_hard_CPU_seconds"] == 4
    assert budget["worker_hard_CPU_seconds"] == 53
    assert budget["sum_hard_CPU_caps_seconds"] == 60
    with pytest.raises(ValueError, match="exhausted"):
        pilot.allocate_CPU_budget(55, 4.1)
    report = pilot.evaluate_ledger(ledger, lambda q: pytest.fail("aggregate55s ceiling admitted evaluation"),
                                   tmp_path, beta=model.beta, energy_offset=model.energy_offset,
                                   registration_sha="fixture", ledger_sha="fixture", overhead_cpu_charge=7,
                                   clock=lambda: 48.)
    assert report["status"] == "incomplete_CPU_admission_stop"
    assert report["CPU_accounting"]["charged_total_before_report_seconds"] == 55


def test_namespace_PID_reuse_allows_distinct_interpreters_and_enforces_CPU_caps(tmp_path):
    pytest.importorskip("resource", reason="the actual bounded execution is POSIX")
    import subprocess
    # Limits are installed only in this isolated fixture process and its child.
    program = '''
import json, os, resource, sys
from pathlib import Path
from benchmarks.hch4 import reference_pilot as p
output=Path(sys.argv[1]); output.mkdir()
registration=output/'registration.json'; p.atomic_json(registration, {'synthetic':True})
guard=output/'guard'; guard.write_text('synthetic unchanged')
reg={'output_root':str(output),'source_sha256':{str(guard):p.sha256(guard)},
     'prepare_hard_CPU_seconds':2,'prepare_CPU_seconds_before_registration_write':.3,
     'prepare_process_PID':os.getpid(),'prepare_process_token':'e'*32}
p.load_registration=lambda path:(reg,{},[])
command=[sys.executable,'-c','import json,resource; import benchmarks.hch4.reference_pilot; print(json.dumps(list(resource.getrlimit(resource.RLIMIT_CPU))))']
result=p.supervise(registration,command=command,wall_seconds=10)
budget=p.read_json(output/'CPU-budget.json')
print(json.dumps({'budget':budget,'parent_limit':list(resource.getrlimit(resource.RLIMIT_CPU)),
                  'child_limit':json.loads((output/'worker.log').read_text()),
                  'worker_exit_code':result['worker_exit_code'],
                  'same_namespace_PID':reg['prepare_process_PID']==os.getpid(),
                  'supervisor_process_token':result['supervisor_process_token'],
                  'prepare_process_token':reg['prepare_process_token']}))
'''
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path/"isolated")], cwd=pilot.ROOT,
                            text=True, capture_output=True, check=True, timeout=20)
    record = json.loads(result.stdout)
    budget = record["budget"]
    assert budget["sum_hard_CPU_caps_seconds"] == 60
    assert record["parent_limit"] == [budget["supervisor_hard_CPU_seconds"]]*2
    assert record["child_limit"] == [budget["worker_hard_CPU_seconds"]]*2
    assert record["worker_exit_code"] == 0
    assert record["same_namespace_PID"] is True
    assert record["supervisor_process_token"] == budget["supervisor_process_token"]
    assert record["prepare_process_token"] != record["supervisor_process_token"]


def test_same_interpreter_cannot_claim_separate_preparation_CPU_budget(tmp_path, monkeypatch):
    pytest.importorskip("resource", reason="the actual bounded execution is POSIX")
    regpath = tmp_path/"registration.json"
    pilot.atomic_json(regpath, {"synthetic": True})
    reg = {"output_root": str(tmp_path), "source_sha256": {}, "prepare_hard_CPU_seconds": 2,
           "prepare_process_PID": -1, "prepare_process_token": pilot.PROCESS_TOKEN}
    monkeypatch.setattr(pilot, "load_registration", lambda path: (reg, {}, []))
    monkeypatch.setattr(pilot.time, "process_time", lambda: .3)
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *a, **k: pytest.fail("same interpreter admitted a child"))
    with pytest.raises(ValueError, match="separate dedicated interpreter"):
        pilot.supervise(regpath)
    assert (tmp_path/"RUN_ONCE.json").exists()
    assert not (tmp_path/"worker.log").exists()


def test_interpreter_identity_uses_no_numpy_scientific_random_draw():
    import subprocess
    program = '''
import json, numpy as np, uuid
np.random.seed(28317)
before=np.random.get_state()
from benchmarks.hch4 import reference_pilot as p
after=np.random.get_state()
print(json.dumps({'numpy_state_unchanged':before[0]==after[0] and
  np.array_equal(before[1],after[1]) and before[2:]==after[2:],
  'uuid_version':uuid.UUID(hex=p.PROCESS_TOKEN).version}))
'''
    result = subprocess.run([sys.executable, "-c", program], cwd=pilot.ROOT,
                            text=True, capture_output=True, check=True, timeout=10)
    assert json.loads(result.stdout) == {"numpy_state_unchanged": True, "uuid_version": 4}
