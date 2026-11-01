#!/usr/bin/env python3
"""Excluded, bounded P1 finite-domain flux-reference cost/weight pilot.

This tool performs no dynamics, quantum correction or physical-rate comparison.
The reviewed proposal is immutable. All 768 latent target draws and geometric
zeros are saved before the first PES evaluation. A registered output is run
at most once; failures, zero draws and incomplete attempts are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
# An invocation identity, independent of every scientific RNG. Separate managed
# exec namespaces may reuse the same PID even while these are distinct processes.
PROCESS_TOKEN = uuid.uuid4().hex

# This cap counts the worker's entire process lifetime, including scientific
# imports. The supervisor retains a larger inherited hard limit until spawn.
_early_CPU_cap = os.environ.get("PYRPMD_HCH4_REFERENCE_WORKER_CPU_CAP")
if _early_CPU_cap is not None:
    import resource
    _seconds = int(_early_CPU_cap)
    if not 1 <= _seconds <= 60:
        raise ValueError("invalid worker CPU allocation")
    resource.setrlimit(resource.RLIMIT_CPU, (_seconds, _seconds))

import numpy as np
from scipy.special import logsumexp

from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, KB_AU

SELF = "benchmarks/hch4/reference_pilot.py"
PROPOSAL = "docs/validation/literature-convergence-2026-11-01/review/hch4-reference-feasibility/pilot-design/proposal.json"
PROPOSAL_SHA = "85de4292058f9ac4fce9b8c5327d3aba0bea0204e02ede17fb629fa20ee23029"
PLAN_SCHEMA = "pyrpmd.hch4-reference-pilot-execution-plan.v1"
REG_SCHEMA = "pyrpmd.hch4-reference-pilot-registration.v2"
SEEDS = tuple(range(2026111201, 2026111209))
HYDROGENS = (0, 2, 3, 4)
ALPHAS = (.999, 1.007)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
OUTPUT_FIELDS = (
    "schema", "status", "scope", "registration_sha256", "ledger_sha256",
    "completed_target_draws", "geometric_zero_draws", "attempted_PES_geometries",
    "successful_PES_geometries", "pending_target_draws", "CPU_accounting",
    "call_CPU_seconds", "failure", "all_eight_diagnostic_seeds",
    "zero_reasons", "proposal_component_coverage", "source_guards_unchanged",
    "weight_diagnostics", "seed_summaries", "pooled_log_flux_ratios",
    "paired_log_surface_difference", "limitations",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Model:
    masses: np.ndarray
    beta: float
    c: float
    reference: np.ndarray
    length: float = 30.
    radius_low: float = 1.4
    radius_high: float = 4.5
    minimum_pair: float = .7
    energy_offset: float = -.7

    @property
    def methane_mass(self):
        return float(self.masses[:5].sum())

    @property
    def reduced_mass(self):
        return float(self.masses[5]*self.methane_mass/(self.masses[5]+self.methane_mass))


def fixed_model():
    proposal = read_json(ROOT / PROPOSAL)
    require(sha256(ROOT / PROPOSAL) == PROPOSAL_SHA, "reviewed proposal bytes changed")
    for path, expected in proposal["source_sha256"].items():
        require(sha256(ROOT/path) == expected, f"reviewed scientific input changed: {path}")
    values = proposal["physics"]["coordinate_constants"]
    reference = np.asarray(read_json(ROOT/"benchmarks/hch4/geometries.json")["official_input"]["coordinates"]) * ANGSTROM_TO_AU
    masses = np.asarray(proposal["physics"]["masses_amu"]) * AMU_TO_AU
    beta = 1/(300*KB_AU)
    c = float(np.linalg.norm(reference[5]-reference[3])-np.linalg.norm(reference[3]-reference[1]))
    require(np.array_equal(masses, np.asarray(values["masses_electron_mass"]))
            and beta == values["beta_hartree_inverse"] and c == values["reference_difference_bohr"],
            "runtime constants differ from the frozen proposal")
    return Model(masses, beta, c, reference)


def chart_labels(h):
    require(h in HYDROGENS, "unknown methane-H label")
    rest = sorted(set(HYDROGENS)-{h})
    return h, rest[0], rest[1], rest[2]


def to_chart(q, h):
    h, j, a, b = chart_labels(h)
    q = np.asarray(q, float)-q[1]
    ez = q[h]/np.linalg.norm(q[h])
    ex = q[j]-np.dot(q[j], ez)*ez
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    axes = np.array([ex, ey, ez])
    return np.r_[np.linalg.norm(q[h]), np.linalg.norm(q[j]),
                 np.dot(q[j], ez)/np.linalg.norm(q[j]), axes@q[a], axes@q[b]]


def from_chart(z, h):
    z = np.asarray(z, float)
    require(z.shape == (9,) and np.isfinite(z).all(), "invalid latent shape")
    r, s, v = z[:3]
    if r <= 0 or s <= 0 or not -1 < v < 1:
        return None
    h, j, a, b = chart_labels(h)
    q = np.zeros((6, 3))
    q[h] = [0, 0, r]
    q[j] = [s*np.sqrt(1-v*v), 0, s*v]
    q[a], q[b] = z[3:6], z[6:9]
    return q


def shape_jacobian(z):
    return float(z[0]**2*z[1]**2)


def proposal_means(model, h, *, entry=False):
    if entry:
        require(h == 3, "the entry sphere uses one fixed chart")
        radius = 1.0890*ANGSTROM_TO_AU
        q = np.zeros((6, 3)); q[3] = [0, 0, radius]
        q[0] = radius*np.array([np.sqrt(8/9), 0, -1/3])
        for atom, phi in ((2, 2*np.pi/3), (4, -2*np.pi/3)):
            q[atom] = radius*np.array([np.sqrt(8/9)*np.cos(phi), np.sqrt(8/9)*np.sin(phi), -1/3])
        sigma = np.array([.12, .12, .12, .20, .20, .20, .20, .20, .20])
    else:
        q = model.reference.copy()
        q[[h, 3]] = q[[3, h]]
        sigma = np.array([.15, .12, .12, .20, .20, .20, .20, .20, .20])
    plus = to_chart(q, h); minus = plus.copy(); minus[[4, 7]] *= -1
    return plus, minus, sigma


def box_bounds(model):
    low = np.array([model.radius_low, model.radius_low, -1., *([-model.radius_high]*6)])
    high = np.array([model.radius_high, model.radius_high, 1., *([model.radius_high]*6)])
    return low, high


def normal_logpdf(z, mean, sigma):
    return float(-.5*np.sum(((z-mean)/sigma)**2)-np.log(sigma).sum()-len(z)/2*np.log(2*np.pi))


def shape_logpdf(z, model, means):
    plus, minus, sigma = means
    low, high = box_bounds(model)
    uniform = -np.log(high-low).sum() if np.all((z >= low)&(z <= high)) else -np.inf
    return float(logsumexp([np.log(.2)+uniform, np.log(.4)+normal_logpdf(z, plus, sigma),
                           np.log(.4)+normal_logpdf(z, minus, sigma)]))


def draw_shape(rng, model, means):
    plus, minus, sigma = means
    choice = float(rng.random())
    if choice < .2:
        low, high = box_bounds(model)
        return rng.uniform(low, high), "uniform_box"
    if choice < .6:
        return plus+sigma*rng.normal(size=9), "positive_mirror_gaussian"
    return minus+sigma*rng.normal(size=9), "negative_mirror_gaussian"


def vmf_inverse(uniform):
    return float(1+np.log(uniform+(1-uniform)*np.exp(-16))/8)


def direction_logpdf(u, *, entry=False):
    if entry:
        return float(-np.log(4*np.pi))
    return float(logsumexp([np.log(.2)-np.log(4*np.pi),
                           np.log(.8*8/(4*np.pi*np.sinh(8)))+8*u[2]]))


def draw_direction(rng, *, entry=False):
    component = "uniform_sphere" if entry or rng.random() < .2 else "von_Mises_Fisher"
    uniform = float(rng.random())
    cosine = 2*uniform-1 if component == "uniform_sphere" else vmf_inverse(uniform)
    phi = float(rng.uniform(0, 2*np.pi))
    sine = np.sqrt(max(0., 1-cosine*cosine))
    return np.array([sine*np.cos(phi), sine*np.sin(phi), cosine]), component


def channel_values(q, model):
    return np.array([model.c+np.linalg.norm(q[h]-q[1])-np.linalg.norm(q[5]-q[h]) for h in HYDROGENS])


def root_radius(q, h, u, alpha, model):
    require(alpha > abs(1-alpha), "radial monotonicity is not guaranteed")
    com = np.average(q[:5], weights=model.masses[:5], axis=0)
    v = q[h]-com
    radius = float(np.linalg.norm(q[h]-q[1]))
    def surface(rho):
        return (1-alpha)*(model.length-np.linalg.norm(v+rho*u))+alpha*(model.c+radius-rho)
    first = surface(0.)
    if first <= 0:
        return None
    low, high = 0., float(first/(alpha-abs(1-alpha))+1e-12)
    require(surface(high) <= 1e-10, "failed monotone root bracket")
    for _ in range(72):
        middle = (low+high)/2
        if surface(middle) > 0:
            low = middle
        else:
            high = middle
    rho = (low+high)/2
    require(abs(surface(rho)) < 1e-10, "radial root residual exceeds fixed tolerance")
    return rho


def inverse_mass_metric(q, h, alpha, model):
    ub = q[h]-q[1]; ub /= np.linalg.norm(ub)
    u = q[5]-q[h]; u /= np.linalg.norm(u)
    uR = q[5]-np.average(q[:5], weights=model.masses[:5], axis=0)
    require(np.linalg.norm(uR) > 0, "singular coincident fragment COM")
    uR /= np.linalg.norm(uR)
    m, carbon, mu = model.masses[5], model.masses[1], model.reduced_mass
    metric = ((1-alpha)**2/mu+alpha**2*((3+2*np.dot(ub, u))/m+1/carbon)
              +2*alpha*(1-alpha)*np.dot(uR, u)/mu)
    derivative = -alpha-(1-alpha)*np.dot(u, uR)
    require(metric > 0 and derivative < 0, "invalid surface metric/derivative")
    return float(metric), float(derivative)


def candidate(z, u, h, alpha, model, means):
    entry = alpha is None
    logqz = shape_logpdf(z, model, means)
    logqu = direction_logpdf(u, entry=entry)
    row = {"shape": z.tolist(), "direction": u.tolist(), "channel": h if not entry else None,
           "alpha": alpha, "target": "entry" if entry else f"alpha_{alpha:.3f}",
           "shape_log_density": logqz, "direction_log_density": logqu,
           "coordinates_bohr": None, "zero_reason": None,
           "log_geometric_measure": None, "log_importance_prefactor": None}
    q = from_chart(z, h)
    if q is None:
        row["zero_reason"] = "invalid_chart"; return row
    radii = np.linalg.norm(q[list(HYDROGENS)]-q[1], axis=1)
    if not np.all((radii >= model.radius_low)&(radii <= model.radius_high)):
        row["zero_reason"] = "methane_radius_domain"; return row
    if entry:
        q[5] = np.average(q[:5], weights=model.masses[:5], axis=0)+model.length*u
        metric = 1/model.reduced_mass
        log_measure = np.log(shape_jacobian(z))+2*np.log(model.length)+.5*np.log(metric/(2*np.pi*model.beta))
    else:
        rho = root_radius(q, h, u, alpha, model)
        if rho is None:
            row["zero_reason"] = "no_positive_root"; return row
        q[5] = q[h]+rho*u
        metric, derivative = inverse_mass_metric(q, h, alpha, model)
        row.update(rho_bohr=rho, radial_derivative=derivative, inverse_mass_metric=metric)
        log_measure = (np.log(shape_jacobian(z))+2*np.log(rho)-np.log(-derivative)
                       +.5*np.log(metric/(2*np.pi*model.beta)))
    row["coordinates_bohr"] = q.tolist()
    pair = np.linalg.norm(q[:, None]-q[None, :], axis=-1)[np.triu_indices(6, 1)]
    row["minimum_pair_bohr"] = float(pair.min())
    values = channel_values(q, model)
    active = int(HYDROGENS[int(np.argmax(values))])
    distance = float(np.linalg.norm(q[5]-np.average(q[:5], weights=model.masses[:5], axis=0)))
    row.update(active_channel=active, channel_values_bohr=values.tolist(), fragment_distance_bohr=distance,
               xi_denominator_bohr=model.length-distance-float(values.max()))
    if pair.min() < model.minimum_pair:
        row["zero_reason"] = "minimum_pair_domain"
    elif not entry and active != h:
        row["zero_reason"] = "inactive_max_channel"
    elif row["xi_denominator_bohr"] <= 0:
        row["zero_reason"] = "disconnected_xi_component"
    else:
        row["log_geometric_measure"] = float(log_measure)
        row["log_importance_prefactor"] = float(log_measure-logqz-logqu)
    return row


def build_ledger(model):
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for h in HYDROGENS:
            means = proposal_means(model, h)
            for draw in range(8):
                z, component = draw_shape(rng, model, means)
                u, direction_component = draw_direction(rng)
                for alpha in ALPHAS:
                    row = candidate(z, u, h, alpha, model, means)
                    row.update(index=len(rows), seed=seed, latent_id=f"{seed}:channel{h}:draw{draw}",
                               shape_component=component, direction_component=direction_component)
                    rows.append(row)
        means = proposal_means(model, 3, entry=True)
        for draw in range(32):
            z, component = draw_shape(rng, model, means)
            u, direction_component = draw_direction(rng, entry=True)
            row = candidate(z, u, 3, None, model, means)
            row.update(index=len(rows), seed=seed, latent_id=f"{seed}:entry:draw{draw}",
                       shape_component=component, direction_component=direction_component)
            rows.append(row)
    require(len(rows) == 768 and [r["index"] for r in rows] == list(range(768)), "ledger size/order mismatch")
    return rows


def importance_statistics(log_weights):
    logs = np.asarray(log_weights, float)
    require(logs.ndim == 1 and len(logs) > 0 and not np.isnan(logs).any() and not np.isposinf(logs).any(), "invalid importance weights")
    positive = np.isfinite(logs)
    if not positive.any():
        return {"draws": len(logs), "positive_weights": 0, "log_mean": None, "empirical_ESS": 0., "maximum_normalized_weight": None}
    total = float(logsumexp(logs))
    normalized = np.exp(logs[positive]-total)
    return {"draws": len(logs), "positive_weights": int(positive.sum()),
            "log_mean": float(total-np.log(len(logs))),
            "empirical_ESS": float(1/np.sum(normalized**2)),
            "maximum_normalized_weight": float(normalized.max())}


def combined_integrals(rows, logweights):
    summaries, logs = {}, {}
    for target in ("entry", "alpha_0.999", "alpha_1.007"):
        chosen = [i for i, row in enumerate(rows) if row["target"] == target]
        values = importance_statistics([logweights[i] for i in chosen])
        if target == "entry":
            logs[target] = values["log_mean"]
        else:
            channels = {}
            for h in HYDROGENS:
                channels[str(h)] = importance_statistics([logweights[i] for i in chosen if rows[i]["channel"] == h])
            means = [v["log_mean"] for v in channels.values() if v["log_mean"] is not None]
            logs[target] = float(logsumexp(means)) if means else None
            values["channel_means"] = channels
        values["log_surface_integral_without_common_rotation"] = logs[target]
        summaries[target] = values
    ratios = {name: float(value-logs["entry"]) if value is not None and logs["entry"] is not None else None
              for name, value in logs.items() if name != "entry"}
    difference = (float(logs["alpha_1.007"]-logs["alpha_0.999"])
                  if logs["alpha_1.007"] is not None and logs["alpha_0.999"] is not None else None)
    return summaries, ratios, difference


def environment_identity():
    import scipy
    return {"python_executable": str(Path(sys.executable).resolve()), "python_version": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform()}


def check_hashes(mapping):
    for name, expected in mapping.items():
        require(sha256(name) == expected, f"registered source/model changed: {name}")


def load_plan(path):
    path = Path(path).resolve(); plan = read_json(path)
    require(plan.get("schema") == PLAN_SCHEMA and plan["proposal_sha256"] == PROPOSAL_SHA,
            "unreviewed plan/proposal identity")
    require(plan["source_sha256"].get(str((ROOT/SELF).resolve())) == sha256(ROOT/SELF), "execution source is not frozen in plan")
    require(plan["diagnostic_seeds"] == list(SEEDS) and plan["maximum_PES_geometries"] == 768
            and plan["resource_ceiling"] == {"CPUs": 1, "soft_CPU_seconds": 55, "hard_CPU_seconds": 60, "wall_seconds": 120}
            and plan["output_fields"] == list(OUTPUT_FIELDS) and plan["parent_adoption"] is True,
            "execution settings differ from adopted bounded design")
    check_hashes(plan["source_sha256"])
    for name in ("synthetic_gate", "implementation_review"):
        receipt = plan[name]
        require(sha256(receipt["path"]) == receipt["sha256"], f"{name} receipt changed")
    require(sha256(ROOT/PROPOSAL) == PROPOSAL_SHA, "proposal bytes changed")
    return plan


def prepare(plan_path, output):
    plan = load_plan(plan_path)
    model = fixed_model()
    output = Path(output).resolve()
    require(not output.exists() or not any(output.iterdir()), "pilot output must be new/empty")
    rows = build_ledger(model)
    check_hashes(plan["source_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"ledger.json", {"schema": "pyrpmd.hch4-reference-latent-ledger.v1", "draws": rows,
                                      "PES_geometries_evaluated_before_this_ledger": 0})
    # Charge the whole dedicated preparation process, not just this function.
    # The extra integer second bounds registration writing and interpreter exit.
    measured_cpu = time.process_time()
    preparation_cap = math.ceil(measured_cpu)+1
    require(preparation_cap < 55, "preparation exhausted the bounded experiment budget")
    install_CPU_limit(preparation_cap)
    registration = {"schema": REG_SCHEMA, "created_utc": now(), "output_root": str(output),
                    "plan_path": str(Path(plan_path).resolve()), "plan_sha256": sha256(plan_path),
                    "ledger_sha256": sha256(output/"ledger.json"), "target_draws": len(rows),
                    "eligible_PES_geometries": sum(row["zero_reason"] is None for row in rows),
                    "source_sha256": plan["source_sha256"], "environment": environment_identity(),
                    "prepare_CPU_seconds_before_registration_write": measured_cpu,
                    "prepare_process_PID": os.getpid(), "prepare_process_token": PROCESS_TOKEN,
                    "prepare_hard_CPU_seconds": preparation_cap,
                    "replay_policy": "one execution only; no automatic retries or replacement draws"}
    atomic_json(output/"registration.json", registration)
    return output/"registration.json"


def load_registration(path):
    path = Path(path).resolve(); reg = read_json(path)
    output = Path(reg["output_root"]).resolve()
    require(reg.get("schema") == REG_SCHEMA and path.parent == output, "invalid registration path/schema")
    require(type(reg["prepare_hard_CPU_seconds"]) is int and 1 <= reg["prepare_hard_CPU_seconds"] < 55
            and 0 <= reg["prepare_CPU_seconds_before_registration_write"] < reg["prepare_hard_CPU_seconds"],
            "invalid preparation CPU accounting")
    token = reg["prepare_process_token"]
    require(isinstance(token, str) and len(token) == 32 and uuid.UUID(hex=token).hex == token,
            "invalid preparation interpreter identity")
    plan = load_plan(reg["plan_path"])
    require(sha256(reg["plan_path"]) == reg["plan_sha256"] and reg["source_sha256"] == plan["source_sha256"]
            and reg["environment"] == environment_identity(), "registered plan/source/runtime changed")
    require(sha256(output/"ledger.json") == reg["ledger_sha256"], "pre-PES ledger changed")
    rows = read_json(output/"ledger.json")["draws"]
    require(len(rows) == reg["target_draws"] == 768 and [r["index"] for r in rows] == list(range(768)), "incomplete ledger")
    require(sorted({r["seed"] for r in rows}) == list(SEEDS), "changed seed cohort")
    # Recreate the deterministic geometry ledger before any external evaluation.
    require(rows == build_ledger(fixed_model()), "ledger is not the fixed proposal's exact draws")
    return reg, plan, rows


def evaluate_ledger(rows, evaluator, output, *, beta, energy_offset, registration_sha,
                    ledger_sha, overhead_cpu_charge=0., clock=time.process_time, guards=None):
    """Generic bounded loop; unit tests use synthetic evaluators, never CBE."""
    output = Path(output)
    events = output/"calls.jsonl"
    require(not events.exists(), "call ledger already exists; do not rerun this pilot")
    status, failure = "complete_cost_pilot", None
    logweights = [None]*len(rows)
    attempted, successes, call_times = 0, 0, []
    for i, row in enumerate(rows):
        if row["zero_reason"] is not None:
            logweights[i] = -np.inf
            continue
        if overhead_cpu_charge+clock() >= 55:
            status = "incomplete_CPU_admission_stop"; break
        attempted += 1
        require(attempted <= 768, "PES geometry ceiling exceeded")
        begin = clock()
        append_event(events, {"event": "attempt", "index": row["index"], "seed": row["seed"],
                              "attempt_number": attempted, "CPU_before_call": begin, "utc": now()})
        try:
            energy = float(evaluator(np.asarray(row["coordinates_bohr"], float)))
            elapsed = clock()-begin
            require(np.isfinite(energy), "PES returned a nonfinite energy")
            value = float(row["log_importance_prefactor"]-beta*(energy-energy_offset))
            require(np.isfinite(value), "nonfinite logarithmic importance weight")
        except BaseException as error:
            failure = {"index": row["index"], "error_type": type(error).__name__, "message": str(error)}
            append_event(events, {"event": "failure", **failure, "CPU_after_call": clock(), "utc": now()})
            status = "failed_PES_or_weight"; break
        logweights[i] = value
        successes += 1; call_times.append(elapsed)
        append_event(events, {"event": "success", "index": row["index"], "energy_hartree": energy,
                              "log_weight": value, "call_CPU_seconds": elapsed, "utc": now()})
    unchanged = True
    try:
        if guards:
            check_hashes(guards)
    except Exception as error:
        unchanged = False; status = "failed_source_guard"
        failure = {"error_type": type(error).__name__, "message": str(error)}
    complete = all(value is not None for value in logweights) and status == "complete_cost_pilot"
    report = {"schema": "pyrpmd.hch4-reference-cost-pilot.v1", "status": status,
              "scope": "excluded finite-domain cost/weight pilot; no physical rate, quantum correction or literature comparison",
              "registration_sha256": registration_sha, "ledger_sha256": ledger_sha,
              "completed_target_draws": sum(x is not None for x in logweights),
              "geometric_zero_draws": sum(row["zero_reason"] is not None for row in rows),
              "attempted_PES_geometries": attempted, "successful_PES_geometries": successes,
              "pending_target_draws": sum(x is None for x in logweights),
              "CPU_accounting": {"charged_preparation_and_supervisor_caps_seconds": overhead_cpu_charge,
                                 "measured_worker_process_CPU_seconds_before_report": clock(),
                                 "charged_total_before_report_seconds": overhead_cpu_charge+clock(),
                                 "all_log_integrals": "include common beta*Eoffset and omit common8*pi^2; both cancel in ratios"},
              "call_CPU_seconds": call_times,
              "failure": failure, "all_eight_diagnostic_seeds": list(SEEDS),
              "zero_reasons": dict(Counter(row["zero_reason"] for row in rows if row["zero_reason"])),
              "proposal_component_coverage": dict(Counter(f'{row["target"]}:{row["shape_component"]}:{row["direction_component"]}' for row in rows)),
              "source_guards_unchanged": unchanged,
              "weight_diagnostics": None, "seed_summaries": None, "pooled_log_flux_ratios": None,
              "paired_log_surface_difference": None,
              "limitations": ["Finite-domain/PES reliability and missing tails are unvalidated.",
                              "Empirical ESS is not a bound on unseen importance tails.",
                              "A completed cost pilot is not a scientific accuracy or equivalence pass.",
                              "Eight independent diagnostic seed vectors; channels/paired surfaces are not extra replications."]}
    if complete:
        summary, ratios, difference = combined_integrals(rows, logweights)
        seeds = []
        for seed in SEEDS:
            chosen = [i for i, row in enumerate(rows) if row["seed"] == seed]
            seed_summary, seed_ratios, seed_difference = combined_integrals([rows[i] for i in chosen], [logweights[i] for i in chosen])
            seeds.append({"seed": seed, "weights": seed_summary, "log_flux_ratios": seed_ratios,
                          "paired_log_surface_difference": seed_difference})
        report.update(weight_diagnostics=summary, seed_summaries=seeds,
                      pooled_log_flux_ratios=ratios, paired_log_surface_difference=difference)
    require(set(report) == set(OUTPUT_FIELDS), "undeclared pilot output field")
    atomic_json(output/"report.json", report)
    return report


def install_CPU_limit(seconds):
    import resource
    require(type(seconds) is int and 1 <= seconds <= 60, "CPU allocation must be an integer1..60")
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
    return seconds


def allocate_CPU_budget(preparation_cap, supervisor_cpu):
    require(type(preparation_cap) is int and preparation_cap >= 1 and np.isfinite(supervisor_cpu)
            and supervisor_cpu >= 0, "invalid prior CPU accounting")
    supervisor_cap = math.ceil(supervisor_cpu)+2
    worker_cap = 60-preparation_cap-supervisor_cap
    require(worker_cap >= 1, "preflight exhausted the total CPU budget")
    return {"preparation_hard_CPU_seconds": preparation_cap,
            "supervisor_hard_CPU_seconds": supervisor_cap,
            "worker_hard_CPU_seconds": worker_cap,
            "sum_hard_CPU_caps_seconds": preparation_cap+supervisor_cap+worker_cap,
            "soft_total_CPU_admission_seconds": 55}


def claim_worker(registration, output):
    """The private worker also refuses direct bypass or a second attempt."""
    output = Path(output)
    supervisor = read_json(output/"RUN_ONCE.json")
    require(supervisor["registration_sha256"] == sha256(registration)
            and supervisor["supervisor_PID"] == os.getppid()
            and supervisor["supervisor_process_token"] == os.environ.get("PYRPMD_HCH4_REFERENCE_SUPERVISOR_TOKEN")
            and supervisor["supervisor_process_token"] != PROCESS_TOKEN,
            "worker must be the one child of the registered supervisor")
    require(supervisor["CPU_budget_sha256"] == sha256(output/"CPU-budget.json"), "CPU budget changed")
    budget = read_json(output/"CPU-budget.json")
    require(budget["registration_sha256"] == sha256(registration)
            and budget["supervisor_PID"] == os.getppid()
            and budget["supervisor_process_token"] == supervisor["supervisor_process_token"],
            "CPU budget has another supervisor/registration")
    with (output/"WORKER_ONCE.json").open("x") as stream:
        json.dump({"created_utc": now(), "registration_sha256": sha256(registration),
                   "worker_PID": os.getpid(), "worker_process_token": PROCESS_TOKEN,
                   "supervisor_PID": os.getppid(),
                   "supervisor_process_token": supervisor["supervisor_process_token"]}, stream)
        stream.flush(); os.fsync(stream.fileno())
    return budget


def worker(registration):
    reg, plan, rows = load_registration(registration)
    output = Path(reg["output_root"])
    budget = claim_worker(registration, output)
    require(all(os.environ.get(name) == "1" for name in THREAD_VARIABLES), "native/BLAS threads must be one")
    require(len(os.sched_getaffinity(0)) == 1, "pilot must use exactly one CPU")
    require(budget["hard_caps_requested"] is True and budget["sum_hard_CPU_caps_seconds"] == 60
            and budget["preparation_hard_CPU_seconds"] == reg["prepare_hard_CPU_seconds"]
            and int(os.environ["PYRPMD_HCH4_REFERENCE_WORKER_CPU_CAP"]) == budget["worker_hard_CPU_seconds"],
            "worker did not receive the frozen aggregate resource budget")
    hard_seconds = install_CPU_limit(budget["worker_hard_CPU_seconds"])
    imported = {}
    for name in ("pyrpmd.constants", "pyrpmd.potentials.base", "pyrpmd.potentials.cabi", "benchmarks.hch4.pes"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(str(path) in plan["source_sha256"] and sha256(path) == plan["source_sha256"][str(path)],
                f"actual imported module differs from frozen source: {name}")
        imported[name] = {"path": str(path), "sha256": sha256(path)}
    atomic_json(output/"runtime.json", {"environment": environment_identity(), "PID": os.getpid(),
                "worker_process_token": PROCESS_TOKEN, "supervisor_process_token": budget["supervisor_process_token"],
                "affinity": sorted(os.sched_getaffinity(0)), "thread_environment": {k: os.environ[k] for k in THREAD_VARIABLES},
                "hard_CPU_seconds_in_this_process": hard_seconds, "actual_imports": imported, "started_utc": now()})
    from benchmarks.hch4.pes import CBE2008Potential
    potential = CBE2008Potential(plan["library"])
    require(potential.metadata["library_sha256"] == plan["library_sha256"], "loaded model identity differs")
    def energy(q):
        result = potential.evaluate(q[None])
        require(result.status[0] == 0, f"PES error status {result.status[0]}: {result.messages[0]}")
        return float(result.energy[0])
    model = fixed_model()
    report = evaluate_ledger(rows, energy, output, beta=model.beta, energy_offset=model.energy_offset,
                             registration_sha=sha256(registration), ledger_sha=reg["ledger_sha256"],
                             overhead_cpu_charge=budget["preparation_hard_CPU_seconds"]+budget["supervisor_hard_CPU_seconds"],
                             guards=reg["source_sha256"])
    return 0 if report["status"] == "complete_cost_pilot" else 2


def supervise(registration, *, command=None, wall_seconds=120, enforce_resources=True):
    started = time.perf_counter()
    reg, plan, rows = load_registration(registration)
    output = Path(reg["output_root"])
    # Exclusive persistent marker prevents a second attempt, including after a crash.
    with (output/"RUN_ONCE.json").open("x") as stream:
        json.dump({"created_utc": now(), "registration_sha256": sha256(registration),
                   "supervisor_PID": os.getpid(), "supervisor_process_token": PROCESS_TOKEN}, stream)
        stream.flush(); os.fsync(stream.fileno())
    require(0 < wall_seconds <= 120, "wall ceiling cannot exceed the reviewed120s")
    budget = allocate_CPU_budget(reg["prepare_hard_CPU_seconds"], time.process_time())
    budget.update(registration_sha256=sha256(registration), supervisor_PID=os.getpid(),
                  supervisor_process_token=PROCESS_TOKEN,
                  measured_supervisor_CPU_seconds_at_allocation=time.process_time(),
                  hard_caps_requested=enforce_resources, created_utc=now())
    atomic_json(output/"CPU-budget.json", budget)
    marker = read_json(output/"RUN_ONCE.json")
    marker["CPU_budget_sha256"] = sha256(output/"CPU-budget.json")
    atomic_json(output/"RUN_ONCE.json", marker)
    command = command or [sys.executable, str(Path(__file__).resolve()), "_worker", "--registration", str(Path(registration).resolve())]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **{name: "1" for name in THREAD_VARIABLES})
    environment["PYRPMD_HCH4_REFERENCE_SUPERVISOR_TOKEN"] = PROCESS_TOKEN
    if enforce_resources:
        import resource
        require(reg["prepare_process_token"] != PROCESS_TOKEN,
                "preparation and supervisor must be separate dedicated interpreter invocations")
        environment["PYRPMD_HCH4_REFERENCE_WORKER_CPU_CAP"] = str(budget["worker_hard_CPU_seconds"])
        # Bound supervisor CPU during spawn, retaining the inherited hard limit
        # until the worker can install its independently allocated cap.
        _, inherited_hard = resource.getrlimit(resource.RLIMIT_CPU)
        require(inherited_hard == resource.RLIM_INFINITY or inherited_hard >= budget["worker_hard_CPU_seconds"],
                "inherited hard CPU limit cannot accommodate the worker allocation")
        signal.signal(signal.SIGXCPU, signal.SIG_DFL)
        resource.setrlimit(resource.RLIMIT_CPU, (budget["supervisor_hard_CPU_seconds"], inherited_hard))
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        previous_child_cpu = before.ru_utime+before.ru_stime
    timed_out = False
    with (output/"worker.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=environment)
        try:
            if enforce_resources:
                install_CPU_limit(budget["supervisor_hard_CPU_seconds"])
            code = child.wait(timeout=max(.001, wall_seconds-(time.perf_counter()-started)))
        except subprocess.TimeoutExpired:
            timed_out = True; child.kill(); code = child.wait()
        except BaseException:
            child.kill(); child.wait()
            atomic_json(output/"supervisor.json", {"status": "interrupted", "child_PID": child.pid,
                        "wall_seconds": time.perf_counter()-started, "output_retained": True})
            raise
    check_hashes(reg["source_sha256"])
    report = read_json(output/"report.json") if (output/"report.json").exists() else None
    outcome = {"status": "wall_timeout" if timed_out else "worker_exited", "worker_exit_code": code,
               "supervisor_process_token": PROCESS_TOKEN,
               "child_PID": child.pid, "wall_seconds": time.perf_counter()-started,
               "report_status": report["status"] if report else None,
               "complete_cost_pilot": bool(code == 0 and report and report["status"] == "complete_cost_pilot"),
               "no_retry_or_budget_extension": True, "registration_sha256": sha256(registration)}
    if enforce_resources:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        outcome["CPU_accounting"] = {"three_enforced_caps": budget,
                "measured_worker_entire_process_CPU_seconds": usage.ru_utime+usage.ru_stime-previous_child_cpu,
                "measured_supervisor_CPU_seconds_before_receipt": time.process_time(),
                "prepare_CPU_seconds_before_registration_write": reg["prepare_CPU_seconds_before_registration_write"],
                "unmeasured_finalization_bounded_by_allocated_reserves": True}
    atomic_json(output/"supervisor.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("prepare", help="zero-PES registration and full deterministic latent ledger")
    create.add_argument("--plan", type=Path, required=True); create.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="one bounded evaluated pilot; no automatic retry")
    run.add_argument("--registration", type=Path, required=True)
    execute = sub.add_parser("_worker", help=argparse.SUPPRESS)
    execute.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.plan, args.output))
    elif args.action == "run":
        result = supervise(args.registration)
        print(json.dumps(result, indent=2)); raise SystemExit(0 if result["complete_cost_pilot"] else 2)
    else:
        raise SystemExit(worker(args.registration))


if __name__ == "__main__":
    main()
