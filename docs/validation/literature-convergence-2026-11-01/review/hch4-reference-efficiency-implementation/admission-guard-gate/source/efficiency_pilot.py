#!/usr/bin/env python3
"""Excluded second-proposal HCH4 pilot; no execution without separate adoption.

The physical PES and original coarea target are unchanged. Every real geometry,
including proposal calibration, is charged to the fixed 4000 maximum. Scientific
imports occur only after the CLI role's resource and ownership checks.
"""
from __future__ import annotations

import time
ENTRY_MONOTONIC = time.monotonic()

import os
import sys
import signal

# These stdlib-only admission checks precede NumPy/SciPy/PES imports. The
# supervisor retains its inherited hard CPU limit until the child has spawned;
# its 3s soft SIGXCPU default is already active during preflight/spawn.
ROLE = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else None
RESOURCE_ROLE = ROLE in {"prepare", "run", "_worker"}
if RESOURCE_ROLE:
    import resource
    if sys.platform != "linux":
        raise SystemExit("the adopted process/resource contract requires Linux")
    if ROLE == "_worker":
        if os.environ.get("PYRPMD_HCH4_EFFICIENCY_WORKER_CAP") != "55":
            raise SystemExit("a worker requires its supervisor's exact resource admission")
        _deadline = float(os.environ.get("PYRPMD_HCH4_EFFICIENCY_DEADLINE", "0"))
        _remaining = _deadline-time.monotonic()
        if not 0 < _remaining <= 120 or time.process_time()+5 >= 55:
            raise SystemExit("worker CPU/wall admission expired before scientific imports")
        resource.setrlimit(resource.RLIMIT_CPU, (55, 55))
    elif ROLE == "prepare":
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        _remaining = 120.
    else:
        _, _hard = resource.getrlimit(resource.RLIMIT_CPU)
        if _hard != resource.RLIM_INFINITY and _hard < 55:
            raise SystemExit("inherited CPU limit cannot admit the independently capped worker")
        signal.signal(signal.SIGXCPU, signal.SIG_DFL)
        resource.setrlimit(resource.RLIMIT_CPU, (3, _hard))
        _remaining = 120.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, _remaining)

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))
sys.dont_write_bytecode = True
SELF = ROOT/"benchmarks/hch4/efficiency_pilot.py"
TESTS = ROOT/"tests/test_hch4_efficiency_pilot.py"
DESIGN = ROOT/"docs/validation/literature-convergence-2026-11-01/review/hch4-reference-efficiency-design/proposal.json"
DESIGN_SHA = "8c9c576f37714cead7beb6a788a1262f88b7a1f34914029f041643774b78d962"
METHOD_REVIEW = DESIGN.parent/"independent-method-review.json"
METHOD_REVIEW_SHA = "694590d63fb8d153bef43286e90bcd218271d0977e9372f6849d8d70f3247ff7"
LEGACY = ROOT/"benchmarks/hch4/reference_pilot.py"
LEGACY_SHA = "a164ff9ab148e24b5175d7087224efd95a95435665b90f4a353b2ddf0778a65a"
SEEDS = tuple(range(2026112201, 2026112209))
CHANNELS = (0, 2, 3, 4)
TARGETS = ("alpha_0.999", "alpha_1.007", "entry")
DOMAINS = {"inner": (1.6, 4., .9), "original": (1.4, 4.5, .7), "outer": (1.2, 5., .6)}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PROCESS_TOKEN = uuid.uuid4().hex
PLAN_SCHEMA = "pyrpmd.hch4-efficiency-execution-plan.v1"
REG_SCHEMA = "pyrpmd.hch4-efficiency-registration.v1"
OUTPUT_FIELDS = ("schema", "status", "scope", "registration_sha256", "source_guards_unchanged",
                 "stage", "ledger_sha256", "proposal_sha256", "attempted_PES_geometries",
                 "successful_PES_geometries", "calibration_complete", "main_draws_complete",
                 "control_draws_complete", "failure", "resource_accounting", "analysis", "limitations")
_NUMPY = None
_LOGSUMEXP = None
_GAMMALN = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_new_json(path, value):
    """Exclusive immutable document; no automatic retry or overwrite."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def event(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        stream.flush(); os.fsync(stream.fileno())


def science():
    global _NUMPY, _LOGSUMEXP, _GAMMALN
    if _NUMPY is None:
        if ROLE == "_worker":
            require(time.process_time()+5 < 55
                    and time.monotonic() < float(os.environ["PYRPMD_HCH4_EFFICIENCY_DEADLINE"])-2,
                    "CPU/wall admission expired before scientific imports")
        import numpy
        from scipy.special import gammaln, logsumexp
        _NUMPY, _LOGSUMEXP, _GAMMALN = numpy, logsumexp, gammaln
    return _NUMPY


def legacy():
    require(sha(LEGACY) == LEGACY_SHA, "frozen original coarea helper changed")
    module = importlib.import_module("benchmarks.hch4.reference_pilot")
    require(Path(module.__file__).resolve() == LEGACY.resolve(), "coarea helper import route differs")
    return module


def sphere(t):
    np = science(); t = np.asarray(t, dtype=float)
    require(t.shape == (2,) and np.isfinite(t).all(), "invalid stereographic coordinate")
    square = float(t@t); denominator = 1+square
    return np.r_[2*t, 1-square]/denominator, 4/(denominator*denominator)


def inverse_sphere(u):
    np = science(); u = np.asarray(u, dtype=float)
    require(u.shape == (3,) and np.isfinite(u).all() and abs(float(u@u)-1) < 1e-12,
            "invalid unit direction")
    if u[2] == -1:
        return None, 0.
    t = u[:2]/(1+u[2])
    return t, float((1+u[2])**2)


def reflection(dimension):
    np = science(); signs = np.ones(dimension)
    require(dimension in (9, 11), "unsupported chart dimension")
    signs[[4, 7]] = -1
    if dimension == 11:
        signs[10] = -1
    return signs


def log_density(x, mean, covariance, *, student=False):
    np = science(); x, mean, covariance = map(np.asarray, (x, mean, covariance))
    require(x.shape == mean.shape and mean.ndim == 1 and covariance.shape == (len(mean),)*2
            and np.isfinite(x).all() and np.isfinite(mean).all() and np.isfinite(covariance).all(),
            "invalid elliptical density parameters")
    scale = covariance*(12/5) if student else covariance
    chol = np.linalg.cholesky(scale)
    difference = np.linalg.solve(chol, x-mean)
    logdet = 2*float(np.log(np.diag(chol)).sum()); radius = float(difference@difference); d = len(mean)
    if student:
        return float(_GAMMALN((5+d)/2)-_GAMMALN(5/2)-.5*(d*math.log(5*math.pi)+logdet)
                     -(5+d)/2*math.log1p(radius/5))
    return -.5*(d*math.log(2*math.pi)+logdet+radius)


def components(models, *, entry=False):
    np = science(); result = []
    names = ("entry",) if entry else TARGETS[:2]
    for kind in ("normal", "student5"):
        for name in names:
            model = models[name]
            mean, covariance = np.asarray(model["mean_x"]), np.asarray(model["covariance_x"])
            for mirror in ("positive", "negative"):
                signs = np.ones(len(mean)) if mirror == "positive" else reflection(len(mean))
                result.append({"name": f"{kind}:{name}:{mirror}", "kind": kind, "mirror": mirror,
                               "mean": signs*mean, "covariance": covariance*signs[:, None]*signs[None, :],
                               "count": (48 if entry else 6) if kind == "normal" else (8 if entry else 1)})
    return result


def mixture_log_density(z, u, models, *, entry=False):
    np = science(); z = np.asarray(z, float)
    low = np.r_[1.2, 1.2, -1., [-5.]*6]; high = np.r_[5., 5., 1., [5.]*6]
    terms = []
    if np.all((z >= low)&(z <= high)):
        terms.append(-math.log(8)-float(np.log(high-low).sum())-math.log(4*math.pi))
    if entry:
        x, jac = z, 4*math.pi
    else:
        t, jac = inverse_sphere(u)
        if t is None:
            return float(_LOGSUMEXP(terms)) if terms else -math.inf
        x = np.r_[z, t]
    count = 128 if entry else 32
    for component in components(models, entry=entry):
        terms.append(math.log(component["count"]/count)
                     +log_density(x, component["mean"], component["covariance"], student=component["kind"] == "student5")
                     -math.log(jac))
    return float(_LOGSUMEXP(terms))


def draw_component(rng, component, *, entry=False):
    np = science()
    if component is None:
        z = rng.uniform(np.r_[1.2, 1.2, -1., [-5.]*6], np.r_[5., 5., 1., [5.]*6])
        cosine, phi = float(rng.uniform(-1, 1)), float(rng.uniform(0, 2*math.pi))
        radius = math.sqrt(max(0., 1-cosine*cosine))
        return z, np.array([radius*math.cos(phi), radius*math.sin(phi), cosine])
    scale = component["covariance"]*(12/5 if component["kind"] == "student5" else 1)
    displacement = np.linalg.cholesky(scale)@rng.standard_normal(len(component["mean"]))
    if component["kind"] == "student5":
        displacement /= math.sqrt(float(rng.chisquare(5))/5)
    x = component["mean"]+displacement
    if not entry:
        return x[:9], sphere(x[9:])[0]
    cosine, phi = float(rng.uniform(-1, 1)), float(rng.uniform(0, 2*math.pi))
    radius = math.sqrt(max(0., 1-cosine*cosine))
    return x, np.array([radius*math.cos(phi), radius*math.sin(phi), cosine])


def geometry(z, u, h, alpha, model, *, length=30.):
    """Unchanged coarea algebra, explicit nested domains; no PES operation."""
    np = science(); old = legacy(); z, u = np.asarray(z, float), np.asarray(u, float)
    row = {"coordinates_bohr": None, "common_zero_reason": None,
           "domains": {key: False for key in DOMAINS}, "log_geometric_measure": None,
           "minimum_pair_bohr": None, "CH_radii_bohr": None}
    q = old.from_chart(z, h)
    if q is None:
        row["common_zero_reason"] = "invalid_chart"; return row
    radii = np.linalg.norm(q[list(CHANNELS)]-q[1], axis=1)
    row["CH_radii_bohr"] = radii.tolist()
    if not np.all((radii >= 1.2)&(radii <= 5.)):
        row["common_zero_reason"] = "outside_outer_methane_domain"; return row
    local = replace(model, length=length)
    if alpha is None:
        q[5] = np.average(q[:5], weights=local.masses[:5], axis=0)+length*u
        log_measure = math.log(old.shape_jacobian(z))+2*math.log(length)-.5*math.log(2*math.pi*local.beta*local.reduced_mass)
    else:
        rho = old.root_radius(q, h, u, alpha, local)
        if rho is None:
            row["common_zero_reason"] = "no_positive_root"; return row
        q[5] = q[h]+rho*u
        metric, derivative = old.inverse_mass_metric(q, h, alpha, local)
        log_measure = (math.log(old.shape_jacobian(z))+2*math.log(rho)-math.log(-derivative)
                       +.5*math.log(metric/(2*math.pi*local.beta)))
    row["coordinates_bohr"] = q.tolist()
    pairs = np.linalg.norm(q[:, None]-q[None, :], axis=-1)[np.triu_indices(6, 1)]
    minimum = float(pairs.min()); row["minimum_pair_bohr"] = minimum
    values = old.channel_values(q, local); active = CHANNELS[int(np.argmax(values))]
    R = float(np.linalg.norm(q[5]-np.average(q[:5], weights=local.masses[:5], axis=0)))
    if minimum < .6:
        row["common_zero_reason"] = "outside_outer_pair_domain"
    elif alpha is not None and active != h:
        row["common_zero_reason"] = "inactive_max_channel"
    elif length-R-float(values.max()) <= 0:
        row["common_zero_reason"] = "disconnected_xi_component"
    else:
        row["log_geometric_measure"] = float(log_measure)
        row["domains"] = {name: bool(np.all((radii >= lo)&(radii <= hi)) and minimum >= pair)
                          for name, (lo, hi, pair) in DOMAINS.items()}
    return row


def training_specs(model):
    np = science(); old = legacy()
    q = np.asarray(model.reference)-model.reference[1]
    ez = q[3]/np.linalg.norm(q[3]); ex = q[0]-np.dot(q[0], ez)*ez; ex /= np.linalg.norm(ex)
    axes = np.array([ex, np.cross(ez, ex), ez])
    u = axes@(q[5]-q[3]); u /= np.linalg.norm(u)
    z = old.to_chart(q, 3)
    if z[4] < 0:
        z[[4, 7]] *= -1; u[1] *= -1
    t, _ = inverse_sphere(u); require(t is not None, "reference at unsupported stereographic pole")
    entry = old.proposal_means(model, 3, entry=True)[0].copy()
    if entry[4] < 0:
        entry[[4, 7]] *= -1
    return {name: {"x0": (entry if name == "entry" else np.r_[z, t]).tolist(),
                   "scales": ([.12, .12, .12]+[.20]*6 if name == "entry" else
                              [.15, .12, .12]+[.20]*6+[1/(2*math.sqrt(8))]*2),
                   "alpha": None if name == "entry" else float(name.split("_")[1])}
            for name in TARGETS}


def stencil(dimension):
    np = science(); zero = np.zeros(dimension)
    rows = [("center", zero.copy())]
    for i in range(dimension):
        for sign in (-1, 1):
            point = zero.copy(); point[i] = sign*.02; rows.append((f"diag:{i}:{sign}", point))
    for i in range(dimension):
        for j in range(i+1, dimension):
            for a in (-1, 1):
                for b in (-1, 1):
                    point = zero.copy(); point[i] = a*.02; point[j] = b*.02
                    rows.append((f"cross:{i}:{j}:{a}:{b}", point))
    for i in range(dimension):
        for sign in (-1, 1):
            point = zero.copy(); point[i] = sign*.01; rows.append((f"refine:{i}:{sign}", point))
    require(len(rows) == 1+2*dimension*dimension+2*dimension, "stencil count differs")
    return rows


def calibration_point(name, tag, x, model):
    np = science(); x = np.asarray(x, float)
    u, jacobian = (np.array([0., 0., 1.]), 1.) if name == "entry" else sphere(x[9:])
    alpha = None if name == "entry" else float(name.split("_")[1])
    row = geometry(x[:9], u, 3, alpha, model)
    row.update(target=name, tag=tag, x=x.tolist(),
               log_training_measure=(None if row["log_geometric_measure"] is None else row["log_geometric_measure"]+math.log(jacobian)))
    return row


def initial_calibration_ledger(specs, model):
    np = science(); rows = []
    for name in TARGETS:
        spec = specs[name]; center, scales = np.asarray(spec["x0"]), np.asarray(spec["scales"])
        for tag, y in stencil(len(center)):
            row = calibration_point(name, tag, center+scales*y, model)
            row.update(index=len(rows), y=y.tolist()); rows.append(row)
    require(len(rows) == 711, "initial calibration ledger must have711 points")
    return rows


def fit_quadratic(spec, values):
    np = science(); d = len(spec["x0"]); h = .02
    H = np.zeros((d, d)); gradient = np.zeros(d); refined = np.zeros(d); center = values["center"]
    require(all(math.isfinite(value) for value in values.values()), "nonfinite calibration objective")
    for i in range(d):
        plus, minus = values[f"diag:{i}:1"], values[f"diag:{i}:-1"]
        H[i, i] = (plus-2*center+minus)/h**2; gradient[i] = (plus-minus)/(2*h)
        refined[i] = (values[f"refine:{i}:1"]-2*center+values[f"refine:{i}:-1"])/(.01**2)
        for j in range(i+1, d):
            H[i, j] = H[j, i] = sum(a*b*values[f"cross:{i}:{j}:{a}:{b}"] for a in (-1, 1) for b in (-1, 1))/(4*h*h)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    require(np.isfinite(H).all() and eigenvalues[0] > 1e-6 and eigenvalues[-1]/eigenvalues[0] <= 1e8,
            "failed positive-Hessian construction; no clipping or fallback")
    discrepancy = float(np.max(abs(np.diag(H)-refined)/np.maximum(1., abs(refined))))
    require(discrepancy <= .05, "Hessian finite-difference refinement failed")
    delta = -np.linalg.solve(H, gradient); distance = float(math.sqrt(max(0., delta@H@delta)))
    require(distance <= 3, "Newton displacement exceeds fixed bound; no clipping or optimizer")
    scales = np.asarray(spec["scales"]); covariance_y = np.linalg.inv(H)
    mean = np.asarray(spec["x0"])+scales*delta
    covariance = covariance_y*scales[:, None]*scales[None, :]
    axes = scales[:, None]*eigenvectors/np.sqrt(eigenvalues)[None, :]
    return {"mean_x": mean.tolist(), "covariance_x": covariance.tolist(), "H_y": H.tolist(),
            "gradient_y": gradient.tolist(), "delta_y": delta.tolist(), "eigenvalues_y": eigenvalues.tolist(),
            "unit_probe_axes_x": axes.tolist(), "Newton_Mahalanobis_length": distance,
            "maximum_diagonal_refinement_discrepancy": discrepancy}


def followup_calibration_ledger(models, model):
    np = science(); rows = []
    for name in TARGETS:
        fitted = models[name]; mean, axes = np.asarray(fitted["mean_x"]), np.asarray(fitted["unit_probe_axes_x"])
        originals = [("shifted_center", mean)]
        for i in range(len(mean)):
            for sign in (-1, 1):
                originals.append((f"probe:{i}:{sign}", mean+sign*axes[:, i]))
        for tag, x in originals:
            row = calibration_point(name, tag, x, model); row["index"] = 711+len(rows); rows.append(row)
        reflected_tags = ["shifted_center", *[f"probe:{i}:{sign}" for i in range(3) for sign in (-1, 1)], "probe:3:1"]
        lookup = dict(originals)
        for tag in reflected_tags:
            row = calibration_point(name, "mirror:"+tag, reflection(len(mean))*lookup[tag], model)
            row.update(index=711+len(rows), reflected_original_tag=tag); rows.append(row)
    require(len(rows) == 89, "followup calibration ledger must have89 points")
    return rows


def check_followup(rows, energy, objective):
    np = science(); records = {}
    lookup = {(row["target"], row["tag"]): row for row in rows}
    for name in TARGETS:
        center = objective[(name, "shifted_center")]
        residuals = [abs(value-center-.5) for (target, tag), value in objective.items() if target == name and tag.startswith("probe:")]
        require(residuals and max(residuals) <= 5, "unit-width quadratic probe failed; no refit")
        differences, distances = [], []
        for row in rows:
            if row["target"] != name or "reflected_original_tag" not in row:
                continue
            tag = row["reflected_original_tag"]
            differences.append(abs(energy[(name, row["tag"])]-energy[(name, tag)]))
            first = np.asarray(row["coordinates_bohr"]); second = np.asarray(lookup[(name, tag)]["coordinates_bohr"])
            distances.append(float(np.max(abs(np.linalg.norm(first[:, None]-first[None, :], axis=-1)
                                              -np.linalg.norm(second[:, None]-second[None, :], axis=-1)))))
        require(len(differences) == 8 and max(differences) <= 1e-10 and max(distances) <= 1e-12,
                "reflection calibration failed")
        records[name] = {"maximum_quadratic_probe_residual": max(residuals),
                         "maximum_reflection_energy_difference_hartree": max(differences),
                         "maximum_reflection_distance_difference_bohr": max(distances)}
    return records


def diagnostic_ledger(models, model):
    np = science(); rows, controls = [], []
    for seed in SEEDS:
        rng = np.random.Generator(np.random.PCG64(seed))
        for h in CHANNELS:
            strata = [(None, 4), *[(component, component["count"]) for component in components(models)]]
            for component, count in strata:
                label = "uniform" if component is None else component["name"]
                for number in range(count):
                    z, u = draw_component(rng, component)
                    logq = mixture_log_density(z, u, models)
                    for alpha in (.999, 1.007):
                        row = geometry(z, u, h, alpha, model)
                        row.update(index=len(rows), seed=seed, channel=h, target=f"alpha_{alpha:.3f}",
                                   latent_id=f"{seed}:{h}:{label}:{number}", component=label, component_draw=number,
                                   shape=z.tolist(), direction=u.tolist(), log_proposal_density=logq, separation_bohr=30.)
                        rows.append(row)
        strata = [(None, 16), *[(component, component["count"]) for component in components(models, entry=True)]]
        for component, count in strata:
            label = "uniform" if component is None else component["name"]
            control_count = 2 if component is None else 6 if component["kind"] == "normal" else 1
            for number in range(count):
                z, u = draw_component(rng, component, entry=True)
                logq = mixture_log_density(z, u, models, entry=True)
                row = geometry(z, u, 3, None, model)
                row.update(index=len(rows), seed=seed, channel=None, target="entry", latent_id=f"{seed}:entry:{label}:{number}",
                           component=label, component_draw=number, shape=z.tolist(), direction=u.tolist(),
                           log_proposal_density=logq, separation_bohr=30.)
                rows.append(row)
                if number < control_count:
                    control = geometry(z, u, 3, None, model, length=60.)
                    control.update(seed=seed, channel=None, target="entry", latent_id=row["latent_id"], component=label,
                                   component_draw=number, shape=z.tolist(), direction=u.tolist(),
                                   log_proposal_density=logq, separation_bohr=60., paired_main_index=row["index"])
                    controls.append(control)
    require(len(rows) == 3072 and len(controls) == 128, "complete fresh ledger counts differ")
    for i, row in enumerate(controls):
        row["index"] = 3072+i
    return rows, controls


class AdmissionStop(RuntimeError):
    pass


class EvaluationBudget:
    """One attempted geometry is charged even when the evaluator fails."""

    def __init__(self, output, evaluator, *, cpu_clock=time.process_time, wall_clock=time.monotonic,
                 deadline=math.inf):
        self.path = Path(output)/"calls.jsonl"
        require(not self.path.exists(), "call journal already exists; no repeated attempt")
        self.evaluator, self.cpu_clock, self.wall_clock, self.deadline = evaluator, cpu_clock, wall_clock, deadline
        self.attempted, self.successful = 0, 0
        self.completed_rows = Counter()

    def admit(self):
        if 5+self.cpu_clock() >= 55:
            raise AdmissionStop("aggregate CPU admission exhausted: two preparation + three supervisor seconds charged")
        if self.wall_clock() >= self.deadline-2:
            raise AdmissionStop("wall admission exhausted; two-second finalization reserve")
        if self.attempted >= 4000:
            raise AdmissionStop("the fixed4000-geometry ceiling is exhausted")

    def evaluate(self, row, stage):
        self.admit(); self.attempted += 1; begin = self.cpu_clock()
        event(self.path, {"event": "attempt", "attempt_number": self.attempted, "stage": stage,
                          "index": row["index"], "target": row["target"], "CPU_before_call": begin, "utc": now()})
        try:
            np = science(); value = float(self.evaluator(np.asarray(row["coordinates_bohr"], float)))
            require(math.isfinite(value), "evaluator returned a nonfinite energy")
        except BaseException as error:
            event(self.path, {"event": "failure", "attempt_number": self.attempted, "stage": stage,
                              "index": row["index"], "error_type": type(error).__name__, "message": str(error), "utc": now()})
            raise
        self.successful += 1
        event(self.path, {"event": "success", "attempt_number": self.attempted, "stage": stage,
                          "index": row["index"], "energy_hartree": value,
                          "call_CPU_seconds": self.cpu_clock()-begin, "utc": now()})
        return value


def evaluate_calibration(rows, budget, model, stage):
    require(all(row["domains"]["original"] and row["common_zero_reason"] is None for row in rows),
            "calibration leaves the original domain/chart; no fallback or discarded stencil")
    energy, objective = {}, {}
    for row in rows:
        value = budget.evaluate(row, stage)
        key = (row["target"], row["tag"])
        energy[key] = value
        objective[key] = float(model.beta*(value-model.energy_offset)-row["log_training_measure"])
        require(math.isfinite(objective[key]), "nonfinite calibration objective")
        budget.completed_rows[stage] += 1
    return energy, objective


def evaluate_diagnostics(rows, budget, model, stage):
    result = []
    for row in rows:
        if row["common_zero_reason"] is not None or not row["domains"]["outer"]:
            result.append({"index": row["index"], "energy_hartree": None,
                           "log_weight": {name: None for name in DOMAINS}})
            budget.completed_rows[stage] += 1
            continue
        energy = budget.evaluate(row, stage)
        weight = row["log_geometric_measure"]-row["log_proposal_density"]-model.beta*(energy-model.energy_offset)
        require(math.isfinite(weight), "nonfinite importance weight; never silently zero/drop")
        result.append({"index": row["index"], "energy_hartree": energy,
                       "log_weight": {name: float(weight) if row["domains"][name] else None for name in DOMAINS}})
        budget.completed_rows[stage] += 1
    return result


def logmean(logs):
    science(); finite = [value for value in logs if value is not None]
    return None if not finite else float(_LOGSUMEXP(finite)-math.log(len(logs)))


def concentration(logs):
    np = science(); finite = np.asarray([value for value in logs if value is not None], float)
    if not len(finite):
        return {"draws": len(logs), "positive_weights": 0, "log_mean": None, "empirical_ESS": 0.,
                "maximum_normalized_weight": None, "largest_one_percent_weight_mass": None}
    total = float(_LOGSUMEXP(finite)); weights = np.exp(finite-total)
    return {"draws": len(logs), "positive_weights": len(finite), "log_mean": total-math.log(len(logs)),
            "empirical_ESS": float(1/(weights@weights)), "maximum_normalized_weight": float(weights.max()),
            "largest_one_percent_weight_mass": float(np.sort(weights)[::-1][:math.ceil(.01*len(logs))].sum())}


def integral_logs(rows, results, *, domain="original"):
    science(); table = {row["index"]: row for row in results}; outcome = {}
    for target in TARGETS:
        selected = [row for row in rows if row["target"] == target]
        if target == "entry":
            outcome[target] = logmean([table[row["index"]]["log_weight"][domain] for row in selected])
        else:
            means = [logmean([table[row["index"]]["log_weight"][domain] for row in selected if row["channel"] == h])
                     for h in CHANNELS]
            finite = [value for value in means if value is not None]
            outcome[target] = None if not finite else float(_LOGSUMEXP(finite))
    return outcome


def paired_log_ratio(first, second):
    """Ratio of pooled means and nominal eight-vector delta-method uncertainty."""
    np = science()
    require(len(first) == len(second) == 8, "eight complete independent seed vectors required")
    a, b = logmean(first), logmean(second)
    if a is None or b is None:
        return {"log_ratio_of_pooled_means": None, "resolved": False,
                "reason": "zero pooled integral; no epsilon substitution", "seed_vectors": 8}
    normalized_a = np.array([0. if value is None else math.exp(value-a) for value in first])
    normalized_b = np.array([0. if value is None else math.exp(value-b) for value in second])
    influence = normalized_a-normalized_b
    standard_error = float(influence.std(ddof=1)/math.sqrt(8))
    from scipy.stats import t as student_t
    half = float(student_t.ppf(.975, 7)*standard_error)
    return {"log_ratio_of_pooled_means": a-b, "resolved": True, "seed_vectors": 8,
            "degrees_of_freedom": 7, "seed_influence": influence.tolist(),
            "nominal_delta_standard_error": standard_error, "nominal_delta_t95_interval": [a-b-half, a-b+half],
            "interpretation": "Conditional small-sample Monte Carlo approximation; not a tail bound or physical-rate confidence interval."}


def validate_fresh_membership(rows, controls, results, control_results):
    require(len(rows) == len(results) == 3072 and len(controls) == len(control_results) == 128,
            "partial fresh cohorts cannot be analyzed")
    require([row["index"] for row in rows] == [row["index"] for row in results] == list(range(3072))
            and [row["index"] for row in controls] == [row["index"] for row in control_results] == list(range(3072, 3200)),
            "fresh draw identities differ")
    for seed in SEEDS:
        subset = [row for row in rows if row["seed"] == seed]
        require(len(subset) == 384, "each seed requires both128-point TS surfaces and128 entry points")
        for h in CHANNELS:
            pair = [[row for row in subset if row["target"] == target and row["channel"] == h] for target in TARGETS[:2]]
            require(len(pair[0]) == len(pair[1]) == 32, "four fixed TS channel counts required")
            expected = {"uniform": 4, **{f"{kind}:{target}:{mirror}": 6 if kind == "normal" else 1
                                         for kind in ("normal", "student5") for target in TARGETS[:2]
                                         for mirror in ("positive", "negative")}}
            for selected in pair:
                require(Counter(row["component"] for row in selected) == expected, "fixed TS mixture strata differ")
                for component, count in expected.items():
                    require(sorted(row["component_draw"] for row in selected if row["component"] == component) == list(range(count)),
                            "TS stratum draw identities differ")
            for a, b in zip(*pair):
                require(all(a[key] == b[key] for key in ("latent_id", "component", "component_draw", "shape", "direction", "log_proposal_density")),
                        "paired alpha draws differ")
        entry = [row for row in subset if row["target"] == "entry"]
        require(len(entry) == 128 and all(row["channel"] is None for row in entry), "one128-draw entry chart required")
        entry_counts = {"uniform": 16, **{f"{kind}:entry:{mirror}": 48 if kind == "normal" else 8
                                         for kind in ("normal", "student5") for mirror in ("positive", "negative")}}
        require(Counter(row["component"] for row in entry) == entry_counts, "fixed entry mixture strata differ")
        for component, count in entry_counts.items():
            require(sorted(row["component_draw"] for row in entry if row["component"] == component) == list(range(count)),
                    "entry stratum draw identities differ")
        chosen = [row for row in controls if row["seed"] == seed]
        require(len(chosen) == 16 and len({row["paired_main_index"] for row in chosen}) == 16,
                "the matched16-draw separation subset differs")
        expected_control = {key: (2 if key == "uniform" else 6 if key.startswith("normal:") else 1) for key in entry_counts}
        require(Counter(row["component"] for row in chosen) == expected_control, "fixed matched-control strata differ")
        for row in chosen:
            parent = rows[row["paired_main_index"]]
            require(parent["target"] == row["target"] == "entry" and parent["seed"] == row["seed"]
                    and parent["separation_bohr"] == 30. and row["separation_bohr"] == 60.
                    and all(parent[key] == row[key] for key in ("latent_id", "component", "component_draw", "shape", "direction", "log_proposal_density")),
                    "separation control must use exactly the same selected entry geometry")
            threshold = 2 if row["component"] == "uniform" else 6 if row["component"].startswith("normal:") else 1
            require(0 <= row["component_draw"] < threshold, "post-energy separation selection is forbidden")


def analyze(rows, controls, results, control_results):
    np = science(); validate_fresh_membership(rows, controls, results, control_results)
    table = {row["index"]: row for row in results}; control_table = {row["index"]: row for row in control_results}
    seed_logs = {seed: integral_logs([row for row in rows if row["seed"] == seed], results) for seed in SEEDS}
    pooled = integral_logs(rows, results)
    summaries, channel_summaries, screen = {}, {}, {}
    for target in TARGETS:
        selected = [row for row in rows if row["target"] == target]
        logs = [table[row["index"]]["log_weight"]["original"] for row in selected]
        summary = concentration(logs)
        leave_changes = []
        for omitted in SEEDS:
            value = logmean([seed_logs[seed][target] for seed in SEEDS if seed != omitted])
            leave_changes.append(None if value is None or pooled[target] is None else abs(value-pooled[target]))
        summary["leave_one_seed_out_absolute_log_integral_change"] = leave_changes
        summary["component_contributions"] = {}
        total_log = None if summary["log_mean"] is None else summary["log_mean"]+math.log(len(logs))
        for component in sorted({row["component"] for row in selected}):
            indices = [i for i, row in enumerate(selected) if row["component"] == component]
            group = [logs[i] for i in indices]; stats = concentration(group)
            stats["observed_fraction_of_total_weight"] = (None if total_log is None else 0. if stats["log_mean"] is None
                                                          else math.exp(stats["log_mean"]+math.log(len(group))-total_log))
            summary["component_contributions"][component] = stats
        summary["nested_domains"] = {}
        for domain in DOMAINS:
            values = [table[row["index"]]["log_weight"][domain] for row in selected]
            summary["nested_domains"][domain] = concentration(values)
        shell = [table[row["index"]]["log_weight"]["outer"] if not row["domains"]["original"] else None for row in selected]
        shell_stats = concentration(shell)
        shell_stats["positive_seed_count"] = len({row["seed"] for row, value in zip(selected, shell) if value is not None})
        shell_stats["zero_observation_upper_bound_claimed"] = False
        summary["outer_shell"] = shell_stats
        summaries[target] = summary
        screen[target] = bool(summary["empirical_ESS"] >= 128 and summary["maximum_normalized_weight"] is not None
                              and summary["maximum_normalized_weight"] <= .025
                              and summary["largest_one_percent_weight_mass"] <= .20
                              and all(value is not None and value <= .25 for value in leave_changes))
        if target != "entry":
            for h in CHANNELS:
                values = [table[row["index"]]["log_weight"]["original"] for row in selected if row["channel"] == h]
                stats = concentration(values); key = f"{target}:channel{h}"
                channel_summaries[key] = stats
                screen[key] = bool(stats["empirical_ESS"] >= 32 and stats["maximum_normalized_weight"] is not None
                                   and stats["maximum_normalized_weight"] <= .10)
    ratios = {target: paired_log_ratio([seed_logs[seed][target] for seed in SEEDS], [seed_logs[seed]["entry"] for seed in SEEDS])
              for target in TARGETS[:2]}
    paired_alpha = paired_log_ratio([seed_logs[seed][TARGETS[1]] for seed in SEEDS], [seed_logs[seed][TARGETS[0]] for seed in SEEDS])
    before, after = [], []
    for seed in SEEDS:
        chosen = [row for row in controls if row["seed"] == seed]
        first = [table[row["paired_main_index"]]["log_weight"]["original"] for row in chosen]
        second = [control_table[row["index"]]["log_weight"]["original"] for row in chosen]
        before.append(logmean([None if value is None else value-2*math.log(30) for value in first]))
        after.append(logmean([None if value is None else value-2*math.log(60) for value in second]))
    separation = paired_log_ratio(after, before)
    separation.update(draws_at_each_separation=128, matched_draws_per_seed=16,
                      primary_entry_denominator_draws_per_seed=128, full_finite_Rinf_equivalence_claimed=False)
    return {"all_eight_seeds": list(SEEDS), "pooled_log_integrals_without_common_rotation": pooled,
            "seed_log_integrals_without_common_rotation": {str(seed): value for seed, value in seed_logs.items()},
            "dimensionless_flux_ratio_diagnostics": ratios, "paired_alpha_log_integral_difference": paired_alpha,
            "entry_only_separation_control": separation, "weight_diagnostics": summaries,
            "channel_weight_diagnostics": channel_summaries, "efficiency_screen_components": screen,
            "original_domain_efficiency_screen_passed": all(screen.values()),
            "efficiency_screen_is_scientific_validation": False, "automatic_continuation": False}


def run_stages(output, model, evaluator, *, registration_sha, guards=None,
               cpu_clock=time.process_time, wall_clock=time.monotonic, deadline=math.inf):
    """Caller supplies an evaluator; tests use analytic stand-ins, never CBE."""
    output = Path(output)
    require(not (output/"report.json").exists(), "a report already exists; no automatic restart")
    budget = EvaluationBudget(output, evaluator, cpu_clock=cpu_clock, wall_clock=wall_clock, deadline=deadline)
    stage, ledgers, proposal_sha = "initial_calibration", {}, None
    calibration_complete, main_complete, control_complete = False, 0, 0
    result, failure, status, unchanged = None, None, "failed", True
    try:
        budget.admit()
        specs = training_specs(model); initial = initial_calibration_ledger(specs, model)
        write_new_json(output/"calibration-initial-ledger.json", {"design_sha256": DESIGN_SHA, "specifications": specs,
                                                                 "rows": initial, "PES_evaluations_before_this_ledger": 0})
        ledgers["calibration-initial-ledger.json"] = sha(output/"calibration-initial-ledger.json")
        if guards:
            check_hashes(guards)
        _, objectives = evaluate_calibration(initial, budget, model, stage)
        models = {name: fit_quadratic(specs[name], {tag: value for (target, tag), value in objectives.items() if target == name})
                  for name in TARGETS}
        stage = "followup_calibration"; followup = followup_calibration_ledger(models, model)
        write_new_json(output/"calibration-followup-ledger.json", {"initial_ledger_sha256": ledgers["calibration-initial-ledger.json"],
                                                                  "models": models, "rows": followup,
                                                                  "PES_evaluations_before_this_ledger": budget.attempted})
        ledgers["calibration-followup-ledger.json"] = sha(output/"calibration-followup-ledger.json")
        if guards:
            check_hashes(guards)
        energies, objectives = evaluate_calibration(followup, budget, model, stage)
        checks = check_followup(followup, energies, objectives)
        require(budget.attempted == budget.successful == 800, "complete calibration must account for all800 evaluations")
        write_new_json(output/"frozen-proposal.json", {"design_sha256": DESIGN_SHA, "models": models,
                                                      "calibration_checks": checks, "calibration_ledgers": ledgers,
                                                      "calibration_PES_evaluations": 800,
                                                      "fresh_diagnostic_draws_generated": 0})
        proposal_sha = sha(output/"frozen-proposal.json"); calibration_complete = True
        stage = "main_diagnostics"; main, controls = diagnostic_ledger(models, model)
        write_new_json(output/"diagnostic-ledger.json", {"design_sha256": DESIGN_SHA, "proposal_sha256": proposal_sha,
                                                        "main_rows": main, "control_rows": controls,
                                                        "fresh_PES_evaluations_before_this_ledger": 0})
        ledgers["diagnostic-ledger.json"] = sha(output/"diagnostic-ledger.json")
        if guards:
            check_hashes(guards)
        main_results = evaluate_diagnostics(main, budget, model, stage); main_complete = len(main_results)
        write_new_json(output/"main-results.json", {"ledger_sha256": ledgers["diagnostic-ledger.json"], "results": main_results})
        ledgers["main-results.json"] = sha(output/"main-results.json")
        stage = "entry_separation_control"
        control_results = evaluate_diagnostics(controls, budget, model, stage); control_complete = len(control_results)
        write_new_json(output/"control-results.json", {"ledger_sha256": ledgers["diagnostic-ledger.json"], "results": control_results})
        ledgers["control-results.json"] = sha(output/"control-results.json")
        require(budget.attempted <= 4000 and budget.successful == budget.attempted, "geometry budget or success accounting differs")
        result = analyze(main, controls, main_results, control_results)
        require(sha(output/"frozen-proposal.json") == proposal_sha, "frozen proposal changed during diagnostics")
        for name, digest in ledgers.items():
            require(sha(output/name) == digest, "staged ledger or result changed during execution")
        status, stage = "complete_excluded_efficiency_pilot", "complete"
    except BaseException as error:
        failure = {"stage": stage, "error_type": type(error).__name__, "message": str(error)}
        status = "incomplete_resource_admission" if isinstance(error, AdmissionStop) else "failed_calibration_or_evaluation"
    if budget.path.exists():
        ledgers["calls.jsonl"] = sha(budget.path)
    if guards:
        try:
            check_hashes(guards)
        except BaseException as error:
            unchanged, result, status = False, None, "failed_source_guard"
            failure = {"stage": stage, "error_type": type(error).__name__, "message": str(error)}
    report = {"schema": "pyrpmd.hch4-reference-efficiency-pilot.v1", "status": status,
              "scope": "Excluded P1 proposal-efficiency study; no physical rate, quantum or literature result.",
              "registration_sha256": registration_sha, "source_guards_unchanged": unchanged, "stage": stage,
              "ledger_sha256": ledgers, "proposal_sha256": proposal_sha,
              "attempted_PES_geometries": budget.attempted, "successful_PES_geometries": budget.successful,
              "calibration_complete": calibration_complete, "main_draws_complete": budget.completed_rows["main_diagnostics"],
              "control_draws_complete": budget.completed_rows["entry_separation_control"], "failure": failure,
              "resource_accounting": {"charged_prior_process_caps_seconds": 5,
                                       "worker_process_CPU_seconds_before_report": cpu_clock(),
                                       "charged_CPU_seconds_before_report": 5+cpu_clock(), "geometry_ceiling": 4000},
              "analysis": result,
              "limitations": ["Efficiency does not establish physical-domain, tail or absolute-normalization accuracy.",
                              "Eight conditional independent seed vectors; all paired observations stay together.",
                              "The60bohr entry control does not establish full finite-Rinf surface/rate convergence.",
                              "No automatic extension, proposal refit, replacement draw or rate-target decision."]}
    require(set(report) == set(OUTPUT_FIELDS), "undeclared output fields")
    write_new_json(output/"report.json", report)
    return report


def check_hashes(values):
    for path, expected in values.items():
        require(sha(path) == expected, f"frozen source/model changed: {path}")


def required_sources(design):
    """Pure file inventory; no scientific package import or PES loading."""
    values = {str((ROOT/name).resolve()): digest for name, digest in design["protected_source_sha256"].items()}
    # Bind the complete Python package imported by the potential adapter; this
    # avoids an unrecorded transitive helper. A current-file inventory must
    # never overwrite the frozen design's protected identities.
    paths = [SELF, TESTS, LEGACY, DESIGN, METHOD_REVIEW,
             ROOT/"benchmarks/hch4/wrapper.f90", ROOT/design["unchanged_measure_source"]["proposal"],
             *sorted((ROOT/"src/pyrpmd").rglob("*.py"))]
    for path in paths:
        name, observed = str(path.resolve()), sha(path)
        require(name not in values or values[name] == observed,
                f"reviewed protected source changed; cannot rebaseline: {name}")
        values.setdefault(name, observed)
    return values


def environment_identity():
    executable = Path(sys.executable).resolve()
    return {"python": platform.python_version(), "python_executable": str(executable),
            "python_executable_sha256": sha(executable), "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"), "scipy": importlib.metadata.version("scipy")}


def receipt_identity(row):
    path = Path(row["path"]).resolve()
    require(path.is_file() and sha(path) == row["sha256"] and path.stat().st_size == row["bytes"],
            "bound receipt identity differs")
    return path


def validate_implementation_receipts(plan):
    """A file hash binds bytes; explicit semantics also bind their approval."""
    expected = {str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in (SELF, TESTS)}
    review = read_json(plan["implementation_review"]["path"])
    require(review.get("approved_for_single_excluded_efficiency_pilot") is True
            and review.get("remaining_blockers") == []
            and review.get("design_sha256") == DESIGN_SHA
            and review.get("source_sha256") == sha(SELF)
            and review.get("tests_sha256") == sha(TESTS),
            "an approving implementation review for the exact source/test/design is required")
    gate = read_json(plan["synthetic_gate"]["path"])
    require(gate.get("schema") == "pyrpmd.hch4.efficiency-pilot.synthetic-gate.v1"
            and type(gate.get("exit_code")) is int and gate["exit_code"] == 0
            and gate.get("source_unchanged") is True and gate.get("source_files") == expected
            and gate.get("real_PES_evaluations") == 0 and gate.get("real_dynamics_steps") == 0
            and gate.get("actual_pilot_registration_or_execution") is False,
            "a successful current-source synthetic gate is required")
    suites = gate.get("testsuites")
    require(isinstance(suites, list) and suites and all(isinstance(row, dict) for row in suites),
            "complete synthetic test-suite counts are required")
    require(all(int(row["tests"]) > 0 and all(int(row[key]) == 0 for key in ("errors", "failures", "skipped"))
                for row in suites), "the required synthetic gate may contain no failures, errors or skips")


def load_plan(path):
    path = Path(path).resolve(); plan = read_json(path)
    require(sha(DESIGN) == DESIGN_SHA and sha(METHOD_REVIEW) == METHOD_REVIEW_SHA
            and sha(LEGACY) == LEGACY_SHA, "reviewed design/method/coarea source changed")
    design = read_json(DESIGN)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("parent_adoption") is True
            and plan.get("design_sha256") == DESIGN_SHA and plan.get("method_review_sha256") == METHOD_REVIEW_SHA,
            "separate reviewed execution adoption is required")
    require(plan.get("seeds") == list(SEEDS) and plan.get("geometry_counts") == {
        "initial_calibration": 711, "followup_calibration": 89, "main": 3072, "control": 128, "maximum_total": 4000},
        "fixed cohort or staged geometry ceiling differs")
    require(plan.get("resources") == {"CPUs": 1, "prepare_CPU_seconds": 2, "supervisor_CPU_seconds": 3,
                                     "worker_CPU_seconds": 55, "sum_CPU_caps_seconds": 60,
                                     "combined_soft_CPU_seconds": 55, "wall_seconds": 120}
            and plan.get("output_fields") == list(OUTPUT_FIELDS)
            and plan.get("automatic_retry") is False and plan.get("automatic_extension") is False,
            "resource/output/one-attempt policy differs from the reviewed design")
    require(type(plan.get("cpu_affinity")) is int and plan["cpu_affinity"] in os.sched_getaffinity(0),
            "one actually available adopted CPU is required")
    require(plan.get("environment") == environment_identity(), "adopted runtime identity differs")
    required = required_sources(design)
    expected_model = design["fixed_PES_identity"]
    library = Path(plan["library"]["path"]).resolve()
    require(str(library) == expected_model["absolute_path"] and plan["library"]["sha256"] == expected_model["sha256"],
            "fixed CBE model path/hash differs")
    receipt_identity(plan["library"]); build = receipt_identity(plan["library_build_receipt"])
    require(build == library.with_suffix(".build.json"), "model build receipt route differs")
    required[str(library)] = plan["library"]["sha256"]; required[str(build)] = plan["library_build_receipt"]["sha256"]
    for key in ("synthetic_gate", "implementation_review", "resource_adoption"):
        item = receipt_identity(plan[key]); required[str(item)] = plan[key]["sha256"]
    validate_implementation_receipts(plan)
    require(plan.get("source_sha256") == required, "complete source/model/receipt inventory differs")
    check_hashes(required)
    resource_adoption = read_json(plan["resource_adoption"]["path"])
    require(resource_adoption.get("authorize_single_bounded_pilot") is True
            and resource_adoption.get("available_CPU_count") == 1
            and resource_adoption.get("maximum_PES_geometries") == 4000,
            "a concrete one-CPU resource adoption is required")
    output = Path(plan["output"]).resolve()
    require(plan["output"] == str(output) and output.is_relative_to(ROOT.parent)
            and output != ROOT.parent and not output.is_relative_to(ROOT),
            "live pilot output must be a separate adopted workspace directory")
    return plan, required


def prepare(plan_path, output):
    require(ROLE == "prepare" and RESOURCE_ROLE, "preparation requires its dedicated capped interpreter")
    plan, guards = load_plan(plan_path); output = Path(output).resolve()
    require(output == Path(plan["output"]).resolve(), "preparation output differs from adoption")
    os.sched_setaffinity(0, {plan["cpu_affinity"]})
    require(time.process_time() < 2 and time.monotonic()-ENTRY_MONOTONIC < 118,
            "preparation resource admission exhausted")
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output/"PREPARATION_ONCE.json", {"process_token": PROCESS_TOKEN, "PID": os.getpid(), "created_utc": now()})
    registration = {"schema": REG_SCHEMA, "created_utc": now(), "output": str(output),
                    "plan_path": str(Path(plan_path).resolve()), "plan_sha256": sha(plan_path),
                    "design_sha256": DESIGN_SHA, "source_sha256": guards, "environment": environment_identity(),
                    "preparation_process_token": PROCESS_TOKEN, "preparation_PID": os.getpid(),
                    "preparation_CPU_seconds_before_registration_write": time.process_time(),
                    "preparation_hard_CPU_seconds": 2, "PES_evaluations": 0, "new_random_draws": 0,
                    "one_attempt_no_automatic_resume": True}
    check_hashes(guards); write_new_json(output/"registration.json", registration)
    return output/"registration.json"


def load_registration(path):
    path = Path(path).resolve(); reg = read_json(path)
    require(reg.get("schema") == REG_SCHEMA and Path(reg["output"]).resolve() == path.parent
            and reg["preparation_hard_CPU_seconds"] == 2
            and 0 <= reg["preparation_CPU_seconds_before_registration_write"] < 2,
            "invalid preparation identity or CPU accounting")
    token = reg["preparation_process_token"]
    require(isinstance(token, str) and uuid.UUID(hex=token).hex == token, "invalid preparation invocation token")
    plan, guards = load_plan(reg["plan_path"])
    require(sha(reg["plan_path"]) == reg["plan_sha256"] and reg["source_sha256"] == guards
            and reg["environment"] == environment_identity() and reg["output"] == plan["output"],
            "registered plan/source/runtime/output differs")
    return reg, plan, guards


def claim_worker(registration, reg):
    output = Path(reg["output"]); owner = read_json(output/"RUN_ONCE.json")
    token = os.environ.get("PYRPMD_HCH4_EFFICIENCY_SUPERVISOR_TOKEN")
    require(ROLE == "_worker" and RESOURCE_ROLE and owner["registration_sha256"] == sha(registration)
            and owner["supervisor_PID"] == os.getppid() and owner["supervisor_process_token"] == token
            and token not in (None, PROCESS_TOKEN), "private worker ownership/admission differs")
    require(owner["deadline_monotonic"] == float(os.environ["PYRPMD_HCH4_EFFICIENCY_DEADLINE"])
            and time.monotonic() < owner["deadline_monotonic"]-2,
            "worker deadline identity/admission differs")
    write_new_json(output/"WORKER_ONCE.json", {"created_utc": now(), "registration_sha256": sha(registration),
                                               "PID": os.getpid(), "process_token": PROCESS_TOKEN,
                                               "supervisor_PID": os.getppid(), "supervisor_process_token": token})
    return owner


@contextmanager
def stop_flags():
    flags = {"signal": None}; previous = {}
    def handler(number, frame):
        flags["signal"] = number
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, handler)
    try:
        yield flags
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def stop_and_reap(child, *, grace=.5):
    """Only this supervisor's child session; a failed drain is never success."""
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return child.wait(timeout=grace)


def supervise(registration, *, command=None, enforce_resources=True, wall_seconds=120):
    require(0 < wall_seconds <= 120, "wall ceiling cannot exceed the reviewed120 seconds")
    if enforce_resources:
        require(ROLE == "run" and RESOURCE_ROLE, "supervision requires its dedicated capped interpreter")
    reg, plan, guards = load_registration(registration); output = Path(reg["output"])
    require(reg["preparation_process_token"] != PROCESS_TOKEN,
            "preparation and supervisor require distinct invocations, regardless of namespace PIDs")
    if enforce_resources:
        os.sched_setaffinity(0, {plan["cpu_affinity"]})
    deadline = ENTRY_MONOTONIC+wall_seconds
    require(time.monotonic() < deadline-2, "supervisor preflight exhausted the wall admission")
    owner = {"schema": "pyrpmd.hch4-efficiency-one-attempt.v1", "created_utc": now(),
             "registration_sha256": sha(registration), "supervisor_PID": os.getpid(),
             "supervisor_process_token": PROCESS_TOKEN, "deadline_monotonic": deadline,
             "whole_process_CPU_caps_seconds": {"prepare": 2, "supervisor": 3, "worker": 55},
             "combined_soft_CPU_seconds": 55, "maximum_PES_geometries": 4000}
    write_new_json(output/"RUN_ONCE.json", owner)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **{name: "1" for name in THREADS})
    environment.update(PYRPMD_HCH4_EFFICIENCY_WORKER_CAP="55", PYRPMD_HCH4_EFFICIENCY_DEADLINE=str(deadline),
                       PYRPMD_HCH4_EFFICIENCY_SUPERVISOR_TOKEN=PROCESS_TOKEN)
    command = command or [sys.executable, str(SELF), "_worker", "--registration", str(Path(registration).resolve())]
    child, code, failure, stop, drained = None, None, None, None, False
    started = time.monotonic()
    import resource
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with stop_flags() as flags:
        try:
            with (output/"worker.log").open("x") as log:
                require(time.monotonic() < deadline-2 and flags["signal"] is None, "admission stopped before Popen")
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                         env=environment, start_new_session=True)
                # Child ownership is assigned before this limit/journal operation.
                if enforce_resources:
                    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
                write_new_json(output/"child-started.json", {"PID": child.pid, "supervisor_process_token": PROCESS_TOKEN,
                                                             "started_utc": now(), "command": command})
                while child.poll() is None:
                    if flags["signal"] is not None or time.monotonic() >= deadline-2:
                        stop = "signal" if flags["signal"] is not None else "wall_admission"; break
                    try:
                        code = child.wait(timeout=min(.05, max(.001, deadline-2-time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                if stop is None:
                    code = child.wait()
        except BaseException as error:
            failure = {"error_type": type(error).__name__, "message": str(error)}
        finally:
            if child is not None:
                try:
                    code = stop_and_reap(child); drained = True
                except BaseException as error:
                    failure = {"error_type": type(error).__name__, "message": str(error), "drain_unconfirmed": True}
    if child is None:
        drained = True
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    unchanged = True
    try:
        check_hashes(guards)
    except BaseException as error:
        unchanged = False; failure = {"error_type": type(error).__name__, "message": str(error)}
    report_path = output/"report.json"
    try:
        report = read_json(report_path) if report_path.exists() else None
        if report is not None:
            require(set(report) == set(OUTPUT_FIELDS), "worker report schema differs")
            for name, digest in report["ledger_sha256"].items():
                require(Path(name).name == name and sha(output/name) == digest, "worker raw ledger identity differs")
            if report["proposal_sha256"] is not None:
                require(sha(output/"frozen-proposal.json") == report["proposal_sha256"], "worker proposal identity differs")
    except BaseException as error:
        report = None; failure = {"error_type": type(error).__name__, "message": str(error), "report_validation_failed": True}
    complete = bool(code == 0 and drained and failure is None and stop is None and unchanged
                    and report is not None and report["status"] == "complete_excluded_efficiency_pilot"
                    and report["source_guards_unchanged"] is True and report["registration_sha256"] == sha(registration)
                    and report["main_draws_complete"] == 3072 and report["control_draws_complete"] == 128
                    and report["calibration_complete"] is True and report["attempted_PES_geometries"] <= 4000)
    outcome = {"schema": "pyrpmd.hch4-efficiency-supervisor.v1", "created_utc": now(),
               "registration_sha256": sha(registration), "supervisor_process_token": PROCESS_TOKEN,
               "worker_PID": None if child is None else child.pid, "worker_exit_code": code,
               "all_owned_workers_drained": drained, "stop": stop, "failure": failure,
               "source_guards_unchanged": unchanged, "complete_excluded_efficiency_pilot": complete,
               "wall_seconds_since_entry": time.monotonic()-ENTRY_MONOTONIC,
               "wall_seconds_since_preflight": time.monotonic()-started,
               "supervisor_CPU_seconds_before_receipt": time.process_time(),
               "reaped_child_CPU_seconds": after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
               "fixed_whole_process_CPU_caps_sum_seconds": 60,
               "report_sha256": sha(report_path) if report_path.exists() else None,
               "automatic_retry": False, "automatic_extension": False}
    write_new_json(output/"supervisor.json", outcome)
    return outcome


def worker(registration):
    reg, plan, guards = load_registration(registration); output = Path(reg["output"])
    owner = claim_worker(registration, reg)
    require(all(os.environ.get(name) == "1" for name in THREADS)
            and os.sched_getaffinity(0) == {plan["cpu_affinity"]}, "worker requires its one-CPU/one-thread allocation")
    np = science(); import scipy
    require(np.__version__ == plan["environment"]["numpy"] and scipy.__version__ == plan["environment"]["scipy"],
            "actual scientific runtime differs from the adopted metadata")
    old = legacy(); model = old.fixed_model()
    imported = {}
    for name in ("pyrpmd.constants", "pyrpmd.potentials.base", "pyrpmd.potentials.cabi", "benchmarks.hch4.pes"):
        module = importlib.import_module(name); path = Path(module.__file__).resolve()
        require(str(path) in guards and sha(path) == guards[str(path)], "actual imported scientific source differs")
        imported[name] = {"path": str(path), "sha256": sha(path)}
    import resource
    require(resource.getrlimit(resource.RLIMIT_CPU) == (55, 55), "worker hard CPU cap was not installed before imports")
    write_new_json(output/"runtime.json", {"environment": environment_identity(), "actual_imports": imported,
                                           "affinity": sorted(os.sched_getaffinity(0)),
                                           "thread_environment": {name: os.environ[name] for name in THREADS},
                                           "worker_process_token": PROCESS_TOKEN, "worker_PID": os.getpid(),
                                           "actual_CPU_limit": list(resource.getrlimit(resource.RLIMIT_CPU)),
                                           "deadline_monotonic": owner["deadline_monotonic"]})
    require(time.process_time()+5 < 55 and time.monotonic() < owner["deadline_monotonic"]-2,
            "CPU/wall admission expired before external PES loading")
    from benchmarks.hch4.pes import CBE2008Potential
    potential = CBE2008Potential(plan["library"]["path"])
    require(potential.metadata["library_sha256"] == plan["library"]["sha256"], "actual loaded model differs")
    def energy(q):
        value = potential.evaluate(q[None])
        require(value.status[0] == 0, f"PES returned status {value.status[0]}: {value.messages[0]}")
        return float(value.energy[0])
    report = run_stages(output, model, energy, registration_sha=sha(registration), guards=guards,
                        deadline=owner["deadline_monotonic"])
    return 0 if report["status"] == "complete_excluded_efficiency_pilot" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("preflight", "prepare"):
        part = subparsers.add_parser(name); part.add_argument("--plan", type=Path, required=True)
        if name == "prepare":
            part.add_argument("--output", type=Path, required=True)
    for name in ("run", "_worker"):
        part = subparsers.add_parser(name); part.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "preflight":
        plan, guards = load_plan(args.plan)
        print(json.dumps({"status": "read_only_preflight_passed", "source_guards": len(guards), "PES_evaluations": 0}))
        return 0
    if args.action == "prepare":
        print(prepare(args.plan, args.output)); return 0
    if args.action == "run":
        return 0 if supervise(args.registration)["complete_excluded_efficiency_pilot"] else 2
    return worker(args.registration)


if __name__ == "__main__":
    raise SystemExit(main())
