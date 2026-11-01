#!/usr/bin/env python3
"""Reproducible enhanced-sampling / ML validation with explicit reference scope.

The Muller model is from Invernizzi & Parrinello, JCTC 18, 3988 (2022),
doi:10.1021/acs.jctc.2c00152. Local runs do not reproduce its 200M-step budget
or PLUMED's bandwidth adaptation / kernel compression. All trajectory kernels
below call the public PyRPMD Dynamics and bias APIs, except the separately
labelled overdamped kinetics reference experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp
from scipy.stats import t as student_t

from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.opes import OPESBias
from pyrpmd.potentials import PESResult, HarmonicPotential


AUTHOR_COMMIT = "47b5d4c92f87920dd7750900c751a1e37011ed6a"
AUTHOR_STARTS = {
    57859: (-0.693111, 1.40842), 28661: (-0.682513, 1.34308),
    38795: (-0.639895, 1.33947), 53730: (-0.43134, 1.54469),
}
DEFAULT_SEEDS = tuple(AUTHOR_STARTS)
DEFAULT_OUTPUT = Path("docs/validation/tier2-literature-2026-11-01/sampling")
RECTANGLE = ((-1.3, 1.0), (-0.2, 2.0))
COEFFICIENTS = np.array([
    [-200., -1., 0., -10., 1., 0.],
    [-100., -1., 0., -10., 0., .5],
    [-170., -6.5, 11., -6.5, -.5, 1.5],
    [15., .7, .6, .7, -1., 1.],
])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def muller_xy(x, y, *, gradient=False, walls=False):
    """Author potential in natural units; analytic derivatives, no copied code."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    energy = np.full(x.shape, 146.7)
    gx, gy = np.zeros_like(x), np.zeros_like(y)
    for amplitude, a, b, c, center_x, center_y in COEFFICIENTS:
        dx, dy = x - center_x, y - center_y
        term = amplitude * np.exp(a*dx*dx + b*dx*dy + c*dy*dy)
        energy += term
        if gradient:
            gx += term*(2*a*dx + b*dy)
            gy += term*(b*dx + 2*c*dy)
    energy *= .15
    gx *= .15
    gy *= .15
    if walls:
        outside = np.minimum(x + 1.3, 0) + np.maximum(x - 1., 0)
        energy += 1000*outside**2
        gx += 2000*outside
    return (energy, gx, gy) if gradient else energy


class MullerPotential:
    metadata = {"provider": "literature_muller", "batch_independent": True}

    def evaluate(self, positions):
        q = np.asarray(positions)
        energy, gx, gy = muller_xy(q[:, 0, 0], q[:, 0, 1], gradient=True, walls=True)
        # Independent harmonic z embeds the 2D problem in the [P,N,3] API.
        # It factors out of all x/y statistics and does not change their forces.
        energy = energy + .5*q[:, 0, 2]**2
        forces = -q.copy()
        forces[:, 0, 0], forces[:, 0, 1] = -gx, -gy
        return PESResult(energy, forces)


class SplineBias:
    """Numerically audited acceleration of the actual PyRPMD bias object.

    Between deposits V is immutable. Energy and force use one cubic spline;
    every refresh compares its midpoint values/derivatives with the direct API.
    An out-of-grid point always falls back to the direct API.
    """
    def __init__(self, source, axis, *, grid_points=2049):
        self.source, self.axis = source, axis
        self.grid = np.linspace(-2., 2., grid_points) if axis == 0 else np.linspace(-.8, 2.8, grid_points)
        self.max_energy_error = self.max_force_error = 0.0
        self.fallbacks = 0
        self.refresh()

    def refresh(self):
        values = np.asarray(self.source.physical_energy(self.grid[:, None]))
        self.spline = CubicSpline(self.grid, values)
        test = (self.grid[:-1:17] + self.grid[1::17]) / 2
        exact_energy, exact_grad = self.source.physical_energy_gradient(test[:, None])
        self.max_energy_error = max(self.max_energy_error, float(np.max(np.abs(self.spline(test)-exact_energy))))
        self.max_force_error = max(self.max_force_error, float(np.max(np.abs(self.spline(test, 1)-exact_grad[:, 0]))))

    def energy_force(self, q):
        if q.shape != (1, 1, 3):
            raise ValueError("literature spline acceleration is only for P=N=1")
        value = float(q[0, 0, self.axis])
        if value < self.grid[0] or value > self.grid[-1]:
            self.fallbacks += 1
            return self.source.energy_force(q)
        force = np.zeros_like(q)
        force[0, 0, self.axis] = -float(self.spline(value, 1))
        return float(self.spline(value)), force


def reference(output):
    output = Path(output)
    x = np.linspace(*RECTANGLE[0], 1151)
    y = np.linspace(*RECTANGLE[1], 1101)
    density = np.exp(-muller_xy(x[:, None], y[None, :]))
    px, py = simpson(density, x=y, axis=1), simpson(density, x=x, axis=0)
    fx, fy = -np.log(px), -np.log(py)
    fx -= fx.min(); fy -= fy.min()
    def integral(a, b, lower, upper, tol):
        return quad(lambda xx: quad(lambda yy: math.exp(-float(muller_xy(xx, yy))),
                    lower(xx), upper(xx), epsabs=tol, epsrel=tol)[0],
                    a, b, epsabs=tol, epsrel=tol, points=[0.] if a < 0 < b else None)[0]
    computed = []
    for tol in (1e-7, 2e-10):
        left = integral(-1.3, 0, lambda x: -.2, lambda x: 2., tol)
        right = integral(0, 1., lambda x: -.2, lambda x: 2., tol)
        # A: y>x+1.3; B is the complementary rectangle, with an exact kink split.
        above = integral(-1.3, .7, lambda x: x+1.3, lambda x: 2., tol)
        below = left+right-above
        computed.append({"epsabs_epsrel": tol, "delta_f_x": -math.log(right/left),
                         "delta_f_oblique": -math.log(below/above), "partition_rectangle": left+right})
    checks = []
    rng = np.random.default_rng(12)
    points = rng.uniform([-1.2, -.1], [.9, 1.9], size=(30, 2))
    for xx, yy in points:
        _, gx, gy = muller_xy(xx, yy, gradient=True, walls=True)
        h = 1e-6
        numeric = [(muller_xy(xx+h, yy, walls=True)-muller_xy(xx-h, yy, walls=True))/(2*h),
                   (muller_xy(xx, yy+h, walls=True)-muller_xy(xx, yy-h, walls=True))/(2*h)]
        checks.append(float(np.max(np.abs(np.array([gx, gy])-numeric))))
    result = {"reference_type": "computed adaptive quadrature of the authors' published model",
              "published_delta_f_x_printed": "5.712(6)",
              "published_delta_f_oblique_printed": "5.690(5)",
              "published_parentheses_interpretation": "Retained literally; no statistical uncertainty is inferred from the printed notation.",
              "published_source": "https://arxiv.org/pdf/2201.09950#page=17, Supporting Information Sec. S3",
              "author_input_commit": AUTHOR_COMMIT, "rectangle": RECTANGLE,
              "quad_refinement": computed, "delta_f_x": computed[-1]["delta_f_x"],
              "delta_f_oblique": computed[-1]["delta_f_oblique"],
              "maximum_force_finite_difference_error": max(checks)}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"reference.npz", x=x, y=y, fx=fx, fy=fy, px=px, py=py)
    write_json(output/"reference.json", result)
    write_csv(output/"reference_x.csv", ({"x": xx, "F_kBT": ff, "density_unnormalized": pp}
                                         for xx, ff, pp in zip(x, fx, px)))
    write_csv(output/"reference_y.csv", ({"y": yy, "F_kBT": ff, "density_unnormalized": pp}
                                         for yy, ff, pp in zip(y, fy, py)))
    return result


def js_divergence(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p/p.sum(), q/q.sum()
    m = (p+q)/2
    return float(.5*np.sum(p[p > 0]*np.log(p[p > 0]/m[p > 0]))
                 + .5*np.sum(q[q > 0]*np.log(q[q > 0]/m[q > 0])))


def transitions(xy):
    """Core-to-core switches, retaining last basin in the transition strip."""
    last, count = None, 0
    for x, y in xy:
        score = y-x-1.3
        current = 1 if score > .2 else 0 if score < -.2 else None
        if current is not None:
            count += last is not None and current != last
            last = current
    return count


def aligned_rmse(estimate, exact, weights, valid):
    if not np.any(valid):
        return None
    diff = np.asarray(estimate)[valid] - np.asarray(exact)[valid]
    weights = np.asarray(weights)[valid]
    weights = weights/weights.sum()
    diff -= weights@diff
    return float(np.sqrt(weights@(diff*diff)))


def opes_run(output, *, seed, axis, mode, adaptive, steps, freeze, dt=.005, sigma=.1, pace=500,
             sample_stride=10, grid_points=2049, direct=False):
    output = Path(output)
    label = f"{mode}_{'adaptive' if adaptive else 'fixed'}_{'xy'[axis]}_dt{dt:g}_s{sigma:g}_seed{seed}"
    destination = output/"opes_v2"/label
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"metrics.json").is_file():
        saved = json.loads((destination/"metrics.json").read_text())
        requested = {"steps_adapting": steps, "steps_frozen": freeze, "dt": dt,
                     "pace": pace, "initial_sigma": sigma, "sample_stride": sample_stride,
                     "spline_grid_points": None if direct else grid_points}
        if any(saved.get(key) != value for key, value in requested.items()):
            raise ValueError(f"saved benchmark settings differ at {destination}; use a distinct output directory")
        if mode == "metad" and saved.get("bias_state_schema_version", 1) != 2:
            raise ValueError(f"legacy METAD trajectory at {destination}; preserve it separately before rerunning")
        return saved
    bias = OPESBias(Cartesian(0, axis=axis), [sigma], 1., barrier=20., bias_factor=20.,
                    mode=mode, adaptive_sigma=adaptive, sigma_min=[.015] if adaptive else None)
    engine_bias = bias if direct else SplineBias(bias, axis, grid_points=grid_points)
    dyn = Dynamics(MullerPotential(), [1.], 1., 1, dt, seed=seed, integrator="normal-mode-bab",
                   native=True, loop_backend="auto")
    state = State(np.array([[list(AUTHOR_STARTS.get(seed, AUTHOR_STARTS[57859]))+[0.]]]), dyn.sample_momenta())
    snapshots, updates, records = [], [], []
    started = time.perf_counter()
    ref = np.load(output/"reference.npz")
    coords, free_energy = (ref["x"], ref["fx"]) if axis == 0 else (ref["y"], ref["fy"])
    for index in range(1, steps+freeze+1):
        dyn.sample_step(state, engine_bias, gamma_centroid=10.)
        if index <= steps and index % pace == 0:
            updates.append(bias.update([state.q[0, 0, axis]], step=index, time=index*dt))
            if not direct:
                engine_bias.refresh()
            if index % (20*pace) == 0 or index == steps:
                estimate = -bias.log_probability(coords[:, None]) * (20. if mode == "explore" else 1.)
                snapshots.append((index, estimate-estimate.min()))
        if index % sample_stride == 0:
            records.append([index, index*dt, state.q[0, 0, 0], state.q[0, 0, 1],
                            float(bias.physical_energy([state.q[0, 0, axis]]))])
    history = np.asarray(records)
    frozen = history[history[:, 0] > steps+min(10000, freeze//5)]
    inside = ((frozen[:, 2] >= -1.3) & (frozen[:, 2] <= 1.)
              & (frozen[:, 3] >= -.2) & (frozen[:, 3] <= 2.))
    samples = frozen[inside]
    weights = np.exp(samples[:, 4]-samples[:, 4].max())
    left, right = weights[samples[:, 2] < 0].sum(), weights[samples[:, 2] > 0].sum()
    delta_f = -float(np.log(right/left)) if min(left, right) > 0 else None
    above = samples[:, 3] > samples[:, 2]+1.3
    wa, wb = weights[above].sum(), weights[~above].sum()
    delta_oblique = -float(np.log(wb/wa)) if min(wa, wb) > 0 else None
    edges = np.linspace(*RECTANGLE[axis], 81)
    centers = (edges[1:]+edges[:-1])/2
    exact = np.interp(centers, coords, free_energy)
    exact_prob = np.exp(-exact)
    weighted, _ = np.histogram(samples[:, 2+axis], bins=edges, weights=weights)
    unweighted, _ = np.histogram(samples[:, 2+axis], bins=edges)
    estimated = np.full_like(weighted, np.nan)
    np.log(weighted, out=estimated, where=weighted > 0)
    estimated = -estimated
    direct_fes = -bias.log_probability(centers[:, None]) * (20. if mode == "explore" else 1.)
    bias_values = bias.physical_energy(centers[:, None])
    ideal_wt = np.exp(-exact/20.)
    exact_frozen = np.exp(-exact-bias_values+np.min(bias_values))
    mask = exact-exact.min() <= 10.
    metrics = {"label": label, "mode": mode, "adaptive_sigma": adaptive, "cv": "xy"[axis],
               "seed": seed, "steps_adapting": steps, "steps_frozen": freeze, "dt": dt,
               "pace": pace, "initial_sigma": sigma, "bias_factor": 20., "barrier_kBT": 20.,
               "friction": 10., "sample_stride": sample_stride,
               "initial_empty_history_bias": bias.state_dict().get("initial_bias", 0.),
               "bias_state_schema_version": bias.state_dict()["schema_version"],
               "elapsed_seconds": time.perf_counter()-started,
               "kernels": len(bias), "kernel_weight_ESS": bias.effective_sample_size,
               "final_sigma": bias._sigmas[-1, 0] if adaptive else sigma,
               "transitions_adapting": transitions(history[history[:, 0] <= steps, 2:4]),
               "transitions_frozen": transitions(frozen[:, 2:4]),
               "frozen_samples": len(frozen), "fraction_inside_paper_rectangle": float(inside.mean()),
               "reweighting_Kish_ESS": float(weights.sum()**2/(weights@weights)),
               "reweighting_ESS_scope": "weight-only; independent-seed spread is reported separately",
               "frozen_delta_F_x_kBT": delta_f, "frozen_delta_F_oblique_kBT": delta_oblique,
               "frozen_FES_RMSE_kBT": aligned_rmse(estimated, exact, np.ones_like(exact), mask & (weighted > 0)),
               "frozen_FES_Boltzmann_weighted_RMSE_kBT": aligned_rmse(estimated, exact, exact_prob, mask & (weighted > 0)),
               "unvisited_reference_probability": float(exact_prob[weighted == 0].sum()/exact_prob.sum()),
               "direct_FES_RMSE_kBT": aligned_rmse(direct_fes, exact, np.ones_like(exact), mask),
               "JS_unbiased_reweighted": js_divergence(weighted, exact_prob),
               "JS_sample_to_exact_frozen_ensemble": js_divergence(unweighted, exact_frozen),
               "JS_sample_to_ideal_WT": js_divergence(unweighted, ideal_wt),
               "JS_frozen_exact_to_ideal_WT": js_divergence(exact_frozen, ideal_wt),
               "spline_grid_points": None if direct else grid_points,
               "spline_max_energy_error_kBT": 0. if direct else engine_bias.max_energy_error,
               "spline_max_force_error": 0. if direct else engine_bias.max_force_error,
               "direct_fallback_force_evaluations": 0 if direct else engine_bias.fallbacks,
               "full_paper_reproduction": False}
    np.savez_compressed(destination/"trajectory.npz", history=history,
                        columns=np.array(["step", "time", "x", "y", "physical_bias"]),
                        centers=centers, exact_fes=exact, reweighted_counts=weighted,
                        unweighted_counts=unweighted, direct_fes=direct_fes, bias_values=bias_values,
                        snapshots_steps=np.array([entry[0] for entry in snapshots]),
                        snapshots_fes=np.array([entry[1] for entry in snapshots]),
                        snapshot_grid=coords)
    write_json(destination/"metrics.json", metrics)
    write_json(destination/"bias_state.json", bias.state_dict())
    write_csv(destination/"updates.csv", updates)
    print(json.dumps(metrics, allow_nan=False), flush=True)
    return metrics


def summarize_opes(output):
    output = Path(output)
    rows = [json.loads(path.read_text()) for path in sorted((output/"opes_v2").glob("*/metrics.json"))]
    write_csv(output/"opes_metrics.csv", rows)
    groups = {}
    for row in rows:
        key = (row["mode"], row["adaptive_sigma"], row["cv"], row["dt"], row["initial_sigma"])
        groups.setdefault(key, []).append(row)
    grouped = []
    fields = ["frozen_delta_F_x_kBT", "frozen_delta_F_oblique_kBT", "frozen_FES_RMSE_kBT",
              "direct_FES_RMSE_kBT", "JS_unbiased_reweighted", "JS_sample_to_exact_frozen_ensemble",
              "JS_sample_to_ideal_WT", "transitions_adapting", "transitions_frozen", "reweighting_Kish_ESS"]
    for key, group in groups.items():
        record = dict(zip(["mode", "adaptive_sigma", "cv", "dt", "sigma"], key))
        record["independent_seeds"] = len(group)
        for field in fields:
            values = [row[field] for row in group if row[field] is not None]
            record[field+"_valid_seeds"] = len(values)
            record[field+"_mean"] = float(np.mean(values)) if values else None
            record[field+"_std_between_seeds"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            record[field+"_95CI_halfwidth"] = (float(student_t.ppf(.975, len(values)-1)
                  *np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None)
        grouped.append(record)
    write_json(output/"opes_summary.json", grouped)
    return grouped


def mean_interval(values):
    values = np.asarray(values, float)
    mean = float(values.mean())
    error = float(student_t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else None
    return {"mean": mean, "95CI_halfwidth": error, "samples": len(values)}


def replica_exchange(output, *, seeds=DEFAULT_SEEDS, steps=40000):
    """Measured thermal ensembles; analytical finite-P harmonic references."""
    from pyrpmd.replica_exchange import ReplicaExchange
    output = Path(output)/"replica_exchange"
    output.mkdir(parents=True, exist_ok=True)
    records, datasets = [], {}
    for scheme in ("temperature", "hamiltonian"):
        betas = [1., 2., 4.] if scheme == "temperature" else [2., 2., 2.]
        constants = [1., 1., 1.] if scheme == "temperature" else [.5, 1., 2.]
        for seed in seeds:
            dynamics = [Dynamics(HarmonicPotential(k=k), [1.], beta, 4, .02,
                         seed=np.random.SeedSequence([seed, index]), native=True,
                         integrator="normal-mode-bab") for index, (beta, k) in enumerate(zip(betas, constants))]
            states = [State(np.zeros((4, 1, 3)), dyn.sample_momenta()) for dyn in dynamics]
            exchange = ReplicaExchange(dynamics, states, seed=seed+100)
            samples, labels = [], []
            for iteration in range(steps//10):
                exchange.advance(10, gamma_centroid=1.)
                if iteration >= steps//50:  # Discard the initial 20%.
                    samples.append([[np.mean(s.q, axis=0)[0]**2,
                                     np.mean(s.q*s.q, axis=0)[0],
                                     np.mean(s.p*s.p, axis=0)[0]] for s in exchange.states])
                    labels.append(exchange.labels.copy())
            values = np.array(samples)  # [frames,slot,observable,xyz]
            datasets[f"{scheme}_{seed}_moments"] = values
            datasets[f"{scheme}_{seed}_labels"] = labels
            for slot, (beta, k) in enumerate(zip(betas, constants)):
                omega = 2*4/beta*np.sin(np.pi*np.arange(4)/4)
                expected = [1/(beta*k), float(np.sum(1/(omega*omega+k))/beta), 4/beta]
                for index, observable in enumerate(("centroid_variance", "bead_variance", "momentum_variance")):
                    measured = float(values[:, slot, index].mean())
                    block_means = np.array_split(values[:, slot, index].mean(axis=1), 20)
                    interval = mean_interval([block.mean() for block in block_means])
                    records.append({"scheme": scheme, "seed": seed, "slot": slot, "beta": beta,
                                    "k": k, "observable": observable, "measured": measured,
                                    "reference": expected[index], "relative_error": measured/expected[index]-1,
                                    "block_95CI_halfwidth": interval["95CI_halfwidth"],
                                    "exchange_attempts": int(exchange.attempts.sum()),
                                    "acceptance": float(exchange.accepts.sum()/exchange.attempts.sum()),
                                    "dt": .02, "beads": 4, "steps": steps})
            print(f"replica_exchange {scheme} seed {seed}: complete", flush=True)
    write_csv(output/"moments.csv", records)
    np.savez_compressed(output/"raw_moments.npz", **datasets)
    groups = []
    for scheme in ("temperature", "hamiltonian"):
        for slot in range(3):
            for observable in ("centroid_variance", "bead_variance", "momentum_variance"):
                rows = [r for r in records if (r["scheme"], r["slot"], r["observable"]) == (scheme, slot, observable)]
                groups.append({"scheme": scheme, "slot": slot, "observable": observable,
                               "reference": rows[0]["reference"], **mean_interval([r["measured"] for r in rows])})
    write_json(output/"summary.json", groups)
    return groups


def shared_metadynamics(output, *, seeds=DEFAULT_SEEDS, steps=30000, freeze=30000):
    """Shared WT history, then an independently sampled stationary bias.

    Adaptive deposition has no stationary detailed-balance claim. The frozen
    phase checks exp[-beta*(U+V)] and the well-tempered target separately.
    """
    from pyrpmd.shared_metadynamics import SharedMetadynamicsBias
    output = Path(output)/"shared_metadynamics"
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for seed in seeds:
        bias = SharedMetadynamicsBias(Cartesian(0), [.15], .05, 1., walkers=4,
                                      mode="well_tempered", bias_factor=5.)
        spline = SplineBias(bias, 0, grid_points=4097)
        # The harmonic WT distribution has sd=sqrt(gamma/k); extend the grid.
        spline.grid = np.linspace(-10., 10., 4097)
        spline.refresh()
        dynamics = [Dynamics(HarmonicPotential(k=1.), [1.], 1., 1, .02,
                    seed=np.random.SeedSequence([seed, walker]), integrator="normal-mode-bab",
                    native=True) for walker in range(4)]
        states = [State(np.zeros((1, 1, 3)), dyn.sample_momenta()) for dyn in dynamics]
        records, deposit_error, reorder_error = [], 0., 0.
        for step in range(1, steps+freeze+1):
            for dyn, state in zip(dynamics, states):
                dyn.sample_step(state, spline, gamma_centroid=1.)
            if step <= steps and step % 100 == 0:
                centers = np.array([[state.q[0, 0, 0]] for state in states])
                before = bias.physical_energy(centers)
                expected = .05*np.exp(-before/4.)
                if step == steps:
                    reverse = SharedMetadynamicsBias(Cartesian(0), [.15], .05, 1., walkers=4,
                                mode="well_tempered", bias_factor=5., hills=bias.hills)
                    reverse.deposit_walkers(centers[::-1], step=step, time=step*.02)
                hills = bias.deposit_walkers(centers, step=step, time=step*.02)
                deposit_error = max(deposit_error, float(np.max(np.abs(expected-[h["height"] for h in hills]))))
                if step == steps:
                    grid = np.linspace(-7., 7., 201)[:, None]
                    reorder_error = float(np.max(np.abs(bias.physical_energy(grid)-reverse.physical_energy(grid))))
                spline.refresh()
            if step % 5 == 0:
                x = np.array([state.q[0, 0, 0] for state in states])
                records.append([step, *x, *bias.physical_energy(x[:, None])])
        history = np.asarray(records)
        frozen = history[history[:, 0] > steps+freeze//5]
        x, potential = frozen[:, 1:5], frozen[:, 5:9]
        edges = np.linspace(-7., 7., 101)
        grid = (edges[1:]+edges[:-1])/2
        vb = bias.physical_energy(grid[:, None])
        target = np.exp(-.5*grid**2/5.)
        exact = np.exp(-.5*grid**2-vb+vb.min())
        counts = np.histogram(x.ravel(), edges)[0]
        weights = np.exp(potential-potential.max())
        reweighted = np.histogram(x.ravel(), edges, weights=weights.ravel())[0]
        unbiased = np.exp(-.5*grid**2)
        mean = float(np.sum(weights*x)/np.sum(weights))
        variance = float(np.sum(weights*(x-mean)**2)/np.sum(weights))
        record = {"seed": seed, "walkers": 4, "steps_adapting_per_walker": steps,
                  "steps_frozen_per_walker": freeze, "hill_count": bias.hill_count,
                  "JS_sample_to_exact_frozen": js_divergence(counts, exact),
                  "JS_sample_to_WT": js_divergence(counts, target),
                  "JS_exact_frozen_to_WT": js_divergence(exact, target),
                  "JS_reweighted_to_unbiased": js_divergence(reweighted, unbiased),
                  "reweighted_mean": mean, "reweighted_variance": variance,
                  "exact_unbiased_variance": 1., "Kish_ESS": float(weights.sum()**2/np.sum(weights**2)),
                  "maximum_simultaneous_height_error": deposit_error,
                  "final_permuted_walker_bias_difference": reorder_error,
                  "spline_energy_error": spline.max_energy_error,
                  "spline_force_error": spline.max_force_error,
                  "adaptive_detailed_balance_claim": False}
        metrics.append(record)
        np.savez_compressed(output/f"seed_{seed}.npz", history=history, grid=grid,
                            bias=vb, histogram=counts, weighted_histogram=reweighted)
        write_json(output/f"seed_{seed}.json", record)
        print(json.dumps(record), flush=True)
    write_csv(output/"metrics.csv", metrics)
    write_json(output/"summary.json", {key: mean_interval([r[key] for r in metrics]) for key in (
        "JS_sample_to_exact_frozen", "JS_sample_to_WT", "JS_reweighted_to_unbiased", "reweighted_variance")})
    return metrics


def kinetics(output, *, replicas=256, dt=.0005, seed=8701):
    """Actual overdamped SDE + strictly transition-free static flooding bias.

    This tests the clock/rate analysis APIs, not native infrequent MetaD or an
    OPES-flooding driver. Euler-Maruyama dynamics is independently implemented
    here and checked against the exact reflecting/absorbing MFPT equation.
    """
    from array import array
    from pyrpmd.biased_kinetics import rescale_biased_time, estimate_biased_rate
    output = Path(output)/"kinetics"
    output.mkdir(parents=True, exist_ok=True)
    left, target, initial, barrier, width, amplitude = -1.8, .8, -1., 6., .65, 2.
    def potential(x):
        return barrier*(np.asarray(x)**2-1)**2
    def flood(x):
        z = (np.asarray(x)+1)/width
        core = np.maximum(1-z*z, 0.)
        return amplitude*core**3, -6*amplitude*z*core**2/width
    exact = quad(lambda y: math.exp(float(potential(y))) * quad(
                lambda z: math.exp(-float(potential(z))), left, y,
                epsabs=1e-10, epsrel=1e-10)[0], initial, target, epsabs=1e-8, epsrel=1e-10)[0]
    all_rows, summaries = [], []
    for use_bias in (False, True):
        rng = np.random.default_rng(seed+int(use_bias))
        q = np.full(replicas, initial)
        histories = [array("d") for _ in range(replicas)] if use_bias else None
        completed = np.zeros(replicas, bool)
        times = np.zeros(replicas)
        scaled = np.zeros(replicas)
        traces = []
        ts_bias_max = ts_gradient_max = 0.
        limit = int(2000/dt)
        for step in range(limit):
            active = np.flatnonzero(~completed)
            x = q[active]
            vb, gb = flood(x) if use_bias else (np.zeros_like(x), np.zeros_like(x))
            audited = x >= -.35
            if audited.any():
                ts_bias_max = max(ts_bias_max, float(np.max(np.abs(vb[audited]))))
                ts_gradient_max = max(ts_gradient_max, float(np.max(np.abs(gb[audited]))))
            if use_bias:
                for index, value in zip(active, vb):
                    histories[index].append(float(value))
            force = -4*barrier*x*(x*x-1)-gb
            proposed = x + dt*force + np.sqrt(2*dt)*rng.normal(size=len(active))
            proposed = np.where(proposed < left, 2*left-proposed, proposed)
            q[active] = proposed
            ended = active[proposed >= target]
            for index in ended:
                completed[index] = True
                times[index] = (step+1)*dt
                if use_bias:
                    history = np.frombuffer(histories[index], dtype=np.float64)
                    scaled[index] = rescale_biased_time(np.arange(len(history)+1)*dt, history,
                        beta=1., transition_bias=0., transition_unbiased=True)["rescaled_time_au"]
                    histories[index] = array("d")
                else:
                    scaled[index] = rescale_biased_time([0., times[index]], [0.],
                        beta=1., transition_bias=0., transition_unbiased=True)["rescaled_time_au"]
            if step % 100 == 0:
                traces.append([step*dt, *np.where(completed[:4], np.nan, q[:4])])
            if completed.all():
                break
        if not completed.all():
            raise RuntimeError("extend the kinetics exposure limit; no censored path is silently discarded")
        estimate = estimate_biased_rate(scaled, bootstrap=499, seed=seed)
        interval = mean_interval(scaled)
        summary = {"static_flooding_bias": use_bias, "dt": dt, "replicas": replicas,
                   "exact_unbiased_MFPT": exact, "rescaled_MFPT": interval,
                   "raw_mean_elapsed": float(times.mean()), "mean_boost": float(np.mean(scaled/times)),
                   "relative_error_to_exact": interval["mean"]/exact-1,
                   "transition_region_x_min": -.35,
                   "maximum_bias_in_transition_region": ts_bias_max,
                   "maximum_bias_gradient_in_transition_region": ts_gradient_max,
                   "deposition_events": 0, "clock_sample_interval": dt,
                   "rate_analysis": estimate, "reference_scope": "computed exact overdamped MFPT; not a published rate"}
        summaries.append(summary)
        for index in range(replicas):
            all_rows.append({"biased": use_bias, "replica": index, "dt": dt,
                             "elapsed": times[index], "rescaled_time": scaled[index], "event": True})
        np.savez_compressed(output/f"traces_bias{int(use_bias)}_dt{dt:g}.npz", traces=np.asarray(traces))
        print(json.dumps(summary), flush=True)
    write_csv(output/f"lifetimes_dt{dt:g}.csv", all_rows)
    write_json(output/f"summary_dt{dt:g}.json", summaries)
    return summaries


def ml_lifecycle(output):
    """Supervised small TorchScript models and an actual monitored trajectory.

    This is an engineering benchmark on a held-out known analytic function,
    with no claim to reproduce a molecular ML-potential paper.
    """
    import torch
    from pyrpmd.checkpoint import load_checkpoint
    from pyrpmd.potentials import make_potential
    from pyrpmd.potentials.lifecycle import DeltaLearningPotential
    from pyrpmd.system import System
    from pyrpmd.workflows.materials import run_materials
    output = Path(output)/"ml_lifecycle"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    def truth(q):
        energy = np.sum(.3*q*q + .04*q**4 + .02*(1-np.cos(3*q)), axis=(1, 2))
        force = -.6*q-.16*q**3-.06*np.sin(3*q)
        return energy, force
    class Learned(torch.nn.Module):
        def __init__(self, coefficients, reference):
            super().__init__()
            self.register_buffer("coefficients", coefficients)
            self.reference = reference
        def forward(self, q):
            features = torch.stack([(q*q).sum(dim=(1, 2)), (q**4).sum(dim=(1, 2)),
                                    (q**6).sum(dim=(1, 2))], dim=1)
            return features @ self.coefficients + self.reference*(q*q).sum(dim=(1, 2))
    rng = np.random.default_rng(8319)
    train = rng.uniform(-.8, .8, size=(512, 1, 3))
    energy, force = truth(train)
    correction = energy-.3*np.sum(train*train, axis=(1, 2))
    noisy = correction+rng.normal(scale=2e-4, size=len(train))
    features = np.stack([np.sum(train**power, axis=(1, 2)) for power in (2, 4, 6)], axis=1)
    members, coefficients = [], []
    for index in range(4):
        bootstrap = np.random.default_rng(20+index).integers(0, len(train), len(train))
        x, target = torch.from_numpy(features[bootstrap]), torch.from_numpy(noisy[bootstrap])
        fitted = torch.linalg.solve(x.T@x+1e-7*torch.eye(3), x.T@target)
        coefficients.append(fitted.numpy().copy())
        full = torch.jit.trace(Learned(fitted, .3), torch.from_numpy(train[:2]))
        delta = torch.jit.trace(Learned(fitted, 0.), torch.from_numpy(train[:2]))
        path, delta_path = output/f"member_{index}.pt", output/f"delta_{index}.pt"
        full.save(str(path)); delta.save(str(delta_path))
        members.append({"provider": "torch", "model": str(path.resolve()), "device": "cpu", "virial": "none"})
    committee = make_potential({"provider": "committee", "members": members})
    inner = rng.uniform(-.8, .8, size=(400, 1, 3))
    outer = rng.uniform(-2., 2., size=(800, 1, 3))
    outer = outer[np.max(np.abs(outer), axis=(1, 2)) > 1.2][:400]
    heldout = np.concatenate([inner, outer])
    exact_e, exact_f = truth(heldout)
    prediction = committee.evaluate_uncertainty(heldout)
    force_error = np.linalg.norm(prediction.mean.forces-exact_f, axis=-1).max(axis=1)
    threshold = float(np.quantile(prediction.max_force_std[:400], .95))
    stats = {}
    for label, ids in (("interpolation", slice(0, 400)), ("extrapolation", slice(400, None))):
        stats[label] = {"energy_RMSE": float(np.sqrt(np.mean((prediction.mean.energy[ids]-exact_e[ids])**2))),
                       "force_RMSE": float(np.sqrt(np.mean((prediction.mean.forces[ids]-exact_f[ids])**2))),
                       "mean_force_disagreement": float(np.mean(prediction.max_force_std[ids])),
                       "acquired_fraction": float(np.mean(prediction.max_force_std[ids] >= threshold)),
                       "error_disagreement_Pearson": float(np.corrcoef(force_error[ids], prediction.max_force_std[ids])[0, 1])}
    config = {"task": "pimd", "beads": 4, "beta": 10., "dt_au": .01, "seed": 43,
              "potential": {"provider": "committee", "members": members},
              "dynamics": {"integrator": "normal-mode-bab", "loop_backend": "auto"},
              "sampling": {"equilibration_steps": 100, "production_steps": 300,
                           "sample_interval": 10, "replicas": 1, "thermostat": "pile-l", "gamma_centroid": 2.},
              "output": {"checkpoint_steps": 50, "trajectory_storage": "memory"}}
    system = System(("H",), np.array([1.]), np.array([[1.8, .1, 0.]]))
    plain = run_materials(config, system, output/"unmonitored", return_data=True)
    config["active_learning"] = {"force_threshold": threshold, "interval_steps": 5, "max_frames": 32}
    monitored = run_materials(config, system, output/"monitored", return_data=True)
    drift = max(float(np.max(np.abs(plain["streams"][0][key]-monitored["streams"][0][key])))
                for key in ("positions", "velocities", "observables"))
    records = sorted((output/"monitored"/"active_learning"/"replica_0000").glob("step_*.npz"))
    selected, labelled = 0, False
    for path in records:
        with load_checkpoint(path, require_integrity=True) as record:
            selected += len(record["selected_beads"])
            labelled |= bool(record["labels_available"])
    learned_correction = make_potential({"provider": "torch", "model": str((output/"delta_0.pt").resolve()),
                                         "device": "cpu", "virial": "none"})
    delta = DeltaLearningPotential(HarmonicPotential(k=.6), learned_correction)
    q = rng.uniform(-.8, .8, size=(4, 1, 3))
    full, split = delta.evaluate(q), delta.split(4, 1)
    contracted = split.evaluate(q)
    fd_errors = []
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        numerical = -(split.evaluate(plus).energy.sum()-split.evaluate(minus).energy.sum())/2e-6
        fd_errors.append(abs(numerical-contracted.forces[index]))
    full_member = make_potential(members[0])
    stats.update({"benchmark_scope": "supervised TorchScript engineering test, not molecular-paper reproduction",
                  "torch_version": torch.__version__, "training_samples": 512,
                  "training_energy_noise_sigma": 2e-4, "coefficients": np.asarray(coefficients).tolist(),
                  "force_acquisition_threshold": threshold, "monitored_trajectory_max_difference": drift,
                  "inspection_records": len(records), "selected_bead_geometries": selected,
                  "reference_labels_available": labelled,
                  "full_delta_vs_full_model_force_max_error": float(np.max(np.abs(full.forces-full_member.evaluate(q).forces))),
                  "contracted_force_chain_rule_max_error": max(fd_errors),
                  "contraction_total_energy_change": float(contracted.energy.sum()-full.energy.sum()),
                  "disagreement_is_calibrated_error_bound": False})
    np.savez_compressed(output/"training_and_heldout.npz", train=train, train_energy=energy,
                        train_energy_noisy=noisy, heldout=heldout, exact_energy=exact_e,
                        predicted_energy=prediction.mean.energy, force_error=force_error,
                        force_disagreement=prediction.max_force_std, exact_forces=exact_f,
                        predicted_forces=prediction.mean.forces)
    write_json(output/"summary.json", stats)
    committee.close(); full_member.close(); learned_correction.close()
    print(json.dumps(stats), flush=True)
    return stats


def plumed_comparator(output, *, depositions=80, probes=21):
    """Same prescribed depositions, direct energies/forces from both codes.

    Unlike the long trajectory benchmark this performs NO interpolation.
    A second PyRPMD object deliberately restores legacy schema-1 semantics
    to retain a quantitative regression witness for the initialization bug.
    """
    import plumed
    output = Path(output)/"plumed_comparator"
    output.mkdir(parents=True, exist_ok=True)
    all_rows, summaries = [], []
    for mode in ("metad", "explore"):
        kernel = plumed.Plumed()
        kernel.cmd("setRealPrecision", 8)
        kernel.cmd("setNatoms", 1)
        kernel.cmd("setMDEngine", "PyRPMD-literature-equations")
        kernel.cmd("setTimestep", .005)
        kernel.cmd("setKbT", 1.)
        kernel.cmd("setLogFile", str((output/f"{mode}.log").resolve()))
        kernel.cmd("init")
        sigma = .1 if mode == "metad" else .1/np.sqrt(20.)
        action = "OPES_METAD" if mode == "metad" else "OPES_METAD_EXPLORE"
        script = (f"UNITS NATURAL\np: POSITION ATOM=1\n"
                  f"b: {action} ARG=p.x PACE=1 SIGMA={sigma:.17g} FIXED_SIGMA "
                  f"BIASFACTOR=20 BARRIER=20 COMPRESSION_THRESHOLD=0 KERNEL_CUTOFF=12 "
                  f"FILE={(output/f'{mode}_KERNELS').resolve()}\n")
        (output/f"{mode}.dat").write_text(script)
        kernel.cmd("readInputLines", script)
        masses, box, virial = np.ones(1), np.zeros((3, 3)), np.zeros((3, 3))
        positions, forces, energy = np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1)
        # The arrays above retain ownership for the entire kernel lifetime.
        def calculate(step, value):
            positions[:] = [value, 0., 0.]
            forces[:] = 0.; virial[:] = 0.
            kernel.cmd("setStep", step)
            kernel.cmd("setPositions", positions)
            kernel.cmd("setMasses", masses)
            kernel.cmd("setForces", forces)
            kernel.cmd("setVirial", virial)
            kernel.cmd("setBox", box)
            kernel.cmd("prepareCalc")
            kernel.cmd("performCalcNoUpdate")
            kernel.cmd("getBias", energy)
            return float(energy[0]), float(forces[0, 0])
        options = dict(cv=Cartesian(0), sigma=[.1], beta=1., barrier=20., bias_factor=20., mode=mode)
        local = OPESBias(**options)
        legacy_state = local.state_dict()
        legacy_state["schema_version"] = 1
        del legacy_state["initial_bias"]
        legacy = OPESBias(**options, state=legacy_state)
        local_initial = float(local.physical_energy([-.3]))
        initial, _ = calculate(0, -.3)
        kernel.cmd("update")
        try:
            for index in range(1, depositions+1):
                center = .7*np.sin(.67*index) + .15*np.cos(.19*index)
                calculate(index, center)
                kernel.cmd("update")
                local.update([center], step=index, time=index*.005)
                legacy.update([center], step=index, time=index*.005)
                for value in np.linspace(-1., 1., probes):
                    actual_e, actual_f = calculate(index, value)
                    expected_e, expected_g = local.physical_energy_gradient([value])
                    legacy_e, legacy_g = legacy.physical_energy_gradient([value])
                    all_rows.append({"mode": mode, "deposition": index, "x": value,
                                     "plumed_energy": actual_e, "pyrpmd_energy": float(expected_e),
                                     "plumed_force": actual_f, "pyrpmd_force": -float(expected_g[0]),
                                     "legacy_v1_energy": float(legacy_e),
                                     "legacy_v1_force": -float(legacy_g[0])})
        finally:
            kernel.finalize()
        rows = [row for row in all_rows if row["mode"] == mode]
        summary = {"mode": mode, "depositions": depositions, "probe_points_per_deposition": probes,
                   "plumed_initial_empty_bias": initial, "pyrpmd_initial_empty_bias": local_initial,
                   "pyrpmd_state_schema_version": local.state_dict()["schema_version"],
                   "literal_PyRPMD_sigma": .1, "PLUMED_input_sigma": sigma,
                   "compression_threshold": 0, "kernel_cutoff": 12,
                   "fixed_sigma": True, "interpolation": False,
                   "max_energy_difference": max(abs(r["plumed_energy"]-r["pyrpmd_energy"]) for r in rows),
                   "max_force_difference": max(abs(r["plumed_force"]-r["pyrpmd_force"]) for r in rows),
                   "legacy_v1_max_energy_difference": max(abs(r["plumed_energy"]-r["legacy_v1_energy"]) for r in rows),
                   "legacy_v1_max_force_difference": max(abs(r["plumed_force"]-r["legacy_v1_force"]) for r in rows),
                   "legacy_scope": "schema-1 restarts preserve zero initial bias; new runs use -barrier"}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_csv(output/"direct_energy_force.csv", all_rows)
    write_json(output/"summary.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["reference", "opes", "summarize", "exchange", "shared", "kinetics", "ml", "plumed"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--freeze", type=int, default=80000)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--cvs", default="x,y")
    parser.add_argument("--variants", default="metad_fixed,explore_fixed,metad_adaptive,explore_adaptive")
    parser.add_argument("--dt", type=float, default=.005)
    parser.add_argument("--sigma", type=float, default=.1)
    parser.add_argument("--pace", type=int, default=500)
    parser.add_argument("--grid-points", type=int, default=2049)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--replicas", type=int, default=256)
    args = parser.parse_args()
    if args.task == "reference" or not (args.output/"reference.npz").is_file():
        print(json.dumps(reference(args.output), indent=2))
    if args.task == "opes":
        for cv in args.cvs.split(","):
            for variant in args.variants.split(","):
                mode, widths = variant.split("_")
                for seed in map(int, args.seeds.split(",")):
                    opes_run(args.output, seed=seed, axis="xy".index(cv), mode=mode,
                             adaptive=widths == "adaptive", steps=args.steps, freeze=args.freeze,
                             dt=args.dt, sigma=args.sigma, pace=args.pace,
                             grid_points=args.grid_points, direct=args.direct)
        summarize_opes(args.output)
    if args.task == "summarize":
        summarize_opes(args.output)
    if args.task == "exchange":
        replica_exchange(args.output, seeds=tuple(map(int, args.seeds.split(","))), steps=args.steps)
    if args.task == "shared":
        shared_metadynamics(args.output, seeds=tuple(map(int, args.seeds.split(","))), steps=args.steps, freeze=args.freeze)
    if args.task == "kinetics":
        kinetics(args.output, replicas=args.replicas, dt=args.dt)
    if args.task == "ml":
        ml_lifecycle(args.output)
    if args.task == "plumed":
        plumed_comparator(args.output)


if __name__ == "__main__":
    main()
