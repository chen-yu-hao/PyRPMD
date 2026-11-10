"""Transfer error of the multiplicative subspace correction on 2D Eckart models.

For a full 2D Hamiltonian H and a 1D projected model H_S along the valley,

    Delta = ln[k_QM(H) / k_RPMD(H)] - ln[k_QM(H_S) / k_RPMD(H_S)]
          = G_QM - G_RPMD,      G_X = ln k_X(H) - ln k_X(H_S),

is the error of k_RPMD(H) * C_S (docs/subspace-transfer-theory.md).  Two
families of couplings to a transverse vibration y (mass 1060 au in s and y):

    curvature:  V = V_b(s) + m w_y^2 (y - h sech^2(s/a))^2 / 2        (w_y = 0.010 Eh)
    frequency:  V = V_b(s) + m w(s)^2 y^2 / 2,  w(s) = w_B (1 - 0.3 sech^2(s/a))

with V_b(s) = V0 sech^2(s/a) - D (1 + tanh(s/a)) / 2 (V0 = 0.425 eV, a = 0.66
bohr; D = 0 symmetric or 0.2 eV asymmetric).  The symmetric barrier has
T_c = 413 K.  H_S is the vibrationally adiabatic (zero-point) or the harmonic
free-energy projection along the valley floor (pyrpmd project_path_subspace).

    python benchmarks/subspace_theory/transfer.py --family curvature --parameter 0.6 \\
        --barrier symmetric --temperature 200 --output point.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import time

import numpy as np

from pyrpmd.analysis.subspace_correction import harmonic_bead_factor
from pyrpmd.analysis.subspace_model import model_from_function, path_curvature, project_path_subspace
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate

KELVIN = 3.166811563e-6
MASS = 1060.0
V0, A = 0.425 / 27.211386, 0.66
OMEGA_Y = 0.010
SOFTEN = 0.3
ASYMMETRY = 0.2 / 27.211386
S_RANGE = 8.0
REFERENCE_S = -4.0


class Model:
    def __init__(self, family, parameter, barrier):
        if family not in ("curvature", "frequency"):
            raise ValueError("family must be 'curvature' or 'frequency'")
        self.family, self.parameter = family, float(parameter)
        self.drop = ASYMMETRY if barrier == "asymmetric" else 0.0

    # --- pieces of V(s, y) and their s-derivatives -----------------------------------------
    def barrier(self, s):
        sech2, tanh = 1.0 / np.cosh(s / A) ** 2, np.tanh(s / A)
        return V0 * sech2 - 0.5 * self.drop * (1.0 + tanh), (-2.0 * V0 * tanh * sech2 - 0.5 * self.drop * sech2) / A

    def centre(self, s):
        if self.family == "frequency":
            return 0.0 * s, 0.0 * s
        sech2 = 1.0 / np.cosh(s / A) ** 2
        return self.parameter * sech2, -2.0 * self.parameter / A * np.tanh(s / A) * sech2

    def omega(self, s):
        if self.family == "curvature":
            return OMEGA_Y + 0.0 * s, 0.0 * s
        sech2 = 1.0 / np.cosh(s / A) ** 2
        return (self.parameter * (1.0 - SOFTEN * sech2),
                self.parameter * SOFTEN * 2.0 / A * np.tanh(s / A) * sech2)

    def __call__(self, points):
        points = np.asarray(points, dtype=float)
        s, y = points[..., 0], points[..., 1]
        vb, _ = self.barrier(s)
        c, _ = self.centre(s)
        w, _ = self.omega(s)
        return vb + 0.5 * MASS * w**2 * (y - c) ** 2

    def gradient(self, points):
        points = np.asarray(points, dtype=float)
        s, y = points[..., 0], points[..., 1]
        _, dvb = self.barrier(s)
        c, dc = self.centre(s)
        w, dw = self.omega(s)
        out = np.empty_like(points)
        out[..., 0] = dvb + MASS * w * dw * (y - c) ** 2 - MASS * w**2 * (y - c) * dc
        out[..., 1] = MASS * w**2 * (y - c)
        return out

    def scalar(self, x):
        return float(self(np.asarray(x)[None])[0])

    def scalar_gradient(self, x):
        return self.gradient(np.asarray(x)[None])[0]

    def frequencies(self):
        """Transverse frequency at the reactant reference and at the barrier top of V_b."""
        s = np.linspace(-S_RANGE, S_RANGE, 4001)
        top = s[np.argmax(self.barrier(s)[0])]
        return float(self.omega(np.array(REFERENCE_S))[0]), float(self.omega(np.array(top))[0]), float(top)


def bead_number(model, beta, minimum=16):
    w_max = max(model.omega(np.linspace(-S_RANGE, S_RANGE, 201))[0].max(), OMEGA_Y if model.family == "curvature" else 0)
    return int(max(minimum, 8 * np.ceil(2.0 * beta * w_max / 8.0)))


def full_model(model, beta, spacing_s=0.1, sigma_fraction=2.3):
    s_axis = np.arange(-S_RANGE, S_RANGE + 1e-9, spacing_s)
    w = model.omega(s_axis)[0]
    w_min = float(w.min())
    sigma0 = np.sqrt(1.0 / (2.0 * MASS * w_min))
    sigma_th = sigma0 * np.sqrt(1.0 / np.tanh(0.5 * beta * w_min))
    c = model.centre(s_axis)[0]
    # The grid edge is a wall too: it must lie beyond the truncation energy.
    extent = max(7.0 * sigma_th, np.sqrt(2.0 * truncation_offset(beta) / (MASS * w_min**2)) + 2.0 * sigma0)
    lo, hi = float(c.min() - extent), float(c.max() + extent)
    y_axis = np.arange(lo, hi + 1e-9, sigma0 / sigma_fraction)
    top = model.frequencies()[2]
    return model_from_function(model, [s_axis, y_axis], gradient=model.gradient, masses=[MASS, MASS],
                               dividing_surface=float(s_axis[np.argmin(abs(s_axis - top))]),
                               reactant_reference=REFERENCE_S,
                               provenance={"family": model.family, "parameter": model.parameter})


def valley_path(model, samples=3201, half_length=2 * S_RANGE):
    # Twice the 2D box: RPMD products accelerate downhill and must stay on the 1D grid.
    s = np.linspace(-half_length, half_length, samples)
    return np.column_stack([s, model.centre(s)[0]])


def path_projection(model, beta, environment, spacing=1.0, curvature_mass="none", zero_point="harmonic"):
    path = valley_path(model)
    arc, curvature = path_curvature(path, [MASS, MASS], samples=2001)
    axis = np.arange(0.0, arc[-1] - 1e-9, spacing)
    # Arc length of the reference point s = REFERENCE_S along the valley.
    s_of_arc = np.interp(arc, np.linspace(0.0, arc[-1], len(path)), path[:, 0])
    reference = float(np.interp(REFERENCE_S, s_of_arc, arc))
    sub = project_path_subspace(model.scalar, model.scalar_gradient, path, [MASS, MASS], axis,
                                environment=environment, beta=beta if environment == "free_energy" else None,
                                reactant_reference=reference, curvature=curvature_mass, zero_point=zero_point)
    return sub, arc, curvature


def bead_width(omega, beta):
    """sqrt(2 x bead-about-centroid variance) of a harmonic mode (mass-weighted)."""
    variance = 0.5 / omega / np.tanh(0.5 * beta * omega) - 1.0 / (beta * omega**2)
    return np.sqrt(2.0 * variance)


def curvature_width(model, beta):
    path = valley_path(model)
    arc, curvature = path_curvature(path, [MASS, MASS], samples=2001)
    s_of_arc = np.interp(arc, np.linspace(0.0, arc[-1], len(path)), path[:, 0])
    w = model.omega(s_of_arc)[0]
    # Normal-mode frequency of the transverse vibration: w * sqrt(1 + c'^2) on a tilted valley.
    dc = model.centre(s_of_arc)[1]
    product = curvature * bead_width(w * np.sqrt(1.0 + dc**2), beta)
    return float(product.max()), float(curvature.max())


# Energy-truncated DVR: points above (barrier + largest zero-point energy + offset) are
# removed.  With the sinc-DVR's non-local kinetic matrix the rate converges slowly in the
# offset: for the curvature model at 200 K it is 43% low at 0.016 Eh, 13% at 0.03 Eh and
# within 0.5% from 0.05 Eh (docs/validation/2026-11-09-subspace-transfer).
REFINE = {"ladder": [1.0, 0.8], "truncation_offset": 0.07, "spacing": 1.0}


def truncation_offset(beta):
    return max(REFINE["truncation_offset"], 30.0 / beta)


def quantum_options(model, beta):
    w_reactant = model.omega(np.array(REFERENCE_S))[0]
    threshold = 0.5 * float(w_reactant)
    return {"method": "sylvester", "dense_limit": 11000, "energy_floor": threshold - 1.0 / beta,
            "truncation_energy": max(V0, 0.0) + 0.5 * float(model.omega(np.linspace(-S_RANGE, S_RANGE, 201))[0].max())
            + truncation_offset(beta)}


def _limit_threads(count):
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional
        return None
    return threadpool_limits(count)


def _quantum(args, threads=1):
    kind, family, parameter, barrier, beta, environment, curvature_mass, zero_point = args[:8]
    if len(args) > 8:
        REFINE.update(args[8])
    _limit_threads(threads)
    model = Model(family, parameter, barrier)
    start = time.time()
    if kind == "2d":
        # Spacing ladder: the finest grid is used, the others measure grid convergence.
        ladder = []
        for spacing in REFINE["ladder"]:
            start = time.time()
            sub = full_model(model, beta, spacing_s=0.1 * spacing, sigma_fraction=2.3 / spacing)
            result = exact_quantum_rate(sub, beta, **quantum_options(model, beta))
            ladder.append({"spacing": spacing, "rate": result.rate, "seconds": time.time() - start,
                           "retained_points": result.diagnostics.get("retained_points"),
                           "grid_points": result.diagnostics.get("grid_points"),
                           "states_below_floor": result.diagnostics.get("states_below_floor")})
            print(f"2D exact spacing {spacing:g}: k = {result.rate:.6e} "
                  f"({ladder[-1]['retained_points']} points, {ladder[-1]['seconds']:.0f} s)", flush=True)
        finest = min(ladder, key=lambda item: item["spacing"])
        coarse = max(ladder, key=lambda item: item["spacing"])
        return dict(finest, ladder=ladder, truncation_offset=truncation_offset(beta),
                    grid_relative_change=float(finest["rate"] / coarse["rate"] - 1.0))
    else:
        sub = path_projection(model, beta, environment, curvature_mass=curvature_mass, zero_point=zero_point)[0]
        # Reactant threshold of this H_S (zero-point or free energy of the reactant vibration).
        threshold = float(np.interp(sub.reactant_reference, sub.axes[0], sub.values))
        opts = {"method": "sylvester", "energy_floor": threshold - 1.0 / beta}
    result = exact_quantum_rate(sub, beta, **opts)
    return {"rate": result.rate, "seconds": time.time() - start,
            "retained_points": result.diagnostics.get("retained_points"),
            "grid_points": result.diagnostics.get("grid_points"),
            "states_below_floor": result.diagnostics.get("states_below_floor"),
            "model_provenance": dict(sub.provenance)}


def _rpmd(args):
    kind, family, parameter, barrier, beta, environment, curvature_mass, zero_point, nbeads, sampling, seed = args
    _limit_threads(1)
    model = Model(family, parameter, barrier)
    sub = full_model(model, beta) if kind == "2d" else path_projection(model, beta, environment,
                                                                         curvature_mass=curvature_mass, zero_point=zero_point)[0]
    start = time.time()
    r = ring_polymer_rate(sub, beta, nbeads, seed=seed, **sampling)
    return {"rate": r.rate, "log_standard_error": r.log_rate_standard_error, "qtst": r.qtst_rate,
            "kappa": r.transmission, "kappa_standard_error": r.transmission_standard_error,
            "max_energy_error_beta_p": r.diagnostics.get("max_energy_error_beta_p"),
            "spline_outside_fraction": r.diagnostics.get("spline_outside_fraction"),
            "seconds": time.time() - start}


def run_point(family, parameter, barrier, temperature, *, nbeads=None, environments=("adiabatic", "free_energy"),
              curvature_mass="none", zero_point="harmonic",
              sampling=None, seed=0, workers=None):
    beta = 1.0 / (KELVIN * temperature)
    model = Model(family, parameter, barrier)
    nbeads = bead_number(model, beta) if nbeads is None else int(nbeads)
    sampling = dict(sampling or {})
    tasks_q = [("2d", family, parameter, barrier, beta, None, None, zero_point, json.loads(json.dumps(REFINE)))] + [
        ("1d", family, parameter, barrier, beta, e, curvature_mass, zero_point) for e in environments]
    tasks_r = [("2d", family, parameter, barrier, beta, None, None, zero_point, nbeads, sampling, seed)] + [
        ("1d", family, parameter, barrier, beta, e, curvature_mass, zero_point, nbeads, sampling, seed + 1 + i)
        for i, e in enumerate(environments)]
    cores = workers or os.cpu_count() or 1
    # The dense 2D solve dominates: run it here with every BLAS thread while
    # the RPMD runs and the cheap 1D solves share a process pool.
    import multiprocessing

    with ProcessPoolExecutor(max_workers=max(1, min(cores - 1, len(tasks_r) + len(tasks_q) - 1)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures_q = [pool.submit(_quantum, task) for task in tasks_q[1:]]
        futures_r = [pool.submit(_rpmd, task) for task in tasks_r]
        full_quantum = _quantum(tasks_q[0], threads=cores)
        quantum = [full_quantum] + [f.result() for f in futures_q]
        rpmd = [f.result() for f in futures_r]
    w_r, w_ts, top = model.frequencies()
    bead = harmonic_bead_factor([w_ts], [w_r], beta, nbeads)
    kr, kc = curvature_width(model, beta)
    record = {
        "family": family, "parameter": float(parameter), "barrier": barrier, "temperature": float(temperature),
        "curvature_mass": curvature_mass, "zero_point": zero_point,
        "beta": beta, "nbeads": nbeads, "seed": seed, "sampling": sampling,
        "crossover_temperature_symmetric": 413.1,
        "curvature_width_max": kr, "path_curvature_max": kc,
        "beta_omega_reactant": beta * w_r, "beta_omega_barrier": beta * w_ts,
        "environment_bead_factor": bead["factor"],
        "quantum_grid": dict(REFINE),
        "full": {"quantum": quantum[0], "rpmd": rpmd[0]},
        "projections": {},
    }
    c_full = quantum[0]["rate"] / rpmd[0]["rate"]
    for e, q, r in zip(environments, quantum[1:], rpmd[1:]):
        c_s = q["rate"] / r["rate"]
        delta = np.log(c_full) - np.log(c_s)
        record["projections"][e] = {
            "quantum": q, "rpmd": r, "C_S": c_s,
            "delta": float(delta),
            "delta_bead_corrected": float(delta + np.log(bead["factor"])),
            "delta_standard_error": float(np.hypot(rpmd[0]["log_standard_error"], r["log_standard_error"])),
            # H rates are per bohr of s, H_S rates per mass-weighted unit of arc length (sqrt(m) bohr).
            "G_quantum": float(np.log(quantum[0]["rate"] / (q["rate"] / np.sqrt(MASS)))),
            "G_rpmd": float(np.log(rpmd[0]["rate"] / (r["rate"] / np.sqrt(MASS)))),
        }
    record["C_full"] = c_full
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True, choices=["curvature", "frequency"])
    p.add_argument("--parameter", type=float, required=True, help="h (bohr) or w_B (Eh)")
    p.add_argument("--barrier", default="symmetric", choices=["symmetric", "asymmetric"])
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--beads", type=int, default=None, help="default: max(16, 2 beta w_max rounded up to 8)")
    p.add_argument("--environments", nargs="+", default=["adiabatic", "free_energy"])
    p.add_argument("--curvature-mass", choices=["none", "small"], default="none",
                   help="SCT coordinate-dependent mass for 1D path projections")
    p.add_argument("--zero-point", choices=["harmonic", "rph", "dvr"], default="harmonic")
    p.add_argument("--windows", type=int, default=48)
    p.add_argument("--walkers", type=int, default=64)
    p.add_argument("--sweeps", type=int, default=1000)
    p.add_argument("--trajectories", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--ladder", type=float, nargs="+", default=[1.0, 0.8],
                   help="2D DVR spacing factors; the finest is used, the spread is the grid check")
    p.add_argument("--truncation-offset", type=float, default=0.07,
                   help="truncation energy above barrier + zero-point energy (Eh; at least 30 kT)")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    start = time.time()
    REFINE.update(ladder=list(args.ladder), truncation_offset=args.truncation_offset)
    sampling = {"windows": args.windows, "walkers": args.walkers, "sweeps": args.sweeps,
                "trajectories": args.trajectories}
    record = run_point(args.family, args.parameter, args.barrier, args.temperature, nbeads=args.beads,
                       environments=tuple(args.environments), sampling=sampling, seed=args.seed,
                       workers=args.workers, curvature_mass=args.curvature_mass, zero_point=args.zero_point)
    record["wall_seconds"] = time.time() - start
    with open(args.output, "w") as handle:
        json.dump(record, handle, indent=1, default=float)
    summary = {k: record[k] for k in ("family", "parameter", "barrier", "temperature", "nbeads",
                                       "curvature_width_max", "C_full")}
    summary.update({e: {k: v[k] for k in ("C_S", "delta", "delta_standard_error", "G_quantum", "G_rpmd")}
                    for e, v in record["projections"].items()})
    print("RESULT " + json.dumps(summary, default=float))


if __name__ == "__main__":
    main()
