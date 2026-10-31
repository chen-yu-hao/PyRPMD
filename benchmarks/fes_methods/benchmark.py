"""Reproducible 2D sampling comparison using PyRPMD production workflows.

The harmonic centroid FES is exact for every bead count, including in the
quantum regime beta*hbar*omega=4. This is an implementation benchmark, not
a molecular rate calculation. Independent seeds are never pooled for errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import ndtr

from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, write_json
from pyrpmd.workflows.metadynamics import run_metadynamics, _new_bias, resolve_metadynamics
from pyrpmd.analysis.metadynamics import bias_fes

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("fixed", "adaptive", "unbiased", "standard", "well_tempered")


def compile_harmonic(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "native/examples/harmonic_pes.c"
    library = output / "harmonic.so"
    cmd = ["cc", "-shared", "-fPIC", "-O2", "-std=c11", "-I",
           str(ROOT / "native/include"), str(source), "-o", str(library)]
    subprocess.run(cmd, check=True, capture_output=True)
    write_json(output / "build.json", {"command": cmd,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest()})
    return library


def configuration(method, beads, seed, library, scale=1):
    config = {
        "task": "fes", "beta": 4.0, "beads": beads, "dt_au": 0.04,
        "seed": seed, "native": "required",
        "potential": {"provider": "cabi", "library": str(library)},
        "dynamics": {"loop_backend": "auto", "block_steps": 256},
        "sampling": {"equilibration_steps": 1000, "production_steps": 8000 * scale,
                     "sample_interval": 10, "andersen_interval": 40, "replicas": 1},
        "cvs": [{"name": a, "kind": "cartesian", "atom": 0, "axis": a,
                 "estimator": "centroid"} for a in ("x", "y")],
        "analysis": {"bins": [24, 24], "ranges": [[-1.5, 1.5]] * 2,
                     "bootstrap": 0, "min_count": 5},
        "output": {"checkpoint_steps": 1000000},
    }
    if method in ("fixed", "adaptive"):
        axis = np.linspace(-1, 1, 5 if method == "fixed" else 3)
        config["umbrella"] = {"centers": [[x, y] for x in axis for y in axis], "stiffness": 8.0}
        if method == "adaptive":
            config["adaptive_umbrella"] = {
                "pilot_steps": 2000, "pilot_equilibration_steps": 500,
                "max_rounds": 6, "max_windows": 64, "target_overlap": 0.02,
                "min_effective_samples": 20.0,
            }
    elif method == "unbiased":
        # A negligible known bias allows exactly the same saved MBAR pipeline.
        config["umbrella"] = {"centers": [[0, 0]], "stiffness": 1e-12}
        config["sampling"]["production_steps"] = 60000 * scale
    else:
        config["task"] = "metadynamics"
        config["sampling"]["production_steps"] = 60000 * scale
        config["metadynamics"] = {"mode": method, "sigma": [0.25, 0.25],
                                  "height": 0.025, "pace": 100}
        if method == "well_tempered":
            config["metadynamics"]["bias_factor"] = 5.0
    return config


def harmonic_reference(edges, point=False):
    centers = [0.5 * (e[1:] + e[:-1]) for e in edges]
    x, y = np.meshgrid(*centers, indexing="ij")
    point_f = 0.5 * (x*x + y*y)
    if point:
        return point_f, point_f
    densities = [np.diff(ndtr(e * 2.0)) / np.diff(e) for e in edges]
    return -np.log(densities[0][:, None] * densities[1][None, :]) / 4.0, point_f


def metrics(data, method, reference=None):
    exact, point_f = harmonic_reference(data["edges"], point=method in ("standard", "well_tempered")) if reference is None else reference
    beta = 4.0
    roi = beta * (point_f - point_f.min()) <= 4.0
    good = roi & ~np.asarray(data["mask"]) & np.isfinite(data["free_energy"])
    residual = beta * (data["free_energy"] - exact)
    offset = float(np.mean(residual[good])) if good.any() else np.nan
    residual = residual - offset
    return {"roi_bins": int(roi.sum()), "supported_roi_bins": int(good.sum()),
            "roi_coverage": float(good.sum() / roi.sum()),
            "rmse_kbt": float(np.sqrt(np.mean(residual[good]**2))) if good.any() else None,
            "max_abs_error_kbt": float(np.max(np.abs(residual[good]))) if good.any() else None,
            "gauge_offset_kbt": offset, "all_supported_bins": int(np.sum(~data["mask"]))}, residual, good


def run_case(method, beads, seed, output, library, scale=1):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = configuration(method, beads, seed, library, scale)
    system = System(("H",), np.array([1.0]), np.zeros((1, 3)))
    write_json(output / "input-resolved.json", {"config": cfg, "system": system.to_dict()})
    start = time.monotonic()
    result = (run_metadynamics if cfg["task"] == "metadynamics" else run_fes)(
        cfg, system, output / "run", return_data=True)
    wall = time.monotonic() - start
    score, residual, good = metrics(result["data"], method)
    windows = result["summary"].get("windows", 1)
    pilot_steps = 0
    adaptive = None
    planfile = output / "run/adaptive-plan.json"
    if planfile.exists():
        adaptive = json.loads(planfile.read_text())
        pilot_steps = sum(len(r["centers"]) * (r["pilot"]["equilibration_steps"] +
                         r["pilot"]["production_steps"]) for r in adaptive["rounds"])
    total_steps = windows * (cfg["sampling"]["equilibration_steps"] +
                            cfg["sampling"]["production_steps"]) + pilot_steps
    report = {"method": method, "beads": beads, "seed": seed, "scale": scale,
              "wall_seconds_this_call": wall, "integration_steps": total_steps,
              "bead_steps": beads * total_steps, "pilot_steps": pilot_steps,
              "samples": result["summary"]["samples"], "windows": windows,
              **score, "physical_convergence_claimed": False,
              "oracle": "exact harmonic centroid point FES" if cfg["task"] == "metadynamics"
                        else "exact harmonic centroid bin-integrated density",
              "overlap_connected": result["summary"].get("overlap_connected"),
              "adaptive_rounds": len(adaptive["rounds"]) if adaptive else 0,
              "pilot_samples_used_in_production": 0,
              "statistical_note": "Finite sample error on supported ROI; no pass/fail by tuning this error.",
    }
    if cfg["task"] == "metadynamics":
        cv, _, opts = resolve_metadynamics(cfg, system)
        stream = result["streams"][0]
        report["hills"] = len(stream["hills"])
        report["prefix_errors"] = []
        for fraction in (0.25, 0.5, 1.0):
            cutoff = cfg["sampling"]["equilibration_steps"] + int(fraction * cfg["sampling"]["production_steps"])
            hills = [h for h in stream["hills"] if h["step"] <= cutoff]
            values = stream["samples"][stream["sample_steps"] <= cutoff]
            bias = _new_bias(cv, opts, cfg["beta"], hills)
            estimate = bias_fes([bias], [values], bins=cfg["analysis"]["bins"],
                                ranges=cfg["analysis"]["ranges"], min_count=5)
            report["prefix_errors"].append({"fraction": fraction, **metrics(estimate, method)[0]})
    write_json(output / "comparison.json", report)
    np.savez_compressed(output / "comparison.npz", residual_kbt=residual, roi_support=good,
                        free_energy=result["data"]["free_energy"], mask=result["data"]["mask"],
                        edges_0=result["data"]["edges"][0], edges_1=result["data"]["edges"][1])
    print(json.dumps(report), flush=True)
    return report


def dynamics_check(output, library):
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.potentials.cabi import CABIPotential
    rows = []
    for scheme in ("cayley-bcb", "normal-mode-bab", "velocity-verlet"):
        for beads in (1, 8, 32):
            for dt in (0.04, 0.02):
                pes = CABIPotential(library)
                dyn = Dynamics(pes, [1.0], 4.0, beads, dt, native="required",
                               integrator=scheme, loop_backend="native")
                q = np.zeros((beads, 1, 3)); p = q.copy()
                q[:, 0, 0] = 0.5
                if beads > 1:
                    q[:, 0, 0] += 0.1 * np.cos(2*np.pi*np.arange(beads)/beads)
                    p[:, 0, 0] = 0.2 * np.sin(2*np.pi*np.arange(beads)/beads)
                state = State(q, p)
                trace = dyn.propagate(state, round(20/dt), sample_interval=1)
                t = trace.steps * dt
                correlation = 0.5 * trace.centroids[:, 0, 0]
                exact = 0.25 * np.cos(t)
                rows.append({"scheme": scheme, "beads": beads, "dt": dt,
                             "max_error_relative_C0": float(np.max(np.abs(correlation-exact))/0.25),
                             "max_relative_energy_drift": float((trace.maximum_energy-trace.minimum_energy)/abs(trace.initial_energy))})
                pes.close()
    write_json(Path(output) / "harmonic-dynamics.json", {
        "reference": "Craig and Manolopoulos 2004, DOI 10.1063/1.1777575; exact harmonic Kubo Cqq(t)=cos(t)/4",
        "construction": "Exact linear response of equilibrium q covariance; internal modes have zero centroid. No stochastic fitting, bias or thermostat.",
        "rows": rows})
    print(json.dumps(rows), flush=True)


def quantum_check(output, library):
    """Check bead fluctuations, which differ from the classical centroid FES."""
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.potentials.cabi import CABIPotential
    from scipy.linalg import solve_discrete_lyapunov
    rows = []
    for beads in (1, 4, 8, 16, 32):
        pes = CABIPotential(library)
        dyn = Dynamics(pes, [1.0], 4.0, beads, .04, seed=9329+beads,
                       native="required", loop_backend="native")
        state = State(np.zeros((beads, 1, 3)), dyn.sample_momenta())
        values = []
        for block in range(10200):
            state.p[:] = dyn.sample_momenta()
            dyn.propagate(state, 40, sample_interval=40, record_centroids=False)
            if block >= 200:
                x = state.q[:, 0, 0]
                values.append([np.mean(x*x), np.mean(x)**2])
        values = np.asarray(values)
        batch = values.reshape(100, 100, 2).mean(1)
        omega = 2*beads/4 * np.sin(np.pi*np.arange(beads)/beads)
        finite_p = float(np.sum(1/(1+omega*omega))/4)
        # Independent stationary covariance of the actual linear update,
        # including Andersen momentum variance P/beta. This distinguishes
        # finite simulation noise from an incorrect quantum invariant.
        maps = []
        for dt in (.04, .02):
            mapped = Dynamics(pes, [1.0], 4.0, beads, dt, native="required",
                              loop_backend="native")
            aa = np.empty((beads, beads)); bb = aa.copy()
            for col in range(beads):
                for momentum, matrix in ((False, aa), (True, bb)):
                    q0 = np.zeros((beads, 1, 3)); p0 = q0.copy()
                    (p0 if momentum else q0)[col, 0, 0] = 1
                    basis_state = State(q0, p0)
                    mapped.propagate(basis_state, 40, sample_interval=40, record_centroids=False)
                    matrix[:, col] = basis_state.q[:, 0, 0]
            covariance = solve_discrete_lyapunov(aa, (beads/4) * bb @ bb.T)
            var = float(np.trace(covariance)/beads)
            maps.append({"dt":dt,"stationary_bead_second_moment":var,
                         "relative_error_vs_finite_P":var/finite_p-1,
                         "spectral_radius":float(np.max(abs(np.linalg.eigvals(aa))))})
        rows.append({"beads": beads, "bead_second_moment": float(values[:, 0].mean()),
                     "centroid_second_moment": float(values[:, 1].mean()),
                     "bead_batch_se": float(batch[:, 0].std(ddof=1)/10),
                     "centroid_batch_se": float(batch[:, 1].std(ddof=1)/10),
                     "finite_P_exact_bead_second_moment": finite_p,
                     "infinite_P_exact_bead_second_moment": float(.5/np.tanh(2)),
                     "exact_centroid_second_moment": .25,
                     "integration_steps": 408000, "saved_states": 10000,
                     "deterministic_stationary_checks":maps})
        pes.close()
    write_json(Path(output) / "harmonic-quantum.json", {
        "beta_hbar_omega": 4.0, "dt": .04,
        "reference": "Finite-P normal-mode Gaussian integral; quantum limit coth(beta*hbar*omega/2)/(2*m*omega)",
        "uncertainty": "100 sequential batches of 100 thermostat intervals; diagnostic standard errors, no molecular convergence claim",
        "rows": rows})
    print(json.dumps(rows), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["build", "run", "dynamics", "quantum"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=METHODS, default="fixed")
    p.add_argument("--beads", type=int, default=8)
    p.add_argument("--seed", type=int, default=3101)
    p.add_argument("--scale", type=int, default=1)
    args = p.parse_args()
    base = args.output.resolve()
    if args.action == "build":
        print(compile_harmonic(base / "build")); return
    library = base / "build/harmonic.so"
    if not library.exists():
        library = compile_harmonic(base / "build")
    if args.action == "dynamics":
        dynamics_check(base, library); return
    if args.action == "quantum":
        quantum_check(base, library); return
    case = base / f"{args.method}-p{args.beads}-s{args.seed}-n{args.scale}"
    run_case(args.method, args.beads, args.seed, case, library, args.scale)


if __name__ == "__main__":
    main()
