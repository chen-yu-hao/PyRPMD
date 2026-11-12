"""Run the input-file path workflows against an independent quartic reference.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/sampling/path_sampling_quartic.py --output runs/path-reference

Every model quantity is in atomic units.  This is a classical one-bead
implementation check, not evidence of physical convergence for a molecule.
The exact rate comes from equilibrium quadrature and the Boltzmann flux
factor, independently of any path-sampling estimator.  FFS uses the explicitly
named basin-Andersen process; its outside-basin excursions are Hamiltonian.
Independent complete runs, including zero-event runs, supply the FFS error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from scipy.integrate import quad

from pyrpmd.constants import KB_AU
from pyrpmd.potentials import PESResult


class QuarticDoubleWell:
    """U(x,y,z)=(x*x-1)**2+(y*y+z*z)/2 for the single reference particle."""

    metadata = {"thread_safety": "reentrant", "execution_device": "cpu"}

    def evaluate(self, q):
        x = q[:, :, 0]
        energy = np.sum((x * x - 1) ** 2, axis=1)
        energy += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * x * (x * x - 1)
        return PESResult(energy, force)


def make_potential():
    """Factory referenced by the accompanying TOML input files."""
    return QuarticDoubleWell()


def exact_reference(beta=1.0, first_interface=-1.0):
    """Canonical A-history flux at an uphill interface and first-order rate.

    A/B are x<=-1 / x>=1 and m=1.  Symmetry gives <h_A>=1/2;
    harmonic spectator integrals cancel.  The flux at the first interface
    already includes reaching it from A.  No fitted simulation data enter.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not -1 <= first_interface <= 0:
        raise ValueError("reference first_interface must be uphill between -1 and 0")
    partition, quadrature_error = quad(
        lambda x: math.exp(-beta * (x * x - 1) ** 2),
        -math.inf, math.inf, epsabs=1e-12, epsrel=1e-12,
    )
    prefactor = 2 / (math.sqrt(2 * math.pi * beta) * partition)
    barrier_to_first = (first_interface * first_interface - 1) ** 2
    return {
        "partition_x": partition,
        "quadrature_error": quadrature_error,
        "flux_au": prefactor * math.exp(-beta * barrier_to_first),
        "crossing_probability": math.exp(-beta * (1 - barrier_to_first)),
        "rate_au": prefactor * math.exp(-beta),
    }


def phase_point(x, momentum):
    """Explicit phase point in the input's [N,3] atomic-unit representation."""
    return {"q": [[float(x), 0.0, 0.0]], "p": [[float(momentum), 0.0, 0.0]]}


def canonical_state(rng, beta=1.0):
    """Exact unbounded canonical state by Gaussian rejection sampling.

    The proposal is proportional to exp(-x*x); the log-density difference
    -beta*(x*x-1)**2+x*x has maximum 1+1/(4*beta).
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    while True:
        x = rng.normal(scale=math.sqrt(0.5))
        log_acceptance = -beta * (x*x - 1)**2 + x*x - 1 - 1/(4*beta)
        if math.log(max(float(rng.random()), np.finfo(float).tiny)) <= log_acceptance:
            break
    q = np.zeros((1, 3))
    q[0, 0] = x
    q[0, 1:] = rng.normal(size=2) / math.sqrt(beta)
    p = rng.normal(size=q.shape) / math.sqrt(beta)
    return {"q": q.tolist(), "p": p.tolist()}


def workflow_input(method, *, seed=981, dt=0.04, samples=2400,
                   burn_in=300, block_size=120, workers=2,
                   flux_steps=6000, trials=160, initial_seed=None):
    """Build the same checked input grammar as the four .inp examples."""
    infinite = method == "infinite-retis"
    selected = "retis" if infinite else method
    if selected not in ("retis", "repptis", "ffs"):
        raise ValueError("method must be retis, infinite-retis, repptis or ffs")
    options = {
        "method": selected, "cv": "x", "basin_a": -1.0, "basin_b": 1.0,
        "interfaces": [-1.0, -0.5], "block_size": block_size,
    }
    if selected == "ffs":
        options.update({
            "dynamics": "basin-andersen", "collision_rate": 1.0,
            "flux_steps": flux_steps, "trials_per_interface": trials,
            "max_trial_steps": 2000, "equilibration_steps": 500,
            "initial_state": canonical_state(np.random.default_rng(
                np.random.SeedSequence([seed, 89171] if initial_seed is None
                                       else [initial_seed, 89171]))),
        })
    else:
        options.update({
            "samples": samples, "burn_in": burn_in,
            "max_path_steps": 600,
        })
        if selected == "retis":
            options.update({
                "minus": True, "seeds": [phase_point(-0.3, 1.4)],
                "minus_seed": phase_point(-1.1, -0.7),
            })
        else:
            centers = [-0.8, -0.4]
            options.update({
                "interfaces": centers,
                "seeds": [phase_point(x, 1.5) for x in centers],
                "external_flux": {
                    "basin_a": -1.0, "basin_b": 1.0,
                    "first_interface": centers[0], "dt_au": dt,
                    "flux_au": exact_reference(first_interface=centers[0])["flux_au"],
                },
            })
    if infinite:
        options.update({"exchange": "infinite", "workers": workers})
    config = {
        "task": "path_sampling", "temperature": 1 / KB_AU,
        "beads": 1, "seed": seed, "native": False,
        "system": {"symbols": ["H"], "masses": [1.0],
                   "geometry": [[0.0, 0.0, 0.0]], "geometry_unit": "bohr"},
        "potential": {"provider": "python",
                      "factory": str(Path(__file__).resolve()) + ":make_potential"},
        "dynamics": {"timestep": dt, "integrator": "cayley-bcb"},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "path_sampling": options,
        "output": {"checkpoint_steps": 100},
    }
    if infinite:
        config["parallel"] = {"backend": "process", "mode": "serial", "scheduling": "dynamic",
                              "trajectory_workers": workers, "threads_per_rank": 1}
    return config


def main():
    from pyrpmd.runner import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=(
        "retis", "repptis", "ffs", "infinite-retis"),
        default=["retis", "repptis", "ffs", "infinite-retis"])
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--burn-in", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=120)
    parser.add_argument("--flux-steps", type=int, default=6000)
    parser.add_argument("--trials", type=int, default=160)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=981)
    args = parser.parse_args()
    if (args.replicates < 2 or args.samples < 1 or args.burn_in < 0
            or args.block_size < 1 or args.flux_steps < 1 or args.trials < 1
            or args.workers < 1 or args.seed < 0
            or not math.isfinite(args.dt) or args.dt <= 0):
        parser.error("use at least two replicates and valid positive budgets/dt")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = exact_reference()
    report = {
        "model": "quartic_x_harmonic_yz_mass1_beta1_beads1",
        "reference": reference,
        "dt_au": args.dt, 
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "cpu_count": os.cpu_count(),
                     "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "methods": {},
    }
    for method in args.methods:
        details = []
        started = time.perf_counter()
        for replica in range(args.replicates):
            config = workflow_input(
                method, seed=args.seed + replica * 1009, dt=args.dt,
                samples=args.samples, burn_in=args.burn_in,
                block_size=args.block_size, workers=args.workers,
                flux_steps=args.flux_steps, trials=args.trials,
            )
            result = run(config, output=args.output / method / f"replica-{replica:03d}")
            details.append({"seed": config["seed"],
                            "rate_au": result["rate_estimate_au"],
                            "status": result.get("status"),
                            "output": result["output"]})
        rates = np.array([item["rate_au"] for item in details], dtype=float)
        if not np.isfinite(rates).all():
            raise RuntimeError("a replicate has no rate estimate; do not drop it")
        mean = float(rates.mean())
        error = float(rates.std(ddof=1) / math.sqrt(len(rates)))
        report["methods"][method] = {
            "replicates": details, "mean_rate_au": mean,
            "whole_run_standard_error_au": error if error > 0 else None,
            "absolute_error_au": abs(mean - reference["rate_au"]),
            "within_five_standard_errors": error > 0 and abs(mean - reference["rate_au"]) <= 5 * error,
            "wall_seconds": time.perf_counter() - started,
            "uncertainty_scope": "Independent whole runs; finite timestep error is separate.",
        }
    path = args.output / "reference.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
