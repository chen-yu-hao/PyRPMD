"""Compare forward-flux sampling with the canonical quartic-well rate.

Run after installation, using one thread for reproducible resource reporting::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/sampling/ffs_quartic_reference.py

All quantities are in atomic units. The one-bead model is
U(x,y,z)=(x*x-1)**2+(y*y+z*z)/2, with unit masses and A/B boundaries -1/+1.
An NVE step is followed by a Maxwell momentum refresh with probability
1-exp(-gamma*dt), but only if the endpoint is inside A or B. The supplied FFS
RNG supplies every random number. There is no thermostat in a transition
excursion, no momentum redraw at an FFS interface, and no time reversal.

This process has different dynamics inside the basins from Hamiltonian RETIS.
Position-conditioned Maxwell refresh preserves the canonical density, while
the law of an A-to-A/B excursion outside the basins is unchanged. Consequently
its continuous-time canonical flux and crossing-probability reference agree
with the Hamiltonian quartic reference. The numerical NVE map introduces a
finite-dt error; this script does not establish its convergence.

Each run starts from an independently sampled canonical state (Gaussian
rejection sampling for x, exact Gaussian spectators/momenta), then receives
additional equilibration under the same forward callback. Rate uncertainty
comes from complete independent FFS runs, keeping flux/bank/genealogy error
together. The script prints one JSON object and creates no output files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time

import numpy as np
from scipy.integrate import quad

from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.ffs import ForwardFluxSampler, combine_ffs_replicates
from pyrpmd.potentials import PESResult


class QuarticDoubleWell:
    """Separable multidimensional teaching PES; every energy is in hartree."""

    def evaluate(self, q):
        x = q[:, :, 0]
        energy = np.sum((x * x - 1) ** 2, axis=1)
        energy += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * x * (x * x - 1)
        return PESResult(energy, force)


def canonical_state(rng, beta):
    """Sample the unbounded canonical distribution without a box cutoff.

    For proposal density proportional to exp(-x*x),
    -beta*(x*x-1)**2+x*x <= 1+1/(4*beta), giving a rigorous rejection bound.
    """
    while True:
        x = rng.normal(scale=math.sqrt(0.5))
        log_acceptance = -beta * (x * x - 1) ** 2 + x * x - 1 - 1 / (4 * beta)
        if math.log(max(float(rng.random()), np.finfo(float).tiny)) <= log_acceptance:
            break
    q = np.zeros((1, 1, 3))
    q[0, 0, 0] = x
    q[0, 0, 1:] = rng.normal(size=2) / math.sqrt(beta)
    p = rng.normal(size=q.shape) / math.sqrt(beta)
    return State(q, p)


def positive_integer(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def positive_real(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=positive_integer, default=8)
    parser.add_argument("--flux-steps", type=positive_integer, default=6000)
    parser.add_argument("--trials", type=positive_integer, default=160)
    parser.add_argument("--equilibration-steps", type=positive_integer, default=2000)
    parser.add_argument("--max-trial-steps", type=positive_integer, default=2000)
    parser.add_argument("--dt", type=positive_real, default=0.04)
    parser.add_argument("--beta", type=positive_real, default=1.0)
    parser.add_argument("--collision-rate", type=positive_real, default=1.0)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("at least two independent replicates are needed for a rate standard error")
    if args.seed < 0:
        parser.error("seed must be nonnegative")

    start = time.perf_counter()
    partition, integration_error = quad(
        lambda x: math.exp(-args.beta * (x * x - 1) ** 2),
        -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12,
    )
    exact_flux = 2 / (math.sqrt(2 * math.pi * args.beta) * partition)
    exact_probability = math.exp(-args.beta)
    exact_rate = exact_flux * exact_probability
    probability = -math.expm1(-args.collision_rate * args.dt)
    sigma_p = 1 / math.sqrt(args.beta)
    identity = {
        "model": "quartic_x_plus_harmonic_y_z.v1", "mass_au": 1.0,
        "nbeads": 1, "beta_au": args.beta, "dt_au": args.dt,
        "integrator": "cayley-bcb", "native": False,
        "momentum_refresh": "after_NVE_step_if_x<=-1_or_x>=1",
        "collision_rate_au_inv": args.collision_rate,
        "collision_probability": probability,
    }
    streams = np.random.SeedSequence(args.seed).spawn(args.replicates)
    runs, details = [], []
    md_steps = 0
    for index, stream in enumerate(streams):
        rng = np.random.default_rng(stream)
        dynamics = Dynamics(QuarticDoubleWell(), [1.0], args.beta, 1, args.dt, native=False)

        def forward_step(state, generator):
            dynamics.step(state)
            x = state.q[0, 0, 0]
            if (x <= -1 or x >= 1) and generator.random() < probability:
                state.p[:] = generator.normal(size=state.p.shape) * sigma_p

        state = canonical_state(rng, args.beta)
        for _ in range(args.equilibration_steps):
            forward_step(state, rng)
        sampler = ForwardFluxSampler(
            forward_step, Cartesian(0), dt=args.dt, basin_a=-1.0, basin_b=1.0,
            interfaces=(-1.0, -0.5), dynamics_id=json.dumps(identity, sort_keys=True),
            coordinate_id="centroid_cartesian_x_atom0.v1",
            flux_steps=args.flux_steps, trials_per_interface=args.trials,
            max_trial_steps=args.max_trial_steps, seed=rng,
        )
        result = sampler.run(state)
        runs.append(result)
        summary = result.summary(block_size=100)
        trial_steps = sum(int(stage.steps.sum()) for stage in result.stages)
        run_steps = args.equilibration_steps + args.flux_steps + trial_steps
        md_steps += run_steps
        details.append({
            "replicate": index, "status": summary["status"],
            "run_identity": result.run_identity,
            "flux_crossings": summary["effective_crossings"],
            "a_history_time_au": summary["a_history_time_au"],
            "discarded_unknown_history_time_au": summary["discarded_unknown_history_time_au"],
            "flux_au": summary["flux_au"],
            "conditional_probabilities": summary["conditional_probabilities"],
            "overall_crossing_probability": summary["overall_crossing_probability"],
            "rate_estimate_au": summary["rate_estimate_au"],
            "distinct_flux_ancestors": [stage["distinct_flux_ancestors"] for stage in summary["interfaces"]],
            "trial_md_steps": trial_steps, "total_md_steps": run_steps,
        })
    combined = combine_ffs_replicates(runs)
    error = combined["rate_standard_error_au"]
    difference = combined["rate_estimate_au"] - exact_rate
    allowance = 0.01 * exact_rate
    flux_error = combined["flux_standard_error_au"]
    flux_difference = combined["flux_au"] - exact_flux
    crossing_error = combined["crossing_probability_standard_error"]
    crossing = combined["overall_crossing_probability"]
    crossing_difference = None if crossing is None else crossing - exact_probability
    rate_agreement = error is not None and abs(difference) <= 3 * error + allowance
    flux_agreement = flux_error is not None and abs(flux_difference) <= 3 * flux_error + 0.01 * exact_flux
    crossing_agreement = (
        crossing_error is not None and crossing_difference is not None
        and abs(crossing_difference) <= 3 * crossing_error + 0.01 * exact_probability
    )
    agreement = rate_agreement and flux_agreement and crossing_agreement
    elapsed = time.perf_counter() - start
    report = {
        "model": identity,
        "basin_a": -1.0, "basin_b": 1.0, "interfaces": [-1.0, -0.5],
        "seed": args.seed, "equilibration_steps_per_run": args.equilibration_steps,
        "flux_steps_per_run": args.flux_steps, "trials_per_interface_per_run": args.trials,
        "exact_continuous_time_reference": {
            "x_partition_integral": partition, "quadrature_absolute_error": integration_error,
            "flux_au": exact_flux, "overall_crossing_probability": exact_probability,
            "rate_au": exact_rate,
        },
        "ffs": combined,
        "comparison": {
            "rate_difference_au": difference,
            "rate_difference_in_standard_errors": None if error is None or error == 0 else difference / error,
            "flux_difference_au": flux_difference,
            "flux_difference_in_standard_errors": (
                None if flux_error is None or flux_error == 0 else flux_difference / flux_error
            ),
            "crossing_probability_difference": crossing_difference,
            "crossing_probability_difference_in_standard_errors": (
                None if crossing_error is None or crossing_error == 0
                else crossing_difference / crossing_error
            ),
            "finite_dt_allowance_fraction": 0.01,
            "rate_agreement": bool(rate_agreement),
            "flux_agreement": bool(flux_agreement),
            "crossing_probability_agreement": bool(crossing_agreement),
            "agrees_within_three_standard_errors_plus_dt_allowance": bool(agreement),
        },
        "runs": details,
        "resources": {
            "wall_seconds": elapsed, "total_md_steps": md_steps,
            "md_steps_per_wall_second": md_steps / elapsed,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "platform": platform.system(), "machine": platform.machine(),
            "processor": platform.processor() or None,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "native": False,
        },
        "converged": False, "physical_converged": False,
        "qualification": (
            "In-basin dynamics differs from Hamiltonian RETIS: a region-conditioned "
            "Maxwell refresh makes the flux run ergodic in energy. Canonical density "
            "and outside-basin excursion dynamics supply the common continuous-time "
            "reference. Starts are independently canonical-sampled and equilibrated; "
            "finite-dt stationary bias and stationarity still require a convergence "
            "study. Independent whole-run error includes finite banks and shared "
            "ancestry. The fixed 1% dt allowance is a validation allowance, not a "
            "measured integration error bound."
        ),
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if agreement else 1


if __name__ == "__main__":
    raise SystemExit(main())
