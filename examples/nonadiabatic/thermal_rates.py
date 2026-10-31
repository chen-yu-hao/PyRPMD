"""Run a small, uncalibrated two-state thermal NRPMD rate-curve example.

The sample budget exercises the full API.  It does not establish a kinetic
plateau, independent-chain agreement or physical convergence.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.thermal_nrpmd import (
    ThermalNRPMDSampler, ThermalNRPMDDynamics,
    thermal_population_rate, thermal_flux_side,
)


class ThermalTwoStateModel:
    """Three confined Cartesian coordinates and one vibronic coupling."""

    def evaluate(self, q):
        b = len(q)
        common = 0.5*np.sum(q*q, axis=(1, 2))
        gap = .05+.2*q[:, 0, 0]
        matrix = np.empty((b, 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = common+gap, common-gap
        matrix[:, 0, 1] = matrix[:, 1, 0] = .15
        gradients = np.zeros((b, 2, 2, 1, 3))
        gradients[:, 0, 0], gradients[:, 1, 1] = q, q
        gradients[:, 0, 0, 0, 0] += .2
        gradients[:, 1, 1, 0, 0] -= .2
        return DiabaticResult(matrix, gradients)


def run(samples=64, beads=2, steps=30, seed=20261031):
    potential = ThermalTwoStateModel()
    beta, masses, dt = 2., [1.], .02
    sampler = ThermalNRPMDSampler(potential, masses, beta, np.zeros((beads, 1, 3)), 2,
                                  seed=seed, nuclear_step=.3, centroid_step=.4)
    parents = sampler.sample(samples, burn_in=100, stride=5)
    dynamics = ThermalNRPMDDynamics(potential, masses, beta, beads, dt, 2)
    electronic = thermal_population_rate(parents, dynamics, 0, 1, steps=steps)

    def dividing_surface(centroid):
        return centroid[0, 0], np.array([[1., 0., 0.]])

    nuclear = thermal_flux_side(parents, dynamics, dividing_surface, steps=steps,
                                smoothing_width=.2)
    return {
        "model": "uncalibrated_two_state_harmonic_vibronic_model",
        "beta_au": beta, "masses_au": masses, "dt_au": dt, "beads": beads,
        "seed": seed, "sampling": parents.sampling_diagnostics,
        "electronic": {
            "summary": electronic.summarize(), "time_au": electronic.time.tolist(),
            "rate_curve_au": electronic.rate_curve.tolist() if electronic.rate_curve is not None else None,
        },
        "nuclear": {
            "summary": nuclear.summarize(), "time_au": nuclear.time.tolist(),
            "rate_curve_au": nuclear.rate_curve.tolist() if nuclear.rate_curve is not None else None,
        },
        "physical_converged": False,
        "rate_constant": None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--beads", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.samples, args.beads, args.steps, args.seed)
    content = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")
