#!/usr/bin/env python3
"""Exact stationary covariance of the actual harmonic sampling integrator.

An affine Gaussian integrator obeys C=M C M^T+B B^T. Extract M and B by
propagating basis vectors with the public Dynamics.sample_step API, then solve
the discrete Lyapunov equation. Compare to an independently integrated action.
This removes trajectory sampling noise from the finite-timestep error test.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from path_integral import sc_variance, pigs_variance, write_csv
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.pimd_methods import SuzukiChinPotential, PIGSPotential, GaussianTrialWavefunction
from pyrpmd.potentials.builtin import HarmonicPotential


class PrescribedNoise:
    def __init__(self, beads):
        self.bit_generator = np.random.default_rng(0).bit_generator
        self.values = np.zeros((beads, 1, 3))

    def normal(self, *, size):
        assert tuple(size) == self.values.shape
        return self.values.copy()


def covariance(method, dt):
    if method == 'suzuki_chin':
        p, beta = 8, 4.
        model = SuzukiChinPotential(HarmonicPotential(), [1.], beta, p)
        target = sc_variance(beta, p, fourth_order=True)
        indices = np.arange(0, p, 2)
    else:
        p = 33
        model = PIGSPotential(HarmonicPotential(), [1.], p, 8., GaussianTrialWavefunction(.25))
        beta = model.beta
        target = pigs_variance(8., p)[0]
        indices = np.array([p//2])
    dyn = Dynamics(model, [1.], beta, p, dt, native=True)
    noise = PrescribedNoise(p)
    dyn.rng = noise
    def propagate(vector, random):
        noise.values[:, 0, 0] = random
        q, momenta = np.zeros((p, 1, 3)), np.zeros((p, 1, 3))
        q[:, 0, 0], momenta[:, 0, 0] = vector[:p], vector[p:]
        state = State(q, momenta)
        dyn.sample_step(state)
        return np.concatenate((state.q[:, 0, 0], state.p[:, 0, 0]))
    matrix = np.column_stack([propagate(v, np.zeros(p)) for v in np.eye(2*p)])
    injection = np.column_stack([propagate(np.zeros(2*p), v) for v in np.eye(p)])
    stationary = solve_discrete_lyapunov(matrix, injection@injection.T)
    variance = float(np.diag(stationary)[indices].mean())
    residual = stationary-matrix@stationary@matrix.T-injection@injection.T
    return dict(method=method, timestep=dt, beads=p, finite_action_variance=target,
                stationary_integrator_variance=variance, relative_timestep_bias=(variance-target)/target,
                lyapunov_max_residual=float(np.max(abs(residual))),
                spectral_radius=float(np.max(abs(np.linalg.eigvals(matrix)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/validation/tier2-literature-2026-11-01/path_integral'))
    args = parser.parse_args()
    rows = [covariance(method, dt) for method in ('suzuki_chin', 'pigs') for dt in (.08, .04, .02, .01)]
    write_csv(args.output/'stationary_integrator_covariance.csv', rows)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
