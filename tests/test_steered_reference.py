"""Two-dimensional Muller--Brown steering against quadrature and MBAR."""

import numpy as np
from scipy.special import logsumexp

from pyrpmd.analysis.mbar import solve_mbar
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.base import PESResult
from pyrpmd.steered import SteeredBias, SteeredMD, crooks


class MullerBrown:
    """Original four-Gaussian model, with energy scaled to atomic toy units."""

    def evaluate(self, q):
        x, y = q[:, 0, 0, None], q[:, 0, 1, None]
        dx, dy = x - [1, 0, -0.5, -1], y - [0, 0.5, 1.5, 1]
        a, b, c = np.array([-1, -1, -6.5, 0.7]), np.array([0, 0, 11, 0.6]), np.array([-10, -10, -6.5, 0.7])
        terms = np.array([-2, -1, -1.7, 0.15]) * np.exp(a * dx**2 + b * dx * dy + c * dy**2)
        forces = np.zeros_like(q)
        forces[:, 0, 0] = -np.sum(terms * (2 * a * dx + b * dy), axis=1)
        forces[:, 0, 1] = -np.sum(terms * (b * dx + 2 * c * dy), axis=1)
        forces[:, 0, 2] = -q[:, 0, 2]
        return PESResult(np.sum(terms, axis=1) + 0.5 * q[:, 0, 2]**2, forces)


def test_muller_brown_steering_matches_converged_quadrature_and_umbrella_mbar():
    beta, stiffness, samples = 1.1, 2.0, 300
    centers = np.array([[-0.5, 0.7], [0.2, 0.2]])
    provider = MullerBrown()
    norms = []
    for bins in (401, 801):
        axis_x, axis_y = np.linspace(-4, 3.5, bins), np.linspace(-3, 4.5, bins)
        points = np.stack(np.meshgrid(axis_x, axis_y, indexing="ij"), axis=-1).reshape(-1, 2)
        qgrid = np.zeros((len(points), 1, 3))
        qgrid[:, 0, :2] = points
        physical = provider.evaluate(qgrid).energy
        log_density = -beta * (physical[None] + 0.5 * stiffness *
                               np.sum((points[None] - centers[:, None])**2, axis=-1))
        log_z = logsumexp(log_density, axis=1)
        norms.append(-(log_z[1] - log_z[0]) / beta)
    assert abs(norms[1] - norms[0]) < 1e-9
    rng = np.random.default_rng(732)
    works, initial = [], []
    for direction in range(2):
        indices = rng.choice(len(points), size=samples, p=np.exp(log_density[direction] - log_z[direction]))
        positions = points[indices]
        initial.append(positions)
        values = []
        for point in positions:
            q = np.zeros((1, 1, 3))
            q[0, 0, :2] = point
            p = np.zeros_like(q)
            p[0, 0, :2] = rng.normal(size=2) / np.sqrt(beta)
            dynamics = Dynamics(provider, [1.0], beta, 1, 0.035, native=False)
            cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
            bias = SteeredBias(cv, np.linspace(centers[direction], centers[1 - direction], 6), stiffness)
            driver = SteeredMD(dynamics, State(q, p), bias, thermostat="none", initial_equilibrium=True)
            values.append(driver.run()["total_work"][-1])
        works.append(np.asarray(values))
    estimate = crooks(*works, beta, bootstrap=50, seed=142)
    assert abs(estimate.delta_free_energy - norms[-1]) < 4 * estimate.standard_error
    joined = np.concatenate(initial)
    reduced = beta * 0.5 * stiffness * np.sum((joined[None] - centers[:, None])**2, axis=-1)
    mbar = solve_mbar(reduced, [samples, samples])
    mbar_delta = (mbar.free_energies[1] - mbar.free_energies[0]) / beta
    assert abs(estimate.delta_free_energy - mbar_delta) < 4 * estimate.standard_error
    assert estimate.diagnostics["overlap_resolved"]
