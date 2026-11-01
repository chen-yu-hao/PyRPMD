"""Independent force, recurrence and Gaussian-action checks for PI engines."""
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.pimd_methods import (SuzukiChinPotential, BosonicExchangePotential,
    bosonic_exchange, PIGSPotential, GaussianTrialWavefunction)
from pyrpmd.potentials.base import PESResult
from pyrpmd.potentials.builtin import HarmonicPotential, FreePotential


def numerical_force(energy, q, h=1e-5):
    force = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += h
        minus[index] -= h
        force[index] = -(energy(plus) - energy(minus)) / (2 * h)
    return force


class Quartic:
    def evaluate(self, q):
        return PESResult(np.sum(0.15*q**2+0.03*q**4, axis=(1,2)), -0.3*q-0.12*q**3)


@pytest.mark.parametrize('alpha', [0.0, 0.5, 1.0])
def test_suzuki_chin_force_is_gradient_of_modified_energy(alpha):
    q = np.random.default_rng(2).normal(size=(4, 2, 3))
    potential = SuzukiChinPotential(Quartic(), [2., 3.], 3., 4, alpha=alpha, fd_step=1e-5)
    expected = numerical_force(lambda x: potential.evaluate(x).energy.sum(), q)
    np.testing.assert_allclose(potential.evaluate(q).forces, expected, rtol=2e-7, atol=2e-8)


def test_suzuki_chin_even_bead_variance_is_fourth_order():
    # Exact Gaussian integral of the action; no Monte Carlo convergence noise.
    beta, mass, frequency = 4., 2., 0.7
    k = mass * frequency**2
    exact = 1 / (2 * mass * frequency * np.tanh(beta * frequency / 2))
    errors = []
    for p in (8, 16, 32):
        model = SuzukiChinPotential(HarmonicPotential(k=k), [mass], beta, p)
        laplacian = 2*np.eye(p)-np.roll(np.eye(p), 1, axis=0)-np.roll(np.eye(p), -1, axis=0)
        stiffness = mass*(p/beta)**2*laplacian + np.diag(model.weights*k + 2*model.coefficients*k*k/mass)
        variance = np.diag(np.linalg.inv(stiffness) * p / beta)[::2].mean()
        errors.append(abs(variance-exact))
    assert errors[0]/errors[1] > 12
    assert errors[1]/errors[2] > 14


def representative_terms(q, mass, beta):
    """Exponential reference: enumerate compositions and explicit cycle springs."""
    p, n, _ = q.shape
    coefficient = mass * (p/beta)**2
    def partitions(stop):
        if stop == 0:
            yield [], 0.
        else:
            for start in range(stop):
                for before, logweight in partitions(start):
                    yield before + [(start, stop)], logweight - math.log(stop)
    energies, forces, weights = [], [], []
    for blocks, logweight in partitions(n):
        energy, force = 0., np.zeros_like(q)
        for start, stop in blocks:
            nodes = [(bead, atom) for atom in range(start, stop) for bead in range(p)]
            for a, b in zip(nodes, nodes[1:]+nodes[:1]):
                delta = q[a]-q[b]
                energy += 0.5*coefficient*np.dot(delta,delta)
                force[a] -= coefficient*delta
                force[b] += coefficient*delta
        energies.append(energy)
        forces.append(force)
        weights.append(logweight-beta/p*energy)
    logz = logsumexp(weights)
    expected_force = np.einsum('i,ipnc->pnc', np.exp(np.array(weights)-logz), forces)
    return -p/beta*logz, expected_force


@pytest.mark.parametrize('beads', [1, 2, 5])
@pytest.mark.parametrize('particles', [1, 2, 4, 6])
def test_quadratic_bosonic_algorithm_matches_enumerated_representatives(beads, particles):
    q = np.random.default_rng(beads+particles).normal(size=(beads, particles, 3))
    result = bosonic_exchange(q, 2., 3.)
    energy, force = representative_terms(q, 2., 3.)
    assert result.energy == pytest.approx(energy, rel=2e-13, abs=2e-13)
    np.testing.assert_allclose(result.forces, force, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.connections.sum(axis=0), 1, atol=2e-12)
    np.testing.assert_allclose(result.connections.sum(axis=1), 1, atol=2e-12)


def test_bosonic_force_gradient_large_offsets_and_main_engine():
    q = np.random.default_rng(3).normal(size=(3, 3, 3))
    masses, beta = np.array([2., 2., 2.]), 2.5
    force = numerical_force(lambda x: bosonic_exchange(x, 2., beta).energy, q)
    np.testing.assert_allclose(bosonic_exchange(q, 2., beta).forces, force, rtol=2e-8, atol=2e-8)
    # main Dynamics adds springs exactly once to the exchange correction.
    potential = BosonicExchangePotential(FreePotential(), masses, beta, 3)
    dyn = Dynamics(potential, masses, beta, 3, 0.001, native=False)
    state = State(q, np.zeros_like(q))
    assert dyn.energy(state) == pytest.approx(bosonic_exchange(q, 2., beta).energy)
    total = potential.evaluate(q).forces + dyn.integrator.spring_force(q)
    np.testing.assert_allclose(total, force, atol=2e-8)
    assert np.isfinite(bosonic_exchange(q*1e3, 2., 0.01).forces).all()


def test_pigs_action_and_force_match_open_chain_definition():
    q = np.random.default_rng(4).normal(size=(7, 2, 3))
    masses = np.array([2., 3.])
    physical, trial = HarmonicPotential(k=0.4), GaussianTrialWavefunction([0.6,0.8])
    model = PIGSPotential(physical, masses, 7, 4., trial)
    logpsi = trial.evaluate(q[[0,-1]])[0]
    e = physical.evaluate(q).energy
    expected = (np.sum(masses[None,:,None]*np.diff(q,axis=0)**2)/(2*model.tau)
        + model.tau*(e.sum()-0.5*(e[0]+e[-1]))-logpsi.sum())
    assert model.action(q) == pytest.approx(expected)
    dyn = Dynamics(model, masses, model.beta, 7, 0.002, native=False)
    force = numerical_force(lambda x: dyn.energy(State(x, np.zeros_like(x))), q)
    np.testing.assert_allclose(model.evaluate(q).forces+dyn.integrator.spring_force(q), force, atol=2e-8)


def test_pigs_midpoint_converges_to_harmonic_ground_state():
    # Nonexact positive trial; independently integrate the finite open Gaussian.
    mass, omega, alpha, projection = 2., 0.7, 0.3, 14.
    target = 1/(2*mass*omega)
    errors = []
    for p in (33,65,129):
        model = PIGSPotential(HarmonicPotential(k=mass*omega**2), [mass], p, projection,
                              GaussianTrialWavefunction(alpha))
        laplacian = 2*np.eye(p)-np.roll(np.eye(p),1,axis=0)-np.roll(np.eye(p),-1,axis=0)
        laplacian[0,-1]=laplacian[-1,0]=0
        laplacian[0,0]=laplacian[-1,-1]=1
        matrix = mass/model.tau*laplacian + np.diag(model.tau*model.weights*mass*omega**2)
        matrix[0,0]+=alpha
        matrix[-1,-1]+=alpha
        variance = np.linalg.inv(matrix)[p//2,p//2]
        errors.append(abs(variance-target))
    assert errors[-1] < 5e-4
    assert errors[0] > 3*errors[1] > 9*errors[2]


def test_methods_reject_invalid_ensembles():
    with pytest.raises(ValueError, match='even'):
        SuzukiChinPotential(FreePotential(), [1.], 2., 3)
    with pytest.raises(ValueError, match='equal masses'):
        BosonicExchangePotential(FreePotential(), [1.,2.], 2., 4)
    with pytest.raises(ValueError, match='odd'):
        PIGSPotential(FreePotential(), [1.], 4, 2., GaussianTrialWavefunction(1.))
