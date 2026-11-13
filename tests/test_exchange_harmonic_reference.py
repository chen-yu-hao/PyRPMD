"""Independent finite-P harmonic partition functions and exchange sampling.

The stochastic oracle draws *independent* Gaussian mixture configurations
from the recursive bosonic density.  It neither advances Dynamics nor uses
the production exchange recurrence to construct or select its samples.
These checks validate finite-P estimators, not MD mixing or P convergence.
"""

import itertools
import math

import numpy as np
import pytest

from pyrpmd.pimd_methods import BosonicExchangePotential, bosonic_exchange
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.quantum_statistics import exchange_thermodynamics


def _cycle_partition(beta, omega, beads, length, dimensions=3):
    """Trace of length*P primitive harmonic slices of duration beta/P.

    A length-k exchange cycle is Z_1(k*beta, k*P), not Z_1(k*beta, P).
    Its energy is the derivative with respect to the *original* beta.
    """
    x = beta * omega / (2 * beads)
    argument = length * beads * np.arcsinh(x)
    partition = (2 * np.sinh(argument)) ** (-dimensions)
    energy = (dimensions * length * omega
              / (2 * np.sqrt(1 + x * x) * np.tanh(argument)))
    return partition, energy


def _canonical_reference(particles, beta, omega, beads, statistics):
    """Closed N=2,3 cycle-index polynomials, without the production recurrence."""
    parity = 1 if statistics == "bosonic" else -1
    z1, e1 = _cycle_partition(beta, omega, beads, 1)
    z2, e2 = _cycle_partition(beta, omega, beads, 2)
    if particles == 2:
        partition = (z1 * z1 + parity * z2) / 2
        numerator = (2 * e1 * z1 * z1 + parity * e2 * z2) / 2
    elif particles == 3:
        z3, e3 = _cycle_partition(beta, omega, beads, 3)
        partition = (z1**3 + parity * 3 * z1 * z2 + 2 * z3) / 6
        numerator = (3 * e1 * z1**3
                     + parity * 3 * (e1 + e2) * z1 * z2
                     + 2 * e3 * z3) / 6
    else:
        raise ValueError("this independent reference is for two or three particles")
    return partition, numerator / partition


def _permutation_gaussian_reference(particles, beta, mass, omega, beads, statistics):
    """Integrate each of all N! explicit spring matrices by its determinant."""
    nodes = particles * beads
    spring_coefficient = mass * beads / beta
    physical_coefficient = beta * mass * omega**2 / beads
    partitions, energy_numerators = [], []
    for permutation in itertools.permutations(range(particles)):
        laplacian = np.zeros((nodes, nodes))
        for atom in range(particles):
            for bead in range(beads):
                first = atom * beads + bead
                second = (first + 1 if bead + 1 < beads
                          else permutation[atom] * beads)
                difference = np.zeros(nodes)
                difference[first] += 1
                difference[second] -= 1
                laplacian += np.outer(difference, difference)
        precision = spring_coefficient * laplacian + physical_coefficient * np.eye(nodes)
        determinant_sign, log_determinant = np.linalg.slogdet(precision)
        assert determinant_sign == 1
        # Three Cartesian axes and the free-particle kernel normalization.
        weight = np.exp(1.5 * (nodes * np.log(spring_coefficient) - log_determinant))
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(particles) for j in range(i + 1, particles))
        if statistics == "fermionic":
            weight *= (-1) ** inversions
        derivative = (-spring_coefficient * laplacian
                      + physical_coefficient * np.eye(nodes)) / beta
        energy = 1.5 * (nodes / beta + np.trace(np.linalg.solve(precision, derivative)))
        partitions.append(weight)
        energy_numerators.append(weight * energy)
    partition = math.fsum(partitions) / math.factorial(particles)
    energy = math.fsum(energy_numerators) / math.fsum(partitions)
    return partition, energy


def _independent_bosonic_paths(particles, beta, mass, omega, beads, samples, seed):
    """Sample contiguous-cycle Gaussian components with their integrated weights.

    The recursive representative density is not the pointwise factorial sum
    when N >= 3.  The component coefficient is the inverse product of block
    endpoints; for N=3 it is 1/6, 1/6, 1/3, 1/3 for the four compositions.
    """
    rng = np.random.default_rng(seed)
    components, component_weights = [], []
    covariance_factors = {}
    for length in range(1, particles + 1):
        nodes = length * beads
        eye = np.eye(nodes)
        laplacian = 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
        precision = (mass * beads / beta * laplacian
                     + beta * mass * omega**2 / beads * eye)
        eigenvalues, eigenvectors = np.linalg.eigh(precision)
        covariance_factors[length] = eigenvectors / np.sqrt(eigenvalues)[None, :]
    for cuts in itertools.product((False, True), repeat=particles - 1):
        ends = [i + 1 for i, cut in enumerate(cuts) if cut] + [particles]
        starts = [0] + ends[:-1]
        blocks = list(zip(starts, ends))
        weight = math.prod(_cycle_partition(beta, omega, beads, stop - start)[0] / stop
                           for start, stop in blocks)
        components.append(blocks)
        component_weights.append(weight)
    probabilities = np.array(component_weights) / math.fsum(component_weights)
    selected = rng.choice(len(components), samples, p=probabilities)
    paths = np.empty((samples, beads, particles, 3))
    for component, blocks in enumerate(components):
        indices = np.flatnonzero(selected == component)
        for start, stop in blocks:
            length = stop - start
            noise = rng.normal(size=(len(indices), length * beads, 3))
            draws = np.einsum("ij,sjc->sic", covariance_factors[length], noise)
            paths[indices, :, start:stop] = draws.reshape(
                len(indices), length, beads, 3).transpose(0, 2, 1, 3)
    return paths


def _mean_and_error(values):
    values = np.asarray(values)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


@pytest.mark.parametrize("particles", [2, 3])
@pytest.mark.parametrize("beads", [1, 2, 4])
@pytest.mark.parametrize("statistics", ["bosonic", "fermionic"])
def test_finite_p_closed_form_matches_all_permutation_gaussian_integrals(
        particles, beads, statistics):
    beta, mass, omega = 2.4, 1.7, 0.8
    expected = _permutation_gaussian_reference(
        particles, beta, mass, omega, beads, statistics)
    actual = _canonical_reference(particles, beta, omega, beads, statistics)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("particles", [2, 3])
def test_exchange_energies_and_beta_dependent_sign_match_finite_p_harmonic_trace(particles):
    # Fifteen thousand IID paths over both particle counts and all temperatures.
    # Primitive and virial estimators are checked independently; the tolerance
    # uses the measured IID error, retaining numerator/sign covariance.
    mass, omega, beads, samples = 1.7, 0.8, 4, 2500
    sampled_signs = []
    for index, beta in enumerate((0.6, 1.4, 2.4)):
        paths = _independent_bosonic_paths(
            particles, beta, mass, omega, beads, samples, seed=810 + particles * 10 + index)
        potential = 0.5 * mass * omega**2 * np.sum(paths**2, axis=(1, 2, 3)) / beads
        signs, bosonic_energy, signed_energy = [], [], []
        for path, physical in zip(paths, potential):
            exchange = bosonic_exchange(path, mass, beta, statistics="fermionic")
            signs.append(exchange.fermionic_sign)
            normalization = 1.5 * particles * beads / beta
            bosonic_energy.append(normalization + physical - exchange.spring_energy / beads)
            signed_energy.append((normalization + physical) * exchange.fermionic_sign
                                 - exchange.fermionic_spring_energy_numerator / beads)
        signs = np.asarray(signs)
        signed_energy = np.asarray(signed_energy)
        z_b, energy_b = _canonical_reference(particles, beta, omega, beads, "bosonic")
        z_f, energy_f = _canonical_reference(particles, beta, omega, beads, "fermionic")
        sign, sign_error = _mean_and_error(signs)
        sampled_signs.append(sign)
        assert sign == pytest.approx(z_f / z_b, abs=5 * sign_error)
        bosonic_mean, bosonic_error = _mean_and_error(bosonic_energy)
        assert bosonic_mean == pytest.approx(energy_b, abs=5 * bosonic_error)
        fermionic_mean = signed_energy.mean() / sign
        _, numerator_error = _mean_and_error(signed_energy - fermionic_mean * signs)
        assert fermionic_mean == pytest.approx(energy_f, abs=5 * numerator_error / sign)
        # For a harmonic trap the unshifted virial total energy is twice V.
        virial_b, virial_b_error = _mean_and_error(2 * potential)
        assert virial_b == pytest.approx(energy_b, abs=5 * virial_b_error)
        virial_f = np.mean(2 * potential * signs) / sign
        _, virial_f_error = _mean_and_error((2 * potential - virial_f) * signs)
        assert virial_f == pytest.approx(energy_f, abs=5 * virial_f_error / sign)
        reported = exchange_thermodynamics({
            "sign": signs,
            "potential_energy_numerator_hartree": potential * signs,
            "kinetic_energy_numerator_hartree": signed_energy - potential * signs,
            "internal_energy_numerator_hartree": signed_energy,
            "virial_internal_energy_numerator_hartree": 2 * potential * signs,
        }, block_size=1)
        assert reported["average_sign"] == pytest.approx(sign)
        assert reported["average_sign_standard_error"] == pytest.approx(sign_error)
        assert reported["internal_energy_hartree"] == pytest.approx(
            energy_f, abs=5 * reported["internal_energy_standard_error_hartree"])
        assert reported["virial_internal_energy_hartree"] == pytest.approx(
            energy_f, abs=5 * reported["virial_internal_energy_standard_error_hartree"])
        assert reported["sign_resolved"] is True
    assert sampled_signs[0] > sampled_signs[1] > sampled_signs[2] > 0


@pytest.mark.parametrize("statistics", ["bosonic", "fermionic"])
def test_harmonic_action_estimator_matches_independent_spring_numerator(statistics):
    particles, beads, beta, mass, omega = 3, 4, 1.4, 1.7, 0.8
    action = BosonicExchangePotential(
        HarmonicPotential(k=mass * omega**2), [mass] * particles, beta, beads,
        statistics=statistics)
    paths = _independent_bosonic_paths(particles, beta, mass, omega, beads, 5, seed=41)
    for path in paths:
        exchange = bosonic_exchange(path, mass, beta, statistics="fermionic")
        physical = 0.5 * mass * omega**2 * np.sum(path**2) / beads
        sign = exchange.fermionic_sign if statistics == "fermionic" else 1.0
        spring = (exchange.fermionic_spring_energy_numerator if statistics == "fermionic"
                  else exchange.spring_energy)
        expected = (1.5 * particles * beads / beta + physical) * sign - spring / beads
        estimated = action.estimators(path)
        assert estimated["sign"] == pytest.approx(sign, abs=2e-14)
        assert estimated["potential_energy_numerator_hartree"] == pytest.approx(physical * sign)
        assert estimated["internal_energy_numerator_hartree"] == pytest.approx(expected)
        assert estimated["kinetic_energy_numerator_hartree"] == pytest.approx(expected - physical * sign)
        assert estimated["virial_internal_energy_numerator_hartree"] == pytest.approx(2 * physical * sign)


def test_all_bead_mic_affine_virial_cannot_supply_centroid_bzp_pressure():
    """A bead-image change leaves the torus action invariant, not BZP pressure.

    After the ordinary ring subtraction cancels, the proposed exchange term
    in the existing centroid BZP numerator would be Tr(W)-sum(offset*F).
    Its image dependence demonstrates why merely exposing an affine virial
    is insufficient to remove the explicit exchange-NPT rejection.
    """
    q = np.random.default_rng(607).uniform(-0.8, 0.8, size=(3, 2, 3))
    cell = np.diag([4.0, 4.2, 4.4])
    shifted = q.copy()
    shifted[0, 0] += cell[0]
    before = bosonic_exchange(q, 1.3, 2.0, cell=cell, periodic_mode="minimum_image")
    after = bosonic_exchange(shifted, 1.3, 2.0, cell=cell, periodic_mode="minimum_image")
    assert before.energy == pytest.approx(after.energy, abs=1e-13)
    np.testing.assert_allclose(before.forces, after.forces, atol=2e-13)
    np.testing.assert_allclose(before.virial, after.virial, atol=2e-13)
    raw = np.trace(before.virial) - np.sum((q - q.mean(axis=0)) * before.forces)
    changed = np.trace(after.virial) - np.sum((shifted - shifted.mean(axis=0)) * after.forces)
    exact_change = np.dot(cell[0], before.forces[:, 0].mean(axis=0) - before.forces[0, 0])
    assert changed - raw == pytest.approx(exact_change, rel=2e-13, abs=2e-13)
    assert abs(changed - raw) > 0.1
