"""Independent representative sums, image geometry and signed derivatives."""
import itertools
import math

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import logsumexp

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.pimd_methods import BosonicExchangePotential, bosonic_exchange
from pyrpmd.potentials.base import PESResult, PotentialError
from pyrpmd.potentials.builtin import FreePotential, HarmonicPotential


def representative_reference(q, mass, beta, cell=None, pbc=(True, True, True)):
    """Enumerate 2**(N-1) compositions and each explicit quadratic edge."""
    p, n, _ = q.shape
    coefficient = mass * (p / beta) ** 2

    def compositions(stop):
        if not stop:
            yield [], 0.0
        for start in range(stop):
            for blocks, weight in compositions(start):
                yield blocks + [(start, stop)], weight - math.log(stop)

    def displacement(a, b):
        delta = q[a] - q[b]
        if cell is None:
            return delta
        # Independent bounded image enumeration for the modest test cells.
        ranges = [range(-3, 4) if flag else (0,) for flag in pbc]
        images = np.asarray(list(itertools.product(*ranges))) @ cell
        candidates = delta - images
        return candidates[np.argmin(np.sum(candidates**2, axis=1))]

    energies, forces, virials, parity, logweights = [], [], [], [], []
    for blocks, logweight in compositions(n):
        energy, force, virial = 0.0, np.zeros_like(q), np.zeros((3, 3))
        for start, stop in blocks:
            nodes = [(bead, atom) for atom in range(start, stop) for bead in range(p)]
            for a, b in zip(nodes, nodes[1:] + nodes[:1]):
                delta = displacement(a, b)
                energy += 0.5 * coefficient * np.dot(delta, delta)
                force[a] -= coefficient * delta
                force[b] += coefficient * delta
                virial -= coefficient * np.outer(delta, delta)
        energies.append(energy)
        forces.append(force)
        virials.append(virial)
        parity.append((-1.0) ** (n - len(blocks)))
        logweights.append(logweight - beta / p * energy)
    logz = logsumexp(logweights)
    weight = np.exp(np.asarray(logweights) - logz)
    return {
        "energy": -p / beta * logz,
        "forces": np.einsum("i,ipnc->pnc", weight, forces),
        "virial": np.einsum("i,ijk->jk", weight, virials),
        "spring_energy": float(weight @ energies),
        "fermionic_sign": float(weight @ parity),
        "fermionic_spring_energy_numerator": float((weight * parity) @ energies),
    }


def numerical_force(function, q, h=1e-6):
    result = np.empty_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += h
        minus[index] -= h
        result[index] = -(function(plus) - function(minus)) / (2 * h)
    return result


@pytest.mark.parametrize("beads", [1, 2, 5])
@pytest.mark.parametrize("particles", [1, 2, 3, 6])
def test_signed_quadratic_recurrence_matches_exponential_reference(beads, particles):
    q = np.random.default_rng(beads * 100 + particles).normal(size=(beads, particles, 3))
    actual = bosonic_exchange(q, 1.3, 2.8, statistics="fermionic")
    expected = representative_reference(q, 1.3, 2.8)
    for key, value in expected.items():
        np.testing.assert_allclose(getattr(actual, key), value, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(np.trace(actual.virial), -2 * actual.spring_energy)
    assert abs(actual.fermionic_sign) <= 1 + 1e-13


def test_large_common_actions_preserve_signed_weight_bound_without_clipping():
    # Subtracting the O(10^4) log partition values in the conditional
    # normalization previously produced sign=-1.0000000000000873 here,
    # causing the workflow to reject its own otherwise valid observations.
    q = np.random.default_rng(37).normal(size=(8, 32, 3)) * 10
    sampled = bosonic_exchange(q, 1.0, 3.0)
    signed = bosonic_exchange(q, 1.0, 3.0, statistics="fermionic")
    assert abs(signed.fermionic_sign) <= 1 + 4 * np.finfo(float).eps
    assert signed.energy == sampled.energy
    np.testing.assert_array_equal(signed.forces, sampled.forces)
    np.testing.assert_array_equal(signed.connections, sampled.connections)


@pytest.mark.parametrize("pbc", [(True, True, True), (True, False, True)])
def test_periodic_images_force_virial_and_all_lattice_translations(pbc):
    cell = np.array([[3.0, 0.0, 0.0], [1.1, 2.8, 0.0], [0.3, 0.4, 3.2]])
    q = np.random.default_rng(251).uniform(-0.9, 0.9, size=(3, 3, 3))
    kwargs = dict(cell=cell, pbc=pbc, periodic_mode="minimum_image", statistics="fermionic")
    actual = bosonic_exchange(q, 1.2, 3.4, **kwargs)
    expected = representative_reference(q, 1.2, 3.4, cell, pbc)
    for key, value in expected.items():
        np.testing.assert_allclose(getattr(actual, key), value, rtol=2e-12, atol=2e-12)
    force = numerical_force(lambda x: bosonic_exchange(x, 1.2, 3.4, **kwargs).energy, q)
    np.testing.assert_allclose(actual.forces, force, rtol=2e-7, atol=2e-8)
    integers = np.random.default_rng(59).integers(-3, 4, size=q.shape)
    integers[..., np.logical_not(pbc)] = 0
    shifted = bosonic_exchange(q + integers @ cell, 1.2, 3.4, **kwargs)
    for key in expected:
        np.testing.assert_allclose(getattr(shifted, key), getattr(actual, key), rtol=2e-12, atol=2e-12)
    derivative = np.zeros((3, 3))
    for a, b in np.ndindex(3, 3):
        strain = np.zeros((3, 3))
        strain[a, b] = 1e-6
        plus = bosonic_exchange(q @ (np.eye(3) + strain), 1.2, 3.4,
            **{**kwargs, "cell": cell @ (np.eye(3) + strain)}).energy
        minus = bosonic_exchange(q @ (np.eye(3) - strain), 1.2, 3.4,
            **{**kwargs, "cell": cell @ (np.eye(3) - strain)}).energy
        derivative[a, b] = -(plus - minus) / 2e-6
    np.testing.assert_allclose(actual.virial, derivative, rtol=2e-7, atol=2e-8)


def test_wrapper_cancels_raw_springs_and_includes_distinguishable_particles():
    cell = np.diag([3.0, 4.0, 5.0])
    masses, beta = np.array([1.2, 1.2, 2.4]), 3.0
    q = np.random.default_rng(812).normal(size=(3, 3, 3))
    potential = BosonicExchangePotential(FreePotential(), masses, beta, 3,
        groups=[[0, 1]], cell=cell, periodic_mode="minimum_image")
    dynamics = Dynamics(potential, masses, beta, 3, 0.001, native=False)
    group = representative_reference(q[:, :2], masses[0], beta, cell)
    unlisted = representative_reference(q[:, 2:], masses[2], beta, cell)
    actual = potential.evaluate(q)
    total_force = actual.forces + dynamics.integrator.spring_force(q)
    expected_force = np.concatenate((group["forces"], unlisted["forces"]), axis=1)
    np.testing.assert_allclose(total_force, expected_force, rtol=2e-13, atol=2e-13)
    assert dynamics.energy(State(q, np.zeros_like(q))) == pytest.approx(group["energy"] + unlisted["energy"])
    shifted_q = q.copy()
    shifted_q[1, 2] += cell[0]
    shifted = potential.evaluate(shifted_q)
    np.testing.assert_allclose(shifted.forces + dynamics.integrator.spring_force(shifted_q),
                               total_force, atol=1e-13)
    assert dynamics.energy(State(shifted_q, np.zeros_like(q))) == pytest.approx(group["energy"] + unlisted["energy"])
    # The PES wrapper virial is the correction only: raw rings are subtracted.
    delta = q - np.roll(q, -1, axis=0)
    raw_virial = -(3 / beta)**2 * np.einsum("n,pna,pnb->ab", masses, delta, delta)
    np.testing.assert_allclose(actual.virial.sum(axis=0) + raw_virial,
                               group["virial"] + unlisted["virial"], atol=1e-13)
    derivative = np.zeros((3, 3))
    for a, b in np.ndindex(3, 3):
        strain = np.zeros((3, 3))
        strain[a, b] = 1e-6
        energies = []
        for direction in (1, -1):
            transform = np.eye(3) + direction * strain
            changed = BosonicExchangePotential(FreePotential(), masses, beta, 3,
                groups=[[0, 1]], cell=cell @ transform, periodic_mode="minimum_image")
            energies.append(changed.evaluate(q @ transform).energy.sum())
        derivative[a, b] = -(energies[0] - energies[1]) / 2e-6
    np.testing.assert_allclose(actual.virial.sum(axis=0), derivative, rtol=2e-7, atol=2e-8)


def test_signed_temperature_derivative_and_nontrivial_node():
    rng = np.random.default_rng(1302)
    for _ in range(7):
        q = rng.normal(size=(2, 3, 3))
    beta = brentq(lambda b: representative_reference(q, 1.0, b)["fermionic_sign"], 0.2, 5.0)
    model = BosonicExchangePotential(HarmonicPotential(k=0.4), [1.0] * 3, beta, 2,
                                    statistics="fermionic")
    result = model.estimators(q)
    assert abs(result["sign"]) < 3e-13
    expected = representative_reference(q, 1.0, beta)
    assert abs(expected["fermionic_spring_energy_numerator"]) > 0.4
    assert result["internal_energy_numerator_hartree"] == pytest.approx(
        -expected["fermionic_spring_energy_numerator"] / 2, abs=2e-12)

    # Differentiate the complete signed finite-P density relative to its
    # positive bosonic value at beta, including the free normalization.
    v = float(model.potential.evaluate(q).energy.mean())
    reference_log = -beta / 2 * expected["energy"] - beta * v - 9 * math.log(beta)

    def relative_signed_density(b):
        terms = representative_reference(q, 1.0, b)
        log_weight = -b / 2 * terms["energy"] - b * v - 9 * math.log(b)
        return math.exp(log_weight - reference_log) * terms["fermionic_sign"]

    h = 2e-6
    derivative = -(relative_signed_density(beta + h) - relative_signed_density(beta - h)) / (2 * h)
    assert result["internal_energy_numerator_hartree"] == pytest.approx(derivative, rel=1e-7, abs=1e-9)


def test_multiple_fermionic_groups_and_distinguishable_energy():
    q = np.random.default_rng(671).normal(size=(3, 6, 3))
    masses = [1.1] * 3 + [1.8] * 2 + [2.7]
    beta = 4.3
    model = BosonicExchangePotential(HarmonicPotential(k=0.2), masses, beta, 3,
                                    groups=[[0, 1, 2], [3, 4]], statistics="fermionic")
    a = representative_reference(q[:, :3], 1.1, beta)
    b = representative_reference(q[:, 3:5], 1.8, beta)
    c = representative_reference(q[:, 5:], 2.7, beta)
    sign = a["fermionic_sign"] * b["fermionic_sign"]
    spring = (a["fermionic_spring_energy_numerator"] * b["fermionic_sign"]
              + b["fermionic_spring_energy_numerator"] * a["fermionic_sign"]
              + c["spring_energy"] * sign)
    physical = model.potential.evaluate(q)
    result = model.estimators(q, physical_result=physical)
    assert result["sign"] == pytest.approx(sign, abs=1e-14)
    assert result["kinetic_energy_numerator_hartree"] == pytest.approx(27 / beta * sign - spring / 3)
    assert result["potential_energy_numerator_hartree"] == pytest.approx(sign * physical.energy.mean())
    assert result["virial_internal_energy_numerator_hartree"] == pytest.approx(2 * sign * physical.energy.mean())


def test_periodic_primitive_pressure_signed_numerator_is_volume_derivative():
    cell, beta = np.diag([4.0, 4.2, 4.4]), 3.1
    q = np.random.default_rng(33).uniform(-1.2, 1.2, size=(3, 3, 3))
    model = BosonicExchangePotential(FreePotential(), [1.2] * 3, beta, 3,
        statistics="fermionic", cell=cell, periodic_mode="minimum_image")
    volume = np.linalg.det(cell)
    result = model.estimators(q, volume=volume, require_pressure=True)
    reference = representative_reference(q, 1.2, beta, cell)
    base_log = -beta / 3 * reference["energy"]

    def scaled_density(log_volume_scale):
        scale = math.exp(log_volume_scale / 3)
        other = representative_reference(q * scale, 1.2, beta, cell * scale)
        log_weight = -beta / 3 * other["energy"] + 9 * log_volume_scale
        return math.exp(log_weight - base_log) * other["fermionic_sign"]

    h = 1e-6
    pressure = (scaled_density(h) - scaled_density(-h)) / (2 * h * beta * volume)
    assert result["pressure_numerator_au"] == pytest.approx(pressure, rel=2e-7, abs=1e-10)


def test_pressure_never_fabricates_missing_physical_virial_and_rejects_npt():
    model = BosonicExchangePotential(HarmonicPotential(), [1.0, 1.0], 2.0, 2,
        cell=np.eye(3) * 4, periodic_mode="minimum_image")
    q = np.zeros((2, 2, 3))
    assert model.evaluate(q).virial is None
    assert "pressure_numerator_au" not in model.estimators(q, volume=64)
    with pytest.raises(PotentialError, match="physical configurational virials"):
        model.estimators(q, volume=64, require_pressure=True)
    with pytest.raises(PotentialError, match="centroid-only BZP"):
        model.set_cell(np.eye(3) * 5)
    with pytest.raises(ValueError, match="volume disagrees"):
        model.estimators(q, volume=65)


@pytest.mark.parametrize("kwargs,match", [
    ({"cell": np.eye(3)}, "explicit periodic_mode"),
    ({"periodic_mode": "minimum_image"}, "requires periodic"),
    ({"periodic_mode": "winding"}, "winding sums"),
    ({"cell": np.eye(3), "periodic_mode": np.array(["minimum_image"])}, "periodic_mode"),
    ({"cell": np.eye(3), "periodic_mode": np.array("minimum_image")}, "periodic_mode"),
    ({"statistics": "boltzmann"}, "statistics"),
    ({"pbc": [True, False, False]}, "require a cell"),
])
def test_invalid_statistics_and_periodic_combinations_fail_explicitly(kwargs, match):
    with pytest.raises(ValueError, match=match):
        bosonic_exchange(np.zeros((2, 2, 3)), 1.0, 2.0, **kwargs)


def test_physical_result_can_be_reused_without_evaluation():
    class Once:
        def evaluate(self, q):
            pytest.fail("an explicitly supplied physical result must be reused")

    q = np.zeros((2, 2, 3))
    model = BosonicExchangePotential(Once(), [1.0, 1.0], 2.0, 2)
    result = model.estimators(q, physical_result=PESResult(np.zeros(2), np.zeros_like(q)))
    assert result["sign"] == 1
    assert result["internal_energy_numerator_hartree"] == 3
