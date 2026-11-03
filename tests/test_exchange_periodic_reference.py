"""Periodic exchange against reciprocal spectra and independent torus bridges.

The free torus heat kernel is exp(-tau*T), so its semigroup property makes
the N-particle trace exact at *every* P.  References here sum momentum states,
enumerate short representative compositions, or condition independent Wiener
increments.  None calls the production winding kernel to construct samples.

The periodic path domain follows Higer, Feldman and Hirshberg,
https://arxiv.org/abs/2501.17618 .  The fixed-N sign is Z_F/Z_B; see also
https://doi.org/10.1002/ctpp.70091 .  NPT volume references are independently
integrated occupation-number partition functions, including the V=0 boundary.
"""

from functools import lru_cache
import itertools
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import logsumexp

from pyrpmd.pimd_methods import BosonicExchangePotential, bosonic_exchange
from pyrpmd.potentials.builtin import FreePotential
from pyrpmd.quantum_statistics import exchange_thermodynamics


def _integer_vectors(cutoff):
    axis = np.arange(-cutoff, cutoff + 1)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _reciprocal_spectrum(cell, mass, cutoff=8):
    wavevectors = 2 * np.pi * _integer_vectors(cutoff) @ np.linalg.inv(cell).T
    return wavevectors, np.sum(wavevectors**2, axis=1) / (2 * mass)


def _cycle_trace(length, beta, mass, cell, cutoff=8):
    if np.array_equal(cell, np.diag(np.diag(cell))):
        partition, energy = 1.0, 0.0
        for side in np.diag(cell):
            gap = 2 * np.pi**2 / (mass * side**2)
            limit = max(cutoff, int(np.ceil(np.sqrt(45 / (length * beta * gap)))))
            squared = np.arange(-limit, limit + 1)**2
            weights = np.exp(-length * beta * gap * squared)
            partition *= weights.sum()
            energy += length * gap * np.dot(squared, weights) / weights.sum()
        return float(partition), float(energy)
    _, energies = _reciprocal_spectrum(cell, mass, cutoff)
    weights = np.exp(-length * beta * energies)
    partition = float(weights.sum())
    return partition, float(length * np.dot(energies, weights) / partition)


def _canonical_reference(particles, beta, mass, cell, statistics, cutoff=8):
    """Explicit N<=3 cycle-index polynomials in reciprocal one-body traces."""
    parity = 1 if statistics == "bosonic" else -1
    z1, e1 = _cycle_trace(1, beta, mass, cell, cutoff)
    if particles == 1:
        return z1, e1
    z2, e2 = _cycle_trace(2, beta, mass, cell, cutoff)
    if particles == 2:
        partition = (z1**2 + parity * z2) / 2
        numerator = (2 * e1 * z1**2 + parity * e2 * z2) / 2
    elif particles == 3:
        z3, e3 = _cycle_trace(3, beta, mass, cell, cutoff)
        partition = (z1**3 + 3 * parity * z1 * z2 + 2 * z3) / 6
        numerator = (3 * e1 * z1**3 + 3 * parity * (e1 + e2) * z1 * z2 + 2 * e3 * z3) / 6
    else:
        raise ValueError("the independent cycle-index reference covers N=1,2,3")
    return partition, numerator / partition


def _compositions(particles):
    for cuts in itertools.product((False, True), repeat=particles - 1):
        ends = [i + 1 for i, cut in enumerate(cuts) if cut] + [particles]
        yield list(zip([0] + ends[:-1], ends))


def _torus_bosonic_paths(particles, beta, mass, cell, beads, samples, seed):
    """IID bosonic representative paths from exact conditioned free bridges.

    For a k-cycle select its winding w with probability proportional to
    exp[-m|w H|^2/(2 k beta)], then condition kP independent increments on
    their sum w H.  Folding the bridge after construction sums all individual
    link images, whereas selecting a minimum-image link would change density.
    A cutoff of twelve standard deviations has negligible reference tail.
    """
    rng = np.random.default_rng(seed)
    cell = np.asarray(cell, dtype=float)
    inverse = np.linalg.inv(cell)
    components = list(_compositions(particles))
    trace = {k: _cycle_trace(k, beta, mass, cell)[0] for k in range(1, particles + 1)}
    weights = np.array([math.prod(trace[stop - start] / stop for start, stop in blocks)
                        for blocks in components])
    selected = rng.choice(len(components), samples, p=weights / weights.sum())
    paths = np.empty((samples, beads, particles, 3))
    smallest_length = np.linalg.svd(cell, compute_uv=False)[-1]
    for index, blocks in enumerate(components):
        ids = np.flatnonzero(selected == index)
        if not len(ids):
            continue
        for start, stop in blocks:
            length, nodes = stop - start, (stop - start) * beads
            if np.array_equal(cell, np.diag(np.diag(cell))):
                # Separate the three integer Gaussians; small NPT cells can
                # have very broad winding distributions without large grids.
                windings = np.empty((len(ids), 3))
                for axis, side in enumerate(np.diag(cell)):
                    cutoff = max(1, int(np.ceil(12 * np.sqrt(length * beta / mass) / side)))
                    values = np.arange(-cutoff, cutoff + 1) * side
                    probabilities = np.exp(-mass * values**2 / (2 * length * beta))
                    windings[:, axis] = rng.choice(values, len(ids), p=probabilities / probabilities.sum())
            else:
                cutoff = max(1, int(np.ceil(12 * np.sqrt(length * beta / mass) / smallest_length)))
                winding_vectors = _integer_vectors(cutoff) @ cell
                winding_weights = np.exp(-mass * np.sum(winding_vectors**2, axis=1) / (2 * length * beta))
                windings = winding_vectors[rng.choice(len(winding_vectors), len(ids),
                                                      p=winding_weights / winding_weights.sum())]
            noise = rng.normal(scale=np.sqrt(beta / (mass * beads)), size=(len(ids), nodes, 3))
            increments = noise - noise.mean(axis=1, keepdims=True) + windings[:, None] / nodes
            start_point = rng.random((len(ids), 1, 3)) @ cell
            positions = np.concatenate((start_point, start_point + np.cumsum(increments[:, :-1], axis=1)), axis=1)
            positions = np.remainder(positions @ inverse, 1.0) @ cell
            paths[ids, :, start:stop] = positions.reshape(len(ids), length, beads, 3).transpose(0, 2, 1, 3)
    return paths


def _reciprocal_kernel(displacement, tau, mass, cell, cutoff=10):
    """Fourier kernel, its energy derivative, and Cartesian log gradient."""
    vectors, energies = _reciprocal_spectrum(cell, mass, cutoff)
    phase = np.asarray(displacement) @ vectors.T
    weights = np.exp(-tau * energies)
    cosine = np.cos(phase)
    density = cosine @ weights
    assert np.all(density > 1e-10), "reference Fourier cancellation exceeds this test's budget"
    energy = (cosine @ (energies * weights)) / density
    gradient = -(np.sin(phase) * weights) @ vectors / density[..., None]
    normalization = 1.5 * np.log(mass / (2 * np.pi * tau))
    log_unnormalized = np.log(density) - np.log(abs(np.linalg.det(cell))) - normalization
    return log_unnormalized, energy, gradient


def _pointwise_reciprocal_reference(q, beta, mass, cell):
    """Enumerate at most four compositions, differentiating exact Fourier links."""
    beads, particles, _ = q.shape
    logs, parities, energies, forces = [], [], [], []
    for blocks in _compositions(particles):
        edges = []
        for start, stop in blocks:
            nodes = [(bead, atom) for atom in range(start, stop) for bead in range(beads)]
            edges.extend(zip(nodes, nodes[1:] + nodes[:1]))
        displacement = np.array([q[first] - q[second] for first, second in edges])
        link_log, link_energy, link_gradient = _reciprocal_kernel(displacement, beta / beads, mass, cell)
        logs.append(float(link_log.sum()) - sum(math.log(stop) for _, stop in blocks))
        parities.append((-1) ** (particles - len(blocks)))
        energies.append(float(link_energy.sum()) / beads)
        force = np.zeros_like(q)
        for (first, second), gradient in zip(edges, link_gradient):
            force[first] += beads / beta * gradient
            force[second] -= beads / beta * gradient
        forces.append(force)
    log_weight = float(logsumexp(logs))
    probabilities = np.exp(np.asarray(logs) - log_weight)
    parities = np.asarray(parities)
    return {"energy": -beads / beta * log_weight,
            "forces": np.einsum("s,spnc->pnc", probabilities, forces),
            "sign": float(probabilities @ parities),
            "bosonic_energy": float(probabilities @ energies),
            "fermionic_energy_numerator": float((probabilities * parities) @ energies)}


@lru_cache(maxsize=64)
def _cubic_shells(cutoff):
    squared = np.sum(_integer_vectors(cutoff)**2, axis=1)
    degeneracies = np.bincount(squared)
    return [(index, int(count)) for index, count in enumerate(degeneracies) if count]


def _cubic_occupation_partition(particles, beta, mass, volume, statistics):
    """Positive occupation-number polynomial; avoids fermion cancellation at V=0."""
    if volume == 0:
        return (1.0 if statistics == "bosonic" or particles == 1 else 0.0), 0.0
    energy_unit = 2 * np.pi**2 / (mass * volume**(2 / 3))
    cutoff = max(1, int(np.ceil(np.sqrt(45 / (beta * energy_unit)))))
    partitions, numerators = np.zeros(particles + 1), np.zeros(particles + 1)
    partitions[0] = 1.0
    for shell, degeneracy in _cubic_shells(cutoff):
        energy = energy_unit * shell
        occupancy = np.arange(particles + 1)
        coefficients = np.array([
            math.comb(degeneracy + number - 1, number) if statistics == "bosonic"
            else math.comb(degeneracy, number) if number <= degeneracy else 0
            for number in occupancy], dtype=float) * np.exp(-beta * energy * occupancy)
        previous, previous_numerator = partitions.copy(), numerators.copy()
        for number in range(particles + 1):
            indices = np.arange(number + 1)
            partitions[number] = np.dot(coefficients[:number + 1], previous[number - indices])
            numerators[number] = np.dot(coefficients[:number + 1],
                previous_numerator[number - indices] + indices * energy * previous[number - indices])
    return float(partitions[-1]), float(numerators[-1])


@lru_cache(maxsize=16)
def _volume_reference(particles, beta, mass, pressure):
    """Integrate Z_N(V) exp(-beta*p*V) dV with a momentum-occupation oracle.

    The dimensionless integration variable beta*p*V balances the moments.
    At V->0 the Bose partition tends to one, so its mean pressure is
    p-1/(beta*Delta), not p.  For N>=2 fermions that boundary vanishes.
    The universal relation is p*<V>-<V*P>=1/beta.
    """
    def integrand(t):
        volume = t / (beta * pressure)
        rows = []
        for statistics in ("bosonic", "fermionic"):
            z, numerator = _cubic_occupation_partition(particles, beta, mass, volume, statistics)
            rows += [z, t * z, t * t * z, beta * numerator,
                     0.0 if t == 0 else 2 * beta * numerator / (3 * t)]
        return np.exp(-t) * np.array(rows)

    # Z grows only polynomially (degree N<=3); t>80 contributes <1e-28 here.
    values, error = quad_vec(integrand, 0.0, 80.0, epsabs=1e-10, epsrel=1e-10)
    result = {"integration_error_bound": float(error)}
    for index, statistics in enumerate(("bosonic", "fermionic")):
        z, first, second, energy, instantaneous_pressure = values[5 * index:5 * index + 5]
        result[statistics] = {"partition_NPT": float(z / (beta * pressure)),
            "volume_bohr3": float(first / z / (beta * pressure)),
            "volume_squared_bohr6": float(second / z / (beta * pressure)**2),
            "internal_energy_hartree": float(energy / z / beta),
            "pressure_au": float(instantaneous_pressure / z * pressure)}
    result["average_sign"] = float(values[5] / values[0])
    return result


def _npt_bosonic_volumes(particles, beta, mass, pressure, samples, seed):
    """Exact rejection sampling using an analytic theta-function envelope.

    Monotonicity gives sum_n exp(-a*n*n) <= 1+sqrt(pi/a).  Consequently
    z_k(V) <= (1+c*V**(1/3)/sqrt(k))**3, c=sqrt(m/(2*pi*beta)).
    The positive Bose cycle-index polynomial gives a polynomial envelope in
    V**(1/3). Each term times exp(-beta*p*V) is an elementary gamma density.
    """
    coefficients = lambda k: np.array([1, 3 / np.sqrt(k), 3 / k, k**(-1.5)])
    first = coefficients(1)
    if particles == 1:
        envelope = first
    elif particles == 2:
        envelope = (np.polynomial.polynomial.polymul(first, first)
                    + np.pad(coefficients(2), (0, 3))) / 2
    elif particles == 3:
        cube = np.polynomial.polynomial.polypow(first, 3)
        product = np.polynomial.polynomial.polymul(first, coefficients(2))
        envelope = (cube + 3 * np.pad(product, (0, 3)) + 2 * np.pad(coefficients(3), (0, 6))) / 6
    else:
        raise ValueError("NPT reference sampler covers N=1,2,3")
    degrees = np.arange(len(envelope)) / 3
    envelope *= (mass / (2 * np.pi * beta))**(1.5 * degrees)
    rate = beta * pressure
    weights = envelope * np.array([math.gamma(1 + degree) for degree in degrees]) / rate**(1 + degrees)
    rng = np.random.default_rng(seed)
    accepted = []
    while len(accepted) < samples:
        component = rng.choice(len(weights), p=weights / weights.sum())
        volume = float(rng.gamma(1 + degrees[component], 1 / rate))
        # The positive Bose cycle polynomial needs only short 1-D momentum
        # sums.  Keep the slower occupation polynomial for the integral oracle,
        # where it avoids low-volume cancellation in the fermion partition.
        value = _canonical_reference(particles, beta, mass, np.eye(3) * np.cbrt(volume), "bosonic")[0]
        upper = float(envelope @ volume**degrees)
        assert value <= upper * (1 + 1e-13)
        if rng.random() * upper <= value:
            accepted.append(volume)
    return np.asarray(accepted)


@pytest.mark.parametrize("particles", [1, 2, 3])
@pytest.mark.parametrize("beads", [1, 2, 4])
def test_full_winding_pointwise_action_matches_independent_reciprocal_links(particles, beads):
    cell = np.array([[2.8, 0.0, 0.0], [0.4, 3.2, 0.0], [-0.2, 0.3, 3.6]])
    beta, mass = 2.1, 1.3
    q = np.random.default_rng(731 + particles + beads).normal(scale=0.18, size=(beads, particles, 3))
    expected = _pointwise_reciprocal_reference(q, beta, mass, cell)
    actual = bosonic_exchange(q, mass, beta, cell=cell, periodic_mode="full_winding",
                              statistics="fermionic", winding_tolerance=1e-12)
    assert actual.energy == pytest.approx(expected["energy"], abs=2e-10)
    np.testing.assert_allclose(actual.forces, expected["forces"], atol=3e-10, rtol=2e-10)
    assert actual.fermionic_sign == pytest.approx(expected["sign"], abs=2e-11)
    normalization = 1.5 * particles * beads / beta
    assert normalization - actual.spring_energy / beads == pytest.approx(expected["bosonic_energy"], abs=3e-10)
    assert (normalization * actual.fermionic_sign - actual.fermionic_spring_energy_numerator / beads
            == pytest.approx(expected["fermionic_energy_numerator"], abs=3e-10))


@pytest.mark.parametrize("particles", [1, 2, 3])
@pytest.mark.parametrize("statistics", ["bosonic", "fermionic"])
def test_canonical_cycle_index_matches_positive_occupation_polynomial(particles, statistics):
    beta, mass, length = 1.7, 1.3, 3.6
    partition, energy = _canonical_reference(particles, beta, mass, np.eye(3) * length, statistics)
    occupation, numerator = _cubic_occupation_partition(particles, beta, mass, length**3, statistics)
    assert partition == pytest.approx(occupation, rel=2e-13)
    assert energy == pytest.approx(numerator / occupation, rel=2e-13)


@pytest.mark.parametrize("particles", [1, 2])
def test_npt_reference_obeys_volume_identity_and_small_volume_pressure_boundary(particles):
    beta, mass, pressure = 2.0, 1.0, 0.02
    expected = _volume_reference(particles, beta, mass, pressure)
    for statistics in ("bosonic", "fermionic"):
        row = expected[statistics]
        assert pressure * row["volume_bohr3"] - 2 * row["internal_energy_hartree"] / 3 == pytest.approx(1 / beta, abs=2e-10)
        boundary = 1.0 if statistics == "bosonic" or particles == 1 else 0.0
        assert row["pressure_au"] == pytest.approx(pressure - boundary / (beta * row["partition_NPT"]), abs=2e-11)
    assert expected["bosonic"]["pressure_au"] < pressure


@pytest.mark.parametrize("particles", [1, 2, 3])
@pytest.mark.parametrize("beads", [1, 2, 4])
def test_torus_bridge_samples_give_p_independent_free_gas_energy_pressure_and_sign(particles, beads):
    beta, mass, samples = 1.7, 1.3, 500
    cell = np.diag([2.9, 3.6, 4.1])
    volume = float(np.linalg.det(cell))
    paths = _torus_bosonic_paths(particles, beta, mass, cell, beads, samples,
                                seed=9071 + 10 * particles + beads)
    z_b, e_b = _canonical_reference(particles, beta, mass, cell, "bosonic")
    z_f, e_f = _canonical_reference(particles, beta, mass, cell, "fermionic")
    signs, bosonic_energy, fermionic_energy = [], [], []
    for q in paths:
        exchange = bosonic_exchange(q, mass, beta, cell=cell, periodic_mode="full_winding",
                                    statistics="fermionic", winding_tolerance=1e-12)
        normalization = 1.5 * particles * beads / beta
        signs.append(exchange.fermionic_sign)
        bosonic_energy.append(normalization - exchange.spring_energy / beads)
        fermionic_energy.append(normalization * exchange.fermionic_sign
                                - exchange.fermionic_spring_energy_numerator / beads)
    signs, bosonic_energy, fermionic_energy = map(np.asarray, (signs, bosonic_energy, fermionic_energy))
    error = bosonic_energy.std(ddof=1) / np.sqrt(samples)
    assert bosonic_energy.mean() == pytest.approx(e_b, abs=5 * error + 2e-10)
    data = {"sign": signs, "potential_energy_numerator_hartree": np.zeros(samples),
            "kinetic_energy_numerator_hartree": fermionic_energy,
            "internal_energy_numerator_hartree": fermionic_energy,
            "pressure_numerator_au": 2 * fermionic_energy / (3 * volume)}
    reported = exchange_thermodynamics(data, block_size=1)
    assert reported["sign_resolved"]
    assert reported["average_sign"] == pytest.approx(z_f / z_b, abs=5 * reported["average_sign_standard_error"] + 2e-10)
    assert reported["internal_energy_hartree"] == pytest.approx(e_f, abs=5 * reported["internal_energy_standard_error_hartree"] + 2e-10)
    assert reported["pressure_au"] == pytest.approx(2 * e_f / (3 * volume), abs=5 * reported["pressure_standard_error_au"] + 2e-11)
    # Check the actual action-level pressure channel, not only the free-gas
    # relation used to reduce the larger reference sample loop's runtime.
    action = BosonicExchangePotential(FreePotential(), [mass] * particles, beta, beads,
        statistics="fermionic", cell=cell, periodic_mode="full_winding", winding_tolerance=1e-12)
    for q, sign, numerator in zip(paths[:3], signs[:3], fermionic_energy[:3]):
        row = action.estimators(q, volume=volume, require_pressure=True)
        assert row["sign"] == pytest.approx(sign, abs=2e-13)
        assert row["internal_energy_numerator_hartree"] == pytest.approx(numerator, abs=2e-13)
        assert row["pressure_numerator_au"] == pytest.approx(2 * numerator / (3 * volume), abs=2e-13)


def test_one_bead_small_periodic_cell_reaches_exact_zero_momentum_ground_state():
    beta, mass = 4.0, 1.0
    cell = np.eye(3) * 2.0
    q = np.array([[[0.3, 0.7, 1.1]]])
    action = BosonicExchangePotential(FreePotential(), [mass], beta, 1,
        cell=cell, periodic_mode="full_winding", winding_tolerance=1e-12)
    estimated = action.estimators(q, volume=8.0, require_pressure=True)
    partition, energy = _canonical_reference(1, beta, mass, cell, "bosonic")
    exchange = bosonic_exchange(q, mass, beta, cell=cell, periodic_mode="full_winding",
                                winding_tolerance=1e-12)
    reconstructed = 8.0 * (mass / (2 * np.pi * beta))**1.5 * np.exp(-beta * exchange.energy)
    assert reconstructed == pytest.approx(partition, rel=2e-11)
    assert estimated["internal_energy_numerator_hartree"] == pytest.approx(energy, abs=2e-11)
    assert estimated["pressure_numerator_au"] == pytest.approx(energy / 12, abs=2e-12)
    assert energy < 1e-6
    # MIC would give 3/(2 beta)=0.375 hartree at P=1 for this same free gas.
    assert abs(estimated["internal_energy_numerator_hartree"] - 1.5 / beta) > 0.3


def test_signed_npt_volume_energy_pressure_and_enthalpy_against_integrated_free_trace():
    particles, beads, beta, mass, pressure, samples = 2, 2, 2.0, 1.0, 0.02, 600
    expected = _volume_reference(particles, beta, mass, pressure)
    volumes = _npt_bosonic_volumes(particles, beta, mass, pressure, samples, seed=1192)
    rows = []
    for index, volume in enumerate(volumes):
        cell = np.eye(3) * np.cbrt(volume)
        q = _torus_bosonic_paths(particles, beta, mass, cell, beads, 1, seed=70114 + index)[0]
        action = BosonicExchangePotential(FreePotential(), [mass] * particles, beta, beads,
            statistics="fermionic", cell=cell, periodic_mode="full_winding", winding_tolerance=1e-12)
        row = action.estimators(q, volume=volume, require_pressure=True)
        row["volume_numerator_bohr3"] = row["sign"] * volume
        row["enthalpy_numerator_hartree"] = row["internal_energy_numerator_hartree"] + pressure * row["sign"] * volume
        rows.append(row)
    reported = exchange_thermodynamics(rows, block_size=1)
    assert reported["sign_resolved"]
    assert reported["average_sign"] == pytest.approx(expected["average_sign"], abs=5 * reported["average_sign_standard_error"])
    for name, error in (("volume_bohr3", "volume_standard_error_bohr3"),
                        ("internal_energy_hartree", "internal_energy_standard_error_hartree"),
                        ("pressure_au", "pressure_standard_error_au")):
        assert reported[name] == pytest.approx(expected["fermionic"][name], abs=5 * reported[error])
    enthalpy = expected["fermionic"]["internal_energy_hartree"] + pressure * expected["fermionic"]["volume_bohr3"]
    assert reported["enthalpy_hartree"] == pytest.approx(enthalpy, abs=5 * reported["enthalpy_standard_error_hartree"])
    error = volumes.std(ddof=1) / np.sqrt(samples)
    assert volumes.mean() == pytest.approx(expected["bosonic"]["volume_bohr3"], abs=5 * error)
    assert reported["volume_bohr3"] > 1.5 * volumes.mean()


@pytest.mark.parametrize("particles", [1, 2])
def test_all_bead_bzp_samples_finite_box_quantum_npt_volume_measure(
        particles, record_testsuite_property):
    """The NP coordinate Jacobian is tested with P=2, not a classical P=1 gas.

    The oracle predicts <V>=31.7608 (N=1) or 36.2748 (N=2); the classical
    Gamma limits would give 50 or 75.  Complete 40-au blocks give conditional
    errors for this finite seed, rather than a fixed percentage tolerance.
    The N=2 case also checks the sign from the actual bosonic NPT chain. This
    modest validation chain does not establish production convergence.
    """
    from pyrpmd.analysis.fes import statistical_inefficiency
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.exchange_barostat import ExchangeBZPBarostat
    from pyrpmd.system import System

    beads, beta, mass, pressure = 2, 2.0, 1.0, 0.02
    reference = _volume_reference(particles, beta, mass, pressure)
    expected = reference["bosonic"]
    cell = np.eye(3) * np.cbrt(expected["volume_bohr3"])
    masses = [mass] * particles
    system = System(("H",) * particles, masses, np.zeros((particles, 3)), cell=cell)
    action = BosonicExchangePotential(FreePotential(), masses, beta, beads,
        cell=cell, periodic_mode="full_winding", winding_tolerance=1e-12)
    dynamics = Dynamics(action, masses, beta, beads, 0.04, native=False,
        integrator="velocity-verlet", loop_backend="python", seed=35199)
    q = _torus_bosonic_paths(particles, beta, mass, cell, beads, 1, seed=21571)[0]
    state = State(q, dynamics.sample_momenta())
    piston = ExchangeBZPBarostat(action, system, beta, beads,
        pressure_au=pressure, tau_au=1.0, friction_au=1.0)
    rows, volumes, signs = [], [], []
    for step in range(24000):
        piston.step(state, dynamics, thermostat="andersen", andersen_interval=5)
        if step >= 2000 and (step + 1) % 10 == 0:
            volume = piston.volume
            row = action.estimators(state.q, volume=volume, require_pressure=True)
            row["volume_numerator_bohr3"] = volume
            row["enthalpy_numerator_hartree"] = row["internal_energy_numerator_hartree"] + pressure * volume
            rows.append(row)
            volumes.append(volume)
            if particles > 1:
                exchange = bosonic_exchange(state.q, mass, beta, cell=piston.cell,
                    periodic_mode="full_winding", statistics="fermionic", winding_tolerance=1e-12)
                signs.append(exchange.fermionic_sign)
    volumes = np.asarray(volumes)
    block_length = 100
    assert max(statistical_inefficiency(volumes), statistical_inefficiency(volumes**2)) < block_length
    report = exchange_thermodynamics(rows, block_size=block_length)
    for name, error in (("volume_bohr3", "volume_standard_error_bohr3"),
                        ("internal_energy_hartree", "internal_energy_standard_error_hartree"),
                        ("pressure_au", "pressure_standard_error_au")):
        assert report[name] == pytest.approx(expected[name], abs=5 * report[error])
    second_blocks = (volumes**2).reshape(-1, block_length).mean(axis=1)
    second_error = second_blocks.std(ddof=1) / np.sqrt(len(second_blocks))
    assert np.mean(volumes**2) == pytest.approx(expected["volume_squared_bohr6"], abs=5 * second_error)
    if particles > 1:
        signs = np.asarray(signs)
        assert statistical_inefficiency(signs) < block_length
        sign_blocks = signs.reshape(-1, block_length).mean(axis=1)
        sign_error = sign_blocks.std(ddof=1) / np.sqrt(len(sign_blocks))
        assert signs.mean() == pytest.approx(reference["average_sign"], abs=5 * sign_error)
    assert report["physical_converged"] is False
    measurements = {
        name: {"exact": expected[name], "mean": report[name], "standard_error": report[error]}
        for name, error in (("volume_bohr3", "volume_standard_error_bohr3"),
                            ("internal_energy_hartree", "internal_energy_standard_error_hartree"),
                            ("pressure_au", "pressure_standard_error_au"))}
    measurements["volume_squared_bohr6"] = {"exact": expected["volume_squared_bohr6"],
        "mean": float(np.mean(volumes**2)), "standard_error": float(second_error)}
    if particles > 1:
        measurements["average_sign"] = {"exact": reference["average_sign"],
            "mean": float(signs.mean()), "standard_error": float(sign_error)}
    record_testsuite_property(f"exchange_npt_N{particles}", json.dumps({
        "particles": particles, "beads": beads, "beta_au": beta, "mass_au": mass,
        "external_pressure_au": pressure, "steps": 24000, "equilibration_steps": 2000,
        "dt_au": 0.04, "sample_interval_steps": 10, "samples": len(rows),
        "seed": 35199, "block_size_frames": block_length,
        "block_duration_au": 40.0, "complete_blocks": len(second_blocks),
        "uncertainty_scope": "conditional_on_the_reported_blocks_and_finite_seed",
        "measurements": measurements}, sort_keys=True, allow_nan=False))
