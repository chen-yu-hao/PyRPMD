"""Harmonic nuclear parents checked with independent dense precision matrices."""

import copy

import numpy as np
import pytest

from pyrpmd.nonadiabatic_sampling import sample_harmonic_ring


class UnitCovarianceGenerator(np.random.Generator):
    """Deterministic zero-mean points test a Gaussian sampler's linear map.

    These are not Gaussian draws. Their exact unit second moment allows the
    position and momentum transforms to be checked without Monte Carlo noise.
    """

    def __init__(self):
        super().__init__(np.random.PCG64(0))

    def standard_normal(self, size=None, dtype=np.float64, out=None):
        dimension = int(np.prod(size[1:]))
        assert size[0] == 2*dimension
        values = np.sqrt(dimension)*np.concatenate((np.eye(dimension), -np.eye(dimension)))
        return values.reshape(size)


@pytest.mark.parametrize("beads", [1, 2, 5, 8])
def test_exact_covariance_matches_dense_periodic_stiffness_including_one_and_two_beads(beads):
    masses, beta, omega = np.array([2., 5.]), 3.7, np.array([.4, 1.3])
    centers, shifts = np.array([.6, -.2]), np.array([.1, .8])
    coordinates = np.array([0, 4])  # atom 0 x, atom 1 y; not mass-weighted
    count = 4*beads
    states = sample_harmonic_ring(masses, beta, beads, omega, coordinates=coordinates,
                                  centers=centers, centroid_momenta=shifts,
                                  count=count, seed=UnitCovarianceGenerator())
    q = np.array([state.q for state in states]).reshape(count, beads, -1)
    p = np.array([state.p for state in states]).reshape(count, beads, -1)
    laplacian = (2*np.eye(beads)-np.roll(np.eye(beads), 1, axis=0)
                 - np.roll(np.eye(beads), -1, axis=0))
    for mode, coordinate in enumerate(coordinates):
        delta = q[:, :, coordinate]-centers[mode]
        dp = p[:, :, coordinate]-shifts[mode]
        stiffness = masses[mode]*(omega[mode]**2*np.eye(beads)+(beads/beta)**2*laplacian)
        expected = (beads/beta)*np.linalg.inv(stiffness)
        np.testing.assert_allclose(delta.T@delta/count, expected, rtol=8e-14, atol=1e-14)
        np.testing.assert_allclose(dp.T@dp/count, masses[mode]*beads/beta*np.eye(beads), atol=4e-15)
        np.testing.assert_allclose(delta.mean(axis=0), 0., atol=3e-16)
        np.testing.assert_allclose(dp.mean(axis=0), 0., atol=3e-16)
        if beads == 1:
            assert expected[0, 0] == pytest.approx(1/(beta*masses[mode]*omega[mode]**2))
        if beads == 2:
            # P=2 has two links between the same beads, so its internal
            # Laplacian eigenvalue is 4, not 2.
            diagonal = (beads/beta)/(2*masses[mode])*(
                1/omega[mode]**2 + 1/(omega[mode]**2+4*(beads/beta)**2))
            assert expected[0, 0] == pytest.approx(diagonal)
    inactive = [1, 2, 3, 5]
    np.testing.assert_array_equal(q[:, :, inactive], 0.)
    np.testing.assert_array_equal(p[:, :, inactive], 0.)
    cross = (q[:, :, 0]-centers[0]).T@(q[:, :, 4]-centers[1])/count
    np.testing.assert_allclose(cross, 0., atol=1e-15)


def test_actual_gaussian_draws_have_expected_centroid_moments_and_no_position_momentum_correlation():
    mass, beta, omega, beads, count = 1.7, 2.3, .8, 4, 6000
    states = sample_harmonic_ring([mass], beta, beads, [omega], centers=[.4],
                                  centroid_momenta=[-.3], count=count, seed=9182)
    qc = np.array([state.q[:, 0, 0].mean() for state in states])
    pc = np.array([state.p[:, 0, 0].mean() for state in states])
    expected_q_variance = 1/(beta*mass*omega**2)
    expected_p_variance = mass/beta
    assert abs(qc.mean()-.4) < .035
    assert abs(pc.mean()+.3) < .035
    assert qc.var() == pytest.approx(expected_q_variance, rel=.055)
    assert pc.var() == pytest.approx(expected_p_variance, rel=.055)
    assert abs(np.corrcoef(qc, pc)[0, 1]) < .05
    assert abs(np.corrcoef(qc[:-1], qc[1:])[0, 1]) < .05


def test_default_coordinate_assignment_reproducibility_and_array_ownership():
    masses, omega, centers, shifts = ([2., 3.], [.6, 1.1], [.3, -.4], [.1, .2])
    inputs = [np.array(value) for value in (masses, omega, centers, shifts)]
    copies = [value.copy() for value in inputs]
    result = sample_harmonic_ring(inputs[0], 4., 3, inputs[1], centers=inputs[2],
                                  centroid_momenta=inputs[3], count=2, seed=12)
    repeated = sample_harmonic_ring(masses, 4., 3, omega, centers=centers,
                                    centroid_momenta=shifts, count=2, seed=12)
    assert isinstance(result, tuple) and len(result) == 2
    for state, duplicate in zip(result, repeated):
        np.testing.assert_array_equal(state.q, duplicate.q)
        np.testing.assert_array_equal(state.p, duplicate.p)
        np.testing.assert_array_equal(state.q[:, :, 1:], 0.)
        np.testing.assert_array_equal(state.p[:, :, 1:], 0.)
        assert state.q.shape == (3, 2, 3)
        assert state.step == 0 and state.time == 0.
        assert not np.shares_memory(state.q, state.p)
    assert not np.shares_memory(result[0].q, result[1].q)
    assert not np.shares_memory(result[0].p, result[1].p)
    result[0].q[:] = 41.
    result[0].p[:] = -5.
    np.testing.assert_array_equal(result[1].q, repeated[1].q)
    for given, before in zip(inputs, copies):
        np.testing.assert_array_equal(given, before)


def test_generator_advances_on_success_and_is_not_consumed_by_invalid_arguments():
    rng = np.random.default_rng(73)
    initial = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        sample_harmonic_ring([2.], 3., 4, [.5], centers=[np.nan], seed=rng)
    assert rng.bit_generator.state == initial
    first = sample_harmonic_ring([2.], 3., 4, [.5], seed=rng)[0]
    after = copy.deepcopy(rng.bit_generator.state)
    assert after != initial
    second = sample_harmonic_ring([2.], 3., 4, [.5], seed=rng)[0]
    assert not np.array_equal(first.q, second.q)
    rng.bit_generator.state = initial
    repeated = sample_harmonic_ring([2.], 3., 4, [.5], seed=rng)[0]
    np.testing.assert_array_equal(first.q, repeated.q)
    np.testing.assert_array_equal(first.p, repeated.p)


@pytest.mark.parametrize("keyword,value", [
    ("masses", []), ("masses", [0.]), ("masses", [True]),
    ("beta", 0.), ("beta", np.inf), ("beta", True),
    ("nbeads", 0), ("nbeads", 2.5), ("nbeads", True),
    ("frequencies", []), ("frequencies", [0.]), ("frequencies", [-.2]),
    ("frequencies", [np.nan]), ("frequencies", [1.+.1j]),
    ("coordinates", [3]), ("coordinates", [-1]), ("coordinates", [0.]),
    ("coordinates", [True]), ("coordinates", [0, 1]),
    ("centers", [np.nan]), ("centers", [0., 0.]),
    ("centroid_momenta", [[1.]]), ("centroid_momenta", [np.inf]),
    ("count", 0), ("count", 1.5), ("count", False),
    ("seed", -1), ("seed", True), ("seed", None),
])
def test_strict_parameter_validation(keyword, value):
    options = {"masses": [2.], "beta": 3., "nbeads": 4, "frequencies": [.5], keyword: value}
    with pytest.raises(ValueError):
        sample_harmonic_ring(**options)


def test_modes_cannot_share_a_cartesian_coordinate_and_default_needs_one_atom_per_mode():
    with pytest.raises(ValueError, match="distinct integer Cartesian"):
        sample_harmonic_ring([2.], 3., 4, [.5, .7], coordinates=[0, 0])
    with pytest.raises(ValueError, match="distinct integer Cartesian"):
        sample_harmonic_ring([2.], 3., 4, [.5, .7])
    states = sample_harmonic_ring([2.], 3., 4, [.5, .7], coordinates=[2, 0])
    np.testing.assert_array_equal(states[0].q[:, 0, 1], 0.)
    assert np.any(states[0].q[:, 0, 2] != 0)
