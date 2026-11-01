"""Independent numerical and composition boundary checks for Tier 2 APIs."""

import numpy as np
import pytest

from pyrpmd.advanced_cv import PathCV, RMSD
from pyrpmd.bias import CompositeBias, MetadynamicsBias, WallBias
from pyrpmd.biased_kinetics import estimate_biased_rate, rescale_biased_time
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential
from pyrpmd.replica_exchange import ReplicaExchange, reduced_configurational_energy


@pytest.mark.parametrize("dtype", [np.uint64, ">u8", np.int64])
def test_biased_clock_rejects_backwards_integer_boundaries(dtype):
    with pytest.raises(ValueError, match="increasing"):
        rescale_biased_time(np.array([2, 1], dtype=dtype), [0.], beta=1,
                           transition_bias=0., transition_unbiased=True)


def test_biased_clock_uses_floating_durations_without_signed_overflow():
    limit = np.iinfo(np.int64).max
    result = rescale_biased_time(np.array([-limit, 0, limit]), [0., 0.], beta=1,
                                transition_bias=0., transition_unbiased=True)
    assert result["elapsed_biased_time_au"] == 2 * float(limit)
    assert result["rescaled_time_au"] == pytest.approx(2 * float(limit))
    assert result["acceleration_factor"] == pytest.approx(1.)


def test_biased_clock_rejects_intervals_lost_in_float_conversion():
    with pytest.raises(ValueError, match="representable"):
        rescale_biased_time(np.array([2**63, 2**63 + 1], dtype=np.uint64), [0.],
                           beta=1., transition_bias=0., transition_unbiased=True)


@pytest.mark.parametrize("dtype", [np.int64, ">i8", np.uint64])
def test_integer_exposures_cannot_wrap_into_negative_or_zero_rates(dtype):
    limit = np.iinfo(np.int64).max
    result = estimate_biased_rate(np.array([limit, limit, limit], dtype=dtype),
                                 events=[True, True, False], bootstrap=0)
    assert result["exposure_au"] == 3 * float(limit)
    assert result["rate_au"] == pytest.approx(2 / (3 * float(limit)), rel=1e-14, abs=0.)
    assert result["rate_s_inv"] > 0.
    assert result["mean_first_passage_au"] == pytest.approx(1.5 * float(limit))


def test_exposure_overflow_is_rejected_before_rate_estimation():
    with pytest.raises(ValueError, match="exposure.*representable"):
        estimate_biased_rate([np.finfo(float).max] * 2, bootstrap=0)


def _replicas(biases):
    dynamics = [Dynamics(FreePotential(), [1.], beta, 1, .01, native=False)
                for beta in (2., 3.)]
    states = [State(np.full((1, 1, 3), q), np.zeros((1, 1, 3))) for q in (.2, .4)]
    return dynamics, states, ReplicaExchange(dynamics, states, biases=biases)


def test_nested_composite_metadynamics_cannot_enter_equilibrium_exchange():
    cv = Cartesian(0)
    history = MetadynamicsBias(cv, [.1], .05, 2.)
    nested = CompositeBias(WallBias(cv, 2., upper=1.), CompositeBias(history))
    with pytest.raises(ValueError, match="time-independent"):
        _replicas([nested, None])


def test_static_compositions_are_supported_and_rechecked_after_mutation():
    cv = Cartesian(0)
    wall = WallBias(cv, 2., upper=.1)
    nested = CompositeBias(CompositeBias(wall))
    dynamics, states, exchange = _replicas([nested, nested])
    qi, qj = states[0].q, states[1].q
    # V(q)=2*(x-.1)^2 and beta_i=2, beta_j=3; P=1 and U=0.
    assert exchange.log_acceptance_ratio(0, 1) == pytest.approx((2 - 3) * 2 * (.1**2 - .3**2))
    nested.biases = (MetadynamicsBias(cv, [.1], .05, 2.),)
    with pytest.raises(ValueError, match="time-independent"):
        reduced_configurational_energy(dynamics[0], qi, nested)
    with pytest.raises(ValueError, match="time-independent"):
        exchange.attempt(0, 1)
    np.testing.assert_array_equal(exchange.states[0].q, qi)
    np.testing.assert_array_equal(exchange.states[1].q, qj)


def test_cyclic_bias_composition_is_rejected_before_energy_recursion():
    nested = CompositeBias()
    nested.biases = (nested,)
    with pytest.raises(ValueError, match="acyclic"):
        _replicas([nested, None])


@pytest.mark.parametrize("collapsed", [False, True])
def test_singular_alignment_rejects_branch_dependent_rmsd_and_path_forces(collapsed):
    reference = np.array([[-1., -1., 0.], [0., 1., 0.], [1., 0., 0.]])
    current = (np.zeros((3, 3)) if collapsed else
               np.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]]))
    with pytest.raises(ValueError, match="nonunique gradient"):
        RMSD(reference).squared_value_gradient(current)
    with pytest.raises(ValueError, match="nonunique gradient"):
        PathCV([reference, 2 * reference], path_lambda=2.).value_and_jacobian(current[None])


@pytest.mark.parametrize("reference", [
    np.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]]),
    np.full((3, 3), .7),
])
def test_low_rank_reference_symmetry_still_has_the_correct_unique_gradient(reference):
    current = np.array([[-1.1, .2, .5], [.3, .4, -.1], [.8, .1, .3]])
    cv = RMSD(reference)
    _, analytic = cv.squared_value_gradient(current)
    numerical = np.empty_like(current)
    for index in np.ndindex(current.shape):
        plus, minus = current.copy(), current.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (cv.squared_value_gradient(plus)[0]
                            - cv.squared_value_gradient(minus)[0]) / 2e-6
    np.testing.assert_allclose(analytic, numerical, atol=2e-9, rtol=1e-7)


def test_single_point_reference_squared_distance_remains_well_defined():
    value, gradient = RMSD([[1., 2., 3.]]).squared_value_gradient([[4., 8., 12.]])
    assert value == 0.
    np.testing.assert_array_equal(gradient, 0.)
