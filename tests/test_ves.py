"""Independent variational, derivative, boundary and restart checks for VES."""

import copy
import json

import numpy as np
import pytest
from scipy.special import logsumexp

from pyrpmd.cv import Cartesian, CVStack, Distance
from pyrpmd.ves import LegendreBasis, VESBias


def cv(dimension=1):
    return CVStack([Cartesian(0, axis) for axis in range(dimension)])


def test_tensor_basis_matches_explicit_polynomials_and_removes_only_constant():
    basis = LegendreBasis([[-2, 2], [-3, 1]], [2, 2])
    points = np.array([[0.3, -0.4], [-1.1, 0.7]])
    x, y = points[:, 0] / 2, (points[:, 1] + 1) / 2
    one = np.ones(2)
    px, py = [one, x, (3*x*x - 1)/2], [one, y, (3*y*y - 1)/2]
    expected = np.stack([px[i]*py[j] for i, j in basis.indices], axis=-1)
    assert basis.size == 8
    assert (1, 1) in map(tuple, basis.indices)
    np.testing.assert_allclose(basis.evaluate(points), expected, atol=2e-16)
    assert basis.evaluate(points[0]).shape == (8,)


@pytest.mark.parametrize("dimension", [1, 2])
def test_legendre_quadrature_orthogonality(dimension):
    basis = LegendreBasis([[-2., 3.]]*dimension, [4]*dimension)
    points, weights = basis.quadrature(8)
    values = basis.evaluate(points)
    volume = 5**dimension
    np.testing.assert_allclose(weights.sum(), volume, rtol=2e-15)
    np.testing.assert_allclose(weights @ values, 0., atol=8*np.finfo(float).eps*volume)
    expected = volume / np.prod(2*basis.indices + 1, axis=1)
    np.testing.assert_allclose(values.T @ (weights[:, None]*values), np.diag(expected), atol=2e-14)


@pytest.mark.parametrize("points", [
    [[0.1, -0.2], [-0.7, 0.4]],
    [[-1.7, 2.1], [1.4, 0.3]],
])
def test_basis_derivatives_interior_and_bounded_tangent_extension(points):
    basis = LegendreBasis([[-1, 1], [-1, 1]], [4, 3])
    points = np.asarray(points)
    values, gradient = basis.values_and_gradients(points)
    np.testing.assert_array_equal(values, basis.evaluate(points))
    for d in range(2):
        shift = np.eye(2)[d]*1e-6
        finite = (basis.evaluate(points+shift)-basis.evaluate(points-shift))/(2e-6)
        np.testing.assert_allclose(gradient[..., d], finite, rtol=2e-8, atol=5e-8)


def test_exterior_is_bounded_and_c1_at_each_boundary():
    basis = LegendreBasis([[-2., 2.]], 5)
    for endpoint in (-2., 2.):
        _, middle = basis.values_and_gradients([endpoint])
        for direction in (-1, 1):
            _, nearby = basis.values_and_gradients([endpoint+direction*1e-8])
            np.testing.assert_allclose(nearby, middle, rtol=1e-7, atol=1e-7)
    endpoints = basis.evaluate(np.array([[-1e6], [1e6]]))
    farther = basis.evaluate(np.array([[-1e12], [1e12]]))
    np.testing.assert_allclose(endpoints, farther, rtol=3e-6)


@pytest.mark.parametrize("beads", [1, 3, 8])
@pytest.mark.parametrize("outside", [False, True])
def test_bias_force_and_bead_scaling_against_energy_finite_differences(beads, outside):
    bias = VESBias(cv(2), 2., [[-1, 1], [-1, 1]], [3, 2], wall_stiffness=[2., 3.])
    bias.coefficients[:] = np.linspace(-0.1, 0.2, bias.basis.size)
    q = np.zeros((beads, 1, 3))
    q[:, 0, :2] = [1.3, -1.4] if outside else [0.2, -0.3]
    q[:, 0, 2] = np.linspace(-0.2, 0.4, beads)
    energy, force = bias.energy_force(q)
    assert energy == pytest.approx(beads * bias.physical_energy(q.mean(0)[0, :2]))
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numeric = -(bias.energy_force(plus)[0]-bias.energy_force(minus)[0])/(2e-6)
        assert force[index] == pytest.approx(numeric, abs=5e-8, rel=2e-7)
    np.testing.assert_allclose(force, np.repeat(force[:1], beads, axis=0), rtol=1e-15)


def test_nonlinear_centroid_distance_force():
    bias = VESBias(CVStack([Distance([0, 1])]), 1., [[0.5, 3]], orders=3)
    bias.coefficients[:] = [0.1, -0.3, 0.2]
    q = np.array([[[0., 0., 0.], [1., 1., 0.]], [[0.1, 0., 0.], [1.4, 0.8, 0.]]])
    _, force = bias.energy_force(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        derivative = -(bias.energy_force(plus)[0]-bias.energy_force(minus)[0])/(2e-6)
        assert force[index] == pytest.approx(derivative, abs=1e-9)


def test_variational_gradient_and_hessian_match_independent_functional_differences():
    beta = 1.7
    bias = VESBias(cv(2), beta, [[-1.3, 1.7], [-1.2, 1.2]], [2, 2], quadrature_order=20)
    points, weights = bias.basis.quadrature(35)
    features = bias.basis.evaluate(points)
    x, y = points.T
    physical = (x*x-0.5)**2+0.7*y*y+0.3*x*y
    coefficients = np.linspace(-0.1, 0.12, bias.basis.size)
    log_w = np.log(weights)
    reference_z = logsumexp(log_w-beta*physical)
    def objective(alpha):
        return (logsumexp(log_w-beta*(physical+features@alpha))-reference_z)/beta
    def moment_gradient(alpha):
        logmass = log_w-beta*(physical+features@alpha)
        mass = np.exp(logmass-logsumexp(logmass))
        mean = mass @ features
        centered = features-mean
        return -mean, beta * (centered.T @ (mass[:, None]*centered))
    gradient, hessian = moment_gradient(coefficients)
    for i in range(bias.basis.size):
        step = np.eye(bias.basis.size)[i]*1e-5
        numeric = (objective(coefficients+step)-objective(coefficients-step))/(2e-5)
        assert gradient[i] == pytest.approx(numeric, abs=3e-11)
        numeric_h = (moment_gradient(coefficients+step)[0]-moment_gradient(coefficients-step)[0])/(2e-5)
        np.testing.assert_allclose(hessian[:, i], numeric_h, atol=4e-11)
    bias.coefficients[:] = coefficients
    bias.instantaneous[:] = coefficients+0.05
    old = bias.instantaneous.copy()
    bias.update_from_moments(-gradient, hessian/beta)
    expected = old-bias.learning_rate*(gradient+hessian@(old-coefficients))
    np.testing.assert_allclose(bias.instantaneous, expected, atol=1e-16)
    np.testing.assert_allclose(bias.coefficients, (coefficients+expected)/2, atol=1e-16)


def test_uniform_exact_polynomial_optimum_is_stationary():
    bias = VESBias(cv(), 1.3, [[-2, 2]], 4, pace=9)
    points, weights = bias.basis.quadrature(32)
    features = bias.basis.evaluate(points)
    x = points[:, 0]
    physical = (x*x-1)**2+0.3*x
    # Independent weighted projection; no VES optimizer formula used to fit.
    coefficient = np.linalg.lstsq(np.column_stack([np.ones(len(x)), features]), physical, rcond=None)[0][1:]
    bias.coefficients[:] = -coefficient
    bias.instantaneous[:] = -coefficient
    mass = weights/weights.sum()  # F+V is constant throughout the domain.
    mean = mass @ features
    centered = features-mean
    before = bias.coefficients.copy()
    bias.update_from_moments(mean, centered.T@(mass[:, None]*centered))
    np.testing.assert_allclose(bias.coefficients, before, atol=5e-16)
    recovered = bias.free_energy(points)
    np.testing.assert_allclose(recovered-recovered[0], physical-physical[0], atol=1e-14)


def test_well_tempered_target_update_and_matching_finite_stage_estimate():
    bias = VESBias(cv(), 2., [[-1, 1]], 2, target="well_tempered", bias_factor=4.,
                   learning_rate=.3, pace=4, target_update_interval=1, quadrature_order=40)
    points, weights = bias.basis.quadrature(80)
    for _ in range(3):
        old_target = bias.target_coefficients.copy()
        bias.update_from_moments(np.array([0.2, -0.1]), np.diag([.2, .1]))
        expected_target = (bias.beta*bias.coefficients+old_target)/bias.bias_factor
        np.testing.assert_array_equal(bias.target_coefficients, expected_target)
        np.testing.assert_array_equal(bias.reconstruction_target, old_target)
        estimate = bias.basis.evaluate(points) @ (-bias.coefficients-old_target/bias.beta)
        np.testing.assert_allclose(bias.free_energy(points), estimate, atol=1e-16)
        assert np.sum(weights*np.exp(bias.log_target(points))) == pytest.approx(1., abs=2e-14)
    assert bias.target_updates == 3
    assert not np.allclose(bias.free_energy(points), -4/3*bias.basis.evaluate(points)@bias.coefficients)
    assert np.isneginf(bias.log_target([[2.]])).all()


@pytest.mark.parametrize("target", ["uniform", "well_tempered"])
@pytest.mark.parametrize("cut", [0, 1, 6, 7, 14, 15, 30])
def test_checkpoint_restores_partial_blocks_and_dynamic_targets_exactly(target, cut):
    kwargs = dict(target=target, pace=7, target_update_interval=2, wall_stiffness=[3., 4.])
    sequence = np.random.default_rng(713).uniform(-1.4, 1.4, (43, 2))
    original = VESBias(cv(2), 1.2, [[-1, 1]]*2, [2, 3], **kwargs)
    for value in sequence[:cut]:
        original.observe(value)
    encoded = json.loads(json.dumps(original.state_dict(), allow_nan=False))
    continued = VESBias(cv(2), 1.2, [[-1, 1]]*2, [2, 3], state=encoded, **kwargs)
    for value in sequence[cut:]:
        original.observe(value)
        continued.observe(value)
    assert continued.state_dict() == original.state_dict()
    assert original.pending_count == len(sequence)%7
    assert original.observations == len(sequence)
    assert original.outside == sum(not original.basis.contains(x) for x in sequence)


def test_outside_points_are_excluded_not_clipped_and_empty_block_failure_is_atomic():
    bias = VESBias(cv(), 1., [[-1, 1]], orders=2, pace=3)
    bias.observe([0.2])
    bias.observe([2.])
    bias.observe([-3.])
    np.testing.assert_allclose(bias.last_gradient, -bias.basis.evaluate([0.2]))
    np.testing.assert_array_equal(bias.last_covariance, np.zeros((2, 2)))
    assert bias.outside == 2
    bias.observe([2.])
    bias.observe([3.])
    before = bias.state_dict()
    with pytest.raises(ValueError, match="no observations inside"):
        bias.observe([4.])
    assert bias.state_dict() == before


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(schema_version=True),
    lambda s: s.update(observations=3.5),
    lambda s: s.update(outside=100),
    lambda s: s.update(pending_count=100),
    lambda s: s.update(pending_inside=100),
    lambda s: s.update(updates=100),
    lambda s: s.update(target_updates=100),
    lambda s: s["coefficients"].append(0),
    lambda s: s["mean"].__setitem__(0, 3.),
    lambda s: s["m2"][0].__setitem__(0, -100.),
    lambda s: s["last_covariance"][0].__setitem__(0, -1.),
    lambda s: s["target_coefficients"].__setitem__(0, 10.),
    lambda s: s["parameters"].update(pace=15),
    lambda s: s.update(unknown=1),
])
def test_malformed_checkpoint_rejected_before_mutation(mutation):
    bias = VESBias(cv(), 1., [[-1, 1]], orders=2, pace=5, target="well_tempered", target_update_interval=1)
    for value in [0.1, -.2, .3, .2, -.1, .4]:
        bias.observe([value])
    original = bias.state_dict()
    invalid = copy.deepcopy(original)
    mutation(invalid)
    with pytest.raises(ValueError):
        bias.load_state_dict(invalid)
    assert bias.state_dict() == original


@pytest.mark.parametrize("kwargs", [
    {"orders": True}, {"orders": 2.5}, {"orders": 0}, {"orders": 33},
    {"pace": 0}, {"pace": True}, {"target_update_interval": 0},
    {"target": "well-tempered"}, {"target": "unknown"},
    {"bias_factor": 1.}, {"bias_factor": True}, {"learning_rate": 0},
    {"learning_rate": float("nan")}, {"learning_rate": "0.1"},
    {"quadrature_order": 4}, {"quadrature_order": True},
    {"wall_stiffness": 0.}, {"wall_stiffness": True},
])
def test_invalid_core_parameters_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        VESBias(cv(), 1., [[-1., 1.]], **kwargs)


def test_noncentroid_and_periodic_cvs_rejected():
    with pytest.raises(ValueError, match="centroid"):
        VESBias(CVStack([Distance([0, 1], estimator="bead_average")]), 1., [[0.1, 3.]])
    periodic = Cartesian(0, 0)
    periodic.periods = (2.,)
    with pytest.raises(ValueError, match="nonperiodic"):
        VESBias(periodic, 1., [[-1., 1.]])
