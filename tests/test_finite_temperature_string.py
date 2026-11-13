"""Thermal references, centroid metrics, uncertainty and exact restart."""

import copy
import json

import numpy as np
import pytest
from scipy.integrate import quad

from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.finite_temperature_string import (
    RestrainedStringSampler, StringStatistics, finite_temperature_string,
    restrained_string_statistics, _metric,
)
from pyrpmd.potentials.builtin import HarmonicPotential


def _sampler(*, nbeads=1, sampling_steps=80, block_size=10, stiffness=3., seed=30):
    images = np.array([[-0.6, 0.], [0., 0.3], [0.6, 0.]])
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    dynamics, states = [], []
    for index, image in enumerate(images):
        dyn = Dynamics(HarmonicPotential(k=[2., 4., 1.]), [1.], beta=1.,
                       nbeads=nbeads, dt=.05, seed=seed + index, native=False,
                       potential_identity="test:harmonic:2,4,1")
        q = np.zeros((nbeads, 1, 3))
        q[:, 0, :2] = image
        states.append(State(q, dyn.sample_momenta()))
        dynamics.append(dyn)
    return images, RestrainedStringSampler(dynamics, cv, states, stiffness=stiffness,
                                           equilibration_steps=20, sampling_steps=sampling_steps,
                                           sample_interval=1, block_size=block_size,
                                           gamma_centroid=2.)


@pytest.mark.parametrize("beads", [1, 4, 9])
def test_centroid_metric_has_no_spurious_bead_factor(beads):
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    q = np.zeros((beads, 1, 3))
    jac = cv.value_and_jacobian(q).jacobian
    assert jac.shape == (2, beads, 1, 3)
    np.testing.assert_allclose(_metric(jac, np.array([2.])), np.eye(2) / 2., atol=1e-15)


def test_restrained_gaussian_samples_match_exact_multidimensional_free_energy():
    rng = np.random.default_rng(44)
    images = np.column_stack((np.linspace(-1, 1, 9), np.linspace(-.3, .8, 9)))
    a = np.array([[2., .5], [.5, 3.]])
    k = np.array([[10., 1.], [1., 12.]])
    beta, samples = 1.7, 40000
    covariance = np.linalg.inv(beta * (a + k))
    mean = np.linalg.solve(a + k, (k @ images.T)).T
    values = mean[:, None, :] + rng.multivariate_normal(np.zeros(2), covariance, size=(len(images), samples))
    metrics = np.broadcast_to(np.diag([.5, 2.]), (len(images), samples, 2, 2))
    stats = restrained_string_statistics(images, values, metrics, k, block_size=100)
    exact_gradient = (k @ (images - mean).T).T
    errors = np.sqrt(np.diagonal(stats.gradient_covariance, axis1=1, axis2=2))
    assert np.max(np.abs(stats.gradient - exact_gradient) / errors) < 4.5
    # An independent partition-function calculation gives F_K(z).
    effective = k - k @ np.linalg.solve(a + k, k)
    exact_f = .5 * np.einsum("mi,ij,mj->m", images, effective, images)
    result = finite_temperature_string(images, lambda path: stats, tolerance=1e6, max_iterations=1)
    difference = result.free_energy - (exact_f - exact_f[0])
    assert np.all(np.abs(difference[1:]) < 4.5 * result.free_energy_standard_error[1:])
    assert result.converged
    assert stats.correlation_resolved


def test_radial_jacobian_is_in_thermal_measure_not_a_second_fixman_bias():
    # In 2D isotropic harmonic confinement the radial density includes r.
    # Integrate that Jacobian exactly and sample its inverse CDF.  Omitting r
    # would produce a different force, even in the infinite-sample limit.
    rng = np.random.default_rng(948)
    images = np.array([[.5, -.2], [1., 0.], [1.5, .2]])
    beta, spring, kappa, samples = 2., 1.3, 15., 50000
    radius = np.linspace(0, 4, 30001)
    values, exact = [], []
    for center in images:
        density = radius * np.exp(-beta * (.5 * spring * radius**2 + .5 * kappa * (radius - center[0])**2))
        cdf = np.cumsum((density[:-1] + density[1:]) * np.diff(radius) / 2)
        cdf = np.r_[0, cdf / cdf[-1]]
        draws = np.interp(rng.random(samples), cdf, radius)
        y_mean = kappa * center[1] / (spring + kappa)
        values.append(np.column_stack((draws, rng.normal(y_mean, np.sqrt(1 / (beta * (spring + kappa))), samples))))
        def weight(r):
            return r * np.exp(-beta * (.5 * spring * r*r + .5 * kappa * (r - center[0])**2))
        mean_radius = quad(lambda r: r * weight(r), 0, 5, epsabs=1e-12)[0] / quad(weight, 0, 5, epsabs=1e-12)[0]
        exact.append(kappa * (center - [mean_radius, y_mean]))
    metrics = np.broadcast_to(np.eye(2), (3, samples, 2, 2))
    stats = restrained_string_statistics(images, values, metrics, kappa, block_size=100)
    se = np.sqrt(np.diagonal(stats.gradient_covariance, axis1=1, axis2=2))
    assert np.max(np.abs(stats.gradient - exact) / se) < 4.5
    # Cartesian treatment would predict k*a/(k+a)*r, missing radial entropy.
    wrong = kappa * spring / (kappa + spring) * images[:, 0]
    assert np.min(np.abs(stats.gradient[:, 0] - wrong) / se[:, 0]) > 20


def test_thermal_string_relaxes_entropic_valley_and_integrates_free_energy():
    beta, transverse, coupling = 2., 10., 3.
    y0 = -coupling / (2 * beta * transverse)
    x = np.linspace(-1, 1, 31)
    initial = np.column_stack((x, y0 + .1 * np.sin(np.linspace(0, np.pi, len(x)))))
    def sampled_oracle(path):
        # Exact conditional Gaussian integration over hidden z in
        # V=(x*x-1)^2+10*y*y/2+exp(c*y)*z*z/2 gives the entropic term c*y/(2 beta).
        gradient = np.column_stack((4 * path[:, 0] * (path[:, 0]**2 - 1),
                                    transverse * path[:, 1] + coupling / (2 * beta)))
        cov = np.zeros((len(path), 2, 2))
        return StringStatistics(gradient, np.broadcast_to(np.eye(2), cov.shape), cov, cov,
                                np.full(len(path), 1e6), 100, True)
    result = finite_temperature_string(initial, sampled_oracle, step=.03,
                                        tolerance=1e-5, max_iterations=300)
    assert result.converged
    np.testing.assert_array_equal(result.images[[0, -1]], initial[[0, -1]])
    assert np.max(np.abs(result.images[:, 1] - y0)) < 1e-5
    reference = []
    for x_i, y_i in result.images:
        integral = quad(lambda z: np.exp(-beta * ((x_i*x_i - 1)**2 +
                              transverse*y_i*y_i/2 + np.exp(coupling*y_i)*z*z/2)), -np.inf, np.inf)[0]
        reference.append(-np.log(integral) / beta)
    reference = np.array(reference) - reference[0]
    np.testing.assert_allclose(result.free_energy, reference, atol=.005)
    assert len(result.history) == result.iterations + 1
    assert result.history[-1] == result.maximum_perpendicular_drift


@pytest.mark.parametrize("beads", [1, 4])
def test_actual_restrained_rpmd_means_match_harmonic_reference(beads):
    images, sampler = _sampler(nbeads=beads, sampling_steps=1500, block_size=150)
    sampler.equilibration_steps = 200
    stats = sampler(images)
    exact = images * np.array([3*2/5, 3*4/7])
    errors = np.sqrt(np.diagonal(stats.gradient_covariance, axis1=1, axis2=2))
    assert np.max(np.abs(stats.gradient - exact) / errors) < 4.
    np.testing.assert_allclose(stats.metric, np.broadcast_to(np.eye(2), (3, 2, 2)), atol=1e-14)
    assert not stats.correlation_resolved  # only ten blocks: deliberately conservative


def test_split_restart_matches_uninterrupted_images_states_and_rng(tmp_path):
    images, continuous = _sampler()
    full = finite_temperature_string(images, continuous, step=.003, tolerance=1e-12, max_iterations=3)
    images, first = _sampler()
    path = tmp_path / "string.npz"
    finite_temperature_string(images, first, step=.003, tolerance=1e-12,
                              max_iterations=1, checkpoint=path)
    images, restored = _sampler(seed=999)
    resumed = finite_temperature_string(images, restored, step=.003, tolerance=1e-12,
                                         max_iterations=3, checkpoint=path, resume=True)
    np.testing.assert_array_equal(full.images, resumed.images)
    np.testing.assert_array_equal(full.free_energy, resumed.free_energy)
    np.testing.assert_array_equal(full.statistics.gradient, resumed.statistics.gradient)
    assert full.history == resumed.history
    for a, b, d, e in zip(continuous.states, restored.states, continuous.dynamics, restored.dynamics):
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.p, b.p)
        assert (a.step, a.time) == (b.step, b.time)
        assert json.dumps(d.operator_state_dict(), sort_keys=True) == json.dumps(e.operator_state_dict(), sort_keys=True)


def test_restart_rejects_changed_scientific_identity(tmp_path):
    images, first = _sampler()
    path = tmp_path / "string.npz"
    finite_temperature_string(images, first, max_iterations=1, checkpoint=path)
    images, different = _sampler(stiffness=4.)
    before = different.snapshot()
    with pytest.raises(ValueError, match="identity"):
        finite_temperature_string(images, different, max_iterations=2, checkpoint=path, resume=True)
    assert different.snapshot() == before


def test_restart_rejects_incomplete_batch_clock_before_restoring_rng(tmp_path):
    images, sampler = _sampler()
    path = tmp_path / "string.npz"
    finite_temperature_string(images, sampler, max_iterations=1, checkpoint=path)
    with load_checkpoint(path) as stored:
        payload = {name: stored[name] for name in stored.files}
    metadata = json.loads(str(payload["metadata"].item()))
    metadata["sampler_state"]["states"][1]["step"] += 1
    payload["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(path, kind="PyRPMD.thermal-string.v1", **payload)
    images, fresh = _sampler(seed=999)
    before = fresh.snapshot()
    with pytest.raises(ValueError, match="clock"):
        finite_temperature_string(images, fresh, max_iterations=2, checkpoint=path, resume=True)
    assert fresh.snapshot() == before


def test_sampler_failure_restores_already_advanced_other_images():
    images, sampler = _sampler()
    before = copy.deepcopy(sampler.snapshot())
    class FailingPotential:
        def evaluate(self, x):
            raise RuntimeError("deliberate reference failure")
    sampler.dynamics[1].potential = FailingPotential()
    with pytest.raises(RuntimeError, match="deliberate"):
        sampler(images)
    assert sampler.snapshot() == before


@pytest.mark.parametrize("option", [{"step": True}, {"max_iterations": True}, {"confidence_z": 0.}, {"resume": 1}])
def test_invalid_algorithm_controls_rejected(option):
    images, sampler = _sampler()
    with pytest.raises(ValueError):
        finite_temperature_string(images, sampler, **option)


def test_invalid_samples_and_singular_metric_fail_explicitly():
    images, _ = _sampler()
    values = np.zeros((3, 20, 2))
    metrics = np.zeros((3, 20, 2, 2))
    with pytest.raises(ValueError, match="metric"):
        restrained_string_statistics(images, values, metrics, 1.)
    metrics = np.broadcast_to(np.eye(2), (3, 20, 2, 2))
    with pytest.raises(ValueError, match="blocks"):
        restrained_string_statistics(images, values[:, :19], metrics[:, :19], 1.)
    with pytest.raises(ValueError, match="finite"):
        restrained_string_statistics(images, values + 1j, metrics, 1.)


def test_checkpoint_requires_explicit_potential_identity(tmp_path):
    images, sampler = _sampler()
    sampler.dynamics[0].potential_identity = None
    with pytest.raises(ValueError, match="potential_identity"):
        finite_temperature_string(images, sampler, checkpoint=tmp_path / "bad.npz")


def test_bead_average_and_periodic_cvs_rejected():
    _, sampler = _sampler()
    bad = CVStack([Cartesian(0, 0, estimator="bead_average"), Cartesian(0, 1)])
    with pytest.raises(ValueError, match="centroid"):
        RestrainedStringSampler(sampler.dynamics, bad, sampler.states, stiffness=3.)
    periodic = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    periodic.periods = (2., None)
    with pytest.raises(ValueError, match="nonperiodic"):
        RestrainedStringSampler(sampler.dynamics, periodic, sampler.states, stiffness=3.)


def test_separate_generators_with_identical_streams_are_rejected():
    _, sampler = _sampler()
    for dynamics in sampler.dynamics:
        dynamics.rng = np.random.default_rng(9)
    with pytest.raises(ValueError, match="distinct states"):
        RestrainedStringSampler(sampler.dynamics, sampler.cv, sampler.states, stiffness=3.)
