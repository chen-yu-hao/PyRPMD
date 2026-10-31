"""Independent action/proposal checks for equilibrium path sampling."""

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.path_integral import (
    OpenPathAction, PathIntegralAction, _ring_noise,
    sample_open_path, sample_path_integral,
)
from pyrpmd.potentials.builtin import HarmonicPotential


@pytest.mark.parametrize("method", ["primitive", "takahashi_imada"])
@pytest.mark.parametrize("beads", [1, 7])
def test_action_beta_derivatives_include_normalization_and_ti(method, beads):
    beta = 2.3
    q = np.random.default_rng(20).normal(size=(beads, 2, 3))
    mass = [1.0, 3.0]
    pes = HarmonicPotential(0.7)
    action = PathIntegralAction(pes, mass, 1 / (KB_AU * beta), method=method)
    terms = action.evaluate(q)
    eps = beta * 1e-4
    values = [PathIntegralAction(pes, mass, 1 / (KB_AU * b), method=method).evaluate(q).total
              for b in (beta - eps, beta, beta + eps)]
    first = (values[2] - values[0]) / (2 * eps)
    second = (values[2] - 2 * values[1] + values[0]) / eps**2
    assert first == pytest.approx(terms.energy - 6 * beads / (2 * beta), rel=3e-8, abs=1e-7)
    assert second == pytest.approx(terms.second_beta_derivative, rel=3e-7, abs=1e-6)
    expected_ti = beta**3 * np.sum((0.7 * q)**2 / np.array(mass)[None, :, None]) / (24 * beads**3)
    assert terms.correction == pytest.approx(expected_ti if method == "takahashi_imada" else 0)


def test_free_ring_noise_has_zero_centroid_and_correct_mode_covariance():
    rng = np.random.default_rng(14)
    # Independent atom slots serve as a large vectorized reference ensemble.
    beta, beads, atoms, mass = 3.0, 8, 16000, 2.0
    q = _ring_noise(rng, (beads, atoms, 3), beta, np.full(atoms, mass))
    assert np.max(abs(q.mean(axis=0))) < 2e-15
    modes = np.fft.rfft(q, axis=0, norm="ortho")
    expected = beta / (mass * beads * 4 * np.sin(np.pi * np.arange(1, beads // 2 + 1) / beads)**2)
    measured = np.mean(abs(modes[1:])**2, axis=(1, 2))
    np.testing.assert_allclose(measured, expected, rtol=0.025)


def test_closed_sampler_targets_harmonic_centroid_and_is_reproducible():
    beta, beads, stiffness = 2.0, 8, 0.64
    action = PathIntegralAction(HarmonicPotential(stiffness), [1.0], 1 / (KB_AU * beta), method="ti")
    initial = np.zeros((beads, 1, 3))
    options = dict(steps=12000, burn_in=1200, stride=4, centroid_step_bohr=0.7, internal_mix=1.0, seed=3)
    samples = sample_path_integral(action, initial, **options)
    # The TI action changes the harmonic curvature; centroid precision is
    # beta*k*(1 + beta²*k/(12*m*P²)).
    expected = 1 / (beta * stiffness * (1 + beta**2 * stiffness / (12 * beads**2)))
    measured = np.mean(samples.positions.mean(axis=1)**2)
    assert measured == pytest.approx(expected, rel=0.10)
    np.testing.assert_array_equal(initial, 0)
    assert len(samples.positions) == options["steps"] // options["stride"]
    assert all(0 < value < 1 for value in samples.acceptance.values())
    short = dict(steps=24, burn_in=0, stride=2, seed=5)
    np.testing.assert_array_equal(sample_path_integral(action, initial, **short).positions,
                                  sample_path_integral(action, initial, **short).positions)


def test_open_action_matches_independent_quadratic_form_and_endpoint_weights():
    beta, links, mass, stiffness = 1.7, 5, 2.0, 0.8
    action = OpenPathAction(HarmonicPotential(stiffness), [mass], 1 / (KB_AU * beta), axis=1)
    q = np.zeros((links + 1, 1, 3))
    q[:, 0, 1] = np.random.default_rng(8).normal(size=links + 1)
    spring, potential = action.evaluate(q)
    laplacian = np.diag([1.0] + [2.0] * (links - 1) + [1.0])
    laplacian -= np.diag(np.ones(links), 1) + np.diag(np.ones(links), -1)
    weights = np.array([0.5] + [1.0] * (links - 1) + [0.5])
    precision = mass * links / beta * laplacian + np.diag(beta * stiffness / links * weights)
    assert spring + potential == pytest.approx(0.5 * q[:, 0, 1] @ precision @ q[:, 0, 1])


def test_open_sampler_end_to_end_matches_finite_bead_gaussian():
    beta, links, stiffness = 2.0, 8, 0.64
    action = OpenPathAction(HarmonicPotential(stiffness), [1.0], 1 / (KB_AU * beta))
    samples = sample_open_path(action, np.zeros((links + 1, 1, 3)), steps=16000,
                               burn_in=1000, stride=4, centroid_step_bohr=0.6, internal_mix=1.0, seed=9)
    laplacian = np.diag([1.0] + [2.0] * (links - 1) + [1.0])
    laplacian -= np.diag(np.ones(links), 1) + np.diag(np.ones(links), -1)
    weights = np.array([0.5] + [1.0] * (links - 1) + [0.5])
    precision = links / beta * laplacian + np.diag(beta * stiffness / links * weights)
    difference = np.zeros(links + 1)
    difference[0], difference[-1] = -1, 1
    expected = difference @ np.linalg.solve(precision, difference)
    assert np.mean(samples.displacements**2) == pytest.approx(expected, rel=0.08)
    np.testing.assert_array_equal(samples.positions[:, 0, 0, 1:], samples.positions[:, -1, 0, 1:])


@pytest.mark.parametrize("masses", [[], [0], [-1], [np.nan], [True], [[1]], [1 + 1j]])
def test_invalid_masses_rejected(masses):
    with pytest.raises(ValueError):
        PathIntegralAction(HarmonicPotential(), masses, 300)


def test_scientific_topology_and_sampling_guards():
    pes = HarmonicPotential()
    with pytest.raises(ValueError, match="closed traces"):
        OpenPathAction(pes, [1], 300, method="ti")
    opened = OpenPathAction(pes, [1], 300)
    bad = np.zeros((3, 1, 3))
    bad[-1, 0, 2] = 1
    with pytest.raises(ValueError, match="endpoint"):
        opened.evaluate(bad)
    action = PathIntegralAction(pes, [1], 300)
    for options in ({"steps": True}, {"steps": 3, "stride": 4}, {"steps": 4, "internal_mix": 2}):
        with pytest.raises(ValueError):
            sample_path_integral(action, np.zeros((3, 1, 3)), **options)
    with pytest.raises(TypeError):
        sample_path_integral(opened, np.zeros((3, 1, 3)), steps=10)
    with pytest.raises(ValueError):
        action.evaluate_batch(np.zeros((3, 2, 2, 3)))
    one = sample_open_path(opened, np.zeros((3, 1, 3)), steps=1, burn_in=0, stride=1, seed=2)
    assert one.positions.shape == (1, 3, 1, 3)
    assert one.acceptance["internal"] == 0.0
