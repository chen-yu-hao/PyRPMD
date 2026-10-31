"""Physics and estimator checks for internal-only TRPMD and spectra."""

import copy

import numpy as np
import pytest
from scipy.constants import c

from pyrpmd.constants import ATOMIC_TIME_S
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.analysis.spectroscopy import (
    bead_averaged_dipoles, centroid_velocities, time_correlation,
    velocity_autocorrelation, dipole_autocorrelation, flux_correlation,
    power_spectrum, relative_ir_spectrum,
)


NATIVE_CHOICES = [False] + ([True] if native_capabilities()["available"] else [])
SCHEMES = ["normal-mode-bab", "cayley-bcb"]


def real_modes(beads, beta):
    # Independently construct eigenvectors of the cyclic spring Laplacian.
    eye = np.eye(beads)
    laplacian = 2 * eye - np.roll(eye, 1, axis=0) - np.roll(eye, -1, axis=0)
    values, vectors = np.linalg.eigh(laplacian)
    return vectors.T, beads / beta * np.sqrt(np.maximum(values, 0))


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("native", NATIVE_CHOICES)
@pytest.mark.parametrize("beads", [2, 7, 16, 31])
def test_trpmd_leaves_free_centroid_unthermostatted(scheme, native, beads):
    rng = np.random.default_rng(15)
    state = State(rng.normal(size=(beads, 2, 3)), rng.normal(size=(beads, 2, 3)))
    initial = state.copy()
    masses = np.array([1.7, 4.3])
    dynamics = Dynamics(FreePotential(), masses, 2.8, beads, .03,
                        seed=33, integrator=scheme, native=native)
    for _ in range(80):
        dynamics.trpmd_step(state, friction_scale=.7)
    np.testing.assert_allclose(state.p.mean(axis=0), initial.p.mean(axis=0), atol=2e-13)
    np.testing.assert_allclose(
        state.q.mean(axis=0), initial.q.mean(axis=0) + 80 * .03 * initial.p.mean(axis=0) / masses[:, None],
        atol=3e-13,
    )
    assert not np.allclose(state.p - state.p.mean(0), initial.p - initial.p.mean(0))


@pytest.mark.parametrize("native", NATIVE_CHOICES)
def test_trpmd_preserves_fixed_substrate_exactly(native):
    reference = np.array([[.4, -.2, 1.3], [.7, .2, .4]])
    q = np.broadcast_to(reference, (8, 2, 3)).copy()
    state = State(q, np.zeros_like(q))
    dynamics = Dynamics(
        FreePotential(), [3., 1.], 2.5, 8, .05, seed=54, native=native,
        fixed_atoms=(0,), fixed_positions=reference,
    )
    for _ in range(25):
        dynamics.trpmd_step(state)
    np.testing.assert_array_equal(state.q[:, 0], q[:, 0])
    np.testing.assert_array_equal(state.p[:, 0], 0)
    assert np.any(state.p[:, 1] != 0)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("beads,friction", [(1, 1.0), (7, 0.0)])
@pytest.mark.parametrize("native", NATIVE_CHOICES)
def test_trpmd_classical_and_zero_friction_limits_are_identical_rpmd(scheme, beads, friction, native):
    dynamics = Dynamics(HarmonicPotential(.3), [2.0], 2.3, beads, .1,
                        seed=18, integrator=scheme, native=native)
    rng = np.random.default_rng(21)
    a = State(rng.normal(size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3)))
    b = a.copy()
    rng_before = copy.deepcopy(dynamics.rng.bit_generator.state)
    for _ in range(20):
        dynamics.trpmd_step(a, friction_scale=friction)
        dynamics.step(b)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.p, b.p)
    assert (a.step, a.time) == (b.step, b.time)
    assert dynamics.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("friction", [.25, 1.0])
def test_free_internal_modes_preserve_gibbs_covariance(scheme, friction):
    # 12,000 independent Cartesian components give a stable statistical check
    # of the full free/OU/free map, independent of its real-mode ordering.
    beads, atoms, beta, mass = 8, 4000, 3.1, 2.3
    modes, omega = real_modes(beads, beta)
    rng = np.random.default_rng(245)
    qk = np.zeros((beads, atoms, 3))
    qk[1:] = rng.normal(size=qk[1:].shape) * np.sqrt(
        beads / (beta * mass * omega[1:, None, None] ** 2)
    )
    pk = rng.normal(size=qk.shape) * np.sqrt(mass * beads / beta)
    pk[0] = 0
    state = State(np.einsum("kb,kac->bac", modes, qk), np.einsum("kb,kac->bac", modes, pk))
    dynamics = Dynamics(FreePotential(), np.full(atoms, mass), beta, beads, .23,
                        integrator=scheme, native=False, seed=751)
    for _ in range(3):
        dynamics.trpmd_step(state, friction_scale=friction)
    final_q = np.einsum("kb,bac->kac", modes, state.q)
    final_p = np.einsum("kb,bac->kac", modes, state.p)
    q_variance = beads / (beta * mass * omega[1:] ** 2)
    p_variance = mass * beads / beta
    np.testing.assert_allclose(np.mean(final_q[1:] ** 2, axis=(1, 2)), q_variance, rtol=.045)
    np.testing.assert_allclose(np.mean(final_p[1:] ** 2, axis=(1, 2)), p_variance, rtol=.045)
    cross = np.mean(final_q[1:] * final_p[1:], axis=(1, 2))
    assert np.max(np.abs(cross / np.sqrt(q_variance * p_variance))) < .04
    np.testing.assert_allclose(final_p[0], 0, atol=5e-14)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("native", NATIVE_CHOICES)
def test_trpmd_harmonic_centroid_is_independent_of_friction(scheme, native):
    rng = np.random.default_rng(812)
    initial = State(rng.normal(size=(8, 1, 3)), rng.normal(size=(8, 1, 3)))
    reference = initial.copy()
    rpmd = Dynamics(HarmonicPotential(.7), [1.5], 3.7, 8, .01,
                    integrator=scheme, native=native)
    actual = initial.copy()
    trpmd = Dynamics(HarmonicPotential(.7), [1.5], 3.7, 8, .01,
                     integrator=scheme, native=native, seed=613)
    for _ in range(200):
        rpmd.step(reference)
        trpmd.trpmd_step(actual, friction_scale=.5)
    np.testing.assert_allclose(actual.q.mean(0), reference.q.mean(0), atol=2e-13)
    np.testing.assert_allclose(actual.p.mean(0), reference.p.mean(0), atol=2e-13)


@pytest.mark.parametrize("native", NATIVE_CHOICES)
def test_trpmd_failure_rolls_back_state_rng_clock_and_retry(native):
    class FailSecond(HarmonicPotential):
        def __init__(self):
            super().__init__(.7)
            self.calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 2:
                raise PotentialError("injected TRPMD failure")
            return super().evaluate(q)

    a = State(np.ones((8, 1, 3)), np.zeros((8, 1, 3)))
    original = a.copy()
    dynamics = Dynamics(FailSecond(), [1.1], 3.1, 8, .02, native=native, seed=43)
    rng_before = copy.deepcopy(dynamics.rng.bit_generator.state)
    with pytest.raises(PotentialError, match="injected"):
        dynamics.trpmd_step(a)
    np.testing.assert_array_equal(a.q, original.q)
    np.testing.assert_array_equal(a.p, original.p)
    assert (a.step, a.time) == (original.step, original.time)
    assert dynamics.rng.bit_generator.state == rng_before
    dynamics.potential = HarmonicPotential(.7)
    reference = Dynamics(HarmonicPotential(.7), [1.1], 3.1, 8, .02, native=native, seed=43)
    dynamics.trpmd_step(a)
    reference.trpmd_step(original)
    np.testing.assert_array_equal(a.q, original.q)
    np.testing.assert_array_equal(a.p, original.p)
    assert dynamics.rng.bit_generator.state == reference.rng.bit_generator.state


@pytest.mark.parametrize("options,match", [
    ({"thermostat": "trpmd", "gamma_centroid": .1}, "gamma_centroid=0"),
    ({"thermostat": "trpmd", "friction_scale": -.1}, "friction_scale"),
    ({"thermostat": "trpmd", "friction_scale": float("nan")}, "friction_scale"),
    ({"thermostat": "trpmd", "friction_scale": True}, "friction_scale"),
    ({"thermostat": "pile-l", "friction_scale": .5}, "TRPMD option"),
    ({"thermostat": "pile-l", "gamma_centroid": 0}, "positive"),
    ({"thermostat": "unknown"}, "thermostat"),
])
def test_invalid_thermostat_options_do_not_mutate(options, match):
    dynamics = Dynamics(FreePotential(), [1], 3, 4, .01, native=False)
    state = State(np.zeros((4, 1, 3)), np.ones((4, 1, 3)))
    old = state.copy()
    with pytest.raises(ValueError, match=match):
        dynamics.sample_step(state, **options)
    np.testing.assert_array_equal(state.q, old.q)
    np.testing.assert_array_equal(state.p, old.p)


def test_trpmd_alias_and_changing_friction_cache_match_python():
    rng = np.random.default_rng(732)
    initial = State(rng.normal(size=(8, 1, 3)), rng.normal(size=(8, 1, 3)))
    a, b = initial.copy(), initial.copy()
    automatic = Dynamics(HarmonicPotential(.2), [1.3], 2.5, 8, .05, seed=12)
    python = Dynamics(HarmonicPotential(.2), [1.3], 2.5, 8, .05, seed=12, native=False)
    for thermostat, friction in [("trpmd", .2), ("pile-l", 1.), ("trpmd", 1.), ("trpmd", .5)]:
        automatic.sample_step(a, thermostat=thermostat, friction_scale=friction)
        python.sample_step(b, thermostat=thermostat, friction_scale=friction)
        np.testing.assert_allclose(a.q, b.q, atol=1e-13)
        np.testing.assert_allclose(a.p, b.p, atol=1e-13)
        assert automatic.rng.bit_generator.state == python.rng.bit_generator.state


@pytest.mark.parametrize("demean", [False, True])
def test_fft_cross_correlation_matches_direct_without_wraparound(demean):
    rng = np.random.default_rng(312)
    a, b = rng.normal(size=(3, 19, 4)), rng.normal(size=(3, 19, 4))
    a[0, 0] += 40
    b[-1, -1] -= 30
    actual = time_correlation(a, .7, other=b, max_lag=18, demean=demean)
    expected = time_correlation(a, .7, other=b, max_lag=18, demean=demean, method="direct")
    np.testing.assert_allclose(actual["per_replica"], expected["per_replica"], atol=5e-13)
    np.testing.assert_allclose(actual["standard_error"], expected["standard_error"], atol=5e-13)
    np.testing.assert_array_equal(actual["origins_per_replica"], np.arange(19, 0, -1))
    np.testing.assert_array_equal(actual["lag_time_au"], .7 * np.arange(19))


def test_replica_boundaries_are_not_concatenated_and_origins_are_not_error_samples():
    values = np.array([[[2.], [2.], [2.]], [[-4.], [-4.], [-4.]]])
    result = time_correlation(values, 1, max_lag=2)
    np.testing.assert_allclose(result["correlation"], 10)
    np.testing.assert_allclose(result["standard_error"], 6)
    assert time_correlation(values[0], 1)["standard_error"] is None
    np.testing.assert_allclose(time_correlation(values, 1, demean=True)["correlation"], 0)


def test_velocity_uses_centroid_and_real_masses_and_atom_average():
    p = np.zeros((2, 7, 4, 2, 3))
    p[..., 0, 0] = 2
    p[..., 1, 1] = 6
    p[:, :, 0] += 9
    p[:, :, 1] -= 9  # Internal fluctuations cancel in the estimator.
    velocity = centroid_velocities(p, [2, 3])
    np.testing.assert_array_equal(velocity[0, 0], [[1, 0, 0], [0, 2, 0]])
    result = velocity_autocorrelation(p, [2, 3], 1)
    np.testing.assert_allclose(result["correlation"], 2.5)
    np.testing.assert_allclose(result["standard_error"], 0)


def test_nonlinear_dipole_is_averaged_after_evaluation():
    q = np.array([1., 3.])
    bead_mu = np.zeros((10, 2, 3))
    bead_mu[:, :, 0] = q ** 2
    np.testing.assert_array_equal(bead_averaged_dipoles(bead_mu)[:, 0], 5)
    assert q.mean() ** 2 != 5
    result = dipole_autocorrelation(bead_mu, 1)
    np.testing.assert_allclose(result["correlation"], 25)


def test_flux_side_retains_sign_and_correct_time_ordering_without_rate_normalization():
    flux = np.array([[1., -2., 3., -4.], [-1., 0., 1., 0.]])
    side = np.array([[0., 1., 1., 0.], [1., 0., 1., 0.]])
    result = flux_correlation(flux, .2, side=side, max_lag=3)
    direct = time_correlation(flux[..., None], .2, other=side[..., None], max_lag=3, method="direct")
    np.testing.assert_allclose(result["correlation"], direct["correlation"], atol=1e-15)
    assert np.any(result["per_replica"] < 0)
    assert result["rate_normalized"] is False


@pytest.mark.parametrize("frames,nfft", [(128, 128), (127, 127), (128, 512)])
def test_periodogram_parseval_dc_nyquist_and_zero_padding(frames, nfft):
    rng = np.random.default_rng(937)
    values = rng.normal(size=(2, frames, 3))
    result = power_spectrum(values, .25, window="boxcar", demean=False, nfft=nfft)
    df = result["frequency_au"][1]
    np.testing.assert_allclose(np.sum(result["per_replica"], axis=1) * df,
                               np.mean(np.sum(values ** 2, axis=2), axis=1), rtol=1e-14)
    assert np.min(result["power"]) >= 0
    np.testing.assert_allclose(result["wavenumber_cm1"], result["frequency_au"] / (ATOMIC_TIME_S * c * 100))
    assert result["resolution_au"] == 1 / (frames * .25)


def test_known_harmonic_peak_and_ir_omega_squared_weight():
    frames, dt, harmonic_bin = 1024, .5, 64
    x = np.sin(2 * np.pi * harmonic_bin * np.arange(frames) / frames)
    dipoles = np.zeros((frames, 3, 3))
    dipoles[:, :, 0] = x[:, None]
    result = relative_ir_spectrum(dipoles, dt, beta=3, window="boxcar", nfft=frames)
    peak = np.argmax(result["relative_ir"])
    assert peak == harmonic_bin
    assert result["frequency_au"][peak] == harmonic_bin / (frames * dt)
    assert result["relative_ir"][peak] == 1
    assert result["relative_ir"][0] == 0
    assert result["absolute_intensity"] is False
    np.testing.assert_allclose(result["unnormalized_relative_ir"],
                               3 * result["angular_frequency_au"] ** 2 * result["power"])


@pytest.mark.parametrize("function,args,kwargs", [
    (time_correlation, ([1., 2.], 0), {}),
    (time_correlation, ([1., 2.], 1), {"max_lag": 2}),
    (time_correlation, ([1., 2.], 1), {"other": [1., 2., 3.]}),
    (time_correlation, ([1., np.nan], 1), {}),
    (time_correlation, ([1 + 2j, 3j], 1), {}),
    (power_spectrum, ([1., 2.], 1), {}),
    (power_spectrum, ([1., 2., 3.], 1), {"nfft": 2}),
    (power_spectrum, ([1., 2., 3.], 1), {"window": "unknown"}),
    (flux_correlation, ([1., 2., 3.], 1), {"side": [0., 1., 2.]}),
    (centroid_velocities, (np.zeros((3, 2, 1, 3)), [0.]), {}),
    (bead_averaged_dipoles, (np.zeros((3, 2, 2)),), {}),
])
def test_invalid_correlation_or_spectrum_inputs_are_rejected(function, args, kwargs):
    with pytest.raises(ValueError):
        function(*args, **kwargs)
