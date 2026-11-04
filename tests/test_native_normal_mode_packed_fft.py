"""Independent spectral reference for the packed two-real native FFT path."""

import numpy as np
import pytest


@pytest.mark.parametrize("beads", [32, 128])
def test_normal_mode_free_step_matches_numpy_spectral_solution(beads):
    native = pytest.importorskip("pyrpmd._native")
    masses = np.array([2.0, 3.0, 5.0])
    beta, dt = 41.0, 0.07
    rng = np.random.default_rng(618)
    q = rng.normal(size=(beads, len(masses), 3))
    p = rng.normal(size=q.shape)
    expected_q, expected_p = q.copy(), p.copy()

    wave = np.arange(beads)
    omega = 2 * beads / beta * np.sin(
        np.pi * np.minimum(wave, beads - wave) / beads
    )
    theta = dt * omega
    a = np.cos(theta)
    b = np.empty_like(omega)
    zero = omega == 0.0
    b[zero] = dt
    b[~zero] = np.sin(theta[~zero]) / omega[~zero]
    c = omega * np.sin(theta)
    qk = np.fft.fft(expected_q, axis=0)
    pk = np.fft.fft(expected_p, axis=0)
    qnext = a[:, None, None] * qk + b[:, None, None] * pk / masses[None, :, None]
    pnext = a[:, None, None] * pk - c[:, None, None] * masses[None, :, None] * qk
    expected_q = np.fft.ifft(qnext, axis=0).real
    expected_p = np.fft.ifft(pnext, axis=0).real

    context = native.create(masses, beta, beads, dt, 1, "normal-mode-bab")
    native.free_step(context, q, p)
    np.testing.assert_allclose(q, expected_q, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(p, expected_p, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("beads", [3, 32])
def test_packed_two_real_pile_map_matches_independent_numpy_fft(scheme, beads):
    native = pytest.importorskip("pyrpmd._native")
    masses = np.array([2.0, 3.0])
    beta, dt = 40.0, 0.07
    rng = np.random.default_rng(619 + beads)
    q = rng.normal(size=(beads, len(masses), 3))
    p = rng.normal(size=q.shape)
    noise = rng.normal(size=q.shape)
    decay = np.linspace(0.2, 0.9, beads)
    amplitude = np.linspace(0.1, 0.3, beads)
    fixed = np.zeros(len(masses), dtype=np.uint8)

    expected_q = q.reshape(beads, -1).copy()
    expected_p = p.reshape(beads, -1).copy()
    flattened_noise = noise.reshape(beads, -1)
    wave = np.arange(beads)
    omega = 2 * beads / beta * np.sin(
        np.pi * np.minimum(wave, beads - wave) / beads
    )
    theta = 0.5 * dt * omega
    if scheme == "normal-mode-bab":
        half_a = np.cos(theta)
        half_b = np.empty_like(theta)
        zero = theta == 0.0
        half_b[zero] = 0.5 * dt
        half_b[~zero] = 0.5 * dt * np.sin(theta[~zero]) / theta[~zero]
        half_c = omega * np.sin(theta)
    else:
        denominator = np.sqrt(1.0 + theta * theta)
        half_a = 1.0 / denominator
        half_b = 0.5 * dt / denominator
        half_c = 0.5 * dt * omega * omega / denominator
    scale0 = np.sqrt(float(beads))
    scale = np.sqrt(0.5 * beads)
    for coordinate in range(expected_q.shape[1]):
        mass = masses[coordinate // 3]
        qk = np.fft.fft(expected_q[:, coordinate])
        pk = np.fft.fft(expected_p[:, coordinate])
        for mode_index in range(beads):
            x = half_a[mode_index] * qk[mode_index] + half_b[mode_index] * pk[mode_index] / mass
            u = half_a[mode_index] * pk[mode_index] - half_c[mode_index] * mass * qk[mode_index]
            if mode_index == 0:
                mode = 0
                random_value = scale0 * flattened_noise[0, coordinate]
            elif beads % 2 == 0 and mode_index == beads // 2:
                mode = beads - 1
                random_value = scale0 * flattened_noise[beads - 1, coordinate]
            else:
                frequency = min(mode_index, beads - mode_index)
                mode = 2 * frequency - 1
                sign = -1.0 if mode_index < beads - mode_index else 1.0
                random_value = scale * (
                    flattened_noise[mode, coordinate]
                    + 1j * sign * flattened_noise[mode + 1, coordinate]
                )
            sine_mode = mode if (
                mode_index == 0 or (beads % 2 == 0 and mode_index == beads // 2)
            ) else mode + 1
            damped = (
                decay[mode] * u.real + amplitude[mode] * random_value.real
                + 1j * (decay[sine_mode] * u.imag + amplitude[sine_mode] * random_value.imag)
            )
            qk[mode_index] = half_a[mode_index] * x + half_b[mode_index] * damped / mass
            pk[mode_index] = half_a[mode_index] * damped - half_c[mode_index] * mass * x
        expected_q[:, coordinate] = np.fft.ifft(qk).real
        expected_p[:, coordinate] = np.fft.ifft(pk).real

    context = native.create(masses, beta, beads, dt, 1, scheme)
    native.sample_free(context, q, p, noise, decay, amplitude, fixed)
    np.testing.assert_allclose(q, expected_q.reshape(q.shape), atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(p, expected_p.reshape(p.shape), atol=3e-12, rtol=3e-12)
