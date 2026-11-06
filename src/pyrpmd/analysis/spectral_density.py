r"""RPMD Kubo force correlations and exponential bath representations.

This module contains the numerical part of the QE--RPMD input pipeline.  The
functions deliberately operate on arrays and do not make any assumptions about
the RPMD integrator, so a trajectory analysis can pass either a plain NumPy
array or a block-averaged Kubo correlation function.

The convention is ``hbar = 1``.  For a real Kubo correlation of a bath force
the spectral density is

.. math::

   J(\omega) = \beta\omega \int_0^\infty C_K(t)\cos(\omega t)\,dt.

The finite-time transform below uses the trapezoidal rule and an explicitly
reported window.  No positivity projection is performed unless requested;
this is important when reporting the error introduced by a finite trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import least_squares


def _validate_time_series(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time, dtype=float)
    y = np.asarray(values)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("time and values must be one-dimensional arrays of equal length")
    if t.size < 2:
        raise ValueError("at least two time points are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("time and values must be finite")
    if abs(t[0]) > 1e-13 or np.any(np.diff(t) <= 0):
        raise ValueError("time must start at zero and be strictly increasing")
    return t, y


def _window_values(n: int, kind: str | None) -> np.ndarray:
    if kind is None or str(kind).lower() in {"none", "rect", "rectangle"}:
        return np.ones(n)
    key = str(kind).lower()
    if key in {"hann", "hanning"}:
        return np.hanning(n)
    if key == "blackman":
        return np.blackman(n)
    if key == "tukey":
        # A short cosine taper is less biased than a Hann window for a
        # correlation that is already close to zero at the trajectory end.
        x = np.linspace(0.0, 1.0, n)
        a = 0.1
        w = np.ones(n)
        left = x < a / 2
        right = x > 1 - a / 2
        w[left] = 0.5 * (1 + np.cos(2 * np.pi * (x[left] / a - 0.5)))
        w[right] = 0.5 * (1 + np.cos(2 * np.pi * ((x[right] - 1) / a + 0.5)))
        return w
    raise ValueError(f"unknown window {kind!r}; use None, 'hann', 'blackman', or 'tukey'")


def spectral_density_from_kubo(
    time: Iterable[float],
    kubo_tcf: Iterable[float],
    beta: float,
    frequencies: Optional[Iterable[float]] = None,
    *,
    window: str | None = "hann",
    pad_factor: int = 4,
    positive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``J(omega)`` from a real Kubo force correlation.

    Parameters
    ----------
    time, kubo_tcf:
        Samples at nonuniform or uniform times beginning at zero.  For
        accurate FFT-like frequency spacing, use a uniform grid.
    beta:
        Inverse temperature in the same units as the frequencies.
    frequencies:
        Frequencies at which to evaluate the cosine transform.  If omitted,
        an evenly spaced grid up to the Nyquist frequency is generated.  The
        grid is zero-padded by ``pad_factor`` to make interpolation smooth;
        zero padding does not add information.
    window:
        Finite-time window.  ``"hann"`` is the conservative default.  Set
        ``None`` when the supplied correlation is known to be zero at the end.
    positive:
        If true, clip tiny/negative values caused by finite-time noise.  The
        default preserves the raw estimate for error accounting.

    Returns
    -------
    frequencies, J:
        Real arrays.  ``J[0]`` is exactly zero by definition.
    """
    t, c = _validate_time_series(np.asarray(time), np.asarray(kubo_tcf))
    if np.iscomplexobj(c):
        if np.max(np.abs(c.imag)) > 1e-10 * max(1.0, np.max(np.abs(c))):
            raise ValueError("the Kubo TCF must be real")
        c = c.real
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    dt = np.diff(t)
    if frequencies is None:
        # A direct quadrature has no FFT-grid requirement.  The Nyquist limit
        # is still a useful upper bound for a sampled trajectory.
        nyquist = 0.5 / np.min(dt)
        nfreq = max(2, int(pad_factor) * (t.size - 1) + 1)
        freq = np.linspace(0.0, nyquist, nfreq)
    else:
        freq = np.asarray(frequencies, dtype=float)
        if freq.ndim != 1 or np.any(~np.isfinite(freq)) or np.any(freq < 0):
            raise ValueError("frequencies must be a finite one-dimensional nonnegative array")
    w = _window_values(t.size, window)
    # np.trapezoid is unavailable in older NumPy versions used by some PyRPMD
    # environments, while np.trapz remains widely supported.
    integrand = c * w
    phase = np.cos(np.outer(freq, t))
    integral = np.trapz(phase * integrand[None, :], t, axis=1)
    J = beta * freq * integral
    # Only an explicitly requested zero-frequency sample is fixed to zero.
    # A user-supplied grid may legitimately start at a small positive omega;
    # overwriting its first sample would create a spurious 100% endpoint error
    # in the harmonic-bath recovery diagnostic.
    zero = np.isclose(freq, 0.0, rtol=0.0, atol=1e-15)
    J[zero] = 0.0
    if positive:
        J = np.maximum(J, 0.0)
    return freq, J


def force_tcf_to_spectral_density(
    time: Iterable[float],
    kubo_tcf: Iterable[float],
    beta: float,
    frequencies: Optional[Iterable[float]] = None,
    *,
    window: str | None = "hann",
    pad_factor: int = 4,
    positive: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Public QE--RPMD spelling for :func:`spectral_density_from_kubo`.

    The explicit alias is kept because the pipeline is normally written as
    ``force_tcf_to_spectral_density`` in validation scripts.  Both names
    implement exactly the same convention and numerical quadrature.
    """
    return spectral_density_from_kubo(
        time,
        kubo_tcf,
        beta,
        frequencies,
        window=window,
        pad_factor=pad_factor,
        positive=positive,
    )


def correlation_from_spectral_density(
    frequencies: Iterable[float],
    spectral_density: Iterable[float],
    beta: float,
    time: Iterable[float],
) -> np.ndarray:
    """Evaluate the quantum bath correlation from a spectral density.

    This implements the convention used by QE--RPMD,

    ``C(t) = 1/pi integral J(w)[coth(beta*w/2) cos(w*t) - i sin(w*t)] dw``.

    The zero-frequency value is evaluated by its limiting value and therefore
    does not produce a NaN for an Ohmic density.
    """
    omega = np.asarray(frequencies, dtype=float)
    J = np.asarray(spectral_density, dtype=float)
    t = np.asarray(time, dtype=float)
    if omega.ndim != 1 or J.ndim != 1 or omega.size != J.size:
        raise ValueError("frequencies and spectral_density must be equal-length vectors")
    if omega.size < 2 or np.any(~np.isfinite(omega)) or np.any(~np.isfinite(J)):
        raise ValueError("spectral density arrays must contain at least two finite points")
    if np.any(omega < 0) or np.any(np.diff(omega) <= 0):
        raise ValueError("frequencies must be strictly increasing and nonnegative")
    beta = float(beta)
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    if t.ndim != 1 or np.any(~np.isfinite(t)):
        raise ValueError("time must be a finite one-dimensional array")
    x = 0.5 * beta * omega
    coth = np.empty_like(x)
    nz = x != 0
    coth[nz] = 1.0 / np.tanh(x[nz])
    # For a regular Ohmic J, J*coth has a finite limit.  In the absence of a
    # slope estimate, setting the isolated endpoint to zero is the trapezoid
    # convention and avoids injecting an arbitrary delta contribution.
    coth[~nz] = 0.0
    cos_part = np.cos(np.outer(t, omega))
    sin_part = np.sin(np.outer(t, omega))
    real = np.trapz(cos_part * (J * coth)[None, :], omega, axis=1) / np.pi
    imag = -np.trapz(sin_part * J[None, :], omega, axis=1) / np.pi
    return real + 1j * imag


def spectral_density_to_correlation(
    frequencies: Iterable[float],
    spectral_density: Iterable[float],
    beta: float,
    time: Iterable[float],
) -> np.ndarray:
    """Public QE--RPMD spelling for :func:`correlation_from_spectral_density`."""
    return correlation_from_spectral_density(frequencies, spectral_density, beta, time)


def harmonic_bath_spectral_density(
    frequencies: Iterable[float],
    mode_frequencies: Iterable[float],
    couplings: Iterable[float],
    *,
    masses: Iterable[float] | float = 1.0,
    broadening: float | None = None,
) -> np.ndarray:
    """Return a discretized harmonic-bath spectral density.

    For modes with Hamiltonian ``p_j**2/(2 m_j) + m_j Ω_j**2 q_j**2/2``
    and force ``g = sum_j c_j q_j``,

    ``J(ω) = π/2 sum_j c_j²/(m_j Ω_j) δ(ω-Ω_j)``.

    A finite ``broadening`` (standard deviation in frequency units) replaces
    each delta function by a normalized Gaussian.  With ``broadening=None``
    the returned array is zero except at grid points that coincide with a
    mode to floating-point tolerance; this form is useful for constructing
    exact discrete bath checks but is not a smooth quadrature rule.
    """
    omega = np.asarray(frequencies, dtype=float)
    modes = np.asarray(mode_frequencies, dtype=float)
    c = np.asarray(couplings, dtype=float)
    if omega.ndim != 1 or omega.size < 1 or np.any(~np.isfinite(omega)):
        raise ValueError("frequencies must be a nonempty finite vector")
    if np.any(omega < 0) or (omega.size > 1 and np.any(np.diff(omega) <= 0)):
        raise ValueError("frequencies must be sorted and nonnegative")
    if modes.ndim != 1 or c.ndim != 1 or modes.size != c.size or modes.size < 1:
        raise ValueError("mode_frequencies and couplings must be equal-length nonempty vectors")
    if np.any(~np.isfinite(modes)) or np.any(modes <= 0) or np.any(~np.isfinite(c)):
        raise ValueError("mode frequencies must be positive and all inputs finite")
    if np.isscalar(masses):
        m = np.full_like(modes, float(masses))
    else:
        m = np.asarray(masses, dtype=float)
    if m.shape != modes.shape or np.any(~np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("masses must be positive and match mode_frequencies")
    weights = 0.5 * np.pi * c**2 / (m * modes)
    if broadening is None:
        out = np.zeros_like(omega)
        for Om, weight in zip(modes, weights):
            idx = int(np.argmin(np.abs(omega - Om)))
            if np.isclose(omega[idx], Om, rtol=1e-10, atol=1e-12):
                out[idx] += weight
        return out
    sigma = float(broadening)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("broadening must be finite and positive")
    out = np.zeros_like(omega)
    # Normalize each Gaussian on the supplied half-line grid.  This keeps the
    # discretized integrated weight equal to the delta-mode weight even when
    # the grid is finite and the mode lies close to a boundary.
    for Om, weight in zip(modes, weights):
        g = np.exp(-0.5 * ((omega - Om) / sigma) ** 2)
        norm = np.trapz(g, omega) if omega.size > 1 else g[0]
        if norm > 0:
            out += weight * g / norm
    return out


def harmonic_bath_kubo_tcf(
    time: Iterable[float],
    mode_frequencies: Iterable[float],
    couplings: Iterable[float],
    *,
    beta: float,
    masses: Iterable[float] | float = 1.0,
) -> np.ndarray:
    """Exact real Kubo force TCF for a linearly coupled harmonic bath.

    In the ``hbar=1`` convention this is

    ``C_K(t) = sum_j c_j²/(beta*m_j*Ω_j²) cos(Ω_j t)``.

    It is the analytic input used by the P2 harmonic-bath recovery test.
    """
    t = np.asarray(time, dtype=float)
    modes = np.asarray(mode_frequencies, dtype=float)
    c = np.asarray(couplings, dtype=float)
    if t.ndim != 1 or np.any(~np.isfinite(t)):
        raise ValueError("time must be a finite one-dimensional array")
    if modes.ndim != 1 or c.ndim != 1 or modes.size != c.size or modes.size < 1:
        raise ValueError("mode_frequencies and couplings must be equal-length nonempty vectors")
    if np.any(modes <= 0) or np.any(~np.isfinite(modes)) or np.any(~np.isfinite(c)):
        raise ValueError("mode frequencies must be positive and all inputs finite")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if np.isscalar(masses):
        m = np.full_like(modes, float(masses))
    else:
        m = np.asarray(masses, dtype=float)
    if m.shape != modes.shape or np.any(~np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("masses must be positive and match mode_frequencies")
    amplitudes = c**2 / (beta * m * modes**2)
    return np.cos(np.outer(t, modes)) @ amplitudes


@dataclass(frozen=True)
class ExponentialFit:
    """Result of :func:`fit_exponential_correlation`.

    ``exponents`` have nonnegative real parts and the model is
    ``sum(amplitudes[k] * exp(-exponents[k] * t))``.

    ``absolute_l2_error`` and ``relative_l2_error`` are computed on the
    supplied fitting grid.  The latter is normalized by ``||correlation||_2``
    (with a machine-precision floor), so both quantities can be recorded in a
    validation manifest without reconstructing the residual.
    """

    amplitudes: np.ndarray
    exponents: np.ndarray
    fitted: np.ndarray
    relative_l2_error: float
    absolute_l2_error: float
    rms_error: float
    nfev: int
    success: bool
    message: str

    def evaluate(self, time: Iterable[float]) -> np.ndarray:
        t = np.asarray(time, dtype=float)
        return np.exp(-np.outer(t, self.exponents)) @ self.amplitudes


@dataclass(frozen=True)
class SpectralPipelineResult:
    """Complete deterministic P2 conversion record.

    The object keeps the raw Kubo TCF, transformed spectral density, quantum
    correlation reconstructed from that density, and its finite exponential
    fit together.  Keeping all intermediate arrays makes it possible to write
    a validation file without rerunning any numerical step.
    """

    beta: float
    tcf_time: np.ndarray
    kubo_tcf: np.ndarray
    frequencies: np.ndarray
    spectral_density: np.ndarray
    correlation_time: np.ndarray
    correlation: np.ndarray
    fit: ExponentialFit
    window: str | None
    pad_factor: int


def analyze_kubo_tcf(
    time: Iterable[float],
    kubo_tcf: Iterable[float],
    beta: float,
    *,
    frequencies: Optional[Iterable[float]] = None,
    correlation_time: Optional[Iterable[float]] = None,
    n_terms: int = 4,
    initial_exponents: Optional[Iterable[complex]] = None,
    window: str | None = "hann",
    pad_factor: int = 4,
    positive: bool = False,
    max_nfev: int = 3000,
) -> SpectralPipelineResult:
    """Run the full TCF -> ``J`` -> ``C`` -> exponential-fit pipeline.

    ``n_terms`` and all transform choices are explicit arguments so a caller
    can freeze them before comparing against an exact reference.  The returned
    :class:`SpectralPipelineResult` exposes ``fit.relative_l2_error`` and
    ``fit.absolute_l2_error`` as the required fit diagnostics.
    """
    t, c = _validate_time_series(np.asarray(time), np.asarray(kubo_tcf))
    beta_value = float(beta)
    if not np.isfinite(beta_value) or beta_value <= 0:
        raise ValueError("beta must be finite and positive")
    freq, j = spectral_density_from_kubo(
        t,
        c,
        beta_value,
        frequencies=frequencies,
        window=window,
        pad_factor=pad_factor,
        positive=positive,
    )
    ct = t.copy() if correlation_time is None else np.asarray(correlation_time, dtype=float)
    if ct.ndim != 1 or ct.size < 2 or np.any(~np.isfinite(ct)) or np.any(ct < 0):
        raise ValueError("correlation_time must be a finite nonnegative vector")
    if ct.size > 1 and np.any(np.diff(ct) <= 0):
        raise ValueError("correlation_time must be strictly increasing")
    reconstructed = correlation_from_spectral_density(freq, j, beta_value, ct)
    fit = fit_exponential_correlation(
        ct,
        reconstructed,
        n_terms=n_terms,
        initial_exponents=initial_exponents,
        max_nfev=max_nfev,
    )
    return SpectralPipelineResult(
        beta=beta_value,
        tcf_time=t.copy(),
        kubo_tcf=np.array(c, copy=True),
        frequencies=np.array(freq, copy=True),
        spectral_density=np.array(j, copy=True),
        correlation_time=np.array(ct, copy=True),
        correlation=np.array(reconstructed, copy=True),
        fit=fit,
        window=window,
        pad_factor=int(pad_factor),
    )


def _initial_frequency_candidates(t: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Pick deterministic frequencies from the discrete Fourier spectrum."""
    if n == 0:
        return np.empty(0)
    # The complex signal can have positive and negative oscillatory poles.
    f = 2 * np.pi * np.fft.fftfreq(t.size, d=np.median(np.diff(t)))
    power = np.abs(np.fft.fft(y - y[-1]))
    order = np.argsort(power)[::-1]
    chosen: list[float] = []
    max_frequency = np.pi / np.median(np.diff(t))
    for idx in order:
        candidate = float(f[idx])
        if abs(candidate) > max_frequency * (1 - 1e-12):
            continue
        if all(abs(candidate - old) > 1e-8 for old in chosen):
            chosen.append(candidate)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        chosen.extend([0.0] * (n - len(chosen)))
    return np.asarray(chosen[:n])


def fit_exponential_correlation(
    time: Iterable[float],
    correlation: Iterable[complex],
    n_terms: int,
    *,
    initial_exponents: Optional[Iterable[complex]] = None,
    max_nfev: int = 3000,
    enforce_stability: bool = True,
) -> ExponentialFit:
    """Fit ``C(t)`` to a finite sum of damped complex exponentials.

    The fit uses variable projection: for each trial set of exponents the
    amplitudes are solved by linear least squares, while decay rates and
    frequencies are optimized nonlinearly.  This avoids a poorly conditioned
    optimization over both amplitudes and exponents and makes the reported
    residual directly comparable between choices of ``K``.

    ``initial_exponents`` is recommended when a prior pole estimate (AAA or
    ESPRIT) is available.  Without it, deterministic log-spaced decay rates
    and FFT frequency peaks are used.  No random starts are used, ensuring a
    reproducible pipeline.
    """
    t, y = _validate_time_series(np.asarray(time), np.asarray(correlation, dtype=complex))
    k = int(n_terms)
    if k < 1:
        raise ValueError("n_terms must be positive")
    if not np.iscomplexobj(y):
        y = y.astype(complex)
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")
    tmax = max(float(t[-1]), np.finfo(float).eps)
    dt = max(float(np.median(np.diff(t))), np.finfo(float).eps)
    if initial_exponents is not None:
        ex0 = np.asarray(list(initial_exponents), dtype=complex)
        if ex0.shape != (k,) or np.any(~np.isfinite(ex0)):
            raise ValueError("initial_exponents must contain n_terms finite complex values")
        if enforce_stability and np.any(ex0.real < 0):
            raise ValueError("initial_exponents must have nonnegative real parts")
        rates0 = np.maximum(ex0.real, 1.0 / tmax * 1e-6)
        freq0 = ex0.imag
    else:
        rates0 = np.geomspace(1.0 / tmax, min(20.0 / tmax, 1.0 / dt), k)
        freq0 = _initial_frequency_candidates(t, y, k)
    theta0 = np.concatenate([np.log(rates0), freq0])

    def unpack(theta: np.ndarray) -> np.ndarray:
        rates = np.exp(theta[:k])
        return rates + 1j * theta[k:]

    def solve_coeff(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ex = unpack(theta)
        design = np.exp(-np.outer(t, ex))
        coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
        return ex, coeff

    scale = max(float(np.linalg.norm(y)), np.finfo(float).eps)

    def residual(theta: np.ndarray) -> np.ndarray:
        ex, coeff = solve_coeff(theta)
        err = np.exp(-np.outer(t, ex)) @ coeff - y
        return np.concatenate([err.real, err.imag]) / scale

    result = least_squares(residual, theta0, max_nfev=int(max_nfev), method="trf")
    ex, coeff = solve_coeff(result.x)
    fitted = np.exp(-np.outer(t, ex)) @ coeff
    err = fitted - y
    rel = float(np.linalg.norm(err) / scale)
    rms = float(np.sqrt(np.mean(np.abs(err) ** 2)))
    return ExponentialFit(
        amplitudes=np.asarray(coeff),
        exponents=np.asarray(ex),
        fitted=np.asarray(fitted),
        relative_l2_error=rel,
        absolute_l2_error=float(np.linalg.norm(err)),
        rms_error=rms,
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


__all__ = [
    "ExponentialFit",
    "SpectralPipelineResult",
    "analyze_kubo_tcf",
    "correlation_from_spectral_density",
    "fit_exponential_correlation",
    "force_tcf_to_spectral_density",
    "harmonic_bath_kubo_tcf",
    "harmonic_bath_spectral_density",
    "spectral_density_to_correlation",
    "spectral_density_from_kubo",
]
