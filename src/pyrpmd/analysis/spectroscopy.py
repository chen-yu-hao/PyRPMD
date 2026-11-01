"""Correlations and spectra of explicitly supplied RPMD/TRPMD observables.

Trajectories must already sample the equilibrium ensemble and use physical
production time. A PILE-L/Andersen PIMD sampling trajectory is not a dynamical
correlation trajectory. Bead-averaged position observables and centroid
velocities give the usual RPMD/TRPMD approximations to Kubo correlations;
this module does not turn them into exact quantum dynamics.

All times are atomic time. Dipoles for IR must use atomic dipole units. The
one-sided periodogram is a density with respect to cycles per atomic time;
frequency and wavenumber conversions do not silently rescale that density.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import c

from ..constants import ATOMIC_TIME_S


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain finite real numeric values")
    result = np.array(raw, dtype=float, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numeric values")
    return result


def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isreal(value) or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive real number")
    return float(value)


def _series(value, name):
    """Canonical [replica, time, component], with no flattening of replicas."""
    result = _real_array(value, name)
    if result.ndim == 1:
        result = result[None, :, None]
    elif result.ndim == 2:
        result = result[None, :, :]
    if result.ndim != 3 or min(result.shape) < 1 or result.shape[1] < 2:
        raise ValueError(
            f"{name} needs [time], [time,component], or [replica,time,component] "
            "with at least two frames"
        )
    return result


def _replica_error(values):
    # Time origins and Cartesian components are correlated. Neither is treated
    # as an independent sample in an error bar.
    return (None if len(values) < 2
            else np.std(values, axis=0, ddof=1) / np.sqrt(len(values)))


def centroid_velocities(momenta, masses):
    """Convert real-mass bead momenta to v_c=<p>_beads/m in atomic units.

    Input: [time,P,N,3] or [replica,time,P,N,3]. Output drops the bead axis.
    The caller decides whether removing a molecular/box COM velocity is part of
    the observable; no drift or rotational projection is applied implicitly.
    """
    p = _real_array(momenta, "momenta")
    m = _real_array(masses, "masses")
    if p.ndim not in (4, 5) or p.shape[-1] != 3 or min(p.shape) < 1:
        raise ValueError("momenta need [time,P,N,3] or [replica,time,P,N,3]")
    if m.ndim != 1 or len(m) != p.shape[-2] or np.any(m <= 0):
        raise ValueError("masses must be positive real masses matching N")
    return p.mean(axis=-3) / m[:, None]


def bead_averaged_dipoles(dipoles):
    """Average a dipole surface evaluated on each bead, not on the centroid.

    Input: [time,P,3] or [replica,time,P,3]. For a nonlinear dipole surface,
    mu(q_centroid) and mean_beads(mu(q_bead)) are different estimators.
    """
    mu = _real_array(dipoles, "dipoles")
    if mu.ndim not in (3, 4) or mu.shape[-1] != 3 or min(mu.shape) < 1:
        raise ValueError("dipoles need [time,P,3] or [replica,time,P,3]")
    return mu.mean(axis=-2)


def time_correlation(
    values, sample_interval, *, other=None, max_lag=None, demean=False,
    method="fft",
):
    """Compute C_AB(l)=mean_r mean_t sum_d A[r,t,d]*B[r,t+l,d].

    ``values`` and ``other`` have shape [time], [time,component], or
    [replica,time,component]. Scalar replicas therefore need a trailing length-1
    component axis. All replicas have the same number of frames and receive
    equal weight. The denominator per replica is T-l (the available origins),
    not T. No products cross a replica boundary. By default the full observable
    correlation is retained; ``demean=True`` subtracts each replica's time mean.

    ``standard_error`` uses independent replicas only, and is None for one
    replica. It does not establish independence or physical convergence.
    """
    dt = _positive(sample_interval, "sample_interval")
    a = _series(values, "values")
    b = a.copy() if other is None else _series(other, "other")
    if a.shape != b.shape:
        raise ValueError("values and other must have identical shapes")
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError("demean must be boolean")
    if method not in ("fft", "direct"):
        raise ValueError("method must be fft or direct")
    frames = a.shape[1]
    lag = frames // 2 if max_lag is None else _integer(max_lag, "max_lag", 0)
    if lag >= frames:
        raise ValueError("max_lag must be less than the number of frames")
    if demean:
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
    origins = frames - np.arange(lag + 1)
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "fft":
            # At least 2*T-1 points prevent cyclic aliasing of time origins.
            nfft = 1 << (2 * frames - 2).bit_length()
            fa = np.fft.rfft(a, n=nfft, axis=1)
            fb = np.fft.rfft(b, n=nfft, axis=1)
            cross = np.sum(fa.conj() * fb, axis=2)
            per_replica = np.fft.irfft(cross, n=nfft, axis=1)[:, :lag + 1]
            per_replica /= origins[None, :]
        else:
            per_replica = np.stack([
                np.sum(a[:, :frames - shift] * b[:, shift:], axis=(1, 2))
                / (frames - shift) for shift in range(lag + 1)
            ], axis=1)
    if not np.isfinite(per_replica).all():
        raise ValueError("observable magnitudes overflow the correlation estimator")
    with np.errstate(over="ignore", invalid="ignore"):
        lag_time = dt * np.arange(lag + 1)
    if not np.isfinite(lag_time).all():
        raise ValueError("lag times overflow; check sample_interval")
    return {
        "lag_time_au": lag_time,
        "correlation": per_replica.mean(axis=0),
        "per_replica": per_replica,
        "standard_error": _replica_error(per_replica),
        "origins_per_replica": origins,
        "replicas": len(a),
        "sample_interval_au": dt,
        "demeaned": bool(demean),
        "normalization": "dot product; available time origins; equal replica weights",
    }


def velocity_autocorrelation(momenta, masses, sample_interval, **options):
    """Atom-averaged dot product of centroid velocities (Kubo approximation)."""
    velocity = centroid_velocities(momenta, masses)
    if velocity.ndim == 3:
        velocity = velocity[None, ...]
    atoms = velocity.shape[-2]
    result = time_correlation(
        velocity.reshape(*velocity.shape[:2], -1), sample_interval, **options,
    )
    for key in ("correlation", "per_replica", "standard_error"):
        if result[key] is not None:
            result[key] = result[key] / atoms
    result["observable"] = "atom-averaged centroid-velocity dot product"
    result["estimator"] = "RPMD/TRPMD approximation to the Kubo velocity correlation"
    return result


def dipole_autocorrelation(dipoles, sample_interval, **options):
    """Dot product of bead-averaged dipoles, retaining nonlinear surface values."""
    result = time_correlation(bead_averaged_dipoles(dipoles), sample_interval, **options)
    result["observable"] = "bead-averaged total-dipole dot product"
    result["estimator"] = "RPMD/TRPMD approximation to the Kubo dipole correlation"
    return result


def flux_correlation(flux, sample_interval, *, side=None, **options):
    """Signed equilibrium flux-flux or flux-side correlation, without a rate.

    Supply the scalar flux observable itself, including the chosen delta-surface
    measure/bin normalization if appropriate. Shape is [time] or [replica,time].
    ``side`` has the same shape and lies in [0,1]. The result does not divide by
    a reactant population, entrance flux, or volume, and is not an automatic
    substitute for the constrained parent/paired-child recrossing estimator.
    """
    f = _real_array(flux, "flux")
    if f.ndim not in (1, 2):
        raise ValueError("flux needs [time] or [replica,time]")
    b = None
    if side is not None:
        b = _real_array(side, "side")
        if b.shape != f.shape or np.any((b < 0) | (b > 1)):
            raise ValueError("side must match flux shape and have values in [0,1]")
        b = b[..., None]
    result = time_correlation(f[..., None], sample_interval, other=b, **options)
    result["observable"] = "signed flux-flux" if side is None else "signed flux-side"
    result["rate_normalized"] = False
    return result


def power_spectrum(
    values, sample_interval, *, demean=True, window="hann", nfft=None,
):
    """Nonnegative one-sided modified periodogram, averaged across replicas.

    Shapes match ``time_correlation``. Components are summed as a vector dot
    product. S(f)=dt*|FFT(w*x)|^2/sum(w^2), doubled except at DC/Nyquist.
    The power density is per cycles/atomic-time, even though convenient Hz and
    cm^-1 frequency axes are also returned. Zero padding interpolates the
    frequency grid and does not improve the resolution of the finite record.
    """
    dt = _positive(sample_interval, "sample_interval")
    x = _series(values, "values")
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError("demean must be boolean")
    if demean:
        x -= x.mean(axis=1, keepdims=True)
    frames = x.shape[1]
    if window == "hann":
        weights = np.hanning(frames)
    elif window in ("boxcar", None):
        weights = np.ones(frames)
    else:
        raise ValueError("window must be hann, boxcar or None")
    normalization = np.dot(weights, weights)
    if normalization <= 0:
        raise ValueError("window has zero weight; use more frames or a boxcar window")
    size = (1 << (frames - 1).bit_length()
            if nfft is None else _integer(nfft, "nfft", frames))
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.fft.rfft(x * weights[None, :, None], n=size, axis=1)
        per_replica = dt / normalization * np.sum(np.abs(transformed) ** 2, axis=2)
        per_replica[:, 1:-1 if size % 2 == 0 else None] *= 2
    if not np.isfinite(per_replica).all():
        raise ValueError("observable magnitudes overflow the spectrum estimator")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        frequency = np.fft.rfftfreq(size, d=dt)
        frequency_hz = frequency / ATOMIC_TIME_S
        duration = frames * dt
    if not np.isfinite(frequency_hz).all() or not np.isfinite(duration):
        raise ValueError("frequency/time range overflows; check sample_interval")
    return {
        "frequency_au": frequency,
        "angular_frequency_au": 2 * np.pi * frequency,
        "frequency_hz": frequency_hz,
        "wavenumber_cm1": frequency_hz / (c * 100),
        "power": per_replica.mean(axis=0),
        "per_replica": per_replica,
        "standard_error": _replica_error(per_replica),
        "frequency_density_unit": "per cycles/atomic-time",
        "sample_interval_au": dt,
        "record_duration_au": duration,
        "resolution_au": 1 / duration,
        "replicas": len(x),
        "window": "boxcar" if window is None else window,
        "demeaned": bool(demean),
    }


def relative_ir_spectrum(
    dipoles, sample_interval, *, beta, normalize=True, **options,
):
    """Relative IR from beta*omega^2 times the Kubo dipole power estimate.

    ``dipoles`` are bead dipoles in atomic units; beta is inverse hartree. No
    extra harmonic quantum correction factor is applied to a Kubo estimate.
    The electromagnetic/volume/refractive-index prefactor is not supplied, so
    this is neither absolute absorbance nor an absorption cross section.
    Multiplying finite-record spectra by omega^2 can amplify leakage; check
    record length, sampling cadence, and the chosen window.

    With peak normalization, ``relative_ir_standard_error`` divides the
    unnormalized replica standard error by the common fitted peak scale. It
    conditions on that scale and does not propagate uncertainty in its value
    or in the location of the maximum. Use ``normalize=False`` when comparing
    quantitative replica uncertainty without this fitted normalization.
    """
    inverse_temperature = _positive(beta, "beta")
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be boolean")
    result = power_spectrum(bead_averaged_dipoles(dipoles), sample_interval, **options)
    with np.errstate(over="ignore", invalid="ignore"):
        factor = inverse_temperature * result["angular_frequency_au"] ** 2
        raw = result["power"] * factor
    if not np.isfinite(raw).all():
        raise ValueError("IR intensity overflows; check beta, sample interval and dipole units")
    scale = float(np.max(raw)) if normalize else 1.0
    if scale == 0:
        scale = 1.0
    result["relative_ir"] = raw / scale
    result["relative_ir_standard_error"] = (
        None if result["standard_error"] is None
        else result["standard_error"] * factor / scale
    )
    result["unnormalized_relative_ir"] = raw
    result["peak_normalized"] = bool(normalize)
    result["normalization_scale"] = scale
    result["absolute_intensity"] = False
    result["beta_au"] = inverse_temperature
    result["estimator"] = "beta*omega^2 times RPMD/TRPMD Kubo dipole spectrum"
    return result


def bead_averaged_polarizabilities(polarizabilities):
    """Average a real symmetric total polarizability surface over beads.

    Input: [time,P,3,3] or [replica,time,P,3,3], in atomic polarizability
    units. Lab-frame axes, including the normal to an interface, are preserved.
    """
    alpha = _real_array(polarizabilities, "polarizabilities")
    if alpha.ndim not in (4, 5) or alpha.shape[-2:] != (3, 3) or min(alpha.shape) < 1:
        raise ValueError("polarizabilities need [time,P,3,3] or [replica,time,P,3,3]")
    if not np.allclose(alpha, alpha.swapaxes(-1, -2), rtol=1e-8, atol=1e-10):
        raise ValueError("nonresonant polarizability tensors must be symmetric")
    return alpha.mean(axis=-3)


def relative_raman_spectrum(
    polarizabilities, sample_interval, *, beta=None,
    laser_wavenumber_cm1=None, normalize=True, **options,
):
    """Isotropic/anisotropic nonresonant Raman spectra from a Kubo estimator.

    Let a=Tr(alpha)/3 and b=sqrt(3/2)*(alpha-a*I). The scalar a spectrum and
    the Frobenius b spectrum give the rotational invariants I_iso and I_aniso.
    VV=45*I_iso+4*I_aniso and VH=3*I_aniso; the unpolarized sum is 45*I_iso
    +7*I_aniso. Tensor components are correlated and never counted as replicas.

    With ``beta=None``, outputs are reduced Kubo band shapes. Supplying beta
    converts the Kubo approximation to the positive-frequency Stokes spectrum
    via beta*omega/(1-exp(-beta*omega)); this is a detailed-balance conversion,
    not another correction to a quantum Kubo correlation. An optional positive
    laser wavenumber also supplies (omega_laser-omega)**4 and removes bins at or
    above the laser frequency. Electromagnetic, volume and instrument factors
    are absent, so these remain relative intensities.

    Peak normalization uses one common unpolarized peak. Replica error bars
    condition on that scale, as in ``relative_ir_spectrum``.
    """
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be boolean")
    alpha = bead_averaged_polarizabilities(polarizabilities)
    if alpha.ndim == 3:
        alpha = alpha[None, ...]
    isotropic = np.trace(alpha, axis1=-2, axis2=-1) / 3
    anisotropic = np.sqrt(1.5) * (alpha - isotropic[..., None, None] * np.eye(3))
    iso = power_spectrum(isotropic[..., None], sample_interval, **options)
    aniso = power_spectrum(anisotropic.reshape(*alpha.shape[:2], 9), sample_interval, **options)
    omega = iso["angular_frequency_au"]
    factor = np.ones_like(omega)
    if beta is not None:
        inverse_temperature = _positive(beta, "beta")
        with np.errstate(over="ignore", invalid="ignore"):
            x = inverse_temperature * omega
        if not np.isfinite(x).all():
            raise ValueError("Raman detailed-balance factor overflows")
        positive = x > 0
        factor[positive] = x[positive] / -np.expm1(-x[positive])
    if laser_wavenumber_cm1 is not None:
        if beta is None:
            raise ValueError("a laser-frequency Stokes spectrum requires beta")
        laser = _positive(laser_wavenumber_cm1, "laser_wavenumber_cm1")
        laser_omega = laser * c * 100 * ATOMIC_TIME_S * 2 * np.pi
        factor *= np.maximum(laser_omega - omega, 0.0) ** 4
    per_iso = iso["per_replica"] * factor
    per_aniso = aniso["per_replica"] * factor
    per_vv = 45 * per_iso + 4 * per_aniso
    per_vh = 3 * per_aniso
    per_total = per_vv + per_vh
    if not np.isfinite(per_total).all():
        raise ValueError("Raman intensity overflows; check property units and frequency range")
    raw = per_total.mean(axis=0)
    scale = float(raw.max()) if normalize else 1.0
    if scale == 0:
        scale = 1.0
    vv, vh = per_vv.mean(axis=0), per_vh.mean(axis=0)
    ratio = np.full_like(vv, np.nan)
    np.divide(vh, vv, out=ratio, where=vv > 0)
    result = {k: v for k, v in iso.items() if k not in ("power", "per_replica", "standard_error")}
    result.update({
        "relative_raman": raw / scale,
        "unnormalized_relative_raman": raw,
        "per_replica": per_total / scale,
        "standard_error": None if len(per_total) < 2 else _replica_error(per_total) / scale,
        "isotropic": per_iso.mean(axis=0) / scale,
        "anisotropic": per_aniso.mean(axis=0) / scale,
        "vv": vv / scale, "vh": vh / scale,
        "depolarization_ratio": ratio,
        "peak_normalized": bool(normalize), "normalization_scale": scale,
        "absolute_intensity": False, "beta_au": beta,
        "laser_wavenumber_cm1": laser_wavenumber_cm1,
        "estimator": "rotationally averaged Kubo polarizability spectrum",
        "representation": "reduced_kubo" if beta is None else "stokes",
        "invariant_convention": "a=Tr(alpha)/3; gamma^2=(3/2)Tr[(alpha-a*I)^2]",
    })
    return result


def _polarization_component(component):
    if isinstance(component, str):
        if len(component) != 3 or any(letter not in "xyz" for letter in component):
            raise ValueError("component must contain three axes, for example xxz")
        return tuple("xyz".index(letter) for letter in component)
    raw = np.asarray(component)
    if raw.shape != (3,) or raw.dtype.kind not in "iu" or np.any((raw < 0) | (raw > 2)):
        raise ValueError("component must be xxz-style text or three axis indices")
    return tuple(int(i) for i in raw)


def relative_sfg_spectrum(
    dipoles, polarizabilities, sample_interval, *, beta, component="xxz",
    max_lag=None, demean=True, window="cosine", nfft=None, normalize=False,
):
    """Causal complex nonresonant SFG response with its signed imaginary part.

    For component ijk, C(t)=<delta mu_k(0) delta alpha_ij(t)> and
    R(t)=-beta*dC/dt. We integrate R(t)*exp(+i*omega*t) on t>=0 with the
    trapezoidal rule. Thus reversing the dipole/interface orientation reverses
    chi, including Im(chi), while |chi|**2 is unchanged. Squaring an ordinary
    dipole-polarizability cross-periodogram would lose this causal phase and is
    deliberately not used.

    Inputs have the same bead-property shapes as the IR/Raman routines. The
    lab axes and chosen interface must be supplied by the caller. This gives
    a Kubo/classical approximate resonant response; Fresnel factors, area or
    volume normalization, incident beams and nonresonant electronic background
    are not included. The default cosine lag taper equals 1 at t=0 and 0 at
    max_lag. A symmetric Hann time-series window is inappropriate here.

    Derivatives use finite differences of the available-origin correlation;
    convergence with sampling cadence, record length and max_lag is required.
    """
    dt = _positive(sample_interval, "sample_interval")
    inverse_temperature = _positive(beta, "beta")
    if not isinstance(normalize, (bool, np.bool_)):
        raise ValueError("normalize must be boolean")
    mu = bead_averaged_dipoles(dipoles)
    alpha = bead_averaged_polarizabilities(polarizabilities)
    if mu.shape[:-1] != alpha.shape[:-2]:
        raise ValueError("dipoles and polarizabilities must have equal replica/time axes")
    if mu.ndim == 2:
        mu, alpha = mu[None, ...], alpha[None, ...]
    i, j, k = _polarization_component(component)
    corr = time_correlation(mu[..., k, None], dt, other=alpha[..., i, j, None],
                            max_lag=max_lag, demean=demean)
    count = len(corr["lag_time_au"])
    if count < 3:
        raise ValueError("SFG needs at least three lag samples (max_lag >= 2)")
    if window == "cosine":
        taper = 0.5 * (1 + np.cos(np.pi * np.arange(count) / (count - 1)))
    elif window in ("boxcar", None):
        taper = np.ones(count)
    else:
        raise ValueError("SFG lag window must be cosine, boxcar or None")
    size = (1 << (2 * count - 2).bit_length()
            if nfft is None else _integer(nfft, "nfft", count))
    response = -inverse_temperature * np.gradient(corr["per_replica"], dt, axis=1, edge_order=2)
    weighted = response * taper
    # Finite interval trapezoidal weights; zeros beyond max_lag do not add data.
    weighted[:, [0, -1]] *= 0.5
    susceptibility = dt * np.fft.rfft(weighted, n=size, axis=1).conj()
    if not np.isfinite(susceptibility).all():
        raise ValueError("SFG response overflows; check property units and beta")
    frequency = np.fft.rfftfreq(size, d=dt)
    frequency_hz = frequency / ATOMIC_TIME_S
    if not np.isfinite(frequency_hz).all():
        raise ValueError("SFG frequency range overflows; check sample_interval")
    mean = susceptibility.mean(axis=0)
    scale = float(np.abs(mean).max()) if normalize else 1.0
    if scale == 0:
        scale = 1.0
    susceptibility /= scale
    mean /= scale
    # Coherent response is averaged before squaring. Replica |chi|^2 average
    # would spuriously keep inversion-paired contributions that must cancel.
    return {
        "frequency_au": frequency, "angular_frequency_au": 2 * np.pi * frequency,
        "frequency_hz": frequency_hz, "wavenumber_cm1": frequency_hz / (c * 100),
        "susceptibility": mean, "real": mean.real, "imaginary": mean.imag,
        "relative_sfg_intensity": np.abs(mean) ** 2,
        "per_replica_susceptibility": susceptibility,
        "real_standard_error": _replica_error(susceptibility.real),
        "imaginary_standard_error": _replica_error(susceptibility.imag),
        "lag_time_au": corr["lag_time_au"], "correlation": corr["correlation"],
        "response": response.mean(axis=0), "sample_interval_au": dt,
        "component": "".join("xyz"[axis] for axis in (i, j, k)),
        "beta_au": inverse_temperature, "replicas": len(mu),
        "peak_normalized": bool(normalize), "normalization_scale": scale,
        "window": "boxcar" if window is None else window,
        "absolute_intensity": False,
        "estimator": "causal -beta*d/dt Kubo dipole-polarizability cross correlation",
        "fourier_convention": "integral_0^tmax R(t)*exp(+i*omega*t) dt",
    }
