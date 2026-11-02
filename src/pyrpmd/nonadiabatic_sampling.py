"""Independent harmonic ring-polymer nuclear parents in explicit coordinates."""

from __future__ import annotations

import copy
import numbers

import numpy as np

from .dynamics import State


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _real_vector(value, name, *, size=None, positive=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 1 or not len(raw):
        raise ValueError(f"{name} must be a nonempty real vector")
    result = np.array(raw, dtype=float, copy=True)
    if (not np.isfinite(result).all() or size is not None and result.shape != (size,)
            or positive and np.any(result <= 0)):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must contain {qualifier} values with the required shape")
    return result


def sample_harmonic_ring(masses, beta, nbeads, frequencies, *, coordinates=None,
                         centers=None, centroid_momenta=None, count=1, seed=0):
    """Draw independent finite-P harmonic Boltzmann nuclear ring states.

    Atomic units and ordinary, non-mass-weighted Cartesian coordinates are used.
    ``masses[N]`` gives atom masses and ``frequencies[D]`` gives positive angular
    frequencies of independent modes. ``coordinates[D]`` contains distinct
    flattened Cartesian indices in [0,3*N); its default is ``3*arange(D)``, one
    atom's x coordinate per mode. Inactive coordinates and momenta remain zero.

    ``centers[D]`` gives the position mean of every bead. Optional
    ``centroid_momenta[D]`` adds the same momentum shift to each bead; the
    realized centroid momenta still fluctuate thermally. These vectors default
    to zero. There is no electronic initialization or electronic thermal trace.

    For each mode j, q has covariance
    ``(P/beta)/m_j * inv(omega_j**2*I + (P/beta)**2*L)``, where L is the periodic
    ring Laplacian. Each bead's p is independent with variance ``m_j*P/beta``.
    This is the finite-P primitive path-integral distribution, not a Wigner
    distribution. All count states are independent draws, not steps of a chain.

    ``seed`` is a nonnegative integer or a NumPy Generator. A supplied Generator
    advances on success; if sampling fails its state is restored. Inputs and
    returned states own separate arrays, and returned clocks start at zero.
    """
    masses = _real_vector(masses, "masses", positive=True)
    frequencies = _real_vector(frequencies, "frequencies", positive=True)
    nbeads = _positive_integer(nbeads, "nbeads")
    count = _positive_integer(count, "count")
    if isinstance(beta, (bool, np.bool_)) or not isinstance(beta, numbers.Real):
        raise ValueError("beta must be a positive finite real scalar")
    try:
        beta = float(beta)
    except (OverflowError, ValueError) as exc:
        raise ValueError("beta must be a positive finite real scalar") from exc
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite real scalar")
    modes = len(frequencies)
    if coordinates is None:
        coordinates = 3*np.arange(modes, dtype=np.int64)
    else:
        coordinates = np.asarray(coordinates)
    if (coordinates.shape != (modes,) or coordinates.dtype.kind not in "iu"
            or np.any(coordinates < 0) or np.any(coordinates >= 3*len(masses))
            or len(np.unique(coordinates)) != modes):
        raise ValueError("coordinates must be distinct integer Cartesian indices, one per frequency")
    coordinates = np.array(coordinates, dtype=np.int64, copy=True)
    centers = np.zeros(modes) if centers is None else _real_vector(centers, "centers", size=modes)
    momentum_mean = (np.zeros(modes) if centroid_momenta is None
                     else _real_vector(centroid_momenta, "centroid_momenta", size=modes))
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif (not isinstance(seed, (bool, np.bool_))
          and isinstance(seed, (int, np.integer)) and seed >= 0):
        rng = np.random.default_rng(int(seed))
    else:
        raise ValueError("seed must be a nonnegative integer or a NumPy Generator")

    selected_masses = masses[coordinates//3]
    # Real-bead white noise can be filtered with the positive square root of
    # the covariance. This automatically handles the real zero/Nyquist modes,
    # including the P=1 zero spring and the double link of a P=2 ring.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        bead_temperature = nbeads/beta
        ring_frequencies = (2*np.sin(np.pi*np.arange(nbeads//2+1)/nbeads)
                            * bead_temperature)
        denominator = np.hypot(ring_frequencies[:, None], frequencies[None, :])
        position_scale = (np.sqrt(bead_temperature)/np.sqrt(selected_masses))[None, :]/denominator
        momentum_scale = np.sqrt(selected_masses)*np.sqrt(bead_temperature)
    if (not np.isfinite(bead_temperature) or bead_temperature <= 0
            or not np.isfinite(position_scale).all() or np.any(position_scale <= 0)
            or not np.isfinite(momentum_scale).all() or np.any(momentum_scale <= 0)):
        raise ValueError("harmonic ring variances cannot be represented with these parameters")

    saved_rng = copy.deepcopy(rng.bit_generator.state)
    try:
        white = rng.standard_normal((count, nbeads, modes))
        with np.errstate(over="ignore", invalid="ignore"):
            mode_positions = np.fft.irfft(
                np.fft.rfft(white, axis=1)*position_scale[None], n=nbeads, axis=1,
            ) + centers
            mode_momenta = (rng.standard_normal((count, nbeads, modes))
                            * momentum_scale + momentum_mean)
        if not np.isfinite(mode_positions).all() or not np.isfinite(mode_momenta).all():
            raise FloatingPointError("harmonic ring sampling produced nonfinite coordinates or momenta")
        q = np.zeros((count, nbeads, 3*len(masses)))
        p = np.zeros_like(q)
        q[:, :, coordinates], p[:, :, coordinates] = mode_positions, mode_momenta
        q = q.reshape(count, nbeads, len(masses), 3)
        p = p.reshape(count, nbeads, len(masses), 3)
        return tuple(State(position, momentum) for position, momentum in zip(q, p))
    except BaseException:
        rng.bit_generator.state = saved_rng
        raise
