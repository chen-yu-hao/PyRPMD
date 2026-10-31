"""Finite-time self-diffusion estimates from unwrapped centroid trajectories.

The Einstein estimator uses the slope of the displacement second moment without
subtracting a fitted particle drift.  Coordinates must already be continuous
across periodic boundaries; independently wrapped beads or atom positions are
not accepted substitutes.  Optional reference coordinates remove the motion
of a declared lattice centre of mass.  No sampling thermostat, fictitious
mass, or PIMD sampling time is interpreted as physical RPMD dynamics here.

Multiple time origins improve a mean, but are not independent measurements.
Errors are therefore computed across independent replicas.  An explicit long
non-overlapping block length enables a conditional batch estimate for one
trajectory; independence of these batches remains a user convergence check.
Linear MSD and a stable slope are only diagnostics, never proof of physical
convergence.  In particular, no observed motion does not establish D = 0.

The centroid Einstein relation follows Kataoka et al., Phys. Rev. Research 6,
043224 (2024), doi:10.1103/PhysRevResearch.6.043224.  The tensor extension and
uncertainty/finite-record diagnostics below use the same displacement data.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from ..constants import ATOMIC_TIME_S, AU_TIME_FS, BOHR_M
from ..validation import integer_count

DIFFUSION_AU_TO_M2_S = BOHR_M**2 / ATOMIC_TIME_S


_integer = partial(
    integer_count,
    type_message="{name} must be an integer",
    bounds_message="{name} must be >= {minimum}",
)


def _trajectory_array(value, name, *, copy=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")
    result = (np.array(raw, dtype=float, order="C", copy=True) if copy
              else np.asarray(raw, dtype=float))
    if result.ndim == 3:
        result = result[None, ...]
    if result.ndim != 4 or result.shape[-1] != 3 or min(result.shape) < 1:
        raise ValueError(f"{name} needs shape [replica,frame,atom,3] or [frame,atom,3]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _indices(value, size, name):
    if value is None:
        return tuple(range(size))
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of unique integer indices") from exc
    indices = tuple(_integer(i, name) for i in values)
    if not indices or len(set(indices)) != len(indices) or max(indices) >= size:
        raise ValueError(f"{name} must be nonempty, unique and in 0..{size - 1}")
    return indices


def _correlation(values, max_lag):
    """All-origin, atom-averaged C_ab(l) = <x_a(t) x_b(t+l)>.

    Zero padding prevents periodic wraparound of the *time* series.  FFTs
    avoid quadratic time for long saved trajectories.  Correlations retain
    their per-replica identity; atoms are never counted as independent errors.
    """
    _, frames, atoms, dimension = values.shape
    nfft = 1 << (2 * frames - 1).bit_length()
    transform = np.fft.rfft(values, n=nfft, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        spectrum = np.einsum("rfna,rfnb->rfab", transform.conj(), transform,
                             optimize=True) / atoms
    if not np.isfinite(spectrum).all():
        raise ValueError("trajectory magnitudes overflow the correlation estimator")
    sums = np.fft.irfft(spectrum, n=nfft, axis=1)[:, :max_lag + 1]
    return sums / (frames - np.arange(max_lag + 1))[None, :, None, None]


def _msd_tensor(positions, max_lag):
    # Translation reduces cancellation in the squared-coordinate identity.
    centered = positions - positions.mean(axis=1, keepdims=True)
    _, frames, atoms, dimension = centered.shape
    products = np.einsum("rfna,rfnb->rfab", centered, centered,
                         optimize=True) / atoms
    prefix = np.concatenate((np.zeros((len(centered), 1, dimension, dimension)),
                             np.cumsum(products, axis=1)), axis=1)
    lag = np.arange(max_lag + 1)
    edge = (prefix[:, -1:, :, :] - prefix[:, lag, :, :] +
            prefix[:, frames - lag, :, :])
    edge /= (frames - lag)[None, :, None, None]
    correlation = _correlation(centered, max_lag)
    msd = edge - correlation - correlation.swapaxes(-1, -2)
    msd = (msd + msd.swapaxes(-1, -2)) / 2
    if not np.isfinite(msd).all():
        raise ValueError("trajectory magnitudes overflow the displacement estimator")
    msd[:, 0] = 0.0
    # Only floating-point cancellation can produce negative diagonal MSDs.
    for axis in range(dimension):
        msd[..., axis, axis] = np.maximum(msd[..., axis, axis], 0.0)
    return msd


def _fit_tensor(lag_time, msd, selection):
    time = lag_time[selection]
    values = msd[selection]
    scaled_time = time / time[-1]
    offset = scaled_time - scaled_time.mean()
    slope = np.einsum("l,lab->ab", offset, values) / np.dot(offset, offset) / time[-1]
    intercept = values.mean(axis=0) - slope * time.mean()
    if not np.isfinite(slope).all() or not np.isfinite(intercept).all():
        raise ValueError("time or displacement magnitudes overflow the diffusion fit")
    tensor = (slope + slope.T) / 4
    dimension = tensor.shape[0]
    trace = np.trace(values, axis1=-2, axis2=-1)
    fitted = np.trace(intercept) + np.trace(slope) * time
    total = float(np.sum((trace - trace.mean())**2))
    residual = float(np.sum((trace - fitted)**2))
    r_squared = 1 - residual / total if total > 0 else None
    positive = (time > 0) & (trace > 0)
    log_slope = None
    if np.count_nonzero(positive) >= 3:
        log_time = np.log(time[positive])
        log_trace = np.log(trace[positive])
        delta = log_time - log_time.mean()
        log_slope = float(np.dot(delta, log_trace - log_trace.mean()) / np.dot(delta, delta))
    scalar = float(np.trace(tensor) / dimension)
    half_slopes = None
    slope_variation = None
    if len(time) >= 6:
        # Disjoint halves are a stability diagnostic, not independent data.
        halves = [slice(None, len(time) // 2), slice(len(time) // 2, None)]
        half_slopes = []
        for part in halves:
            t, y = time[part], trace[part]
            dt = t - t.mean()
            half_slopes.append(float(np.dot(dt, y - y.mean()) /
                                     np.dot(dt, dt) / (2 * dimension)))
        scale = max(abs(scalar), np.finfo(float).tiny)
        slope_variation = float(abs(half_slopes[1] - half_slopes[0]) / scale)
    return {
        "diffusion_au": scalar,
        "diffusion_m2_s": scalar * DIFFUSION_AU_TO_M2_S,
        "diffusion_tensor_au": tensor.tolist(),
        "diffusion_tensor_m2_s": (tensor * DIFFUSION_AU_TO_M2_S).tolist(),
        "intercept_tensor_bohr2": intercept.tolist(),
        "r_squared": r_squared,
        "log_msd_slope": log_slope,
        "half_window_diffusion_au": half_slopes,
        "relative_half_window_difference": slope_variation,
        "minimum_tensor_eigenvalue_au": float(np.linalg.eigvalsh(tensor).min()),
    }


def _empty_uncertainty(bootstrap, reason):
    return {
        "method": "unavailable", "independent_units": 0,
        "quantity": "raw_fit.diffusion", "unit_kind": None,
        "bootstrap_replicates": bootstrap, "reason": reason,
        "standard_error_au": None, "standard_error_m2_s": None,
        "tensor_standard_error_au": None, "tensor_standard_error_m2_s": None,
        "confidence_level": 0.95, "confidence_interval_au": None,
        "confidence_interval_m2_s": None, "tensor_confidence_interval_au": None,
        "tensor_confidence_interval_m2_s": None,
    }


def _uncertainty(tensors, bootstrap, seed, method):
    count, dimension, _ = tensors.shape
    scalars = np.trace(tensors, axis1=-2, axis2=-1) / dimension
    tensor_error = tensors.std(axis=0, ddof=1) / np.sqrt(count)
    scalar_error = float(scalars.std(ddof=1) / np.sqrt(count))
    scalar_interval = tensor_interval = None
    if bootstrap:
        rng = np.random.default_rng(seed)
        # Bound memory even when users request many bootstrap replicates.
        estimates = np.empty((bootstrap, dimension, dimension))
        for start in range(0, bootstrap, 256):
            take = min(256, bootstrap - start)
            indices = rng.integers(0, count, size=(take, count))
            estimates[start:start + take] = tensors[indices].mean(axis=1)
        tensor_interval = np.quantile(estimates, [0.025, 0.975], axis=0)
        scalar_interval = np.quantile(
            np.trace(estimates, axis1=-2, axis2=-1) / dimension, [0.025, 0.975])
    return {
        "method": method, "independent_units": count,
        "quantity": "raw_fit.diffusion",
        "unit_kind": "replica" if method == "independent_replicas" else "nonoverlapping_block",
        "bootstrap_replicates": bootstrap, "reason": None,
        "standard_error_au": scalar_error,
        "standard_error_m2_s": scalar_error * DIFFUSION_AU_TO_M2_S,
        "tensor_standard_error_au": tensor_error.tolist(),
        "tensor_standard_error_m2_s": (tensor_error * DIFFUSION_AU_TO_M2_S).tolist(),
        "confidence_level": 0.95,
        "confidence_interval_au": None if scalar_interval is None else scalar_interval.tolist(),
        "confidence_interval_m2_s": None if scalar_interval is None else
            (scalar_interval * DIFFUSION_AU_TO_M2_S).tolist(),
        "tensor_confidence_interval_au": None if tensor_interval is None else tensor_interval.tolist(),
        "tensor_confidence_interval_m2_s": None if tensor_interval is None else
            (tensor_interval * DIFFUSION_AU_TO_M2_S).tolist(),
    }


def analyze_diffusion(
    positions, times, *, velocities=None, atom_indices=None, axes=(0, 1, 2),
    reference_positions=None, reference_masses=None, reference_velocities=None,
    fit_range=None, max_lag=None, bootstrap=200, seed=0, block_length=None,
):
    """Analyze self-diffusion from one or more independent RPMD trajectories.

    Parameters are atomic units.  ``positions`` is unwrapped ring-centroid
    coordinates with shape ``[R,F,N,3]``; ``[F,N,3]`` denotes one replica.
    ``times[F]`` must be uniform and increasing.  ``velocities`` has the same
    shape, and contains physical centroid velocities, not finite differences.
    ``axes`` selects Cartesian components (two for a coordinate-aligned slab).
    Arbitrarily oriented surfaces must first be rotated to the analysis frame.

    ``reference_positions`` has the same R/F dimensions but may contain other
    atoms.  Its mass-weighted centre is subtracted at each frame, before atom
    selection.  Equal masses are used if ``reference_masses`` is omitted.
    Reference velocities are required for a matching reference-frame VACF.
    Individual particle drifts are never fitted/subtracted automatically.

    ``max_lag`` counts saved-frame intervals and defaults to a quarter of the
    record.  ``fit_range=(start, stop)`` is a closed interval in atomic time;
    either endpoint can be None.  Default fitting uses the latter half of the
    available positive lags.  Windows with fewer than four points are reported
    as insufficient.  OLS estimates the slope; its pointwise fit error is never
    used as a diffusion uncertainty because time origins/lags are correlated.

    For a single replica, ``block_length`` optionally specifies non-overlapping
    batches in saved frames, at least four times ``max_lag``.  At least four
    such batches are required.  Errors from batches are explicitly conditional
    on that user-chosen length exceeding the physical correlation time.

    ``diffusive_regime_candidate`` requires >=16 frames, >=8 origins at the
    largest lag, a positive semidefinite fitted tensor, log-MSD slope in
    [0.7,1.3], R² >= 0.8, and (when measurable) half-window slopes differing by
    no more than 75% of the fitted D.  These are screening heuristics, not a
    physical convergence test.  Other statuses retain ``raw_fit`` but expose
    no accepted ``diffusion_*`` values.  All return values are strict JSON data.
    """
    # Validate the full input without copying every unselected substrate atom.
    # Advanced indexing below owns the measured data before frame subtraction.
    original = _trajectory_array(positions, "positions", copy=False)
    replicas, frames, atoms, _ = original.shape
    if frames < 2:
        raise ValueError("diffusion needs at least two saved frames")
    raw_time = np.asarray(times)
    if raw_time.dtype.kind not in "iuf" or raw_time.shape != (frames,):
        raise ValueError("times must be a real vector with one value per saved frame")
    time = np.asarray(raw_time, float)
    if not np.isfinite(time).all():
        raise ValueError("times must be finite")
    differences = np.diff(time)
    dt = float(differences[0])
    if dt <= 0 or np.any(differences <= 0) or not np.allclose(
        differences, dt, rtol=1e-8, atol=np.finfo(float).eps * abs(dt) * 8
    ):
        raise ValueError("times must be strictly increasing and uniformly spaced")
    selected_atoms = _indices(atom_indices, atoms, "atom_indices")
    selected_axes = _indices(axes, 3, "axes")
    bootstrap = _integer(bootstrap, "bootstrap")
    if bootstrap == 1:
        raise ValueError("bootstrap must be zero or at least two")
    max_lag = max(1, (frames - 1) // 4) if max_lag is None else _integer(max_lag, "max_lag", 1)
    if max_lag >= frames:
        raise ValueError("max_lag must be smaller than the saved frame count")
    if block_length is not None:
        block_length = _integer(block_length, "block_length", 1)
        if block_length < 4 * max_lag:
            raise ValueError("block_length must be at least four times max_lag")
    velocity = None
    if velocities is not None:
        velocity = _trajectory_array(velocities, "velocities", copy=False)
        if velocity.shape != original.shape:
            raise ValueError("velocities must match positions")
    diagnostics = []
    reference_count = 0
    position_reference = velocity_reference = None
    if reference_positions is not None:
        reference = _trajectory_array(reference_positions, "reference_positions", copy=False)
        if reference.shape[:2] != original.shape[:2]:
            raise ValueError("reference_positions must match replica and frame counts")
        reference_count = reference.shape[2]
        if reference_masses is None:
            masses = np.ones(reference_count)
        else:
            raw_masses = np.asarray(reference_masses)
            if raw_masses.dtype.kind not in "iuf" or raw_masses.shape != (reference_count,):
                raise ValueError("reference_masses must contain one real mass per reference atom")
            masses = np.asarray(raw_masses, float)
            if not np.isfinite(masses).all() or np.any(masses <= 0):
                raise ValueError("reference_masses must be finite and positive")
        weights = masses / masses.max()
        weights /= weights.sum()
        position_reference = np.einsum("rfnc,n->rfc", reference, weights)
        if velocity is not None:
            if reference_velocities is None:
                raise ValueError("reference_velocities are required for a reference-frame VACF")
            reference_velocity = _trajectory_array(reference_velocities, "reference_velocities", copy=False)
            if reference_velocity.shape != reference.shape:
                raise ValueError("reference_velocities must match reference_positions")
            velocity_reference = np.einsum("rfnc,n->rfc", reference_velocity, weights)
        elif reference_velocities is not None:
            raise ValueError("reference_velocities require velocities")
        diagnostics.append("Coordinates are relative to the declared reference centre of mass.")
    elif reference_masses is not None or reference_velocities is not None:
        raise ValueError("reference masses/velocities require reference_positions")
    positions = original[:, :, selected_atoms, :][..., selected_axes]
    if position_reference is not None:
        positions -= position_reference[..., selected_axes][:, :, None, :]
    dimension = len(selected_axes)
    if velocity is not None:
        velocity = velocity[:, :, selected_atoms, :][..., selected_axes]
        if velocity_reference is not None:
            velocity -= velocity_reference[..., selected_axes][:, :, None, :]
    lag_time = np.arange(max_lag + 1) * dt
    default_start = lag_time[max(1, max_lag // 2)]
    lower, upper = default_start, float(lag_time[-1])
    if fit_range is not None:
        try:
            if len(fit_range) != 2:
                raise ValueError
            if any(isinstance(bound, (bool, np.bool_)) for bound in fit_range):
                raise ValueError
            lower = default_start if fit_range[0] is None else float(fit_range[0])
            upper = float(lag_time[-1]) if fit_range[1] is None else float(fit_range[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("fit_range needs two time bounds, each real or None") from exc
        if not np.isfinite([lower, upper]).all() or lower < 0 or upper <= lower:
            raise ValueError("fit_range must be finite, nonnegative and increasing")
        if upper > lag_time[-1] * (1 + 1e-12) or lower > lag_time[-1]:
            raise ValueError("fit_range extends beyond the available lag time")
    fit_mask = (lag_time > 0) & (lag_time >= lower) & (lag_time <= upper * (1 + 1e-12))
    tensor_per_replica = _msd_tensor(positions, max_lag)
    tensor = tensor_per_replica.mean(axis=0)
    msd = np.trace(tensor, axis1=-2, axis2=-1)
    output = {
        "schema": "pyrpmd.diffusion", "schema_version": 1,
        "estimator": "centroid_einstein", "status": "insufficient_data",
        "physical_converged": False,
        "diffusion_au": None, "diffusion_m2_s": None,
        "diffusion_tensor_au": None, "diffusion_tensor_m2_s": None,
        "raw_fit": None, "per_replica_fits": [],
        "uncertainty": _empty_uncertainty(bootstrap, "Insufficient points in the fitting window."),
        "lag_time_au": lag_time.tolist(), "lag_time_fs": (lag_time * AU_TIME_FS).tolist(),
        "msd_au": msd.tolist(), "msd_tensor_au": tensor.tolist(),
        "vacf_au": None, "vacf_tensor_au": None,
        "vacf_integral_au": None, "vacf_integral_tensor_au": None,
        "vacf_integral_m2_s": None, "vacf_relative_difference": None,
        "diagnostics": diagnostics,
        "input": {
            "replicas": replicas, "frames": frames, "atoms": atoms,
            "atom_indices": list(selected_atoms), "axes": list(selected_axes),
            "dimension": dimension, "time_step_au": dt, "max_lag": max_lag,
            "fit_range_au": [float(lower), float(upper)],
            "fit_lag_indices": np.flatnonzero(fit_mask).tolist(),
            "reference_atom_count": reference_count, "block_length": block_length,
            "coordinate_convention": "continuous_unwrapped_centroids",
            "length_unit": "bohr", "time_unit": "atomic", "msd_unit": "bohr^2",
            "diffusion_unit_au": "bohr^2/atomic_time", "diffusion_unit_si": "m^2/s",
        },
    }
    if velocity is not None:
        vacf = _correlation(velocity, max_lag).mean(axis=0)
        vacf = (vacf + vacf.swapaxes(-1, -2)) / 2
        integral = np.zeros_like(vacf)
        integral[1:] = np.cumsum(0.5 * (vacf[:-1] + vacf[1:]) * dt, axis=0)
        scalar_integral = np.trace(integral, axis1=-2, axis2=-1) / dimension
        output.update({
            "vacf_au": np.trace(vacf, axis1=-2, axis2=-1).tolist(),
            "vacf_tensor_au": vacf.tolist(),
            "vacf_integral_au": scalar_integral.tolist(),
            "vacf_integral_tensor_au": integral.tolist(),
            "vacf_integral_m2_s": (scalar_integral * DIFFUSION_AU_TO_M2_S).tolist(),
        })
        diagnostics.append("VACF integrals are finite-time estimates; no plateau is presumed.")
    if np.count_nonzero(fit_mask) < 4:
        diagnostics.append("At least four positive lag points are required in the diffusion fit.")
        return output
    fit = _fit_tensor(lag_time, tensor, fit_mask)
    replica_fits = [_fit_tensor(lag_time, item, fit_mask) for item in tensor_per_replica]
    output["raw_fit"] = fit
    output["per_replica_fits"] = replica_fits
    if replicas >= 2:
        fit_tensors = np.asarray([item["diffusion_tensor_au"] for item in replica_fits])
        output["uncertainty"] = _uncertainty(fit_tensors, bootstrap, seed, "independent_replicas")
        diagnostics.append("Uncertainty resamples whole independent replicas, not overlapping time origins or atoms.")
    elif block_length is not None and frames // block_length >= 4:
        count = frames // block_length
        blocks = positions[0, :count * block_length].reshape(
            count, block_length, len(selected_atoms), dimension)
        block_msd = _msd_tensor(blocks, max_lag)
        block_fits = [_fit_tensor(lag_time, item, fit_mask) for item in block_msd]
        block_tensors = np.asarray([item["diffusion_tensor_au"] for item in block_fits])
        output["uncertainty"] = _uncertainty(
            block_tensors, bootstrap, seed, "conditional_nonoverlapping_blocks")
        output["uncertainty"]["reason"] = (
            "Conditional on blocks exceeding the physical correlation time; check longer blocks and independent replicas.")
        diagnostics.append("Temporal block errors are conditional; correlation beyond a block is not excluded.")
    else:
        output["uncertainty"] = _empty_uncertainty(
            bootstrap, "One trajectory is not an independent ensemble; supply replicas or at least four long non-overlapping blocks.")
        diagnostics.append("No sampling error bar is inferred from overlapping time origins.")
    if velocity is not None:
        final_integral = output["vacf_integral_au"][-1]
        scale = max(abs(fit["diffusion_au"]), abs(final_integral), np.finfo(float).tiny)
        output["vacf_relative_difference"] = float(abs(final_integral - fit["diffusion_au"]) / scale)
    coordinate_span = np.max(np.ptp(positions, axis=1))
    no_motion = coordinate_span == 0 or msd.max() <= np.finfo(float).eps * coordinate_span**2
    alpha = fit["log_msd_slope"]
    if no_motion:
        status = "no_observed_motion"
        diagnostics.append("No resolved displacement is observed; this does not establish zero diffusivity.")
    elif frames < 16 or frames - max_lag < 8:
        status = "insufficient_data"
        diagnostics.append("The record is too short for the finite-record diffusion screen.")
    elif alpha is not None and alpha > 1.5:
        status = "ballistic_or_drifting"
        diagnostics.append("MSD grows faster than a diffusion-like window; inspect ballistic motion and the lattice reference.")
    elif (fit["diffusion_au"] <= 0 or alpha is None or not 0.7 <= alpha <= 1.3 or
          fit["r_squared"] is None or fit["r_squared"] < 0.8):
        status = "non_diffusive_window"
        diagnostics.append("The selected window is not approximately linear diffusive growth; inspect caging, transients or sampling.")
    elif fit["minimum_tensor_eigenvalue_au"] < -1e-10 * max(abs(fit["diffusion_au"]), np.finfo(float).tiny):
        status = "indefinite_diffusion_tensor"
        diagnostics.append("The fitted tensor is not positive semidefinite; increase sampling or revise the fit window.")
    elif (fit["relative_half_window_difference"] is not None and
          fit["relative_half_window_difference"] > 0.75):
        status = "unstable_fit_window"
        diagnostics.append("Diffusion estimates vary substantially between halves of the selected fit window.")
    else:
        status = "diffusive_regime_candidate"
        for key in ("diffusion_au", "diffusion_m2_s", "diffusion_tensor_au", "diffusion_tensor_m2_s"):
            output[key] = fit[key]
        diagnostics.append("Linear growth is a candidate diffusion regime, not a physical convergence certificate.")
    output["status"] = status
    diagnostics.append("Bead count, time step, cell size, production length, model accuracy and fit-window convergence remain separate checks.")
    return output


__all__ = ["analyze_diffusion", "DIFFUSION_AU_TO_M2_S"]
