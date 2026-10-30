"""Numerical and correlated-sampling diagnostics for saved trajectories."""

import numpy as np

from ..constants import FS_TO_AU
from .fes import statistical_inefficiency


def nve_energy_diagnostics(energies, times, *, beads, beta, tolerance_kbt=0.1):
    """Energy-zero-independent errors of H_P at the saved production frames.

    The tolerance is expressed per bead in kBT, matching the rate workflow's
    energy scale. Saved-frame diagnostics can miss excursions between frames;
    satisfying this diagnostic does not establish timestep convergence.
    """
    energy = np.asarray(energies, dtype=float)
    time = np.asarray(times, dtype=float)
    if energy.ndim == 1:
        energy = energy[None]
    if (energy.ndim != 2 or min(energy.shape) < 1
            or time.shape != (energy.shape[1],) or not np.isfinite(energy).all()
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)):
        raise ValueError("NVE diagnostics require finite energy streams and increasing times")
    if (isinstance(beads, (bool, np.bool_)) or not isinstance(beads, (int, np.integer))
            or beads < 1 or not np.isfinite(beta) or beta <= 0
            or not np.isfinite(tolerance_kbt) or tolerance_kbt <= 0):
        raise ValueError("NVE diagnostics require positive beads, beta and tolerance")
    errors = (energy - energy[:, :1]) / beads
    scaled = beta * errors
    clock = time - time[0]
    centered = clock - clock.mean()
    norm = float(centered @ centered)
    slopes = (scaled - scaled.mean(axis=1, keepdims=True)) @ centered / norm if norm > 0 else None
    per_replica = []
    for replica, row in enumerate(scaled):
        peak = float(np.max(np.abs(row)))
        per_replica.append({
            "replica": replica,
            "maximum_absolute_error_hartree_per_bead": float(np.max(np.abs(errors[replica]))),
            "maximum_absolute_error_kbt_per_bead": peak,
            "endpoint_error_kbt_per_bead": float(row[-1]),
            "linear_drift_kbt_per_bead_per_fs": None if slopes is None else float(slopes[replica] * FS_TO_AU),
            "within_tolerance": peak <= tolerance_kbt,
        })
    return {
        "scope": "saved_unthermostatted_production_frames",
        "reference": "first_saved_production_frame",
        "normalization": "(H_P(t)-H_P(0))/(P*kBT)",
        "samples_per_replica": energy.shape[1],
        "tolerance_kbt_per_bead": float(tolerance_kbt),
        "within_tolerance": all(row["within_tolerance"] for row in per_replica),
        "per_replica": per_replica,
        "physical_converged": False,
    }


def correlated_mean_diagnostics(values):
    """Conditional batch error, with the correlation estimate kept explicit.

    Four complete nonoverlapping blocks are required. The estimated block
    length is a diagnostic choice, not proof that the blocks are independent.
    The ordinary sample mean still uses all frames; only the batch error omits
    an incomplete final block.
    """
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("sampling diagnostics require a nonempty finite series")
    inefficiency = statistical_inefficiency(data)
    length = max(1, int(np.ceil(inefficiency)))
    count = len(data) // length
    available = count >= 4
    error = None
    if available:
        means = data[:count * length].reshape(count, length).mean(axis=1)
        error = float(means.std(ddof=1) / np.sqrt(count))
    return {
        "samples": len(data),
        "statistical_inefficiency": float(inefficiency),
        "effective_samples": float(len(data) / inefficiency),
        "block_length_frames": length,
        "complete_blocks": count,
        "conditional_standard_error": error,
        "uncertainty_scope": "conditional_on_block_length_exceeding_correlation_time",
        "reason": None if available else "fewer_than_four_complete_blocks",
    }
