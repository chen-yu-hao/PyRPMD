"""Analysis-only uncertainty estimates for the maintained gas-water example.

Kept outside fqcmd_water.py because that file is the physical provider source:
changing it would change existing fqcmd_base_identity model fingerprints.
This module never imports or modifies the potential or its propagation code.
"""

from __future__ import annotations

import numpy as np


def stretch_statistics(wavenumber_cm1, molecular_ir, *, samples=2000, seed=0):
    """Bootstrap the OH-band peak and first moment over whole molecules.

    Each row must be an unnormalised spectrum of an independently propagated,
    noninteracting molecule, computed with the same record and uniform FFT
    frequency grid. The common bin width cancels in the band first moment.
    Frames, FFT bins and Cartesian components are never resampled as independent
    observations. The result is conditional on the fitted potential: independent
    repetitions of PIMD and IBI are still needed to assess training uncertainty.
    A grid maximum can switch between unresolved components, so its percentile
    interval need not resemble a Gaussian error bar. No interpolation of the
    peak or increase in the finite-record resolution is implied.
    """
    wave = np.asarray(wavenumber_cm1)
    values = np.asarray(molecular_ir)
    if (wave.dtype.kind not in "iuf" or wave.ndim != 1 or len(wave) < 2
            or not np.isfinite(wave).all() or np.any(np.diff(wave) <= 0)):
        raise ValueError("wavenumbers must be a finite strictly increasing vector")
    if (values.dtype.kind not in "iuf" or values.ndim != 2
            or values.shape[1] != len(wave) or values.shape[0] < 1
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("molecular IR must be a finite nonnegative [molecule,frequency] array")
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("bootstrap samples must be an integer >= 2")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    selected = (wave >= 2500) & (wave <= 4500)
    x, y = wave[selected], values[:, selected]
    if x.size < 2 or np.any(y.sum(axis=1) <= 0):
        raise ValueError("each molecular spectrum needs a resolved positive OH-band signal")
    mean = y.mean(axis=0)
    estimates = {"peak_grid_cm1": float(x[np.argmax(mean)]),
                 "band_first_moment_cm1": float(mean @ x / mean.sum())}
    result = {"analysis_band_cm1": [2500.0, 4500.0], "estimates": estimates,
              "sampling_unit": "whole independent noninteracting molecule",
              "molecules": len(y), "bootstrap_samples": int(samples),
              "bootstrap_seed": int(seed), "conditional_on_fixed_potential_and_dynamics": True,
              "includes_training_uncertainty": False,
              "per_molecule_peak_grid_cm1": x[np.argmax(y, axis=1)].tolist(),
              "per_molecule_band_first_moment_cm1": ((y @ x) / y.sum(axis=1)).tolist(),
              "standard_error_cm1": {name: None for name in estimates},
              "percentile_95_interval_cm1": {name: None for name in estimates},
              "physical_converged": False}
    if len(y) < 2:
        result["uncertainty_status"] = "unresolved: only one independent molecule"
        return result
    rng = np.random.default_rng(seed)
    # Multinomial weights equal sampling molecule rows with replacement, while
    # avoiding a [bootstrap,molecule,frequency] temporary array.
    weights = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=samples)
    boot = weights @ y / len(y)
    draws = {"peak_grid_cm1": x[np.argmax(boot, axis=1)],
             "band_first_moment_cm1": (boot @ x) / boot.sum(axis=1)}
    result["standard_error_cm1"] = {name: float(value.std(ddof=1)) for name, value in draws.items()}
    result["percentile_95_interval_cm1"] = {
        name: np.quantile(value, [0.025, 0.975], method="nearest").tolist()
        for name, value in draws.items()}
    result["uncertainty_status"] = "conditional production bootstrap; no systematic-error bound"
    return result

