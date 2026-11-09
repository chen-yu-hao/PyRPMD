"""Reproducible three-dimensional quantum-correction stress test.

Run from the repository root with::

    PYTHONPATH=src python examples/three_d_quantum_correction.py

The quantum reference is a finite normal-mode Fock-space Kubo sum for a coupled
three-dimensional harmonic oscillator.  The RPMD reference is the analytic
classical Gaussian result for the same quadratic model.  The intermediate
subspace result is compared with an independent full-system quantum response;
the full-subspace algebraic limit is checked separately.
"""

from __future__ import annotations

import json

import numpy as np

from pyrpmd.analysis import (
    causal_susceptibility,
    frequency_response,
    run_quantum_correction,
)


def kubo_quadratic(operator, energies, populations, beta, times):
    mean = float(np.sum(populations * np.diag(operator).real))
    centered = operator - mean * np.eye(operator.shape[0])
    gap = energies[None, :] - energies[:, None]
    coefficient = np.empty_like(gap, dtype=float)
    mask = np.abs(gap) > 1.0e-12
    coefficient[mask] = (populations[:, None] - populations[None, :])[mask] / (beta * gap[mask])
    coefficient[~mask] = np.broadcast_to(populations[:, None], gap.shape)[~mask]
    weight = coefficient * np.abs(centered) ** 2
    correlation = np.sum(
        weight[None, :, :] * np.exp(1j * times[:, None, None] * gap[None, :, :]),
        axis=(1, 2),
    )
    return correlation.real, mean


def run_three_dimensional_test(nmax=7, beta=2.0):
    force_constants = np.array(
        [[1.00, 0.12, 0.05], [0.12, 1.40, 0.08], [0.05, 0.08, 1.80]],
        dtype=float,
    )
    eigenvalues, normal_modes = np.linalg.eigh(force_constants)
    omega = np.sqrt(eigenvalues)
    levels = np.arange(nmax + 1)
    states = np.array(np.meshgrid(levels, levels, levels, indexing="ij")).reshape(3, -1).T
    energies = (states + 0.5) @ omega
    populations = np.exp(-beta * (energies - energies.min()))
    populations /= populations.sum()
    thermal_tail = float(1.0 - np.prod(1.0 - np.exp(-beta * omega * (nmax + 1))))

    mode_dim = nmax + 1
    mode_coordinates = []
    for w in omega:
        q = np.zeros((mode_dim, mode_dim))
        for n in range(nmax):
            q[n + 1, n] = q[n, n + 1] = np.sqrt(n + 1) / np.sqrt(2.0 * w)
        mode_coordinates.append(q)
    normal_coordinates = []
    for mode in range(3):
        factors = [mode_coordinates[mode] if axis == mode else np.eye(mode_dim) for axis in range(3)]
        normal_coordinates.append(np.kron(np.kron(factors[0], factors[1]), factors[2]))
    cartesian = [
        sum(normal_modes[i, j] * normal_coordinates[j] for j in range(3))
        for i in range(3)
    ]

    a_sub = cartesian[0] @ cartesian[0]
    a_full = sum(coordinate @ coordinate for coordinate in cartesian)
    time = np.linspace(0.0, 30.0, 601)
    frequencies = np.linspace(0.03, 3.5, 180)
    cq_sub, mean_sub = kubo_quadratic(a_sub, energies, populations, beta, time)
    cq_full, mean_full = kubo_quadratic(a_full, energies, populations, beta, time)

    covariance = np.array(
        [
            np.sum(
                (normal_modes[i] * normal_modes[j])[None, :] / (beta * omega[None, :] ** 2)
                * np.cos(time[:, None] * omega[None, :]),
                axis=1,
            )
            for i in range(3)
            for j in range(3)
        ]
    ).reshape(3, 3, time.size)
    cr_sub = 2.0 * covariance[0, 0] ** 2
    cr_full = 2.0 * np.sum(covariance**2, axis=(0, 1))

    result = run_quantum_correction(
        time,
        frequencies,
        cr_full,
        cr_sub,
        cq_sub,
        beta=beta,
        damping=0.08,
        provenance={
            "system": "coupled three-dimensional harmonic oscillator",
            "force_constants": force_constants.tolist(),
            "normal_frequencies": omega.tolist(),
            "basis": f"{mode_dim}^3 normal-mode Fock truncation",
            "basis_dimension": int(mode_dim**3),
            "observable_full": "x^2 + y^2 + z^2",
            "observable_subspace": "x^2",
            "rpmd_reference": "analytic classical Gaussian quadratic correlation",
            "quantum_reference": "finite spectral Kubo sum",
        },
    )
    q_full_response = frequency_response(
        time,
        causal_susceptibility(time, cq_full, beta),
        frequencies,
        damping=0.08,
    )
    intermediate_error = float(
        np.linalg.norm(result.response - q_full_response)
        / max(np.linalg.norm(q_full_response), np.finfo(float).eps)
    )

    # In the full-subspace limit, the inverse-response defect must cancel the
    # RPMD error algebraically, independent of the nonzero defect magnitude.
    full_limit = run_quantum_correction(
        time, frequencies, cr_full, cr_full, cq_full, beta=beta, damping=0.08
    )
    full_limit_error = float(np.max(np.abs(full_limit.response - q_full_response)))

    return {
        "force_constants": force_constants.tolist(),
        "normal_frequencies": omega.tolist(),
        "basis_dimension": int(mode_dim**3),
        "fock_levels_per_mode": int(mode_dim),
        "thermal_omitted_bound": thermal_tail,
        "time_points": int(time.size),
        "frequency_points": int(frequencies.size),
        "quantum_sub_correlation_at_zero": float(cq_sub[0]),
        "rpmd_sub_correlation_at_zero": float(cr_sub[0]),
        "quantum_full_correlation_at_zero": float(cq_full[0]),
        "rpmd_full_correlation_at_zero": float(cr_full[0]),
        "quantum_sub_mean": mean_sub,
        "quantum_full_mean": mean_full,
        "inverse_defect_rms": float(result.diagnostics["inverse_defect_rms"]),
        "relative_inverse_defect": float(result.diagnostics["relative_inverse_defect"]),
        "negative_real_fraction_full": float(result.diagnostics["negative_real_fraction_full"]),
        "negative_real_fraction_corrected": float(result.diagnostics["negative_real_fraction_corrected"]),
        "corrected_condition_number_max": float(result.diagnostics["corrected_condition_number_max"]),
        "corrected_effective_rank_min": int(result.diagnostics["corrected_effective_rank_min"]),
        "intermediate_subspace_vs_full_quantum_relative_rms": intermediate_error,
        "full_subspace_max_abs_error": full_limit_error,
        "full_subspace_inverse_defect_rms": float(full_limit.diagnostics["inverse_defect_rms"]),
        "physical_converged": bool(result.physical_converged),
    }


if __name__ == "__main__":
    print(json.dumps(run_three_dimensional_test(), indent=2))
