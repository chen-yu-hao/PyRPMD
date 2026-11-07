"""Experimental force-informed active subspace and harmonic-memory embedding.

Inputs must use consistent mass-weighted coordinates and forces (hbar=1).
The cubic force fit is a small-system prototype, not a general molecular
selector. Thresholds are heuristic, not bounds on quantum dynamical error.
Embedding assumes a stable harmonic residual bath with linear coupling.
It is not an exact nonlinear system-bath or reaction-rate theory.
"""
from __future__ import annotations
from math import comb
from itertools import combinations_with_replacement
import numpy as np
from scipy.linalg import null_space, eigh


def polynomial_features(z, degree=3):
    z = np.asarray(z, float)
    powers = [()] + [p for d in range(1, degree + 1)
                    for p in combinations_with_replacement(range(z.shape[1]), d)]
    return np.column_stack([
        np.prod(z[:, p], axis=1) if p else np.ones(len(z)) for p in powers
    ]), powers


def select_subspace(positions, forces, max_dimension=3, coupling_tolerance=.12,
                    nonlinear_fraction=.02, target=None):
    """Discover non-affine force directions, then close their harmonic coupling.

    The optional target is a prescribed observable direction, not a quantum
    reference. Full quantum data are never used by this function. A returned
    dimension equal to the full dimension provides no dimensional reduction.
    Inspect fit residuals, bath stability and budget_exhausted before use.
    """
    z = np.asarray(positions, float)
    f = np.asarray(forces, float)
    if z.ndim != 2 or z.shape[1] == 0 or f.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(f).all():
        raise ValueError('Finite matching [samples, coordinates] arrays required')
    if isinstance(max_dimension, bool) or int(max_dimension) != max_dimension or not 1 <= max_dimension <= 3:
        raise ValueError('Active dimension budget must be an integer from 1 to 3')
    max_dimension = min(int(max_dimension), z.shape[1])
    if not np.isfinite(coupling_tolerance) or coupling_tolerance < 0 or not 0 < nonlinear_fraction <= 1:
        raise ValueError('Invalid selection thresholds')
    terms = comb(z.shape[1] + 3, 3)
    if terms > 5000 or len(z) <= terms:
        raise ValueError('Cubic force model requires more samples than features and at most 5000 features')
    if target is not None:
        target = np.array(target, dtype=float, copy=True)
        if target.shape != (z.shape[1],) or not np.isfinite(target).all() or np.linalg.norm(target) == 0:
            raise ValueError('Finite nonzero target direction required')
    affine = np.column_stack([np.ones(len(z)), z])
    residual = f - affine @ np.linalg.lstsq(affine, f, rcond=None)[0]
    values, U = eigh(residual.T @ residual / len(z))
    values, U = values[::-1], U[:, ::-1]
    if values[0] <= 1e-20 * max(np.mean(f * f), 1):
        seed = np.array(target if target is not None else np.eye(z.shape[1])[0], dtype=float, copy=True)
        seed /= np.linalg.norm(seed)
        U, rank = seed[:, None], 0
    else:
        rank = max(1, int(np.count_nonzero(values >= nonlinear_fraction * values.sum())))
        U = U[:, :min(rank, max_dimension)]
    if target is not None:
        t = target / np.linalg.norm(target)
        candidates, orthogonal = np.column_stack([t, U]), []
        for v in candidates.T:
            v = v.copy()
            for old in orthogonal:
                v -= old * (old @ v)
            if np.linalg.norm(v) > 1e-10:
                orthogonal.append(v / np.linalg.norm(v))
            if len(orthogonal) >= max_dimension:
                break
        U = np.column_stack(orthogonal)
    design, _ = polynomial_features(z)
    coefficients = np.linalg.lstsq(design, f, rcond=1e-12)[0]
    raw_K = -coefficients[1:1 + z.shape[1]].T
    K = .5 * (raw_K + raw_K.T)
    fit = float(np.linalg.norm(f - design @ coefficients) / max(np.linalg.norm(f), 1e-30))
    history, bases = [], {}
    while True:
        W = null_space(U.T)
        d = U.shape[1]
        bases[d] = U.copy()
        A = U.T @ K @ U
        if W.shape[1]:
            B, C = W.T @ K @ W, W.T @ K @ U
            eigenvalues, _ = eigh(B)
            left, singular, _ = np.linalg.svd(C, full_matrices=False)
            score = float(singular[0] / np.sqrt(max(np.linalg.norm(A, 2), 1e-12)
                                                * max(np.linalg.norm(B, 2), 1e-12)))
            stable = bool(eigenvalues.min() > 0)
        else:
            score, stable = 0., True
        history.append({'dimension': d, 'residual_coupling_score': score,
                        'stable_harmonic_complement': stable})
        if score <= coupling_tolerance or d >= max_dimension or rank == 0:
            break
        v = W @ left[:, 0]
        U = np.column_stack([U, v / np.linalg.norm(v)])
    bath_residual = (float(np.linalg.norm((f + z @ K.T - coefficients[0]) @ W)
                           / max(np.linalg.norm(f @ W), 1e-30)) if W.shape[1] else 0.)
    return {'basis': U, 'nested_bases': bases, 'stiffness': K,
            'force_fit_relative_residual': fit, 'nonlinear_force_eigenvalues': values,
            'nonlinear_rank': rank, 'history': history, 'selected_dimension': U.shape[1],
            'selection_tolerance': coupling_tolerance,
            'budget_exhausted': bool((rank > 0 and score > coupling_tolerance) or rank > max_dimension),
            'quantum_correction_candidate': bool(rank > 0),
            'residual_bath_nonlinearity_fraction': bath_residual,
            'interpretation': 'heuristic coupling closure, not an error bound'}


def spectral_matrix(spec, frequencies, damping):
    """Thermal retarded response from positive-gap transition coefficients.

    spec['matrix_coefficients'] has shape [observables, observables, transitions].
    For real coordinate operators its kth slice is
    2*(p_n-p_m)*A_nm*A_mn, with gap_k=E_m-E_n>0. The caller is responsible for
    a converged Hamiltonian, temperature, and physically valid spectral weights.
    """
    s = damping - 1j * np.asarray(frequencies)
    coefficients = np.asarray(spec['matrix_coefficients'], complex)
    gap = np.asarray(spec['gap'], float)
    if gap.ndim != 1 or len(gap) == 0 or not np.isfinite(gap).all() or np.any(gap <= 0):
        raise ValueError('Positive finite transition gaps required')
    if coefficients.ndim != 3 or coefficients.shape[0] != coefficients.shape[1] or coefficients.shape[2] != len(gap) or not np.isfinite(coefficients).all():
        raise ValueError('Finite [observables, observables, transitions] coefficients required')
    if not np.allclose(coefficients, coefficients.conj().swapaxes(0, 1)):
        raise ValueError('Hermitian transition weights required')
    return np.einsum('abk,kw->wab', coefficients,
                     gap[:, None] / (gap[:, None] ** 2 + s[None, :] ** 2))


def memory_embed(spec, frequencies, damping, cross, bath,
                 static_covariance=None, beta=None):
    """Return (I+chi_Q*(memory+static_shift))^-1 chi_Q and diagnostics.

    Active references must use the statically relaxed stiffness
    K_AA-K_AB*K_BB^-1*K_BA. Restoring the residual harmonic environment adds
    K_AB*(K_BB^-1-(K_BB+s^2 I)^-1)*K_BA, s=eta-i*omega. This is exactly the
    harmonic Schur complement, but its transfer to a nonlinear quantum active
    system is an approximation. It does not restore full system-bath entanglement.
    Optional static matching uses a matching PIMD Kubo centroid covariance;
    it is a separate estimator and must be audited for sampling and bead error.
    No spectra are clipped, renormalized or fitted to full quantum answers.
    """
    w = np.asarray(frequencies, float)
    if w.ndim != 1 or len(w) == 0 or not np.isfinite(w).all() or np.any(w < 0) or np.any(np.diff(w) <= 0):
        raise ValueError('Sorted finite nonnegative frequency grid required')
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError('Positive finite damping required')
    Q = spectral_matrix(spec, w, damping)
    d = Q.shape[1]
    cross, bath = np.asarray(cross, float), np.asarray(bath, float)
    if bath.ndim != 2 or bath.shape[0] != bath.shape[1] or cross.shape != (d, len(bath)):
        raise ValueError('Incompatible active-bath matrix shapes')
    if not np.isfinite(bath).all() or not np.isfinite(cross).all():
        raise ValueError('Finite bath matrices required')
    if len(bath) and (not np.allclose(bath, bath.T) or np.linalg.eigvalsh(bath).min() <= 0):
        raise ValueError('Stable symmetric harmonic residual bath required')
    s = damping - 1j * w
    kernel, shift = np.zeros_like(Q), np.zeros((d, d))
    if len(bath):
        for i, value in enumerate(s):
            kernel[i] = cross @ (np.linalg.inv(bath)
                - np.linalg.inv(bath + value * value * np.eye(len(bath)))) @ cross.T
    if static_covariance is not None and len(bath):
        if beta is None or beta <= 0:
            raise ValueError('Positive beta required for static matching')
        static_Q = np.einsum('abk,k->ab', spec['matrix_coefficients'], 1 / spec['gap'])
        covariance = np.asarray(static_covariance, float)
        if covariance.shape != (d, d) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError('Positive definite matching Kubo covariance required')
        shift = np.linalg.inv(beta * covariance) - np.linalg.inv(static_Q)
        kernel += shift
    denominator = np.eye(d)[None] + Q @ kernel
    result = np.linalg.solve(denominator, Q)
    dissipative = (result - result.conj().swapaxes(-1, -2)) / (2j)
    eigenvalues = np.linalg.eigvalsh(dissipative)
    negative = float(np.sum(np.maximum(-eigenvalues, 0)) / max(np.sum(np.abs(eigenvalues)), 1e-30))
    return result, {'minimum_dissipative_eigenvalue': float(eigenvalues.min()),
        'negative_spectral_fraction': negative,
        'denominator_condition_max': float(np.linalg.cond(denominator).max()),
        'static_shift': shift.tolist(), 'physical_converged': False,
        'closure': 'harmonic-memory, nonlinear quantum active response',
        'passivity_grid_passed': bool(eigenvalues.min() > -1e-10)}


__all__ = ['select_subspace', 'memory_embed', 'spectral_matrix']
