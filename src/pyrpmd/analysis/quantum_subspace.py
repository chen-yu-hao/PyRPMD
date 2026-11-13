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
from dataclasses import dataclass
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
        'static_shift': shift.tolist(),
        'closure': 'harmonic-memory, nonlinear quantum active response',
        'passivity_grid_passed': bool(eigenvalues.min() > -1e-10)}


__all__ = ['select_subspace', 'select_quantum_subspace', 'QuantumSubspaceSelection', 'memory_embed', 'spectral_matrix']

@dataclass(frozen=True)
class QuantumSubspaceSelection:
    """Auditable coordinate selection based on force nonlinearity and coupling.

    ``selected_indices`` is a coordinate set, while ``basis`` is the matching
    column subset of the identity. The score is a ranking diagnostic and is
    not an estimate or bound on the quantum-dynamical error.
    """
    selected_indices: tuple[int, ...]
    basis: np.ndarray
    scores: np.ndarray
    nonlinear_scores: np.ndarray
    coupling_scores: np.ndarray
    target_scores: np.ndarray
    diagnostics: dict

    def __post_init__(self):
        basis = np.array(self.basis, dtype=float, copy=True)
        arrays = {
            "scores": np.asarray(self.scores, dtype=float),
            "nonlinear_scores": np.asarray(self.nonlinear_scores, dtype=float),
            "coupling_scores": np.asarray(self.coupling_scores, dtype=float),
            "target_scores": np.asarray(self.target_scores, dtype=float),
        }
        if basis.ndim != 2 or basis.shape[1] != len(self.selected_indices):
            raise ValueError("basis must have one column per selected coordinate")
        for name, value in arrays.items():
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite one-dimensional array")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        basis.flags.writeable = False
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "selected_indices", tuple(int(i) for i in self.selected_indices))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _selection_features(z):
    """Return centered linear/quadratic features and their coordinate labels."""
    n = z.shape[1]
    columns, labels = [], []
    for i in range(n):
        columns.append(z[:, i])
        labels.append((i,))
    for i in range(n):
        columns.append(z[:, i] * z[:, i])
        labels.append((i, i))
    for i in range(n):
        for j in range(i + 1, n):
            columns.append(z[:, i] * z[:, j])
            labels.append((i, j))
    return np.column_stack(columns), labels


def _normalise_scores(values):
    values = np.asarray(values, dtype=float)
    scale = float(np.max(values)) if len(values) else 0.0
    return values / scale if scale > 0 else np.zeros_like(values)


def select_quantum_subspace(
    positions,
    forces,
    *,
    max_dimension=3,
    joint_coordinates=None,
    target_observables=None,
    masses=None,
    nonlinear_weight=1.0,
    coupling_weight=1.0,
    target_weight=1.0,
    coupling_threshold=0.15,
    min_score=0.0,
):
    """Select quantum coordinates from nonlinear and joint force diagnostics.

    The selector combines three independently reported signals: quadratic
    force-response strength, linear/nonlinear force coupling to residual
    coordinates, and correlations with target observables including quadratic
    features. ``joint_coordinates`` defines inseparable coordinate groups;
    selecting one coordinate selects the complete group. A greedy closure then
    adds the strongest coupled group while the dimension budget permits it.

    This is an automatic *ranking* procedure. Its score is not an error bound,
    and it does not establish that the resulting low-dimensional inverse
    response defect is transferable to the full system. A full-dimensional
    selection is the algebraic no-reduction limit.
    """
    z = np.asarray(positions, dtype=float)
    force = np.asarray(forces, dtype=float)
    if (z.ndim != 2 or force.shape != z.shape or len(z) < 8 or z.shape[1] < 1
            or not np.isfinite(z).all() or not np.isfinite(force).all()):
        raise ValueError("positions and forces must be finite matching [samples, coordinates] arrays")
    samples, ndim = z.shape
    if isinstance(max_dimension, (bool, np.bool_)) or int(max_dimension) != max_dimension:
        raise ValueError("max_dimension must be an integer")
    max_dimension = int(max_dimension)
    if max_dimension < 1 or max_dimension > ndim:
        raise ValueError("max_dimension must be between one and the coordinate count")
    weights = np.asarray([nonlinear_weight, coupling_weight, target_weight], dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("selection weights must be finite and nonnegative")
    if not np.isfinite(coupling_threshold) or coupling_threshold < 0:
        raise ValueError("coupling_threshold must be finite and nonnegative")
    if not np.isfinite(min_score) or min_score < 0:
        raise ValueError("min_score must be finite and nonnegative")

    if masses is None:
        mass = np.ones(ndim)
    else:
        mass = np.asarray(masses, dtype=float)
        if mass.shape != (ndim,) or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("masses must be positive and match the coordinate count")
    # Mass weighting makes a coordinate score comparable across heterogeneous
    # coordinates while preserving the input force convention.
    zc = z - z.mean(axis=0)
    fc = force - force.mean(axis=0)
    zw = zc * np.sqrt(mass)[None, :]
    fw = fc / np.sqrt(mass)[None, :]
    affine = np.column_stack([np.ones(samples), zw])
    affine_coeff = np.linalg.lstsq(affine, fw, rcond=None)[0]
    residual = fw - affine @ affine_coeff
    total_force = np.linalg.norm(fw, axis=0)
    nonlinear = np.linalg.norm(residual, axis=0) / np.maximum(total_force, np.finfo(float).tiny)

    features, labels = _selection_features(zw)
    if features.shape[1] > 5000 or samples <= features.shape[1] + 1:
        raise ValueError("quadratic force model requires more samples than features and at most 5000 features")
    # The linear and quadratic model is deliberately used only as a diagnostic;
    # its residual and coefficient norms are both retained for auditability.
    design = np.column_stack([np.ones(samples), features])
    coeff = np.linalg.lstsq(design, fw, rcond=None)[0]
    model_residual = fw - design @ coeff
    nonlinear_strength = np.zeros(ndim)
    for column, label in enumerate(labels):
        if len(label) == 2:
            contribution = np.linalg.norm(features[:, column, None] * coeff[column + 1], axis=0)
            nonlinear_strength += contribution
    nonlinear_strength /= np.maximum(total_force, np.finfo(float).tiny)
    nonlinear = _normalise_scores(.5 * nonlinear + .5 * nonlinear_strength)

    # Harmonic coupling plus nonlinear force cross-coupling. The diagonal is
    # excluded so a stiff uncoupled coordinate is not selected by itself.
    linear_K = -affine_coeff[1:].T
    linear_K = .5 * (linear_K + linear_K.T)
    denom = np.sqrt(np.maximum(np.diag(linear_K) ** 2, np.finfo(float).tiny))
    linear_coupling = np.abs(linear_K) / np.maximum(denom[:, None], np.finfo(float).tiny)
    np.fill_diagonal(linear_coupling, 0.0)
    nonlinear_coupling = np.zeros((ndim, ndim))
    for column, label in enumerate(labels):
        if len(label) == 2:
            i, j = label
            value = np.linalg.norm(coeff[column + 1]) / max(np.linalg.norm(fw, axis=0).mean(), np.finfo(float).tiny)
            nonlinear_coupling[i, j] += value
            nonlinear_coupling[j, i] += value
    coupling_matrix = np.maximum(linear_coupling, nonlinear_coupling)
    coupling = _normalise_scores(np.max(coupling_matrix, axis=1))

    target_score = np.zeros(ndim)
    if target_observables is not None:
        target = np.asarray(target_observables, dtype=float)
        if target.ndim == 1:
            target = target[:, None]
        if target.ndim != 2 or target.shape[0] != samples or target.shape[1] < 1 or not np.isfinite(target).all():
            raise ValueError("target_observables must be finite [samples] or [samples, observables]")
        target = target - target.mean(axis=0)
        target_norm = np.linalg.norm(target, axis=0)
        # Correlation with x_i, x_i^2 and x_i*x_j catches even symmetry-forced
        # resonance partners such as x*y^2 that have zero linear covariance.
        for column, label in enumerate(labels):
            feature = features[:, column]
            score = np.linalg.norm(target.T @ feature) / np.maximum(target_norm * np.linalg.norm(feature), np.finfo(float).tiny)
            for i in label:
                target_score[i] = max(target_score[i], float(np.max(score)))
        target_score = _normalise_scores(target_score)

    total = weights[0] * nonlinear + weights[1] * coupling + weights[2] * target_score
    scale = max(float(np.sum(weights)), np.finfo(float).tiny)
    total = total / scale

    if joint_coordinates is None:
        groups = [(i,) for i in range(ndim)]
    else:
        groups = []
        seen = set()
        for raw in joint_coordinates:
            group = tuple(sorted(set(int(i) for i in raw)))
            if not group or any(i < 0 or i >= ndim for i in group) or seen.intersection(group):
                raise ValueError("joint_coordinates must be nonempty disjoint in-range groups")
            seen.update(group)
            groups.append(group)
        groups.extend((i,) for i in range(ndim) if i not in seen)
    if any(len(group) > max_dimension for group in groups):
        raise ValueError("joint coordinate group exceeds max_dimension")
    group_scores = {group: float(np.mean(total[list(group)])) for group in groups}
    selected_groups = []
    selected = set()
    # Primary ranking is score; closure adds a coupled group only when it is
    # supported by the current active set and there is dimension budget.
    while True:
        candidates = [g for g in groups if g not in selected_groups and len(selected) + len(g) <= max_dimension]
        if not candidates:
            break
        if not selected:
            group = max(candidates, key=lambda g: (group_scores[g], tuple(-i for i in g)))
        else:
            def candidate_key(g):
                cross = max((coupling_matrix[i, j] for i in selected for j in g), default=0.0)
                return (group_scores[g] + float(coupling_weight) * cross, group_scores[g], tuple(-i for i in g))
            group = max(candidates, key=candidate_key)
        cross = max((coupling_matrix[i, j] for i in selected for j in group), default=0.0)
        if selected and group_scores[group] <= min_score + 1e-15 and cross < coupling_threshold:
            break
        selected_groups.append(group)
        selected.update(group)
        if len(selected) >= max_dimension:
            break
        if all(group_scores[g] < min_score for g in groups if g not in selected_groups):
            break
    if not selected:
        selected_groups = [min(groups, key=lambda g: (len(g), g))]
        selected.update(selected_groups[0])
    selected_indices = tuple(sorted(selected))
    basis = np.eye(ndim)[:, selected_indices]
    ranking = tuple(sorted(range(ndim), key=lambda i: (-total[i], i)))
    diagnostics = {
        "feature_model": "mass-weighted linear plus quadratic force response",
        "target_feature_model": "linear, squares and pair products",
        "selected_groups": [list(g) for g in selected_groups],
        "candidate_groups": [list(g) for g in groups],
        "ranking": list(ranking),
        "score_components": {"nonlinear": nonlinear.tolist(), "coupling": coupling.tolist(), "target": target_score.tolist(), "combined": total.tolist()},
        "linear_stiffness": linear_K.tolist(),
        "coupling_matrix": coupling_matrix.tolist(),
        "force_model_relative_residual": float(np.linalg.norm(model_residual) / max(np.linalg.norm(fw), np.finfo(float).tiny)),
        "max_dimension": max_dimension,
        "budget_exhausted": bool(len(selected) >= max_dimension),
        "closure_threshold": float(coupling_threshold),
        "selection_is_error_bound": False,
        "interpretation": "automatic nonlinear-force and joint-coordinate ranking; validate by nested quantum-defect convergence",
    }
    return QuantumSubspaceSelection(selected_indices, basis, total, nonlinear, coupling, target_score, diagnostics)
