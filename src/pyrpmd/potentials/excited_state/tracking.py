"""State tracking and overlap-derived nonadiabatic couplings.

The overlap convention throughout this module is ``O[i, j] = <old_i|new_j>``.
All routines operate in atomic units.  They never invent a gauge when the
retained electronic subspace is singular or poorly connected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import logm
from scipy.optimize import linear_sum_assignment


class StateTrackingError(RuntimeError):
    """Raised when adjacent electronic subspaces cannot be tracked safely."""


@dataclass(frozen=True)
class TrackingDiagnostics:
    """Diagnostics for one old-to-new state assignment."""

    permutation: tuple[int, ...]
    minimum_assigned_overlap: float
    maximum_subspace_defect: float
    minimum_energy_gap: float
    near_degenerate_pairs: tuple[tuple[int, int], ...]
    ambiguous_pairs: tuple[tuple[int, int], ...]

    @property
    def near_degenerate(self) -> bool:
        return bool(self.near_degenerate_pairs)

    @property
    def ambiguous(self) -> bool:
        return bool(self.ambiguous_pairs)


def _square_finite(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or array.shape[0] < 1
        or array.dtype.kind not in "fciu"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{label} must be a finite square numerical matrix")
    return np.asarray(array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64)


def polar_unitary(overlap: object, *, singular_tolerance: float = 0.25) -> tuple[np.ndarray, float]:
    """Return the closest unitary overlap and its retained-subspace defect.

    ``singular_tolerance`` bounds ``max(abs(singular_values - 1))``.  A large
    defect means that the selected states do not span the same subspace at the
    two geometries; silently projecting such an overlap would hide missing
    states or an excessive nuclear step.
    """

    matrix = _square_finite(overlap, "overlap")
    if (
        isinstance(singular_tolerance, (bool, np.bool_))
        or not np.isfinite(singular_tolerance)
        or not 0.0 < float(singular_tolerance) < 1.0
    ):
        raise ValueError("singular_tolerance must be finite and between zero and one")
    left, singular, right = np.linalg.svd(matrix)
    defect = float(np.max(np.abs(singular - 1.0)))
    if defect > float(singular_tolerance):
        raise StateTrackingError(
            "electronic overlap loses retained-subspace norm "
            f"(maximum singular-value defect {defect:.3e}); reduce the step or retain more states"
        )
    return left @ right, defect


def _phase_factor(diagonal: np.ndarray, tolerance: float) -> np.ndarray:
    magnitude = np.abs(diagonal)
    if np.any(magnitude < tolerance):
        index = int(np.argmin(magnitude))
        raise StateTrackingError(
            f"assigned state {index} has overlap {magnitude[index]:.3e}, below "
            f"phase_overlap_tolerance={tolerance:.3e}"
        )
    phase = np.exp(-1j * np.angle(diagonal))
    # Preserve exact real arrays for real electronic-structure backends.
    if np.max(np.abs(phase.imag)) < 1e-13:
        phase = phase.real
    return phase


def _transform_couplings(couplings: np.ndarray | None, permutation: np.ndarray, phase: np.ndarray):
    if couplings is None:
        return None
    raw = np.asarray(couplings)
    nstates = len(permutation)
    if raw.ndim != 4 or raw.shape[:2] != (nstates, nstates):
        raise ValueError("derivative couplings must have shape [S,S,N,3]")
    transformed = raw[permutation][:, permutation]
    transformed = transformed * np.conjugate(phase)[:, None, None, None]
    transformed = transformed * phase[None, :, None, None]
    return transformed


def align_states(
    overlap: object,
    energies: object,
    gradients: object,
    *,
    derivative_couplings: object | None = None,
    oscillator_strengths: object | None = None,
    transition_dipoles: object | None = None,
    phase_overlap_tolerance: float = 1e-8,
    subspace_tolerance: float = 0.25,
    degeneracy_tolerance: float = 1e-6,
    ambiguity_tolerance: float = 0.05,
):
    """Assign, reorder and phase-align current states to the previous frame.

    The rows of ``overlap`` are already in the *reported* old-state gauge; its
    columns are raw current states.  The maximum-absolute-overlap assignment is
    deterministic.  Each assigned current ket is then multiplied by a phase so
    that the corresponding diagonal overlap is positive real.

    Returns a mapping containing transformed arrays, the raw-state permutation,
    phases, and :class:`TrackingDiagnostics`.
    """

    matrix = _square_finite(overlap, "overlap")
    nstates = matrix.shape[0]
    energy = np.asarray(energies)
    gradient = np.asarray(gradients)
    if energy.shape != (nstates,) or energy.dtype.kind not in "fiu" or not np.isfinite(energy).all():
        raise ValueError("energies must be finite real [S]")
    if (
        gradient.ndim != 3
        or gradient.shape[0] != nstates
        or gradient.shape[-1] != 3
        or gradient.dtype.kind not in "fiu"
        or not np.isfinite(gradient).all()
    ):
        raise ValueError("gradients must be finite real [S,N,3]")
    for value, name in (
        (phase_overlap_tolerance, "phase_overlap_tolerance"),
        (degeneracy_tolerance, "degeneracy_tolerance"),
        (ambiguity_tolerance, "ambiguity_tolerance"),
    ):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or float(value) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if phase_overlap_tolerance == 0:
        raise ValueError("phase_overlap_tolerance must be positive")

    unitary, defect = polar_unitary(matrix, singular_tolerance=subspace_tolerance)
    rows, columns = linear_sum_assignment(-np.abs(unitary))
    if not np.array_equal(rows, np.arange(nstates)):
        raise StateTrackingError("state assignment did not cover every previous state")
    permutation = np.asarray(columns, dtype=np.int64)
    assigned = unitary[np.arange(nstates), permutation]
    phases = _phase_factor(assigned, float(phase_overlap_tolerance))

    tracked_overlap = matrix[:, permutation] * phases[None, :]
    tracked_unitary = unitary[:, permutation] * phases[None, :]
    minimum_assigned = float(np.min(np.abs(assigned)))

    reordered_energy = np.asarray(energy[permutation], dtype=np.float64)
    reordered_gradient = np.asarray(gradient[permutation], dtype=np.float64)
    transformed_coupling = _transform_couplings(
        None if derivative_couplings is None else np.asarray(derivative_couplings),
        permutation,
        phases,
    )

    strengths = None
    if oscillator_strengths is not None:
        strengths = np.asarray(oscillator_strengths)
        if strengths.shape != (nstates,) or strengths.dtype.kind not in "fiu" or not np.isfinite(strengths).all():
            raise ValueError("oscillator strengths must be finite real [S]")
        strengths = np.asarray(strengths[permutation], dtype=np.float64)

    dipoles = None
    if transition_dipoles is not None:
        dipoles = np.asarray(transition_dipoles)
        if (
            dipoles.shape != (nstates, 3)
            or dipoles.dtype.kind not in "fciu"
            or not np.isfinite(dipoles).all()
        ):
            raise ValueError("transition dipoles must be finite [S,3]")
        if int(permutation[0]) != 0:
            raise StateTrackingError(
                "ground-to-state transition dipoles cannot be reassigned when the tracked "
                "reference state is not raw state 0"
            )
        relative_phase = np.conjugate(phases[0]) * phases
        dipoles = dipoles[permutation] * relative_phase[:, None]

    pairs: list[tuple[int, int]] = []
    minimum_gap = float("inf")
    for i in range(nstates):
        for j in range(i + 1, nstates):
            gap = abs(float(reordered_energy[j] - reordered_energy[i]))
            minimum_gap = min(minimum_gap, gap)
            if gap <= float(degeneracy_tolerance):
                pairs.append((i, j))
    if nstates == 1:
        minimum_gap = float("inf")

    ambiguous: list[tuple[int, int]] = []
    absolute = np.abs(unitary)
    for row in range(nstates):
        ordered = np.sort(absolute[row])[::-1]
        if len(ordered) > 1 and ordered[0] - ordered[1] <= float(ambiguity_tolerance):
            ambiguous.append((row, int(permutation[row])))

    diagnostics = TrackingDiagnostics(
        permutation=tuple(int(index) for index in permutation),
        minimum_assigned_overlap=minimum_assigned,
        maximum_subspace_defect=defect,
        minimum_energy_gap=minimum_gap,
        near_degenerate_pairs=tuple(pairs),
        ambiguous_pairs=tuple(ambiguous),
    )
    return {
        "energies": reordered_energy,
        "gradients": reordered_gradient,
        "derivative_couplings": transformed_coupling,
        "oscillator_strengths": strengths,
        "transition_dipoles": dipoles,
        "overlap": tracked_overlap,
        "unitary_overlap": tracked_unitary,
        "permutation": permutation,
        "phases": phases,
        "diagnostics": diagnostics,
    }


def align_reference_overlap(
    overlap: object,
    *,
    phase_overlap_tolerance: float = 1e-8,
    subspace_tolerance: float = 0.25,
) -> np.ndarray:
    """Assign and phase-align displaced states against one fixed reference frame.

    This helper is intended for centered finite differences of wavefunction
    overlaps.  It returns the physical overlap after a maximum-overlap
    permutation and makes every assigned diagonal element positive real.
    """

    matrix = _square_finite(overlap, "overlap")
    if (
        isinstance(phase_overlap_tolerance, (bool, np.bool_))
        or not np.isfinite(phase_overlap_tolerance)
        or float(phase_overlap_tolerance) <= 0.0
    ):
        raise ValueError("phase_overlap_tolerance must be finite and positive")
    unitary, _ = polar_unitary(matrix, singular_tolerance=subspace_tolerance)
    rows, columns = linear_sum_assignment(-np.abs(unitary))
    if not np.array_equal(rows, np.arange(len(rows))):
        raise StateTrackingError("displaced-state assignment did not cover every reference state")
    reordered = matrix[:, columns]
    assigned = unitary[np.arange(len(rows)), columns]
    phases = _phase_factor(assigned, float(phase_overlap_tolerance))
    result = reordered * phases[None, :]
    return np.asarray(np.real_if_close(result, tol=1000))



def integrated_connection(
    overlap: object,
    *,
    method: str = "npi",
    subspace_tolerance: float = 0.25,
    logarithm_tolerance: float = 1e-8,
) -> np.ndarray:
    """Return the anti-Hermitian connection integrated over one frame step.

    ``method='hst'`` is the Hammes--Schiffer--Tully finite-difference form,
    ``(U-U^H)/2``.  ``method='npi'`` uses the principal matrix logarithm of the
    polar-unitary overlap, i.e. the constant norm-preserving connection whose
    exponential is the old-to-new basis rotation.
    """

    if method not in ("hst", "npi"):
        raise ValueError("coupling method must be 'hst' or 'npi'")
    if (
        isinstance(logarithm_tolerance, (bool, np.bool_))
        or not np.isfinite(logarithm_tolerance)
        or float(logarithm_tolerance) <= 0
    ):
        raise ValueError("logarithm_tolerance must be finite and positive")
    unitary, _ = polar_unitary(overlap, singular_tolerance=subspace_tolerance)
    if method == "hst":
        connection = 0.5 * (unitary - unitary.conj().T)
    else:
        raw = logm(unitary)
        anti = 0.5 * (raw - raw.conj().T)
        scale = max(1.0, float(np.linalg.norm(raw)))
        defect = float(np.linalg.norm(raw - anti) / scale)
        if not np.isfinite(defect) or defect > float(logarithm_tolerance):
            raise StateTrackingError(
                "principal overlap logarithm is not anti-Hermitian within tolerance "
                f"({defect:.3e} > {float(logarithm_tolerance):.3e}); reduce the step"
            )
        connection = anti
    diagonal = np.diag_indices_from(connection)
    if np.iscomplexobj(connection):
        connection[diagonal] = 1j * np.imag(np.diag(connection))
        if np.max(np.abs(connection.imag)) < 1e-13:
            connection = connection.real
    else:
        connection[diagonal] = 0.0
    return np.asarray(connection)


def time_derivative_coupling(
    overlap: object,
    time_step: float,
    *,
    method: str = "npi",
    subspace_tolerance: float = 0.25,
) -> np.ndarray:
    """Return the average ``<old_i|d/dt new_j>`` over one positive time step."""

    if isinstance(time_step, (bool, np.bool_)) or not np.isfinite(time_step) or float(time_step) <= 0:
        raise ValueError("time_step must be finite and positive")
    return integrated_connection(
        overlap, method=method, subspace_tolerance=subspace_tolerance
    ) / float(time_step)


def path_derivative_couplings(
    overlap: object,
    displacement: object,
    *,
    method: str = "npi",
    subspace_tolerance: float = 0.25,
    displacement_tolerance: float = 1e-14,
) -> np.ndarray:
    """Lift an overlap connection to a minimum-norm nuclear NAC vector.

    For the supplied path displacement ``dR``, the returned tensor satisfies
    ``sum_A d_ij,A dR_A = log(O)_ij`` (NPI) or the corresponding HST connection.
    Components perpendicular to the sampled path are set to zero; they are not
    inferable from two frames.  This explicit minimum-norm convention is useful
    for path-following dynamics but is not a replacement for full analytic NAC
    vectors when momentum rescaling outside the path direction is required.
    """

    delta = np.asarray(displacement)
    if (
        delta.ndim != 2
        or delta.shape[-1] != 3
        or delta.dtype.kind not in "fiu"
        or not np.isfinite(delta).all()
    ):
        raise ValueError("displacement must be finite real [N,3]")
    if (
        isinstance(displacement_tolerance, (bool, np.bool_))
        or not np.isfinite(displacement_tolerance)
        or float(displacement_tolerance) <= 0
    ):
        raise ValueError("displacement_tolerance must be finite and positive")
    norm2 = float(np.vdot(delta, delta).real)
    connection = integrated_connection(
        overlap, method=method, subspace_tolerance=subspace_tolerance
    )
    if norm2 <= float(displacement_tolerance) ** 2:
        if np.linalg.norm(connection) <= 10.0 * np.finfo(float).eps:
            return np.zeros((*connection.shape, *delta.shape), dtype=connection.dtype)
        raise StateTrackingError(
            "nontrivial electronic rotation at zero nuclear displacement; the two contexts are inconsistent"
        )
    return connection[..., None, None] * delta[None, None, :, :] / norm2
