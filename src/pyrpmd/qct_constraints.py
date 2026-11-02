"""Approximate QCT phase constraints with fixed harmonic mode actions.

The nonlinear map changes the joint phase distribution. It is not a derivation
of a conditional microcanonical measure or an exact rovibrational quantum state.
No additional random draws, retries from new phases, or energy-changing fallback
are used when the constraints cannot be satisfied.
"""

from __future__ import annotations

import numpy as np


_ANGULAR_RTOL = 1e-12
_ACTION_RTOL = 2e-10
_DAMPING = (0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0)


def constrain_qct_phases(phase, frequencies, quanta, modes, geometry, masses, *,
                         max_iterations=128):
    """Return one-dimensional ``(mode_q, mode_p)`` with fixed actions and j=0.

    Inputs use atomic units and orthonormal mass-weighted mode rows. ``modes``
    has shape ``[Nv, Natoms, 3]`` or ``[Nv, 3*Natoms]``. Each input phase is the
    starting point of a deterministic damped Gauss--Newton solve; output phases
    are generally correlated. Positions and momenta are parameterized by phase
    so every iterate has ``E_k = (quanta[k] + 1/2) * frequencies[k]``.

    Angular momentum is evaluated about the instantaneous fragment COM. The
    residual is divided by a phase-independent angular-momentum scale, and the
    final Cartesian reconstruction is checked separately. Ill-conditioned mode
    references, stagnation, or exhaustion of the finite iteration budget raise
    ``RuntimeError`` rather than return an unconstrained or rescaled state.
    ``max_iterations=0`` permits only an already admissible initial phase.
    """
    phase = np.array(phase, dtype=float, copy=True)
    frequencies = np.asarray(frequencies, dtype=float)
    quanta = np.asarray(quanta, dtype=float)
    modes = np.asarray(modes, dtype=float)
    geometry = np.asarray(geometry, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if (phase.ndim != 1 or frequencies.shape != phase.shape or quanta.shape != phase.shape
            or masses.ndim != 1 or len(masses) == 0 or geometry.shape != (len(masses), 3)):
        raise ValueError("QCT action constraints require matching phase, mode, geometry and mass arrays")
    nmodes = len(phase)
    if modes.shape == (nmodes, 3 * len(masses)):
        modes = modes.reshape(nmodes, len(masses), 3)
    if modes.shape != (nmodes, len(masses), 3):
        raise ValueError("QCT action constraints require mass-weighted mode rows [Nv,Natoms,3]")
    if (not all(np.isfinite(array).all() for array in
                (phase, frequencies, quanta, modes, geometry, masses))
            or np.any(frequencies <= 0) or np.any(masses <= 0)
            or np.any(quanta < 0) or np.any(quanta != np.floor(quanta))):
        raise ValueError("QCT action constraints require finite inputs, positive masses/frequencies and integer quanta")
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0):
        raise ValueError("QCT constraint max_iterations must be a nonnegative integer")
    if nmodes == 0:
        return phase.copy(), phase.copy()

    actions = quanta + 0.5
    amplitude_q = np.sqrt(2 * actions / frequencies)
    amplitude_p = np.sqrt(2 * actions * frequencies)
    root_mass = np.sqrt(masses)
    centered_geometry = geometry - np.average(geometry, axis=0, weights=masses)

    # Work about the actual COM even for explicit modes with small, accepted
    # reference-orthogonality errors. The following correction is used only to
    # form the angular-momentum coefficients, not to change the supplied basis.
    translation = np.einsum("i,kix->kx", root_mass, modes) / masses.sum()
    centered_modes = modes - root_mass[None, :, None] * translation[:, None, :]
    linear = np.sum(np.cross(centered_geometry[None] * root_mass[None, :, None],
                             centered_modes), axis=1)
    # Sum the mode cross products without a [Nv,Nv,Natoms,3] intermediate.
    coupling = np.empty((nmodes, nmodes, 3))
    for axis, first, second in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        coupling[:, :, axis] = (centered_modes[:, :, first] @ centered_modes[:, :, second].T
                               - centered_modes[:, :, second] @ centered_modes[:, :, first].T)
    angular_scale = (np.linalg.norm(amplitude_q) * np.linalg.norm(amplitude_p)
                     + np.linalg.norm(linear * amplitude_p[:, None]))
    if (not np.isfinite(amplitude_q).all() or not np.isfinite(amplitude_p).all()
            or not np.isfinite(angular_scale) or angular_scale <= 0
            or not np.isfinite(coupling).all()):
        raise ValueError("QCT action constraints overflow; check reference frequencies, masses and geometry")
    linear = linear / angular_scale
    coupling = coupling / angular_scale

    def residual(current):
        mode_q = amplitude_q * np.cos(current)
        mode_p = amplitude_p * np.sin(current)
        dq = -amplitude_q * np.sin(current)
        dp = amplitude_p * np.cos(current)
        angular = (mode_p @ linear
                   + np.einsum("a,b,abx->x", mode_q, mode_p, coupling))
        jacobian = (dq[:, None] * np.einsum("b,abx->ax", mode_p, coupling)
                    + dp[:, None] * (linear + np.einsum("a,abx->bx", mode_q, coupling))).T
        return angular, jacobian

    def checked_result(current):
        mode_q = amplitude_q * np.cos(current)
        mode_p = amplitude_p * np.sin(current)
        rows = modes.reshape(nmodes, -1)
        displacement = (mode_q @ rows).reshape(len(masses), 3) / root_mass[:, None]
        cartesian_q = centered_geometry + displacement
        cartesian_p = (mode_p @ rows).reshape(len(masses), 3) * root_mass[:, None]
        relative_q = cartesian_q - np.average(cartesian_q, axis=0, weights=masses)
        angular = np.sum(np.cross(relative_q, cartesian_p), axis=0)
        # Independent projection checks catch a supplied basis whose accepted
        # orthogonality error is too large for this stricter preparation policy.
        projected_q = (displacement * root_mass[:, None]).ravel() @ rows.T
        projected_p = (cartesian_p / root_mass[:, None]).ravel() @ rows.T
        actual = 0.5 * (projected_p ** 2 + (frequencies * projected_q) ** 2)
        target = actions * frequencies
        action_error = np.max(np.abs(actual / target - 1))
        angular_error = np.linalg.norm(angular) / angular_scale
        if (not np.isfinite(action_error) or not np.isfinite(angular_error)
                or action_error > _ACTION_RTOL or angular_error > 2 * _ANGULAR_RTOL):
            raise RuntimeError(
                "QCT preserve_actions failed its Cartesian constraint check: "
                f"relative mode-energy error={action_error:.3g}, "
                f"scaled angular momentum={angular_error:.3g}; check the harmonic reference"
            )
        return mode_q, mode_p

    norm = float("inf")
    for iteration in range(max_iterations + 1):
        angular, jacobian = residual(phase)
        norm = np.linalg.norm(angular)
        if not np.isfinite(norm) or not np.isfinite(jacobian).all():
            break
        if norm <= _ANGULAR_RTOL:
            return checked_result(phase)
        if iteration == max_iterations:
            break
        try:
            left, singular, right = np.linalg.svd(jacobian, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("QCT preserve_actions could not solve its phase constraints") from exc
        if singular[0] == 0:
            break
        best = None
        for damping in _DAMPING:
            if damping == 0:
                factors = np.divide(1.0, singular, out=np.zeros_like(singular),
                                    where=singular > 1e-12 * singular[0])
            else:
                factors = singular / (singular ** 2 + damping * singular[0] ** 2)
            step = -(right.T * factors) @ (left.T @ angular)
            step_norm = np.linalg.norm(step)
            if not np.isfinite(step_norm) or step_norm == 0:
                continue
            step *= min(1.0, 1.0 / step_norm)
            fraction = 1.0
            for _ in range(12):
                proposal = phase + fraction * step
                next_angular, _ = residual(proposal)
                candidate_norm = np.linalg.norm(next_angular)
                if candidate_norm < norm * (1 - 1e-4 * fraction):
                    if best is None or candidate_norm < best[0]:
                        best = (candidate_norm, proposal)
                    break
                fraction *= 0.5
        if best is None:
            break
        phase = best[1]
    raise RuntimeError(
        "QCT preserve_actions phase constraints did not converge "
        f"within {max_iterations} iterations (scaled angular momentum={norm:.3g}); "
        "no state was returned or resampled"
    )
