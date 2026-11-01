"""Local first-order saddle refinement in Cartesian atomic units.

Partitioned rational-function optimization (P-RFO) follows Banerjee et al.,
J. Phys. Chem. 89, 52--57 (1985), https://doi.org/10.1021/j100247a015.
The uphill and downhill augmented Hessians are solved separately. Numerical
Hessians are refreshed at every iteration; no extra potential backend is needed.
"""

from __future__ import annotations

from numbers import Real

import numpy as np


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive number") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _geometry(value, name):
    array = np.asarray(value)
    if (
        array.dtype.kind not in "fiu"
        or array.ndim != 2
        or array.shape[1:] != (3,)
        or len(array) == 0
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real nonempty [N,3] array")
    return np.array(array, dtype=float, order="C", copy=True)


def _rfo_partition(curvatures, gradient, maximize):
    """Solve one augmented-Hessian problem in the local principal basis."""
    n = len(curvatures)
    if n == 0:
        return np.empty(0)
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = np.diag(curvatures)
    augmented[:n, n] = gradient
    augmented[n, :n] = gradient
    _, vectors = np.linalg.eigh(augmented)
    vector = vectors[:, -1 if maximize else 0]
    # A vanishing last component represents an unbounded RFO step (e.g. a
    # zero gradient along a mode of the wrong sign). Retain its direction;
    # the Cartesian trust limit below makes the displacement finite.
    tail = vector[-1]
    if abs(tail) < 1e-12:
        tail = np.copysign(1e-12, tail if tail != 0 else 1.0)
    return vector[:-1] / tail


def refine_transition_state(
    provider,
    geometry,
    *,
    max_steps=100,
    gradient_tolerance=1e-5,
    hessian_step=1e-4,
    max_step=0.2,
    initial_mode=None,
    negative_eigenvalue_tolerance=1e-8,
    masses=None,
):
    """Refine a TS guess with bounded, mode-following P-RFO steps.

    Coordinates, displacements and ``max_step`` are in bohr; force tolerance
    is in hartree/bohr. ``max_step`` bounds each atom's Cartesian displacement.
    ``initial_mode`` is an optional [N,3] reaction direction, such as an NEB
    tangent. Otherwise the lowest resolved Cartesian Hessian mode is followed.
    Subsequent modes are selected by overlap with the previous direction,
    favoring resolved unstable modes once they are available.

    Masses are used only for the final saddle-index test, matching preparation's
    mass-weighted eigenvalue tolerance. They default to one. Exact or numerical
    zero-curvature modes are excluded from mode selection, without assuming a
    translation/rotation-invariant potential. No constraints are imposed.

    The result contains a retained ``geometry`` and explicit ``converged``
    flag. Force convergence alone is insufficient: exactly one mass-weighted
    eigenvalue must be below ``-negative_eigenvalue_tolerance``. Stationary
    minima or higher-index saddles return a failure diagnostic. This is a local
    refinement method; convergence to a particular reaction is not guaranteed.
    """
    # Keep the numerical PES contract in one place and avoid an import cycle
    # when prepare_rate dispatches to this optimizer.
    from .prepare import _potential_values, numerical_hessian

    q = _geometry(geometry, "TS geometry")
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 1
    ):
        raise ValueError("max_steps must be a positive integer")
    gradient_tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    hessian_step = _positive(hessian_step, "hessian_step")
    max_step = _positive(max_step, "max_step")
    negative_eigenvalue_tolerance = _positive(
        negative_eigenvalue_tolerance, "negative_eigenvalue_tolerance"
    )
    if masses is None:
        masses = np.ones(len(q))
    mass_array = np.asarray(masses)
    if (
        mass_array.dtype.kind not in "fiu"
        or mass_array.shape != (len(q),)
        or not np.isfinite(mass_array).all()
        or np.any(mass_array <= 0)
    ):
        raise ValueError("masses must contain one finite positive mass per atom")
    repeated = np.repeat(np.asarray(mass_array, dtype=float), 3)
    root_mass = np.sqrt(repeated)
    previous_mode = None
    if initial_mode is not None:
        mode = _geometry(initial_mode, "initial_mode")
        if mode.shape != q.shape:
            raise ValueError("initial_mode must have the same shape as TS geometry")
        norm = np.linalg.norm(mode)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("initial_mode must have nonzero finite norm")
        previous_mode = mode.ravel() / norm

    trust = max_step
    history = []
    converged = False
    reason = "P-RFO reached max_steps without convergence"
    for iteration in range(int(max_steps) + 1):
        energy, forces = _potential_values(provider, q)
        gradient = -forces.ravel()
        maximum = float(np.max(np.abs(forces)))
        hessian = numerical_hessian(provider, q, hessian_step)
        values, vectors = np.linalg.eigh(hessian)
        weighted_values = np.linalg.eigvalsh(
            hessian / root_mass[:, None] / root_mass[None, :]
        )
        negative = int(
            np.count_nonzero(weighted_values < -negative_eigenvalue_tolerance)
        )
        history.append(
            {
                "iteration": iteration,
                "energy": energy,
                "max_force": maximum,
                "negative_modes": negative,
            }
        )
        if maximum <= gradient_tolerance:
            converged = negative == 1
            reason = (
                "P-RFO converged to a first-order saddle"
                if converged
                else "P-RFO reached a stationary point with "
                f"{negative} resolved unstable modes; exactly one is required"
            )
            break
        if iteration == max_steps:
            break

        # Avoid selecting translational, rotational, or other numerical null
        # modes as the uphill direction. This cutoff is numerical and separate
        # from the physical mass-weighted saddle-index tolerance above.
        cutoff = max(1e-12, float(np.max(np.abs(values))) * 1e-10)
        resolved = np.flatnonzero(np.abs(values) > cutoff)
        if len(resolved) == 0:
            reason = "P-RFO found no resolved Hessian mode to follow"
            break
        if previous_mode is None:
            uphill = int(resolved[0])
        else:
            overlap = vectors[:, resolved].T @ previous_mode
            uphill = int(resolved[np.argmax(np.abs(overlap))])
            unstable = np.flatnonzero(values < -cutoff)
            if values[uphill] >= -cutoff and len(unstable):
                # Across an avoided mode crossing, overlap alone can retain
                # a stable mode and climb away from an existing saddle. Once
                # an unstable subspace exists, follow its best-matching mode.
                overlap = vectors[:, unstable].T @ previous_mode
                uphill = int(unstable[np.argmax(np.abs(overlap))])
        mode = vectors[:, uphill].copy()
        if previous_mode is not None and np.dot(mode, previous_mode) < 0:
            mode *= -1
        previous_mode = mode

        principal_gradient = vectors.T @ gradient
        downhill = np.arange(len(values)) != uphill
        components = np.zeros_like(values)
        components[uphill : uphill + 1] = _rfo_partition(
            values[uphill : uphill + 1],
            principal_gradient[uphill : uphill + 1],
            True,
        )
        components[downhill] = _rfo_partition(
            values[downhill], principal_gradient[downhill], False
        )
        step = (vectors @ components).reshape(q.shape)
        largest = float(np.max(np.linalg.norm(step, axis=1)))
        if not np.isfinite(largest):
            raise ValueError("P-RFO produced a nonfinite displacement")
        if largest > trust:
            step *= trust / largest
        if not np.any(q + step != q):
            reason = "P-RFO stalled before force convergence"
            break

        # A TS search need not lower the energy or force norm on every step.
        # Instead, assess the local Hessian's prediction of the trial gradient.
        # Reduce the trust radius when that model is inaccurate.
        accepted = False
        for _ in range(12):
            trial = q + step
            _, trial_forces = _potential_values(provider, trial)
            change = hessian @ step.ravel()
            predicted = gradient + change
            error = float(np.linalg.norm(-trial_forces.ravel() - predicted))
            scale = max(
                float(np.linalg.norm(gradient)),
                float(np.linalg.norm(change)),
                gradient_tolerance,
            )
            if error <= 0.5 * scale:
                q = trial
                accepted = True
                if error < 0.1 * scale:
                    trust = min(max_step, trust * 1.25)
                break
            step *= 0.5
            trust *= 0.5
        if not accepted:
            reason = "P-RFO could not find a reliable step; try a closer TS guess"
            break

    return {
        "geometry": q.copy(),
        "converged": converged,
        "iterations": iteration,
        "energy": energy,
        "max_force": maximum,
        "hessian_eigenvalues": weighted_values.tolist(),
        "mode": (
            None if previous_mode is None else previous_mode.reshape(q.shape).copy()
        ),
        "history": history,
        "diagnostics": [
            reason,
            f"P-RFO max force = {maximum:.6g} hartree/bohr",
            f"P-RFO negative mass-weighted Hessian eigenvalues = {negative}",
        ],
    }
