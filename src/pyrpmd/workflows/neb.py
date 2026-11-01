"""Climbing-image NEB for finite molecular paths in atomic units.

The endpoints have a fixed atom correspondence and remain fixed throughout the
search.  The band uses the improved tangent of Henkelman and Jonsson,
J. Chem. Phys. 113, 9978 (2000), doi:10.1063/1.1323224, and the climbing force
of Henkelman, Uberuaga and Jonsson, J. Chem. Phys. 113, 9901 (2000),
doi:10.1063/1.1329672.  FIRE supplies the fictitious optimization dynamics
(Bitzek et al., Phys. Rev. Lett. 97, 170201, doi:10.1103/PhysRevLett.97.170201).

Convergence of a band is not a full Hessian or reaction-channel validation.
Use its highest image as the starting geometry for transition-state refinement.
"""

from __future__ import annotations

import numbers

import numpy as np


def _positive(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _count(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _geometry(value, name):
    q = np.asarray(value)
    if (
        q.dtype.kind not in "fiu"
        or q.ndim != 2
        or q.shape[0] < 1
        or q.shape[1] != 3
        or not np.isfinite(q).all()
    ):
        raise ValueError(f"{name} must be a finite real [atom,3] array")
    return np.array(q, dtype=float, order="C", copy=True)


def _tangents(path, energies):
    """Energy-weighted path tangents, including the equal-energy limit."""
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments.reshape(len(segments), -1), axis=1)
    if np.any(lengths <= 1e-12) or not np.isfinite(lengths).all():
        raise ValueError("NEB path contains coincident adjacent images")
    tangents = np.empty_like(path[1:-1])
    for i in range(1, len(path) - 1):
        back, forward = segments[i - 1], segments[i]
        left, center, right = energies[i - 1 : i + 2]
        if left < center < right:
            tangent = forward
        elif left > center > right:
            tangent = back
        else:
            high = max(abs(right - center), abs(left - center))
            low = min(abs(right - center), abs(left - center))
            if high == 0.0:
                tangent = forward + back
            elif right > left:
                tangent = forward + (low / high) * back
            else:
                tangent = (low / high) * forward + back
        norm = np.linalg.norm(tangent)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(
                f"NEB tangent is undefined at image {i}; supply a non-retracing path"
            )
        tangents[i - 1] = tangent / norm
    return tangents, lengths


def _band_forces(path, energies, physical_forces, spring_constant, climbing):
    tangents, lengths = _tangents(path, energies)
    projected = np.empty_like(physical_forces)
    for j, tangent in enumerate(tangents):
        force = physical_forces[j]
        parallel = np.sum(force * tangent) * tangent
        if j + 1 == climbing:
            projected[j] = force - 2.0 * parallel
        else:
            projected[j] = (
                force - parallel
                + spring_constant * (lengths[j + 1] - lengths[j]) * tangent
            )
    return projected, tangents


def climbing_image_neb(
    provider,
    reactant,
    product,
    *,
    images=7,
    spring_constant=0.1,
    max_steps=1000,
    gradient_tolerance=1e-4,
    max_step=0.1,
    climb_after=20,
    initial_path=None,
):
    """Find a saddle-point seed between atom-mapped molecular endpoints.

    Coordinates and ``max_step`` are in bohr, forces in hartree/bohr and the
    spring constant in hartree/bohr**2. ``images`` counts both fixed endpoints.
    Endpoints should already describe the intended reactant/product basins in
    the same Cartesian frame; this function does not optimize or reorder them.
    By default, interior coordinates are linearly interpolated. ``initial_path``
    may instead specify all images, including matching endpoints.

    ``max_force`` is the largest absolute component of the active band forces.
    A successful result also meets the true-force tolerance at the climbing
    image and has strictly negative curvature along its path tangent. This is
    only a directional sign check; the full Hessian check and its configurable
    mass-weighted eigenvalue tolerance establish a first-order saddle. On an
    iteration limit or a stationary path without negative curvature, ``converged`` is
    false and the last evaluated path is returned for inspection.
    """
    # Import locally so prepare can call this routine without a module cycle.
    from .prepare import _potential_values

    nimages = _count(images, "images", 3)
    max_steps = _count(max_steps, "max_steps", 1)
    climb_after = _count(climb_after, "climb_after", 0)
    spring_constant = _positive(spring_constant, "spring_constant")
    gradient_tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    max_step = _positive(max_step, "max_step")
    start = _geometry(reactant, "reactant")
    end = _geometry(product, "product")
    if start.shape != end.shape:
        raise ValueError("NEB endpoints must have the same [atom,3] shape")
    if np.linalg.norm(end - start) <= 1e-12:
        raise ValueError("NEB endpoints must be distinct")
    if initial_path is None:
        fraction = np.linspace(0.0, 1.0, nimages)[:, None, None]
        path = start[None] + fraction * (end - start)[None]
    else:
        supplied = np.asarray(initial_path)
        if (
            supplied.dtype.kind not in "fiu"
            or supplied.shape != (nimages,) + start.shape
            or not np.isfinite(supplied).all()
        ):
            raise ValueError("initial_path must be a finite real [images,atom,3] array")
        path = np.array(supplied, dtype=float, order="C", copy=True)
        if not (
            np.allclose(path[0], start, rtol=0, atol=1e-10)
            and np.allclose(path[-1], end, rtol=0, atol=1e-10)
        ):
            raise ValueError("initial_path endpoints must match reactant and product")
    path[0], path[-1] = start, end
    # Fail on geometrically degenerate input before making expensive PES calls.
    _tangents(path, np.zeros(nimages))
    energies = np.empty(nimages)
    energies[0], _ = _potential_values(provider, start)
    energies[-1], _ = _potential_values(provider, end)
    velocity = np.zeros_like(path[1:-1])
    dt, alpha, positive_steps = 0.1, 0.1, 0
    converged = False
    curvature = None
    reason = "CI-NEB reached max_steps before convergence."
    for iteration in range(max_steps + 1):
        physical_forces = np.empty_like(path[1:-1])
        for j in range(1, nimages - 1):
            energies[j], physical_forces[j - 1] = _potential_values(provider, path[j])
        highest = 1 + int(np.argmax(energies[1:-1]))
        climbing = highest if iteration >= climb_after else None
        band_force, tangents = _band_forces(
            path, energies, physical_forces, spring_constant, climbing
        )
        maximum = float(np.max(np.abs(band_force)))
        physical_maximum = float(np.max(np.abs(physical_forces[highest - 1])))
        if climbing is not None and max(maximum, physical_maximum) <= gradient_tolerance:
            tangent = tangents[highest - 1]
            displacement = 1e-4
            _, plus = _potential_values(provider, path[highest] + displacement * tangent)
            _, minus = _potential_values(provider, path[highest] - displacement * tangent)
            curvature = float(-np.sum((plus - minus) * tangent) / (2 * displacement))
            if curvature < 0.0:
                converged = True
                reason = "CI-NEB converged with a stationary climbing image."
            else:
                reason = (
                    "Stationary NEB band has no negative curvature along "
                    "the climbing tangent; a saddle was not established."
                )
            break
        if iteration == max_steps:
            break
        if iteration == climb_after:
            # Warmup velocities describe a different force field.
            velocity.fill(0.0)
            dt, alpha, positive_steps = 0.1, 0.1, 0
        power = float(np.sum(velocity * band_force))
        if power > 0.0:
            force_norm = np.linalg.norm(band_force)
            velocity = (1.0 - alpha) * velocity + (
                alpha * np.linalg.norm(velocity) / force_norm
            ) * band_force
            positive_steps += 1
            if positive_steps > 5:
                dt = min(1.1 * dt, 1.0)
                alpha *= 0.99
        elif power < 0.0:
            velocity.fill(0.0)
            dt *= 0.5
            alpha, positive_steps = 0.1, 0
        velocity += dt * band_force
        delta = dt * velocity
        largest_step = float(np.max(np.linalg.norm(delta, axis=2)))
        if largest_step > max_step:
            scale = max_step / largest_step
            delta *= scale
            velocity *= scale
        path[1:-1] += delta
        if not np.isfinite(path).all():
            raise ValueError("CI-NEB produced nonfinite coordinates")

    return {
        "geometry": path[highest].copy(),
        "energy": float(energies[highest]),
        "converged": converged,
        "iterations": iteration,
        "images": path.copy(),
        "energies": energies.copy(),
        "climbing_image": highest if climbing is not None else None,
        "max_force": maximum,
        "max_climbing_force": physical_maximum,
        "tangent": tangents[highest - 1].copy(),
        "directional_curvature": curvature,
        "diagnostics": [reason, f"max NEB force = {maximum:.6g} hartree/bohr"],
    }
