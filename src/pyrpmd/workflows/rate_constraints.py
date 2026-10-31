"""Centroid dividing-surface geometry and deterministic RATTLE projections."""

from __future__ import annotations

import numpy as np

from ..cv import RPMDrateCoordinate


def rate_coordinate(config, system):
    if config.get("task") == "surface_rate":
        from .surface import surface_definition

        return surface_definition(config, system)["coordinate"]
    reaction = config["reaction"]
    return RPMDrateCoordinate(
        reaction["fragments"],
        system.masses,
        system.geometry,
        reaction["forming_bonds"],
        reaction["breaking_bonds"],
        reaction["rinf"],
        equivalent_paths=reaction.get("equivalent_paths"),
    )


def _normal(surface, q, masses):
    value, g = surface.centroid_value_gradient(np.mean(q, axis=0))
    metric = float(np.sum(g * g / np.asarray(masses)[:, None]))
    if not np.isfinite(metric) or metric <= 1e-28:
        raise ValueError("singular constrained dividing surface")
    return value, g, metric


def project_positions(q, surface, masses, tolerance=1e-10, max_iterations=100):
    """Project an initial centroid onto S=0; used only before equilibration."""
    q = np.asarray(q, float).copy()
    m = np.asarray(masses)[:, None]
    for _ in range(max_iterations):
        value, g, z = _normal(surface, q, masses)
        if abs(value) < tolerance:
            return q
        correction = -value * g / (z * m)
        maximum = np.max(np.linalg.norm(correction, axis=1))
        if maximum > 0.2:
            correction *= 0.2 / maximum
        q += correction[None]
    raise RuntimeError("initial dividing-surface projection did not converge")


def project_momenta(p, q, surface, masses):
    """Gaussian conditional momentum projection onto the centroid tangent."""
    _, g, z = _normal(surface, q, masses)
    velocity = float(np.sum(g * np.mean(p, axis=0) / np.asarray(masses)[:, None]))
    p -= velocity / z * g[None]
    return p


def constrained_step(dynamics, state, surface, tolerance=1e-10, max_iterations=50):
    """Symmetric centroid Cayley-RATTLE, transactional on PES/constraint failure.

    The old-normal multiplier is solved *inside* the position step. Because
    centroid motion is a free drift in the ring Cayley map, its corrections
    commute with all internal modes. A new-normal tangent projection follows
    the force at the constrained final position, as required by RATTLE.
    """
    if getattr(dynamics, "_stochastic_propagation", False):
        raise ValueError("constrained_step requires deterministic dynamics; a constrained electronic bath is not implemented")
    dynamics._check_parameters()
    dynamics.integrator._check(state.q, state.p)
    dynamics._check_fixed(state.q, state.p)
    q = state.q.copy()
    p = state.p.copy()
    m = dynamics.masses[:, None]
    dt = dynamics.dt
    initial, g0, _ = _normal(surface, q, dynamics.masses)
    if dynamics.fixed_atoms and np.any(g0[list(dynamics.fixed_atoms)] != 0):
        raise ValueError("constrained surface gradients must exclude fixed atoms")
    if abs(initial) > max(1e-8, 10 * tolerance):
        raise ValueError("parent state is not on the dividing surface")
    dynamics.integrator.kick(p, dynamics._force(q), 0.5 * dt)
    dynamics.integrator.free_step(q, p)
    dynamics._project_fixed(q, p)
    free = q.copy()
    direction = g0 / m
    lam = 0.0
    for _ in range(max_iterations):
        value, g, _ = _normal(surface, q, dynamics.masses)
        if dynamics.fixed_atoms and np.any(g[list(dynamics.fixed_atoms)] != 0):
            raise ValueError("constrained surface gradients must exclude fixed atoms")
        if abs(value) < tolerance:
            break
        derivative = float(np.sum(g * direction))
        if abs(derivative) < 1e-25:
            raise RuntimeError("RATTLE old/new normals became orthogonal")
        lam -= value / derivative
        q[:] = free + lam * direction[None]
    else:
        raise RuntimeError(
            "Cayley-RATTLE constraint failed to converge; reduce timestep"
        )
    p += lam / dt * g0[None]
    dynamics.integrator.kick(p, dynamics._force(q), 0.5 * dt)
    project_momenta(p, q, surface, dynamics.masses)
    dynamics._project_fixed(q, p)
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise FloatingPointError("nonfinite parent state")
    state.q[:] = q
    state.p[:] = p
    state.step += 1
    state.time += dt
    return state
