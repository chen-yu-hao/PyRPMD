"""Configurational equilibrium moves for BrHCl umbrella sampling.

These moves change configurations non-dynamically. Apply them immediately before
an independent Maxwell/Andersen momentum refresh; they do not define kinetics.
The Br-H and H-Cl centroid distances and every ring-spring norm are preserved.
"""
from __future__ import annotations

import numbers
import numpy as np
from pyrpmd.potentials import PotentialError
from pyrpmd.potentials.base import coordinates, validate_result


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _haar_rotations(rng, count):
    """Independent Haar SO(3) matrices, using three uniform variates each."""
    u = np.asarray(rng.random((count, 3)), dtype=float)
    a, b = 2*np.pi*u[:, 1], 2*np.pi*u[:, 2]
    x, y = np.sqrt(1-u[:, 0])*np.sin(a), np.sqrt(1-u[:, 0])*np.cos(a)
    z, w = np.sqrt(u[:, 0])*np.sin(b), np.sqrt(u[:, 0])*np.cos(b)
    return np.stack([
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ], axis=-1).reshape(count, 3, 3)


def _invalid_core(q, minimum):
    """q is [proposal,bead,atom,3]; only the declared finite-distance core."""
    invalid = np.zeros(len(q), dtype=bool)
    if minimum > 0:
        for a in range(q.shape[2]-1):
            for b in range(a+1, q.shape[2]):
                invalid |= np.any(np.linalg.norm(q[:, :, a]-q[:, :, b], axis=-1) < minimum, axis=1)
    return invalid


def _actions(potential, q, beta, batch_size):
    """Evaluate valid proposals in batches; any undeclared PES failure aborts."""
    count, beads, atoms, _ = q.shape
    flat = q.reshape(count*beads, atoms, 3)
    energies = np.empty(len(flat), dtype=float)
    for start in range(0, len(flat), batch_size):
        stop = min(len(flat), start+batch_size)
        block = flat[start:stop]
        result = validate_result(potential.evaluate(block), block.shape, require_success=True)
        energies[start:stop] = result.energy
    actions = (beta/beads)*np.sum(energies.reshape(count, beads), axis=1)
    if not np.isfinite(actions).all():
        raise PotentialError("Nonfinite orientational Metropolis action")
    return actions


def orientational_metropolis(potential, q, beta, rng, trials=32, atom=2, anchor=1,
                             *, batch_size=None, record_trace=False):
    """Return ``(new_q, diagnostics)`` after reversible Haar-orbit MH moves.

    ``q`` is [bead,atom,3] in bohr and is never modified. Each proposal rotates
    the entire ``atom`` ring about the ``anchor`` centroid. The candidates are
    independent Haar rotations of the original configuration, on the same
    fixed orbit as every accepted state. Acceptance is sequential relative to
    the *latest accepted state*, with log ratio ``beta/P * (sum V_old-sum V_new)``.
    Here V includes any common physical confinement supplied by ``potential``.

    For the default Br,H,Cl order, the two centroid-distance umbrella energies
    and all springs cancel exactly. No extra sin(theta) factor is needed: Haar
    measure already supplies the orientational measure. Other non-invariant
    biases, position constraints or thermostats require separate treatment.

    The declared ``metadata.minimum_pair_distance_bohr`` core is rejected before
    PES evaluation. Every other provider exception or failed frame is raised,
    including a source singularity; it is not silently counted as an MC reject.
    ``batch_size`` changes only PES batching, not proposals or random draws.
    The function draws orientation and acceptance uniforms, never momenta.
    """
    q = coordinates(q).copy()
    beads, atoms, _ = q.shape
    trials = _integer(trials, "trials")
    atom, anchor = _integer(atom, "atom"), _integer(anchor, "anchor")
    if atom >= atoms or anchor >= atoms or atom == anchor:
        raise ValueError("atom and anchor must be distinct valid atom indices")
    if isinstance(beta, (bool, np.bool_)):
        raise ValueError("beta must be positive and finite")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    minimum = float(getattr(potential, "metadata", {}).get("minimum_pair_distance_bohr", 0.0))
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("minimum_pair_distance_bohr must be finite and nonnegative")
    if _invalid_core(q[None], minimum)[0]:
        raise PotentialError("Initial configuration is inside the declared PES pair-distance core")
    batch_size = max(1, (trials+1)*beads) if batch_size is None else _integer(batch_size, "batch_size", 1)
    rotations = _haar_rotations(rng, trials)
    center = np.mean(q[:, anchor], axis=0)
    displacement = q[:, atom]-center
    proposals = np.broadcast_to(q, (trials, *q.shape)).copy()
    proposals[:, :, atom] = center + np.einsum("tij,pj->tpi", rotations, displacement)
    invalid = _invalid_core(proposals, minimum)
    valid = np.flatnonzero(~invalid)
    # Include the initial configuration so initial/proposal evaluations have
    # precisely the same energy and domain-check contract.
    to_evaluate = np.concatenate([q[None], proposals[valid]], axis=0)
    actions = _actions(potential, to_evaluate, beta, batch_size)
    current_action = float(actions[0])
    initial_action = current_action
    proposal_actions = np.full(trials, np.inf)
    proposal_actions[valid] = actions[1:]
    with np.errstate(divide="ignore"):
        log_uniform = np.log(np.asarray(rng.random(trials), dtype=float))
    accepted = np.zeros(trials, dtype=bool)
    log_ratios = np.full(trials, np.nan)
    current = q
    for index in range(trials):
        if invalid[index]:
            continue
        log_ratios[index] = current_action-proposal_actions[index]
        if log_uniform[index] < min(0.0, log_ratios[index]):
            current = proposals[index]
            current_action = float(proposal_actions[index])
            accepted[index] = True
    diagnostics = {
        "attempted": trials, "accepted": int(np.count_nonzero(accepted)),
        "invalid_domain": int(np.count_nonzero(invalid)),
        "initial_action": initial_action, "final_action": current_action,
    }
    if record_trace:
        diagnostics.update({
            "proposal_actions": [None if invalid[i] else float(proposal_actions[i]) for i in range(trials)],
            "log_acceptance_ratios": [None if invalid[i] else float(log_ratios[i]) for i in range(trials)],
            "log_uniforms": log_uniform.tolist(), "accepted_flags": accepted.tolist(),
        })
    return current.copy(), diagnostics
