"""Reversible two-level multiple-time-step propagation of a split ring Hamiltonian.

The outer step is Bslow(dt/2) [Bfast(h/2) A(h) Bfast(h/2)]**M Bslow(dt/2),
with h=dt/M.  A is the existing real-mass ring propagator.  This module offers
an explicit deterministic Python API; it does not silently change a workflow's
integrator, thermostat, checkpoint format, or reported physical Hamiltonian.
"""

from __future__ import annotations

import math
import numbers
import numpy as np

from .dynamics import CayleyIntegrator, State
from .potentials.base import validate_result
from .potentials.contraction import SplitPotential, _positive_integer


class MTSDynamics:
    """Two-level r-RESPA using a :class:`SplitPotential`.

    ``dt`` and State.step count OUTER steps.  Ring springs and fast forces use
    ``inner_steps`` substeps.  ``propagate`` rolls the entire block back on
    failure, while ``propagate_block`` retains completed outer steps.  Endpoint
    forces are reused within one block only, so external coordinate changes
    cannot leave a persistent stale force cache.  No thermostat is applied.
    """

    def __init__(
        self, potential, masses, beta, nbeads, dt, *, inner_steps=1,
        native=True, threads=1, integrator="normal-mode-bab", seed=0,
    ):
        if not isinstance(potential, SplitPotential):
            raise TypeError("MTSDynamics requires an explicit SplitPotential")
        self.inner_steps = _positive_integer(inner_steps, "inner_steps")
        self.nbeads = _positive_integer(nbeads, "nbeads")
        if self.nbeads != potential.nbeads:
            raise ValueError("MTSDynamics and SplitPotential bead counts must agree")
        self.dt = float(dt)
        self.integrator = CayleyIntegrator(
            masses, beta, self.nbeads, self.dt / self.inner_steps,
            native=native, threads=threads, scheme=integrator,
        )
        self.potential = potential
        self.masses = self.integrator.masses
        self.beta = self.integrator.beta
        self.integrator_name = self.integrator.scheme
        self.native = self.integrator.native
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._parameters = (self.dt, self.inner_steps, self.nbeads, self.beta, self.integrator_name)
        self.force_evaluations = {"fast": 0, "slow": 0}

    def _check(self, state):
        if not isinstance(state, State):
            raise TypeError("state must be a pyrpmd.dynamics.State")
        current = (self.dt, self.inner_steps, self.nbeads, self.beta, self.integrator_name)
        if current != self._parameters or self.integrator.dt != self.dt / self.inner_steps:
            raise ValueError("MTS parameters are immutable; create a new MTSDynamics")
        if self.integrator.beta != self.beta or self.integrator.nbeads != self.nbeads or self.integrator.scheme != self.integrator_name:
            raise ValueError("MTS integrator parameters are immutable")
        if (
            isinstance(state.step, (bool, np.bool_))
            or not isinstance(state.step, (int, np.integer)) or state.step < 0
            or isinstance(state.time, (bool, np.bool_))
            or not isinstance(state.time, numbers.Real)
            or not math.isfinite(state.time)
        ):
            raise ValueError("invalid state clock")
        self.integrator._check(state.q, state.p)

    def sample_momenta(self):
        """Independent ring momenta for exp[-(beta/P) H_P], with real masses."""
        return self.rng.normal(size=self.integrator.shape) * np.sqrt(
            self.masses[None, :, None] * self.nbeads / self.beta
        )

    def _force(self, kind, q):
        self.force_evaluations[kind] += 1
        evaluator = self.potential.evaluate_fast if kind == "fast" else self.potential.evaluate_slow
        # Both endpoint caches must survive another call into either component.
        return validate_result(evaluator(q), q.shape).forces.copy()

    def step(self, state):
        """Advance one outer step atomically, returning the same State object."""
        return self.propagate(state, 1)

    def propagate(self, state, steps):
        """Advance atomically; return State (not a trajectory/observation record)."""
        return self._advance(state, steps, keep_prefix=False)

    def propagate_block(self, state, steps):
        """Retain completed outer steps if a later outer step fails."""
        return self._advance(state, steps, keep_prefix=True)

    def _advance(self, state, steps, keep_prefix):
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        self._check(state)
        if steps == 0:
            return state
        q, p = state.q.copy(), state.p.copy()
        fast = self._force("fast", q)
        slow = self._force("slow", q)
        h = self.integrator.dt
        initial_step, initial_time = state.step, state.time
        for outer in range(int(steps)):
            next_time = initial_time + (outer + 1) * self.dt
            if not math.isfinite(next_time):
                raise FloatingPointError("nonfinite MTS state clock")
            self.integrator.kick(p, slow, 0.5 * self.dt)
            for _ in range(self.inner_steps):
                self.integrator.kick(p, fast, 0.5 * h)
                self.integrator.free_step(q, p)
                fast = self._force("fast", q)
                self.integrator.kick(p, fast, 0.5 * h)
            slow = self._force("slow", q)
            self.integrator.kick(p, slow, 0.5 * self.dt)
            if not np.isfinite(q).all() or not np.isfinite(p).all():
                raise FloatingPointError("nonfinite MTS integration state")
            if keep_prefix:
                state.q[:] = q
                state.p[:] = p
                state.step = initial_step + outer + 1
                state.time = next_time
        if not keep_prefix:
            state.q[:] = q
            state.p[:] = p
            state.step = initial_step + int(steps)
            state.time = next_time
        return state

    def energy(self, state):
        """Sum-beads Hamiltonian, including the explicit RPC approximation."""
        self._check(state)
        result = validate_result(self.potential.evaluate(state.q), state.q.shape)
        kinetic = np.sum(state.p * state.p / (2 * self.masses[None, :, None]))
        dq = state.q - np.roll(state.q, -1, axis=0)
        spring = 0.5 * (self.nbeads / self.beta) ** 2 * np.sum(
            self.masses[None, :, None] * dq * dq
        )
        return float(np.sum(result.energy) + kinetic + spring)


__all__ = ["MTSDynamics"]
