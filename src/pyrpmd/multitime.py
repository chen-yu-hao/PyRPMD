"""Compatibility adapter for the common Dynamics force-splitting operator."""
from __future__ import annotations

import copy
import math
import numpy as np

from .dynamics import Dynamics, State, _state_clock
from .engine import ForceSplitting
from .potentials.contraction import SplitPotential


class MTSDynamics(Dynamics):
    """Two-level r-RESPA, implemented by :class:`Dynamics`' shared operators.

    ``dt`` and State.step count outer steps. Endpoint forces are reused only
    within this adapter's block, never across a caller's coordinate edits.
    Existing ``propagate``/``propagate_block`` methods still return State;
    new code can request ``Dynamics(..., force_splitting=ForceSplitting(M))``
    for the common observation/result API. Baths and fixed atoms compose with
    the same force split rather than selecting a separate integrator loop.
    """

    def __init__(self, potential, masses, beta, nbeads, dt, *, inner_steps=1,
                 native=True, threads=1, integrator="normal-mode-bab", seed=0,
                 thermostat=None, fixed_atoms=(), fixed_positions=None,
                 loop_backend="auto"):
        if not isinstance(potential, SplitPotential):
            raise TypeError("MTSDynamics requires an explicit SplitPotential")
        super().__init__(potential, masses, beta, nbeads, dt, seed=seed,
                         native=native, threads=threads, integrator=integrator,
                         loop_backend=loop_backend, thermostat=thermostat,
                         fixed_atoms=fixed_atoms, fixed_positions=fixed_positions,
                         force_splitting=ForceSplitting(inner_steps))

    def _check(self, state):
        if not isinstance(state, State):
            raise TypeError("state must be a pyrpmd.dynamics.State")
        self._check_parameters()
        _state_clock(state.step, state.time)
        self.integrator._check(state.q, state.p)
        self._check_fixed(state.q, state.p)

    def step(self, state, bias=None):
        return self._advance(state, 1, keep_prefix=False, bias=bias)

    def propagate(self, state, steps, *, bias=None):
        """Advance atomically, returning the same State as the legacy API."""
        return self._advance(state, steps, keep_prefix=False, bias=bias)

    def propagate_block(self, state, steps, *, bias=None):
        """Retain completed outer steps if a later outer step fails."""
        return self._advance(state, steps, keep_prefix=True, bias=bias)

    def _advance(self, state, steps, keep_prefix, bias=None):
        if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        self._check(state)
        if not steps:
            return state
        trial = state if keep_prefix else state.copy()
        operators = self.operator_state_dict()
        stats = copy.deepcopy(self.propagation_stats)
        cache = {}
        initial_step, initial_time = state.step, state.time
        try:
            for outer in range(int(steps)):
                next_time = initial_time + (outer + 1) * self.dt
                if not math.isfinite(next_time):
                    raise FloatingPointError("nonfinite MTS state clock")
                self._composition.step(trial, bias, cache=cache)
                trial.step, trial.time = initial_step + outer + 1, next_time
        except BaseException:
            if not keep_prefix:
                self.load_operator_state_dict(operators)
                self.propagation_stats = stats
            raise
        if not keep_prefix:
            state.q[:], state.p[:] = trial.q, trial.p
            state.step, state.time = trial.step, trial.time
        return state

    def energy(self, state, bias=None):
        self._check(state)
        return super().energy(state, bias)


__all__ = ["MTSDynamics"]
