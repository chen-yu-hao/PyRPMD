"""Canonical temperature/Hamiltonian replica exchange for real-mass rings.

The configurational reduced potential includes the temperature-dependent
spring action. Using only a classical potential-energy difference would give
the wrong acceptance probability for P>1 temperature exchange.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .dynamics import Dynamics, State
from .bias import CompositeBias
from .potentials.cache import StepEvaluationCache
from .potentials.contraction import DifferencePotential, SplitPotential
from .potentials.lifecycle import CommitteePotential, DeltaLearningPotential
from .potentials.python import PythonPotential


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _static(obj, _ancestors=frozenset()):
    if id(obj) in _ancestors:
        raise ValueError("replica exchange requires an acyclic static composition")
    # Composite providers advertise their own execution contract. Inspect the
    # borrowed components too: a member may be replaced or marked dynamic after
    # construction, and four energy evaluations no longer define a static
    # Metropolis ratio in that case. Do this before any provider is called.
    if isinstance(obj, CompositeBias):
        children = obj.biases
    elif isinstance(obj, CommitteePotential):
        children = obj.members
    elif isinstance(obj, DeltaLearningPotential):
        children = (obj.reference, obj.correction)
    elif isinstance(obj, DifferencePotential):
        children = (obj.target, obj.reference)
    elif isinstance(obj, SplitPotential):
        children = (obj.fast, obj.slow)
    elif isinstance(obj, PythonPotential):
        children = (obj.obj,)
    elif isinstance(obj, StepEvaluationCache):
        children = (obj._provider,)
    else:
        children = ()
    # A cache forwards attribute access. Traverse known composition edges first
    # so a cache cycle cannot recurse through metadata before cycle detection.
    for child in children:
        _static(child, _ancestors | {id(obj)})
    metadata = getattr(obj, "metadata", {})
    capabilities = metadata.get("capabilities", {}) if isinstance(metadata, dict) else {}
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    markers = ("history_dependent", "time_dependent", "stateful")
    if any(getattr(obj, key, False)
           or isinstance(metadata, dict) and metadata.get(key, False)
           or capabilities.get(key, False) for key in markers):
        raise ValueError("replica exchange requires time-independent Hamiltonians and biases")


def reduced_configurational_energy(dynamics, q, bias=None):
    """Return beta/P times U_P(q), including each target's spring constant."""
    dynamics._check_parameters()
    _static(dynamics.potential)
    if bias is not None:
        _static(bias)
    state = State(q, np.zeros_like(q))
    result = dynamics.beta / dynamics.nbeads * dynamics.energy(state, bias)
    if not np.isfinite(result):
        raise ValueError("replica reduced energy must be finite")
    return float(result)


@dataclass(frozen=True)
class ReplicaExchangeMove:
    pair: tuple[int, int]
    accepted: bool
    log_acceptance_ratio: float


class ReplicaExchange:
    """Exchange configurations and temperature-scaled momenta between slots.

    Slot Hamiltonians, temperatures, RNGs and clocks remain at their slot.
    All replicas need the same bead count, masses and fixed-atom constraints.
    States are owned copies. Calls to ``advance`` thermostat each slot with
    PILE-L, then perform one random-parity nearest-neighbor exchange sweep.
    This serial reference engine does not claim asynchronous/MPI execution.
    """

    def __init__(self, dynamics, states, *, biases=None, seed=0):
        self.dynamics = tuple(dynamics)
        if len(self.dynamics) < 2 or any(type(d) is not Dynamics for d in self.dynamics):
            raise TypeError("replica exchange needs at least two built-in Dynamics instances")
        first = self.dynamics[0]
        for dyn in self.dynamics:
            dyn._check_parameters()
            if (dyn.nbeads != first.nbeads or not np.array_equal(dyn.masses, first.masses)
                    or dyn.fixed_atoms != first.fixed_atoms or dyn.dt != first.dt
                    or (first.fixed_atoms and not np.array_equal(dyn.fixed_positions, first.fixed_positions))):
                raise ValueError("replicas must share beads, masses, timestep and fixed coordinates")
            _static(dyn.potential)
        if len(states) != len(self.dynamics) or any(not isinstance(s, State) for s in states):
            raise ValueError("provide one State per replica")
        self.states = [state.copy() for state in states]
        if len({(s.step, s.time) for s in self.states}) != 1:
            raise ValueError("replica clocks must be synchronized")
        for dyn, state in zip(self.dynamics, self.states):
            dyn.integrator._check(state.q, state.p)
            dyn._check_fixed(state.q, state.p)
        self.biases = tuple([None] * len(states) if biases is None else biases)
        if len(self.biases) != len(states):
            raise ValueError("provide one bias or None per replica")
        for bias in self.biases:
            if bias is not None:
                _static(bias)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.labels = np.arange(len(states))
        self.attempts = np.zeros((len(states), len(states)), dtype=np.int64)
        self.accepts = self.attempts.copy()

    def _pair(self, i, j):
        i, j = _integer(i, "replica index"), _integer(j, "replica index")
        if i == j or max(i, j) >= len(self.states):
            raise ValueError("replica pair needs two distinct in-range slots")
        return i, j

    def log_acceptance_ratio(self, i, j):
        i, j = self._pair(i, j)
        di, dj = self.dynamics[i], self.dynamics[j]
        qi, qj = self.states[i].q, self.states[j].q
        energy = np.asarray([
            reduced_configurational_energy(di, qi, self.biases[i]),
            reduced_configurational_energy(dj, qj, self.biases[j]),
            reduced_configurational_energy(di, qj, self.biases[i]),
            reduced_configurational_energy(dj, qi, self.biases[j]),
        ], dtype=np.longdouble)
        result = float(energy[0] + energy[1] - energy[2] - energy[3])
        if np.isnan(result):
            raise ValueError("indeterminate replica exchange probability")
        return result

    def attempt(self, i, j):
        i, j = self._pair(i, j)
        log_ratio = self.log_acceptance_ratio(i, j)
        bi, bj = self.dynamics[i].beta, self.dynamics[j].beta
        scale = np.sqrt(bj / bi)
        qi, qj = self.states[j].q.copy(), self.states[i].q.copy()
        pi, pj = self.states[j].p * scale, self.states[i].p / scale
        if not np.isfinite(pi).all() or not np.isfinite(pj).all():
            raise ValueError("temperature momentum rescaling is not representable")
        uniform = self.rng.random()
        accepted = bool((-np.inf if uniform == 0 else np.log(uniform)) < min(0.0, log_ratio))
        self.attempts[i, j] += 1
        if accepted:
            self.states[i].q[:], self.states[j].q[:] = qi, qj
            self.states[i].p[:], self.states[j].p[:] = pi, pj
            self.labels[[i, j]] = self.labels[[j, i]]
            self.accepts[i, j] += 1
        return ReplicaExchangeMove((i, j), accepted, log_ratio)

    def exchange(self):
        """One symmetric, nonoverlapping random-parity neighboring sweep."""
        parity = int(self.rng.integers(0, 2)) if len(self.states) > 2 else 0
        return tuple(self.attempt(i, i + 1) for i in range(parity, len(self.states) - 1, 2))

    def advance(self, steps, *, gamma_centroid=1.0):
        """Thermostat all slots, then exchange; roll back the full event on error."""
        steps = _integer(steps, "steps", 1)
        # Reject a changed Hamiltonian across every slot before the first MD
        # call: restoring q/p and RNG cannot undo a provider's external history.
        for dyn, bias in zip(self.dynamics, self.biases):
            _static(dyn.potential)
            if bias is not None:
                _static(bias)
        saved_states = [state.copy() for state in self.states]
        saved_rngs = [copy.deepcopy(dyn.rng.bit_generator.state) for dyn in self.dynamics]
        saved_stats = [copy.deepcopy(dyn.propagation_stats) for dyn in self.dynamics]
        own_rng = copy.deepcopy(self.rng.bit_generator.state)
        labels, attempts, accepts = self.labels.copy(), self.attempts.copy(), self.accepts.copy()
        try:
            for dyn, state, bias in zip(self.dynamics, self.states, self.biases):
                for _ in range(steps):
                    dyn.sample_step(state, bias, gamma_centroid=gamma_centroid)
            return self.exchange()
        except Exception:
            for dyn, state, saved, rng, stats in zip(self.dynamics, self.states, saved_states, saved_rngs, saved_stats):
                state.q[:], state.p[:] = saved.q, saved.p
                state.step, state.time = saved.step, saved.time
                dyn.rng.bit_generator.state = rng
                dyn.propagation_stats = stats
            self.rng.bit_generator.state = own_rng
            self.labels[:], self.attempts[:], self.accepts[:] = labels, attempts, accepts
            raise
