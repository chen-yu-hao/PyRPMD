"""Equilibrium sampling of a ring polymer and one shared electron number.

The target is ``exp[-beta/P * (H_P(q,p; N) - P*mu*N)]``.  Nuclear
proposals reuse PyRPMD's reversible real-mass integrator; electron proposals
are symmetric additive moves with rejection at the model's number bounds.
Monte Carlo sweeps and integration proposal lengths are not physical time.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from ..dynamics import Dynamics, State
from ..potentials.base import coordinates
from ..potentials.cache import StepEvaluationCache
from .potentials import FixedElectronNumberPotential, evaluate_electronic
from .reservoir import ElectronReservoir, _finite_real as _real, _electron_bounds


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _static_bias(bias):
    if bias is None:
        return
    if not callable(getattr(bias, "energy_force", None)):
        raise TypeError("bias must implement energy_force(q)")
    if getattr(bias, "history_dependent", False):
        raise ValueError("GrandCanonicalSampler requires a static bias")
    for child in getattr(bias, "biases", ()):
        _static_bias(child)


@dataclass
class GrandCanonicalState:
    """Equilibrium configuration: AU positions, one number, and MC sweeps.

    Positions have shape ``[P, atoms, 3]`` and are copied on construction.
    There is deliberately no real-time clock or persistent fictitious electron
    momentum.  A state alone is not a complete random-stream checkpoint.
    """

    q: np.ndarray
    electron_number: float
    sweeps: int = 0

    def __post_init__(self):
        self.q = coordinates(self.q).copy()
        self.electron_number = _real(self.electron_number, "electron_number")
        self.sweeps = _integer(self.sweeps, "sweeps")

    def copy(self):
        return GrandCanonicalState(self.q, self.electron_number, self.sweeps)


class GrandCanonicalSampler:
    """Alternate fixed-number nuclear HMC and shared-number Metropolis moves.

    ``beta`` and ``dt`` use atomic units. ``electron_bounds`` is the valid
    continuous electron-number interval of the model. ``proposal_width`` is
    the half-width of a uniform additive proposal, not a standard deviation.
    Proposals outside that interval are rejected without clipping or redrawing.

    The model must implement the electronic PES protocol and must already be
    bound to its physical system if it needs cell/PBC information.  ``bias``
    uses the ordinary PyRPMD convention: its ``energy_force`` returns the
    contribution to the *summed* ring Hamiltonian, e.g. ``P*W`` for an umbrella.
    Custom biases must remain deterministic and unchanged during sampling.
    """

    def __init__(
        self, model, masses, beta, nbeads, dt, *, reservoir, electron_bounds,
        proposal_width=0.5, seed=0, native=True, integrator="cayley-bcb",
        fixed_atoms=(), fixed_positions=None,
    ):
        beta = _real(beta, "beta", positive=True)
        dt = _real(dt, "dt", positive=True)
        lower, upper = _electron_bounds(electron_bounds)
        width = _real(proposal_width, "proposal_width", positive=True)
        if not math.isfinite(2 * width):
            raise ValueError("proposal_width is too large for a finite uniform interval")
        if not isinstance(reservoir, ElectronReservoir):
            raise TypeError("reservoir must be an ElectronReservoir")
        mu = reservoir.chemical_potential_au
        self._model = model
        self._bounds = (lower, upper)
        self._proposal_width = width
        self._mu = mu
        self._fixed_potential = FixedElectronNumberPotential(model, lower + (upper - lower) / 2)
        self._potential_cache = StepEvaluationCache(self._fixed_potential)
        self._dynamics = Dynamics(
            self._potential_cache, masses, beta, nbeads, dt, seed=seed,
            native=native, integrator=integrator, loop_backend="python",
            fixed_atoms=fixed_atoms, fixed_positions=fixed_positions,
        )
        self._counts = {
            "sweeps": 0, "nuclear_attempted": 0, "nuclear_accepted": 0,
            "electron_attempted": 0, "electron_accepted": 0,
            "electron_out_of_bounds": 0,
        }

    @property
    def rng(self):
        """The sampler-owned NumPy random generator, also used for momenta."""
        return self._dynamics.rng

    @property
    def native(self):
        """Whether the existing compiled free-ring microkernel is active."""
        return self._dynamics.native

    @property
    def statistics(self):
        """A detached snapshot of attempted/accepted Monte Carlo moves."""
        result = dict(self._counts)
        for kind in ("nuclear", "electron"):
            total = result[kind + "_attempted"]
            result[kind + "_acceptance"] = (
                result[kind + "_accepted"] / total if total else None
            )
        return result

    def _state(self, state):
        if not isinstance(state, GrandCanonicalState):
            raise TypeError("state must be a GrandCanonicalState")
        q = state.q
        if (not isinstance(q, np.ndarray) or q.dtype != np.float64
                or q.shape != self._dynamics.integrator.shape
                or not q.flags.c_contiguous or not q.flags.writeable
                or not np.isfinite(q).all()):
            raise ValueError("state.q must be writable finite float64 coordinates matching the sampler")
        number = _real(state.electron_number, "state.electron_number")
        if not self._bounds[0] <= number <= self._bounds[1]:
            raise ValueError("state electron number is outside electron_bounds")
        _integer(state.sweeps, "state.sweeps")
        self._dynamics._check_fixed(q, np.zeros_like(q))
        return number

    def _grand_energy(self, q, number):
        result = evaluate_electronic(self._model, q, np.full(len(q), number))
        value = float(np.mean(result.energy)) - self._mu * number
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite grand potential energy")
        return value

    def _accept(self, log_ratio):
        if not math.isfinite(log_ratio):
            raise FloatingPointError("nonfinite Metropolis log ratio")
        uniform = float(self.rng.random())
        return uniform == 0 or math.log(uniform) < min(0.0, log_ratio)

    def step(self, state, *, nuclear_steps=5, electron_moves=1, bias=None):
        """Perform one MC sweep and return acceptance diagnostics.

        Set ``nuclear_steps=0`` for conditional number sampling or
        ``electron_moves=0`` for a fixed-number nuclear HMC sweep.  At least
        one move must be requested.  All sampler-owned state, random draws,
        and counters roll back if an evaluation or proposal raises; normal
        Metropolis rejections still count as completed attempts.

        ``physical_grand_energy_au`` is only ``mean_b E(q_b,N) - mu*N``.
        It excludes spring, momentum, and bias contributions; it is neither
        the complete ring-polymer action nor a thermodynamic free energy.
        """
        number = self._state(state)
        nuclear_steps = _integer(nuclear_steps, "nuclear_steps")
        electron_moves = _integer(electron_moves, "electron_moves")
        if nuclear_steps == electron_moves == 0:
            raise ValueError("at least one nuclear or electron move is required")
        _static_bias(bias)
        old_rng = deepcopy(self.rng.bit_generator.state)
        old_number = self._fixed_potential.electron_number
        counts = dict(self._counts)
        q = state.q.copy()
        nuclear_accepted = None
        nuclear_log_acceptance = None
        electron_accepted = electron_outside = 0
        try:
            self._fixed_potential.electron_number = number
            if nuclear_steps:
                # One cache transaction covers only this deterministic,
                # fixed-number proposal. Shared endpoints reuse owned E/F
                # arrays; the context clears before any electronic move.
                with self._potential_cache.step():
                    trial = State(q, self._dynamics.sample_momenta())
                    before = self._dynamics.energy(trial, bias)
                    for _ in range(nuclear_steps):
                        self._dynamics.step(trial, bias)
                    trial.p *= -1.0
                    after = self._dynamics.energy(trial, bias)
                ratio = -self._dynamics.beta * (after - before) / self._dynamics.nbeads
                nuclear_accepted = self._accept(ratio)
                nuclear_log_acceptance = min(0.0, ratio)
                counts["nuclear_attempted"] += 1
                if nuclear_accepted:
                    q = trial.q.copy()
                    counts["nuclear_accepted"] += 1
            energy = self._grand_energy(q, number)
            for _ in range(electron_moves):
                counts["electron_attempted"] += 1
                proposal = number + float(self.rng.uniform(-self._proposal_width, self._proposal_width))
                if not self._bounds[0] <= proposal <= self._bounds[1]:
                    electron_outside += 1
                    counts["electron_out_of_bounds"] += 1
                    continue
                candidate = self._grand_energy(q, proposal)
                if self._accept(-self._dynamics.beta * (candidate - energy)):
                    number, energy = proposal, candidate
                    electron_accepted += 1
                    counts["electron_accepted"] += 1
            self._fixed_potential.electron_number = number
            counts["sweeps"] += 1
            result = {
                "sweep": state.sweeps + 1,
                "nuclear_accepted": nuclear_accepted,
                "nuclear_log_acceptance": nuclear_log_acceptance,
                "electron_attempted": electron_moves,
                "electron_accepted": electron_accepted,
                "electron_out_of_bounds": electron_outside,
                "electron_number": number,
                "physical_grand_energy_au": energy,
            }
        except BaseException:
            self.rng.bit_generator.state = old_rng
            self._fixed_potential.electron_number = old_number
            raise
        state.q[:] = q
        state.electron_number = number
        state.sweeps += 1
        self._counts = counts
        return result


__all__ = ["GrandCanonicalState", "GrandCanonicalSampler"]
