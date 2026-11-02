"""Metropolized fixed-internal-coordinate PIMD for bent triatomics.

This is an equilibrium conditional sampler, not a real-time QCMD integrator.
It samples exp(-beta/P H_ring) delta(mean_beads(internals)-target).  RATTLE
proposals use small molecule-local metric blocks; a Fixman determinant in the
Metropolis ratio removes the rigid-surface measure.  No dense Hessian is used.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from .checkpoint import atomic_savez, load_checkpoint
from .cmd import _integer, _scalar
from .dynamics import State, _state_clock
from .io import jsonable
from .potentials.base import validate_result
from ._qcmd_geometry import QuasiCentroidConstraint, TriatomicQuasiCentroid


_KIND = "PyRPMD.quasicentroid-sampler.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _potential_identity(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("potential_identity must be a nonempty scientific model/content fingerprint")
    return value


class ConstrainedQuasiCentroidSampler:
    """Conditional PIMD using fully refreshed RATTLE HMC proposals.

    ``geometry`` specifies nonperiodic disjoint bent triatomics. ``target`` is
    the fixed bead-mean pair of bond lengths and included angle for each
    molecule. Overall translation and rotation are unconstrained. The sum-bead
    ring Hamiltonian uses physical masses and inverse temperature ``beta/P``.

    A proposal follows ``trajectory_steps`` primitive velocity-Verlet RATTLE
    steps, then flips momentum.  Metropolis acceptance includes
    ``-0.5 * (logdet(G_new)-logdet(G_old))`` with ``G=C M^-1 C.T``.  Thus the
    sampled configurational measure is the Cartesian conditional measure
    ``dq delta(C(q)-target)``, not the rigid constrained-surface measure.
    A reverse trajectory is checked before acceptance. Failed projections or
    reversibility checks raise and leave the state, RNG and counters unchanged.

    ``State.step`` counts HMC proposals and ``State.time`` is their accumulated
    proposal integration time. Neither is a physical real-time QCMD trajectory.
    Potential identity must be supplied explicitly when saving/loading; include
    parameter values and any external model's content hash in that identity.
    """

    def __init__(self, potential, geometry, target, *, beta, nbeads, dt,
                 trajectory_steps=1, seed=0, tolerance=1e-10,
                 max_iterations=50, reversibility_tolerance=1e-8):
        if not isinstance(geometry, TriatomicQuasiCentroid):
            raise TypeError("geometry must be a TriatomicQuasiCentroid")
        if not callable(getattr(potential, "evaluate", None)):
            raise TypeError("potential must implement evaluate")
        metadata = getattr(potential, "metadata", {}) or {}
        # The legacy metadata["periodic"] flag declares capability (including
        # FreePotential); it does not mean a periodic cell is currently active.
        if (metadata.get("boundary_conditions") == "periodic"
                or np.asarray(getattr(potential, "pbc", False), dtype=bool).any()):
            raise ValueError("quasi-centroid sampling currently requires a nonperiodic potential")
        self.potential, self.geometry = potential, geometry
        self.constraint = QuasiCentroidConstraint(geometry, target, tolerance=tolerance,
                                                  max_iterations=max_iterations)
        self.beta = _scalar(beta, "beta")
        self.nbeads = _integer(nbeads, "nbeads")
        self.dt = _scalar(dt, "dt")
        self.trajectory_steps = _integer(trajectory_steps, "trajectory_steps")
        self.reversibility_tolerance = _scalar(reversibility_tolerance, "reversibility_tolerance")
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.shape = (self.nbeads, geometry.natoms, 3)
        self.inverse_mass = 1 / geometry.masses[None, :, None]
        self.omega2 = (self.nbeads / self.beta) ** 2
        if (not math.isfinite(self.omega2) or not math.isfinite(self.dt * self.trajectory_steps)
                or not np.isfinite(self.inverse_mass).all()):
            raise ValueError("quasi-centroid sampling mass/frequency scale overflows")
        # Primitive ring Verlet remains subject to its ordinary spring limit.
        if self.nbeads > 1 and self.dt * 2 * self.nbeads / self.beta >= 2:
            raise ValueError("primitive ring RATTLE timestep exceeds the free-spring stability limit")
        self.inverse_mass.flags.writeable = False
        self.attempted = 0
        self.accepted = 0
        self.last_log_acceptance = None
        self._settings_identity = _digest(self.metadata())

    def metadata(self):
        return {"method": "conditional-quasicentroid-pimd", "integrator": "rattle-hmc",
                "beta_au": self.beta, "beads": self.nbeads, "dt_au": self.dt,
                "trajectory_steps": self.trajectory_steps,
                "reversibility_tolerance": self.reversibility_tolerance,
                "constraint": self.constraint.metadata(),
                "measure": "cartesian_delta_constraint_with_fixman_metropolis",
                "momentum_refresh": "full_tangent_maxwell", "real_time_dynamics": False,
                "rng_bit_generator": type(self.rng.bit_generator).__name__}

    def _settings_check(self):
        if _digest(self.metadata()) != self._settings_identity:
            raise ValueError("quasi-centroid sampler settings are immutable; create a new sampler")

    def _positions_check(self, q):
        if np.shape(q) != self.shape:
            raise ValueError(f"quasi-centroid sampler requires shape {self.shape}")
        if np.max(np.abs(self.constraint.values(q))) > self.constraint.tolerance:
            raise ValueError("initial ring does not satisfy the quasi-centroid constraints; project it explicitly")

    def _state_check(self, state):
        self._settings_check()
        if not isinstance(state, State):
            raise TypeError("state must be a State")
        for array in (state.q, state.p):
            if (array.shape != self.shape or array.dtype != np.float64
                    or not array.flags.c_contiguous or not array.flags.writeable
                    or not np.isfinite(array).all()):
                raise ValueError("state must contain writable contiguous finite float64 arrays of the sampler shape")
        if np.shares_memory(state.q, state.p):
            raise ValueError("state positions and momenta cannot share memory")
        step, time = _state_clock(state.step, state.time)
        if step != self.attempted:
            raise ValueError("state proposal clock differs from sampler counters; load the complete checkpoint")
        if not np.isclose(time, step * self.dt * self.trajectory_steps, rtol=2e-12, atol=2e-12):
            raise ValueError("state time differs from the HMC proposal clock")
        self._positions_check(state.q)
        if np.max(np.abs(self.constraint.velocity_residual(state.q, state.p))) > self.reversibility_tolerance:
            raise ValueError("state momenta do not satisfy quasi-centroid velocity constraints")
        return step, time

    def sample_momenta(self, q):
        """Draw momenta from the constrained canonical Gaussian at beta/P."""
        self._settings_check()
        self._positions_check(q)
        before = copy.deepcopy(self.rng.bit_generator.state)
        try:
            p = self.rng.normal(size=self.shape) * np.sqrt(
                self.geometry.masses[None, :, None] * self.nbeads / self.beta)
            return self.constraint.project_momenta(q, p)
        except BaseException:
            self.rng.bit_generator.state = before
            raise

    def _energy_force(self, q):
        result = validate_result(self.potential.evaluate(q), q.shape)
        difference = q - np.roll(q, 1, axis=0)
        mass = self.geometry.masses[None, :, None]
        energy = float(result.energy.sum() + .5 * self.omega2 * np.sum(mass * difference * difference))
        force = result.forces - self.omega2 * mass * (2 * q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0))
        if not math.isfinite(energy) or not np.isfinite(force).all():
            raise FloatingPointError("nonfinite constrained ring energy or force")
        return energy, force

    def _trajectory(self, q, p):
        q, p = q.copy(), p.copy()
        energy, force = self._energy_force(q)
        for _ in range(self.trajectory_steps):
            p += .5 * self.dt * force
            unconstrained = q + self.dt * self.inverse_mass * p
            projected = self.constraint.project_positions(unconstrained, reference=q)
            p += (projected - unconstrained) / (self.dt * self.inverse_mass)
            q = projected
            energy, force = self._energy_force(q)
            p += .5 * self.dt * force
            p = self.constraint.project_momenta(q, p)
        return q, p, energy

    def step(self, state):
        """Advance one Markov proposal transactionally and return ``state``."""
        step, time = self._state_check(state)
        next_time = time + self.dt * self.trajectory_steps
        if not math.isfinite(next_time):
            raise FloatingPointError("nonfinite HMC proposal clock")
        before_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            start_q = state.q.copy()
            start_p = self.sample_momenta(start_q)
            old_energy, _ = self._energy_force(start_q)
            q, p, new_energy = self._trajectory(start_q, start_p)
            reverse_q, reverse_p, _ = self._trajectory(q, -p)
            # Relative scales keep the momentum check meaningful across physical
            # atomic masses while a unit absolute scale protects zeros.
            q_error = np.max(np.abs(reverse_q - start_q) / np.maximum(1., np.abs(start_q)))
            p_error = np.max(np.abs(reverse_p + start_p) / np.maximum(1., np.abs(start_p)))
            if max(q_error, p_error) > self.reversibility_tolerance:
                raise ValueError("RATTLE proposal failed reversibility check; reduce dt or tighten constraint tolerance")
            old_h = old_energy + .5 * np.sum(start_p * start_p * self.inverse_mass)
            new_h = new_energy + .5 * np.sum(p * p * self.inverse_mass)
            log_acceptance = (-self.beta / self.nbeads * (new_h - old_h)
                              - .5 * (self.constraint.logdet_metric(q)
                                      - self.constraint.logdet_metric(start_q)))
            if not math.isfinite(log_acceptance):
                raise FloatingPointError("nonfinite quasi-centroid HMC acceptance ratio")
            uniform = self.rng.random()
            accepted = bool(uniform == 0 or math.log(uniform) < min(0., log_acceptance))
            next_q, next_p = (q, -p) if accepted else (start_q, start_p)
        except BaseException:
            self.rng.bit_generator.state = before_rng
            raise
        state.q[:] = next_q
        state.p[:] = next_p
        state.step, state.time = step + 1, next_time
        self.attempted += 1
        self.accepted += int(accepted)
        self.last_log_acceptance = float(log_acceptance)
        return state

    @property
    def acceptance_rate(self):
        return self.accepted / self.attempted if self.attempted else None

    def save_checkpoint(self, path, state, *, potential_identity):
        """Atomically save the state, complete RNG, counters and model identity."""
        self._state_check(state)
        model = _potential_identity(potential_identity)
        settings = self.metadata()
        identity = _digest({"settings": settings, "potential_identity": model})
        atomic_savez(path, kind=_KIND, fingerprint=identity, settings=json.dumps(settings, sort_keys=True),
                     potential_identity=model, q=state.q, p=state.p, step=state.step, time=state.time,
                     rng=json.dumps(jsonable(self.rng.bit_generator.state), sort_keys=True),
                     attempted=self.attempted, accepted=self.accepted,
                     last_log_acceptance=json.dumps(self.last_log_acceptance))

    def load_checkpoint(self, path, *, potential_identity):
        """Restore a matching checkpoint and return its state, or fail atomically."""
        self._settings_check()
        model = _potential_identity(potential_identity)
        expected = _digest({"settings": self.metadata(), "potential_identity": model})
        with load_checkpoint(path, expected_kind=_KIND, require_integrity=True) as data:
            if (str(data["fingerprint"]) != expected or str(data["potential_identity"]) != model
                    or json.loads(str(data["settings"])) != self.metadata()):
                raise ValueError("quasi-centroid checkpoint physics or potential identity changed")
            for key in ("step", "time", "attempted", "accepted", "rng", "last_log_acceptance"):
                if np.asarray(data[key]).ndim:
                    raise ValueError(f"checkpoint {key} must be scalar")
            step = _integer(data["step"].item(), "checkpoint step", 0)
            attempted = _integer(data["attempted"].item(), "checkpoint attempted", 0)
            accepted = _integer(data["accepted"].item(), "checkpoint accepted", 0)
            if step != attempted or accepted > attempted:
                raise ValueError("invalid quasi-centroid checkpoint counters")
            state = State(data["q"], data["p"], step, data["time"].item())
            last_log_acceptance = json.loads(str(data["last_log_acceptance"]))
            if ((attempted == 0 and last_log_acceptance is not None)
                    or (attempted > 0 and (isinstance(last_log_acceptance, bool)
                        or not isinstance(last_log_acceptance, (int, float))
                        or not math.isfinite(last_log_acceptance)))):
                raise ValueError("invalid checkpoint acceptance diagnostic")
            rng_state = json.loads(str(data["rng"]))
        old = (self.attempted, self.accepted, self.last_log_acceptance,
               copy.deepcopy(self.rng.bit_generator.state))
        try:
            self.attempted, self.accepted = attempted, accepted
            self.last_log_acceptance = last_log_acceptance
            self.rng.bit_generator.state = rng_state
            self._state_check(state)
            if np.max(np.abs(self.constraint.velocity_residual(state.q, state.p))) > self.reversibility_tolerance:
                raise ValueError("checkpoint momenta do not satisfy velocity constraints")
        except BaseException:
            self.attempted, self.accepted, self.last_log_acceptance, rng = old
            self.rng.bit_generator.state = rng
            raise
        return state
