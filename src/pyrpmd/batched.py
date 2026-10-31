"""Independent real-mass ring polymers sharing one physical PES batch.

Only the PES frame axis is flattened.  The free-ring map retains ``P`` beads:
the existing native/NumPy kernel sees ``[P, K*N, 3]`` with repeated atom masses,
so no spring ever connects different trajectories.  All units are atomic.

This is a deterministic NVE API.  It does not change thermostat definitions,
random streams, or the scalar :class:`pyrpmd.dynamics.Dynamics` defaults.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .dynamics import CayleyIntegrator, Dynamics, State
from .potentials.base import PotentialError, validate_result
from .validation import integer_count as _integer


def validate_batch_potential(potential):
    """Require independent, history-independent physical PES frames.

    The built-in analytic providers and stateless C ABI providers have known
    frame semantics.  An arbitrary Python/ML provider must explicitly declare
    ``metadata['trajectory_batch_safe'] = True``.  That promises that changing
    frame order/count does not change any frame's result and that forces do not
    depend on earlier calls.  Explicit contradictory declarations always fail.
    RPC and other ring-coupled wrappers cannot flatten multiple rings.
    """
    metadata = getattr(potential, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        raise ValueError("batched PES metadata must be a mapping")
    capabilities = metadata.get("capabilities", {}) or {}
    if not isinstance(capabilities, dict):
        raise ValueError("batched PES capabilities must be a mapping")
    if (metadata.get("trajectory_batch_safe") is False
            or metadata.get("batch_independent") is False
            or metadata.get("ring_polymer_coupled") is True
            or metadata.get("coupled_frames") is True
            or metadata.get("stateful") is True
            or capabilities.get("stateful") is True):
        raise ValueError("cross-trajectory batching requires independent, stateless PES frames")

    # Recurse through adapters before accepting an outer declaration: a Python
    # wrapper must not hide a ring-coupled or explicitly stateful inner object.
    from .potentials.python import _PYTHON_EVALUATE, PythonPotential
    if type(potential) is PythonPotential:
        if "evaluate" not in potential.__dict__ and PythonPotential.evaluate is _PYTHON_EVALUATE:
            inner = getattr(potential.obj, "metadata", {}) or {}
            if isinstance(inner, dict) and any(
                inner.get(key) is value for key, value in (
                    ("trajectory_batch_safe", False), ("batch_independent", False),
                    ("ring_polymer_coupled", True), ("coupled_frames", True), ("stateful", True),
                )
            ):
                raise ValueError("Python PES wraps a coupled/stateful provider")
            inner_capabilities = inner.get("capabilities", {}) if isinstance(inner, dict) else {}
            if isinstance(inner_capabilities, dict) and inner_capabilities.get("stateful") is True:
                raise ValueError("Python PES wraps a coupled/stateful provider")
    if metadata.get("trajectory_batch_safe") is True:
        return

    from .potentials.builtin import (
        _BUILTIN_EVALUATE,
        DoubleWellPotential,
        FreePotential,
        HarmonicPotential,
        MorsePairPotential,
        PeriodicCosinePotential,
    )
    known = (FreePotential, HarmonicPotential, MorsePairPotential,
             DoubleWellPotential, PeriodicCosinePotential)
    if (type(potential) in known and "evaluate" not in potential.__dict__
            and type(potential).evaluate is _BUILTIN_EVALUATE[type(potential)]):
        if type(potential) is HarmonicPotential and (
            potential.k.ndim > 2 or potential.center.ndim > 2
        ):
            raise ValueError("batched harmonic parameters cannot depend on the frame axis")
        return
    from .potentials.cabi import _CABI_EVALUATE, CABIPotential
    if (type(potential) is CABIPotential
            and "evaluate" not in potential.__dict__
            and CABIPotential.evaluate is _CABI_EVALUATE
            and capabilities.get("stateful") is False):
        return
    if (type(potential) is PythonPotential
            and "evaluate" not in potential.__dict__
            and PythonPotential.evaluate is _PYTHON_EVALUATE):
        # Unit conversion and force/gradient sign conversion are frame-local.
        validate_batch_potential(potential.obj)
        return
    raise ValueError(
        "custom PES must declare metadata['trajectory_batch_safe']=True "
        "before cross-trajectory batching"
    )


@dataclass
class BatchedState:
    """Owned ``q,p[K,P,N,3]`` arrays with one step/time clock per trajectory."""

    q: np.ndarray
    p: np.ndarray
    step: object = 0
    time: object = 0.0

    def __post_init__(self):
        self.q = np.array(self.q, dtype=np.float64, order="C", copy=True)
        self.p = np.array(self.p, dtype=np.float64, order="C", copy=True)
        if (self.q.ndim != 4 or self.q.shape[-1] != 3 or min(self.q.shape) < 1
                or self.p.shape != self.q.shape):
            raise ValueError("batched q,p must have equal nonempty [K,P,N,3] shapes")
        if not np.isfinite(self.q).all() or not np.isfinite(self.p).all():
            raise ValueError("batched q,p must be finite")
        count = len(self.q)
        raw_step = np.asarray(self.step)
        if (raw_step.dtype.kind not in "iu" or raw_step.shape not in ((), (count,))
                or np.any(raw_step < 0)
                or np.any(raw_step > np.iinfo(np.int64).max)):
            raise ValueError("batched step must be a nonnegative integer or integer[K]")
        raw_time = np.asarray(self.time)
        if (raw_time.dtype.kind not in "fiu" or raw_time.shape not in ((), (count,))
                or not np.isfinite(raw_time).all()):
            raise ValueError("batched time must be finite scalar or time[K]")
        self.step = np.broadcast_to(raw_step, (count,)).astype(np.int64, copy=True)
        self.time = np.broadcast_to(raw_time, (count,)).astype(np.float64, copy=True)

    @classmethod
    def from_states(cls, states):
        states = tuple(states)
        if not states or any(not isinstance(state, State) for state in states):
            raise ValueError("from_states requires a nonempty sequence of State objects")
        return cls(np.stack([s.q for s in states]), np.stack([s.p for s in states]),
                   [s.step for s in states], [s.time for s in states])

    def copy(self):
        return BatchedState(self.q, self.p, self.step, self.time)

    def to_states(self):
        """Return independent scalar states in the original trajectory order."""
        return [State(q, p, int(step), float(time))
                for q, p, step, time in zip(self.q, self.p, self.step, self.time)]


@dataclass
class BatchedBlockResult:
    """Sampled observations and completion/error status in original lane order.

    Rows contain only trajectories that completed that observation step;
    ``valid`` identifies them and unavailable numerical samples are NaN.
    A stopping step is always recorded, even when it is between stride points.
    Failed trajectories retain their last completely committed state.
    """

    requested: int
    completed: np.ndarray
    selected: np.ndarray
    stopped: np.ndarray
    failed: np.ndarray
    errors: tuple
    offsets: np.ndarray
    valid: np.ndarray
    steps: np.ndarray
    times: np.ndarray
    energies: np.ndarray
    initial_energy: np.ndarray
    centroids: np.ndarray | None
    centroid_momenta: np.ndarray | None
    values: np.ndarray | None
    pes_calls: int
    pes_frames: int
    backend: str

    @property
    def success(self):
        return not self.failed.any() and bool(np.all(
            ~self.selected | self.stopped | (self.completed == self.requested)
        ))


class BatchedPropagationError(PotentialError):
    def __init__(self, result):
        self.result = result
        messages = [f"trajectory {i}: {error}" for i, error in enumerate(result.errors) if error]
        super().__init__("; ".join(messages[:8]))


class BatchedDynamics:
    """Batch deterministic trajectories using the established ring-polymer map.

    ``stop(index, state)`` is an optional per-trajectory callback returning a
    boolean after a trial full step.  The supplied State is an independent
    snapshot.  A failing callback rolls back that trajectory's trial step.
    ``observe`` uses the existing CV ``value(q)`` / ``dimension`` interface.
    ``bias`` may be one common bias or a sequence with one bias per trajectory;
    each is evaluated on exactly one ring.  Biases must be time independent.

    Potentials remain owned by the caller and are not closed by this object.
    """

    def __init__(self, potential, masses, beta, nbeads, dt, *, native=True,
                 threads=1, integrator="cayley-bcb", fixed_atoms=(), fixed_positions=None):
        validate_batch_potential(potential)
        self._prototype = Dynamics(
            potential, masses, beta, nbeads, dt, native=native, threads=threads,
            integrator=integrator, loop_backend="python", fixed_atoms=fixed_atoms,
            fixed_positions=fixed_positions,
        )
        self.potential = potential
        self.integrator = self._prototype.integrator
        self.masses = self.integrator.masses
        self.beta = self.integrator.beta
        self.nbeads = self.integrator.nbeads
        self.dt = self.integrator.dt
        self.integrator_name = self.integrator.scheme
        self.native = self.integrator.native
        self._native_requested = native
        self._plans = OrderedDict([(1, self.integrator)])
        self._lock = threading.Lock()
        self.pes_calls = 0
        self.pes_frames = 0

    def _check(self, state):
        if not isinstance(state, BatchedState):
            raise TypeError("state must be a BatchedState")
        if (self.beta != self.integrator.beta or self.dt != self.integrator.dt
                or self.nbeads != self.integrator.nbeads
                or self.integrator_name != self.integrator.scheme):
            raise ValueError("BatchedDynamics integration parameters are immutable")
        if (state.q.shape[1:] != self.integrator.shape or state.p.shape != state.q.shape
                or state.step.shape != (len(state.q),) or state.time.shape != (len(state.q),)
                or state.step.dtype != np.int64 or state.time.dtype != np.float64
                or np.any(state.step < 0) or not np.isfinite(state.time).all()):
            raise ValueError("batched state shape/clock differs from the integrator")
        if np.shares_memory(state.q, state.p):
            raise ValueError("batched q and p must not share memory")
        if not state.step.flags.writeable or not state.time.flags.writeable:
            raise ValueError("batched clocks must be writable")
        for q, p in zip(state.q, state.p):
            self.integrator._check(q, p)
            self._prototype._check_fixed(q, p)

    def _plan(self, count):
        plan = self._plans.pop(count, None)
        if plan is None:
            plan = CayleyIntegrator(
                np.tile(self.masses, count), self.beta, self.nbeads, self.dt,
                threads=self.integrator.threads, native=self._native_requested,
                scheme=self.integrator_name,
            )
        self._plans[count] = plan
        # Stop masks can produce many active counts; bound prepared-plan memory.
        while len(self._plans) > 4:
            self._plans.popitem(last=False)
        return plan

    def _free_step(self, q, p):
        count, beads, atoms, _ = q.shape
        packed_q = np.ascontiguousarray(q.transpose(1, 0, 2, 3)).reshape(beads, count * atoms, 3)
        packed_p = np.ascontiguousarray(p.transpose(1, 0, 2, 3)).reshape(beads, count * atoms, 3)
        self._plan(count).free_step(packed_q, packed_p)
        q[:] = packed_q.reshape(beads, count, atoms, 3).transpose(1, 0, 2, 3)
        p[:] = packed_p.reshape(beads, count, atoms, 3).transpose(1, 0, 2, 3)

    def _project_fixed(self, q, p):
        fixed = self._prototype._fixed_indices
        if len(fixed):
            q[:, :, fixed, :] = self._prototype.fixed_positions[fixed]
            p[:, :, fixed, :] = 0.0

    def _evaluate(self, q, indices, biases):
        count, beads, atoms, _ = q.shape
        flat = np.ascontiguousarray(q).reshape(count * beads, atoms, 3)
        self.pes_calls += 1
        self.pes_frames += len(flat)
        result = validate_result(self.potential.evaluate(flat), flat.shape, require_success=False)
        # Some PESs reuse a writable output buffer. Never retain its view.
        force = result.forces.copy().reshape(q.shape)
        energy = result.energy.copy().reshape(count, beads).sum(axis=1)
        status = result.status.reshape(count, beads)
        errors = [None] * count
        for j, index in enumerate(indices):
            bad = np.flatnonzero(status[j])
            if len(bad):
                errors[j] = "PES frames failed: " + ", ".join(
                    f"bead {bead} (status={status[j, bead]}: {result.messages[j*beads+bead]})"
                    for bead in bad[:8]
                )
                continue
            bias = biases[index]
            if bias is not None:
                try:
                    extra_energy, extra_force = bias.energy_force(q[j])
                    extra_force = np.asarray(extra_force)
                    if (np.ndim(extra_energy) != 0 or not np.isfinite(extra_energy)
                            or extra_force.shape != q[j].shape
                            or not np.isfinite(extra_force).all()):
                        raise ValueError("bias returned invalid energy/forces")
                    energy[j] += float(extra_energy)
                    force[j] += extra_force
                except Exception as exc:
                    errors[j] = f"{type(exc).__name__}: {exc}"
        force[:, :, self._prototype._fixed_indices, :] = 0.0
        return energy, force, errors

    def _hamiltonian(self, q, p, energy):
        dq = q - np.roll(q, -1, axis=1)
        return (energy + np.sum(p*p / (2*self.masses[None, None, :, None]), axis=(1, 2, 3))
                + 0.5*(self.nbeads/self.beta)**2
                * np.sum(self.masses[None, None, :, None]*dq*dq, axis=(1, 2, 3)))

    def energy(self, state, *, active=None, bias=None):
        """Return ``H_P[K]``; unselected trajectories have NaN energy."""
        self._check(state)
        selected = self._mask(active, len(state.q))
        indices = np.flatnonzero(selected)
        out = np.full(len(state.q), np.nan)
        if len(indices):
            with self._lock:
                energy, _, errors = self._evaluate(state.q[indices], indices,
                                                   self._biases(bias, len(state.q)))
                if any(errors):
                    raise PotentialError("; ".join(f"trajectory {i}: {error}" for i, error in zip(indices, errors) if error))
                out[indices] = self._hamiltonian(state.q[indices], state.p[indices], energy)
        return out

    @staticmethod
    def _mask(active, count):
        if active is None:
            return np.ones(count, dtype=bool)
        active = np.asarray(active)
        if active.dtype != np.bool_ or active.shape != (count,):
            raise ValueError("active must be a boolean[K] mask")
        return active.copy()

    @staticmethod
    def _biases(bias, count):
        if bias is None or callable(getattr(bias, "energy_force", None)):
            biases = (bias,) * count
        else:
            biases = tuple(bias)
            if len(biases) != count:
                raise ValueError("bias sequence must have one entry per trajectory")
        for item in biases:
            if item is not None and not callable(getattr(item, "energy_force", None)):
                raise ValueError("bias must provide energy_force(q)")
            # Depositing hills between force evaluations invalidates cached forces.
            if item is not None and callable(getattr(item, "deposit", None)):
                raise ValueError("time-dependent/depositing biases require a dedicated sampling workflow")
        return biases

    def step(self, state, *, active=None, bias=None, raise_on_error=True):
        """Advance one full step; return the per-trajectory completion result."""
        return self.propagate_block(state, 1, active=active, bias=bias,
                                    record_centroids=False, raise_on_error=raise_on_error)

    def propagate_block(self, state, steps, *, active=None, bias=None, observe=None,
                        sample_interval=1, stop=None, record_centroids=True,
                        raise_on_error=True):
        """Advance selected trajectories, retaining every valid completed prefix.

        A physical PES call is made once at block entry and once per subsequent
        integration step, on all still-active trajectories.  Force errors are
        never retried with a different backend.  Frame-level PES errors affect
        only their trajectory; exceptions/malformed whole-batch results affect
        every trajectory in that attempted call.  With ``raise_on_error=True``
        the first failing iteration returns through :class:`BatchedPropagationError`
        after committing its successful lanes.  With False, healthy trajectories
        continue to the requested duration.  Stopped/inactive lanes are omitted
        from all later potential calls.
        """
        steps = _integer(steps, "steps")
        stride = _integer(sample_interval, "sample_interval", 1)
        self._check(state)
        count = len(state.q)
        selected = self._mask(active, count)
        biases = self._biases(bias, count)
        if stop is not None and not callable(stop):
            raise TypeError("stop must be callable")
        dimension = 0 if observe is None else _integer(observe.dimension, "CV dimension", 1)
        if observe is not None and not callable(getattr(observe, "value", None)):
            raise TypeError("observe must provide value(q) and dimension")
        if not isinstance(raise_on_error, (bool, np.bool_)):
            raise ValueError("raise_on_error must be boolean")
        with self._lock:
            return self._propagate_block(state, steps, selected, biases, observe, dimension,
                                         stride, stop, bool(record_centroids), bool(raise_on_error))

    def _propagate_block(self, state, steps, selected, biases, observe, dimension,
                         stride, stop, record_centroids, raise_on_error):
        count, _, atoms, _ = state.q.shape
        active = selected.copy()
        completed = np.zeros(count, dtype=np.int64)
        stopped = np.zeros(count, dtype=bool)
        failed = np.zeros(count, dtype=bool)
        errors = [None] * count
        initial_energy = np.full(count, np.nan)
        force = np.zeros_like(state.q)
        rows = []
        calls, frames = self.pes_calls, self.pes_frames

        def fail(indices, reasons):
            for index, reason in zip(indices, reasons):
                if reason is not None:
                    failed[index] = True
                    active[index] = False
                    errors[index] = reason

        def evaluate(q, indices):
            try:
                return self._evaluate(q, indices, biases)
            except Exception as exc:
                return None, None, [f"{type(exc).__name__}: {exc}"] * len(indices)

        if steps and active.any():
            indices = np.flatnonzero(active)
            energy, forces, reasons = evaluate(state.q[indices], indices)
            fail(indices, reasons)
            if forces is not None:
                good = np.array([reason is None for reason in reasons])
                ids = indices[good]
                force[ids] = forces[good]
                initial_energy[ids] = self._hamiltonian(state.q[ids], state.p[ids], energy[good])

        for offset in range(1, steps + 1):
            if not active.any() or (raise_on_error and failed.any()):
                break
            indices = np.flatnonzero(active)
            q, p = state.q[indices].copy(), state.p[indices].copy()
            # Work arrays isolate every attempted step from the live state.
            p += 0.5*self.dt*force[indices]
            self._project_fixed(q, p)
            try:
                self._free_step(q, p)
                self._project_fixed(q, p)
            except Exception as exc:
                fail(indices, [f"{type(exc).__name__}: {exc}"]*len(indices))
                break
            energy, forces, reasons = evaluate(q, indices)
            fail(indices, reasons)
            committed = np.zeros(count, dtype=bool)
            trial_stopped = np.zeros(count, dtype=bool)
            row_energy = np.full(count, np.nan)
            row_values = np.full((count, dimension), np.nan) if observe is not None else None
            if forces is not None:
                for j, index in enumerate(indices):
                    if failed[index]:
                        continue
                    try:
                        p[j] += 0.5*self.dt*forces[j]
                        self._project_fixed(q[j:j+1], p[j:j+1])
                        if not np.isfinite(q[j]).all() or not np.isfinite(p[j]).all():
                            raise FloatingPointError("nonfinite integration state")
                        if state.step[index] == np.iinfo(np.int64).max:
                            raise OverflowError("trajectory step counter overflow")
                        next_step = int(state.step[index]) + 1
                        next_time = float(state.time[index]) + self.dt
                        if not np.isfinite(next_time):
                            raise FloatingPointError("trajectory time overflow")
                        value = None
                        if observe is not None:
                            value = np.asarray(observe.value(q[j]), dtype=float)
                            if value.shape != (dimension,) or not np.isfinite(value).all():
                                raise ValueError("CV returned invalid values")
                        should_stop = False if stop is None else stop(
                            int(index), State(q[j], p[j], next_step, next_time))
                        if not isinstance(should_stop, (bool, np.bool_)):
                            raise ValueError("stop callback must return one boolean")
                        hamiltonian = self._hamiltonian(q[j:j+1], p[j:j+1], energy[j:j+1])[0]
                        if not np.isfinite(hamiltonian):
                            raise FloatingPointError("nonfinite trajectory energy")
                    except Exception as exc:
                        fail([index], [f"{type(exc).__name__}: {exc}"])
                        continue
                    state.q[index], state.p[index] = q[j], p[j]
                    state.step[index], state.time[index] = next_step, next_time
                    force[index] = forces[j]
                    completed[index] += 1
                    committed[index] = True
                    row_energy[index] = hamiltonian
                    if row_values is not None:
                        row_values[index] = value
                    if should_stop:
                        stopped[index] = trial_stopped[index] = True
                        active[index] = False
            if committed.any() and (offset % stride == 0 or offset == steps
                                    or trial_stopped.any() or (raise_on_error and failed.any())):
                row_steps = np.full(count, -1, dtype=np.int64)
                row_times = np.full(count, np.nan)
                row_steps[committed] = state.step[committed]
                row_times[committed] = state.time[committed]
                centroids = momenta = None
                if record_centroids:
                    centroids = np.full((count, atoms, 3), np.nan)
                    momenta = np.full((count, atoms, 3), np.nan)
                    centroids[committed] = state.q[committed].mean(axis=1)
                    momenta[committed] = state.p[committed].mean(axis=1)
                rows.append((offset, committed, row_steps, row_times, row_energy,
                             centroids, momenta, row_values))

        def stack(position, shape, dtype=float):
            return np.stack([row[position] for row in rows]) if rows else np.empty((0, *shape), dtype=dtype)

        result = BatchedBlockResult(
            requested=steps, completed=completed, selected=selected, stopped=stopped,
            failed=failed, errors=tuple(errors), offsets=np.array([row[0] for row in rows], dtype=np.int64),
            valid=stack(1, (count,), bool), steps=stack(2, (count,), np.int64),
            times=stack(3, (count,)), energies=stack(4, (count,)), initial_energy=initial_energy,
            centroids=stack(5, (count, atoms, 3)) if record_centroids else None,
            centroid_momenta=stack(6, (count, atoms, 3)) if record_centroids else None,
            values=stack(7, (count, dimension)) if observe is not None else None,
            pes_calls=self.pes_calls-calls, pes_frames=self.pes_frames-frames,
            backend="native-ring/batched-pes" if self.native else "numpy-ring/batched-pes",
        )
        if raise_on_error and failed.any():
            raise BatchedPropagationError(result)
        return result
