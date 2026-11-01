"""Joint trajectory/bead NRPMD batches with exact electronic impulses.

PES evaluation and electronic propagation each see all selected trajectories in
one call. A full batch step commits only after every lane succeeds. The driver
borrows its potential and does not close it. Stochastic hopping is not batched
by this deterministic mapping API.
"""
from dataclasses import dataclass
import numpy as np

from .batched import BatchedState
from .dynamics import CayleyIntegrator
from .nonadiabatic import NRPMDState, NRPMDDynamics, validate_diabatic_result
from .electronic_backends import electronic_mapping_flow


@dataclass
class BatchedNRPMDState(BatchedState):
    mapping_q: np.ndarray | None = None
    mapping_p: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("mapping_q", "mapping_p"):
            a = np.asarray(getattr(self, name))
            if (a.dtype.kind not in "fiu" or a.ndim != 3 or min(a.shape) < 1
                    or a.shape[:2] != self.q.shape[:2] or not np.isfinite(a).all()):
                raise ValueError("batched mapping variables must be finite real [K,P,S]")
            setattr(self, name, np.array(a, dtype=np.float64, order="C", copy=True))
        if self.mapping_q.shape != self.mapping_p.shape:
            raise ValueError("mapping shapes disagree")

    @classmethod
    def from_states(cls, states):
        states = tuple(states)
        if not states or any(not isinstance(s, NRPMDState) for s in states):
            raise ValueError("a nonempty sequence of NRPMDState objects is required")
        return cls(np.stack([s.q for s in states]), np.stack([s.p for s in states]),
                   [s.step for s in states], [s.time for s in states],
                   np.stack([s.mapping_q for s in states]), np.stack([s.mapping_p for s in states]))

    def copy(self):
        return BatchedNRPMDState(self.q, self.p, self.step, self.time, self.mapping_q, self.mapping_p)

    def to_states(self):
        return tuple(NRPMDState(q,p,int(step),float(time),x,y) for q,p,step,time,x,y in
                     zip(self.q,self.p,self.step,self.time,self.mapping_q,self.mapping_p))


class BatchedNRPMDDynamics:
    """One PES/eigensystem batch across trajectories and beads, double precision."""
    def __init__(self, potential, masses, beta, nbeads, dt, nstates, *, native=True,
                 threads=1, integrator="cayley-bcb", electronic_backend="numpy",
                 electronic_device="cpu"):
        if getattr(potential, "metadata", {}).get("trajectory_batch_safe") is not True:
            raise ValueError("cross-trajectory PES requires metadata['trajectory_batch_safe']=True")
        self._prototype = NRPMDDynamics(
            potential, masses, beta, nbeads, dt, nstates, native=native, threads=threads,
            integrator=integrator, electronic_backend=electronic_backend,
            electronic_device=electronic_device)
        self.potential, self.masses = potential, self._prototype.masses
        self._mass_snapshot = np.array(self.masses, copy=True)
        self.nbeads, self.dt, self.beta, self.nstates = nbeads, dt, beta, nstates
        self.electronic_backend, self.electronic_device = electronic_backend, electronic_device
        self.integrator_name, self.threads = self._prototype.integrator_name, threads
        self._native_requested, self._plan_count, self._plan = native, None, None
        self._parameters = (self.nbeads, self.dt, self.beta, self.nstates,
                            self.electronic_backend, self.electronic_device,
                            self.integrator_name, self.threads)
        self.pes_calls = self.pes_frames = 0

    def _check(self, state):
        current = (self.nbeads, self.dt, self.beta, self.nstates,
                   self.electronic_backend, self.electronic_device,
                   self.integrator_name, self.threads)
        if current != self._parameters:
            raise ValueError("batched dynamics parameters are immutable")
        if not np.array_equal(self.masses, self._mass_snapshot):
            raise ValueError("batched dynamics masses are immutable")
        if not isinstance(state, BatchedNRPMDState):
            raise TypeError("BatchedNRPMDState is required")
        if (state.q.ndim != 4 or state.q.shape[1:] != (self.nbeads, len(self.masses),3)
                or state.p.shape != state.q.shape
                or state.mapping_q.shape != (len(state.q),self.nbeads,self.nstates)
                or state.mapping_p.shape != state.mapping_q.shape):
            raise ValueError("batched state shape mismatch")
        arrays = (state.q,state.p,state.mapping_q,state.mapping_p)
        if any(a.dtype != np.float64 or not a.flags.c_contiguous or not a.flags.writeable
               or not np.isfinite(a).all() for a in arrays):
            raise ValueError("batched state must have finite writable contiguous float64 arrays")
        if (state.step.shape != (len(state.q),) or state.step.dtype != np.int64
                or np.any(state.step < 0) or state.time.shape != (len(state.q),)
                or state.time.dtype != np.float64 or not np.isfinite(state.time).all()
                or not state.step.flags.writeable or not state.time.flags.writeable
                or not state.step.flags.c_contiguous or not state.time.flags.c_contiguous):
            raise ValueError("invalid batched clocks")
        arrays += (state.step, state.time)
        if any(np.shares_memory(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("batched state arrays and clocks must not alias")

    def _evaluate(self, q):
        flat = q.reshape(-1,len(self.masses),3)
        self.pes_calls += 1
        self.pes_frames += len(flat)
        return validate_diabatic_result(self.potential.evaluate(flat), flat.shape, self.nstates)

    def _mapping_step(self, state):
        result = self._evaluate(state.q)
        count = len(state.q)
        z, delta = electronic_mapping_flow(
            state.mapping_q+1j*state.mapping_p,
            result.matrix.reshape(count,self.nbeads,self.nstates,self.nstates),
            result.gradients.reshape(count,self.nbeads,self.nstates,self.nstates,len(self.masses),3),
            .5*self.dt, backend=self.electronic_backend, device=self.electronic_device,
            threads=self.threads)
        state.p += delta
        state.mapping_q[:], state.mapping_p[:] = z.real, z.imag

    def _free_step(self, state):
        count, beads, atoms, _ = state.q.shape
        if self._plan_count != count:
            self._plan = CayleyIntegrator(np.tile(self.masses,count),self.beta,self.nbeads,self.dt,
                                         native=self._native_requested,threads=self.threads,
                                         scheme=self.integrator_name)
            self._plan_count = count
        q = np.ascontiguousarray(state.q.transpose(1,0,2,3)).reshape(beads,count*atoms,3)
        p = np.ascontiguousarray(state.p.transpose(1,0,2,3)).reshape(beads,count*atoms,3)
        self._plan.free_step(q,p)
        state.q[:] = q.reshape(beads,count,atoms,3).transpose(1,0,2,3)
        state.p[:] = p.reshape(beads,count,atoms,3).transpose(1,0,2,3)

    def step(self, state):
        self._check(state)
        trial = state.copy()
        self._mapping_step(trial)
        self._free_step(trial)
        self._mapping_step(trial)
        if np.any(trial.step == np.iinfo(np.int64).max):
            raise OverflowError("batched step clock overflow")
        trial.step += 1
        trial.time += self.dt
        self._check(trial)
        for name in ("q","p","mapping_q","mapping_p","step","time"):
            getattr(state,name)[:] = getattr(trial,name)
        return state

    def energy(self, state):
        self._check(state)
        v = self._evaluate(state.q).matrix.reshape(len(state.q),self.nbeads,self.nstates,self.nstates)
        z = state.mapping_q+1j*state.mapping_p
        electronic = .5*(np.einsum("kbi,kbij,kbj->k",z.conj(),v,z)
                          -np.trace(v,axis1=-2,axis2=-1).sum(axis=1)).real
        kinetic = np.sum(state.p**2/(2*self.masses[None,None,:,None]),axis=(1,2,3))
        dq = state.q-np.roll(state.q,-1,axis=1)
        spring = .5*(self.nbeads/self.beta)**2*np.sum(self.masses[None,None,:,None]*dq**2,axis=(1,2,3))
        return electronic+kinetic+spring

    def populations(self, state):
        self._check(state)
        return .5*np.mean(state.mapping_q**2+state.mapping_p**2-1.,axis=1)

    def density_matrix(self, state):
        self._check(state)
        z = state.mapping_q+1j*state.mapping_p
        return np.einsum("kbi,kbj->kij",z,z.conj())/(2*self.nbeads)-.5*np.eye(self.nstates)
