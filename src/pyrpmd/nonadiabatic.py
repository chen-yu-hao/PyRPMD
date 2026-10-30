"""Experimental MMST NRPMD for state-selected diabatic scattering.

Atomic units (hbar=1), real nuclear masses and one mapping vector per bead.
The electronic estimate is signed; it is not a surface-hopping probability.
This module does not prepare an electronic thermal trace or a QTST rate.
"""
from dataclasses import dataclass
import numpy as np

from .dynamics import CayleyIntegrator, State
from .potentials.base import Potential, PotentialError, PESResult, coordinates


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass
class DiabaticResult:
    """V[B,S,S] in hartree and positive dV/dq[B,S,S,N,3]."""
    matrix: np.ndarray
    gradients: np.ndarray
    status: np.ndarray | None = None
    messages: tuple[str, ...] | None = None


def validate_diabatic_result(result, shape, nstates=None):
    """Reject complex, asymmetric, failed or malformed matrix evaluations."""
    if isinstance(result, (tuple, list)) and 2 <= len(result) <= 4:
        result = DiabaticResult(*result)
    if not isinstance(result, DiabaticResult):
        if not all(hasattr(result, key) for key in ("matrix", "gradients")):
            raise PotentialError("diabatic PES must return matrix and gradients")
        result = DiabaticResult(result.matrix, result.gradients,
                                getattr(result, "status", None), getattr(result, "messages", None))
    matrix, gradients = np.asarray(result.matrix), np.asarray(result.gradients)
    if matrix.dtype.kind not in "fiu" or gradients.dtype.kind not in "fiu":
        raise PotentialError("diabatic matrix and gradients must be real numerical arrays")
    if len(shape) != 3 or shape[-1] != 3 or min(shape) < 1:
        raise PotentialError("expected nonempty coordinates [B,N,3]")
    if matrix.ndim != 3 or matrix.shape[0] != shape[0] or matrix.shape[1] < 1 or matrix.shape[1] != matrix.shape[2]:
        raise PotentialError("diabatic matrix shape must be [B,S,S]")
    if nstates is not None and matrix.shape[1] != _positive_integer(nstates, "nstates"):
        raise PotentialError("diabatic matrix nstates mismatch")
    if gradients.shape != (*matrix.shape, *shape[1:]):
        raise PotentialError("diabatic gradients shape must be [B,S,S,N,3]")
    status = np.zeros(shape[0], dtype=int) if result.status is None else np.asarray(result.status)
    if status.shape != (shape[0],) or status.dtype.kind not in "iu" or np.any(status < 0):
        raise PotentialError("diabatic status must be nonnegative integers [B]")
    messages = ("",) * shape[0] if result.messages is None else tuple(result.messages)
    if len(messages) != shape[0]:
        raise PotentialError("diabatic messages shape mismatch")
    if np.any(status):
        raise PotentialError(f"diabatic PES frames failed: {np.flatnonzero(status).tolist()}")
    if not np.isfinite(matrix).all() or not np.isfinite(gradients).all():
        raise PotentialError("diabatic PES returned nonfinite values")
    # A fixed absolute tolerance prevents a large common energy offset from
    # masking a physically meaningful antisymmetric coupling.
    if not np.allclose(matrix, matrix.swapaxes(1, 2), rtol=0, atol=1e-12):
        raise PotentialError("diabatic matrix must be symmetric")
    if not np.allclose(gradients, gradients.swapaxes(1, 2), rtol=0, atol=1e-12):
        raise PotentialError("diabatic gradients must be symmetric")
    return DiabaticResult(np.ascontiguousarray(0.5*(matrix+matrix.swapaxes(1, 2)), dtype=float),
                         np.ascontiguousarray(0.5*(gradients+gradients.swapaxes(1, 2)), dtype=float),
                         status.copy(), messages)


class DiagonalPotential(Potential):
    """Prepare nuclear reactants on one selected diabatic diagonal surface."""
    def __init__(self, matrix_potential, state, nstates):
        self.matrix_potential = matrix_potential
        self.nstates = _positive_integer(nstates, "nstates")
        if isinstance(state, bool) or not isinstance(state, (int, np.integer)) or not 0 <= state < nstates:
            raise ValueError("initial_state must be an integer in [0,nstates)")
        self.state = int(state)
        self.metadata = {**getattr(matrix_potential, "metadata", {}),
                         "execution_device": "cpu", "nonadiabatic_diagonal_state": self.state}

    def evaluate(self, q):
        q = coordinates(q)
        result = validate_diabatic_result(self.matrix_potential.evaluate(q), q.shape, self.nstates)
        return PESResult(result.matrix[:, self.state, self.state],
                         -result.gradients[:, self.state, self.state])

    def close(self):
        # Borrowed provider: the workflow owns and closes the matrix PES.
        pass


@dataclass
class NRPMDState(State):
    mapping_q: np.ndarray | None = None
    mapping_p: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        for key in ("mapping_q", "mapping_p"):
            value = np.asarray(getattr(self, key))
            if value.dtype.kind not in "fiu" or value.ndim != 2 or value.shape[0] != len(self.q) or value.shape[1] < 1:
                raise ValueError("mapping_q/mapping_p must be real [P,S] arrays")
            if not np.isfinite(value).all():
                raise ValueError("mapping variables must be finite")
            setattr(self, key, np.array(value, dtype=float, order="C", copy=True))
        if self.mapping_q.shape != self.mapping_p.shape:
            raise ValueError("mapping_q/mapping_p shapes must agree")

    @classmethod
    def from_nuclear(cls, state, nstates, initial_state, rng):
        nstates = _positive_integer(nstates, "nstates")
        if isinstance(initial_state, bool) or not isinstance(initial_state, (int, np.integer)) or not 0 <= initial_state < nstates:
            raise ValueError("initial_state must be an integer in [0,nstates)")
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        phase = rng.uniform(0, 2*np.pi, (len(state.q), nstates))
        radius = np.ones(nstates)
        radius[initial_state] = np.sqrt(3.0)
        return cls(state.q, state.p, state.step, state.time,
                   radius*np.cos(phase), radius*np.sin(phase))

    def copy(self):
        return NRPMDState(self.q, self.p, self.step, self.time, self.mapping_q, self.mapping_p)

    def populations(self):
        """Bead-averaged MMST estimator; negative values are retained."""
        return np.mean(0.5*(self.mapping_q**2 + self.mapping_p**2 - 1), axis=0)


def _mint(state, result, duration):
    """Exact fixed-q flow, including integrated nuclear momentum impulse."""
    energies, vectors = np.linalg.eigh(result.matrix)
    z = state.mapping_q + 1j*state.mapping_p
    ze = np.einsum("bsi,bs->bi", vectors, z)
    gaps = energies[:, :, None] - energies[:, None, :]
    phase = 0.5 * duration * gaps
    integral = duration * np.sinc(phase/np.pi) * np.exp(1j*phase)
    density_integral = np.real(ze.conj()[:, :, None] * ze[:, None, :] * integral)
    # Contract in the original diabatic basis; no eigenvector derivatives.
    density = np.einsum("bsi,bij,btj->bst", vectors, density_integral, vectors)
    impulse = 0.5*np.einsum("bij,bijac->bac", density, result.gradients)
    impulse -= 0.5*duration*np.einsum("biiac->bac", result.gradients)
    state.p -= impulse
    rotated = np.einsum("bsi,bi->bs", vectors, ze*np.exp(-1j*duration*energies))
    state.mapping_q[:] = rotated.real
    state.mapping_p[:] = rotated.imag


class NRPMDDynamics:
    """Reversible MInt/free-ring/MInt splitting of the MMST Hamiltonian."""
    def __init__(self, potential, masses, beta, nbeads, dt, nstates, threads=1,
                 native=True, integrator="cayley-bcb"):
        self.nstates = _positive_integer(nstates, "nstates")
        self.potential = potential
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt, threads=threads,
                                          native=native, scheme=integrator)
        self.masses = self.integrator.masses
        self.beta, self.dt, self.nbeads = float(beta), float(dt), int(nbeads)
        self.integrator_name = self.integrator.scheme
        self.native = self.integrator.native

    def _check(self, state):
        if (self.dt != self.integrator.dt or self.beta != self.integrator.beta
                or self.nbeads != self.integrator.nbeads or self.integrator_name != self.integrator.scheme):
            raise ValueError("NRPMDDynamics parameters are immutable")
        if not isinstance(state, NRPMDState):
            raise ValueError("NRPMDDynamics requires NRPMDState")
        self.integrator._check(state.q, state.p)
        if (isinstance(state.step, bool) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid state clock")
        for value in (state.mapping_q, state.mapping_p):
            if value.shape != (self.nbeads, self.nstates) or value.dtype != np.float64 or not value.flags.writeable or not value.flags.c_contiguous or not np.isfinite(value).all():
                raise ValueError("invalid mapping variables")
        arrays = (state.q, state.p, state.mapping_q, state.mapping_p)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")

    def _evaluate(self, q):
        return validate_diabatic_result(self.potential.evaluate(q), q.shape, self.nstates)

    def step(self, state):
        self._check(state)
        trial = state.copy()
        _mint(trial, self._evaluate(trial.q), 0.5*self.dt)
        self.integrator.free_step(trial.q, trial.p)
        _mint(trial, self._evaluate(trial.q), 0.5*self.dt)
        self._check(trial)
        for key in ("q", "p", "mapping_q", "mapping_p"):
            getattr(state, key)[:] = getattr(trial, key)
        state.step += 1
        state.time += self.dt
        return state

    def energy(self, state):
        """Sum-bead propagation Hamiltonian, not a quantum thermal estimator."""
        self._check(state)
        result = self._evaluate(state.q)
        x, y = state.mapping_q, state.mapping_p
        electronic = 0.5*(np.einsum("bi,bij,bj->", x, result.matrix, x)
                          + np.einsum("bi,bij,bj->", y, result.matrix, y)
                          - np.trace(result.matrix, axis1=1, axis2=2).sum())
        kinetic = np.sum(state.p**2/(2*self.masses[None, :, None]))
        dq = state.q-np.roll(state.q, -1, axis=0)
        spring = 0.5*(self.nbeads/self.beta)**2*np.sum(self.masses[None, :, None]*dq**2)
        return float(electronic+kinetic+spring)

    def populations(self, state):
        self._check(state)
        return state.populations()

    def asymptotic_coupling(self, state):
        self._check(state)
        matrix = self._evaluate(state.q).matrix.copy()
        index = np.arange(self.nstates)
        matrix[:, index, index] = 0
        return float(np.max(np.abs(matrix)))
