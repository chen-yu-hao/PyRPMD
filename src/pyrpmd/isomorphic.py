"""Two-state isomorphic RPMD Hamiltonian and explicit iso-Ehrenfest dynamics.

The trace-matching potential follows Tao, Shushkov and Miller, JCP 148,
102327 (2018), Eqs. 13--19, doi:10.1063/1.5005544.  This module uses the
real-mass/sum-bead convention for nuclear propagation.  It supports a fixed
real diabatic basis and a coupling with one nonzero sign throughout the ring.
"""

from dataclasses import dataclass

import numpy as np

from ._matrix_path import (exponential_path, integrator_signature,
                           positive_trace_path, real_diabatic)
from .dynamics import CayleyIntegrator, State
from .potentials.base import PotentialError, coordinates


@dataclass(frozen=True)
class IsomorphicResult:
    """Mean-bead matrix [2,2], dV_iso/dq [2,2,P,N,3], and log thermal trace."""
    matrix: np.ndarray
    gradients: np.ndarray
    log_electronic_weight: float


class IsomorphicPotential:
    """A whole-ring matrix potential; never a separable scalar PES adapter."""

    def __init__(self, potential, beta, nbeads):
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if (isinstance(nbeads, bool) or not isinstance(nbeads, (int, np.integer))
                or nbeads < 1):
            raise ValueError("nbeads must be a positive integer")
        self.potential = potential
        self.beta, self.nbeads = float(beta), int(nbeads)

    def evaluate(self, q):
        q = coordinates(q)
        if len(q) != self.nbeads:
            raise ValueError("isomorphic potential requires the entire configured ring")
        result = real_diabatic(self.potential, q, 2)
        matrix, gradients = result.matrix, result.gradients
        if self.nbeads == 1:
            eig = np.linalg.eigvalsh(matrix[0])
            log_weight = float(np.logaddexp(-self.beta * eig[0], -self.beta * eig[1]))
            return IsomorphicResult(matrix[0].copy(),
                                    gradients.transpose(1, 2, 0, 3, 4).copy(),
                                    log_weight)
        diagonals = np.diagonal(matrix, axis1=1, axis2=2).mean(axis=0)
        diagonal_gradients = np.stack((gradients[:, 0, 0], gradients[:, 1, 1])) / self.nbeads
        out = np.diag(diagonals)
        derivative = np.zeros((2, 2, *q.shape))
        derivative[0, 0], derivative[1, 1] = diagonal_gradients
        coupling = matrix[:, 0, 1]
        if np.all(coupling == 0) and np.all(gradients[:, 0, 1] == 0):
            log_weight = float(np.logaddexp(-self.beta * diagonals[0],
                                            -self.beta * diagonals[1]))
            return IsomorphicResult(out, derivative, log_weight)
        centroid = real_diabatic(self.potential, q.mean(axis=0, keepdims=True), 2)
        sign = float(np.sign(centroid.matrix[0, 0, 1]))
        if sign == 0 or np.any(sign * coupling <= 0):
            raise PotentialError("isomorphic coupling must have one nonzero sign at all beads and centroid")
        # A collapsed electronic path is exactly the physical matrix, even for
        # a coupling whose square is below floating-point trace resolution.
        # Cyclicity of the trace gives dV_iso/dq_b = (dV_b/dq_b)/P here.
        if np.array_equal(matrix, np.broadcast_to(matrix[0], matrix.shape)):
            energies = np.linalg.eigvalsh(matrix[0])
            log_weight = float(np.logaddexp(-self.beta * energies[0],
                                            -self.beta * energies[1]))
            return IsomorphicResult(matrix[0].copy(),
                                    gradients.transpose(1, 2, 0, 3, 4).copy() / self.nbeads,
                                    log_weight)
        path = exponential_path(result, self.beta / self.nbeads)
        log_centered_trace, dlog_centered_trace = positive_trace_path(path)
        log_chi = log_centered_trace - np.log(2.0)
        if log_chi < -2e-13:
            raise PotentialError("isomorphic acosh argument is outside its real domain")
        log_chi = max(0.0, log_chi)
        root = np.sqrt(-np.expm1(-2 * log_chi))
        acosh = log_chi + np.log1p(root)
        radius = acosh / self.beta
        difference = 0.5 * (diagonals[0] - diagonals[1])
        square = (radius - abs(difference)) * (radius + abs(difference))
        prefactor = (2 / self.beta**2 * (1 + 2 * log_chi / 3)
                     if log_chi < 1e-8 else 2 * acosh / (self.beta**2 * root))
        # A small positive subtraction residual can be roundoff too.  Bound
        # accumulated trace/log and final subtraction errors before taking its
        # square root; never turn cancellation noise into a physical coupling.
        resolution = (64 * np.finfo(float).eps * (self.nbeads + 1)
                      * (radius**2 + difference**2 + prefactor*(1 + abs(log_chi))))
        if not np.isfinite(square) or square <= resolution:
            raise PotentialError("isomorphic coupling is numerically unresolved; no force regularization is applied")
        iso_coupling = sign * np.sqrt(square)
        ddifference = 0.5 * (diagonal_gradients[0] - diagonal_gradients[1])
        dcoupling = (prefactor * dlog_centered_trace
                     - 2 * difference * ddifference) / (2 * iso_coupling)
        out[0, 1] = out[1, 0] = iso_coupling
        derivative[0, 1] = derivative[1, 0] = dcoupling
        log_weight = log_centered_trace - self.beta * float(path.common.mean())
        if not np.isfinite(derivative).all() or not np.isfinite(out).all():
            raise PotentialError("isomorphic matrix/gradient overflow")
        return IsomorphicResult(out, derivative, float(log_weight))


@dataclass
class IsomorphicEhrenfestState(State):
    amplitudes: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        value = np.asarray(self.amplitudes)
        if value.shape != (2,) or value.dtype.kind not in "cfiu":
            raise ValueError("amplitudes must contain two complex numbers")
        self.amplitudes = np.array(value, dtype=complex, copy=True)
        if (not np.isfinite(self.amplitudes).all()
                or not np.isclose(np.vdot(value, value).real, 1.0, rtol=0, atol=1e-10)):
            raise ValueError("amplitudes must be finite and normalized to one")

    def copy(self):
        return IsomorphicEhrenfestState(self.q, self.p, self.step, self.time, self.amplitudes)

    def populations(self):
        return np.abs(self.amplitudes)**2


class IsomorphicEhrenfestDynamics:
    """Ehrenfest electronic dynamics on the two-state isomorphic Hamiltonian.

Nuclear energy/force contains P*V_iso; the electronic rotation uses V_iso,
not P*V_iso.  This is an explicitly chosen mixed quantum-classical method,
not a claim that Ehrenfest propagation preserves the quantum thermal trace.
"""

    def __init__(self, potential, masses, beta, nbeads, dt, *, native=True,
                 integrator="cayley-bcb", threads=1):
        self.potential = IsomorphicPotential(potential, beta, nbeads)
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt, native=native,
                                          scheme=integrator, threads=threads)
        self._integrator_signature = integrator_signature(self.integrator)

    def _check(self, state):
        if not isinstance(state, IsomorphicEhrenfestState):
            raise ValueError("iso-Ehrenfest dynamics requires IsomorphicEhrenfestState")
        self.integrator._check(state.q, state.p)
        if (state.amplitudes.shape != (2,) or state.amplitudes.dtype != np.complex128
                or not state.amplitudes.flags.writeable
                or not np.isfinite(state.amplitudes).all()
                or not np.isclose(np.vdot(state.amplitudes, state.amplitudes).real,
                                  1, rtol=0, atol=1e-9)):
            raise ValueError("invalid electronic amplitudes")
        if (isinstance(state.step, bool) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid state clock")
        if any(np.shares_memory(state.amplitudes, value) for value in (state.q, state.p)):
            raise ValueError("state arrays must not share memory")
        if (integrator_signature(self.integrator) != self._integrator_signature
                or self.potential.beta != self.integrator.beta
                or self.potential.nbeads != self.integrator.nbeads):
            raise ValueError("isomorphic dynamics parameters are immutable")

    def _potential_step(self, state, duration):
        result = self.potential.evaluate(state.q)
        energies, vectors = np.linalg.eigh(result.matrix)
        ce = vectors.T @ state.amplitudes
        phase = 0.5 * duration * (energies[:, None] - energies[None, :])
        integrated = (duration * np.sinc(phase / np.pi) * np.exp(1j * phase)
                      * ce.conj()[:, None] * ce[None, :])
        gradient = np.einsum("si,stbac,tj->ijbac", vectors, result.gradients,
                             vectors, optimize=True)
        state.p -= self.integrator.nbeads * np.einsum("ij,ijbac->bac", integrated,
                                                     gradient, optimize=True).real
        state.amplitudes[:] = vectors @ (ce * np.exp(-1j * duration * energies))

    def step(self, state):
        self._check(state)
        trial = state.copy()
        self._potential_step(trial, self.integrator.dt / 2)
        self.integrator.free_step(trial.q, trial.p)
        self._potential_step(trial, self.integrator.dt / 2)
        trial.step += 1
        trial.time += self.integrator.dt
        self._check(trial)
        state.q[:], state.p[:] = trial.q, trial.p
        state.amplitudes[:] = trial.amplitudes
        state.step, state.time = trial.step, trial.time
        return state

    def energy(self, state):
        """Sum-bead conserved model Hamiltonian, in hartree."""
        self._check(state)
        matrix = self.potential.evaluate(state.q).matrix
        kinetic = np.sum(state.p**2 / (2 * self.integrator.masses[None, :, None]))
        difference = state.q - np.roll(state.q, 1, axis=0)
        spring = 0.5 * (self.integrator.nbeads / self.integrator.beta)**2 * np.sum(
            self.integrator.masses[None, :, None] * difference**2)
        electronic = self.integrator.nbeads * np.vdot(state.amplitudes,
                                                     matrix @ state.amplitudes).real
        return float(kinetic + spring + electronic)

