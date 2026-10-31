"""Mapping-variable RPMD with the projected Wigner matrix-path Hamiltonian.

This is the MV-RPMD of Ananth, JCP 139, 124102 (2013),
https://arxiv.org/abs/1309.4476, Eqs. 25--35.  Exact short-time electronic
matrix exponentials replace the paper's first-order electronic approximation.
Signed weights are retained; this is not the NRPMD Hamiltonian.
"""

import copy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from ._matrix_path import (ScaledProduct, exponential_path, integrator_signature,
                           real_diabatic)
from .dynamics import CayleyIntegrator
from .nonadiabatic import NRPMDState, _positive_integer
from .potentials.base import PotentialError


class MappingNodeError(PotentialError):
    """The projected electronic trace is too close to its singular nodal set."""


class MVRPMDState(NRPMDState):
    """Nuclear and Cartesian mapping coordinates for the MV Hamiltonian."""

    @classmethod
    def from_gaussian(cls, nuclear, nstates, rng=None):
        """Gaussian proposal state, not an equilibrated MV-RPMD sample."""
        nstates = _positive_integer(nstates, "nstates")
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        shape = (len(nuclear.q), nstates)
        return cls(nuclear.q, nuclear.p, nuclear.step, nuclear.time,
                   rng.normal(size=shape) / np.sqrt(2),
                   rng.normal(size=shape) / np.sqrt(2))

    def copy(self):
        return MVRPMDState(self.q, self.p, self.step, self.time,
                          self.mapping_q, self.mapping_p)

    def populations(self):
        raise ValueError("MV-RPMD populations depend on the matrix path; use dynamics.populations(state)")


@dataclass(frozen=True)
class MVRPMDResult:
    potential_energy: float
    nuclear_forces: np.ndarray
    mapping_q_gradient: np.ndarray
    mapping_p_gradient: np.ndarray
    sign: int
    log_absolute_theta: float
    populations: np.ndarray


class MVRPMDPotential:
    """Nonseparable electronic Hamiltonian on an entire ring and its mappings."""

    def __init__(self, potential, beta, nbeads, nstates, *, node_tolerance=1e-12):
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        if not np.isfinite(node_tolerance) or not 0 < node_tolerance < 1:
            raise ValueError("node_tolerance must lie strictly between zero and one")
        self.potential, self.beta = potential, float(beta)
        self.nbeads = _positive_integer(nbeads, "nbeads")
        self.nstates = _positive_integer(nstates, "nstates")
        self.node_tolerance = float(node_tolerance)

    def _prepare(self, q):
        if len(q) != self.nbeads:
            raise ValueError("MV-RPMD requires the entire configured ring")
        return exponential_path(real_diabatic(self.potential, q, self.nstates),
                                self.beta / self.nbeads)

    def _from_path(self, path, x, y):
        x, y = np.asarray(x), np.asarray(y)
        expected = (self.nbeads, self.nstates)
        if (x.shape != expected or y.shape != expected
                or x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu"
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("MV mapping variables must be finite real [P,S] arrays")
        z = x + 1j * y
        c = z[:, :, None] * z.conj()[:, None, :] - 0.5 * np.eye(self.nstates)
        factors = c @ path.matrices
        try:
            product = ScaledProduct(factors)
        except PotentialError as exc:
            if np.isfinite(factors).all():
                raise MappingNodeError("MV-RPMD matrix path is at or numerically indistinguishable from a node") from exc
            raise
        theta = float(np.trace(product.matrix).real)
        if abs(theta) <= self.node_tolerance:
            raise MappingNodeError("MV-RPMD projected trace is at or too close to a node")
        holes = product.holes()
        dlog_nuclear = np.einsum("bijac,bji->bac", path.gradients,
                                 holes @ c, optimize=True).real / theta
        bmatrix = path.matrices @ holes
        left = np.einsum("bi,bij->bj", z.conj(), bmatrix)
        right = np.einsum("bij,bj->bi", bmatrix, z)
        dlog_x = (left + right).real / theta
        dlog_y = (1j * (left - right)).real / theta
        coefficient = self.nbeads / self.beta
        centered_logtheta = (np.log(abs(theta)) + product.log_scale
                             + float(path.log_scales.sum()))
        logtheta = centered_logtheta - self.beta / self.nbeads * float(path.common.sum())
        energy = float(path.common.sum() + coefficient * (
            np.sum(x*x + y*y) - centered_logtheta))
        forces = -path.common_gradients + coefficient * dlog_nuclear
        gx, gy = coefficient * (2*x - dlog_x), coefficient * (2*y - dlog_y)
        populations = np.diagonal(factors @ holes, axis1=1, axis2=2).real.mean(axis=0) / theta
        if not (np.isfinite(energy) and np.isfinite(logtheta)
                and np.isfinite(forces).all() and np.isfinite(gx).all()
                and np.isfinite(gy).all() and np.isfinite(populations).all()):
            raise PotentialError("MV-RPMD Hamiltonian or force overflow")
        return MVRPMDResult(energy, forces, gx, gy, 1 if theta > 0 else -1,
                           float(logtheta), populations)

    def evaluate(self, state):
        if not isinstance(state, MVRPMDState):
            raise ValueError("MV-RPMD potential requires MVRPMDState")
        return self._from_path(self._prepare(state.q), state.mapping_q, state.mapping_p)


class MVRPMDDynamics:
    """Symmetric free-ring/implicit-midpoint splitting of the MV Hamiltonian.

The implicit solve is necessary because the electronic Hamiltonian couples
mapping positions and momenta.  Failed solves and detected sign changes leave
the input state untouched.  Work scales linearly in beads per force evaluation;
this reference solver is intended for modest model systems.
"""

    def __init__(self, potential, masses, beta, nbeads, dt, nstates, *,
                 native=True, integrator="cayley-bcb", threads=1,
                 node_tolerance=1e-12, solve_tolerance=1e-10,
                 max_evaluations=1000):
        self.potential = MVRPMDPotential(potential, beta, nbeads, nstates,
                                        node_tolerance=node_tolerance)
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt,
                                          native=native, scheme=integrator, threads=threads)
        self._integrator_signature = integrator_signature(self.integrator)
        if not np.isfinite(solve_tolerance) or not 0 < solve_tolerance < 1e-3:
            raise ValueError("solve_tolerance must lie between zero and 1e-3")
        self.solve_tolerance = float(solve_tolerance)
        self.max_evaluations = _positive_integer(max_evaluations, "max_evaluations")

    def _check(self, state):
        if not isinstance(state, MVRPMDState):
            raise ValueError("MV-RPMD dynamics requires MVRPMDState")
        self.integrator._check(state.q, state.p)
        expected = (self.integrator.nbeads, self.potential.nstates)
        for value in (state.mapping_q, state.mapping_p):
            if (value.shape != expected or value.dtype != np.float64
                    or not value.flags.writeable or not value.flags.c_contiguous
                    or not np.isfinite(value).all()):
                raise ValueError("invalid MV mapping variables")
        arrays = (state.q, state.p, state.mapping_q, state.mapping_p)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")
        if (isinstance(state.step, bool) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid state clock")
        if (integrator_signature(self.integrator) != self._integrator_signature
                or self.integrator.beta != self.potential.beta
                or self.integrator.nbeads != self.potential.nbeads):
            raise ValueError("MV-RPMD dynamics parameters are immutable")

    def _potential_step(self, state, duration):
        path = self.potential._prepare(state.q)
        initial = self.potential._from_path(path, state.mapping_q, state.mapping_p)
        shape = state.mapping_q.shape
        size = state.mapping_q.size
        origin = np.concatenate((state.mapping_q.ravel(), state.mapping_p.ravel()))

        def residual(value):
            midpoint = 0.5 * (origin + value)
            result = self.potential._from_path(path, midpoint[:size].reshape(shape),
                                               midpoint[size:].reshape(shape))
            if result.sign != initial.sign:
                raise MappingNodeError("MV implicit trial crossed a projected-trace node; reduce dt")
            flow = np.concatenate((result.mapping_p_gradient.ravel(),
                                   -result.mapping_q_gradient.ravel()))
            return value - origin - duration * flow

        solution = root(residual, origin, method="hybr",
                        options={"xtol": self.solve_tolerance,
                                 "maxfev": self.max_evaluations})
        error = np.max(np.abs(residual(solution.x)))
        if (not solution.success or not np.isfinite(solution.x).all()
                or error > self.solve_tolerance * (1 + np.max(np.abs(origin)))):
            raise PotentialError("MV implicit midpoint failed to converge; reduce dt")
        endpoint = self.potential._from_path(path, solution.x[:size].reshape(shape),
                                             solution.x[size:].reshape(shape))
        if endpoint.sign != initial.sign:
            raise MappingNodeError("MV step crossed a projected-trace node; reduce dt")
        midpoint = 0.5 * (origin + solution.x)
        middle = self.potential._from_path(path, midpoint[:size].reshape(shape),
                                          midpoint[size:].reshape(shape))
        state.p += duration * middle.nuclear_forces
        state.mapping_q[:] = solution.x[:size].reshape(shape)
        state.mapping_p[:] = solution.x[size:].reshape(shape)

    def step(self, state):
        self._check(state)
        original_sign = self.potential.evaluate(state).sign
        trial = state.copy()
        self._potential_step(trial, self.integrator.dt / 2)
        self.integrator.free_step(trial.q, trial.p)
        if self.potential.evaluate(trial).sign != original_sign:
            raise MappingNodeError("MV nuclear substep crossed a projected-trace node; reduce dt")
        self._potential_step(trial, self.integrator.dt / 2)
        trial.step += 1
        trial.time += self.integrator.dt
        self._check(trial)
        for key in ("q", "p", "mapping_q", "mapping_p"):
            getattr(state, key)[:] = getattr(trial, key)
        state.step, state.time = trial.step, trial.time
        return state

    def _spring_energy(self, state):
        difference = state.q - np.roll(state.q, 1, axis=0)
        return 0.5 * (self.integrator.nbeads / self.integrator.beta)**2 * np.sum(
            self.integrator.masses[None, :, None] * difference**2)

    def energy(self, state):
        self._check(state)
        kinetic = np.sum(state.p**2 / (2 * self.integrator.masses[None, :, None]))
        return float(kinetic + self._spring_energy(state)
                     + self.potential.evaluate(state).potential_energy)

    def populations(self, state):
        self._check(state)
        return self.potential.evaluate(state).populations.copy()

    def log_sampling_density(self, state):
        """Coordinate/mapping |weight|; nuclear Maxwell momenta are separate."""
        self._check(state)
        return -self.integrator.beta / self.integrator.nbeads * (
            self._spring_energy(state) + self.potential.evaluate(state).potential_energy)


class MVRPMDSampler:
    """Symmetric Metropolis sampling of the absolute projected Wigner weight."""

    def __init__(self, dynamics, *, position_step, mapping_step, seed=0):
        if not isinstance(dynamics, MVRPMDDynamics):
            raise ValueError("dynamics must be MVRPMDDynamics")
        for value in (position_step, mapping_step):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("proposal steps must be finite and positive")
        self.dynamics = dynamics
        self.position_step, self.mapping_step = float(position_step), float(mapping_step)
        self.rng = np.random.default_rng(seed)
        self.attempted = self.accepted = 0

    def step(self, state):
        self.dynamics._check(state)
        saved_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            old = self.dynamics.log_sampling_density(state)
            trial = state.copy()
            trial.q += self.rng.normal(scale=self.position_step, size=state.q.shape)
            trial.mapping_q += self.rng.normal(scale=self.mapping_step, size=state.mapping_q.shape)
            trial.mapping_p += self.rng.normal(scale=self.mapping_step, size=state.mapping_p.shape)
            try:
                new = self.dynamics.log_sampling_density(trial)
                accepted = np.log(self.rng.random()) < min(0.0, new - old)
            except MappingNodeError:
                accepted = False
        except BaseException:
            self.rng.bit_generator.state = saved_rng
            raise
        self.attempted += 1
        if accepted:
            for key in ("q", "mapping_q", "mapping_p"):
                getattr(state, key)[:] = getattr(trial, key)
            self.accepted += 1
        return bool(accepted)

    def draw_momenta(self, state):
        self.dynamics._check(state)
        integrator = self.dynamics.integrator
        saved_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            scale = np.sqrt(integrator.masses[None, :, None]) * np.sqrt(integrator.nbeads / integrator.beta)
            momentum = self.rng.normal(size=state.p.shape) * scale
            if not np.isfinite(momentum).all():
                raise ValueError("sampled nuclear momenta overflow")
        except BaseException:
            self.rng.bit_generator.state = saved_rng
            raise
        state.p[:] = momentum
        return state

