"""Centroid-approximation ring-polymer fewest-switches surface hopping.

All beads occupy one adiabatic surface. Electronic amplitudes evolve at the
centroid; they are stored in a fixed diabatic basis to avoid eigenvector-phase
bookkeeping. Nuclear masses, gradients, time and energies are in atomic units.
This experimental API supports real diabatic matrices with nondegenerate
adiabatic states. Complex-SOC surface hopping requires additional physics and
is deliberately rejected; complex matrices are supported by NRPMD instead.
"""
from dataclasses import dataclass
import copy

import numpy as np

from .dynamics import CayleyIntegrator, State
from .nonadiabatic import _positive_integer, validate_diabatic_result
from .potentials.base import PotentialError


def _surface(value, nstates):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < nstates:
        raise ValueError("active_surface must be an integer in [0,nstates)")
    return int(value)


@dataclass
class RPSHState(State):
    """Nuclear ring plus normalized amplitudes in the fixed diabatic basis."""
    coefficients: np.ndarray | None = None
    active_surface: int = 0

    def __post_init__(self):
        super().__post_init__()
        raw = np.asarray(self.coefficients)
        if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "fciu":
            raise ValueError("coefficients must be a numerical vector [S]")
        if not np.isfinite(raw).all() or not np.isclose(np.vdot(raw, raw).real, 1., rtol=0., atol=1e-10):
            raise ValueError("coefficients must be finite and normalized")
        self.coefficients = np.array(raw, dtype=complex, order="C", copy=True)
        self.active_surface = _surface(self.active_surface, raw.size)

    def copy(self):
        return RPSHState(self.q, self.p, self.step, self.time, self.coefficients, self.active_surface)

    def to_dict(self):
        """Detached JSON-compatible state; save the dynamics RNG state as well."""
        return {"q": self.q.tolist(), "p": self.p.tolist(), "step": int(self.step),
                "time": float(self.time), "coefficients_real": self.coefficients.real.tolist(),
                "coefficients_imag": self.coefficients.imag.tolist(),
                "active_surface": int(self.active_surface)}

    @classmethod
    def from_dict(cls, value):
        return cls(value["q"], value["p"], value["step"], value["time"],
                   np.asarray(value["coefficients_real"])+1j*np.asarray(value["coefficients_imag"]),
                   value["active_surface"])


@dataclass(frozen=True)
class HopResult:
    source: int
    target: int
    accepted: bool
    frustrated: bool
    rescaling: str
    ring_energy_change: float


class RPSHDynamics:
    """Centroid RPSH with beadwise adiabatic forces and explicit hop policy.

    ``rescaling='ring'`` conserves the full ring Hamiltonian at a hop by using
    the mean bead gap; the default ``'centroid'`` conserves centroid kinetic energy
    plus the centroid electronic energy. Both shift every bead momentum by the
    same vector parallel to the centroid derivative coupling, leaving internal
    momenta unchanged. Frustrated hops either keep or reverse that component.

    The propagation is an experimental real-symmetric, nonperiodic Python API;
    it adds no decoherence or detailed-balance correction and does not prepare a
    thermal electronic ensemble. Nuclear and electronic timesteps require
    convergence. The outgoing fewest-switches probabilities are never silently
    renormalized when their sum exceeds one: such a step fails transactionally.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates, *, seed=0,
                 native=True, threads=1, integrator="cayley-bcb", electronic_substeps=8,
                 rescaling="centroid", frustrated_hops="keep", gap_tolerance=1e-10):
        self.potential = potential
        self.nstates = _positive_integer(nstates, "nstates")
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt, threads=threads,
                                          native=native, scheme=integrator)
        self.dt, self.beta, self.nbeads = float(dt), float(beta), int(nbeads)
        if self.dt <= 0:
            raise ValueError("RPSH dt must be positive")
        self.masses = self.integrator.masses
        self.integrator_name, self.native = self.integrator.scheme, self.integrator.native
        self.electronic_substeps = _positive_integer(electronic_substeps, "electronic_substeps")
        if rescaling not in ("ring", "centroid"):
            raise ValueError("rescaling must be ring or centroid")
        if frustrated_hops not in ("keep", "reverse"):
            raise ValueError("frustrated_hops must be keep or reverse")
        self.rescaling, self.frustrated_hops = rescaling, frustrated_hops
        if isinstance(gap_tolerance, (bool, np.bool_)) or not np.isfinite(gap_tolerance) or gap_tolerance <= 0:
            raise ValueError("gap_tolerance must be positive and finite")
        self.gap_tolerance = float(gap_tolerance)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.last_hop = None
        self.last_probabilities = tuple(0. for _ in range(self.nstates))
        self._parameters = (self.dt, self.beta, self.nbeads, self.nstates, self.integrator_name,
                            self.electronic_substeps, self.rescaling, self.frustrated_hops, self.gap_tolerance)

    def _check_parameters(self):
        current = (self.dt, self.beta, self.nbeads, self.nstates, self.integrator_name,
                   self.electronic_substeps, self.rescaling, self.frustrated_hops, self.gap_tolerance)
        if (current != self._parameters or self.dt != self.integrator.dt or self.beta != self.integrator.beta
                or self.nbeads != self.integrator.nbeads or self.integrator_name != self.integrator.scheme):
            raise ValueError("RPSHDynamics parameters are immutable")

    def _check(self, state):
        self._check_parameters()
        if not isinstance(state, RPSHState):
            raise ValueError("RPSHDynamics requires RPSHState")
        self.integrator._check(state.q, state.p)
        z = state.coefficients
        if (not isinstance(z, np.ndarray) or z.shape != (self.nstates,) or z.dtype != np.complex128
                or not z.flags.c_contiguous or not z.flags.writeable or not np.isfinite(z).all()
                or not np.isclose(np.vdot(z, z).real, 1., rtol=0., atol=1e-10)):
            raise ValueError("invalid normalized RPSH coefficients")
        _surface(state.active_surface, self.nstates)
        if (isinstance(state.step, (bool, np.bool_)) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid state clock")
        if np.shares_memory(z, state.q) or np.shares_memory(z, state.p):
            raise ValueError("electronic and nuclear state arrays must not share memory")

    def _adiabatic(self, q, couplings=False):
        r = validate_diabatic_result(self.potential.evaluate(q), q.shape, self.nstates)
        if (np.iscomplexobj(r.matrix) and np.any(r.matrix.imag != 0)
                or np.iscomplexobj(r.gradients) and np.any(r.gradients.imag != 0)):
            raise PotentialError("centroid RPSH requires real symmetric matrices; use NRPMD for complex SOC")
        energies, vectors = np.linalg.eigh(r.matrix.real)
        gaps = energies[:, None, :]-energies[:, :, None]
        if self.nstates > 1 and np.any(np.abs(gaps[:, ~np.eye(self.nstates, dtype=bool)]) <= self.gap_tolerance):
            raise PotentialError("RPSH adiabatic states are degenerate or below gap_tolerance")
        transformed = np.einsum("bsi,bstac,btj->bijac", vectors, r.gradients.real, vectors, optimize=True)
        gradients = np.einsum("biiac->biac", transformed).copy()
        if couplings:
            index = np.arange(self.nstates)
            gaps[:, index, index] = 1.
            transformed /= gaps[..., None, None]
            transformed[:, index, index] = 0.
            return energies, vectors, gradients, transformed
        return energies, vectors, gradients

    def initialize(self, nuclear, active_surface=0, *, adiabatic_coefficients=None):
        """Construct an RPSH state at the nuclear centroid in the requested basis."""
        self._check_parameters()
        self.integrator._check(nuclear.q, nuclear.p)
        active = _surface(active_surface, self.nstates)
        _, u, _ = self._adiabatic(nuclear.q.mean(axis=0, keepdims=True))
        c = np.eye(self.nstates, dtype=complex)[active] if adiabatic_coefficients is None else np.asarray(adiabatic_coefficients)
        if c.shape != (self.nstates,):
            raise ValueError("adiabatic_coefficients must have shape [S]")
        state = RPSHState(nuclear.q, nuclear.p, nuclear.step, nuclear.time, u[0]@c, active)
        self._check(state)
        return state

    def populations(self, state):
        """Centroid adiabatic amplitude populations; separate from active-surface counts."""
        self._check(state)
        _, u, _ = self._adiabatic(state.q.mean(axis=0, keepdims=True))
        return np.abs(u[0].T@state.coefficients)**2

    def energy(self, state):
        """Full active-surface sum-bead propagation Hamiltonian."""
        self._check(state)
        energies, _, _ = self._adiabatic(state.q)
        kinetic = np.sum(state.p**2/(2*self.masses[None, :, None]))
        dq = state.q-np.roll(state.q, -1, axis=0)
        spring = .5*(self.nbeads/self.beta)**2*np.sum(self.masses[None, :, None]*dq*dq)
        return float(energies[:, state.active_surface].sum()+kinetic+spring)

    def _electronic_step(self, coefficients, old_q, new_q, active):
        q0, q1 = old_q.mean(axis=0, keepdims=True), new_q.mean(axis=0, keepdims=True)
        velocity = (q1[0]-q0[0])/self.dt
        _, u0, _ = self._adiabatic(q0)
        population = abs((u0[0].T@coefficients)[active])**2
        # A zero-weight active surface cannot define conditional FSSH rates.
        if population <= np.finfo(float).eps:
            raise ValueError("active surface has zero electronic population; FSSH probability is undefined")
        z = coefficients.copy()
        transfer = np.zeros(self.nstates)
        h = self.dt/self.electronic_substeps
        for k in range(self.electronic_substeps):
            q = q0+(q1-q0)*((k+.5)/self.electronic_substeps)
            e, u, _, d = self._adiabatic(q, couplings=True)
            c = u[0].T@z
            phase = np.exp(-.5j*h*e[0])
            midpoint = c*phase
            vd = np.einsum("ac,ijac->ij", velocity, d[0])
            # Outgoing probability current -d|c_active|²/dt into each target.
            transfer += 2*h*np.real(midpoint[active].conj()*midpoint*vd[active])
            z = u[0]@(midpoint*phase)
        probability = np.maximum(transfer/population, 0.)
        probability[active] = 0.
        total = float(probability.sum())
        if not np.isfinite(probability).all() or total > 1.+1e-12:
            raise ValueError("FSSH probabilities exceed one; reduce the nuclear timestep")
        if total > 1.:
            probability /= total  # Roundoff only; finite timestep excess fails above.
        return z, probability

    def _hop(self, state, target, bead_energies=None):
        source = state.active_surface
        target = _surface(target, self.nstates)
        if source == target:
            return HopResult(source, target, False, False, self.rescaling, 0.)
        e, _, _, nac = self._adiabatic(state.q.mean(axis=0, keepdims=True), couplings=True)
        if bead_energies is None:
            bead_energies, _, _ = self._adiabatic(state.q)
        ring_gap = float(np.sum(bead_energies[:, target]-bead_energies[:, source]))
        gap = ring_gap/self.nbeads if self.rescaling == "ring" else float(e[0, target]-e[0, source])
        direction = nac[0, source, target].copy()
        norm = float(np.linalg.norm(direction))
        if norm == 0.:
            return HopResult(source, target, False, True, self.rescaling, 0.)
        direction /= norm
        # Eigenvector signs can flip the real NAC. Fix the otherwise ambiguous
        # downhill root at zero projected momentum in a basis-independent way.
        if direction.flat[np.argmax(np.abs(direction))] < 0.:
            direction *= -1.
        pc = state.p.mean(axis=0)
        a = .5*float(np.sum(direction**2/self.masses[:, None]))
        b = float(np.sum(pc*direction/self.masses[:, None]))
        discriminant = b*b-4*a*gap
        if discriminant < 0.:
            if self.frustrated_hops == "reverse":
                state.p -= (b/a)*direction[None]
            return HopResult(source, target, False, True, self.rescaling, 0.)
        root = np.sqrt(discriminant)
        # Rationalized smaller-magnitude root avoids catastrophic cancellation.
        denominator = b+np.copysign(root, b)
        shift = -2*gap/denominator if denominator != 0. else 0.
        if b == 0. and root != 0.:
            shift = root/(2*a)
        old_kinetic = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        state.p += shift*direction[None]
        state.active_surface = target
        new_kinetic = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        return HopResult(source, target, True, False, self.rescaling, new_kinetic-old_kinetic+ring_gap)

    def attempt_hop(self, state, target):
        """Attempt one deterministic rescaling; useful for restartable custom drivers."""
        self._check(state)
        trial = state.copy()
        result = self._hop(trial, target)
        self._check(trial)
        state.p[:] = trial.p
        state.active_surface = trial.active_surface
        self.last_hop = result
        return result

    def step(self, state):
        self._check(state)
        trial = state.copy()
        saved_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            _, _, gradient = self._adiabatic(trial.q)
            trial.p -= .5*self.dt*gradient[:, trial.active_surface]
            self.integrator.free_step(trial.q, trial.p)
            energies, _, gradient = self._adiabatic(trial.q)
            trial.p -= .5*self.dt*gradient[:, trial.active_surface]
            z, probabilities = self._electronic_step(state.coefficients, state.q, trial.q, trial.active_surface)
            trial.coefficients[:] = z
            hop = None
            if probabilities.sum() > 0.:
                target = int(np.searchsorted(np.cumsum(probabilities), self.rng.random(), side="right"))
                if target < self.nstates:
                    hop = self._hop(trial, target, energies)
            trial.step = int(trial.step) + 1
            trial.time += self.dt
            self._check(trial)
        except BaseException:
            self.rng.bit_generator.state = saved_rng
            raise
        state.q[:], state.p[:], state.coefficients[:] = trial.q, trial.p, trial.coefficients
        state.active_surface = trial.active_surface
        state.step, state.time = trial.step, trial.time
        self.last_hop = hop
        self.last_probabilities = tuple(float(x) for x in probabilities)
        return state
