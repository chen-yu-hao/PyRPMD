"""Original two-state MASH and the 2023 multistate MASH variant.

The active surface is the larger electronic population; there is no stochastic
hop decision. Initial spin sampling, NAC momentum rescaling and frustrated-hop
reflection follow Mannouch and Richardson, JCP 158, 104111 (2023).

This is a fixed-step endpoint-switching implementation. It requires timestep
convergence and does not claim the reversible/event-localized integrator of
Geuther et al., JCTC 21, 2179 (2025). P>1 is an experimental centroid extension,
not a claim to implement every published RP-MASH variant.

MultistateMASHDynamics below instead implements the Runeson--Manolopoulos
P=1 real-matrix method with resolved population-boundary events and its own
observable weights; it does not change the original two-state class.
"""
from dataclasses import dataclass
import copy

import numpy as np
from scipy.integrate import solve_ivp

from .adiabatic import AdiabaticRPSHDynamics
from .dynamics import CayleyIntegrator, State, _state_clock
from .nonadiabatic import _positive_integer, validate_diabatic_result
from .potentials.base import PotentialError
from .rp_mash import _positive_scalar
from .surface_hopping import _surface


class MASHDynamics(AdiabaticRPSHDynamics):
    """Deterministic two-state MASH, with an explicit experimental P>1 extension.

    Pure initial surface populations sample the spin hemisphere with density
    proportional to |S_z|. ``populations`` reports active-surface indicators;
    electronic amplitudes are accessible separately on the state.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates=2, **kwargs):
        if isinstance(nstates, (bool, np.bool_)) or nstates != 2:
            raise ValueError("this MASH implementation requires exactly two states")
        if "frustrated_hops" in kwargs and kwargs["frustrated_hops"] != "reverse":
            raise ValueError("MASH requires NAC momentum reflection at frustrated hops")
        kwargs["frustrated_hops"] = "reverse"
        super().__init__(potential, masses, beta, nbeads, dt, 2, **kwargs)

    def initialize(self, nuclear, active_surface=0, *, adiabatic_coefficients=None, seed=None):
        active = _surface(active_surface, 2)
        if adiabatic_coefficients is None:
            rng = self.rng if seed is None else (seed if isinstance(seed, np.random.Generator)
                                                else np.random.default_rng(seed))
            z = np.sqrt(max(float(rng.random()), np.finfo(float).tiny))
            if active == 1:
                z = -z
            phi = rng.uniform(0., 2*np.pi)
            adiabatic_coefficients = np.array([np.sqrt((1+z)/2),
                                               np.sqrt((1-z)/2)*np.exp(1j*phi)])
        state = super().initialize(nuclear, active, adiabatic_coefficients=adiabatic_coefficients)
        self._check(state)
        return state

    def _check(self, state):
        super()._check(state)
        population = abs(state.coefficients)**2
        if population[state.active_surface] < .5-1e-12:
            raise ValueError("MASH active surface and spin hemisphere disagree")

    def populations(self, state):
        self._check(state)
        return np.eye(2)[state.active_surface]

    def _select_hop(self, trial, probabilities, surfaces):
        crossings = sum(a != b for a, b in zip(surfaces[:-1], surfaces[1:]))
        if crossings > 1:
            raise ValueError("multiple MASH hemisphere crossings in one step; reduce timestep")
        target = int(np.argmax(abs(trial.coefficients)**2))
        if target == trial.active_surface:
            return None
        nac = trial.centroid_frame.result.couplings[0, trial.active_surface, target]
        if np.linalg.norm(nac) <= np.finfo(float).tiny:
            raise PotentialError("MASH crossing has no endpoint NAC rescaling direction; reduce timestep")
        hop = self._hop(trial, target)
        if hop.frustrated:
            # Keep Sx and Sy, reverse Sz: preserve both coefficient phases
            # while swapping their moduli, rather than swapping coefficients.
            moduli = abs(trial.coefficients)
            phase = np.ones(2, dtype=complex)
            np.divide(trial.coefficients, moduli, out=phase, where=moduli > 0)
            trial.coefficients[:] = phase*moduli[::-1]
        return hop

    def attempt_hop(self, state, target):
        raise ValueError("MASH surface changes are determined by spin crossings in step()")


# The original two-state class above deliberately retains its initial sampling,
# endpoint integrator and active-surface population estimator.  Runeson and
# Manolopoulos changed the estimator even for two states; the separately named
# class below must not change the interpretation of existing MASH calculations.
def _reject_boolean_entries(value, name, error=ValueError):
    """Check before heterogeneous Python sequences coerce bools into numbers."""
    if isinstance(value, (bool, np.bool_)):
        raise error(f"{name} must contain numerical values, not booleans")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            raise error(f"{name} must contain numerical values, not booleans")
        if value.dtype.kind == "O":
            for item in value.flat:
                _reject_boolean_entries(item, name, error)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_boolean_entries(item, name, error)


def multistate_mash_parameters(nstates):
    """Return the observable coefficients of JCP 159, 094115 (2023), Eq. 28.

    The mapped density is ``alpha * outer(c, c.conj()) + beta * I``.  Its
    eigenvalues and instantaneous population estimates need not be positive.
    This is the original 2023 Runeson--Manolopoulos prescription, not unSMASH.
    """
    nstates = _positive_integer(nstates, "nstates")
    if nstates < 2:
        raise ValueError("multistate MASH requires at least two states")
    harmonic = float(np.sum(1.0 / np.arange(1, nstates + 1, dtype=float)))
    alpha = (nstates - 1) / (harmonic - 1.0)
    return alpha, (1.0 - alpha) / nstates


@dataclass
class MultistateMASHState(State):
    """Fixed-diabatic amplitudes and the current active adiabatic surface.

    ``sampling`` is either ``population`` for a sampled initial electronic
    pure state or ``custom`` for explicitly supplied coefficients.  All arrays
    are owned.  No local eigenvector phases or hidden electronic frame are
    required to reconstruct this state at a checkpoint.
    """
    coefficients: np.ndarray | None = None
    active_surface: int = 0
    sampling: str = "custom"

    def __post_init__(self):
        for name in ("q", "p", "coefficients"):
            _reject_boolean_entries(getattr(self, name), name)
        super().__post_init__()
        value = np.asarray(self.coefficients)
        if (value.ndim != 1 or value.size < 2 or value.dtype.kind not in "fciu"
                or not np.isfinite(value).all()
                or not np.isclose(np.vdot(value, value).real, 1., rtol=0., atol=1e-8)):
            raise ValueError("coefficients must be a finite normalized electronic vector with at least two states")
        self.coefficients = np.array(value, dtype=np.complex128, order="C", copy=True)
        self.active_surface = _surface(self.active_surface, value.size)
        if self.sampling not in ("population", "custom"):
            raise ValueError("sampling must be population or custom")

    def copy(self):
        return MultistateMASHState(self.q, self.p, self.step, self.time, self.coefficients,
                                   self.active_surface, self.sampling)


@dataclass(frozen=True)
class MultistateMASHHop:
    """Resolved population-boundary event and its instantaneous energy error."""
    time: float
    source: int
    target: int
    accepted: bool
    frustrated: bool
    energy_gap: float
    energy_error: float


class MultistateMASHDynamics:
    """Runeson--Manolopoulos multistate MASH with resolved boundary events.

    This implements the real, nondegenerate, P=1 model of JCP **159**, 094115
    (2023), doi:10.1063/5.0158147, Eqs. 4, 7, 13, 28--37 and Appendix E.
    Electronic coefficients obey the fixed-diabatic Schrodinger equation.
    Nuclei evolve on the state of largest *adiabatic* coefficient modulus.
    The momentum impulse uses the full coherence/NAC sum of Eq. E3, including
    contributions from states other than the two exchanging active status.

    DOP853 and root finding resolve hops and frustrated-hop reflections.  No
    electronic coefficient swap is made at a reflection: the reflected nuclear
    velocity carries the electronic trajectory back into the original region.
    ``dt`` is the output interval; ``rtol``, ``atol`` and ``max_step`` control
    the CPU reference integration.  Convergence in ``max_step`` is necessary:
    sign-change detection can miss an even number of boundary crossings within
    one internal solver step.  Native propagation, P>1, complex SOC and
    degenerate adiabatic surfaces are explicitly unsupported.

    The two-state equations of motion reduce to original MASH, but the 2023
    multistate initial sampling and observable weights differ even for N=2.
    The original :class:`MASHDynamics` is therefore kept separate.  As in the
    publication, this method is not size consistent when uncoupled spectator
    states are added, and signed observable estimates must not be clipped.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates, *, seed=0,
                 rtol=1e-9, atol=1e-11, max_step=None, max_events=64,
                 gap_tolerance=1e-10, native=False, threads=1):
        if callable(getattr(potential, "bare_potential", None)):
            raise ValueError("multistate MASH requires a complete diabatic potential including scalar nuclear terms; "
                             "a separate bare_potential (such as a Newns-Anderson provider) requires IESH "
                             "or an explicit full-potential adapter")
        _reject_boolean_entries(masses, "masses")
        self.nstates = _positive_integer(nstates, "nstates")
        self.alpha, self.observable_offset = multistate_mash_parameters(self.nstates)
        if _positive_integer(nbeads, "nbeads") != 1:
            raise ValueError("multistate MASH currently supports P=1 only")
        if native is not False:
            raise ValueError("multistate MASH uses the CPU adaptive reference solver; native must be False")
        self._validator = CayleyIntegrator(masses, beta, nbeads, dt, native=False, threads=threads)
        self.masses = self._validator.masses
        self.beta, self.nbeads = self._validator.beta, self._validator.nbeads
        self.dt = _positive_scalar(dt, "dt")
        self.rtol, self.atol = _positive_scalar(rtol, "rtol"), _positive_scalar(atol, "atol")
        if self.rtol < 100.0 * np.finfo(float).eps:
            raise ValueError("rtol is below DOP853's double-precision minimum; it would be silently increased")
        self.max_step = self.dt / 4.0 if max_step is None else _positive_scalar(max_step, "max_step")
        self.max_events = _positive_integer(max_events, "max_events")
        self.gap_tolerance = _positive_scalar(gap_tolerance, "gap_tolerance")
        self.potential = potential
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.integrator_name, self.native = "adaptive-dop853", False
        self.last_hops, self.last_hop = (), None
        self._parameters = (self.beta, self.dt, self.nbeads, self.nstates, self.rtol, self.atol,
                            self.max_step, self.max_events, self.gap_tolerance, self.integrator_name,
                            self.native, self.alpha, self.observable_offset)

    def _check_parameters(self):
        if ((self.beta, self.dt, self.nbeads, self.nstates, self.rtol, self.atol,
             self.max_step, self.max_events, self.gap_tolerance, self.integrator_name,
             self.native, self.alpha, self.observable_offset) != self._parameters
                or self.beta != self._validator.beta or self.dt != self._validator.dt
                or self.nbeads != self._validator.nbeads or self.masses is not self._validator.masses):
            raise ValueError("MultistateMASHDynamics parameters are immutable")

    def _frame(self, q):
        cache = getattr(self, "_frame_cache", None)
        if cache is not None and np.array_equal(q, cache[0]):
            return cache[1]
        raw = self.potential.evaluate(q)
        fields = raw[:2] if isinstance(raw, (tuple, list)) else (
            getattr(raw, "matrix", None), getattr(raw, "gradients", None))
        for name, value in zip(("diabatic matrix", "diabatic gradients"), fields):
            _reject_boolean_entries(value, name, PotentialError)
        result = validate_diabatic_result(raw, q.shape, self.nstates)
        if np.any(result.matrix.imag != 0.) or np.any(result.gradients.imag != 0.):
            raise PotentialError("multistate MASH requires real symmetric matrices; complex SOC is unsupported")
        matrix, gradients = result.matrix[0].real, result.gradients[0].real
        energies, vectors = np.linalg.eigh(matrix)
        if np.any(np.diff(energies) <= self.gap_tolerance):
            raise PotentialError("multistate MASH requires nondegenerate adiabatic gaps above gap_tolerance")
        frame = matrix, gradients, energies, vectors
        if hasattr(self, "_frame_cache"):
            # RHS and different pair events often request exactly the same
            # geometry.  Reuse only an exact match, only within this step.
            self._frame_cache = (q.copy(), frame)
        return frame

    def _check(self, state):
        self._check_parameters()
        if not isinstance(state, MultistateMASHState):
            raise ValueError("MultistateMASHDynamics requires MultistateMASHState")
        self._validator._check(state.q, state.p)
        _state_clock(state.step, state.time)
        value = state.coefficients
        if (not isinstance(value, np.ndarray) or value.shape != (self.nstates,)
                or value.dtype != np.complex128 or not value.flags.c_contiguous
                or not value.flags.writeable or not np.isfinite(value).all()
                or not np.isclose(np.vdot(value, value).real, 1., rtol=0., atol=1e-8)):
            raise ValueError("invalid normalized multistate MASH coefficients")
        active = _surface(state.active_surface, self.nstates)
        if state.sampling not in ("population", "custom"):
            raise ValueError("invalid multistate MASH sampling mode")
        arrays = (state.q, state.p, value)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1:]):
            raise ValueError("state arrays must not share memory")
        population = np.abs(self._frame(state.q)[3].T @ value) ** 2
        if population[active] < np.max(population) - 1e-8:
            raise ValueError("active surface disagrees with largest adiabatic population")

    def initialize(self, nuclear, initial_state=0, *, basis="adiabatic", coefficients=None, rng=None):
        """Prepare a pure-state ensemble, or one explicitly specified mapping point.

        For ``coefficients=None``, sample uniformly on the complex unit sphere
        conditional on ``initial_state`` having the largest population in the
        requested basis (2023 Eq. 37).  In the diabatic basis this initial-state
        label can differ from the actual active adiabatic surface.  Custom
        ``coefficients`` are already normalized amplitudes in ``basis`` and do
        not represent a sampled pure-state population ensemble.  The active
        surface is always determined from the adiabatic amplitudes.
        """
        self._check_parameters()
        if not isinstance(nuclear, State):
            raise ValueError("initialize requires a nuclear State")
        self._validator._check(nuclear.q, nuclear.p)
        _state_clock(nuclear.step, nuclear.time)
        initial = _surface(initial_state, self.nstates)
        if basis not in ("adiabatic", "diabatic"):
            raise ValueError("basis must be adiabatic or diabatic")
        vectors = self._frame(nuclear.q)[3]
        generator = self.rng if rng is None else (rng if isinstance(rng, np.random.Generator)
                                                 else np.random.default_rng(rng))
        snapshot = copy.deepcopy(generator.bit_generator.state)
        try:
            if coefficients is None:
                value = generator.normal(size=(2, self.nstates))
                c = value[0] + 1j * value[1]
                norm = np.linalg.norm(c)
                if not np.isfinite(norm) or norm == 0.:
                    raise ValueError("electronic sampling produced a zero or nonfinite vector")
                c /= norm
                # Permutation symmetry gives exactly the conditional uniform
                # measure without the N-fold rejection cost of Eq. 37 sampling.
                largest = int(np.argmax(np.abs(c) ** 2))
                c[initial], c[largest] = c[largest], c[initial]
                sampling = "population"
            else:
                _reject_boolean_entries(coefficients, "coefficients")
                raw = np.asarray(coefficients)
                if (raw.shape != (self.nstates,) or raw.dtype.kind not in "fciu"
                        or not np.isfinite(raw).all()
                        or not np.isclose(np.vdot(raw, raw).real, 1., rtol=0., atol=1e-10)):
                    raise ValueError("coefficients must be a finite normalized vector of length nstates")
                c, sampling = np.array(raw, dtype=complex, copy=True), "custom"
            fixed = vectors @ c if basis == "adiabatic" else c
            adiabatic = c if basis == "adiabatic" else vectors.T @ c
            active = int(np.argmax(np.abs(adiabatic) ** 2))
            state = MultistateMASHState(nuclear.q, nuclear.p, nuclear.step, nuclear.time,
                                        fixed, active, sampling)
            self._check(state)
        except BaseException:
            generator.bit_generator.state = snapshot
            raise
        return state

    def _coefficients_in_basis(self, state, basis):
        if basis == "diabatic":
            return state.coefficients
        if basis == "adiabatic":
            return self._frame(state.q)[3].T @ state.coefficients
        raise ValueError("basis must be adiabatic or diabatic")

    def populations(self, state, *, basis="adiabatic"):
        """Signed 2023 Eq. 29 estimator, not active-surface fractions."""
        self._check(state)
        coefficients = self._coefficients_in_basis(state, basis)
        return self.alpha * np.abs(coefficients) ** 2 + self.observable_offset

    def density_matrix(self, state, *, basis="adiabatic"):
        """Mapped density for electronic observables, including coherences.

        Evaluate a Hermitian observable as ``trace(density_matrix @ operator)``.
        The trace is one; positivity is not imposed on individual samples.
        """
        self._check(state)
        coefficients = self._coefficients_in_basis(state, basis)
        return (self.alpha * np.outer(coefficients, coefficients.conj())
                + self.observable_offset * np.eye(self.nstates))

    def active_populations(self, state):
        """Active-surface fractions diagnostic; not the 2023 population estimator."""
        self._check(state)
        return np.eye(self.nstates)[state.active_surface]

    def electronic_norm(self, state):
        self._check(state)
        return float(np.vdot(state.coefficients, state.coefficients).real)

    def energy(self, state):
        self._check(state)
        energy = self._frame(state.q)[2][state.active_surface]
        return float(energy + np.sum(state.p ** 2 / (2.0 * self.masses[None, :, None])))

    def _unpack(self, y):
        size = int(np.prod(self._validator.shape))
        q = y[:size].reshape(self._validator.shape)
        p = y[size:2 * size].reshape(self._validator.shape)
        c = y[2 * size:2 * size + self.nstates] + 1j * y[2 * size + self.nstates:]
        return q, p, c

    def _rhs(self, time, y, active):
        q, p, coefficients = self._unpack(y)
        matrix, gradients, _, vectors = self._frame(q)
        force = -np.einsum("i,ijac,j->ac", vectors[:, active], gradients, vectors[:, active])
        derivative = -1j * (matrix @ coefficients)
        return np.concatenate(((p / self.masses[None, :, None]).ravel(), force.ravel(),
                               derivative.real, derivative.imag))

    def _boundary_direction(self, q, coefficients, source, target):
        """Half the fixed-c gradient of P_source-P_target (2023 Eq. E3).

        This returns the physical-coordinate direction; mass weighting is
        performed in ``_impulse``.  The pair NAC alone is insufficient for N>2.
        """
        _, gradients, energies, vectors = self._frame(q)
        adiabatic = vectors.T @ coefficients
        gradient = np.einsum("ik,ijac,jl->klac", vectors, gradients, vectors)
        gaps = energies[None, :] - energies[:, None]
        coupling = np.zeros_like(gradient)
        np.divide(gradient, gaps[:, :, None, None], out=coupling,
                  where=~np.eye(self.nstates, dtype=bool)[:, :, None, None])
        direction = (np.einsum("k,kac->ac", np.real(adiabatic.conj() * adiabatic[source]),
                               coupling[:, source])
                     - np.einsum("k,kac->ac", np.real(adiabatic.conj() * adiabatic[target]),
                                 coupling[:, target]))
        return direction, energies

    def _event_value(self, y, source, target):
        q, p, coefficients = self._unpack(y)
        adiabatic = self._frame(q)[3].T @ coefficients
        value = float(abs(adiabatic[source]) ** 2 - abs(adiabatic[target]) ** 2)
        threshold = 64.0 * np.finfo(float).eps
        if abs(value) <= threshold:
            direction, _ = self._boundary_direction(q, coefficients, source, target)
            rate = 2.0 * float(np.sum(direction * p[0] / self.masses[:, None]))
            # Do not rediscover an outgoing or stationary boundary at zero
            # elapsed time after a hop/reflection.  Only roundoff is displaced.
            if rate >= 0.0:
                return threshold
        return value

    def _impulse(self, q, p, coefficients, source, target, event_time):
        source, target = _surface(source, self.nstates), _surface(target, self.nstates)
        if source == target:
            raise ValueError("a MASH impulse requires distinct surfaces")
        direction, energies = self._boundary_direction(q, coefficients, source, target)
        scale = np.linalg.norm(direction)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("population boundary has no finite nonzero rescaling direction")
        direction /= scale
        a = 0.5 * float(np.sum(direction ** 2 / self.masses[:, None]))
        b = float(np.sum(p[0] * direction / self.masses[:, None]))
        if b == 0.0:
            raise ValueError("grazing MASH boundary has zero projected velocity")
        gap = float(energies[target] - energies[source])
        discriminant = b * b - 4.0 * a * gap
        if not np.isfinite([a, b, gap, discriminant]).all() or a <= 0.:
            raise ValueError("multistate MASH projected kinetic energy overflow")
        tolerance = 64.0 * np.finfo(float).eps * max(b * b, abs(4.0 * a * gap), np.finfo(float).tiny)
        accepted = discriminant >= -tolerance
        if accepted:
            root = np.sqrt(max(0., discriminant))
            change = -2.0 * gap / (b + np.copysign(root, b))
        else:
            change = -b / a
        updated = p + change * direction[None]
        energy_error = float(np.sum((updated ** 2 - p ** 2) / (2.0 * self.masses[None, :, None]))
                             + (gap if accepted else 0.0))
        if not np.isfinite(updated).all() or not np.isfinite(energy_error):
            raise ValueError("multistate MASH momentum rescaling overflow")
        return updated, MultistateMASHHop(float(event_time), source, target, bool(accepted),
                                         not accepted, gap, energy_error)

    def step(self, state):
        """Advance atomically: a failed integration leaves state and RNG intact."""
        self._frame_cache = None
        try:
            return self._advance(state)
        finally:
            del self._frame_cache

    def _advance(self, state):
        self._check(state)
        trial = state.copy()
        y = np.concatenate((trial.q.ravel(), trial.p.ravel(), trial.coefficients.real,
                            trial.coefficients.imag))
        elapsed, hops = 0.0, []
        while elapsed < self.dt:
            active = trial.active_surface
            targets = [index for index in range(self.nstates) if index != active]
            events = []
            for target in targets:
                def event(time, value, target=target):
                    return self._event_value(value, active, target)
                event.terminal, event.direction = True, -1.0
                events.append(event)
            solution = solve_ivp(lambda time, value: self._rhs(time, value, active),
                                 (elapsed, self.dt), y, method="DOP853", rtol=self.rtol,
                                 atol=self.atol, max_step=self.max_step, events=events)
            if not solution.success or not np.isfinite(solution.y[:, -1]).all():
                raise ValueError("multistate MASH integration failed: " + solution.message)
            y = solution.y[:, -1].copy()
            elapsed = float(solution.t[-1])
            found = [index for index, times in enumerate(solution.t_events) if times.size]
            if not found:
                break
            if len(hops) >= self.max_events:
                raise ValueError("too many multistate MASH events; reduce dt or inspect a grazing crossing")
            target = targets[found[0]]
            q, p, coefficients = self._unpack(y)
            population = np.abs(self._frame(q)[3].T @ coefficients) ** 2
            if np.count_nonzero(np.abs(population - population[active]) <= 64.0 * np.finfo(float).eps) > 2:
                raise ValueError("simultaneous multistate MASH boundaries are ambiguous")
            updated, hop = self._impulse(q, p, coefficients, active, target, state.time + elapsed)
            p[:] = updated
            if hop.accepted:
                trial.active_surface = target
            hops.append(hop)
        q, p, coefficients = self._unpack(y)
        trial.q[:], trial.p[:], trial.coefficients[:] = q, p, coefficients
        trial.step += 1
        trial.time += self.dt
        self._check(trial)
        state.q[:], state.p[:], state.coefficients[:] = trial.q, trial.p, trial.coefficients
        state.step, state.time = trial.step, trial.time
        state.active_surface, state.sampling = trial.active_surface, trial.sampling
        self.last_hops = tuple(hops)
        self.last_hop = hops[-1] if hops else None
        return state

    def attempt_hop(self, state, target):
        raise ValueError("multistate MASH changes surface only at resolved population boundaries in step()")
