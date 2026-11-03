"""Two-state centroid-approximation RP-MASH with resolved equator events.

Equations 36, 44--50 of Cao, Suarez and Kretchmer,
doi:10.26434/chemrxiv-2025-xps2g (2025 preprint).  This is CA-RP-MASH,
not the bead-average variant, FSSH, or a multistate extension.  Real symmetric
diabatic matrices with nonzero adiabatic gaps are required.  Atomic units.
"""
from dataclasses import dataclass
import copy
import numbers

import numpy as np
from scipy.integrate import solve_ivp

from .dynamics import CayleyIntegrator, State, _state_clock
from .nonadiabatic import _positive_integer, validate_diabatic_result
from .potentials.base import PotentialError
from .spin_mapping import _pauli_components, _spin_kernel, _two_state_index, _unit_vectors


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be positive and finite")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass
class RPMASHState(State):
    """One fixed-diabatic unit Bloch vector and a common active ring surface.

    ``active_surface`` is 0 (lower) or 1 (upper).  ``sampling`` records the
    electronic preparation, not an electronic thermal distribution.
    """
    spin: np.ndarray | None = None
    active_surface: int = 0
    sampling: str = "custom"

    def __post_init__(self):
        super().__post_init__()
        self.spin = _unit_vectors(self.spin, (3,))
        self.active_surface = _two_state_index(self.active_surface, "active_surface")
        if self.sampling not in ("custom", "population", "uniform-sphere"):
            raise ValueError("sampling must be custom, population or uniform-sphere")

    def copy(self):
        return RPMASHState(self.q, self.p, self.step, self.time, self.spin, self.active_surface, self.sampling)

    def populations(self):
        """Unweighted active-surface indicator in the centroid adiabatic basis.

        Its average is a population estimate only for the corresponding
        population importance sampling.  Uniform-sphere samples instead need
        the operator-pair weights in ``RPMASHDynamics.correlation``.
        """
        return np.eye(2)[self.active_surface]

    def spin_density_matrix(self):
        """Electronic propagation diagnostic, NOT the MASH density estimator."""
        return _spin_kernel(self.spin, 0.5)


@dataclass(frozen=True)
class MASHHop:
    time: float
    source: int
    target: int
    accepted: bool
    frustrated: bool
    ring_energy_gap: float
    energy_error: float


def _rescale_ring_momentum(p, direction, masses, gap, source, event_time):
    """Energy-conserving impulse for a supplied bead-space NAC direction.

    ``direction`` may have any finite nonzero common normalization.  The
    quadratic uses the physical mass metric, including unequal atom masses.
    CA and BA differ in their direction, not in this conservation equation.
    """
    a = 0.5 * float(np.sum(direction ** 2 / masses[None, :, None]))
    b = float(np.sum(p * direction / masses[None, :, None]))
    if not np.isfinite([a, b, gap]).all() or a <= 0.0:
        raise ValueError("momentum rescaling overflow")
    if b == 0.0:
        raise ValueError("grazing equator crossing has zero velocity along the derivative coupling")
    discriminant = b*b - 4.0*a*gap
    if not np.isfinite(discriminant):
        raise ValueError("momentum rescaling overflow")
    tolerance = 64.0 * np.finfo(float).eps * max(b*b, abs(4.0*a*gap), np.finfo(float).tiny)
    accepted = discriminant >= -tolerance
    if accepted:
        root = np.sqrt(max(0.0, discriminant))
        # This algebraically equivalent form avoids subtracting nearly
        # equal momenta for a small electronic energy change.
        denominator = b + np.copysign(root, b)
        change = -2.0 * gap / denominator
    else:
        change = -b / a
    updated = p + change * direction
    kinetic_change = float(np.sum((updated**2 - p**2) / (2.0 * masses[None, :, None])))
    error = kinetic_change + (gap if accepted else 0.0)
    if not np.isfinite(updated).all() or not np.isfinite(error):
        raise ValueError("momentum rescaling overflow")
    return updated, MASHHop(event_time, source, 1-source, bool(accepted), not accepted, gap, error)


class RPMASHDynamics:
    """Deterministic CA-RP-MASH using adaptive DOP853 plus equator root finding.

    Every bead feels its own active adiabatic force; the electronic spin
    evolves with V at the centroid.  Equator events rescale all momenta along
    the centroid derivative coupling using the full ring energy gap.  An
    energetically forbidden upward hop reflects the mass-weighted NAC
    momentum component, leaving the active surface unchanged.

    ``dt`` is the output/advance interval.  ``rtol``, ``atol`` and ``max_step``
    control internal integration and event resolution.  This CPU reference
    driver does not use a Cayley splitting or claim a symplectic MASH map.
    """
    _method_label = "CA-RP-MASH"

    def __init__(self, potential, masses, beta, nbeads, dt, nstates=2, *, seed=0,
                 rtol=1e-9, atol=1e-11, max_step=None, max_events=64,
                 gap_tolerance=1e-10, native=False, threads=1):
        if _positive_integer(nstates, "nstates") != 2:
            raise ValueError("RPMASHDynamics implements exactly two electronic states")
        if native is not False:
            raise ValueError(f"{self._method_label} currently uses the CPU adaptive reference solver; native must be False")
        self._validator = CayleyIntegrator(masses, beta, nbeads, dt, native=False, threads=threads)
        self.masses = self._validator.masses
        self.beta, self.dt, self.nbeads, self.nstates = float(beta), float(dt), int(nbeads), 2
        self.dt = _positive_scalar(self.dt, "dt")
        self.rtol, self.atol = _positive_scalar(rtol, "rtol"), _positive_scalar(atol, "atol")
        self.max_step = self.dt / 4.0 if max_step is None else _positive_scalar(max_step, "max_step")
        self.max_events = _positive_integer(max_events, "max_events")
        self.gap_tolerance = _positive_scalar(gap_tolerance, "gap_tolerance")
        self.potential = potential
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.integrator_name, self.native = "adaptive-dop853", False
        self.last_hops, self.last_hop = (), None
        self._parameters = (self.beta, self.dt, self.nbeads, self.nstates, self.rtol, self.atol,
                            self.max_step, self.max_events, self.gap_tolerance, self.integrator_name, self.native)

    def _check_parameters(self):
        if ((self.beta, self.dt, self.nbeads, self.nstates, self.rtol, self.atol, self.max_step,
             self.max_events, self.gap_tolerance, self.integrator_name, self.native) != self._parameters
                or self.beta != self._validator.beta or self.dt != self._validator.dt
                or self.nbeads != self._validator.nbeads):
            raise ValueError(f"{type(self).__name__} parameters are immutable")

    def _evaluate(self, q):
        value = validate_diabatic_result(self.potential.evaluate(q), q.shape, 2)
        if np.any(np.abs(value.matrix.imag) > 1e-12) or np.any(np.abs(value.gradients.imag) > 1e-12):
            raise PotentialError(f"{self._method_label} requires real symmetric matrices; complex SOC is not implemented")
        # Treat accepted imaginary roundoff as zero consistently: the real
        # adiabatic frame below must not contain a residual y-field.
        value.matrix = np.ascontiguousarray(value.matrix.real)
        value.gradients = np.ascontiguousarray(value.gradients.real)
        return value

    def _electronic(self, q):
        value = self._evaluate(q)
        scalar, field, dscalar, dfield = _pauli_components(value.matrix, value.gradients)
        magnitude = np.linalg.norm(field, axis=1)
        if np.any(2.0 * magnitude <= self.gap_tolerance):
            raise PotentialError(f"{self._method_label} requires a nondegenerate adiabatic gap")
        axis = field / magnitude[:, None]
        dmag = np.einsum("bi,biac->bac", axis, dfield)
        return scalar, field, magnitude, axis, dscalar, dfield, dmag

    def _frame(self, q):
        """Right-handed adiabatic Pauli axes; no eigenvector phase choices."""
        values = self._electronic(q)
        axis = values[3]
        xaxis = np.stack((axis[:, 2], np.zeros(len(axis)), -axis[:, 0]), axis=1)
        yaxis = np.broadcast_to([0.0, 1.0, 0.0], axis.shape)
        return np.stack((xaxis, yaxis, axis), axis=-1), values

    def _check(self, state):
        self._check_parameters()
        if not isinstance(state, RPMASHState):
            raise ValueError("RPMASHDynamics requires RPMASHState")
        self._validator._check(state.q, state.p)
        _state_clock(state.step, state.time)
        value = state.spin
        if (not isinstance(value, np.ndarray) or value.shape != (3,) or value.dtype != np.float64
                or not value.flags.writeable or not value.flags.c_contiguous or not np.isfinite(value).all()
                or not np.isclose(value @ value, 1.0, rtol=0.0, atol=1e-8)):
            raise ValueError("invalid unit spin vector")
        _two_state_index(state.active_surface, "active_surface")
        if state.sampling not in ("custom", "population", "uniform-sphere"):
            raise ValueError("invalid sampling mode")
        arrays = (state.q, state.p, state.spin)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")
        _, values = self._frame(state.q.mean(axis=0, keepdims=True))
        if (2 * state.active_surface - 1) * float(state.spin @ values[3][0]) < -1e-8:
            raise ValueError("active surface is inconsistent with the centroid spin hemisphere")

    def initialize(self, nuclear, active_surface=None, *, sampling="population", rng=None):
        """Prepare an electronic ensemble while copying the nuclear parent.

        ``population`` draws Sz = +/-sqrt(U), phi uniformly, for a specified
        initial adiabatic state (default lower).  ``uniform-sphere`` draws
        Sz uniformly on [-1,1] and requires operator-pair correlation weights.
        No focused pole approximation or stochastic hopping is used.
        """
        self._check_parameters()
        if not isinstance(nuclear, State):
            raise ValueError("initialize requires a nuclear State")
        self._validator._check(nuclear.q, nuclear.p)
        if sampling not in ("population", "uniform-sphere"):
            raise ValueError("sampling must be population or uniform-sphere")
        if sampling == "uniform-sphere" and active_surface is not None:
            raise ValueError("uniform-sphere sampling determines the initial active surface")
        active = 0 if active_surface is None else _two_state_index(active_surface, "active_surface")
        frame, _ = self._frame(nuclear.q.mean(axis=0, keepdims=True))
        generator = self.rng if rng is None else (rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng))
        snapshot = copy.deepcopy(generator.bit_generator.state)
        try:
            if sampling == "population":
                # Avoid the measure-zero equator endpoint without rejection or
                # a finite-width cutoff in the target distribution.
                sz = (2 * active - 1) * np.sqrt(1.0 - generator.random())
            else:
                sz = generator.uniform(-1.0, 1.0)
                active = int(sz > 0.0)
            phi = generator.uniform(0.0, 2.0 * np.pi)
            transverse = np.sqrt(max(0.0, 1.0 - sz * sz))
            spin = frame[0] @ np.array([transverse * np.cos(phi), transverse * np.sin(phi), sz])
            state = RPMASHState(nuclear.q, nuclear.p, nuclear.step, nuclear.time, spin, active, sampling)
            self._check(state)
        except Exception:
            generator.bit_generator.state = snapshot
            raise
        return state

    def populations(self, state):
        self._check(state)
        return state.populations()

    def spin_density_matrix(self, state):
        """Pure electronic spin diagnostic; not a MASH observable estimator."""
        self._check(state)
        return state.spin_density_matrix()

    def electronic_norm(self, state):
        self._check(state)
        return float(state.spin @ state.spin)

    def _unpack(self, y):
        size = int(np.prod(self._validator.shape))
        return y[:size].reshape(self._validator.shape), y[size:2*size].reshape(self._validator.shape), y[-3:]

    def _rhs(self, t, y, active):
        q, p, spin = self._unpack(y)
        _, _, _, _, dscalar, _, dmag = self._electronic(q)
        field = self._electronic(q.mean(axis=0, keepdims=True))[1][0]
        spring = (self.nbeads / self.beta) ** 2 * self.masses[None, :, None] * (2*q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0))
        force = -dscalar - (2 * active - 1) * dmag - spring
        return np.concatenate(((p / self.masses[None, :, None]).ravel(), force.ravel(), 2.0 * np.cross(field, spin)))

    def _event_value(self, y, active):
        q, p, spin = self._unpack(y)
        _, _, magnitude, axis, _, dfield, dmag = self._electronic(q.mean(axis=0, keepdims=True))
        sign = 2 * active - 1
        value = sign * float(axis[0] @ spin)
        # A constant or outgoing equatorial spin must not be re-detected as an
        # incoming hop at time zero.  Only a roundoff-width region is affected.
        if abs(value) <= 64.0 * np.finfo(float).eps:
            daxis = (dfield[0] - axis[0, :, None, None] * dmag[0]) / magnitude[0]
            rate = sign * np.einsum("i,iac,ac->", spin, daxis, (p / self.masses[None, :, None]).mean(axis=0))
            if rate >= 0.0:
                return 64.0 * np.finfo(float).eps
        return value

    def _impulse(self, q, p, source, event_time):
        """Solve energy conservation along the centroid NAC, for any masses."""
        _, field, magnitude, _, _, dfield, _ = self._electronic(q.mean(axis=0, keepdims=True))
        direction = (field[0, 0] * dfield[0, 2] - field[0, 2] * dfield[0, 0]) / (2.0 * magnitude[0] ** 2)
        scale = np.linalg.norm(direction)
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("equator crossing has no finite nonzero centroid derivative coupling")
        direction = np.broadcast_to(direction / scale, p.shape)
        target = 1 - source
        gap = float(2.0 * (2 * target - 1) * np.sum(self._electronic(q)[2]))
        return _rescale_ring_momentum(p, direction, self.masses, gap, source, event_time)

    def step(self, state):
        self._check(state)
        trial = state.copy()
        y = np.concatenate((trial.q.ravel(), trial.p.ravel(), trial.spin))
        elapsed, events = 0.0, []
        while elapsed < self.dt:
            active = trial.active_surface
            def event(t, value):
                return self._event_value(value, active)
            event.terminal, event.direction = True, -1.0
            solution = solve_ivp(lambda t, value: self._rhs(t, value, active), (elapsed, self.dt), y,
                                 method="DOP853", rtol=self.rtol, atol=self.atol,
                                 max_step=self.max_step, events=event)
            if not solution.success or not np.isfinite(solution.y[:, -1]).all():
                raise ValueError(self._method_label + " integration failed: " + solution.message)
            y = solution.y[:, -1].copy()
            elapsed = float(solution.t[-1])
            if not solution.t_events[0].size:
                break
            if len(events) >= self.max_events:
                raise ValueError(f"too many {self._method_label} equator events; reduce dt or inspect a grazing crossing")
            q, p, spin = self._unpack(y)
            updated, hop = self._impulse(q, p, active, state.time + elapsed)
            p[:] = updated
            if hop.accepted:
                trial.active_surface = hop.target
            events.append(hop)
            # At threshold an accepted upward hop can have zero outgoing NAC
            # velocity.  A subsequent zero-time grazing event is an explicit
            # error rather than an unreported infinite loop or missed hop.
        q, p, spin = self._unpack(y)
        trial.q[:], trial.p[:], trial.spin[:] = q, p, spin
        trial.step += 1
        trial.time += self.dt
        self._check(trial)
        for key in ("q", "p", "spin"):
            getattr(state, key)[:] = getattr(trial, key)
        state.step, state.time, state.active_surface = trial.step, trial.time, trial.active_surface
        self.last_hops = tuple(events)
        self.last_hop = events[-1] if events else None
        return state

    def energy(self, state):
        """Full ring active-surface energy, conserved across accepted hops."""
        self._check(state)
        scalar, _, magnitude, _, _, _, _ = self._electronic(state.q)
        potential = np.sum(scalar + (2 * state.active_surface - 1) * magnitude)
        kinetic = np.sum(state.p**2 / (2.0 * self.masses[None, :, None]))
        dq = state.q - np.roll(state.q, -1, axis=0)
        spring = 0.5 * (self.nbeads / self.beta)**2 * np.sum(self.masses[None, :, None] * dq**2)
        return float(potential + kinetic + spring)

    def correlation(self, initial, final, initial_operator, final_operator, *, basis="diabatic"):
        """One uniform-full-sphere contribution to Tr[A B(t)], paper Eq. 29.

        A and B are Hermitian 2x2 matrices.  ``basis='adiabatic'`` uses the
        order (upper, lower), matching sigma_z = diag(+1,-1); ``diabatic``
        uses the provider's fixed basis.  Average contributions over complete
        independent initial trajectories; do not divide by sampled weights.
        Nuclear initial-distribution factors are the caller's responsibility.
        """
        self._check(initial)
        self._check(final)
        if initial.sampling != "uniform-sphere":
            raise ValueError("correlation requires uniform-sphere initial sampling")
        if basis not in ("diabatic", "adiabatic"):
            raise ValueError("basis must be diabatic or adiabatic")
        pieces = []
        initial_sz = None
        for state, operator in ((initial, initial_operator), (final, final_operator)):
            raw = np.asarray(operator)
            if (raw.shape != (2, 2) or raw.dtype.kind not in "fciu" or not np.isfinite(raw).all()
                    or not np.allclose(raw, raw.conj().T, rtol=0.0, atol=1e-12)):
                raise ValueError("correlation operators must be finite Hermitian 2x2 matrices")
            frame, _ = self._frame(state.q.mean(axis=0, keepdims=True))
            spin = frame[0].T @ state.spin
            if initial_sz is None:
                initial_sz = spin[2]
            a0 = 0.5 * float(np.trace(raw).real)
            vector = np.array([raw[0, 1].real, -raw[0, 1].imag, 0.5 * (raw[0, 0].real - raw[1, 1].real)])
            if basis == "diabatic":
                vector = frame[0].T @ vector
            population = a0 + vector[2] * (2 * state.active_surface - 1)
            coherence = float(vector[:2] @ spin[:2])
            pieces.append((population, coherence))
        p0, c0 = pieces[0]
        p1, c1 = pieces[1]
        # The paper's dS integrates to 2; uniform probability on the sphere
        # integrates to 1.  The leading factor 2 supplies that normalization.
        return float(2.0 * (2.0 * abs(initial_sz) * p0 * p1 + 2.0 * (p0*c1 + c0*p1) + 3.0 * c0*c1))
