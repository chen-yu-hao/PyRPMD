"""Two-state bead-approximation RP-MASH with a common adiabatic spin.

Cao, Suarez and Kretchmer, doi:10.26434/chemrxiv-2025-xps2g (2025
preprint), Eqs. 21, 29 and 34--43.  All beads share an active surface;
the spin sees bead averages of the gap and NAC dot velocity.  Hops use
the full bead-space NAC and the sum-bead energy, including unequal masses.
Real symmetric, nondegenerate two-state diabatic providers only. Atomic units.
"""
from dataclasses import dataclass
import copy

import numpy as np

from .dynamics import State, _state_clock
from .potentials.base import PotentialError
from .rp_mash import MASHHop, RPMASHDynamics, _positive_scalar
from .spin_mapping import _spin_kernel, _two_state_index, _unit_vectors


@dataclass
class RPMASHBAState(State):
    """One common adiabatic Bloch vector and a common active ring surface.

    ``spin[2]`` is positive on the upper surface (``active_surface=1``),
    negative on the lower (0).  Its Pauli axes follow Eq. 22 of the paper:
    z=(hx,0,hz)/|h|, x=(hz,0,-hx)/|h|, y=(0,1,0) in each bead's fixed
    diabatic basis.  No eigenvector phases or transport history are needed.
    This is a distinct state type from the fixed-diabatic ``RPMASHState``.
    """
    spin: np.ndarray | None = None
    active_surface: int = 0
    sampling: str = "custom"

    def __post_init__(self):
        super().__post_init__()
        self.spin = _unit_vectors(self.spin, (3,))
        self.active_surface = _two_state_index(self.active_surface, "active_surface")
        if self.sampling not in ("custom", "population"):
            raise ValueError("BA-RP-MASH sampling must be custom or population")

    def copy(self):
        return RPMASHBAState(self.q, self.p, self.step, self.time,
                            self.spin, self.active_surface, self.sampling)

    def populations(self):
        """Common adiabatic active indicator, ordered (lower, upper).

        Average only over the specified population importance preparation.
        A custom spin does not by itself define a population ensemble.
        """
        return np.eye(2)[self.active_surface]

    def spin_density_matrix(self):
        """Adiabatic propagation diagnostic, ordered (upper, lower).

        This pure spin density is NOT a MASH observable estimator.  In
        particular it is not a density matrix in a fixed diabatic basis.
        """
        return _spin_kernel(self.spin, 0.5)


class RPMASHBADynamics(RPMASHDynamics):
    """BA-RP-MASH, using the shared transactional DOP853 event integrator.

    The constructor and reporting controls match ``RPMASHDynamics``.
    Electronic evolution evaluates one matrix batch for the entire ring,
    without a separate centroid matrix call.  A hop changes the common
    surface and pays the full ring gap along the bead-resolved NAC.  A
    frustrated hop reflects the mass-weighted NAC velocity.

    Only adiabatic population preparation is exposed.  General operator
    correlations, complex/SOC providers and multistate dynamics are rejected.
    Like the source formulation this is an approximate trajectory method;
    narrow localized couplings can be underestimated by bead averaging.
    """
    _method_label = "BA-RP-MASH"

    def __init__(self, potential, masses, beta, nbeads, dt, nstates=2, *, seed=0,
                 rtol=1e-9, atol=1e-11, max_step=None, max_events=64,
                 gap_tolerance=1e-10, native=False, threads=1):
        rtol = _positive_scalar(rtol, "rtol")
        if rtol < 100.0 * np.finfo(float).eps:
            raise ValueError("rtol is below DOP853's double-precision minimum; "
                             "it would be silently increased")
        super().__init__(potential, masses, beta, nbeads, dt, nstates, seed=seed,
                         rtol=rtol, atol=atol, max_step=max_step, max_events=max_events,
                         gap_tolerance=gap_tolerance, native=native, threads=threads)

    def _check_parameters(self):
        super()._check_parameters()
        if self.masses is not self._validator.masses:
            raise ValueError("RPMASHBADynamics parameters are immutable")

    def _electronic(self, q):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = super()._electronic(q)
        if not all(np.isfinite(value).all() for value in values):
            raise PotentialError("BA-RP-MASH electronic energy or gradient overflow")
        return values

    def _check(self, state):
        self._check_parameters()
        if not isinstance(state, RPMASHBAState):
            raise ValueError("RPMASHBADynamics requires RPMASHBAState with an adiabatic spin")
        self._validator._check(state.q, state.p)
        _state_clock(state.step, state.time)
        spin = state.spin
        if (not isinstance(spin, np.ndarray) or spin.shape != (3,)
                or spin.dtype != np.float64 or not spin.flags.writeable
                or not spin.flags.c_contiguous or not np.isfinite(spin).all()
                or not np.isclose(spin @ spin, 1.0, rtol=0.0, atol=1e-8)):
            raise ValueError("invalid unit adiabatic spin vector")
        _two_state_index(state.active_surface, "active_surface")
        if state.sampling not in ("custom", "population"):
            raise ValueError("BA-RP-MASH sampling must be custom or population")
        arrays = (state.q, state.p, spin)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")
        if (2 * state.active_surface - 1) * spin[2] < -1e-8:
            raise ValueError("active surface is inconsistent with the adiabatic spin hemisphere")
        # Every bead must have a real, nondegenerate PES; a centroid check
        # would miss a degeneracy or invalid provider output on another bead.
        self._electronic(state.q)

    def initialize(self, nuclear, active_surface=0, *, sampling="population", rng=None):
        """Copy a nuclear parent and importance-sample one adiabatic spin.

        abs(Sz)=sqrt(U), uniform azimuth.  This absorbs the population-pair
        weight 2*abs(Sz) of Eq. 29, leaving an unweighted active indicator.
        It does not sample the coupled electron-nuclear thermal distribution.
        """
        self._check_parameters()
        if not isinstance(nuclear, State):
            raise ValueError("initialize requires a nuclear State")
        self._validator._check(nuclear.q, nuclear.p)
        _state_clock(nuclear.step, nuclear.time)
        if sampling != "population":
            raise ValueError("BA-RP-MASH initialize supports only population sampling")
        active = _two_state_index(active_surface, "active_surface")
        self._electronic(nuclear.q)
        generator = self.rng if rng is None else (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng))
        snapshot = copy.deepcopy(generator.bit_generator.state)
        try:
            sz = (2 * active - 1) * np.sqrt(1.0 - generator.random())
            phi = generator.uniform(0.0, 2.0 * np.pi)
            transverse = np.sqrt(max(0.0, 1.0 - sz * sz))
            spin = np.array([transverse * np.cos(phi), transverse * np.sin(phi), sz])
            state = RPMASHBAState(nuclear.q, nuclear.p, nuclear.step,
                                 nuclear.time, spin, active, sampling)
            self._check(state)
        except BaseException:
            generator.bit_generator.state = snapshot
            raise
        return state

    @staticmethod
    def _nac(values):
        """d_b=<upper_b|grad lower_b>, paper Eq. 21, shape [P,N,3]."""
        _, _, magnitude, axis, _, dfield, _ = values
        # Use a unit field for one factor to avoid unnecessarily squaring
        # the gap.  This fixes the adiabatic Pauli-frame gauge analytically.
        direction = (axis[:, 0, None, None] * dfield[:, 2]
                     - axis[:, 2, None, None] * dfield[:, 0]) / (2.0 * magnitude[:, None, None])
        if not np.isfinite(direction).all():
            raise ValueError("BA-RP-MASH derivative coupling overflow")
        return direction

    def _angular_velocity(self, values, p):
        velocity = p / self.masses[None, :, None]
        coupling = np.sum(self._nac(values) * velocity) / self.nbeads
        omega = np.array([0.0, 2.0 * coupling, 2.0 * np.mean(values[2])])
        if not np.isfinite(omega).all():
            raise ValueError("BA-RP-MASH electronic angular velocity overflow")
        return omega

    def _rhs(self, t, y, active):
        q, p, spin = self._unpack(y)
        values = self._electronic(q)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            spring = ((self.nbeads / self.beta) ** 2 * self.masses[None, :, None]
                      * (2*q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0)))
            force = -values[4] - (2 * active - 1) * values[6] - spring
            result = np.concatenate(((p / self.masses[None, :, None]).ravel(),
                                     force.ravel(), np.cross(self._angular_velocity(values, p), spin)))
        if not np.isfinite(result).all():
            raise ValueError("BA-RP-MASH equations of motion overflow")
        return result

    def _event_value(self, y, active):
        q, p, spin = self._unpack(y)
        sign = 2 * active - 1
        value = sign * float(spin[2])
        # Suppress only outgoing/constant, roundoff-width equatorial roots.
        # In particular, do not change the spin or use an endpoint hop test.
        threshold = 64.0 * np.finfo(float).eps
        if abs(value) <= threshold:
            omega = self._angular_velocity(self._electronic(q), p)
            rate = -sign * omega[1] * spin[0]
            if rate >= 0.0:
                return threshold
        return value

    def _impulse(self, q, p, source, event_time):
        """Enforce sum-bead energy along the complete bead-space NAC.

        p'=p+a*d, A=1/2 sum(d^2/m), B=sum(p*d/m), so the accepted
        root obeys A*a^2+B*a+DeltaE=0.  This mass metric is essential
        when nuclei have unequal masses; ordinary Euclidean projection
        plus an effective mass generally fails to conserve kinetic energy.
        """
        values = self._electronic(q)
        direction = self._nac(values)
        scale = np.max(np.abs(direction))
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("equator crossing has no finite nonzero bead derivative coupling")
        # The direction's normalization cancels in the impulse.  Max-scaling
        # keeps the quadratic coefficients bounded even for a sharp NAC.
        direction = direction / scale
        inverse_mass = 1.0 / self.masses[None, :, None]
        a = 0.5 * float(np.sum(direction**2 * inverse_mass))
        b = float(np.sum(p * direction * inverse_mass))
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
            raise ValueError("BA-RP-MASH momentum rescaling overflow")
        if b == 0.0:
            raise ValueError("grazing equator crossing has zero velocity along the derivative coupling")
        target = 1 - source
        gap = float(2.0 * (2 * target - 1) * np.sum(values[2]))
        discriminant = b*b - 4.0*a*gap
        if not np.isfinite(gap) or not np.isfinite(discriminant):
            raise ValueError("BA-RP-MASH momentum rescaling overflow")
        tolerance = (64.0 * np.finfo(float).eps
                     * max(b*b, abs(4.0*a*gap), np.finfo(float).tiny))
        accepted = discriminant >= -tolerance
        if accepted:
            root = np.sqrt(max(0.0, discriminant))
            change = -2.0 * gap / (b + np.copysign(root, b))
        else:
            change = -b / a
        updated = p + change * direction
        # Evaluate the kinetic difference without subtracting large squares.
        kinetic_change = float(np.sum((updated-p) * (updated+p) * (0.5 * inverse_mass)))
        error = kinetic_change + (gap if accepted else 0.0)
        if not np.isfinite(updated).all() or not np.isfinite(error):
            raise ValueError("BA-RP-MASH momentum rescaling overflow")
        return updated, MASHHop(event_time, source, target, bool(accepted),
                                not accepted, gap, error)

    def energy(self, state):
        """Finite sum-bead energy; a nonrepresentable result fails explicitly."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = super().energy(state)
        if not np.isfinite(value):
            raise ValueError("BA-RP-MASH ring energy overflow")
        return value

    def correlation(self, initial, final, initial_operator, final_operator, *, basis="diabatic"):
        """General operator correlations are not part of this BA implementation."""
        raise NotImplementedError("BA-RP-MASH exposes adiabatic population preparation only; "
                                  "general operator correlations are not implemented")
