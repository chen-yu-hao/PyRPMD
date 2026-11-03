"""Two-state bead-approximation RP-MASH, in a shared adiabatic spin basis.

Cao, Suarez and Kretchmer, doi:10.26434/chemrxiv-2025-xps2g (2025 preprint),
Eqs. 21, 29 and 34--43.  Electronic evolution averages the bead gaps and
velocity--NAC contractions; an event changes the *sum* of bead energies.
This practical ring-polymer approximation is not a multistate theory or a
claim of exact quantum dynamics.  Coordinates and parameters use atomic units.
"""
from dataclasses import dataclass
import copy

import numpy as np

from .dynamics import State, _state_clock
from .mash import _reject_boolean_entries
from .nonadiabatic import _positive_integer, validate_diabatic_result
from .potentials.base import PotentialError
from .rp_mash import RPMASHDynamics, _positive_scalar, _rescale_ring_momentum
from .spin_mapping import _pauli_components, _spin_kernel, _two_state_index, _unit_vectors


@dataclass
class BARPMASHState(State):
    """One shared *adiabatic* unit spin for all ring-polymer beads.

    Unlike ``RPMASHState``, this is not a fixed-diabatic spin.  The active
    labels are 0=lower and 1=upper, whereas the Pauli spin and its diagnostic
    density matrix use the order (upper, lower), with Sz>0 on the upper PES.
    The analytic real two-state Pauli frame fixes the NAC convention without
    a saved eigenvector gauge.  ``sampling`` records electronic preparation.
    """
    spin: np.ndarray | None = None
    active_surface: int = 0
    sampling: str = "custom"

    def __post_init__(self):
        for name in ("q", "p", "spin"):
            _reject_boolean_entries(getattr(self, name), name)
        super().__post_init__()
        self.spin = _unit_vectors(self.spin, (3,))
        self.active_surface = _two_state_index(self.active_surface, "active_surface")
        if not isinstance(self.sampling, str) or self.sampling not in ("custom", "population", "uniform-sphere"):
            raise ValueError("sampling must be custom, population or uniform-sphere")

    def copy(self):
        return BARPMASHState(self.q, self.p, self.step, self.time,
                            self.spin, self.active_surface, self.sampling)

    def populations(self):
        """Lower/upper active indicator; an estimator with population sampling."""
        return np.eye(2)[self.active_surface]

    def spin_density_matrix(self):
        """Shared-adiabatic propagation diagnostic, NOT a MASH density estimate.

        Matrix order is (upper, lower), unlike the active-surface labels.
        """
        return _spin_kernel(self.spin, 0.5)


class BARPMASHDynamics(RPMASHDynamics):
    """BA-RP-MASH with event-resolved, deterministic surface changes.

    Real symmetric, nondegenerate two-state diabatic matrices are required
    *at each bead*.  No centroid electronic evaluation is needed.  Each bead
    feels its own active adiabatic force and the physical-mass ring spring.
    The shared adiabatic spin satisfies ``Sdot = (0, 2*D, 2*rbar) cross S``,
    where ``D = mean_b(sum_ac(p_bac*d_bac/m_a))`` and ``rbar`` is the mean
    half-gap.  An equator event applies the full bead-space NAC impulse and
    conserves the full ring energy, including unequal nuclear masses.

    The CPU DOP853 event driver and energy convention are shared with CA.
    ``dt`` is the output interval, not an internal fixed step.  Converge
    ``max_step``, ``rtol`` and ``atol``: sign-change event detection can miss
    an even number of crossings within an internal step.  No native/GPU,
    complex SOC, degenerate bead, or multistate fallback is supplied.
    """
    _method_label = "BA-RP-MASH"

    def __init__(self, potential, masses, beta, nbeads, dt, nstates=2, *, seed=0,
                 rtol=1e-9, atol=1e-11, max_step=None, max_events=64,
                 gap_tolerance=1e-10, native=False, threads=1):
        if _positive_integer(nstates, "nstates") != 2:
            raise ValueError("BA-RP-MASH implements exactly two electronic states")
        if callable(getattr(potential, "bare_potential", None)):
            raise ValueError("BA-RP-MASH requires a complete diabatic potential including scalar nuclear terms; "
                             "a separate bare_potential requires IESH or an explicit full-potential adapter")
        _reject_boolean_entries(masses, "masses")
        rtol = _positive_scalar(rtol, "rtol")
        if rtol < 100.0 * np.finfo(float).eps:
            raise ValueError("rtol is below DOP853's double-precision minimum; it would be silently increased")
        super().__init__(potential, masses, beta, nbeads, dt, nstates, seed=seed,
                         rtol=rtol, atol=atol, max_step=max_step, max_events=max_events,
                         gap_tolerance=gap_tolerance, native=native, threads=threads)
        self._mass_snapshot = self.masses.copy()

    def _check_parameters(self):
        super()._check_parameters()
        if (self.masses is not self._validator.masses
                or not np.array_equal(self.masses, self._mass_snapshot)):
            raise ValueError("BARPMASHDynamics parameters are immutable")

    def _evaluate(self, q):
        raw = self.potential.evaluate(q)
        fields = raw[:2] if isinstance(raw, (tuple, list)) else (
            getattr(raw, "matrix", None), getattr(raw, "gradients", None))
        for name, value in zip(("diabatic matrix", "diabatic gradients"), fields):
            _reject_boolean_entries(value, name, PotentialError)
        value = validate_diabatic_result(raw, q.shape, 2)
        if np.any(value.matrix.imag != 0.0) or np.any(value.gradients.imag != 0.0):
            raise PotentialError("BA-RP-MASH requires real symmetric matrices and gradients; complex SOC is unsupported")
        return value

    def _electronic(self, q):
        value = self._evaluate(q)
        scalar, field, dscalar, dfield = _pauli_components(value.matrix, value.gradients)
        magnitude = np.hypot(field[:, 0], field[:, 2])
        if np.any(2.0 * magnitude <= self.gap_tolerance):
            raise PotentialError("BA-RP-MASH requires a nondegenerate adiabatic gap at every bead")
        axis = field / magnitude[:, None]
        dmag = np.einsum("bi,biac->bac", axis, dfield)
        values = scalar, field, magnitude, axis, dscalar, dfield, dmag
        if not all(np.isfinite(item).all() for item in values):
            raise PotentialError("nonfinite BA-RP-MASH adiabatic quantities")
        return values

    @staticmethod
    def _bead_nac(values):
        """d=<upper|grad lower>, Eq. 21, using the real two-state Pauli frame."""
        _, _, magnitude, axis, _, dfield, _ = values
        # Equivalent to (hx*dhz-hz*dhx)/(2*r**2), without squaring r.
        direction = (axis[:, 0, None, None] * dfield[:, 2]
                     - axis[:, 2, None, None] * dfield[:, 0]) / (2.0 * magnitude[:, None, None])
        if not np.isfinite(direction).all():
            raise PotentialError("nonfinite BA-RP-MASH bead derivative coupling")
        return direction

    def _check(self, state):
        self._check_parameters()
        if not isinstance(state, BARPMASHState):
            raise ValueError("BARPMASHDynamics requires BARPMASHState with a shared adiabatic spin")
        self._validator._check(state.q, state.p)
        _state_clock(state.step, state.time)
        spin = state.spin
        if (not isinstance(spin, np.ndarray) or spin.shape != (3,) or spin.dtype != np.float64
                or not spin.flags.writeable or not spin.flags.c_contiguous or not np.isfinite(spin).all()
                or not np.isclose(spin @ spin, 1.0, rtol=0.0, atol=1e-8)):
            raise ValueError("invalid unit shared adiabatic spin vector")
        active = _two_state_index(state.active_surface, "active_surface")
        if not isinstance(state.sampling, str) or state.sampling not in ("custom", "population", "uniform-sphere"):
            raise ValueError("invalid sampling mode")
        arrays = (state.q, state.p, spin)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")
        if (2 * active - 1) * spin[2] < -1e-8:
            raise ValueError("active surface is inconsistent with the shared adiabatic spin hemisphere")
        self._electronic(state.q)

    def initialize(self, nuclear, active_surface=None, *, sampling="population", rng=None):
        """Copy nuclei and sample an adiabatic population or the full sphere.

        Population importance sampling draws ``Sz = +/-sqrt(U)`` and a
        uniform azimuth.  The |Sz| weight is already absorbed; average active
        indicators without another weight.  Uniform-sphere samples instead
        require ``correlation`` and its operator-pair weights.
        """
        self._check_parameters()
        if not isinstance(nuclear, State):
            raise ValueError("initialize requires a nuclear State")
        self._validator._check(nuclear.q, nuclear.p)
        _state_clock(nuclear.step, nuclear.time)
        if not isinstance(sampling, str) or sampling not in ("population", "uniform-sphere"):
            raise ValueError("sampling must be population or uniform-sphere")
        if sampling == "uniform-sphere" and active_surface is not None:
            raise ValueError("uniform-sphere sampling determines the initial active surface")
        active = 0 if active_surface is None else _two_state_index(active_surface, "active_surface")
        self._electronic(nuclear.q)
        generator = self.rng if rng is None else (rng if isinstance(rng, np.random.Generator)
                                                 else np.random.default_rng(rng))
        snapshot = copy.deepcopy(generator.bit_generator.state)
        try:
            if sampling == "population":
                sz = (2 * active - 1) * np.sqrt(1.0 - generator.random())
            else:
                sz = generator.uniform(-1.0, 1.0)
                active = int(sz > 0.0)
            phi = generator.uniform(0.0, 2.0 * np.pi)
            transverse = np.sqrt(max(0.0, 1.0 - sz * sz))
            spin = np.array([transverse * np.cos(phi), transverse * np.sin(phi), sz])
            state = BARPMASHState(nuclear.q, nuclear.p, nuclear.step, nuclear.time, spin, active, sampling)
            self._check(state)
        except BaseException:
            generator.bit_generator.state = snapshot
            raise
        return state

    def _rhs(self, t, y, active):
        q, p, spin = self._unpack(y)
        values = self._electronic(q)  # One provider batch for all beads.
        velocity = p / self.masses[None, :, None]
        coupling = float(np.sum(velocity * self._bead_nac(values)) / self.nbeads)
        spring = ((self.nbeads / self.beta) ** 2 * self.masses[None, :, None]
                  * (2*q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0)))
        force = -values[4] - (2 * active - 1) * values[6] - spring
        field = np.array([0.0, 2.0 * coupling, 2.0 * float(np.mean(values[2]))])
        return np.concatenate((velocity.ravel(), force.ravel(), np.cross(field, spin)))

    def _event_value(self, y, active):
        q, p, spin = self._unpack(y)
        sign = 2 * active - 1
        value = sign * float(spin[2])
        if abs(value) <= 64.0 * np.finfo(float).eps:
            direction = self._bead_nac(self._electronic(q))
            coupling = float(np.sum(p * direction / self.masses[None, :, None]) / self.nbeads)
            rate = -2.0 * sign * coupling * spin[0]
            if rate >= 0.0:
                return 64.0 * np.finfo(float).eps
        return value

    def _impulse(self, q, p, source, event_time):
        values = self._electronic(q)
        direction = self._bead_nac(values)
        scale = float(np.max(np.abs(direction)))
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("equator crossing has no finite nonzero bead derivative coupling")
        direction = direction / scale
        direction /= np.linalg.norm(direction)
        gap = float(2.0 * (1 - 2 * source) * np.sum(values[2]))
        return _rescale_ring_momentum(p, direction, self.masses, gap, source, event_time)

    def correlation(self, initial, final, initial_operator, final_operator, *, basis="adiabatic"):
        """One full-sphere contribution for shared-adiabatic operators, Eq. 29.

        Operators are Hermitian 2x2 matrices in (upper, lower) order.  Average
        contributions over independent uniform-sphere samples.  Nuclear
        initial-distribution factors remain the caller's responsibility.
        Centroid-transformed diabatic observables are not implemented here.
        """
        self._check(initial)
        self._check(final)
        if not isinstance(basis, str) or basis != "adiabatic":
            raise ValueError("BA-RP-MASH correlation supports only shared adiabatic operators")
        if initial.sampling != "uniform-sphere":
            raise ValueError("correlation requires uniform-sphere initial sampling")
        pieces = []
        for state, operator in ((initial, initial_operator), (final, final_operator)):
            _reject_boolean_entries(operator, "correlation operator")
            raw = np.asarray(operator)
            if (raw.shape != (2, 2) or raw.dtype.kind not in "fciu" or not np.isfinite(raw).all()
                    or not np.allclose(raw, raw.conj().T, rtol=0.0, atol=1e-12)):
                raise ValueError("correlation operators must be finite Hermitian 2x2 matrices")
            scalar = 0.5 * float(np.trace(raw).real)
            vector = np.array([raw[0, 1].real, -raw[0, 1].imag,
                               0.5 * (raw[0, 0].real - raw[1, 1].real)])
            population = scalar + vector[2] * (2 * state.active_surface - 1)
            coherence = float(vector[:2] @ state.spin[:2])
            pieces.append((population, coherence))
        p0, c0 = pieces[0]
        p1, c1 = pieces[1]
        return float(2.0 * (2.0 * abs(initial.spin[2]) * p0 * p1
                            + 2.0 * (p0*c1 + c0*p1) + 3.0 * c0*c1))
