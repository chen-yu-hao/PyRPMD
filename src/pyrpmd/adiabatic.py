"""On-the-fly electronic data and local-basis nonadiabatic propagation.

The provider boundary is adiabatic energies, positive nuclear gradients,
derivative couplings and overlaps, all in atomic units.  No global diabatic
matrix or finite differences of a quantum-chemistry wavefunction are needed.
An overlap is O_ij = <old_i|new_j>; it transports coefficients as O.conj().T @ c.
"""
from dataclasses import dataclass
import copy

import numpy as np
from scipy.linalg import expm, logm

from .dynamics import CayleyIntegrator, State
from .nonadiabatic import DiabaticResult, NRPMDState, _mint, _positive_integer
from .potentials.base import PotentialError, coordinates
from .potentials.python import unit_factors
from .surface_hopping import HopResult, _surface


def _aligned_overlap_log(overlap):
    """Principal log after diagonal-phase alignment, with exact SU(2) algebra.

    A unitary two-state overlap with positive diagonal entries has determinant
    one. Its logarithm is angle/sin(angle) times its anti-Hermitian part.
    This avoids a general Schur decomposition at every two-state step; larger
    electronic spaces retain the general matrix logarithm.
    """
    if overlap.shape != (2, 2):
        return logm(overlap)
    antihermitian = .5*(overlap-overlap.conj().T)
    sine = float(np.sqrt(max(0., -.5*np.trace(antihermitian@antihermitian).real)))
    cosine = float(.5*np.trace(overlap).real)
    factor = np.arctan2(sine, cosine)/sine if sine > 0 else 1.
    return factor*antihermitian


def _hermitian_propagator(hamiltonian, duration):
    """Compute exp(-i duration H); the two-state branch is algebraically exact."""
    if hamiltonian.shape != (2, 2):
        return expm(-1j*duration*hamiltonian)
    mean = float(.5*np.trace(hamiltonian).real)
    centered = hamiltonian-mean*np.eye(2)
    radius = float(np.hypot(centered[0, 0].real, abs(centered[0, 1])))
    angle = duration*radius
    return np.exp(-1j*duration*mean)*(np.cos(angle)*np.eye(2)
                                    -1j*duration*np.sinc(angle/np.pi)*centered)


@dataclass(frozen=True)
class AdiabaticResult:
    """E[B,S], +grad E[B,S,N,3], d_ij=<i|grad j>[B,S,S,N,3].

    ``overlaps[B,S,S]`` connects the explicitly supplied previous frame to
    this frame. ``context`` is an opaque process-local wavefunction handle
    returned to the provider on the next call; the engine never interprets it.
    """
    energies: np.ndarray
    gradients: np.ndarray
    couplings: np.ndarray | None = None
    overlaps: np.ndarray | None = None
    status: np.ndarray | None = None
    context: object = None


def _array(value, shape, label, *, real=False):
    raw = np.asarray(value)
    kinds = "fiu" if real else "fciu"
    if raw.shape != shape or raw.dtype.kind not in kinds or not np.isfinite(raw).all():
        raise PotentialError(f"{label} must be finite {'real ' if real else ''}{list(shape)}")
    result = np.array(raw, dtype=float if real or not np.iscomplexobj(raw) else complex,
                      order="C", copy=True)
    result.flags.writeable = False
    return result


def validate_adiabatic_result(result, shape, nstates=None):
    """Detach and validate electronic data without hiding failed frames."""
    if isinstance(result, dict):
        try:
            result = AdiabaticResult(**result)
        except TypeError as exc:
            raise PotentialError("invalid adiabatic result fields") from exc
    if not isinstance(result, AdiabaticResult):
        raise PotentialError("adiabatic provider must return AdiabaticResult or a field mapping")
    energy = np.asarray(result.energies)
    if len(shape) != 3 or shape[-1] != 3 or min(shape) < 1:
        raise PotentialError("coordinates must be nonempty [B,N,3]")
    if energy.ndim != 2 or energy.shape[0] != shape[0] or energy.shape[1] < 1:
        raise PotentialError("adiabatic energies must have shape [B,S]")
    b, s = energy.shape
    if nstates is not None and s != _positive_integer(nstates, "nstates"):
        raise PotentialError("adiabatic nstates mismatch")
    status = np.zeros(b, dtype=int) if result.status is None else np.asarray(result.status)
    if status.shape != (b,) or status.dtype.kind not in "iu" or np.any(status < 0):
        raise PotentialError("adiabatic status must be nonnegative integers [B]")
    if np.any(status):
        raise PotentialError(f"adiabatic PES frames failed: {np.flatnonzero(status).tolist()}")
    energy = _array(energy, (b, s), "adiabatic energies", real=True)
    gradient = _array(result.gradients, (b, s, *shape[1:]), "adiabatic gradients", real=True)
    coupling = None
    if result.couplings is not None:
        coupling = _array(result.couplings, (b, s, s, *shape[1:]), "derivative couplings")
        if not np.allclose(coupling, -coupling.swapaxes(1, 2).conj(), rtol=0., atol=1e-10):
            raise PotentialError("derivative couplings must be anti-Hermitian in electronic indices")
    overlap = None if result.overlaps is None else _array(result.overlaps, (b, s, s), "overlaps")
    status = np.array(status, copy=True)
    status.flags.writeable = False
    return AdiabaticResult(energy, gradient, coupling, overlap, status, result.context)


@dataclass(frozen=True)
class ElectronicFrame:
    """Owned geometry and immutable electronic arrays in one local basis."""
    q: np.ndarray
    result: AdiabaticResult

    def __post_init__(self):
        q = np.array(coordinates(self.q), copy=True)
        result = validate_adiabatic_result(self.result, q.shape)
        q.flags.writeable = False
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "result", result)


class AdiabaticPythonPotential:
    """Adapt a quantum-chemistry callback with an explicit previous-frame contract.

    ``obj.evaluate(q, previous_q=None, previous_context=None)`` (or a callable)
    returns :class:`AdiabaticResult`. Both q arrays are in provider length
    units. Energies, gradients and NACs are converted independently; overlaps
    and context are dimensionless/opaque. Only set ``continuous_gauge=True``
    when the code supplies consistently phased states between evaluations.
    Otherwise an actual overlap matrix is mandatory for a propagation step.
    """
    def __init__(self, obj, *, units=None, continuous_gauge=False):
        self.obj = obj
        self.fn = getattr(obj, "evaluate", obj)
        if not callable(self.fn):
            raise TypeError("adiabatic provider must be callable or have evaluate")
        if not isinstance(continuous_gauge, (bool, np.bool_)):
            raise ValueError("continuous_gauge must be boolean")
        self.length_factor, self.energy_factor = unit_factors(units)
        self.metadata = {"provider": "adiabatic_python", "execution_device": "cpu",
                         "thread_safety": "context_serial", "continuous_gauge": bool(continuous_gauge)}

    def evaluate(self, q, *, previous=None):
        q = coordinates(q)
        if previous is not None and (not isinstance(previous, ElectronicFrame) or previous.q.shape != q.shape):
            raise ValueError("previous electronic frame must have the same coordinate shape")
        previous_q = None if previous is None else previous.q / self.length_factor
        context = None if previous is None else previous.result.context
        raw = self.fn(q / self.length_factor, previous_q=previous_q, previous_context=context)
        result = validate_adiabatic_result(raw, q.shape)
        return AdiabaticResult(result.energies*self.energy_factor,
                               result.gradients*(self.energy_factor/self.length_factor),
                               None if result.couplings is None else result.couplings/self.length_factor,
                               result.overlaps, result.status, result.context)

    def close(self):
        close = getattr(self.obj, "close", None)
        if callable(close):
            close()


def _frame(potential, q, nstates, previous=None, *, need_couplings=True):
    result = validate_adiabatic_result(potential.evaluate(q, previous=previous), q.shape, nstates)
    if need_couplings and result.couplings is None:
        raise PotentialError("this propagator requires nuclear derivative coupling vectors")
    return ElectronicFrame(q, result)


def electronic_transport(old, new, *, continuous_gauge=False, overlap_tolerance=1e-6):
    """Return a checked unitary old-to-new basis overlap for every frame.

    The polar factor removes only the explicitly tolerated retained-subspace
    leakage. A singular or poorly overlapping electronic subspace is rejected.
    NAC-only transport is a second-order endpoint connection approximation and
    requires an explicitly declared continuous gauge.
    """
    if (isinstance(overlap_tolerance, (bool, np.bool_)) or not np.isfinite(overlap_tolerance)
            or not 0 < overlap_tolerance < 1):
        raise ValueError("overlap_tolerance must be finite and between zero and one")
    if old.q.shape != new.q.shape or old.result.energies.shape != new.result.energies.shape:
        raise ValueError("transport frame shapes differ")
    overlap = new.result.overlaps
    if overlap is None:
        if not continuous_gauge or old.result.couplings is None or new.result.couplings is None:
            raise PotentialError("overlap transport requires overlaps or continuous-gauge NACs")
        connection = np.einsum("bijac,bac->bij", .5*(old.result.couplings+new.result.couplings),
                               new.q-old.q)
        return np.asarray([expm(x) for x in connection])
    u, singular, vh = np.linalg.svd(overlap)
    if np.max(np.abs(singular-1.)) > overlap_tolerance:
        raise PotentialError("electronic overlaps lose retained-subspace norm; reduce timestep or include more states")
    return u @ vh


def adiabatic_gradient_matrix(result):
    """Positive operator derivative in the local adiabatic basis.

    Off-diagonal elements are (E_j-E_i) d_ij, and diagonal elements are
    grad E_i. Keeping these off-diagonal forces is essential for NRPMD.
    """
    if result.couplings is None:
        raise PotentialError("NRPMD forces require derivative coupling vectors")
    gaps = result.energies[:, None, :]-result.energies[:, :, None]
    gradient = gaps[..., None, None]*result.couplings
    diagonal = np.arange(result.energies.shape[1])
    gradient[:, diagonal, diagonal] = result.gradients
    return gradient


@dataclass
class QDNRPMDState(NRPMDState):
    """Mapping variables expressed in each bead's current electronic basis."""
    electronic_frame: ElectronicFrame | None = None

    def copy(self):
        return QDNRPMDState(self.q, self.p, self.step, self.time, self.mapping_q,
                            self.mapping_p, self.electronic_frame)


class QDNRPMDDynamics:
    """Local-basis MInt/free-ring/MInt NRPMD with overlap transport.

    For a complete electronic subspace with exact overlaps this is precisely
    the existing diabatic MInt integrator in changing bases. Truncation and
    approximate NAC-only transport require their own convergence studies.
    This implements focused-state dynamics, not SEO thermal preparation.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates, *, native=True,
                 threads=1, integrator="cayley-bcb", overlap_tolerance=1e-6):
        self.potential = potential
        self.nstates = _positive_integer(nstates, "nstates")
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt, native=native,
                                          threads=threads, scheme=integrator)
        if not self.integrator.dt > 0:
            raise ValueError("QDNRPMD dt must be positive")
        self.masses, self.beta = self.integrator.masses, self.integrator.beta
        self.dt, self.nbeads = self.integrator.dt, self.integrator.nbeads
        if (isinstance(overlap_tolerance, (bool, np.bool_)) or not np.isfinite(overlap_tolerance)
                or not 0 < overlap_tolerance < 1):
            raise ValueError("overlap_tolerance must be between zero and one")
        self.overlap_tolerance = float(overlap_tolerance)
        self.continuous_gauge = getattr(potential, "metadata", {}).get("continuous_gauge") is True

    def initialize(self, nuclear, initial_state=0, *, seed=0):
        self.integrator._check(nuclear.q, nuclear.p)
        mapping = NRPMDState.from_nuclear(nuclear, self.nstates, initial_state, seed)
        frame = _frame(self.potential, nuclear.q, self.nstates)
        return QDNRPMDState(mapping.q, mapping.p, mapping.step, mapping.time,
                            mapping.mapping_q, mapping.mapping_p, frame)

    def _check(self, state):
        if not isinstance(state, QDNRPMDState) or state.electronic_frame is None:
            raise ValueError("QDNRPMDDynamics requires an initialized QDNRPMDState")
        self.integrator._check(state.q, state.p)
        if (self.dt != self.integrator.dt or self.beta != self.integrator.beta
                or self.nbeads != self.integrator.nbeads):
            raise ValueError("QDNRPMDDynamics parameters are immutable")
        if not np.array_equal(state.q, state.electronic_frame.q):
            raise ValueError("nuclear coordinates changed without updating the electronic frame")
        if state.electronic_frame.result.energies.shape != (self.nbeads, self.nstates):
            raise ValueError("electronic frame nstates mismatch")
        for a in (state.mapping_q, state.mapping_p):
            if (a.shape != (self.nbeads, self.nstates) or a.dtype != np.float64
                    or not a.flags.writeable or not a.flags.c_contiguous or not np.isfinite(a).all()):
                raise ValueError("invalid mapping variables")
        arrays = (state.q, state.p, state.mapping_q, state.mapping_p)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")
        if (isinstance(state.step, (bool, np.bool_)) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid state clock")

    @staticmethod
    def _local_result(frame):
        result = frame.result
        matrix = np.zeros((len(result.energies), result.energies.shape[1], result.energies.shape[1]))
        diagonal = np.arange(result.energies.shape[1])
        matrix[:, diagonal, diagonal] = result.energies
        return DiabaticResult(matrix, adiabatic_gradient_matrix(result))

    def step(self, state):
        self._check(state)
        trial = state.copy()
        _mint(trial, self._local_result(state.electronic_frame), .5*self.dt)
        self.integrator.free_step(trial.q, trial.p)
        new = _frame(self.potential, trial.q, self.nstates, state.electronic_frame)
        overlap = electronic_transport(state.electronic_frame, new, continuous_gauge=self.continuous_gauge,
                                       overlap_tolerance=self.overlap_tolerance)
        z = np.einsum("bij,bi->bj", overlap.conj(), trial.mapping_q+1j*trial.mapping_p)
        trial.mapping_q[:], trial.mapping_p[:] = z.real, z.imag
        trial.electronic_frame = new
        _mint(trial, self._local_result(new), .5*self.dt)
        trial.step += 1
        trial.time += self.dt
        self._check(trial)
        for key in ("q", "p", "mapping_q", "mapping_p"):
            getattr(state, key)[:] = getattr(trial, key)
        state.step, state.time, state.electronic_frame = trial.step, trial.time, new
        return state

    def populations(self, state):
        """Signed, bead-averaged instantaneous adiabatic MMST estimator."""
        self._check(state)
        return state.populations()

    def energy(self, state):
        self._check(state)
        population = .5*(state.mapping_q**2+state.mapping_p**2-1.)
        kinetic = np.sum(state.p**2/(2*self.masses[None, :, None]))
        dq = state.q-np.roll(state.q, -1, axis=0)
        spring = .5*(self.nbeads/self.beta)**2*np.sum(self.masses[None, :, None]*dq*dq)
        return float(np.sum(population*state.electronic_frame.result.energies)+kinetic+spring)


@dataclass
class AdiabaticRPSHState(State):
    """Normalized local-adiabatic centroid coefficients and bead/centroid frames."""
    coefficients: np.ndarray | None = None
    active_surface: int = 0
    bead_frame: ElectronicFrame | None = None
    centroid_frame: ElectronicFrame | None = None

    def __post_init__(self):
        super().__post_init__()
        c = np.asarray(self.coefficients)
        if (c.ndim != 1 or c.size < 1 or c.dtype.kind not in "fciu" or not np.isfinite(c).all()
                or not np.isclose(np.vdot(c, c).real, 1., rtol=0, atol=1e-10)):
            raise ValueError("coefficients must be a finite normalized electronic vector")
        self.coefficients = np.array(c, dtype=complex, order="C", copy=True)
        self.active_surface = _surface(self.active_surface, c.size)

    def copy(self):
        return AdiabaticRPSHState(self.q, self.p, self.step, self.time, self.coefficients,
                                  self.active_surface, self.bead_frame, self.centroid_frame)


class AdiabaticRPSHDynamics:
    """Centroid FSSH/RPSH driven directly by quantum-chemistry electronic data.

    All beads occupy one energy-ordered surface. Electronic amplitudes use
    overlap-derived average time couplings and interpolated energies. Hops
    rescale the centroid momentum along the NAC and conserve the full ring
    Hamiltonian. Real electronic surfaces/NACs are required; complex-SOC
    momentum rescaling is not silently approximated.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates, *, native=True,
                 threads=1, integrator="cayley-bcb", seed=0, electronic_substeps=8,
                 frustrated_hops="keep", overlap_tolerance=1e-6):
        self.potential = potential
        self.nstates = _positive_integer(nstates, "nstates")
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt, native=native,
                                          threads=threads, scheme=integrator)
        self.dt, self.beta = self.integrator.dt, self.integrator.beta
        self.nbeads, self.masses = self.integrator.nbeads, self.integrator.masses
        if self.dt <= 0:
            raise ValueError("adiabatic RPSH dt must be positive")
        self.electronic_substeps = _positive_integer(electronic_substeps, "electronic_substeps")
        if frustrated_hops not in ("keep", "reverse"):
            raise ValueError("frustrated_hops must be keep or reverse")
        self.frustrated_hops = frustrated_hops
        if (isinstance(overlap_tolerance, (bool, np.bool_)) or not np.isfinite(overlap_tolerance)
                or not 0 < overlap_tolerance < 1):
            raise ValueError("overlap_tolerance must be between zero and one")
        self.overlap_tolerance = float(overlap_tolerance)
        self.continuous_gauge = getattr(potential, "metadata", {}).get("continuous_gauge") is True
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.last_hop = None
        self.last_probabilities = tuple(0. for _ in range(self.nstates))

    def _new_frame(self, q, previous=None):
        frame = _frame(self.potential, q, self.nstates, previous)
        if np.max(np.abs(np.imag(frame.result.couplings))) > 1e-12:
            raise PotentialError("adiabatic RPSH/MASH requires real NACs; complex SOC rescaling is unsupported")
        if self.nstates > 1 and np.any(np.diff(frame.result.energies, axis=1) <= 1e-12):
            raise PotentialError("RPSH/MASH requires nondegenerate energy-ordered adiabatic states")
        return frame

    def initialize(self, nuclear, active_surface=0, *, adiabatic_coefficients=None):
        self.integrator._check(nuclear.q, nuclear.p)
        active = _surface(active_surface, self.nstates)
        c = np.eye(self.nstates, dtype=complex)[active] if adiabatic_coefficients is None else adiabatic_coefficients
        bead = self._new_frame(nuclear.q)
        centroid = bead if self.nbeads == 1 else self._new_frame(nuclear.q.mean(axis=0, keepdims=True))
        state = AdiabaticRPSHState(nuclear.q, nuclear.p, nuclear.step, nuclear.time, c, active, bead, centroid)
        self._check(state)
        return state

    def _check(self, state):
        if not isinstance(state, AdiabaticRPSHState):
            raise ValueError("adiabatic RPSH requires AdiabaticRPSHState")
        self.integrator._check(state.q, state.p)
        if (self.dt != self.integrator.dt or self.beta != self.integrator.beta
                or self.nbeads != self.integrator.nbeads):
            raise ValueError("adiabatic dynamics parameters are immutable")
        if (state.bead_frame is None or state.centroid_frame is None
                or not np.array_equal(state.q, state.bead_frame.q)
                or not np.array_equal(state.q.mean(axis=0, keepdims=True), state.centroid_frame.q)):
            raise ValueError("nuclear coordinates and electronic frames differ")
        c = state.coefficients
        if (c.shape != (self.nstates,) or c.dtype != np.complex128 or not c.flags.writeable
                or not c.flags.c_contiguous or not np.isfinite(c).all()
                or not np.isclose(np.vdot(c, c).real, 1., rtol=0, atol=1e-10)):
            raise ValueError("invalid electronic coefficients")
        if np.shares_memory(c, state.q) or np.shares_memory(c, state.p):
            raise ValueError("state arrays must not share memory")
        _surface(state.active_surface, self.nstates)
        if (isinstance(state.step, (bool, np.bool_)) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or not np.isfinite(state.time)):
            raise ValueError("invalid state clock")

    def populations(self, state):
        self._check(state)
        return abs(state.coefficients)**2

    def energy(self, state):
        self._check(state)
        kinetic = np.sum(state.p**2/(2*self.masses[None, :, None]))
        dq = state.q-np.roll(state.q, -1, axis=0)
        spring = .5*(self.nbeads/self.beta)**2*np.sum(self.masses[None, :, None]*dq*dq)
        return float(state.bead_frame.result.energies[:, state.active_surface].sum()+kinetic+spring)

    def _electronic_step(self, c, old, new, active):
        overlap = electronic_transport(old, new, continuous_gauge=self.continuous_gauge,
                                       overlap_tolerance=self.overlap_tolerance)[0]
        # Align independent eigenvector phases before taking the matrix log.
        # Undo that gauge in the final coefficients; no probability is changed.
        diagonal = np.diag(overlap)
        if np.any(abs(diagonal) < 1e-8):
            raise PotentialError("electronic states rotate too far in one step; reduce timestep")
        phase = diagonal.conj()/abs(diagonal)
        aligned = overlap*phase[None, :]
        connection = _aligned_overlap_log(aligned)/self.dt
        if np.max(np.abs(connection+connection.conj().T)) > 1e-8:
            raise PotentialError("overlap logarithm is not anti-Hermitian")
        connection = .5*(connection-connection.conj().T)
        initial_population = float(abs(c[active])**2)
        if initial_population <= np.finfo(float).eps:
            raise ValueError("active surface has zero population")
        z = c.copy()
        transfer = np.zeros(self.nstates)
        h = self.dt/self.electronic_substeps
        surfaces = [int(np.argmax(abs(z)**2))]
        for k in range(self.electronic_substeps):
            t = (k+.5)/self.electronic_substeps
            energies = (1-t)*old.result.energies[0]+t*new.result.energies[0]
            hamiltonian = np.diag(energies)-1j*connection
            half = _hermitian_propagator(hamiltonian, .5*h)
            midpoint = half@z
            transfer += 2*h*np.real(midpoint[active].conj()*midpoint*connection[active])
            z = half@midpoint
            surfaces.append(int(np.argmax(abs(z)**2)))
        probability = np.maximum(transfer/initial_population, 0.)
        probability[active] = 0.
        return phase*z, probability, surfaces

    def _hop(self, state, target):
        source = state.active_surface
        target = _surface(target, self.nstates)
        if source == target:
            return HopResult(source, target, False, False, "ring", 0.)
        direction = state.centroid_frame.result.couplings[0, source, target].real.copy()
        norm = np.linalg.norm(direction)
        if norm <= np.finfo(float).tiny:
            return HopResult(source, target, False, True, "ring", 0.)
        direction /= norm
        if direction.flat[np.argmax(abs(direction))] < 0:
            direction *= -1
        gaps = state.bead_frame.result.energies[:, target]-state.bead_frame.result.energies[:, source]
        ring_gap = float(gaps.sum())
        a = .5*float(np.sum(direction**2/self.masses[:, None]))
        b = float(np.sum(state.p.mean(axis=0)*direction/self.masses[:, None]))
        discriminant = b*b-4*a*ring_gap/self.nbeads
        if discriminant < 0:
            if self.frustrated_hops == "reverse":
                state.p -= (b/a)*direction[None]
            return HopResult(source, target, False, True, "ring", 0.)
        root = np.sqrt(discriminant)
        denominator = b+np.copysign(root, b)
        shift = -2*(ring_gap/self.nbeads)/denominator if denominator != 0 else 0.
        if b == 0 and root != 0:
            shift = root/(2*a)
        before = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        state.p += shift*direction[None]
        state.active_surface = target
        after = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        return HopResult(source, target, True, False, "ring", after-before+ring_gap)

    def attempt_hop(self, state, target):
        self._check(state)
        trial = state.copy()
        hop = self._hop(trial, target)
        self._check(trial)
        state.p[:], state.active_surface = trial.p, trial.active_surface
        self.last_hop = hop
        return hop

    def _select_hop(self, trial, probabilities, surfaces):
        total = float(probabilities.sum())
        if not np.isfinite(probabilities).all() or total > 1.+1e-12:
            raise ValueError("FSSH probabilities exceed one; reduce timestep")
        if total > 1.:
            probabilities = probabilities/total
        if total <= 0:
            return None
        target = int(np.searchsorted(np.cumsum(probabilities), self.rng.random(), side="right"))
        return self._hop(trial, target) if target < self.nstates else None

    def step(self, state):
        self._check(state)
        trial = state.copy()
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            trial.p -= .5*self.dt*trial.bead_frame.result.gradients[:, trial.active_surface]
            self.integrator.free_step(trial.q, trial.p)
            trial.bead_frame = self._new_frame(trial.q, state.bead_frame)
            trial.centroid_frame = (trial.bead_frame if self.nbeads == 1 else self._new_frame(
                trial.q.mean(axis=0, keepdims=True), state.centroid_frame))
            trial.p -= .5*self.dt*trial.bead_frame.result.gradients[:, trial.active_surface]
            c, probabilities, surfaces = self._electronic_step(
                state.coefficients, state.centroid_frame, trial.centroid_frame, trial.active_surface)
            trial.coefficients[:] = c
            hop = self._select_hop(trial, probabilities, surfaces)
            trial.step += 1
            trial.time += self.dt
            self._check(trial)
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        state.q[:], state.p[:], state.coefficients[:] = trial.q, trial.p, trial.coefficients
        state.active_surface = trial.active_surface
        state.bead_frame, state.centroid_frame = trial.bead_frame, trial.centroid_frame
        state.step, state.time = trial.step, trial.time
        self.last_hop, self.last_probabilities = hop, tuple(float(x) for x in probabilities)
        return state
