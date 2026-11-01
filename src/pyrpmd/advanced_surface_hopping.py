"""Explicit BA/DM ring-polymer hopping, DBOC and complex-Hermitian hopping.

The methods and their limits are specified in
``docs/nonadiabatic-hopping-extensions.md``.  In particular RPSH-DM means the
bead-wavefunction/centroid-density construction of Shakya et al. (2023), not
an arbitrary mixed-state propagation.  All beads share one active surface.

The matrix provider must supply a position-independent orthonormal diabatic
basis, Hermitian matrices, and their *positive* Cartesian derivatives in atomic
units.  Degenerate adiabatic states are outside this implementation's domain.
"""
from dataclasses import dataclass
import copy

import numpy as np

from .dynamics import CayleyIntegrator, State
from .nonadiabatic import _positive_integer, validate_diabatic_result
from .potentials.base import PotentialError
from .surface_hopping import HopResult, RPSHState, _surface


def _positive(value, name):
    raw = np.asarray(value)
    if (raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw)
            or raw <= 0):
        raise ValueError(f"{name} must be positive and finite")
    result = float(raw)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a representable positive finite float")
    return result


def _amplitudes(value, ndim):
    raw = np.asarray(value)
    if (raw.ndim != ndim or not raw.size or raw.dtype.kind not in "fciu"
            or not np.isfinite(raw).all()):
        raise ValueError("electronic amplitudes must be finite and normalized on their last axis")
    result = np.array(raw, dtype=np.complex128, order="C", copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.sum(np.abs(result)**2, axis=-1)
    if not np.allclose(norms, 1., rtol=0., atol=1e-10):
        raise ValueError("electronic amplitudes must be finite and normalized on their last axis")
    return result


@dataclass
class RPSHDMState(State):
    """One fixed-diabatic normalized wavefunction [P,S] for each bead."""
    coefficients: np.ndarray | None = None
    active_surface: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.coefficients = _amplitudes(self.coefficients, 2)
        if self.coefficients.shape[0] != self.q.shape[0]:
            raise ValueError("DM coefficients must have shape [P,S]")
        self.active_surface = _surface(self.active_surface, self.coefficients.shape[1])

    def copy(self):
        return RPSHDMState(self.q, self.p, self.step, self.time,
                           self.coefficients, self.active_surface)

    def to_dict(self):
        return {"q": self.q.tolist(), "p": self.p.tolist(), "step": int(self.step),
                "time": float(self.time), "coefficients_real": self.coefficients.real.tolist(),
                "coefficients_imag": self.coefficients.imag.tolist(),
                "active_surface": int(self.active_surface)}

    @classmethod
    def from_dict(cls, value):
        return cls(value["q"], value["p"], value["step"], value["time"],
                   np.asarray(value["coefficients_real"])+1j*np.asarray(value["coefficients_imag"]),
                   value["active_surface"])


@dataclass
class RPSHBAState(State):
    """Common adiabatic amplitudes [S], with transported bead frames [P,S,S].

    The frames are part of the restart state: dropping them loses the relative
    sign convention required when averaging derivative couplings across beads.
    """
    coefficients: np.ndarray | None = None
    active_surface: int = 0
    basis_vectors: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        self.coefficients = _amplitudes(self.coefficients, 1)
        nstates = self.coefficients.size
        self.active_surface = _surface(self.active_surface, nstates)
        frames = np.asarray(self.basis_vectors)
        if (frames.shape != (self.q.shape[0], nstates, nstates)
                or frames.dtype.kind not in "fciu" or not np.isfinite(frames).all()):
            raise ValueError("BA basis_vectors must be unitary frames [P,S,S]")
        self.basis_vectors = np.array(frames, dtype=np.complex128, order="C", copy=True)
        if not np.allclose(self.basis_vectors.conj().swapaxes(-1, -2)@self.basis_vectors,
                           np.eye(nstates), rtol=0., atol=1e-10):
            raise ValueError("BA basis_vectors must be unitary frames [P,S,S]")

    def copy(self):
        return RPSHBAState(self.q, self.p, self.step, self.time,
                           self.coefficients, self.active_surface, self.basis_vectors)

    def to_dict(self):
        return {"q": self.q.tolist(), "p": self.p.tolist(), "step": int(self.step),
                "time": float(self.time), "coefficients_real": self.coefficients.real.tolist(),
                "coefficients_imag": self.coefficients.imag.tolist(),
                "active_surface": int(self.active_surface),
                "basis_vectors_real": self.basis_vectors.real.tolist(),
                "basis_vectors_imag": self.basis_vectors.imag.tolist()}

    @classmethod
    def from_dict(cls, value):
        return cls(value["q"], value["p"], value["step"], value["time"],
                   np.asarray(value["coefficients_real"])+1j*np.asarray(value["coefficients_imag"]),
                   value["active_surface"], np.asarray(value["basis_vectors_real"])
                   +1j*np.asarray(value["basis_vectors_imag"]))


@dataclass(frozen=True)
class AdiabaticData:
    """Adiabatic energies/gradients include DBOC if enabled; NACs never do.

    Arrays have respectively [P,S], [P,S,S], [P,S,N,3], [P,S,S,N,3],
    and [P,S] shapes. ``vectors`` stores eigenvectors as columns.
    """
    energies: np.ndarray
    vectors: np.ndarray
    gradients: np.ndarray
    couplings: np.ndarray
    dboc: np.ndarray


def complex_rescaling_direction(coupling, *, tolerance=1e-12):
    """Miao--Bellonzi--Subotnik (2019), Method #2, as a normalized real ray.

    Maximize ||Re(exp(i eta) d)|| over eta.  An overall electronic phase
    cannot change the returned direction.  Equal largest singular values
    leave no unique ray and raise instead of choosing a gauge-dependent one.
    The largest Cartesian component fixes the remaining overall sign.
    """
    raw = np.asarray(coupling)
    tolerance = _positive(tolerance, "tolerance")
    if raw.dtype.kind not in "fciu" or not raw.size or not np.isfinite(raw).all():
        raise ValueError("coupling must be a finite numerical array")
    z = np.asarray(raw, dtype=complex).ravel()
    scale = float(np.max(np.abs(z)))
    if scale == 0.:
        return np.zeros(raw.shape, dtype=float)
    z = z/scale
    a, b = z.real, z.imag
    gram = np.array([[a@a, a@b], [a@b, b@b]])
    eigenvalues, vectors = np.linalg.eigh(gram)
    if eigenvalues[1]-eigenvalues[0] <= tolerance*eigenvalues[1]:
        raise PotentialError("complex NAC has no unique maximum-real-norm rescaling direction")
    direction = vectors[0, 1]*a+vectors[1, 1]*b
    direction /= np.linalg.norm(direction)
    if direction[np.argmax(np.abs(direction))] < 0.:
        direction *= -1.
    return direction.reshape(raw.shape)


def berry_curvature(couplings, active_surface):
    """Cartesian antisymmetric matrix G with F_i = G_ij v_j (hbar=1).

    ``couplings`` has [S,S,N,3] shape. This is the nondegenerate Abelian
    curvature, 2 Im sum_b d_ab,i d_ba,j, not a degenerate-subspace theory.
    """
    d = np.asarray(couplings)
    if (d.ndim != 4 or d.shape[0] != d.shape[1] or d.shape[-1] != 3
            or d.dtype.kind not in "fciu" or not np.isfinite(d).all()):
        raise ValueError("couplings must have finite shape [S,S,N,3]")
    active = _surface(active_surface, d.shape[0])
    a = d[active].reshape(d.shape[0], -1)
    curvature = -2.*np.imag(a.T@a.conj())
    return .5*(curvature-curvature.T)


class AdvancedRPSHDynamics:
    """Explicit ``method='ca'|'ba'|'dm'`` surface-hopping driver.

    CA/DM accept nondegenerate complex Hermitian matrix providers. BA uses
    real Hamiltonians and a continuously transported real adiabatic gauge.
    DBOC is optional and consistently enters energy, gradients, phases and hop
    energy gaps; its gradient is a central difference of the scalar correction.
    ``rescaling='bead'`` uses all bead NACs, while ``'centroid'`` shifts every
    bead by the same vector. Both conserve the *sum-bead* Hamiltonian at hops.
    ``geometric_force=True`` includes the adiabatic Berry-force ansatz with a
    kinetic-energy-preserving momentum rotation. See the method limitations
    and convergence requirements in the accompanying documentation.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates, *, method="ca",
                 seed=0, native=True, threads=1, integrator="cayley-bcb",
                 electronic_substeps=8, rescaling="bead", frustrated_hops="keep",
                 gap_tolerance=1e-10, dboc=False, dboc_step=1e-4,
                 geometric_force=True, decoherence="none", decoherence_constant=0.1):
        if method not in ("ca", "ba", "dm"):
            raise ValueError("method must be ca, ba or dm")
        self.method = method
        self.potential = potential
        self.nstates = _positive_integer(nstates, "nstates")
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt, threads=threads,
                                          native=native, scheme=integrator)
        self.dt, self.beta, self.nbeads = _positive(dt, "dt"), float(beta), int(nbeads)
        self.masses = self.integrator.masses
        self.integrator_name, self.native = self.integrator.scheme, self.integrator.native
        self.electronic_substeps = _positive_integer(electronic_substeps, "electronic_substeps")
        if rescaling not in ("bead", "centroid"):
            raise ValueError("advanced rescaling must be bead or centroid (both conserve ring energy)")
        if frustrated_hops not in ("keep", "reverse"):
            raise ValueError("frustrated_hops must be keep or reverse")
        self.rescaling, self.frustrated_hops = rescaling, frustrated_hops
        self.gap_tolerance = _positive(gap_tolerance, "gap_tolerance")
        if not isinstance(dboc, (bool, np.bool_)) or not isinstance(geometric_force, (bool, np.bool_)):
            raise ValueError("dboc and geometric_force must be boolean")
        self.dboc, self.geometric_force = bool(dboc), bool(geometric_force)
        self.dboc_step = _positive(dboc_step, "dboc_step") if self.dboc else None
        if decoherence not in ("none", "edc") or (method == "dm" and decoherence != "none"):
            raise ValueError("decoherence must be none or edc; RPSH-DM requires none")
        self.decoherence = decoherence
        self.decoherence_constant = (_positive(decoherence_constant, "decoherence_constant")
                                     if decoherence == "edc" else None)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.last_hop = None
        self.last_probabilities = tuple(0. for _ in range(self.nstates))
        self._parameters = self._parameter_values()

    def _parameter_values(self):
        return (self.method, self.dt, self.beta, self.nbeads, self.nstates,
                self.integrator_name, self.electronic_substeps, self.rescaling,
                self.frustrated_hops, self.gap_tolerance, self.dboc, self.dboc_step,
                self.geometric_force, self.decoherence, self.decoherence_constant)

    def _check_parameters(self):
        if (self._parameter_values() != self._parameters or self.dt != self.integrator.dt
                or self.beta != self.integrator.beta or self.nbeads != self.integrator.nbeads
                or self.integrator_name != self.integrator.scheme):
            raise ValueError("AdvancedRPSHDynamics parameters are immutable")

    def _check(self, state):
        self._check_parameters()
        cls = {"ca": RPSHState, "ba": RPSHBAState, "dm": RPSHDMState}[self.method]
        if not isinstance(state, cls):
            raise ValueError(f"method={self.method} requires {cls.__name__}")
        self.integrator._check(state.q, state.p)
        shape = (self.nbeads, self.nstates) if self.method == "dm" else (self.nstates,)
        z = state.coefficients
        if (not isinstance(z, np.ndarray) or z.shape != shape or z.dtype != np.complex128
                or not z.flags.c_contiguous or not z.flags.writeable or not np.isfinite(z).all()
                or not np.allclose(np.sum(abs(z)**2, axis=-1), 1., atol=1e-10, rtol=0.)):
            raise ValueError("invalid normalized electronic amplitudes")
        _surface(state.active_surface, self.nstates)
        arrays = [state.q, state.p, z]
        if self.method == "ba":
            u = state.basis_vectors
            if (not isinstance(u, np.ndarray) or u.shape != (self.nbeads, self.nstates, self.nstates)
                    or u.dtype != np.complex128 or not u.flags.c_contiguous or not u.flags.writeable
                    or not np.isfinite(u).all() or np.any(abs(u.imag) > 1e-10)
                    or not np.allclose(u.conj().swapaxes(-1, -2)@u, np.eye(self.nstates),
                                       atol=1e-10, rtol=0.)):
                raise ValueError("invalid transported real BA basis_vectors")
            arrays.append(u)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")
        clock = np.asarray(state.time)
        if (isinstance(state.step, (bool, np.bool_)) or not isinstance(state.step, (int, np.integer))
                or state.step < 0 or clock.ndim != 0 or clock.dtype.kind not in "fiu"
                or not np.isfinite(clock)):
            raise ValueError("invalid state clock")

    def _eigensystem(self, matrix):
        """Separate seam permits independent eigenvector-gauge tests."""
        return np.linalg.eigh(matrix)

    @staticmethod
    def _align(vectors, reference):
        overlap = np.einsum("bsi,bsi->bi", reference.conj(), vectors)
        if np.any(abs(overlap) <= 1e-10):
            raise PotentialError("adiabatic frame overlap vanishes; refine path or use a different method")
        return vectors*(overlap.conj()/abs(overlap))[:, None, :]

    def _bare(self, q, reference=None):
        r = validate_diabatic_result(self.potential.evaluate(q), q.shape, self.nstates)
        if self.method == "ba" and (np.any(np.imag(r.matrix) != 0)
                                     or np.any(np.imag(r.gradients) != 0)):
            raise PotentialError("RPSH-BA is implemented for real symmetric matrices; use CA or DM for complex SOC")
        e, u = self._eigensystem(r.matrix)
        u = np.asarray(u, dtype=complex)
        if reference is not None:
            u = self._align(u, reference)
        gaps = e[:, None, :]-e[:, :, None]
        offdiag = ~np.eye(self.nstates, dtype=bool)
        if np.any(abs(gaps[:, offdiag]) <= self.gap_tolerance):
            raise PotentialError("adiabatic states are degenerate or below gap_tolerance")
        transformed = np.einsum("bsi,bstac,btj->bijac", u.conj(), r.gradients, u, optimize=True)
        gradient = np.einsum("biiac->biac", transformed).real.copy()
        index = np.arange(self.nstates)
        gaps[:, index, index] = 1.
        d = transformed/gaps[..., None, None]
        d[:, index, index] = 0.
        if not np.isfinite(d).all() or not np.isfinite(gradient).all():
            raise PotentialError("nonfinite adiabatic gradients or couplings")
        return e, u, gradient, d

    def _dboc_values(self, couplings):
        return .5*np.sum(abs(couplings)**2/self.masses[None, None, None, :, None], axis=(2, 3, 4))

    def _geometry(self, q, *, gradients=True, reference=None):
        e, u, g, d = self._bare(q, reference)
        correction = np.zeros_like(e)
        if self.dboc:
            correction = self._dboc_values(d)
            e = e+correction
            if gradients:
                for atom in range(q.shape[1]):
                    for coordinate in range(3):
                        plus, minus = q.copy(), q.copy()
                        plus[:, atom, coordinate] += self.dboc_step
                        minus[:, atom, coordinate] -= self.dboc_step
                        if (np.any(plus[:, atom, coordinate] == q[:, atom, coordinate])
                                or np.any(minus[:, atom, coordinate] == q[:, atom, coordinate])):
                            raise ValueError("dboc_step is not representable at the supplied coordinates")
                        dp = self._dboc_values(self._bare(plus)[3])
                        dm = self._dboc_values(self._bare(minus)[3])
                        g[:, :, atom, coordinate] += (dp-dm)/(2*self.dboc_step)
        if not np.isfinite(e).all() or not np.isfinite(g).all():
            raise PotentialError("nonfinite DBOC-corrected energy or gradient")
        return AdiabaticData(e, u, g, d, correction)

    def surface_data(self, q):
        """Evaluate corrected energies, gradients and bare NACs at bead geometries."""
        self._check_parameters()
        raw = np.asarray(q)
        if (raw.ndim != 3 or raw.shape[1:] != (self.masses.size, 3)
                or not raw.shape[0] or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()):
            raise ValueError("q must be a finite real array [P,N,3]")
        return self._geometry(np.array(raw, dtype=float, copy=True))

    def initialize(self, nuclear, active_surface=0, *, adiabatic_coefficients=None):
        """Initialize pure active-state amplitudes, or supplied normalized amplitudes.

        CA amplitudes refer to the centroid basis. BA amplitudes refer to the
        common, centroid-aligned bead state labels. DM accepts [S] (the same
        *local adiabatic* amplitudes at each bead) or explicit [P,S].
        """
        self._check_parameters()
        self.integrator._check(nuclear.q, nuclear.p)
        active = _surface(active_surface, self.nstates)
        c = (np.eye(self.nstates, dtype=complex)[active] if adiabatic_coefficients is None
             else np.asarray(adiabatic_coefficients))
        centroid = self._geometry(nuclear.q.mean(axis=0, keepdims=True), gradients=False)
        # An explicit initial gauge makes a supplied coherent superposition
        # reproducible even when an eigensolver chooses different column phases.
        initial_vectors = centroid.vectors.copy()
        indices = np.argmax(abs(initial_vectors[0]), axis=0)
        pivots = initial_vectors[0][indices, np.arange(self.nstates)]
        initial_vectors *= (pivots.conj()/abs(pivots))[None, None, :]
        if self.method == "ca":
            if c.shape != (self.nstates,):
                raise ValueError("CA adiabatic_coefficients must have shape [S]")
            state = RPSHState(nuclear.q, nuclear.p, nuclear.step, nuclear.time,
                              initial_vectors[0]@c, active)
        else:
            data = self._geometry(nuclear.q, gradients=False,
                                  reference=np.broadcast_to(initial_vectors, (self.nbeads, self.nstates, self.nstates)))
            if self.method == "ba":
                if c.shape != (self.nstates,):
                    raise ValueError("BA adiabatic_coefficients must have shape [S]")
                state = RPSHBAState(nuclear.q, nuclear.p, nuclear.step, nuclear.time,
                                    c, active, data.vectors)
            else:
                if c.shape == (self.nstates,):
                    c = np.broadcast_to(c, (self.nbeads, self.nstates))
                if c.shape != (self.nbeads, self.nstates):
                    raise ValueError("DM adiabatic_coefficients must have shape [S] or [P,S]")
                state = RPSHDMState(nuclear.q, nuclear.p, nuclear.step, nuclear.time,
                                    np.einsum("bij,bj->bi", data.vectors, c), active)
        self._check(state)
        return state

    def density_matrix(self, state):
        """Electronic amplitude density [S,S] in the fixed diabatic basis.

        This is not the mixed quantum/classical active-state population estimator.
        Active-surface counts must be recorded separately.
        """
        self._check(state)
        if self.method == "ca":
            return np.outer(state.coefficients, state.coefficients.conj())
        if self.method == "ba":
            data = self._geometry(state.q, gradients=False, reference=state.basis_vectors)
            z = np.einsum("bij,j->bi", data.vectors, state.coefficients)
        else:
            z = state.coefficients
        return np.einsum("bi,bj->ij", z, z.conj())/self.nbeads

    def populations(self, state):
        """Method-defined adiabatic amplitude populations, not active-state counts."""
        self._check(state)
        if self.method == "ba":
            return abs(state.coefficients)**2
        u = self._geometry(state.q.mean(axis=0, keepdims=True), gradients=False).vectors[0]
        rho = self.density_matrix(state)
        return np.diag(u.conj().T@rho@u).real.copy()

    def energy(self, state):
        self._check(state)
        data = self._geometry(state.q, gradients=False)
        kinetic = np.sum(state.p**2/(2*self.masses[None, :, None]))
        dq = state.q-np.roll(state.q, -1, axis=0)
        spring = .5*(self.nbeads/self.beta)**2*np.sum(self.masses[None, :, None]*dq*dq)
        energy = float(data.energies[:, state.active_surface].sum()+kinetic+spring)
        if not np.isfinite(energy):
            raise ValueError("ring energy must be finite")
        return energy

    @staticmethod
    def _unitary_action(matrix, amplitudes, duration):
        e, u = np.linalg.eigh(matrix)
        c = np.einsum("...si,...s->...i", u.conj(), amplitudes)
        return np.einsum("...si,...i->...s", u, c*np.exp(-1j*duration*e))

    def _electronic_step(self, state, new_q):
        z = state.coefficients.copy()
        population = float(self.populations(state)[state.active_surface])
        if population <= np.finfo(float).eps:
            raise ValueError("active surface has zero electronic population; FSSH probability is undefined")
        transfer = np.zeros(self.nstates)
        h = self.dt/self.electronic_substeps
        velocity = (new_q-state.q)/self.dt
        centroid_velocity = velocity.mean(axis=0)
        active = state.active_surface
        frames = state.basis_vectors.copy() if self.method == "ba" else None
        for k in range(self.electronic_substeps):
            q = state.q+(new_q-state.q)*((k+.5)/self.electronic_substeps)
            if self.method == "ba":
                data = self._geometry(q, gradients=False, reference=frames)
                frames = data.vectors
                connection = np.einsum("bac,bijac->ij", velocity, data.couplings)/self.nbeads
                generator = np.diag(data.energies.mean(axis=0))-1j*connection
                midpoint = self._unitary_action(generator, z, .5*h)
                transfer += 2*h*np.imag(midpoint[active]*midpoint.conj()*generator[:, active])
                z = self._unitary_action(generator, midpoint, .5*h)
            else:
                center = self._geometry(q.mean(axis=0, keepdims=True), gradients=False)
                u = center.vectors[0]
                connection = np.einsum("ac,ijac->ij", centroid_velocity, center.couplings[0])
                if self.method == "ca":
                    c = u.conj().T@z
                    phase = np.exp(-.5j*h*center.energies[0])
                    midpoint = c*phase
                    generator = np.diag(center.energies[0])-1j*connection
                    transfer += 2*h*np.imag(midpoint[active]*midpoint.conj()*generator[:, active])
                    z = u@(midpoint*phase)
                else:
                    data = self._geometry(q, gradients=False)
                    local = np.einsum("bsi,bs->bi", data.vectors.conj(), z)
                    phase = np.exp(-.5j*h*data.energies)
                    midpoint_fixed = np.einsum("bsi,bi->bs", data.vectors, local*phase)
                    midpoint = np.einsum("si,bs->bi", u.conj(), midpoint_fixed)
                    # In a fixed diabatic representation, Eq.22 of Shakya et al.
                    # simplifies exactly to Uc^dagger V_b Uc - i vc.d_c.  The
                    # centroid connection must remain even for stationary beads.
                    overlaps = np.einsum("si,bsj->bij", u.conj(), data.vectors)
                    generator = np.einsum("bik,bk,bjk->bij", overlaps, data.energies,
                                          overlaps.conj())-1j*connection[None]
                    transfer += 2*h*np.imag(midpoint[:, active, None]*midpoint.conj()
                                            *generator[:, :, active]).mean(axis=0)
                    z = np.einsum("bsi,bi->bs", data.vectors, local*phase*phase)
        probability = np.maximum(transfer/population, 0.)
        probability[active] = 0.
        total = float(probability.sum())
        if not np.isfinite(probability).all() or total > 1.+1e-12:
            raise ValueError("FSSH probabilities exceed one; reduce the nuclear timestep")
        if total > 1.:
            probability /= total  # Roundoff only.
        if frames is not None:
            frames = self._geometry(new_q, gradients=False, reference=frames).vectors
        return z, probability, frames

    def _rotate_momenta(self, q, p, active, duration, data=None):
        if not self.geometric_force:
            return
        data = self._geometry(q, gradients=False) if data is None else data
        rootmass = np.repeat(np.sqrt(self.masses), 3)
        for bead in range(self.nbeads):
            curvature = berry_curvature(data.couplings[bead], active)
            if not np.any(curvature):
                continue
            rotation = curvature/rootmass[:, None]/rootmass[None, :]
            y = p[bead].ravel()/rootmass
            half = .5*duration*rotation
            identity = np.eye(y.size)
            p[bead] = (np.linalg.solve(identity-half, (identity+half)@y)*rootmass).reshape(p[bead].shape)

    def _kick(self, state, duration, data):
        gradient = data.gradients[:, state.active_surface]
        state.p -= .5*duration*gradient
        self._rotate_momenta(state.q, state.p, state.active_surface, duration, data)
        state.p -= .5*duration*gradient

    def _hop(self, state, target):
        source = state.active_surface
        target = _surface(target, self.nstates)
        if source == target:
            return HopResult(source, target, False, False, self.rescaling, 0.)
        center = self._geometry(state.q.mean(axis=0, keepdims=True), gradients=False)
        reference = (state.basis_vectors if self.method == "ba" else
                     np.broadcast_to(center.vectors, (self.nbeads, self.nstates, self.nstates)))
        data = self._geometry(state.q, gradients=False,
                              reference=reference if self.rescaling == "bead" else None)
        gap = float(np.sum(data.energies[:, target]-data.energies[:, source]))
        if self.rescaling == "centroid":
            direction = complex_rescaling_direction(center.couplings[0, source, target])
            direction = np.broadcast_to(direction, state.p.shape).copy()
        else:
            direction = complex_rescaling_direction(data.couplings[:, source, target])
        if not np.any(direction):
            return HopResult(source, target, False, True, self.rescaling, 0.)
        a = .5*float(np.sum(direction**2/self.masses[None, :, None]))
        b = float(np.sum(state.p*direction/self.masses[None, :, None]))
        discriminant = b*b-4*a*gap
        allowance = 64*np.finfo(float).eps*(b*b+abs(4*a*gap))
        old_kinetic = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        if discriminant < -allowance:
            if self.frustrated_hops == "reverse":
                state.p -= (b/a)*direction
            kinetic = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
            return HopResult(source, target, False, True, self.rescaling, kinetic-old_kinetic)
        root = np.sqrt(max(discriminant, 0.))
        denominator = b+np.copysign(root, b)
        shift = -2*gap/denominator if denominator != 0. else 0.
        if b == 0. and root != 0.:
            shift = root/(2*a)
        state.p += shift*direction
        state.active_surface = target
        kinetic = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        return HopResult(source, target, True, False, self.rescaling, kinetic-old_kinetic+gap)

    def attempt_hop(self, state, target):
        self._check(state)
        trial = state.copy()
        result = self._hop(trial, target)
        self._check(trial)
        state.p[:] = trial.p
        state.active_surface = trial.active_surface
        self.last_hop = result
        return result

    def _apply_decoherence(self, state):
        if self.decoherence == "none":
            return
        kinetic = float(np.sum(state.p.mean(axis=0)**2/(2*self.masses[:, None])))
        if not np.isfinite(kinetic):
            raise ValueError("centroid kinetic energy must be finite for EDC")
        if kinetic == 0.:
            return
        data = self._geometry(state.q.mean(axis=0, keepdims=True), gradients=False)
        c = (state.coefficients.copy() if self.method == "ba"
             else data.vectors[0].conj().T@state.coefficients)
        active = state.active_surface
        inactive = np.arange(self.nstates) != active
        with np.errstate(over="ignore"):
            factor = 1./(1.+self.decoherence_constant/kinetic)
            damping = np.exp(-self.dt*abs(data.energies[0]-data.energies[0, active])*factor)
        damped = c.copy()
        damped[inactive] *= damping[inactive]
        if np.array_equal(c, damped):
            return
        if abs(c[active]) == 0.:
            raise ValueError("EDC cannot restore a zero active-state amplitude without a phase")
        remaining = 1.-float(np.sum(abs(damped[inactive])**2))
        if remaining < -1e-10:
            raise ValueError("invalid electronic norm in EDC")
        damped[active] = np.exp(1j*np.angle(c[active]))*np.sqrt(max(remaining, 0.))
        state.coefficients[:] = damped if self.method == "ba" else data.vectors[0]@damped

    def step(self, state):
        self._check(state)
        trial = state.copy()
        saved_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            self._kick(trial, .5*self.dt, self._geometry(trial.q))
            self.integrator.free_step(trial.q, trial.p)
            self._kick(trial, .5*self.dt, self._geometry(trial.q))
            z, probabilities, frames = self._electronic_step(state, trial.q)
            trial.coefficients[:] = z
            if frames is not None:
                trial.basis_vectors[:] = frames
            hop = None
            if probabilities.sum() > 0.:
                target = int(np.searchsorted(np.cumsum(probabilities), self.rng.random(), side="right"))
                if target < self.nstates:
                    hop = self._hop(trial, target)
            self._apply_decoherence(trial)
            trial.step = int(trial.step)+1
            trial.time += self.dt
            self._check(trial)
        except BaseException:
            self.rng.bit_generator.state = saved_rng
            raise
        state.q[:], state.p[:], state.coefficients[:] = trial.q, trial.p, trial.coefficients
        if self.method == "ba":
            state.basis_vectors[:] = trial.basis_vectors
        state.active_surface = trial.active_surface
        state.step, state.time = trial.step, trial.time
        self.last_hop = hop
        self.last_probabilities = tuple(float(x) for x in probabilities)
        return state


class RPSHBADynamics(AdvancedRPSHDynamics):
    """RPSH-BA: one adiabatic vector driven by bead-averaged E and v.d."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="ba", **kwargs)


class RPSHDMDynamics(AdvancedRPSHDynamics):
    """Shakya et al. RPSH-DM with a common-centroid density and hop current."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="dm", **kwargs)


class ComplexRPSHDynamics(AdvancedRPSHDynamics):
    """Complex-Hermitian CA hopping with the stated local Berry-force ansatz."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, method="ca", **kwargs)
