"""Finite-temperature Drude--Lorentz spin--boson HEOM reference.

All quantities are atomic units (hbar = 1).  The convention is
``H = (epsilon*sigma_z + delta*sigma_x)/2``, ``Q = coupling_scale*sigma_z``
and ``J(w) = 2*reorganization_energy*cutoff*w/(w*w+cutoff*cutoff)`` with
``C(t) = integral J(w)/pi * (coth(beta*w/2)*cos(w*t)-1j*sin(w*t)) dw``.
Thus changing Q from sigma_z to sigma_z/2 changes the interaction by two.
The parameter lambda is ``integral J(w)/(pi*w) dw``; the displacement
reorganization energy between the two diabatic surfaces is
``4*coupling_scale**2*lambda``.  Any Q-squared counterterm is a scalar here
and has no effect on the density dynamics.

A finite Matsubara expansion and a hard hierarchy boundary are explicit
approximations.  Sparse exponential propagation has no integration timestep,
but neither its completion nor trace preservation establishes physical
convergence.  See Tanimura, JCP 153, 020901 (2020), doi:10.1063/5.0011599.
"""

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from math import ceil, comb

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def _real(value, name, *, positive=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(raw)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite and "
                         + ("positive" if positive else "nonnegative"))
    return value


def _integer(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < (1 if positive else 0)):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _readonly(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _density(value):
    raw = np.asarray(value)
    if raw.shape != (2, 2) or raw.dtype.kind not in "fciu" or not np.isfinite(raw).all():
        raise ValueError("density must be a finite numerical [2,2] matrix")
    rho = np.array(raw, dtype=np.complex128, copy=True)
    if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0.):
        raise ValueError("density must be Hermitian")
    if not np.isclose(np.trace(rho), 1., atol=1e-12, rtol=0.):
        raise ValueError("density must have unit trace; it is not normalized implicitly")
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("density must be positive semidefinite")
    return rho


@dataclass(frozen=True)
class HEOMState:
    """Complete scaled hierarchy at one time, including every bath-memory ADO.

    ``ados[n,2,2]`` uses the solver's ``labels`` ordering and is scaled by
    ``sqrt(product(n_k! * abs(c_k)**n_k))``.  The zeroth ADO is the density.
    This numerical state can continue in a fresh identically configured
    solver; a reduced density alone would reset the bath memory.  No random
    state is involved.  File checkpoint I/O is not provided by this API.
    """

    time: float
    ados: np.ndarray
    run_identity: str

    def __post_init__(self):
        time = _real(self.time, "state time", nonnegative=True)
        array = np.asarray(self.ados)
        if (array.ndim != 3 or array.shape[1:] != (2, 2) or len(array) < 1
                or array.dtype.kind not in "fciu" or not np.isfinite(array).all()):
            raise ValueError("state ADOs must be a finite numerical [ADO,2,2] array")
        if (not isinstance(self.run_identity, str) or len(self.run_identity) != 64
                or any(c not in "0123456789abcdef" for c in self.run_identity)):
            raise ValueError("state run_identity must be a SHA-256 hexadecimal string")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "ados", _readonly(np.asarray(array, dtype=np.complex128)))

    @property
    def density(self):
        return self.ados[0]


@dataclass(frozen=True)
class HEOMResult:
    """Unclipped reduced densities and complete final bath-memory state.

    Compare at least hierarchy depths and Matsubara cutoffs before treating this
    as a reference.
    No normalization, positivity projection, or Hermitization is performed.
    """

    time: np.ndarray
    density: np.ndarray
    final_state: HEOMState
    max_trace_error: float
    max_hermiticity_error: float
    min_density_eigenvalue: float
    @property
    def populations(self):
        return self.density.diagonal(axis1=-2, axis2=-1).real

    @property
    def population_difference(self):
        return self.populations[:, 0]-self.populations[:, 1]


class SpinBosonHEOM:
    """Scaled sparse HEOM with a finite Drude--Lorentz Matsubara expansion.

    ``max_depth`` retains labels with ``sum(n) <= max_depth`` and sets all
    higher ADOs to zero.  ``matsubara_terms=K`` retains the Drude pole and K
    thermal poles.  With ``terminator=True``, the omitted thermal tail is
    approximated by ``-tail_residue * [Q,[Q,rho_n]]`` on every retained ADO.
    This is a Matsubara-tail approximation, not a hierarchy-depth closure.
    The omitted poles must all be faster than the Drude cutoff in this mode.

    Initial density matrices describe a factorized preparation with a free
    thermal bath; all higher ADOs initially vanish.  Only :class:`HEOMState`
    preserves already accumulated bath memory on continuation.  Hamiltonians,
    coupling, temperatures and bath parameters are time independent.  No
    zero-temperature, arbitrary spectral density or correlated-equilibrium
    preparation is silently substituted.

    ``max_ados`` is an allocation guard and never truncates the requested
    hierarchy.  A zero interaction is reduced exactly to the isolated system.
    """

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("SpinBosonHEOM parameters are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, epsilon, delta, reorganization_energy, cutoff, beta, *,
                 max_depth=4, matsubara_terms=2, terminator=False,
                 coupling_scale=1., max_ados=100000):
        self.epsilon = _real(epsilon, "epsilon")
        self.delta = _real(delta, "delta")
        self.reorganization_energy = _real(reorganization_energy, "reorganization_energy",
                                           nonnegative=True)
        self.cutoff = _real(cutoff, "cutoff", positive=True)
        self.beta = _real(beta, "beta", positive=True)
        self.coupling_scale = _real(coupling_scale, "coupling_scale")
        self.max_depth = _integer(max_depth, "max_depth")
        self.matsubara_terms = _integer(matsubara_terms, "matsubara_terms")
        self.max_ados = _integer(max_ados, "max_ados", positive=True)
        if not isinstance(terminator, (bool, np.bool_)):
            raise ValueError("terminator must be boolean")
        self.terminator = bool(terminator)
        self.hamiltonian = _readonly(np.array([[self.epsilon, self.delta],
                                               [self.delta, -self.epsilon]])/2)
        self.coupling = _readonly(np.diag([self.coupling_scale, -self.coupling_scale]))

        active = self.reorganization_energy != 0 and self.coupling_scale != 0
        count = self.matsubara_terms+1 if active else 0
        self.ado_count = comb(self.max_depth+count, count)
        if self.ado_count > self.max_ados:
            raise ValueError(f"requested hierarchy has {self.ado_count} ADOs, exceeding "
                             f"max_ados={self.max_ados}; no automatic truncation is performed")
        self.coefficients, self.frequencies, self.tail_residue = self._bath(active)
        self.labels = self._labels(count)
        self._generator = self._assemble()
        if not np.isfinite(self._generator.data).all():
            raise ValueError("HEOM parameters overflow the double-precision generator")
        self._trace = self._generator.diagonal().sum()
        shifted = (self._generator - sparse.eye(4*self.ado_count, format="csr")
                   *(self._trace/(4*self.ado_count)))
        self._action_norm = float(abs(shifted).sum(axis=0).max())
        if not np.isfinite(self._trace) or not np.isfinite(self._action_norm):
            raise ValueError("HEOM parameters overflow the exponential-action norm")
        self._identity_json = json.dumps({
            "schema": "pyrpmd.spin_boson_heom.v1", "epsilon": self.epsilon,
            "delta": self.delta, "reorganization_energy": self.reorganization_energy,
            "cutoff": self.cutoff, "beta": self.beta, "coupling_scale": self.coupling_scale,
            "max_depth": self.max_depth, "matsubara_terms": self.matsubara_terms,
            "terminator": self.terminator, "hierarchy_boundary": "zero",
            "ado_scaling": "sqrt_factorial_abs_coefficient", "dtype": "complex128",
            "propagator": "scipy.sparse.linalg.expm_multiply",
            "exponential_action_norm_limit": 16.,
            "spectral_density_convention": "C=integral(J/pi)*(coth*cos-i*sin)",
            "initial_bath": "factorized_free_thermal_or_complete_ado_state",
        }, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.run_identity = sha256(self._identity_json.encode()).hexdigest()
        self._frozen = True

    @property
    def identity(self):
        """Detached JSON-compatible scientific configuration (allocation guard excluded)."""
        return json.loads(self._identity_json)

    @property
    def generator(self):
        """Detached CSR generator, for independent small-system verification."""
        return self._generator.copy()

    def _bath(self, active):
        if not active:
            return _readonly(np.empty(0, complex)), _readonly(np.empty(0)), 0.
        lam, gamma, beta, k = (self.reorganization_energy, self.cutoff,
                               self.beta, self.matsubara_terms)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            angle = beta*gamma/2
            frequencies = np.r_[gamma, 2*np.pi*np.arange(1, k+1)/beta]
        if not np.isfinite(angle) or not np.isfinite(frequencies).all():
            raise ValueError("beta and cutoff overflow the Matsubara frequencies")
        # At coalescing Drude and thermal poles a different limiting expansion
        # is required; subtracting two divergent exponentials is not reliable.
        nearest_pole = round(angle/np.pi)
        if nearest_pole >= 1 and abs(angle-nearest_pole*np.pi) <= 1e-8*max(1., abs(angle)):
            raise ValueError("Drude cutoff coincides with a Matsubara pole; "
                             "coalescing-pole expansions are unsupported")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coeff = np.empty(k+1, dtype=np.complex128)
            coeff[0] = lam*gamma*(1/np.tan(angle)-1j)
            coeff[1:] = 4*lam*gamma*frequencies[1:]/(beta*(frequencies[1:]**2-gamma**2))
            residue = 2*lam/(beta*gamma)-np.sum((coeff/frequencies).real)
        if not np.isfinite(coeff).all() or not np.isfinite(residue) or np.any(abs(coeff) == 0):
            raise ValueError("bath expansion is not representable in finite double precision")
        if self.terminator and (2*np.pi*(k+1)/beta <= gamma or residue < 0):
            raise ValueError("terminator requires omitted Matsubara poles faster than the "
                             "Drude cutoff and a nonnegative tail residue; increase matsubara_terms")
        return _readonly(coeff), _readonly(frequencies), float(residue)

    def _labels(self, count):
        if count == 0:
            return ((),)
        labels = [tuple([0]*count)]
        for tier in range(1, self.max_depth+1):
            for entries in combinations_with_replacement(range(count), tier):
                labels.append(tuple(np.bincount(entries, minlength=count).tolist()))
        return tuple(labels)

    def _assemble(self):
        ident = np.eye(2)
        left_q, right_q = np.kron(self.coupling, ident), np.kron(ident, self.coupling.T)
        commutator = left_q-right_q  # C-order vectorization of density matrices.
        local = -1j*(np.kron(self.hamiltonian, ident)-np.kron(ident, self.hamiltonian.T))
        if self.terminator:
            local -= self.tail_residue*(commutator@commutator)
        damping = np.asarray([np.dot(label, self.frequencies) for label in self.labels])
        result = (sparse.kron(sparse.eye(self.ado_count, format="csr"), local, format="csr")
                  - sparse.kron(sparse.diags(damping), sparse.eye(4), format="csr"))
        lookup = {label: i for i, label in enumerate(self.labels)}
        for k, coefficient in enumerate(self.coefficients):
            rows, columns, upward, downward = [], [], [], []
            magnitude = abs(coefficient)
            for index, label in enumerate(self.labels):
                if sum(label) >= self.max_depth:
                    continue
                higher = list(label)
                higher[k] += 1
                rows.append(index)
                columns.append(lookup[tuple(higher)])
                upward.append(np.sqrt(label[k]+1)*np.sqrt(magnitude))
                downward.append(np.sqrt(label[k]+1)/np.sqrt(magnitude))
            shape = (self.ado_count, self.ado_count)
            up = sparse.csr_matrix((upward, (rows, columns)), shape=shape)
            down = sparse.csr_matrix((downward, (columns, rows)), shape=shape)
            result += sparse.kron(up, -1j*commutator, format="csr")
            result += sparse.kron(down, -1j*(coefficient*left_q-coefficient.conjugate()*right_q),
                                  format="csr")
        result.eliminate_zeros()
        result.sort_indices()
        return result

    def initialize(self, density, *, time=0.):
        """Prepare a factorized free thermal bath; never infer thermal equilibrium."""
        rho = _density(density)
        ados = np.zeros((self.ado_count, 2, 2), dtype=np.complex128)
        ados[0] = rho
        return HEOMState(time, ados, self.run_identity)

    def propagate(self, initial, times):
        """Propagate a density or complete :class:`HEOMState` at requested times.

        Times must increase strictly and start at the initial state's time
        (zero for a density input).  All ADOs propagate between observations;
        only the reduced densities and final full hierarchy are retained.
        Continuing the final state reproduces uninterrupted propagation up to
        floating-point exponential-action error across new observation grids;
        partitioning the same grid is bitwise reproducible in one environment.
        Large intervals are subdivided into exact exponential actions to keep
        SciPy on its deterministic exact-norm path, avoiding randomized norm
        estimation and any consumption of the caller's random-number stream.
        """
        state = initial if isinstance(initial, HEOMState) else self.initialize(initial)
        if state.run_identity != self.run_identity:
            raise ValueError("HEOM state run_identity does not match the solver")
        if state.ados.shape != (self.ado_count, 2, 2):
            raise ValueError("HEOM state ADO count does not match the solver")
        if (not np.isclose(np.trace(state.density), 1., rtol=0., atol=1e-8)
                or not np.allclose(state.density, state.density.conj().T, rtol=0., atol=1e-8)):
            raise ValueError("HEOM state density must have unit trace and be Hermitian")
        raw = np.asarray(times)
        if (raw.ndim != 1 or len(raw) == 0 or raw.dtype.kind not in "fiu"
                or not np.isfinite(raw).all()):
            raise ValueError("times must be a nonempty finite real vector")
        time = np.array(raw, dtype=np.float64, copy=True)
        if not np.isfinite(time).all() or time[0] != state.time or np.any(np.diff(time) <= 0):
            raise ValueError("times must increase strictly and start at the initial state time")
        vector = state.ados.reshape(-1).copy()
        density = np.empty((len(time), 2, 2), dtype=np.complex128)
        density[0] = vector[:4].reshape(2, 2)
        for index, duration in enumerate(np.diff(time), 1):
            action_norm = duration*self._action_norm
            if not np.isfinite(action_norm):
                raise ValueError("observation interval overflows the HEOM generator")
            # SciPy 1.11--1.18 uses m_max=55, ell=2, p_max=8, theta_55=9.9:
            # condition 3.13 for one vector is norm <= 2*2*8*11*9.9/55=63.36.
            # The cap of 16 leaves ample rounding margin in this exact-norm
            # branch.  Repeated actions introduce no time discretization.
            actions = max(1, ceil(action_norm/16.))
            action_time = duration/actions
            scaled = action_time*self._generator
            action_trace = action_time*self._trace
            if not np.isfinite(scaled.data).all() or not np.isfinite(action_trace):
                raise ValueError("observation interval overflows the HEOM generator")
            for _ in range(actions):
                vector = expm_multiply(scaled, vector, traceA=action_trace)
            if not np.isfinite(vector).all():
                raise FloatingPointError("HEOM propagation produced nonfinite ADOs")
            density[index] = vector[:4].reshape(2, 2)
        trace_error = float(np.max(abs(np.trace(density, axis1=1, axis2=2)-1)))
        hermiticity_error = float(np.max(abs(density-density.conj().swapaxes(1, 2))))
        # This Hermitian part is used ONLY to report a diagnostic, never to
        # replace or project the propagated density.
        minimum = float(np.linalg.eigvalsh((density+density.conj().swapaxes(1, 2))/2).min())
        final = HEOMState(time[-1], vector.reshape(self.ado_count, 2, 2), self.run_identity)
        return HEOMResult(_readonly(time), _readonly(density), final,
                          trace_error, hermiticity_error, minimum)
