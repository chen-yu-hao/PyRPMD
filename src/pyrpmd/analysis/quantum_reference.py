"""Small, independent one-dimensional quantum reference calculations.

Atomic units, one uniform periodic grid, a time-independent Hermitian matrix
potential, and a scalar nuclear mass are required. There is no absorber or
trajectory approximation. The Fourier split operator is unitary and second
order in time; a finite grid and finite time step still require convergence.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


def _real(value, name, *, positive=False):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(raw)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be positive and finite" if positive else f"{name} must be finite")
    return value


def _integer(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < (1 if positive else 0)):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _readonly(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _grid(value):
    raw = np.asarray(value)
    if (raw.ndim != 1 or len(raw) < 4 or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw).all()):
        raise ValueError("grid must be a finite real vector with at least four points")
    grid = np.array(raw, dtype=float, copy=True)
    difference = np.diff(grid)
    dx = float(difference[0])
    if (not np.isfinite(difference).all() or dx <= 0
            or not np.allclose(difference, dx, rtol=1e-10, atol=0.)
            or not np.isfinite(dx*len(grid))):
        raise ValueError("grid must be strictly increasing and uniformly spaced")
    return _readonly(grid), dx


def fourier_kinetic_matrix(grid, mass):
    """Dense periodic-grid kinetic operator, in the orthonormal grid basis.

    This O(N**2) utility is for small reference Hamiltonians and thermal states.
    Wavefunction samples themselves have norm ``dx * sum(abs(psi)**2)``.
    """
    grid, dx = _grid(grid)
    mass = _real(mass, "mass", positive=True)
    kinetic = (2*np.pi*np.fft.fftfreq(len(grid), d=dx))**2/(2*mass)
    if not np.isfinite(kinetic).all():
        raise ValueError("grid and mass produce nonfinite kinetic energies")
    matrix = np.fft.ifft(kinetic[:, None]*np.fft.fft(np.eye(len(grid)), axis=0), axis=0)
    return 0.5*matrix+0.5*matrix.conj().T


@dataclass(frozen=True)
class GridThermalEnsemble:
    """Finite-grid canonical eigenstates; each row has unit dx-weighted norm."""
    wavefunctions: np.ndarray
    weights: np.ndarray
    energies: np.ndarray
    omitted_weight: float


def harmonic_canonical_ensemble(grid, mass, omega, beta, *, center=0., tail_tolerance=1e-12):
    """Diagonalize the periodic-grid harmonic Hamiltonian and form exp(-beta H).

    Returns nuclear wavefunctions ``[component,grid]`` and normalized positive
    mixture weights. ``omitted_weight`` measures eigenstate truncation within
    this finite grid only; it does not bound box or grid-discretization errors.
    Set ``tail_tolerance=0`` to retain every grid eigenstate.
    """
    grid, dx = _grid(grid)
    mass, omega = _real(mass, "mass", positive=True), _real(omega, "omega", positive=True)
    beta, center = _real(beta, "beta", positive=True), _real(center, "center")
    tolerance = _real(tail_tolerance, "tail_tolerance")
    if not 0 <= tolerance < 1:
        raise ValueError("tail_tolerance must be in [0,1)")
    with np.errstate(over="ignore", invalid="ignore"):
        potential = 0.5*mass*np.square(omega)*np.square(grid-center)
    if not np.isfinite(potential).all():
        raise ValueError("harmonic potential overflowed")
    energies, vectors = eigh(fourier_kinetic_matrix(grid, mass)+np.diag(potential))
    with np.errstate(over="ignore"):
        weights = np.exp(-beta*(energies-energies[0]))
    weights /= weights.sum()
    keep = len(grid) if tolerance == 0 else min(len(grid), 1+int(np.searchsorted(np.cumsum(weights), 1.-tolerance)))
    omitted = float(weights[keep:].sum())
    selected = weights[:keep].copy()
    selected /= selected.sum()
    return GridThermalEnsemble(_readonly(vectors[:, :keep].T/np.sqrt(dx)),
                               _readonly(selected), _readonly(energies[:keep]), omitted)


@dataclass(frozen=True)
class QuantumReferenceResult:
    """Electronic density time series and a detached final wavefunction batch.

    ``component_norms`` is [frame,component]. ``edge_population`` integrates the
    outer ceil(0.05*N) points at each end; it is a boundary diagnostic, not a
    proof that a periodic box has converged. Times are relative atomic times.
    """
    time: np.ndarray
    density: np.ndarray
    component_norms: np.ndarray
    edge_population: np.ndarray
    final_wavefunctions: np.ndarray
    weights: np.ndarray

    @property
    def populations(self):
        return self.density.diagonal(axis1=-2, axis2=-1).real

    @property
    def norm(self):
        return self.component_norms@self.weights


class SplitOperator1D:
    """Periodic Fourier Strang propagator exp(-iVdt/2) exp(-iTdt) exp(-iVdt/2).

    ``grid[N]`` excludes the duplicated periodic endpoint. ``potential[N,S,S]``
    is a stationary Hermitian matrix in a fixed electronic basis. Inputs are
    copied and frozen because cached propagators depend on them.

    Pure states have shape [N,S]; mixtures use [component,N,S] and explicit
    nonnegative weights summing to one. Each component must already have unit
    dx-weighted norm. No input normalization or signed-weight interpretation is
    performed. NumPy/SciPy are the only numerical dependencies.
    """
    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("SplitOperator1D parameters are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, grid, potential, mass, dt):
        self.grid, self.dx = _grid(grid)
        self.mass, self.dt = _real(mass, "mass", positive=True), _real(dt, "dt", positive=True)
        matrix = np.asarray(potential)
        if (matrix.ndim != 3 or matrix.shape[0] != len(self.grid)
                or matrix.shape[1] < 1 or matrix.shape[1] != matrix.shape[2]
                or matrix.dtype.kind not in "fciu" or not np.isfinite(matrix).all()):
            raise ValueError("potential must be a finite numerical [grid,state,state] array")
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.array(matrix, dtype=complex if np.iscomplexobj(matrix) else float, copy=True)
        if not np.isfinite(matrix).all():
            raise ValueError("potential must be representable with finite double-precision entries")
        if not np.allclose(matrix, matrix.conj().swapaxes(-1, -2), rtol=0., atol=1e-12):
            raise ValueError("potential must be Hermitian; absorbers are unsupported")
        self.potential = _readonly(0.5*matrix+0.5*matrix.conj().swapaxes(-1, -2))
        self.nstates = matrix.shape[1]
        energies, vectors = np.linalg.eigh(self.potential)
        kinetic = (2*np.pi*np.fft.fftfreq(len(self.grid), d=self.dx))**2/(2*self.mass)
        with np.errstate(over="ignore", invalid="ignore"):
            potential_angles, kinetic_angles = 0.5*self.dt*energies, self.dt*kinetic
        if not np.isfinite(potential_angles).all() or not np.isfinite(kinetic_angles).all():
            raise ValueError("grid, mass, potential or dt produce nonfinite propagation phases")
        self._half_potential = _readonly((vectors*np.exp(-1j*potential_angles)[:, None, :])
                                         @vectors.conj().swapaxes(-1, -2))
        self._kinetic_phase = _readonly(np.exp(-1j*kinetic_angles))
        self._frozen = True

    def _wavefunctions(self, value):
        raw = np.asarray(value)
        pure = raw.ndim == 2
        if (raw.ndim not in (2, 3) or raw.shape[-2:] != (len(self.grid), self.nstates)
                or min(raw.shape) < 1 or raw.dtype.kind not in "fciu" or not np.isfinite(raw).all()):
            raise ValueError("wavefunctions must be finite [grid,state] or [component,grid,state] arrays")
        psi = np.array(raw[None] if pure else raw, dtype=complex, copy=True)
        norms = self.dx*np.sum(np.abs(psi)**2, axis=(1, 2))
        if not np.allclose(norms, 1., rtol=0., atol=1e-10):
            raise ValueError("each wavefunction must have unit dx-weighted norm")
        return psi, pure

    @staticmethod
    def _weights(value, count):
        if value is None:
            if count != 1:
                raise ValueError("mixtures require explicit weights")
            return np.ones(1)
        weights = np.asarray(value)
        if (weights.shape != (count,) or weights.dtype.kind not in "fiu"
                or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-10)):
            raise ValueError("weights must be a nonnegative finite vector summing to one")
        return np.array(weights, dtype=float, copy=True)

    def _advance(self, psi):
        psi = np.einsum("xij,mxj->mxi", self._half_potential, psi)
        psi = np.fft.ifft(np.fft.fft(psi, axis=1, norm="ortho")
                         *self._kinetic_phase[None, :, None], axis=1, norm="ortho")
        return np.einsum("xij,mxj->mxi", self._half_potential, psi)

    def step(self, wavefunctions):
        """Return a new state after one step, preserving pure/batch input shape."""
        psi, pure = self._wavefunctions(wavefunctions)
        result = self._advance(psi)
        if not np.isfinite(result).all():
            raise FloatingPointError("quantum propagation produced nonfinite amplitudes")
        return result[0] if pure else result

    def propagate(self, wavefunctions, steps, *, weights=None, sample_stride=1):
        """Record initial/final densities and every sample_stride steps in between."""
        steps, stride = _integer(steps, "steps"), _integer(sample_stride, "sample_stride", positive=True)
        psi, _ = self._wavefunctions(wavefunctions)
        weights = self._weights(weights, len(psi))
        indices = list(range(0, steps+1, stride))
        if indices[-1] != steps:
            indices.append(steps)
        time = np.asarray(indices)*self.dt
        if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("observation times overflow or cannot be resolved")
        densities, norms, edges = [], [], []
        relative_step, edge = 0, max(1, int(np.ceil(0.05*len(self.grid))))
        for index in indices:
            while relative_step < index:
                psi = self._advance(psi)
                relative_step += 1
            if not np.isfinite(psi).all():
                raise FloatingPointError("quantum propagation produced nonfinite amplitudes")
            probability = np.abs(psi)**2
            norms.append(self.dx*probability.sum(axis=(1, 2)))
            densities.append(self.dx*np.einsum("m,mxi,mxj->ij", weights, psi, psi.conj()))
            edges.append(self.dx*np.dot(weights, probability[:, :edge].sum(axis=(1, 2))
                                        +probability[:, -edge:].sum(axis=(1, 2))))
        return QuantumReferenceResult(_readonly(time), _readonly(np.asarray(densities)),
                                      _readonly(np.asarray(norms)), _readonly(np.asarray(edges)),
                                      _readonly(psi), _readonly(weights))
