"""Independent-electron surface hopping (IESH), with classical nuclei (P=1).

The spinless independent-electron wavefunction is a Slater determinant, not a
product of independently hopping probabilities.  Electronic orbitals are stored
in a fixed diabatic basis; all occupied orbitals share the same unitary map.
Nuclei move on U0(q) + sum(occupied one-electron adiabatic energies).  Only one
occupied-to-empty orbital hop may occur in a nuclear step.  The step must be
converged, including its total outgoing probability; probabilities above the
configured limit fail transactionally rather than being clipped or normalized.

References: Shenvi, Roy and Tully, JCP 130, 174107 (2009),
doi:10.1063/1.3125436; Gardner et al., JCP 158, 064101 (2023),
doi:10.1063/5.0137137.  No ring-polymer extension, complex SOC, decoherence
correction, electronic thermostat, or interacting-electron model is implied.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .dynamics import CayleyIntegrator, State, _state_clock
from .nonadiabatic import DiabaticResult, _positive_integer, validate_diabatic_result
from .potentials.base import PESResult, Potential, PotentialError, coordinates, validate_result


def _real(value, name, *, positive=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(raw)
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _occupied(value, nstates, nelectrons=None):
    raw = np.asarray(value)
    if (raw.ndim != 1 or raw.dtype.kind not in "iu" or not raw.size
            or np.any(raw < 0) or np.any(raw >= nstates)
            or len(np.unique(raw)) != len(raw)
            or nelectrons is not None and len(raw) != nelectrons):
        raise ValueError("occupied must contain distinct in-range integer orbital indices")
    return np.array(raw, dtype=np.int64, order="C", copy=True)


def _orbitals(value):
    raw = np.asarray(value)
    if (raw.ndim != 2 or min(raw.shape) < 1 or raw.shape[1] > raw.shape[0]
            or raw.dtype.kind not in "fciu" or not np.isfinite(raw).all()):
        raise ValueError("coefficients must be finite numerical orbitals [nstates,nelectrons]")
    if not np.allclose(raw.conj().T @ raw, np.eye(raw.shape[1]), rtol=0., atol=1e-10):
        raise ValueError("Slater determinant orbitals must be orthonormal")
    return np.array(raw, dtype=np.complex128, order="C", copy=True)


def _immutable(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class NewnsAndersonModel(Potential):
    """Spinless resonant level coupled to a quadrature-discretized metallic band.

    All arguments are in atomic units.  With x=q[coordinate]-center,
    epsilon_d=level_energy+level_slope*x, Gamma=hybridization*exp(
    hybridization_slope*x), and V_k=sqrt(Gamma*w_k/(2*pi)).  The separate bare
    nuclear potential is U0=bare_force_constant*x**2/2.  ``evaluate`` returns
    ONLY the one-electron Hamiltonian; ``bare_potential`` supplies U0 once,
    independent of the number of electrons.

    ``gauss-legendre`` splits the band at chemical_potential before quadrature;
    ``midpoint`` uses uniformly spaced bin centers.  The finite band and number
    of bath states are physical convergence parameters, not an infinite bath.
    """
    _parameter_names = (
        "nband", "band_min", "band_max", "chemical_potential", "discretization",
        "level_energy", "level_slope", "hybridization", "hybridization_slope",
        "bare_force_constant", "center", "coordinate", "natoms",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("Newns-Anderson model parameters are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, nband=20, band_min=-0.5, band_max=0.5,
                 chemical_potential=0., discretization="gauss-legendre",
                 level_energy=0., level_slope=0.01, hybridization=0.02,
                 hybridization_slope=0., bare_force_constant=0.001, center=0.,
                 coordinate=0, natoms=1):
        self.nband = _positive_integer(nband, "nband")
        if self.nband < 2:
            raise ValueError("nband must be at least two")
        self.natoms = _positive_integer(natoms, "natoms")
        if (isinstance(coordinate, (bool, np.bool_))
                or not isinstance(coordinate, (int, np.integer))
                or not 0 <= coordinate < 3*self.natoms):
            raise ValueError("coordinate must be an in-range Cartesian integer index")
        self.coordinate = int(coordinate)
        self.coordinates, self.ncoordinates = (self.coordinate,), 1
        for name, value in (
            ("band_min", band_min), ("band_max", band_max),
            ("chemical_potential", chemical_potential), ("level_energy", level_energy),
            ("level_slope", level_slope), ("hybridization_slope", hybridization_slope),
            ("center", center),
        ):
            setattr(self, name, _real(value, name))
        self.hybridization = _real(hybridization, "hybridization", nonnegative=True)
        self.bare_force_constant = _real(bare_force_constant, "bare_force_constant", nonnegative=True)
        if not self.band_min < self.chemical_potential < self.band_max:
            raise ValueError("chemical_potential must lie strictly inside the band")
        if discretization not in ("gauss-legendre", "midpoint"):
            raise ValueError("discretization must be gauss-legendre or midpoint")
        self.discretization = discretization
        width = self.band_max-self.band_min
        if not np.isfinite(width):
            raise ValueError("bandwidth must be finite")
        if discretization == "midpoint":
            weights = np.full(self.nband, width/self.nband)
            energies = self.band_min+(np.arange(self.nband)+0.5)*weights
        else:
            energies, weights = [], []
            for count, low, high in (
                (self.nband//2, self.band_min, self.chemical_potential),
                (self.nband-self.nband//2, self.chemical_potential, self.band_max),
            ):
                nodes, w = np.polynomial.legendre.leggauss(count)
                energies.extend(low+(nodes+1.)*((high-low)/2.))
                weights.extend(w*((high-low)/2.))
        self.bath_energies, self.bath_weights = _immutable(energies), _immutable(weights)
        self.nstates = self.nband+1
        self._frozen = True

    @property
    def metadata(self):
        return {"provider": "newns_anderson", "execution_device": "cpu",
                "thread_safety": "reentrant", "trajectory_batch_safe": True,
                "periodic": False, "derivative": "gradient", "spin_degeneracy": 1,
                "energy_unit": "hartree", "length_unit": "bohr",
                "nstates": self.nstates, "natoms": self.natoms,
                "model_identity": self.to_dict()}

    def to_dict(self):
        return {"schema": "pyrpmd.newns_anderson_model", "version": 1,
                "model": "newns-anderson",
                "parameters": {name: getattr(self, name) for name in self._parameter_names}}

    @classmethod
    def from_dict(cls, value):
        if (not isinstance(value, Mapping)
                or set(value) != {"schema", "version", "model", "parameters"}
                or value["schema"] != "pyrpmd.newns_anderson_model"
                or isinstance(value["version"], (bool, np.bool_))
                or not isinstance(value["version"], (int, np.integer)) or value["version"] != 1
                or value["model"] != "newns-anderson"
                or not isinstance(value["parameters"], Mapping)
                or set(value["parameters"]) != set(cls._parameter_names)):
            raise ValueError("invalid Newns-Anderson model identity")
        return cls(**value["parameters"])

    def _coordinates(self, q):
        q = coordinates(q)
        if q.shape[1] != self.natoms:
            raise PotentialError("Newns-Anderson coordinate atom count mismatch")
        return q, q.reshape(len(q), -1)[:, self.coordinate]-self.center

    def evaluate(self, q):
        q, x = self._coordinates(q)
        with np.errstate(over="ignore", invalid="ignore"):
            gamma = (np.zeros_like(x) if self.hybridization == 0.
                     else self.hybridization*np.exp(self.hybridization_slope*x))
            coupling = np.sqrt(gamma[:, None]*self.bath_weights[None, :]/(2*np.pi))
            level = self.level_energy+self.level_slope*x
            coupling_gradient = 0.5*self.hybridization_slope*coupling
        if not all(np.isfinite(v).all() for v in (coupling, level, coupling_gradient)):
            raise PotentialError("Newns-Anderson model evaluation overflowed")
        matrix = np.zeros((len(q), self.nstates, self.nstates))
        gradients = np.zeros((*matrix.shape, self.natoms*3))
        matrix[:, 0, 0] = level
        diagonal = np.arange(1, self.nstates)
        matrix[:, diagonal, diagonal] = self.bath_energies
        matrix[:, 0, 1:] = matrix[:, 1:, 0] = coupling
        gradients[:, 0, 0, self.coordinate] = self.level_slope
        gradients[:, 0, 1:, self.coordinate] = coupling_gradient
        gradients[:, 1:, 0, self.coordinate] = coupling_gradient
        return DiabaticResult(matrix, gradients.reshape(*matrix.shape, self.natoms, 3))

    def bare_potential(self, q):
        q, x = self._coordinates(q)
        forces = np.zeros_like(q).reshape(len(q), -1)
        with np.errstate(over="ignore", invalid="ignore"):
            energy = 0.5*self.bare_force_constant*x*x
            forces[:, self.coordinate] = -self.bare_force_constant*x
        return validate_result(PESResult(energy, forces.reshape(q.shape)), q.shape)


@dataclass
class IESHState(State):
    """P=1 nuclei, fixed-diabatic orthonormal orbitals and occupied adiabats.

    The occupied vector is ordered.  Replacing its r-th entry keeps that row
    position, so Slater determinant signs need no permutation convention.
    Save the dynamics RNG state together with this state for exact continuation.
    """
    coefficients: np.ndarray | None = None
    occupied: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        if len(self.q) != 1:
            raise ValueError("IESH supports classical nuclei, P=1 only")
        self.coefficients = _orbitals(self.coefficients)
        self.occupied = _occupied(self.occupied, *self.coefficients.shape)

    def copy(self):
        return IESHState(self.q, self.p, self.step, self.time, self.coefficients, self.occupied)

    def to_dict(self):
        return {"q": self.q.tolist(), "p": self.p.tolist(), "step": int(self.step),
                "time": float(self.time), "coefficients_real": self.coefficients.real.tolist(),
                "coefficients_imag": self.coefficients.imag.tolist(),
                "occupied": self.occupied.tolist()}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {
            "q", "p", "step", "time", "coefficients_real", "coefficients_imag", "occupied"
        }:
            raise ValueError("invalid IESH state fields")
        real, imag = np.asarray(value["coefficients_real"]), np.asarray(value["coefficients_imag"])
        if real.shape != imag.shape or real.dtype.kind not in "fiu" or imag.dtype.kind not in "fiu":
            raise ValueError("IESH real/imaginary orbital arrays must have equal numerical shapes")
        return cls(value["q"], value["p"], value["step"], value["time"],
                   real+1j*imag, value["occupied"])


def slater_determinant_ratios(coefficients, occupied):
    """All one-row replacement determinant ratios, with shape [state,electron].

    For S=C[occupied,:], replacing row r with C[j,:] gives det(S')/det(S)
    = (C @ inverse(S))[j,r].  A single linear solve evaluates every ratio in
    O(ne**3 + nstates*ne**2), without exponentially small determinants or
    division by their squared magnitude.  Singular active determinants fail.
    """
    c = _orbitals(coefficients)
    occupied = _occupied(occupied, *c.shape)
    overlap = c[occupied]
    try:
        ratios = np.linalg.solve(overlap.T, c.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("active Slater determinant has singular overlap; conditional IESH rates are undefined") from exc
    # C is an isometry, so ||C S^-1||_F = ||S^-1||_F.  This inexpensive
    # bound detects unresolved active determinants without a second SVD/LU.
    condition_bound = np.linalg.norm(overlap)*np.linalg.norm(ratios)
    if (not np.isfinite(ratios).all() or not np.isfinite(condition_bound)
            or condition_bound > 1./(64*np.finfo(float).eps)):
        raise ValueError("active Slater overlap is numerically singular; reduce timestep")
    return ratios


def determinant_hopping_probabilities(coefficients, occupied, velocity_couplings, dt,
                                     *, max_probability=1.):
    """IESH probabilities [electron,target] in the convention d_ij=<i|grad j>.

    D_ij=v dot d_ij is real antisymmetric.  The outgoing current for replacing
    occupied[r]=i by j is 2 Re(det(S')/det(S)) D_ij.  Pauli-forbidden targets
    have zero probability.  The total must not exceed max_probability<=1.
    """
    c = _orbitals(coefficients)
    occupied = _occupied(occupied, *c.shape)
    d = np.asarray(velocity_couplings)
    if (d.shape != (len(c), len(c)) or d.dtype.kind not in "fiu"
            or not np.isfinite(d).all()
            or not np.allclose(d, -d.T, rtol=0., atol=1e-12)):
        raise ValueError("velocity_couplings must be finite real antisymmetric [nstates,nstates]")
    dt = _real(dt, "dt", positive=True)
    maximum = _real(max_probability, "max_probability", positive=True)
    if maximum > 1.:
        raise ValueError("max_probability must not exceed one")
    probabilities = np.zeros((len(occupied), len(c)))
    if len(occupied) == len(c):
        return probabilities
    ratios = slater_determinant_ratios(c, occupied)
    with np.errstate(over="ignore", invalid="ignore"):
        probabilities = np.maximum(2*dt*ratios.T.real*d[occupied], 0.)
    probabilities[:, occupied] = 0.
    if not np.isfinite(probabilities).all() or probabilities.sum() > maximum:
        raise ValueError("IESH probabilities exceed max_hop_probability; reduce the nuclear timestep")
    return probabilities


@dataclass(frozen=True)
class IESHHopResult:
    electron: int
    source: int
    target: int
    accepted: bool
    frustrated: bool
    energy_error: float


class IESHDynamics:
    """P=1 IESH with velocity Verlet nuclei and unitary electronic substeps.

    Electronic midpoint quadrature integrates the determinant current along
    the nuclear drift.  At most one hop is selected at the step endpoint;
    reducing electronic_substeps alone does not converge this hop timing.
    Failed upward hops keep momenta unchanged, following original IESH.

    A generic real DiabaticResult provider describes a ONE-electron matrix.
    Its optional bare_potential(q) must return a scalar PESResult, added once.
    Without it U0=0.  Providers remain caller-owned.  ``native`` selects only
    the existing free nuclear map; electronic algebra uses NumPy complex128.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates=None, *,
                 nelectrons=None, seed=0, native=True, threads=1,
                 integrator="cayley-bcb", electronic_substeps=4,
                 gap_tolerance=1e-10, max_hop_probability=1.):
        self.potential = potential
        self.nstates = _positive_integer(
            getattr(potential, "nstates", None) if nstates is None else nstates, "nstates")
        self.nelectrons = _positive_integer(
            max(self.nstates//2, 1) if nelectrons is None else nelectrons, "nelectrons")
        if self.nelectrons > self.nstates:
            raise ValueError("nelectrons cannot exceed nstates (spinless Pauli exclusion)")
        if isinstance(nbeads, (bool, np.bool_)) or nbeads != 1:
            raise ValueError("IESH supports classical nuclei, P=1 only")
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt,
                                          threads=threads, native=native, scheme=integrator)
        self.dt, self.beta, self.nbeads = self.integrator.dt, self.integrator.beta, 1
        if self.dt <= 0.:
            raise ValueError("IESH dt must be positive")
        self.masses = self.integrator.masses
        self.integrator_name, self.native = self.integrator.scheme, self.integrator.native
        self.electronic_substeps = _positive_integer(electronic_substeps, "electronic_substeps")
        self.gap_tolerance = _real(gap_tolerance, "gap_tolerance", positive=True)
        self.max_hop_probability = _real(max_hop_probability, "max_hop_probability", positive=True)
        if self.max_hop_probability > 1.:
            raise ValueError("max_hop_probability must not exceed one")
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.last_hop = None
        self.last_probabilities = np.zeros((self.nelectrons, self.nstates))
        self._parameters = self._parameter_values()

    def _parameter_values(self):
        return (self.dt, self.beta, self.nbeads, self.nstates, self.nelectrons,
                self.integrator_name, self.electronic_substeps,
                self.gap_tolerance, self.max_hop_probability)

    def _check_parameters(self):
        if (self._parameter_values() != self._parameters
                or self.dt != self.integrator.dt or self.beta != self.integrator.beta
                or self.integrator_name != self.integrator.scheme or self.integrator.nbeads != 1):
            raise ValueError("IESHDynamics parameters are immutable")

    def _check(self, state):
        self._check_parameters()
        if not isinstance(state, IESHState):
            raise ValueError("IESHDynamics requires IESHState")
        self.integrator._check(state.q, state.p)
        c, occupied = state.coefficients, state.occupied
        if (not isinstance(c, np.ndarray) or c.dtype != np.complex128
                or c.shape != (self.nstates, self.nelectrons)
                or not c.flags.c_contiguous or not c.flags.writeable):
            raise ValueError("invalid IESH coefficient shape/storage")
        _orbitals(c)
        if (not isinstance(occupied, np.ndarray) or occupied.dtype != np.int64
                or not occupied.flags.c_contiguous or not occupied.flags.writeable):
            raise ValueError("invalid IESH occupation storage")
        _occupied(occupied, self.nstates, self.nelectrons)
        _state_clock(state.step, state.time)
        if np.shares_memory(c, state.q) or np.shares_memory(c, state.p):
            raise ValueError("electronic and nuclear arrays must not share memory")

    def _adiabatic(self, q, couplings=False):
        result = validate_diabatic_result(self.potential.evaluate(q), q.shape, self.nstates)
        if (np.iscomplexobj(result.matrix) and np.any(result.matrix.imag != 0.)
                or np.iscomplexobj(result.gradients) and np.any(result.gradients.imag != 0.)):
            raise PotentialError("IESH requires real symmetric one-electron matrices; complex SOC is unsupported")
        energies, vectors = np.linalg.eigh(result.matrix.real)
        if not np.isfinite(energies).all() or not np.isfinite(vectors).all():
            raise PotentialError("IESH eigensystem evaluation overflowed")
        gaps = energies[:, None, :]-energies[:, :, None]
        mask = ~np.eye(self.nstates, dtype=bool)
        if np.any(np.abs(gaps[:, mask]) <= self.gap_tolerance):
            raise PotentialError("IESH adiabatic states are degenerate or below gap_tolerance")
        transformed = np.einsum("bsi,bstac,btj->bijac", vectors,
                                result.gradients.real, vectors, optimize=True)
        if not np.isfinite(transformed).all():
            raise PotentialError("IESH adiabatic gradients overflowed")
        gradients = np.einsum("biiac->biac", transformed).copy()
        if not couplings:
            return energies, vectors, gradients
        diagonal = np.arange(self.nstates)
        gaps[:, diagonal, diagonal] = 1.
        transformed /= gaps[..., None, None]
        transformed[:, diagonal, diagonal] = 0.
        if not np.isfinite(transformed).all():
            raise PotentialError("IESH derivative couplings overflowed")
        return energies, vectors, gradients, transformed

    def _bare(self, q):
        evaluate = getattr(self.potential, "bare_potential", None)
        if evaluate is None:
            return PESResult(np.zeros(len(q)), np.zeros_like(q))
        return validate_result(evaluate(q), q.shape)

    def initialize(self, nuclear, occupied=None, *, adiabatic_coefficients=None):
        self._check_parameters()
        self.integrator._check(nuclear.q, nuclear.p)
        occupied = _occupied(np.arange(self.nelectrons) if occupied is None else occupied,
                             self.nstates, self.nelectrons)
        _, u, _ = self._adiabatic(nuclear.q)
        c = np.eye(self.nstates, dtype=complex)[:, occupied] if adiabatic_coefficients is None else _orbitals(adiabatic_coefficients)
        if c.shape != (self.nstates, self.nelectrons):
            raise ValueError("adiabatic_coefficients shape disagrees with nstates/nelectrons")
        state = IESHState(nuclear.q, nuclear.p, nuclear.step, nuclear.time, u[0] @ c, occupied)
        self._check(state)
        return state

    def density_matrix(self, state, basis="diabatic"):
        """One-particle density, trace=nelectrons; not normalized to one."""
        self._check(state)
        c = state.coefficients
        if basis == "adiabatic":
            _, u, _ = self._adiabatic(state.q)
            c = u[0].T @ c
        elif basis != "diabatic":
            raise ValueError("basis must be diabatic or adiabatic")
        return c @ c.conj().T

    def populations(self, state):
        """Wavefunction adiabatic occupations (sum=nelectrons)."""
        return np.diag(self.density_matrix(state, "adiabatic")).real.copy()

    def active_populations(self, state):
        self._check(state)
        result = np.zeros(self.nstates)
        result[state.occupied] = 1.
        return result

    def energy(self, state):
        self._check(state)
        energies, _, _ = self._adiabatic(state.q)
        return float(np.sum(state.p**2/(2*self.masses[None, :, None]))
                     + self._bare(state.q).energy[0] + energies[0, state.occupied].sum())

    def _electronic_step(self, state, new_q):
        z = state.coefficients.copy()
        probabilities = np.zeros((self.nelectrons, self.nstates))
        velocity = (new_q[0]-state.q[0])/self.dt
        duration = self.dt/self.electronic_substeps
        # All quadrature geometries are known after the nuclear drift.  Evaluate
        # them in one provider batch and use NumPy's stacked eigensolver; only
        # the wavefunction products remain ordered in time.
        fractions = (np.arange(self.electronic_substeps)+0.5)/self.electronic_substeps
        q = state.q+(new_q-state.q)*fractions[:, None, None]
        energies, vectors, _, couplings = self._adiabatic(q, couplings=True)
        velocity_couplings = np.einsum("ac,bijac->bij", velocity, couplings, optimize=True)
        for k in range(self.electronic_substeps):
            c = vectors[k].T @ z
            phase = np.exp(-0.5j*duration*energies[k])[:, None]
            midpoint = c*phase
            probabilities += determinant_hopping_probabilities(
                midpoint, state.occupied, velocity_couplings[k], duration,
                max_probability=self.max_hop_probability)
            z = vectors[k] @ (midpoint*phase)
        if not np.isfinite(probabilities).all() or probabilities.sum() > self.max_hop_probability:
            raise ValueError("IESH probabilities exceed max_hop_probability; reduce the nuclear timestep")
        return z, probabilities

    def _hop(self, state, electron, target, *, adiabatic=None):
        if (isinstance(electron, (bool, np.bool_)) or not isinstance(electron, (int, np.integer))
                or not 0 <= electron < self.nelectrons):
            raise ValueError("electron must be an in-range integer")
        if (isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer))
                or not 0 <= target < self.nstates):
            raise ValueError("target must be an in-range orbital integer")
        source = int(state.occupied[electron])
        if target in state.occupied:
            raise ValueError("IESH hop target is already occupied (Pauli forbidden)")
        energies, _, _, coupling = self._adiabatic(state.q, couplings=True) if adiabatic is None else adiabatic
        gap = float(energies[0, target]-energies[0, source])
        # Work in mass-weighted coordinates; this conserves physical kinetic
        # energy for unequal nuclear masses, not merely the Euclidean speed.
        sqrt_mass = np.sqrt(self.masses[:, None])
        normal = coupling[0, source, target]/sqrt_mass
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm):
            raise ValueError("IESH mass-weighted hop direction overflowed")
        if norm == 0.:
            return IESHHopResult(int(electron), source, int(target), False, True, 0.)
        normal /= norm
        if normal.flat[np.argmax(np.abs(normal))] < 0.:
            normal *= -1.  # Gauge-independent zero-velocity downhill root.
        projection = float(np.sum(state.p[0]/sqrt_mass*normal))
        discriminant = projection*projection-2*gap
        if not np.isfinite(discriminant):
            raise ValueError("IESH momentum rescaling overflowed")
        if discriminant < 0.:
            return IESHHopResult(int(electron), source, int(target), False, True, 0.)
        root = np.copysign(np.sqrt(discriminant), projection if projection != 0. else 1.)
        # The rationalized root avoids cancellation for small energy changes.
        denominator = root+projection
        shift = -2*gap/denominator if denominator != 0. else 0.
        before = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        state.p[0] += sqrt_mass*shift*normal
        state.occupied[electron] = target
        after = float(np.sum(state.p**2/(2*self.masses[None, :, None])))
        return IESHHopResult(int(electron), source, int(target), True, False, after-before+gap)

    def attempt_hop(self, state, electron, target):
        """Deterministic single-electron energy-conserving hop for diagnostics."""
        self._check(state)
        trial = state.copy()
        result = self._hop(trial, electron, target)
        self._check(trial)
        state.p[:], state.occupied[:] = trial.p, trial.occupied
        self.last_hop = result
        return result

    def step(self, state):
        self._check(state)
        trial = state.copy()
        saved_rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            _, _, gradients = self._adiabatic(trial.q)
            force = self._bare(trial.q).forces-gradients[:, trial.occupied].sum(axis=1)
            trial.p += 0.5*self.dt*force
            self.integrator.free_step(trial.q, trial.p)
            energies, vectors, gradients, couplings = self._adiabatic(trial.q, couplings=True)
            force = self._bare(trial.q).forces-gradients[:, trial.occupied].sum(axis=1)
            trial.p += 0.5*self.dt*force
            z, probabilities = self._electronic_step(state, trial.q)
            trial.coefficients[:] = z
            hop = None
            if probabilities.sum() > 0.:
                index = int(np.searchsorted(np.cumsum(probabilities.ravel()), self.rng.random(), side="right"))
                if index < probabilities.size:
                    electron, target = divmod(index, self.nstates)
                    hop = self._hop(trial, electron, target,
                                    adiabatic=(energies, vectors, gradients, couplings))
            trial.step += 1
            trial.time += self.dt
            self._check(trial)
        except BaseException:
            self.rng.bit_generator.state = saved_rng
            raise
        state.q[:], state.p[:], state.coefficients[:] = trial.q, trial.p, trial.coefficients
        state.occupied[:] = trial.occupied
        state.step, state.time = trial.step, trial.time
        self.last_hop, self.last_probabilities = hop, probabilities.copy()
        return state
