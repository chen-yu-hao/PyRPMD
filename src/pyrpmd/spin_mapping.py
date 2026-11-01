"""Two-state W-spin SM-NRPMD (Bossion, Chowdhury and Huo, 2021).

The propagation Hamiltonian and Stratonovich--Weyl estimators are those of
doi:10.1063/5.0051456.  ``from_nuclear`` prepares focused nonequilibrium
electronic populations; it does not sample the paper's electronic thermal
trace.  Nuclear coordinates, momenta, energies and time use atomic units.
"""
from dataclasses import dataclass

import numpy as np

from .dynamics import CayleyIntegrator, State, _state_clock
from .electronic_backends import electronic_mapping_flow, validate_electronic_backend
from .nonadiabatic import _positive_integer, validate_diabatic_result


W_RADIUS = np.sqrt(3.0) / 2.0


def _unit_vectors(value, shape, name="spin"):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be finite real vectors with shape {shape}")
    result = np.array(raw, dtype=np.float64, order="C", copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.sum(result * result, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} vectors must have unit norm")
    return result


def _two_state_index(value, name="initial_state"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _pauli_components(matrix, gradients):
    """V = h0 I + hx sigma_x + hy sigma_y + hz sigma_z, including SOC."""
    scalar = 0.5 * (matrix[:, 0, 0].real + matrix[:, 1, 1].real)
    field = np.stack((matrix[:, 0, 1].real, -matrix[:, 0, 1].imag,
                      0.5 * (matrix[:, 0, 0].real - matrix[:, 1, 1].real)), axis=1)
    dscalar = 0.5 * (gradients[:, 0, 0].real + gradients[:, 1, 1].real)
    dfield = np.stack((gradients[:, 0, 1].real, -gradients[:, 0, 1].imag,
                       0.5 * (gradients[:, 0, 0].real - gradients[:, 1, 1].real)), axis=1)
    return scalar, field, dscalar, dfield


def _spin_kernel(spin, radius):
    """Construct I/2 + radius * spin.sigma for one vector or a batch."""
    spin = np.asarray(spin)
    result = np.empty((*spin.shape[:-1], 2, 2), dtype=np.complex128)
    result[..., 0, 0] = 0.5 + radius * spin[..., 2]
    result[..., 1, 1] = 0.5 - radius * spin[..., 2]
    result[..., 0, 1] = radius * (spin[..., 0] - 1j * spin[..., 1])
    result[..., 1, 0] = result[..., 0, 1].conj()
    return result


@dataclass
class SMNRPMDState(State):
    """A nuclear ring and one unit Bloch vector per bead in a fixed basis."""
    spin: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        self.spin = _unit_vectors(self.spin, (len(self.q), 3))

    @classmethod
    def from_nuclear(cls, state, initial_state=0, rng=0):
        """Focused W-spin population: uz = +/-1/sqrt(3), uniform azimuth.

        The azimuth is drawn independently for every bead.  A single sample
        has nonzero transverse coherence; its ensemble mean is zero.
        """
        if not isinstance(state, State):
            raise ValueError("from_nuclear requires a State")
        initial_state = _two_state_index(initial_state)
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        phi = rng.uniform(0.0, 2.0 * np.pi, len(state.q))
        z = (1.0 if initial_state == 0 else -1.0) / np.sqrt(3.0)
        radius = np.sqrt(1.0 - z * z)
        spin = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), np.full(len(phi), z)))
        return cls(state.q, state.p, state.step, state.time, spin)

    def copy(self):
        return SMNRPMDState(self.q, self.p, self.step, self.time, self.spin)

    def density_matrix(self):
        """Bead-averaged W kernel, signed and Hermitian with unit trace."""
        return _spin_kernel(self.spin.mean(axis=0), W_RADIUS)

    def populations(self):
        return self.density_matrix().diagonal().real.copy()


def _spin_mint(state, result, duration):
    """Exact fixed-coordinate spin flow and its integrated nuclear impulse."""
    _, field, dscalar, dfield = _pauli_components(result.matrix, result.gradients)
    magnitude = np.linalg.norm(field, axis=1)
    axis = np.divide(field, magnitude[:, None], out=np.zeros_like(field),
                     where=magnitude[:, None] > 0.0)
    parallel = np.sum(state.spin * axis, axis=1)[:, None] * axis
    perpendicular = state.spin - parallel
    cross = np.cross(axis, state.spin)
    angle = 2.0 * duration * magnitude
    integrated = duration * (parallel + np.sinc(angle / np.pi)[:, None] * perpendicular
                             + (0.5 * angle * np.sinc(angle / (2.0 * np.pi)) ** 2)[:, None] * cross)
    state.p -= duration * dscalar + 2.0 * W_RADIUS * np.einsum("bi,biac->bac", integrated, dfield)
    state.spin[:] = parallel + np.cos(angle)[:, None] * perpendicular + np.sin(angle)[:, None] * cross


def _spin_to_mapping(spin):
    """Stable north/south spinor charts, with ||z||^2 = 2*sqrt(3).

    The chart's arbitrary overall phase drops out of the propagation force
    and spin.  Unit normalization removes only validated roundoff drift.
    """
    unit = spin / np.linalg.norm(spin, axis=-1, keepdims=True)
    z = np.empty((*unit.shape[:-1], 2), dtype=np.complex128)
    north = unit[..., 2] >= 0.0
    a = np.sqrt((1.0 + unit[..., 2]) / 2.0)
    b = np.sqrt((1.0 - unit[..., 2]) / 2.0)
    z[..., 0] = np.where(north, a, np.divide(unit[..., 0] - 1j*unit[..., 1], 2*b,
                                           out=np.zeros_like(a, dtype=complex), where=~north))
    z[..., 1] = np.where(north, np.divide(unit[..., 0] + 1j*unit[..., 1], 2*a,
                                         out=np.zeros_like(a, dtype=complex), where=north), b)
    return np.sqrt(4.0 * W_RADIUS) * z


def _mapping_to_spin(z):
    norm = np.sum(np.abs(z)**2, axis=-1)
    if not np.allclose(norm, 4.0 * W_RADIUS, rtol=0.0, atol=1e-8):
        raise ValueError("electronic backend did not preserve the W-spin mapping norm")
    off = z[..., 0] * z[..., 1].conj()
    return np.stack((2.0*off.real, -2.0*off.imag, np.abs(z[..., 0])**2-np.abs(z[..., 1])**2), axis=-1) / norm[..., None]


class SMNRPMDDynamics:
    """Two-state W-spin SM-NRPMD with exact spin/impulse split propagation.

    ``H = sum_b [K_b + spring_b + h0(q_b) + sqrt(3)*h(q_b).u_b]``.
    The nuclear free-ring map is the selected Cayley, normal-mode or Verlet
    scheme.  No clipping of signed population estimates is performed.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, nstates=2, *,
                 threads=1, native=True, integrator="cayley-bcb",
                 electronic_backend="numpy", electronic_device="cpu"):
        if _positive_integer(nstates, "nstates") != 2:
            raise ValueError("SMNRPMDDynamics implements exactly two electronic states")
        self.nstates = 2
        self.potential = potential
        self.integrator = CayleyIntegrator(masses, beta, nbeads, dt, threads=threads,
                                          native=native, scheme=integrator)
        self.masses = self.integrator.masses
        self.beta, self.dt, self.nbeads = float(beta), float(dt), int(nbeads)
        self.integrator_name, self.native = self.integrator.scheme, self.integrator.native
        validate_electronic_backend(electronic_backend, electronic_device)
        self.electronic_backend, self.electronic_device = electronic_backend, electronic_device
        self.threads = self.integrator.threads
        self._parameters = (self.beta, self.dt, self.nbeads, self.nstates, self.integrator_name,
                            self.electronic_backend, self.electronic_device, self.threads)

    def _check(self, state):
        if ((self.beta, self.dt, self.nbeads, self.nstates, self.integrator_name,
             self.electronic_backend, self.electronic_device, self.threads) != self._parameters
                or self.dt != self.integrator.dt or self.beta != self.integrator.beta
                or self.nbeads != self.integrator.nbeads or self.integrator_name != self.integrator.scheme):
            raise ValueError("SMNRPMDDynamics parameters are immutable")
        if not isinstance(state, SMNRPMDState):
            raise ValueError("SMNRPMDDynamics requires SMNRPMDState")
        self.integrator._check(state.q, state.p)
        _state_clock(state.step, state.time)
        value = state.spin
        if (not isinstance(value, np.ndarray) or value.shape != (self.nbeads, 3)
                or value.dtype != np.float64 or not value.flags.writeable or not value.flags.c_contiguous
                or not np.isfinite(value).all()
                or not np.allclose(np.sum(value * value, axis=1), 1.0, rtol=0.0, atol=1e-8)):
            raise ValueError("invalid unit spin vectors")
        arrays = (state.q, state.p, state.spin)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i+1:]):
            raise ValueError("state arrays must not share memory")

    def _evaluate(self, q):
        return validate_diabatic_result(self.potential.evaluate(q), q.shape, 2)

    def initialize(self, nuclear_state, initial_state=0, rng=0):
        state = SMNRPMDState.from_nuclear(nuclear_state, initial_state, rng)
        self._check(state)
        return state

    def _mint(self, state, result, duration):
        if self.electronic_backend == "numpy":
            _spin_mint(state, result, duration)
        else:
            z, delta = electronic_mapping_flow(_spin_to_mapping(state.spin), result.matrix, result.gradients,
                                               duration, gamma=2.0*W_RADIUS-1.0,
                                               backend=self.electronic_backend, device=self.electronic_device,
                                               threads=self.threads)
            state.spin[:] = _mapping_to_spin(z)
            state.p += delta

    def step(self, state):
        self._check(state)
        trial = state.copy()
        self._mint(trial, self._evaluate(trial.q), 0.5 * self.dt)
        self.integrator.free_step(trial.q, trial.p)
        self._mint(trial, self._evaluate(trial.q), 0.5 * self.dt)
        trial.step += 1
        trial.time += self.dt
        self._check(trial)
        for key in ("q", "p", "spin"):
            getattr(state, key)[:] = getattr(trial, key)
        state.step, state.time = trial.step, trial.time
        return state

    def energy(self, state):
        """Sum-bead propagation energy; this is not a thermal estimator."""
        self._check(state)
        result = self._evaluate(state.q)
        scalar, field, _, _ = _pauli_components(result.matrix, result.gradients)
        electronic = np.sum(scalar + 2.0 * W_RADIUS * np.sum(field * state.spin, axis=1))
        kinetic = np.sum(state.p ** 2 / (2.0 * self.masses[None, :, None]))
        dq = state.q - np.roll(state.q, -1, axis=0)
        spring = 0.5 * (self.nbeads / self.beta) ** 2 * np.sum(self.masses[None, :, None] * dq ** 2)
        return float(electronic + kinetic + spring)

    def density_matrix(self, state):
        self._check(state)
        return state.density_matrix()

    def populations(self, state):
        return self.density_matrix(state).diagonal().real.copy()
