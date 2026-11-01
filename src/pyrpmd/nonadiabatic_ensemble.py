"""Unweighted nonequilibrium ensembles for NRPMD and centroid RPSH.

Each trajectory contributes one sample to an ensemble statistic. Ring-polymer
beads and observation times are never treated as independent samples. Thermal
SEO parents and other signed or correlated ensembles require their own weighted
analysis and must not be passed to this API as independent trajectories.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import weakref

import numpy as np

from .dynamics import State
from .nonadiabatic import NRPMDDynamics, NRPMDState
from .surface_hopping import RPSHDynamics, RPSHState


def _integer(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < (1 if positive else 0)):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def density_matrix(state):
    """Return the electronic estimator in the fixed diabatic basis.

    NRPMD returns ``mean_b[(z_b z_b^dagger - I)/2]``. Its diagonal is the
    existing signed MMST population estimator; it is not generally positive
    semidefinite for a trajectory. RPSH returns ``c c^dagger`` from its stored
    diabatic amplitudes, independently of the active adiabatic surface.
    """
    if isinstance(state, NRPMDState):
        x, y = np.asarray(state.mapping_q), np.asarray(state.mapping_p)
        if (x.ndim != 2 or min(x.shape) < 1 or x.shape != y.shape
                or x.shape[0] != len(state.q)
                or x.dtype.kind not in "fiu" or y.dtype.kind not in "fiu"
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("NRPMD mapping variables must be finite real [P,S] arrays")
        z = x + 1j*y
        return (np.einsum("bi,bj->ij", z, z.conj()) / (2*len(z))
                - 0.5*np.eye(z.shape[1]))
    if isinstance(state, RPSHState):
        c = np.asarray(state.coefficients)
        if (c.ndim != 1 or not len(c) or c.dtype.kind not in "fciu"
                or not np.isfinite(c).all()
                or not np.isclose(np.vdot(c, c).real, 1., rtol=0., atol=1e-10)):
            raise ValueError("RPSH coefficients must be a finite normalized [S] vector")
        return np.outer(c, c.conj())
    raise TypeError("density_matrix requires NRPMDState or RPSHState")


class EnsemblePropagationError(RuntimeError):
    """A trajectory failed; no reduced or partially successful result is returned."""

    def __init__(self, trajectory_index, relative_step, stage, cause):
        self.trajectory_index = int(trajectory_index)
        self.relative_step = int(relative_step)
        self.stage = str(stage)
        super().__init__(
            f"nonadiabatic trajectory {trajectory_index} failed at relative step "
            f"{relative_step} during {stage}: {cause}"
        )


@dataclass(frozen=True)
class NonadiabaticEnsembleResult:
    """Per-trajectory observations, using relative times in atomic units.

    ``diabatic_density`` is [trajectory, frame, state, state], ``energy`` is
    [trajectory, frame], and ``electronic_norm`` is [trajectory, frame, bead]
    for NRPMD or [trajectory, frame, 1] for RPSH. Norm means the squared mapping
    or amplitude norm, without the MMST zero-point subtraction.

    SEMs exist only when independence was explicitly asserted and at least two
    trajectories are present. Complex coherence errors are reported separately
    for real and imaginary components. These are pointwise sampling errors,
    not a confidence band or a measure of method/finite-timestep error.
    """

    method: str
    time: np.ndarray
    sample_steps: np.ndarray
    start_times: np.ndarray
    start_steps: np.ndarray
    diabatic_density: np.ndarray
    energy: np.ndarray
    electronic_norm: np.ndarray
    final_states: tuple[State, ...]
    rng_states: tuple[dict, ...]
    independent: bool = False
    adiabatic_amplitude_populations: np.ndarray | None = None
    active_surfaces: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in ("nrpmd", "rpsh"):
            raise ValueError("unknown nonadiabatic ensemble method")
        if not isinstance(self.independent, (bool, np.bool_)):
            raise ValueError("independent must be a boolean")
        converted = {}
        for name in ("time", "sample_steps", "start_times", "start_steps",
                     "diabatic_density", "energy", "electronic_norm"):
            value = np.array(getattr(self, name), copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            converted[name] = value
        density = converted["diabatic_density"]
        if density.ndim != 4 or min(density.shape) < 1 or density.shape[2] != density.shape[3]:
            raise ValueError("diabatic_density must have shape [trajectory,frame,S,S]")
        trajectories, frames, states, _ = density.shape
        time, indices = converted["time"], converted["sample_steps"]
        if (time.shape != (frames,) or indices.shape != (frames,)
                or indices.dtype.kind not in "iu" or time[0] != 0 or indices[0] != 0
                or np.any(np.diff(time) <= 0) or np.any(np.diff(indices) <= 0)
                or converted["start_times"].shape != (trajectories,)
                or converted["start_steps"].shape != (trajectories,)
                or converted["start_steps"].dtype.kind not in "iu"
                or np.any(converted["start_steps"] < 0)
                or converted["energy"].shape != (trajectories, frames)
                or converted["electronic_norm"].ndim != 3
                or converted["electronic_norm"].shape[:2] != (trajectories, frames)
                or converted["electronic_norm"].shape[2] < 1):
            raise ValueError("inconsistent nonadiabatic ensemble observation shapes")
        if not np.allclose(density, density.swapaxes(-1, -2).conj(), rtol=0., atol=1e-10):
            raise ValueError("diabatic density estimators must be Hermitian")
        if len(self.final_states) != trajectories or len(self.rng_states) != trajectories:
            raise ValueError("one final state and RNG record are required per trajectory")
        expected = NRPMDState if self.method == "nrpmd" else RPSHState
        if not all(isinstance(state, expected) for state in self.final_states):
            raise ValueError("final state types do not match the ensemble method")
        if self.method == "rpsh":
            amplitude = np.array(self.adiabatic_amplitude_populations, dtype=float, copy=True)
            active = np.array(self.active_surfaces, copy=True)
            if (amplitude.shape != (trajectories, frames, states)
                    or not np.isfinite(amplitude).all()
                    or np.any(amplitude < 0)
                    or not np.allclose(amplitude.sum(axis=-1), 1., atol=1e-10, rtol=0.)
                    or active.shape != (trajectories, frames) or active.dtype.kind not in "iu"
                    or np.any(active < 0) or np.any(active >= states)
                    or converted["electronic_norm"].shape[2] != 1):
                raise ValueError("invalid RPSH amplitude populations or active surfaces")
            converted.update(adiabatic_amplitude_populations=amplitude, active_surfaces=active)
        elif self.adiabatic_amplitude_populations is not None or self.active_surfaces is not None:
            raise ValueError("NRPMD does not assign an active adiabatic surface")
        for name, value in converted.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "independent", bool(self.independent))
        object.__setattr__(self, "final_states", tuple(state.copy() for state in self.final_states))
        object.__setattr__(self, "rng_states", copy.deepcopy(tuple(self.rng_states)))
        object.__setattr__(self, "metadata", copy.deepcopy(self.metadata))

    @property
    def ntrajectories(self):
        return len(self.diabatic_density)

    @property
    def sem_unavailable_reason(self):
        if not self.independent:
            return "trajectory_independence_not_asserted"
        if self.ntrajectories < 2:
            return "fewer_than_two_trajectories"
        return None

    def _sem(self, values):
        if self.sem_unavailable_reason is not None:
            return None
        return np.std(values, axis=0, ddof=1) / np.sqrt(self.ntrajectories)

    @property
    def populations(self):
        """Fixed-diabatic populations [trajectory,frame,state]; signed for MMST."""
        return self.diabatic_density.diagonal(axis1=-2, axis2=-1).real

    @property
    def mean_density(self):
        return self.diabatic_density.mean(axis=0)

    @property
    def mean_populations(self):
        return self.populations.mean(axis=0)

    @property
    def population_sem(self):
        return self._sem(self.populations)

    @property
    def density_sem_real(self):
        return self._sem(self.diabatic_density.real)

    @property
    def density_sem_imag(self):
        return self._sem(self.diabatic_density.imag)

    @property
    def energy_drift(self):
        return self.energy-self.energy[:, :1]

    @property
    def norm_drift(self):
        return self.electronic_norm-self.electronic_norm[:, :1]

    @property
    def mean_adiabatic_amplitude_populations(self):
        if self.adiabatic_amplitude_populations is None:
            return None
        return self.adiabatic_amplitude_populations.mean(axis=0)

    @property
    def adiabatic_amplitude_population_sem(self):
        if self.adiabatic_amplitude_populations is None:
            return None
        return self._sem(self.adiabatic_amplitude_populations)

    @property
    def active_surface_counts(self):
        """Integer counts of active centroid-adiabatic surfaces [frame,state]."""
        if self.active_surfaces is None:
            return None
        states = np.arange(self.diabatic_density.shape[-1])
        return np.count_nonzero(self.active_surfaces[..., None] == states, axis=0)

    @property
    def active_surface_fractions(self):
        counts = self.active_surface_counts
        return None if counts is None else counts/self.ntrajectories

    @property
    def active_surface_sem(self):
        if self.active_surfaces is None:
            return None
        states = np.arange(self.diabatic_density.shape[-1])
        return self._sem(self.active_surfaces[..., None] == states)


def _signature(dynamics):
    # Exact classes exclude a thermal/sign-weighted propagation subclass from
    # accidentally being analyzed with the unweighted nonequilibrium API.
    if type(dynamics) not in (NRPMDDynamics, RPSHDynamics):
        raise TypeError("dynamics_factory must return NRPMDDynamics or RPSHDynamics")
    dt = dynamics.dt
    if (isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt)
            or not np.isrealobj(dt) or not np.isfinite(dt) or dt <= 0):
        raise ValueError("ensemble dynamics dt must be positive and finite")
    common = (type(dynamics), float(dt), dynamics.beta, dynamics.nbeads,
              dynamics.nstates, tuple(dynamics.masses), dynamics.integrator_name)
    if isinstance(dynamics, RPSHDynamics):
        dynamics._check_parameters()
        return common + (dynamics.electronic_substeps, dynamics.rescaling,
                         dynamics.frustrated_hops, dynamics.gap_tolerance,
                         getattr(dynamics, "decoherence", "none"),
                         getattr(dynamics, "decoherence_constant", None))
    return common


def _initial_state(parent, dynamics, rng, initial_state, active_surface):
    if isinstance(dynamics, NRPMDDynamics):
        if active_surface is not None:
            raise ValueError("active_surface is only an RPSH initialization option")
        if isinstance(parent, NRPMDState):
            if initial_state is not None:
                raise ValueError("initial_state cannot overwrite an already prepared NRPMDState")
            state = parent.copy()
        elif type(parent) is State:
            if initial_state is None:
                raise ValueError("nuclear NRPMD parents require an explicit initial_state")
            state = NRPMDState.from_nuclear(parent, dynamics.nstates, initial_state, rng)
        else:
            raise TypeError("NRPMD parents must be State or NRPMDState")
    else:
        if initial_state is not None:
            raise ValueError("initial_state is only an NRPMD initialization option")
        if isinstance(parent, RPSHState):
            if active_surface is not None:
                raise ValueError("active_surface cannot overwrite an already prepared RPSHState")
            state = parent.copy()
        elif type(parent) is State:
            if active_surface is None:
                raise ValueError("nuclear RPSH parents require an explicit active_surface")
            state = dynamics.initialize(parent, active_surface=active_surface)
        else:
            raise TypeError("RPSH parents must be State or RPSHState")
    dynamics._check(state)
    return state


def run_nonadiabatic_ensemble(parents, dynamics_factory, *, steps, sample_stride=1,
                              seed=0, initial_state=None, active_surface=None,
                              independent=False):
    """Propagate a fixed-size, unweighted nonequilibrium ensemble.

    ``dynamics_factory(index, rng)`` must return a fresh driver per trajectory,
    using the same physical Hamiltonian and numerical parameters. For RPSH,
    construct it with ``seed=rng``; the runner checks RNG identity and does not
    replace or reset it. A factory may restore that RNG's saved state before
    returning a driver. Reused drivers and mixed method/parameter sets fail.

    ``parents`` contains prepared NRPMDState/RPSHState objects, or nuclear State
    parents with explicit ``initial_state`` (focused diabatic NRPMD) or
    ``active_surface`` (pure centroid-adiabatic RPSH). Parent states are copied
    before factory calls and are never advanced in place. Providers are
    borrowed and are never closed by this function.

    Observations include relative time zero and the final step even when it
    is not divisible by sample_stride. Separate initialization and propagation
    streams are spawned for each trajectory from the nonnegative integer seed.
    Means are always available; SEMs require ``independent=True`` and N>=2.
    A failed trajectory raises EnsemblePropagationError with its index, step
    and original cause. Failed trajectories are not silently excluded.
    """
    steps = _integer(steps, "steps")
    stride = _integer(sample_stride, "sample_stride", positive=True)
    seed = _integer(seed, "seed")
    if not isinstance(independent, (bool, np.bool_)):
        raise ValueError("independent must be a boolean")
    if not callable(dynamics_factory):
        raise TypeError("dynamics_factory must be callable")
    parents = tuple(parents)
    if not parents or not all(type(state) in (State, NRPMDState, RPSHState) for state in parents):
        raise ValueError("parents must be a nonempty sequence of supported state objects")
    parents = tuple(state.copy() for state in parents)
    indices = np.arange(0, steps+1, stride, dtype=np.int64)
    if indices[-1] != steps:
        indices = np.append(indices, steps)
    streams = np.random.SeedSequence(seed).spawn(len(parents))
    # Remember live drivers without retaining every native integrator context.
    # A factory that can return the same object again necessarily keeps it alive.
    drivers = weakref.WeakSet()
    records, final_states, rng_states = [], [], []
    reference, time, first = None, None, None
    for index, (parent, sequence) in enumerate(zip(parents, streams)):
        relative_step, stage = 0, "factory"
        initialization_sequence, dynamics_sequence = sequence.spawn(2)
        initialization_rng = np.random.default_rng(initialization_sequence)
        dynamics_rng = np.random.default_rng(dynamics_sequence)
        try:
            dynamics = dynamics_factory(index, dynamics_rng)
            if dynamics in drivers:
                raise ValueError("dynamics_factory reused a driver; return a fresh driver per trajectory")
            signature = _signature(dynamics)
            drivers.add(dynamics)
            if reference is None:
                reference = signature
                first = dynamics
                time = indices*dynamics.dt
                if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
                    raise ValueError("observation times overflow or cannot be resolved")
            elif signature != reference:
                raise ValueError("ensemble dynamics parameters/method must match across trajectories")
            if isinstance(dynamics, RPSHDynamics) and dynamics.rng is not dynamics_rng:
                raise ValueError("RPSH dynamics_factory must use the supplied rng as seed=rng")
            stage = "initialization"
            state = _initial_state(parent, dynamics, initialization_rng, initial_state, active_surface)
            density, energy, norms, amplitudes, surfaces = [], [], [], [], []
            for target in indices:
                stage = "propagation"
                while relative_step < target:
                    relative_step += 1
                    dynamics.step(state)
                stage = "observation"
                density.append(density_matrix(state))
                energy.append(dynamics.energy(state))
                if isinstance(state, NRPMDState):
                    norms.append(np.sum(state.mapping_q**2+state.mapping_p**2, axis=1))
                else:
                    norms.append([float(np.vdot(state.coefficients, state.coefficients).real)])
                    amplitudes.append(dynamics.populations(state))
                    surfaces.append(state.active_surface)
                if (not np.isfinite(density[-1]).all() or not np.isfinite(energy[-1])
                        or not np.isfinite(norms[-1]).all()):
                    raise FloatingPointError("nonfinite ensemble density, energy or electronic norm")
            records.append((density, energy, norms, amplitudes, surfaces))
            final_states.append(state)
            rng_states.append({
                "initialization": copy.deepcopy(initialization_rng.bit_generator.state),
                "dynamics": copy.deepcopy(dynamics_rng.bit_generator.state),
            })
        except Exception as exc:
            raise EnsemblePropagationError(index, relative_step, stage, exc) from exc
    rpsh = isinstance(first, RPSHDynamics)
    metadata = {
        "dt": first.dt, "beta": first.beta, "nbeads": first.nbeads,
        "nstates": first.nstates, "masses": first.masses.tolist(),
        "integrator": first.integrator_name, "seed": seed,
        "time_unit": "atomic_time", "energy_unit": "hartree",
        "density_basis": "fixed_diabatic", "statistics_axis": "trajectory",
        "weighting": "unweighted", "physical_converged": False,
        "sample_stride": stride,
    }
    if rpsh:
        metadata.update(active_surface_basis="centroid_adiabatic",
                        amplitude_population_basis="centroid_adiabatic",
                        rescaling=first.rescaling, frustrated_hops=first.frustrated_hops,
                        electronic_substeps=first.electronic_substeps,
                        decoherence=getattr(first, "decoherence", "none"),
                        decoherence_constant=getattr(first, "decoherence_constant", None))
    return NonadiabaticEnsembleResult(
        method="rpsh" if rpsh else "nrpmd", time=time, sample_steps=indices,
        start_times=np.array([state.time for state in parents]),
        start_steps=np.array([state.step for state in parents], dtype=np.int64),
        diabatic_density=np.array([record[0] for record in records]),
        energy=np.array([record[1] for record in records]),
        electronic_norm=np.array([record[2] for record in records]),
        final_states=tuple(final_states), rng_states=tuple(rng_states),
        independent=independent,
        adiabatic_amplitude_populations=np.array([record[3] for record in records]) if rpsh else None,
        active_surfaces=np.array([record[4] for record in records], dtype=np.int64) if rpsh else None,
        metadata=metadata,
    )
