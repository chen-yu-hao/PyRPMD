"""Transition-interface ensembles sampled by canonical two-way shooting.

The paths use unthermostatted, unbiased, reversible :class:`Dynamics`.  Every
Monte Carlo attempt contributes a path, including rejected attempts.  Uniform
selection among all frames requires the old/new total frame-count ratio.
This is TIS, not a collection of independent committor shots.  See
``docs/transition-interface-sampling.md`` for conventions and limitations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .analysis.fes import statistical_inefficiency
from .constants import ATOMIC_TIME_S
from .dynamics import (
    CayleyIntegrator, Dynamics, State,
    _BLOCK_DYNAMICS_METHODS, _BLOCK_INTEGRATOR_METHODS,
)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError("path arrays and interfaces must contain real numbers")
    result = np.array(raw, dtype=float, copy=True, order="C")
    result.flags.writeable = False
    return result


class PathGenerationError(RuntimeError):
    """A proposed/seed path does not terminate in the requested ensemble."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class TISPath:
    """Owned path snapshots in forward time, including exactly two endpoints.

    Arrays have shapes ``q,p[T,P,N,3]`` and ``values[T]``.  Endpoints lie in A
    and A or B; interior snapshots lie in neither basin.  The sampler validates
    basin membership, coordinate values and dynamics when accepting a seed.
    State clocks are reconstructed relative to the first snapshot.
    """

    q: np.ndarray
    p: np.ndarray
    values: np.ndarray
    dt: float

    def __post_init__(self):
        q, p, values = (_readonly(x) for x in (self.q, self.p, self.values))
        dt = _finite(self.dt, "dt")
        if (
            q.ndim != 4 or q.shape[0] < 2 or min(q.shape[1:]) < 1
            or q.shape[-1] != 3 or p.shape != q.shape
            or values.shape != (len(q),) or dt <= 0
            or not all(np.isfinite(x).all() for x in (q, p, values))
        ):
            raise ValueError("path requires finite q,p[T,P,N,3], values[T], T>=2, and dt>0")
        for name, value in (("q", q), ("p", p), ("values", values), ("dt", dt)):
            object.__setattr__(self, name, value)

    def __len__(self):
        return len(self.q)

    @property
    def interior_frames(self):
        return len(self) - 2

    def state(self, index):
        index = _integer(index, "index")
        if index >= len(self):
            raise IndexError("path snapshot out of range")
        return State(self.q[index], self.p[index], step=index, time=index * self.dt)

    def reversed(self):
        return TISPath(self.q[::-1], -self.p[::-1], self.values[::-1], self.dt)


@dataclass(frozen=True)
class TISMove:
    accepted: bool
    reason: str
    shooting_index: int
    trial_frames: int | None
    log_acceptance_ratio: float | None


@dataclass
class TISResult:
    """A correlated path Monte Carlo chain; repeated paths are retained.

    Only the final full path is stored by default.  ``on_sample`` in ``sample``
    can stream paths to caller-owned storage without accumulating all frames.
    """

    interface_index: int
    interface: float
    target: float
    target_is_product: bool
    basin_a: float
    basin_b: float
    crossing: np.ndarray
    accepted: np.ndarray
    path_frames: np.ndarray
    reasons: tuple[str, ...]
    final_path: TISPath
    rng_state: dict
    burn_in: int
    max_path_steps: int
    length_rejections: int
    beta: float
    nbeads: int

    def summary(self):
        n = len(self.crossing)
        probability = float(np.mean(self.crossing))
        varied = bool(np.any(self.crossing != self.crossing[0]))
        g = statistical_inefficiency(self.crossing.astype(float))
        if np.any(self.path_frames != self.path_frames[0]):
            g = max(g, statistical_inefficiency(self.path_frames))
        standard_error = (
            float(np.sqrt(np.var(self.crossing, ddof=1) * g / n))
            if n > 1 and varied else None
        )
        return {
            "method": "two_way_canonical_tis",
            "interface_index": self.interface_index,
            "interface": self.interface,
            "target": self.target,
            "target_is_product": self.target_is_product,
            "basin_a": self.basin_a,
            "basin_b": self.basin_b,
            "dt_au": self.final_path.dt,
            "beta_au": self.beta,
            "nbeads": self.nbeads,
            "samples": n,
            "burn_in": self.burn_in,
            "crossing_probability": probability,
            "standard_error_diagnostic": standard_error,
            "statistical_inefficiency": float(g),
            "effective_samples_diagnostic": float(n / g),
            "acceptance_fraction": float(np.mean(self.accepted)),
            "mean_path_frames": float(np.mean(self.path_frames)),
            "max_path_steps": self.max_path_steps,
            "length_rejections": self.length_rejections,
            "constant_crossing_indicator": not varied,
            "converged": False,
        }


class TransitionInterfaceSampler:
    """Sample A-originating interface ensembles for real-mass RPMD/classical MD.

    ``coordinate`` is a scalar PyRPMD CV or a callable ``coordinate(q)``.  No
    derivatives are used.  A is ``s <= basin_a`` and B is ``s >= basin_b``.
    Interfaces must satisfy ``basin_a <= lambda[0] < ... < basin_b``; crossing
    an interface means ``s > lambda``.  The coordinate need not approximate a
    committor, but must distinguish the two stable basins.

    ``max_path_steps`` bounds the complete path, not each half separately.
    Exceeding it rejects the proposal: this samples a length-truncated ensemble
    and must be checked by increasing the bound.  Potential/CV errors propagate
    to the caller, rather than silently counting numerical failures as rejection.

    Bare deterministic Dynamics is required.  A thermostat or a time-dependent
    potential would require a different path probability and is not supported.
    """

    def __init__(
        self, dynamics, coordinate, *, basin_a, basin_b, interfaces,
        max_path_steps=10000, seed=0,
    ):
        if type(dynamics) is not Dynamics or type(dynamics.integrator) is not CayleyIntegrator:
            raise TypeError("TIS requires the built-in reversible Dynamics and integrator")
        dynamics._check_parameters()
        if dynamics.dt <= 0:
            raise ValueError("TIS requires a positive dynamics timestep")
        if len(dynamics.fixed_atoms) == len(dynamics.masses):
            raise ValueError("TIS needs at least one mobile atom")
        if callable(getattr(coordinate, "value", None)):
            if getattr(coordinate, "dimension", 1) != 1:
                raise ValueError("TIS requires a scalar order parameter")
            if any(period is not None for period in getattr(coordinate, "periods", (None,))):
                raise ValueError("TIS requires an ordered, nonperiodic scalar coordinate")
            self._coordinate = coordinate.value
        elif callable(coordinate):
            self._coordinate = coordinate
        else:
            raise TypeError("coordinate must be a scalar CV or callable on q")
        self.dynamics = dynamics
        self.basin_a = _finite(basin_a, "basin_a")
        self.basin_b = _finite(basin_b, "basin_b")
        raw_interfaces = np.asarray(interfaces)
        if raw_interfaces.dtype.kind not in "fiu":
            raise ValueError("interfaces must be finite ordered real numbers")
        self.interfaces = _readonly(interfaces)
        if (
            self.interfaces.ndim != 1 or not len(self.interfaces)
            or not np.isfinite(self.interfaces).all()
            or self.basin_a >= self.basin_b
            or self.interfaces[0] < self.basin_a
            or self.interfaces[-1] >= self.basin_b
            or np.any(np.diff(self.interfaces) <= 0)
        ):
            raise ValueError("interfaces require basin_a <= lambda[0] < ... < basin_b")
        self.max_path_steps = _integer(max_path_steps, "max_path_steps", 1)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._signature = self._dynamics_signature()
        self._check_dynamics()

    def _dynamics_signature(self):
        dyn = self.dynamics
        return (
            id(dyn.potential), dyn.beta, dyn.nbeads, dyn.dt, dyn.integrator_name,
            tuple(dyn.masses), dyn.fixed_atoms,
            None if dyn.fixed_positions is None else dyn.fixed_positions.tobytes(),
        )

    def _check_dynamics(self):
        dyn = self.dynamics
        dyn._check_parameters()
        if self._dynamics_signature() != self._signature:
            raise ValueError("Dynamics changed after TIS construction; create a new sampler")
        for obj, cls, methods in (
            (dyn, Dynamics, _BLOCK_DYNAMICS_METHODS),
            (dyn.integrator, CayleyIntegrator, _BLOCK_INTEGRATOR_METHODS),
        ):
            if type(obj) is not cls or any(
                name in obj.__dict__ or getattr(cls, name) is not original
                for name, original in methods.items()
            ):
                raise TypeError("TIS does not support overridden/thermostatted dynamics methods")
        metadata = getattr(dyn.potential, "metadata", {})
        capabilities = metadata.get("capabilities", {}) if isinstance(metadata, dict) else {}
        capabilities = capabilities if isinstance(capabilities, dict) else {}
        if (
            getattr(dyn.potential, "history_dependent", False)
            or getattr(dyn.potential, "time_dependent", False)
            or getattr(dyn.potential, "stateful", False)
            or (isinstance(metadata, dict) and metadata.get("stateful", False))
            or capabilities.get("stateful", False)
        ):
            raise ValueError("TIS requires a deterministic time-independent potential")

    def value(self, q):
        view = np.asarray(q).view()
        view.flags.writeable = False
        result = np.asarray(self._coordinate(view))
        if result.dtype.kind not in "fiu" or result.size != 1 or not np.isfinite(result).all():
            raise ValueError("TIS coordinate must return one finite real value")
        return float(result.reshape(-1)[0])

    def _index(self, index):
        index = _integer(index, "interface_index")
        if index >= len(self.interfaces):
            raise ValueError("interface_index out of range")
        return index

    def _basin(self, value):
        return -1 if value <= self.basin_a else 1 if value >= self.basin_b else 0

    def _member(self, path, index):
        v = path.values
        return bool(
            len(path) - 1 <= self.max_path_steps
            and v[0] <= self.basin_a
            and (v[-1] <= self.basin_a or v[-1] >= self.basin_b)
            and np.all((v[1:-1] > self.basin_a) & (v[1:-1] < self.basin_b))
            and np.any(v > self.interfaces[index])
        )

    def validate_path(self, path, interface_index=0):
        """Validate seed membership and replay every dynamical step once.

        This also rejects geometrically plausible paths made by interpolating
        coordinates or by using a different Hamiltonian/integrator/timestep.
        """
        index = self._index(interface_index)
        self._check_dynamics()
        if not isinstance(path, TISPath):
            raise TypeError("initial_path must be a TISPath")
        if path.dt != self.dynamics.dt or path.q.shape[1:] != self.dynamics.integrator.shape:
            raise ValueError("path shape/timestep does not match Dynamics")
        values = np.asarray([self.value(q) for q in path.q])
        if not np.allclose(values, path.values, rtol=1e-12, atol=1e-12):
            raise ValueError("stored path coordinates do not match order parameter values")
        if not self._member(path, index):
            raise ValueError("initial path is outside the requested interface ensemble")
        state = path.state(0)
        self.dynamics._check_fixed(state.q, state.p)
        for frame in range(1, len(path)):
            self.dynamics.step(state)
            if not (
                np.allclose(state.q, path.q[frame], rtol=2e-10, atol=2e-11)
                and np.allclose(state.p, path.p[frame], rtol=2e-10, atol=2e-11)
            ):
                raise ValueError("initial path is not a trajectory of the configured Dynamics")
        return path

    def _propagate_to_basin(self, state, budget):
        states, values = [], []
        for _ in range(budget):
            self.dynamics.step(state)
            value = self.value(state.q)
            states.append(state.copy())
            values.append(value)
            if self._basin(value):
                return states, values
        raise PathGenerationError("max_path_steps")

    def generate_path(self, shooting_state, interface_index=0):
        """Build a seed from supplied momenta; never alter the supplied State.

        The shooting configuration must be outside both basins.  Backward
        propagation must reach A.  Failure to find a valid seed is explicit;
        an interpolated/forced path must not be substituted for an unbiased one.
        """
        return self._generate_path(shooting_state, interface_index, "interior")

    def _generate_path(self, shooting_state, interface_index, role):
        index = self._index(interface_index)
        self._check_dynamics()
        if not isinstance(shooting_state, State):
            raise TypeError("shooting_state must be a State")
        center = State(shooting_state.q, shooting_state.p)
        self.dynamics.integrator._check(center.q, center.p)
        self.dynamics._check_fixed(center.q, center.p)
        value = self.value(center.q)
        if role == "interior" and self._basin(value):
            raise ValueError("shooting configuration must lie outside both basins")
        back, bv, forward, fv = [], [], [], []
        if role != "start":
            backward = State(center.q, -center.p)
            reserved_forward_step = int(role == "interior")
            back, bv = self._propagate_to_basin(backward, self.max_path_steps - reserved_forward_step)
            if bv[-1] > self.basin_a:
                raise PathGenerationError("backward_not_A")
        if role != "end":
            forward, fv = self._propagate_to_basin(center.copy(), self.max_path_steps - len(back))
        q = [state.q for state in reversed(back)] + [center.q] + [state.q for state in forward]
        p = [-state.p for state in reversed(back)] + [center.p] + [state.p for state in forward]
        path = TISPath(q, p, bv[::-1] + [value] + fv, self.dynamics.dt)
        if not self._member(path, index):
            raise PathGenerationError("interface_not_crossed")
        return path

    def _momenta(self):
        # Same real-mass H_P convention as Dynamics.sample_momenta, but this
        # chain owns its RNG independently of equilibration in Dynamics.
        dyn = self.dynamics
        p = self.rng.normal(size=dyn.integrator.shape) * np.sqrt(
            dyn.masses[None, :, None] * dyn.nbeads / dyn.beta
        )
        p[:, dyn.fixed_atoms, :] = 0
        return p

    def _shooting_log_ratio(self, old, new, old_shooting, new_shooting):
        dyn = self.dynamics
        beta_p = dyn.beta / dyn.nbeads
        delta_k = float(np.sum(
            (new_shooting.p**2 - old_shooting.p**2)
            / (2 * dyn.masses[None, :, None])
        ))
        # Full-refresh proposal densities cancel the shooting kinetic change.
        # Retaining the endpoint-energy difference also corrects the residual
        # drift of a reversible finite-step map for the canonical-start path
        # target. In the energy-conserving limit only the length ratio remains.
        delta_h = dyn.energy(new.state(0)) - dyn.energy(old.state(0))
        return math.log(len(old) / len(new)) - beta_p * (delta_h - delta_k)

    def shoot(self, path, interface_index=0):
        """One Monte Carlo proposal, returning (retained_path, move).

        ``sample`` validates an external seed once.  Direct callers should use
        ``validate_path`` before passing a new externally constructed path here.
        """
        index = self._index(interface_index)
        if not self._member(path, index):
            raise ValueError("current path is outside the requested interface ensemble")
        point = int(self.rng.integers(0, len(path)))
        old_shooting = path.state(point)
        new_shooting = State(old_shooting.q, self._momenta())
        try:
            role = "start" if point == 0 else "end" if point == len(path) - 1 else "interior"
            trial = self._generate_path(new_shooting, index, role)
        except PathGenerationError as exc:
            return path, TISMove(False, exc.reason, point, None, None)
        log_ratio = self._shooting_log_ratio(path, trial, old_shooting, new_shooting)
        if not math.isfinite(log_ratio):
            raise FloatingPointError("nonfinite TIS acceptance ratio")
        accepted = math.log(max(float(self.rng.random()), np.finfo(float).tiny)) < min(0.0, log_ratio)
        return (
            trial if accepted else path,
            TISMove(accepted, "accepted" if accepted else "metropolis", point, len(trial), log_ratio),
        )

    def sample(
        self, initial_path, interface_index=0, *, n_samples, burn_in=0,
        on_sample: Callable[[int, TISPath, TISMove], None] | None = None,
    ):
        """Run a path chain, recording one observation after every proposal.

        ``on_sample(index, path, move)`` receives each retained production path,
        including repeats.  It can stream snapshots or compute mechanism
        observables.  The default memory cost is one current/trial path plus
        O(n_samples) scalars, not O(n_samples * path_length * atoms * beads).
        """
        index = self._index(interface_index)
        n_samples = _integer(n_samples, "n_samples", 1)
        burn_in = _integer(burn_in, "burn_in")
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable")
        path = self.validate_path(initial_path, index)
        crossing, accepted, lengths, reasons = [], [], [], []
        length_rejections = 0
        product = index == len(self.interfaces) - 1
        target = self.basin_b if product else float(self.interfaces[index + 1])
        for attempt in range(burn_in + n_samples):
            path, move = self.shoot(path, index)
            length_rejections += move.reason == "max_path_steps"
            if attempt < burn_in:
                continue
            success = path.values[-1] >= self.basin_b if product else np.any(path.values > target)
            crossing.append(bool(success))
            accepted.append(move.accepted)
            lengths.append(len(path))
            reasons.append(move.reason)
            if on_sample is not None:
                on_sample(attempt - burn_in, path, move)
        return TISResult(
            index, float(self.interfaces[index]), target, product, self.basin_a, self.basin_b,
            np.asarray(crossing, bool), np.asarray(accepted, bool), np.asarray(lengths, int),
            tuple(reasons), path, copy.deepcopy(self.rng.bit_generator.state),
            burn_in, self.max_path_steps, int(length_rejections),
            self.dynamics.beta, self.dynamics.nbeads,
        )

    def sample_interfaces(self, initial_paths, *, n_samples, burn_in=0):
        """Sample all interfaces in order using one supplied seed per ensemble."""
        paths = list(initial_paths)
        if len(paths) != len(self.interfaces):
            raise ValueError("provide one initial path per interface")
        return [
            self.sample(path, i, n_samples=n_samples, burn_in=burn_in)
            for i, path in enumerate(paths)
        ]


def estimate_effective_flux(values, *, basin_a, basin_b, first_interface, dt):
    """Count first outward crossings per A-history time from an unbiased trace.

    Repeated crossings without returning to A count only once.  The denominator
    is time whose *last visited stable basin* was A, including excursions, not
    just time geometrically inside A.  Unknown history before the first A/B
    visit is discarded.  Use every dynamics step and the same dynamics/basins
    as the path ensembles.  Input time is in atomic units.
    """
    values = np.asarray(values)
    if values.dtype.kind not in "fiu" or values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite scalar time series with at least two frames")
    a, b, interface, dt = (
        _finite(value, name) for value, name in (
            (basin_a, "basin_a"), (basin_b, "basin_b"),
            (first_interface, "first_interface"), (dt, "dt"),
        )
    )
    if not a <= interface < b or dt <= 0:
        raise ValueError("require basin_a <= first_interface < basin_b and dt>0")
    history = -1 if values[0] <= a else 1 if values[0] >= b else 0
    armed = history == -1
    crossings = a_intervals = unknown_intervals = 0
    for previous, current in zip(values[:-1], values[1:]):
        a_intervals += history == -1
        unknown_intervals += history == 0
        if armed and previous <= interface < current:
            crossings += 1
            armed = False
        if current <= a:
            history, armed = -1, True
        elif current >= b:
            history, armed = 1, False
    if not a_intervals:
        raise ValueError("trace has no known A-history observation time")
    time = a_intervals * dt
    if not math.isfinite(time) or not math.isfinite(crossings / time / ATOMIC_TIME_S):
        raise ValueError("flux time or value overflow")
    return {
        "first_interface": interface,
        "basin_a": a,
        "basin_b": b,
        "effective_crossings": int(crossings),
        "a_history_time_au": float(time),
        "discarded_unknown_history_time_au": float(unknown_intervals * dt),
        "flux_au": float(crossings / time),
        "flux_s_inv": float(crossings / time / ATOMIC_TIME_S),
        "dt_au": dt,
        "converged": False,
    }


def combine_tis_rate(results, flux):
    """Combine adjacent conditional probabilities with a matching flux estimate.

    This returns a candidate first-order rate, not a bimolecular rate coefficient.
    Correlated path chains and a flux trajectory do not establish uncertainty or
    convergence by multiplication; those require independent sampling studies.
    """
    summaries = [result.summary() if isinstance(result, TISResult) else dict(result) for result in results]
    if not summaries:
        raise ValueError("at least one interface result is required")
    # Imported summaries must obey the same domain as sampler-generated data.
    # Adjacency alone would also accept a decreasing or nonphysical interface
    # chain, and differently truncated ensembles do not share one path measure.
    for result in summaries:
        try:
            for key in ("basin_a", "basin_b", "interface", "target", "dt_au", "beta_au",
                        "crossing_probability"):
                result[key] = _finite(result[key], key)
            for key, minimum in (("interface_index", 0), ("nbeads", 1), ("max_path_steps", 1)):
                result[key] = _integer(result[key], key, minimum)
            result["length_rejections"] = _integer(result.get("length_rejections", 0), "length_rejections")
            if not isinstance(result["target_is_product"], (bool, np.bool_)):
                raise ValueError("target_is_product must be boolean")
        except (KeyError, TypeError) as exc:
            raise ValueError("TIS summary has missing or invalid required metadata") from exc
        if not result["basin_a"] <= result["interface"] < result["target"] <= result["basin_b"]:
            raise ValueError("TIS summaries require finite ordered basin_a <= interface < target <= basin_b")
        if result["dt_au"] <= 0 or result["beta_au"] <= 0:
            raise ValueError("TIS summary dt_au and beta_au must be positive")
        if not 0 <= result["crossing_probability"] <= 1:
            raise ValueError("crossing probabilities must lie in [0,1]")
    a, b = summaries[0]["basin_a"], summaries[0]["basin_b"]
    parameters = {key: summaries[0][key] for key in ("dt_au", "beta_au", "nbeads")}
    path_bound = summaries[0]["max_path_steps"]
    if any(result["max_path_steps"] != path_bound for result in summaries):
        raise ValueError("all TIS ensembles must use the same max_path_steps path support")
    if (
        flux.get("basin_a") != a or flux.get("basin_b") != b
        or flux.get("first_interface") != summaries[0]["interface"]
        or flux.get("dt_au") != parameters["dt_au"]
    ):
        raise ValueError("flux and path ensembles must use the same basins/first interface")
    probabilities = []
    for index, result in enumerate(summaries):
        expected = b if index == len(summaries) - 1 else summaries[index + 1]["interface"]
        if (
            result["interface_index"] != index or result["basin_a"] != a or result["basin_b"] != b
            or result["target"] != expected
            or bool(result["target_is_product"]) != (index == len(summaries) - 1)
            or any(result[key] != value for key, value in parameters.items())
        ):
            raise ValueError("results must cover every adjacent interface through product B in order")
        probabilities.append(result["crossing_probability"])
    flux_au = _finite(flux["flux_au"], "flux_au")
    if flux_au < 0:
        raise ValueError("flux must be nonnegative")
    overall = float(np.prod(probabilities))
    candidate = flux_au * overall
    if not math.isfinite(candidate / ATOMIC_TIME_S):
        raise ValueError("combined rate overflow")
    return {
        "method": "transition_interface_sampling",
        "conditional_probabilities": probabilities,
        "overall_crossing_probability": overall,
        "flux_au": flux_au,
        "rate_estimate_au": candidate,
        "rate_estimate_s_inv": candidate / ATOMIC_TIME_S,
        "rate_s_inv": None,
        "converged": False,
        "max_path_steps": path_bound,
        "max_length_rejections": sum(result.get("length_rejections", 0) for result in summaries),
        "qualification": (
            "Candidate first-order rate only; establish basin, timestep, bead, path-length, "
            "flux and independent path-chain convergence before physical interpretation."
        ),
    }
