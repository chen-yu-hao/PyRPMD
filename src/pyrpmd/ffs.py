"""Direct forward-flux sampling of a user-specified Markov process.

No time reversal or momentum redraw is performed.  The transition callback
``step(state, rng)`` must contain the complete Markov state in ``State.q/p``
and use only the supplied RNG.  Time and step counters are owned by the sampler.
See Allen, Valeriani and ten Wolde, JPCM 21, 463102 (2009),
https://doi.org/10.1088/0953-8984/21/46/463102.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass

import numpy as np

from .analysis.fes import statistical_inefficiency
from .checkpoint import CheckpointError, atomic_savez, load_checkpoint
from .constants import ATOMIC_TIME_S
from .dynamics import State
from .io import fingerprint_payload, jsonable
from .validation import finite_real, integer_count


_FORMAT = "PyRPMD.ffs.v1"


def _owned(value, dtype):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _json(value):
    return json.dumps(jsonable(value, callables=False), sort_keys=True, allow_nan=False)


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty scientific identity string")
    return value


def _correlation(values):
    values = np.asarray(values)
    if len(values) < 2 or not np.any(values != values[0]):
        return None
    values = values.astype(float)
    return float(statistical_inefficiency(values / np.max(np.abs(values))))


def _flux_statistics(events, a_history, dt, block_size):
    """Delete-one-block ratio jackknife, conditional on the observed trace."""
    n = len(events)
    g = max((_correlation(x) or 1.0 for x in (events, a_history)), default=1.0)
    if block_size is None:
        block_size = max(1, int(math.ceil(g)))
    block_size = integer_count(block_size, "block_size", 1)
    n_blocks = n // block_size
    standard_error = None
    if n_blocks >= 2:
        usable = n_blocks * block_size
        numerators = np.asarray(events[:usable], float).reshape(n_blocks, block_size).sum(axis=1)
        denominators = np.asarray(a_history[:usable], float).reshape(n_blocks, block_size).sum(axis=1) * dt
        remaining = denominators.sum() - denominators
        if np.all(remaining > 0) and numerators.sum() > 0:
            estimates = (numerators.sum() - numerators) / remaining
            if np.any(estimates != estimates[0]):
                scale = float(np.max(np.abs(estimates)))
                standard_error = float(np.sqrt((n_blocks - 1) * np.var(estimates / scale)) * scale)
                if not math.isfinite(standard_error):
                    raise ValueError("FFS flux uncertainty overflow")
    return {
        "statistical_inefficiency_diagnostic": g,
        "block_size": block_size,
        "blocks": n_blocks,
        "block_discarded_intervals": n % block_size,
        "standard_error_diagnostic": standard_error,
        "uncertainty_method": "delete_one_equal_time_block_ratio_jackknife",
    }


class FFSUnresolvedError(RuntimeError):
    """A trial is censored, so no success/failure observation was committed."""

    def __init__(self, stage, parent_index, final_state, steps):
        super().__init__(
            f"FFS stage {stage} trial unresolved after max_trial_steps={steps}; "
            "the trial and RNG were rolled back, not counted as a failure"
        )
        self.stage = stage
        self.parent_index = parent_index
        self.final_state = final_state.copy()
        self.steps = steps


@dataclass(frozen=True)
class FFSStage:
    """Trial records; parent indices refer to the preceding crossing bank."""

    interface: float
    target: float
    target_is_product: bool
    outcomes: np.ndarray
    parent_indices: np.ndarray
    root_indices: np.ndarray
    steps: np.ndarray

    def summary(self):
        n = len(self.outcomes)
        successes = int(np.sum(self.outcomes))
        probability = successes / n if n else None
        varied = n > 1 and 0 < successes < n
        return {
            "interface": self.interface,
            "target": self.target,
            "target_is_product": self.target_is_product,
            "trials": n,
            "successes": successes,
            "crossing_probability": probability,
            "conditional_bank_standard_error_diagnostic": (
                math.sqrt(probability * (1 - probability) / (n - 1)) if varied else None
            ),
            "distinct_parents": int(len(np.unique(self.parent_indices))),
            "distinct_flux_ancestors": int(len(np.unique(self.root_indices))),
            "statistical_inefficiency_diagnostic": _correlation(self.outcomes),
        }


@dataclass(frozen=True)
class FFSResult:
    """Owned snapshot of an FFS run, including all crossing-point banks."""

    configuration: dict
    configuration_fingerprint: str
    run_identity: str
    status: str
    flux_values: np.ndarray
    flux_events: np.ndarray
    flux_a_history: np.ndarray
    flux_unknown_history: np.ndarray
    stages: tuple[FFSStage, ...]
    banks: tuple[tuple[State, ...], ...]
    bank_roots: tuple[np.ndarray, ...]
    flux_final_state: State
    rng_state: dict

    @property
    def complete(self):
        # Extinct branch populations are legitimate zero estimates. Omitting
        # them when averaging independent runs would bias the rate upward.
        return self.status in ("complete", "no_successes", "no_flux_crossings") and bool(
            np.any(self.flux_a_history)
        )

    def summary(self, *, block_size=None):
        config = self.configuration
        dt = config["dt_au"]
        crossings = int(self.flux_events.sum())
        time = float(self.flux_a_history.sum() * dt)
        flux = crossings / time if time > 0 else None
        if not math.isfinite(time) or (flux is not None and not math.isfinite(flux / ATOMIC_TIME_S)):
            raise ValueError("FFS flux time or value overflow")
        stage_summaries = [stage.summary() for stage in self.stages]
        probabilities = [stage["crossing_probability"] for stage in stage_summaries]
        probability = None
        if self.status == "complete":
            probability = float(np.prod(probabilities))
        elif self.status == "no_successes":
            probability = 0.0
        rate = flux * probability if self.complete and probability is not None else None
        if self.complete and self.status == "no_flux_crossings":
            rate = 0.0
        if rate is not None and not math.isfinite(rate / ATOMIC_TIME_S):
            raise ValueError("FFS combined rate overflow")
        return {
            "method": "direct_forward_flux_sampling",
            "status": self.status,
            "complete": self.complete,
            "configuration_fingerprint": self.configuration_fingerprint,
            "run_identity": self.run_identity,
            "basin_a": config["basin_a"],
            "basin_b": config["basin_b"],
            "first_interface": config["interfaces"][0],
            "dt_au": dt,
            "flux_intervals": len(self.flux_events),
            "effective_crossings": crossings,
            "a_history_time_au": time,
            "discarded_unknown_history_time_au": float(self.flux_unknown_history.sum() * dt),
            "flux_au": flux,
            "flux_s_inv": flux / ATOMIC_TIME_S if flux is not None else None,
            "flux_diagnostics": _flux_statistics(
                self.flux_events, self.flux_a_history, dt, block_size,
            ),
            "interfaces": stage_summaries,
            "conditional_probabilities": probabilities,
            "overall_crossing_probability": probability,
            "rate_estimate_au": rate,
            "rate_estimate_s_inv": rate / ATOMIC_TIME_S if rate is not None else None,
            "rate_standard_error_au": None,
            "rate_s_inv": None,
            "converged": False,
            "physical_converged": False,
            "qualification": (
                "The flux trace must be stationary. Trial errors are conditional on the "
                "finite crossing banks; use independent complete FFS replicates for "
                "rate uncertainty including shared ancestors and flux correlations. "
                "No observed successes does not establish a zero physical rate."
            ),
        }


class ForwardFluxSampler:
    """Direct FFS with resumable flux collection and interface trials.

    ``step(state, rng)`` mutates its owned ``State`` or returns a new ``State``.
    It must implement one step of duration ``dt``. The sampler advances clocks;
    the callback may either leave them unchanged or advance them by exactly
    one step. All stochastic memory must be in q/p and the supplied generator.
    Auxiliary variables may be encoded in q/p, but opaque solver memory, global
    RNGs and stateful callbacks are unsupported. The process must be stationary
    and time-homogeneous: absolute step/time counters must not change its law.
    No built-in thermostat or
    alternate dynamics is selected automatically.

    Required ``dynamics_id`` and ``coordinate_id`` must identify all scientific
    parameters of those opaque callables, including implementation versions,
    potential, masses, temperature, bead conventions, and stochastic scheme.
    They are part of checkpoint compatibility and do not replace that caller
    responsibility. ``interfaces`` excludes the final product boundary.

    A is s<=basin_a, B is s>=basin_b, and crossing an ordinary interface means
    s>lambda. Every first outward crossing since a visit to A is stored. Full
    states are sampled uniformly with replacement from each crossing bank;
    their momenta are never redrawn. Zero-step continuation handles overshoots.
    ``advance`` stops only between flux steps or complete trials.
    """

    def __init__(
        self, step, coordinate, *, dt, basin_a, basin_b, interfaces,
        dynamics_id, coordinate_id, flux_steps, trials_per_interface,
        max_trial_steps=10000, seed=0,
    ):
        if not callable(step):
            raise TypeError("step must be callable as step(state, rng)")
        self._step_callback = step
        if callable(getattr(coordinate, "value", None)):
            if getattr(coordinate, "dimension", 1) != 1:
                raise ValueError("FFS requires a scalar order parameter")
            if any(p is not None for p in getattr(coordinate, "periods", (None,))):
                raise ValueError("FFS requires an ordered nonperiodic coordinate")
            self._coordinate = coordinate.value
        elif callable(coordinate):
            self._coordinate = coordinate
        else:
            raise TypeError("coordinate must be a scalar CV or callable on q")
        self.dt = finite_real(dt, "dt", positive=True)
        self.basin_a = finite_real(basin_a, "basin_a")
        self.basin_b = finite_real(basin_b, "basin_b")
        raw = np.asarray(interfaces)
        if raw.dtype.kind not in "fiu" or raw.ndim != 1 or not len(raw):
            raise ValueError("interfaces require finite ordered real numbers")
        self.interfaces = _owned(raw, float)
        if (
            not np.isfinite(self.interfaces).all() or self.basin_a >= self.basin_b
            or self.interfaces[0] < self.basin_a or self.interfaces[-1] >= self.basin_b
            or np.any(np.diff(self.interfaces) <= 0)
        ):
            raise ValueError("require basin_a <= lambda[0] < ... < basin_b")
        self.flux_steps = integer_count(flux_steps, "flux_steps", 1)
        self.max_trial_steps = integer_count(max_trial_steps, "max_trial_steps", 1)
        if not all(math.isfinite(count * self.dt) for count in (self.flux_steps, self.max_trial_steps)):
            raise ValueError("FFS simulation horizon overflow")
        if np.isscalar(trials_per_interface):
            trials = [trials_per_interface] * len(self.interfaces)
        else:
            trials = list(trials_per_interface)
        if len(trials) != len(self.interfaces):
            raise ValueError("one positive trial count is required per interface")
        self.trials_per_interface = tuple(
            integer_count(x, "trials_per_interface", 1) for x in trials
        )
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.configuration = {
            "method": "direct_forward_flux_sampling.v1",
            "dt_au": self.dt,
            "basin_a": self.basin_a,
            "basin_b": self.basin_b,
            "interfaces": self.interfaces.tolist(),
            "dynamics_id": _label(dynamics_id, "dynamics_id"),
            "coordinate_id": _label(coordinate_id, "coordinate_id"),
            "flux_steps": self.flux_steps,
            "trials_per_interface": list(self.trials_per_interface),
            "max_trial_steps": self.max_trial_steps,
            "rng_bit_generator": type(self.rng.bit_generator).__name__,
            "flux_normalization": "last_visited_basin_A_time",
            "crossing_convention": "s>lambda; product s>=B",
            "parent_selection": "uniform_with_replacement",
        }
        self.configuration_fingerprint = fingerprint_payload(self.configuration)
        self._initial_configuration = copy.deepcopy(self.configuration)
        self._signature = self._current_signature()
        self._state = None

    def _current_signature(self):
        return (
            self.dt, self.basin_a, self.basin_b, self.interfaces.tobytes(),
            self.flux_steps, self.max_trial_steps, self.trials_per_interface,
            id(self._step_callback), id(self._coordinate), type(self.rng.bit_generator).__name__,
        )

    def _check_configuration(self):
        if self._current_signature() != self._signature or self.configuration != self._initial_configuration:
            raise ValueError("FFS configuration changed; create a new sampler")

    def value(self, q):
        view = np.asarray(q).view()
        view.flags.writeable = False
        value = np.asarray(self._coordinate(view))
        if value.dtype.kind not in "fiu" or value.size != 1 or not np.isfinite(value).all():
            raise ValueError("FFS coordinate must return one finite real value")
        return float(value.reshape(-1)[0])

    def _propagate(self, state):
        old_step, old_time = state.step, state.time
        result = self._step_callback(state, self.rng)
        if result is not None:
            if not isinstance(result, State):
                raise TypeError("FFS step must return State or None")
            state = result
        state = state.copy()  # Revalidate finite arrays, clocks, and ownership.
        if state.q.shape != self._shape:
            raise ValueError("FFS step changed the Markov state shape")
        expected_time = old_time + self.dt
        if not math.isfinite(expected_time):
            raise ValueError("FFS simulation clock overflow")
        if expected_time == old_time:
            raise ValueError("FFS timestep does not advance the floating-point simulation clock")
        if state.step not in (old_step, old_step + 1) or not (
            state.time == old_time or math.isclose(state.time, expected_time, rel_tol=1e-13, abs_tol=1e-14)
        ):
            raise ValueError("FFS step must preserve clocks or advance exactly one timestep")
        state.step, state.time = old_step + 1, expected_time
        return state

    def initialize(self, state):
        """Start flux collection from a caller-prepared stationary state."""
        self._check_configuration()
        if self._state is not None:
            raise ValueError("FFS sampler is already initialized")
        if not isinstance(state, State):
            raise TypeError("initial_state must be a State")
        owned = state.copy()
        value = self.value(owned.q)
        self._shape = owned.q.shape
        self._state = owned
        self._initial_state = owned.copy()
        self._history = -1 if value <= self.basin_a else 1 if value >= self.basin_b else 0
        self._armed = self._history == -1
        self._values = [value]
        self._events, self._a_history, self._unknown_history = [], [], []
        self._banks = [[] for _ in range(len(self.interfaces) + 1)]
        self._bank_roots = [[] for _ in self._banks]
        self._outcomes = [[] for _ in self.interfaces]
        self._parents = [[] for _ in self.interfaces]
        self._roots = [[] for _ in self.interfaces]
        self._steps = [[] for _ in self.interfaces]
        self._stage = 0
        self._status = "collecting_flux"
        self._initial_rng_state = copy.deepcopy(self.rng.bit_generator.state)
        self.run_identity = fingerprint_payload({
            "configuration": self.configuration,
            "initial_q": owned.q, "initial_p": owned.p,
            "initial_step": owned.step, "initial_time": owned.time,
            "initial_rng_state": self._initial_rng_state,
        })
        return self

    def _flux_step(self):
        rng_before = copy.deepcopy(self.rng.bit_generator.state)
        try:
            state = self._propagate(self._state.copy())
            value = self.value(state.q)
        except Exception:
            self.rng.bit_generator.state = rng_before
            raise
        event = self._armed and self._values[-1] <= self.interfaces[0] < value
        self._a_history.append(self._history == -1)
        self._unknown_history.append(self._history == 0)
        self._events.append(bool(event))
        self._values.append(value)
        self._state = state
        if event:
            self._bank_roots[0].append(len(self._banks[0]))
            self._banks[0].append(state.copy())
            self._armed = False
        if value <= self.basin_a:
            self._history, self._armed = -1, True
        elif value >= self.basin_b:
            self._history, self._armed = 1, False
        if len(self._events) == self.flux_steps:
            self._status = "sampling_interfaces" if self._banks[0] else "no_flux_crossings"

    def _target_reached(self, value, stage):
        if stage == len(self.interfaces) - 1:
            return value >= self.basin_b
        return value > self.interfaces[stage + 1]

    def _trial(self):
        i = self._stage
        rng_before = copy.deepcopy(self.rng.bit_generator.state)
        try:
            parent = int(self.rng.integers(len(self._banks[i])))
            state = self._banks[i][parent].copy()
            root = self._bank_roots[i][parent]
            value = self.value(state.q)
            steps = 0
            while not self._target_reached(value, i) and value > self.basin_a:
                if steps == self.max_trial_steps:
                    raise FFSUnresolvedError(i, parent, state, steps)
                state = self._propagate(state)
                value = self.value(state.q)
                steps += 1
            success = self._target_reached(value, i)
        except Exception:
            self.rng.bit_generator.state = rng_before
            raise
        self._outcomes[i].append(bool(success))
        self._parents[i].append(parent)
        self._roots[i].append(root)
        self._steps[i].append(steps)
        if success:
            self._banks[i + 1].append(state.copy())
            self._bank_roots[i + 1].append(root)
        if len(self._outcomes[i]) == self.trials_per_interface[i]:
            if not self._banks[i + 1]:
                self._status = "no_successes"
            elif i == len(self.interfaces) - 1:
                self._status = "complete"
            else:
                self._stage += 1

    def advance(self, n_operations=None):
        """Advance a bounded number of flux steps/trials, or finish the run.

        Numerical exceptions and censored trials restore their pre-operation
        RNG and leave all committed records unchanged. Censoring never counts
        as a failed trial. To change its horizon, start a separately identified
        calculation with a larger ``max_trial_steps``.
        """
        self._check_configuration()
        if self._state is None:
            raise ValueError("initialize FFS before advancing")
        if n_operations is not None:
            n_operations = integer_count(n_operations, "n_operations", 0)
        done = 0
        while self._status in ("collecting_flux", "sampling_interfaces") and (
            n_operations is None or done < n_operations
        ):
            if self._status == "collecting_flux":
                self._flux_step()
            else:
                self._trial()
            done += 1
        return self.result()

    def run(self, initial_state, *, n_operations=None):
        """Initialize and advance; use ``advance`` after a checkpoint restart."""
        self.initialize(initial_state)
        return self.advance(n_operations)

    def result(self):
        """Take an owned snapshot without consuming randomness."""
        self._check_configuration()
        if self._state is None:
            raise ValueError("FFS has not been initialized")
        stages = tuple(FFSStage(
            float(interface),
            self.basin_b if i == len(self.interfaces) - 1 else float(self.interfaces[i + 1]),
            i == len(self.interfaces) - 1,
            _owned(self._outcomes[i], bool), _owned(self._parents[i], np.int64),
            _owned(self._roots[i], np.int64), _owned(self._steps[i], np.int64),
        ) for i, interface in enumerate(self.interfaces))
        return FFSResult(
            copy.deepcopy(self.configuration), self.configuration_fingerprint, self.run_identity,
            self._status, _owned(self._values, float), _owned(self._events, bool),
            _owned(self._a_history, bool), _owned(self._unknown_history, bool), stages,
            tuple(tuple(state.copy() for state in bank) for bank in self._banks),
            tuple(_owned(roots, np.int64) for roots in self._bank_roots),
            self._state.copy(), copy.deepcopy(self.rng.bit_generator.state),
        )

    def save_checkpoint(self, path):
        """Atomically save the exact current path banks, genealogy and RNG."""
        result = self.result()
        arrays = {
            "flux_values": result.flux_values, "flux_events": result.flux_events,
            "flux_a_history": result.flux_a_history,
            "flux_unknown_history": result.flux_unknown_history,
            "state_q": self._state.q, "state_p": self._state.p,
            "initial_q": self._initial_state.q, "initial_p": self._initial_state.p,
        }
        for i, bank in enumerate(self._banks):
            arrays[f"bank_{i}_q"] = np.asarray([s.q for s in bank]).reshape((-1, *self._shape))
            arrays[f"bank_{i}_p"] = np.asarray([s.p for s in bank]).reshape((-1, *self._shape))
            arrays[f"bank_{i}_step"] = np.asarray([s.step for s in bank], dtype=np.int64)
            arrays[f"bank_{i}_time"] = np.asarray([s.time for s in bank], dtype=float)
            arrays[f"bank_{i}_roots"] = np.asarray(self._bank_roots[i], dtype=np.int64)
        for i, stage in enumerate(result.stages):
            for name in ("outcomes", "parent_indices", "root_indices", "steps"):
                arrays[f"stage_{i}_{name}"] = getattr(stage, name)
        metadata = {
            "format": _FORMAT, "configuration": self.configuration,
            "configuration_fingerprint": self.configuration_fingerprint,
            "run_identity": self.run_identity, "status": self._status,
            "history": self._history, "armed": self._armed, "stage": self._stage,
            "state_step": self._state.step, "state_time": self._state.time,
            "initial_step": self._initial_state.step, "initial_time": self._initial_state.time,
            "initial_rng_state": self._initial_rng_state,
            "rng_state": result.rng_state,
        }
        atomic_savez(path, kind=_FORMAT, metadata=np.asarray(_json(metadata)), **arrays)

    def load_checkpoint(self, path):
        """Validate a checkpoint completely before replacing sampler state."""
        self._check_configuration()
        with load_checkpoint(path, expected_kind=_FORMAT, require_integrity=True) as data:
            try:
                metadata = json.loads(str(data["metadata"].item()))
                if metadata["configuration"] != self.configuration or (
                    metadata["configuration_fingerprint"] != self.configuration_fingerprint
                ):
                    raise CheckpointError("FFS scientific identity does not match checkpoint")
                candidate = copy.copy(self)
                candidate.rng = np.random.Generator(type(self.rng.bit_generator)())
                candidate.rng.bit_generator.state = metadata["rng_state"]
                candidate._state = State(
                    data["state_q"], data["state_p"], metadata["state_step"], metadata["state_time"],
                )
                candidate._initial_state = State(
                    data["initial_q"], data["initial_p"], metadata["initial_step"], metadata["initial_time"],
                )
                candidate._shape = candidate._state.q.shape
                candidate._history = metadata["history"]
                candidate._armed = metadata["armed"]
                candidate._stage = integer_count(metadata["stage"], "stage", 0, len(self.interfaces) - 1)
                candidate._status = metadata["status"]
                candidate.run_identity = _label(metadata["run_identity"], "run_identity")
                candidate._initial_rng_state = metadata["initial_rng_state"]
                check_rng = np.random.Generator(type(self.rng.bit_generator)())
                check_rng.bit_generator.state = candidate._initial_rng_state
                expected_names = {"metadata", "state_q", "state_p", "initial_q", "initial_p", "flux_values", "flux_events",
                                  "flux_a_history", "flux_unknown_history"}
                for key, attr in (("flux_values", "_values"), ("flux_events", "_events"),
                                  ("flux_a_history", "_a_history"),
                                  ("flux_unknown_history", "_unknown_history")):
                    array = data[key]
                    expected_kind = "f" if key == "flux_values" else "b"
                    if array.ndim != 1 or array.dtype.kind != expected_kind:
                        raise CheckpointError("invalid FFS flux trace arrays")
                    setattr(candidate, attr, array.tolist())
                candidate._banks, candidate._bank_roots = [], []
                for i in range(len(self.interfaces) + 1):
                    names = [f"bank_{i}_{key}" for key in ("q", "p", "step", "time", "roots")]
                    expected_names.update(names)
                    q, p, steps, times, roots = (data[key] for key in names)
                    count = len(q)
                    if (
                        q.shape != (count, *candidate._shape) or p.shape != q.shape
                        or any(x.shape != (count,) for x in (steps, times, roots))
                        or any(x.dtype.kind not in "iu" for x in (steps, roots))
                        or np.any(steps < 0) or np.any(roots < 0)
                    ):
                        raise CheckpointError("invalid FFS crossing bank shapes or indices")
                    candidate._banks.append([State(a, b, int(s), float(t))
                                             for a, b, s, t in zip(q, p, steps, times)])
                    candidate._bank_roots.append(roots.tolist())
                for attr in ("_outcomes", "_parents", "_roots", "_steps"):
                    setattr(candidate, attr, [])
                for i in range(len(self.interfaces)):
                    for key, attr in (("outcomes", "_outcomes"), ("parent_indices", "_parents"),
                                      ("root_indices", "_roots"), ("steps", "_steps")):
                        name = f"stage_{i}_{key}"
                        expected_names.add(name)
                        array = data[name]
                        kind = "b" if key == "outcomes" else "iu"
                        if array.ndim != 1 or array.dtype.kind not in kind:
                            raise CheckpointError("invalid FFS trial record arrays")
                        getattr(candidate, attr).append(array.tolist())
                if set(data.files) != expected_names:
                    raise CheckpointError("unexpected FFS checkpoint payload inventory")
                candidate._validate_records()
            except CheckpointError:
                raise
            except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
                raise CheckpointError(f"invalid FFS checkpoint: {exc}") from exc
        self.__dict__.update(candidate.__dict__)
        return self

    def _validate_records(self):
        n = len(self._events)
        initial = self._initial_state
        expected_identity = fingerprint_payload({
            "configuration": self.configuration,
            "initial_q": initial.q, "initial_p": initial.p,
            "initial_step": initial.step, "initial_time": initial.time,
            "initial_rng_state": self._initial_rng_state,
        })
        if (
            n > self.flux_steps or len(self._values) != n + 1
            or len(self._a_history) != n or len(self._unknown_history) != n
            or not np.isfinite(self._values).all()
            or type(self._history) is not int or self._history not in (-1, 0, 1)
            or type(self._armed) is not bool or self.run_identity != expected_identity
            or initial.q.shape != self._shape or self._state.step != initial.step + n
            or not math.isclose(self.value(initial.q), self._values[0], rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(self.value(self._state.q), self._values[-1], rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise CheckpointError("inconsistent FFS flux history")
        history = -1 if self._values[0] <= self.basin_a else 1 if self._values[0] >= self.basin_b else 0
        armed = history == -1
        event_frames = []
        event_times = []
        clock = initial.time
        for k, (before, after) in enumerate(zip(self._values[:-1], self._values[1:])):
            clock += self.dt
            event = armed and before <= self.interfaces[0] < after
            if (self._events[k] != event or self._a_history[k] != (history == -1)
                    or self._unknown_history[k] != (history == 0)):
                raise CheckpointError("FFS events disagree with the stored flux trace")
            if event:
                event_frames.append(k + 1)
                event_times.append(clock)
                armed = False
            if after <= self.basin_a:
                history, armed = -1, True
            elif after >= self.basin_b:
                history, armed = 1, False
        if (history != self._history or armed != self._armed or len(event_frames) != len(self._banks[0])
                or clock != self._state.time):
            raise CheckpointError("FFS flux banks or history counters disagree")
        if self._bank_roots[0] != list(range(len(self._banks[0]))):
            raise CheckpointError("invalid FFS initial ancestry")
        for frame, time, state in zip(event_frames, event_times, self._banks[0]):
            if state.step != initial.step + frame or state.time != time or not math.isclose(
                self.value(state.q), self._values[frame], rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise CheckpointError("FFS initial crossing bank disagrees with flux trace")
        for i in range(len(self.interfaces)):
            count = len(self._outcomes[i])
            if (
                count > self.trials_per_interface[i]
                or any(len(getattr(self, attr)[i]) != count for attr in ("_parents", "_roots", "_steps"))
                or sum(self._outcomes[i]) != len(self._banks[i + 1])
                or any(p < 0 or p >= len(self._banks[i]) for p in self._parents[i])
                or any(s < 0 or s > self.max_trial_steps for s in self._steps[i])
                or (i > self._stage and count)
                or (i < self._stage and count != self.trials_per_interface[i])
            ):
                raise CheckpointError("inconsistent FFS stage counts, ancestry or bounds")
            expected_roots = [self._bank_roots[i][p] for p in self._parents[i]]
            if self._roots[i] != expected_roots or self._bank_roots[i + 1] != [
                root for root, success in zip(expected_roots, self._outcomes[i]) if success
            ]:
                raise CheckpointError("inconsistent FFS genealogy")
            successful_trials = [j for j, success in enumerate(self._outcomes[i]) if success]
            for trial, state in zip(successful_trials, self._banks[i + 1]):
                if not self._target_reached(self.value(state.q), i):
                    raise CheckpointError("FFS successful bank state misses its target")
                parent = self._banks[i][self._parents[i][trial]]
                steps = self._steps[i][trial]
                time = parent.time + steps * self.dt
                tolerance = 8 * np.finfo(float).eps * (steps + 1) * max(1.0, abs(time))
                if state.step != parent.step + steps or abs(state.time - time) > tolerance:
                    raise CheckpointError("FFS successful bank clock disagrees with trial duration")
        if n < self.flux_steps:
            expected_status = "collecting_flux"
            if any(self._outcomes):
                raise CheckpointError("FFS trials started before flux collection finished")
        elif not self._banks[0]:
            expected_status = "no_flux_crossings"
        elif len(self._outcomes[self._stage]) < self.trials_per_interface[self._stage]:
            expected_status = "sampling_interfaces"
        elif not self._banks[self._stage + 1]:
            expected_status = "no_successes"
        elif self._stage == len(self.interfaces) - 1:
            expected_status = "complete"
        else:
            raise CheckpointError("FFS completed stage did not advance")
        if self._status != expected_status:
            raise CheckpointError("FFS checkpoint status is inconsistent")


def combine_ffs_replicates(results, *, block_size=1):
    """Block-average *independent complete runs*, never individual branches.

    Entire runs retain their shared genealogy, finite crossing-bank error and
    flux uncertainty. The caller must use independent initial RNG streams and
    independently equilibrated starts; uniqueness of run identity alone cannot
    establish physical independence. Rates are averaged as whole-run estimates.
    """
    runs = tuple(results)
    if len(runs) < 2 or any(not isinstance(r, FFSResult) or not r.complete for r in runs):
        raise ValueError("at least two complete independent FFS results are required")
    if len({r.configuration_fingerprint for r in runs}) != 1:
        raise ValueError("FFS replicates must have identical scientific configurations")
    if len({r.run_identity for r in runs}) != len(runs):
        raise ValueError("duplicate FFS run identities cannot be independent replicates")
    block_size = integer_count(block_size, "block_size", 1)
    if len(runs) % block_size or len(runs) // block_size < 2:
        raise ValueError("block_size must partition the replicates into at least two complete blocks")
    summaries = [r.summary() for r in runs]
    rates = np.asarray([s["rate_estimate_au"] for s in summaries])
    fluxes = np.asarray([s["flux_au"] for s in summaries])
    probabilities = [s["overall_crossing_probability"] for s in summaries]
    def error(data):
        blocks = data.reshape(-1, block_size).mean(axis=1)
        if not np.any(blocks != blocks[0]):
            return None
        scale = float(np.max(np.abs(blocks)))
        result = float(np.std(blocks / scale, ddof=1) / np.sqrt(len(blocks)) * scale)
        if not math.isfinite(result):
            raise ValueError("FFS replicate uncertainty overflow")
        return result
    rate_error = error(rates)
    if rate_error is not None and not math.isfinite(rate_error / ATOMIC_TIME_S):
        raise ValueError("FFS replicate rate uncertainty overflow")
    return {
        "method": "independent_direct_ffs_replicates",
        "replicates": len(runs), "block_size": block_size, "blocks": len(runs) // block_size,
        "configuration_fingerprint": runs[0].configuration_fingerprint,
        "rate_estimate_au": float(rates.mean()), "rate_standard_error_au": rate_error,
        "rate_estimate_s_inv": float(rates.mean() / ATOMIC_TIME_S),
        "rate_standard_error_s_inv": rate_error / ATOMIC_TIME_S if rate_error is not None else None,
        "flux_au": float(fluxes.mean()), "flux_standard_error_au": error(fluxes),
        "overall_crossing_probability": float(np.mean(probabilities)) if None not in probabilities else None,
        "crossing_probability_standard_error": error(np.asarray(probabilities)) if None not in probabilities else None,
        "rate_estimates_au": rates.tolist(),
        "rate_statistical_inefficiency_diagnostic": _correlation(rates),
        "uncertainty_method": "block_mean_of_independent_complete_ffs_runs",
        "rate_s_inv": None, "converged": False, "physical_converged": False,
    }
