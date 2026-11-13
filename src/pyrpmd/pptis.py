"""Partial-path TIS and replica exchange with one-interface memory.

Each local ensemble is bounded by two neighboring interfaces and crosses the
middle one. Both left- and right-origin paths are sampled. Combining these
ensembles uses the PPTIS memory-loss approximation, not a product of ordinary
TIS probabilities. See ``docs/transition-interface-sampling.md``.
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
from .transition_interface import (
    PathGenerationError, TISPath, TransitionInterfaceSampler,
    _finite, _integer,
)


def pptis_crossing_probability(p_lr, p_rl):
    """Return P(B before A | first center crossed from A), under PPTIS memory.

    ``p_lr[i]`` is P(exit right | entered left and crossed center i), and
    ``p_rl[i]`` is its right-to-left counterpart. The centers exclude A/B.
    The recurrence is Eq. (10) of Vervust et al., Biophys. J. 122 (2023),
    doi:10.1016/j.bpj.2023.02.021. It retains the direction of arrival, unlike
    a memoryless product of local probabilities.

    A singular recurrence means sampled local transitions leave a closed
    communicating class. It is explicitly rejected, not regularized.
    """
    arrays = []
    for name, value in (("p_lr", p_lr), ("p_rl", p_rl)):
        value = np.asarray(value)
        if (value.dtype.kind not in "fiu" or value.ndim != 1 or not len(value)
                or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))):
            raise ValueError(f"{name} must be a nonempty finite probability vector")
        arrays.append(value.astype(float))
    lr, rl = arrays
    if lr.shape != rl.shape:
        raise ValueError("p_lr and p_rl must have the same shape")
    plus = minus = 1.0
    for forward, backward in zip(lr, rl):
        denominator = forward + (1.0 - forward) * minus
        if denominator <= 0:
            raise ValueError("PPTIS recurrence is singular; local transition support is unresolved")
        plus, minus = forward * plus / denominator, backward * minus / denominator
    return float(plus)


def _local_probabilities(starts, ends):
    left, right = starts == -1, starts == 1
    lr = float(np.mean(ends[left] == 1)) if np.any(left) else None
    rl = float(np.mean(ends[right] == -1)) if np.any(right) else None
    return lr, rl


def _inefficiency(starts, ends, lengths):
    observables = (starts, ends, lengths, (starts == -1) & (ends == 1),
                   (starts == 1) & (ends == -1))
    return float(max(statistical_inefficiency(np.asarray(x, float)) for x in observables))


@dataclass
class PPTISLocalResult:
    """Retained local paths, including repeats after every rejected move."""

    interface_index: int
    left: float
    interface: float
    right: float
    starts: np.ndarray
    ends: np.ndarray
    accepted: np.ndarray
    path_frames: np.ndarray
    reasons: tuple[str, ...]
    final_path: TISPath
    length_rejections: int

    def summary(self):
        lr, rl = _local_probabilities(self.starts, self.ends)
        g = _inefficiency(self.starts, self.ends, self.path_frames)
        return {
            "interface_index": self.interface_index,
            "left": self.left, "interface": self.interface, "right": self.right,
            "samples": len(self.starts),
            "left_origin_samples": int(np.sum(self.starts == -1)),
            "right_origin_samples": int(np.sum(self.starts == 1)),
            "p_LR": lr, "p_LL": None if lr is None else 1 - lr,
            "p_RL": rl, "p_RR": None if rl is None else 1 - rl,
            "acceptance_fraction": float(np.mean(self.accepted)),
            "mean_path_frames": float(np.mean(self.path_frames)),
            "statistical_inefficiency": g,
            "effective_samples_diagnostic": len(self.starts) / g,
            "length_rejections": self.length_rejections,
            "length_rejections_scope": "shooting_moves_only",
            "converged": False,
        }


@dataclass
class PPTISResult:
    interfaces: tuple[PPTISLocalResult, ...]
    swap_attempts: np.ndarray
    swap_accepts: np.ndarray
    swap_length_rejections: np.ndarray
    walker_labels: np.ndarray
    rng_state: dict
    burn_in: int
    identity: dict
    run_identity: str

    @property
    def final_paths(self):
        return tuple(result.final_path for result in self.interfaces)

    def summary(self, *, block_size=None):
        local = [result.summary() for result in self.interfaces]
        n = local[0]["samples"]
        g = max(item["statistical_inefficiency"] for item in local)
        block = _integer(block_size, "block_size", 1) if block_size is not None else max(1, math.ceil(g))
        probabilities = [(item["p_LR"], item["p_RL"]) for item in local]
        supported = all(a is not None and b is not None for a, b in probabilities)
        overall = None
        if supported:
            try:
                overall = pptis_crossing_probability(*zip(*probabilities))
            except ValueError:
                supported = False
        estimates = []
        blocks = n // block
        for start in range(0, blocks * block, block):
            pairs = [_local_probabilities(r.starts[start:start + block], r.ends[start:start + block])
                     for r in self.interfaces]
            if any(a is None or b is None for a, b in pairs):
                continue
            try:
                estimates.append(pptis_crossing_probability(*zip(*pairs)))
            except ValueError:
                continue
        error = None
        # Missing-origin blocks must not be dropped to manufacture precision.
        if blocks >= 4 and len(estimates) == blocks and np.var(estimates) > 0:
            error = float(np.std(estimates, ddof=1) / np.sqrt(blocks))
        return {
            "method": self.identity["method"], "interfaces": local,
            "overall_crossing_probability": overall,
            "local_transition_support_complete": supported,
            "statistical_inefficiency": g,
            "block_size": block, "complete_blocks": blocks,
            "supported_blocks": len(estimates), "discarded_tail_samples": n % block,
            "crossing_probability_standard_error_block": error,
            "swap_attempts": self.swap_attempts.tolist(),
            "swap_accepts": self.swap_accepts.tolist(),
            "swap_length_rejections": self.swap_length_rejections.tolist(),
            "run_identity": self.run_identity,
            "memory_loss_approximation": True,
            "requires_independent_flux": True,
            "converged": False,
            "qualification": "PPTIS assumes loss of memory beyond the neighboring interfaces; "
                             "vary interface spacing and compare full TIS before interpreting rates.",
        }


def combine_pptis_rate(result, flux, *, block_size=None):
    """Combine a PPTIS ladder with effective A-history flux at its first center.

    The flux interface is ``result.interfaces[0].interface``, not basin A.
    This convention already includes the entrance probability p_0 of the
    original PPTIS formulation. Block error below conditions on supplied flux;
    flux uncertainty and inter-run convergence require separate evidence.
    """
    if not isinstance(result, PPTISResult):
        raise TypeError("result must be a PPTISResult")
    config = result.identity
    expected = {"basin_a": config["basin_a"], "basin_b": config["basin_b"],
                "first_interface": result.interfaces[0].interface,
                "dt_au": config["dt_au"]}
    if any(flux.get(key) != value for key, value in expected.items()):
        raise ValueError("flux must match PPTIS basins, timestep and first center interface")
    value = _finite(flux["flux_au"], "flux_au")
    if value < 0:
        raise ValueError("flux must be nonnegative")
    if not math.isfinite(value / ATOMIC_TIME_S):
        raise ValueError("flux conversion overflow")
    summary = result.summary(block_size=block_size)
    probability = summary["overall_crossing_probability"]
    candidate = None if probability is None else value * probability
    if candidate is not None and not math.isfinite(candidate / ATOMIC_TIME_S):
        raise ValueError("combined rate overflow")
    error = summary["crossing_probability_standard_error_block"]
    summary.update({
        "flux_au": value, "flux_s_inv": value / ATOMIC_TIME_S,
        "rate_estimate_au": candidate,
        "rate_estimate_s_inv": None if candidate is None else candidate / ATOMIC_TIME_S,
        "rate_standard_error_block_conditional_on_flux_au": None if error is None else value * error,
        "rate_s_inv": None,
        "uncertainty_qualification": "Joint ladder blocks retain replica correlations; "
                                     "the reported rate error excludes supplied-flux uncertainty.",
    })
    return summary


class PPTISSampler(TransitionInterfaceSampler):
    """Canonical local partial-path ensembles for reversible classical/RPMD MD.

    ``interfaces`` contains strictly interior *centers*. Ensemble i is bounded
    by its neighboring centers, or A/B at the ends. Unlike ordinary TIS, both
    path endpoints may be on either side. The supplied ``identity`` dictionary
    documents the potential and coordinate (including parameters); it is
    required for persistent checkpoints because arbitrary Python objects cannot
    be scientifically identified from their class names.
    """

    _exchange = False

    def __init__(self, dynamics, coordinate, *, basin_a, basin_b, interfaces,
                 max_path_steps=10000, seed=0, identity=None):
        super().__init__(dynamics, coordinate, basin_a=basin_a, basin_b=basin_b,
                         interfaces=interfaces, max_path_steps=max_path_steps, seed=seed)
        if self.interfaces[0] <= self.basin_a:
            raise ValueError("PPTIS centers require basin_a < interfaces[0] < ... < basin_b")
        if identity is not None and (not isinstance(identity, dict) or not identity):
            raise ValueError("identity must be a nonempty dictionary or None")
        try:
            caller_identity = json.loads(json.dumps(jsonable(identity, callables=False), allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("identity must contain finite JSON data") from error
        self._bounds = np.r_[self.basin_a, self.interfaces, self.basin_b]
        self._bounds.flags.writeable = False
        self.identity = {
            "method": "replica_exchange_pptis" if self._exchange else "partial_path_tis",
            "schema": "pyrpmd.pptis.v1", "caller_identity": caller_identity,
            "basin_a": self.basin_a, "basin_b": self.basin_b,
            "interfaces": self.interfaces.tolist(), "max_path_steps": self.max_path_steps,
            "dt_au": dynamics.dt, "beta_au": dynamics.beta, "nbeads": dynamics.nbeads,
            "masses": dynamics.masses.tolist(), "integrator": dynamics.integrator_name,
            "native": dynamics.native, "threads": dynamics.integrator.threads,
            "cell": None if dynamics.cell is None else dynamics.cell.tolist(),
            "pbc": None if dynamics.pbc is None else list(dynamics.pbc),
            "fixed_atoms": list(dynamics.fixed_atoms),
            "fixed_positions": None if dynamics.fixed_positions is None else dynamics.fixed_positions.tolist(),
            "potential_type": f"{type(dynamics.potential).__module__}.{type(dynamics.potential).__qualname__}",
            "coordinate_type": f"{type(coordinate).__module__}.{type(coordinate).__qualname__}",
            "shooting": "uniform_all_frames_full_maxwell_canonical_start",
            "swap": "independent_half_probability_orientations_terminal_overlap" if self._exchange else None,
            "rng_bit_generator": type(self.rng.bit_generator).__name__,
        }
        self.run_identity = fingerprint_payload(self.identity)
        self._local_configuration = self._local_signature()

    def _local_signature(self):
        return (self.basin_a, self.basin_b, self.interfaces.shape, self.interfaces.dtype.str,
                self.interfaces.tobytes(), self._bounds.shape, self._bounds.dtype.str,
                self._bounds.tobytes(), self.max_path_steps,
                self._exchange, id(self._coordinate), type(self.rng.bit_generator).__name__)

    def _check_dynamics(self):
        super()._check_dynamics()
        if hasattr(self, "_local_configuration") and (
            self._local_signature() != self._local_configuration
            or fingerprint_payload(self.identity) != self.run_identity
        ):
            raise ValueError("PPTIS configuration changed; create a new sampler")

    def _member(self, path, index):
        left, center, right = self._bounds[index:index + 3]
        v = path.values
        start_left, start_right = v[0] <= left, v[0] >= right
        return bool(
            len(path) - 1 <= self.max_path_steps
            and (start_left or start_right)
            and (v[-1] <= left or v[-1] >= right)
            and np.all((v[1:-1] > left) & (v[1:-1] < right))
            and ((start_left and np.any(v > center)) or (start_right and np.any(v < center)))
        )

    def _propagate_local(self, state, index, budget):
        left, right = self._bounds[index], self._bounds[index + 2]
        states, values = [], []
        for _ in range(budget):
            self.dynamics.step(state)
            value = self.value(state.q)
            states.append(state.copy())
            values.append(value)
            if value <= left or value >= right:
                return states, values
        raise PathGenerationError("max_path_steps")

    def _generate_path(self, shooting_state, interface_index, role):
        index = self._index(interface_index)
        self._check_dynamics()
        if not isinstance(shooting_state, State):
            raise TypeError("shooting_state must be a State")
        center = State(shooting_state.q, shooting_state.p)
        self.dynamics.integrator._check(center.q, center.p)
        self.dynamics._check_fixed(center.q, center.p)
        value = self.value(center.q)
        left, right = self._bounds[index], self._bounds[index + 2]
        if role == "interior" and not left < value < right:
            raise ValueError("shooting configuration must lie between the local outer interfaces")
        back, bv, forward, fv = [], [], [], []
        if role != "start":
            back, bv = self._propagate_local(
                State(center.q, -center.p), index, self.max_path_steps - int(role == "interior"))
        if role != "end":
            forward, fv = self._propagate_local(center.copy(), index, self.max_path_steps - len(back))
        path = TISPath(
            [s.q for s in reversed(back)] + [center.q] + [s.q for s in forward],
            [-s.p for s in reversed(back)] + [center.p] + [s.p for s in forward],
            bv[::-1] + [value] + fv, self.dynamics.dt,
        )
        if not self._member(path, index):
            raise PathGenerationError("interface_not_crossed")
        return path

    def _record(self, path, index):
        left = self._bounds[index]
        return (-1 if path.values[0] <= left else 1,
                -1 if path.values[-1] <= left else 1)

    def sample(self, initial_path, interface_index=0, *, n_samples, burn_in=0, on_sample=None):
        """Sample one local ensemble; use sample_interfaces for a complete rate."""
        index = self._index(interface_index)
        n_samples = _integer(n_samples, "n_samples", 1)
        burn_in = _integer(burn_in, "burn_in")
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable")
        path = self.validate_path(initial_path, index)
        starts, ends, accepted, lengths, reasons = [], [], [], [], []
        rejections = 0
        for attempt in range(burn_in + n_samples):
            path, move = self.shoot(path, index)
            rejections += move.reason == "max_path_steps"
            if attempt < burn_in:
                continue
            start, end = self._record(path, index)
            starts.append(start); ends.append(end)
            accepted.append(move.accepted); lengths.append(len(path)); reasons.append(move.reason)
            if on_sample is not None:
                on_sample(attempt - burn_in, path, move)
        return PPTISLocalResult(index, *self._bounds[index:index + 3],
                                np.array(starts, int), np.array(ends, int), np.array(accepted, bool),
                                np.array(lengths, int), tuple(reasons), path, int(rejections))

    def sample_interfaces(self, initial_paths=None, *, n_samples, burn_in=0, on_sample=None, resume=None):
        """Run synchronous sweeps, optionally appending to a validated result.

        ``resume`` is a result from this sampler configuration or load_checkpoint.
        Continuation restores its RNG and final paths, preserves labels and all
        diagnostic observations, and applies no new burn-in.
        """
        n_samples = _integer(n_samples, "n_samples", 1)
        burn_in = _integer(burn_in, "burn_in")
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable")
        n = len(self.interfaces)
        if resume is not None:
            if initial_paths is not None or burn_in:
                raise ValueError("resume cannot be combined with initial_paths or burn_in")
            self._validate_result(resume)
            paths = list(resume.final_paths)
            labels = resume.walker_labels[-1].copy()
            attempts, accepts = resume.swap_attempts.copy(), resume.swap_accepts.copy()
            swap_rejections = resume.swap_length_rejections.copy()
            rejections = np.array([r.length_rejections for r in resume.interfaces], int)
            initial_samples = len(resume.walker_labels)
            self.rng.bit_generator.state = copy.deepcopy(resume.rng_state)
        else:
            paths = list(initial_paths) if initial_paths is not None else []
            if len(paths) != n:
                raise ValueError("provide one initial path per local center interface")
            for i, path in enumerate(paths):
                self.validate_path(path, i)
            labels = np.arange(n)
            attempts, accepts = np.zeros(n - 1, int), np.zeros(n - 1, int)
            swap_rejections = np.zeros(n - 1, int)
            rejections = np.zeros(n, int)
            initial_samples = 0
        starts, ends, accepted, lengths = (np.empty((n_samples, n), dtype=d)
                                          for d in (int, int, bool, int))
        label_history = np.empty((n_samples, n), int)
        reasons = [[] for _ in range(n)]
        for sweep in range(burn_in + n_samples):
            moves = []
            for i, path in enumerate(paths):
                paths[i], move = self.shoot(path, i)
                rejections[i] += move.reason == "max_path_steps"
                moves.append(move)
            if self._exchange:
                parity = int(self.rng.integers(0, 2)) if n > 2 else 0
                for i in range(parity, n - 1, 2):
                    swapped = self.exchange_paths(paths, i, i + 1)
                    attempts[i] += 1
                    accepts[i] += swapped
                    swap_rejections[i] += self._last_swap_reason == "max_path_steps"
                    if swapped:
                        labels[[i, i + 1]] = labels[[i + 1, i]]
            if sweep < burn_in:
                continue
            sample = sweep - burn_in
            for i, (path, move) in enumerate(zip(paths, moves)):
                starts[sample, i], ends[sample, i] = self._record(path, i)
                accepted[sample, i] = move.accepted
                lengths[sample, i] = len(path)
                reasons[i].append(move.reason)
            label_history[sample] = labels
            if on_sample is not None:
                on_sample(initial_samples + sample, tuple(paths), tuple(moves))
        results = []
        for i in range(n):
            previous = None if resume is None else resume.interfaces[i]
            arrays = [starts[:, i], ends[:, i], accepted[:, i], lengths[:, i]]
            why = tuple(reasons[i])
            if previous is not None:
                arrays = [np.r_[getattr(previous, name), array] for name, array in
                          zip(("starts", "ends", "accepted", "path_frames"), arrays)]
                why = previous.reasons + why
            results.append(PPTISLocalResult(i, *self._bounds[i:i + 3], *arrays,
                                            why, paths[i], int(rejections[i])))
        if resume is not None:
            label_history = np.concatenate([resume.walker_labels, label_history])
        return PPTISResult(tuple(results), attempts, accepts, swap_rejections, label_history,
                           copy.deepcopy(self.rng.bit_generator.state),
                           burn_in if resume is None else resume.burn_in,
                           copy.deepcopy(self.identity), self.run_identity)

    def _validate_result(self, result):
        self._check_dynamics()
        if not isinstance(result, PPTISResult) or result.identity != self.identity or result.run_identity != self.run_identity:
            raise ValueError("PPTIS checkpoint scientific identity does not match sampler")
        n = len(self.interfaces)
        if len(result.interfaces) != n:
            raise ValueError("checkpoint path bank has wrong ensemble count")
        count = len(result.walker_labels)
        if (count < 1 or result.walker_labels.shape != (count, n)
                or result.walker_labels.dtype.kind not in "iu"
                or np.any(np.sort(result.walker_labels, axis=1) != np.arange(n))):
            raise ValueError("invalid checkpoint walker labels")
        for counts in (result.swap_attempts, result.swap_accepts, result.swap_length_rejections):
            if counts.shape != (n - 1,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
                raise ValueError("invalid checkpoint swap counters")
        if np.any(result.swap_accepts > result.swap_attempts):
            raise ValueError("checkpoint accepts exceed attempts")
        _integer(result.burn_in, "checkpoint burn_in")
        sweeps = count + result.burn_in
        if (np.any(result.swap_attempts > sweeps)
                or np.any(result.swap_length_rejections > result.swap_attempts - result.swap_accepts)
                or not self._exchange and np.any(result.swap_attempts)):
            raise ValueError("invalid checkpoint swap counters for sweep count")
        for i, local in enumerate(result.interfaces):
            if (local.interface_index != i or (local.left, local.interface, local.right) != tuple(self._bounds[i:i + 3])
                    or len(local.reasons) != count or not all(isinstance(x, str) for x in local.reasons)):
                raise ValueError("invalid checkpoint local metadata")
            for values in (local.starts, local.ends):
                if values.shape != (count,) or values.dtype.kind not in "iu" or not np.isin(values, [-1, 1]).all():
                    raise ValueError("invalid checkpoint endpoint history")
            if local.accepted.shape != (count,) or local.accepted.dtype.kind != "b":
                raise ValueError("invalid checkpoint acceptance history")
            if (local.path_frames.shape != (count,) or local.path_frames.dtype.kind not in "iu"
                    or np.any((local.path_frames < 2) | (local.path_frames > self.max_path_steps + 1))):
                raise ValueError("invalid checkpoint path lengths")
            _integer(local.length_rejections, "checkpoint length_rejections")
            if local.length_rejections > sweeps:
                raise ValueError("checkpoint length rejections exceed sweep count")
            self.validate_path(local.final_path, i)
            if (local.path_frames[-1] != len(local.final_path)
                    or (local.starts[-1], local.ends[-1]) != self._record(local.final_path, i)):
                raise ValueError("checkpoint final path disagrees with last observation")
        # Validate without changing the live RNG on a failed load.
        trial = type(self.rng.bit_generator)()
        try:
            trial.state = copy.deepcopy(result.rng_state)
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("invalid checkpoint RNG state or bit-generator type") from error
        return result

    def save_checkpoint(self, filename, result):
        """Atomically save the complete retained bank, observations and RNG."""
        if not self.identity["caller_identity"]:
            raise ValueError("persistent checkpoints require explicit potential/coordinate identity")
        self._validate_result(result)
        metadata = {"format": "PyRPMD.pptis.v1", "identity": self.identity,
                    "run_identity": self.run_identity, "rng_state": result.rng_state,
                    "burn_in": result.burn_in,
                    "reasons": [r.reasons for r in result.interfaces],
                    "length_rejections": [r.length_rejections for r in result.interfaces]}
        payload = {"metadata": np.asarray(json.dumps(jsonable(metadata), allow_nan=False)),
                   "walker_labels": result.walker_labels,
                   "swap_attempts": result.swap_attempts, "swap_accepts": result.swap_accepts,
                   "swap_length_rejections": result.swap_length_rejections}
        for i, local in enumerate(result.interfaces):
            for name in ("starts", "ends", "accepted", "path_frames"):
                payload[f"{name}_{i}"] = getattr(local, name)
            for name in ("q", "p", "values"):
                payload[f"{name}_{i}"] = getattr(local.final_path, name)
        atomic_savez(filename, kind="PyRPMD.pptis.v1", **payload)

    def load_checkpoint(self, filename):
        """Validate an archive and return a result usable as ``resume=...``."""
        if not self.identity["caller_identity"]:
            raise ValueError("persistent checkpoints require explicit potential/coordinate identity")
        with load_checkpoint(filename, expected_kind="PyRPMD.pptis.v1", require_integrity=True) as data:
            try:
                metadata = json.loads(str(data["metadata"].item()))
                if metadata["identity"] != self.identity or metadata["run_identity"] != self.run_identity:
                    raise ValueError("PPTIS checkpoint scientific identity does not match sampler")
                expected = {"metadata", "walker_labels", "swap_attempts", "swap_accepts", "swap_length_rejections"}
                expected.update(f"{name}_{i}" for name in ("q", "p", "values", "starts", "ends", "accepted", "path_frames")
                                for i in range(len(self.interfaces)))
                if set(data.files) != expected:
                    raise ValueError("unexpected PPTIS checkpoint payload inventory")
                if (len(metadata["reasons"]) != len(self.interfaces)
                        or len(metadata["length_rejections"]) != len(self.interfaces)):
                    raise ValueError("wrong PPTIS checkpoint local metadata count")
                locals_ = []
                for i in range(len(self.interfaces)):
                    path = TISPath(data[f"q_{i}"], data[f"p_{i}"], data[f"values_{i}"], self.dynamics.dt)
                    arrays = [data[f"{name}_{i}"].copy() for name in ("starts", "ends", "accepted", "path_frames")]
                    locals_.append(PPTISLocalResult(i, *self._bounds[i:i + 3], *arrays,
                        tuple(metadata["reasons"][i]), path, metadata["length_rejections"][i]))
                result = PPTISResult(tuple(locals_), data["swap_attempts"].copy(), data["swap_accepts"].copy(),
                    data["swap_length_rejections"].copy(), data["walker_labels"].copy(), metadata["rng_state"], metadata["burn_in"],
                    metadata["identity"], metadata["run_identity"])
                return self._validate_result(result)
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise CheckpointError(f"Invalid PPTIS checkpoint: {error}") from error


class ReplicaExchangePPTIS(PPTISSampler):
    """PPTIS with neighboring overlap-regrowth swaps (REPPTIS).

    An attempted swap chooses one of two orientations independently for each
    path *before* checking compatibility. Invalid directions reject the whole
    move. Selecting only compatible directions would bias path types. The
    terminal overlap is retained and extended into the neighboring window;
    paths are never exchanged by relabeling incompatible complete arrays.
    """

    _exchange = True

    def _extend_overlap(self, path, source, target, reverse):
        oriented = path.reversed() if reverse else path
        going_right = target > source
        endpoint = oriented.values[-1]
        if (going_right and endpoint < self._bounds[source + 2]
                or not going_right and endpoint > self._bounds[source]):
            raise PathGenerationError("incompatible_swap_direction")
        center = self.interfaces[source]
        opposite = oriented.values <= center if going_right else oriented.values >= center
        candidates = np.flatnonzero(opposite)
        if not len(candidates):
            raise PathGenerationError("interface_not_crossed")
        start = int(candidates[-1])
        q, p, values = list(oriented.q[start:]), list(oriented.p[start:]), list(oriented.values[start:])
        state = oriented.state(len(oriented) - 1)
        left, right = self._bounds[target], self._bounds[target + 2]
        # A step can leap across the complete target window. The two-frame
        # overlap is already a complete path in that case.
        if left < values[-1] < right:
            forward, fv = self._propagate_local(state, target, self.max_path_steps - len(q) + 1)
            q.extend(s.q for s in forward); p.extend(s.p for s in forward); values.extend(fv)
        trial = TISPath(q, p, values, self.dynamics.dt)
        if reverse:
            trial = trial.reversed()
        if len(trial) - 1 > self.max_path_steps:
            raise PathGenerationError("max_path_steps")
        if not self._member(trial, target):
            raise PathGenerationError("interface_not_crossed")
        return trial

    def _swap_trial(self, left_path, right_path, index, directions):
        """Deterministic proposal for two independently chosen reversal flags."""
        reverse_left, reverse_right = directions
        trial_left = self._extend_overlap(right_path, index + 1, index, reverse_right)
        trial_right = self._extend_overlap(left_path, index, index + 1, reverse_left)
        return trial_left, trial_right

    def _swap_log_ratio(self, old, new):
        # Overlap shifts and reversible propagation preserve phase volume.
        # Each forward/reverse direction pair has probability 1/4. Unlike
        # shooting, this move does not select a uniformly random path frame.
        delta_h = sum(self.dynamics.energy(p.state(0)) for p in new)
        delta_h -= sum(self.dynamics.energy(p.state(0)) for p in old)
        return -self.dynamics.beta / self.dynamics.nbeads * delta_h

    def exchange_paths(self, paths, i, j):
        i, j = self._index(i), self._index(j)
        if j != i + 1 or len(paths) != len(self.interfaces):
            raise ValueError("REPPTIS exchange requires adjacent ordered ensembles and a complete path bank")
        self._check_dynamics()
        if not self._member(paths[i], i) or not self._member(paths[j], j):
            raise ValueError("current path is outside its local interface ensemble")
        directions = tuple(bool(x) for x in self.rng.integers(0, 2, size=2))
        try:
            proposed = self._swap_trial(paths[i], paths[j], i, directions)
        except PathGenerationError as error:
            self._last_swap_reason = error.reason
            return False
        ratio = self._swap_log_ratio((paths[i], paths[j]), proposed)
        if not math.isfinite(ratio):
            raise FloatingPointError("nonfinite REPPTIS swap acceptance ratio")
        accepted = math.log(max(float(self.rng.random()), np.finfo(float).tiny)) < min(0.0, ratio)
        if accepted:
            paths[i], paths[j] = proposed
        self._last_swap_reason = "accepted" if accepted else "metropolis"
        return bool(accepted)
