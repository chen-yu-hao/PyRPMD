"""Canonical RETIS with optional minus paths, internal flux and exact restart.

The positive-only API remains available. ``minus=True`` adds complementary
excursions inside A, endpoint-continuation exchange, and residence-time flux.
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
    PathGenerationError, TISMove, TISPath, TISResult,
    TransitionInterfaceSampler, _finite, _integer,
)


def _block_diagnostics(data, statistic, block_size=None):
    """Synchronous contiguous blocks preserve cross-observable correlations."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or not len(data) or not np.isfinite(data).all():
        raise ValueError("block observations must be a nonempty finite matrix")
    n = len(data)
    size = max(1, int(np.sqrt(n))) if block_size is None else _integer(block_size, "block_size", 1)
    blocks = n // size
    point = np.atleast_1d(statistic(data))
    estimates = np.asarray([
        np.atleast_1d(statistic(data[i * size:(i + 1) * size])) for i in range(blocks)
    ], dtype=float).reshape(blocks, len(point))
    if not np.isfinite(point).all() or not np.isfinite(estimates).all():
        raise FloatingPointError("nonfinite block statistic")
    errors = [None] * len(point)
    covariance = None
    covariance_reason = "fewer_than_two_blocks"
    if blocks >= 2:
        for j in range(len(point)):
            if np.any(estimates[:, j] != estimates[0, j]):
                scale = float(np.max(np.abs(estimates[:, j])))
                errors[j] = float(np.std(estimates[:, j] / scale, ddof=1) * (scale / np.sqrt(blocks)))
        means = data[:blocks * size].reshape(blocks, size, data.shape[1]).mean(axis=1)
        scales = np.max(np.abs(means), axis=0)
        scales[scales == 0] = 1.
        scaled_cov = np.atleast_2d(np.cov(means / scales, rowvar=False, ddof=1))
        with np.errstate(over="ignore", invalid="ignore"):
            cov = scaled_cov * (scales[:, None] / np.sqrt(blocks)) * (scales[None, :] / np.sqrt(blocks))
        if np.isfinite(cov).all():
            covariance, covariance_reason = cov.tolist(), None
        else:
            covariance_reason = "covariance_exceeds_float64_range"
    # Statistical inefficiency is scale-invariant; normalize to avoid squared
    # intermediate overflow/underflow for otherwise valid finite time units.
    g = [float(statistical_inefficiency(column / max(float(np.max(np.abs(column))), np.finfo(float).tiny)))
         for column in data.T]
    return {
        "block_size": size, "blocks": blocks,
        "samples_excluded_from_block_error": n - blocks * size,
        "block_estimates": estimates.tolist(), "standard_errors": errors,
        "statistical_inefficiencies": g,
        "effective_samples_diagnostic": [n / value for value in g],
        "covariance_of_observable_means": covariance,
        "covariance_unavailable_reason": covariance_reason,
        "qualification": "Joint-block diagnostics; check a block-size plateau and independent chains.",
    }


def estimate_retis_flux(minus_frames, plus_frames, *, dt, basin_a, basin_b,
                        first_interface, block_size=None):
    """A-history flux from synchronized [0-]/[0+] chains, including repeats.

    Each excursion contributes ``(frames-2)*dt``. Minus interior frames count
    A residence including the outward crossing interval; plus interiors count
    the subsequent A-history intervals up to first return to A or arrival at B.
    This also handles a two-frame A-to-B plus path (zero plus residence).
    """
    minus, plus = np.asarray(minus_frames), np.asarray(plus_frames)
    if (minus.dtype.kind not in "iu" or plus.dtype.kind not in "iu"
            or minus.ndim != 1 or not len(minus) or plus.shape != minus.shape
            or np.any(minus < 3) or np.any(plus < 2)):
        raise ValueError("need matching integer minus>=3 and plus>=2 frame counts")
    a, b, first, step = (_finite(value, name) for value, name in (
        (basin_a, "basin_a"), (basin_b, "basin_b"),
        (first_interface, "first_interface"), (dt, "dt"),
    ))
    if a != first or not a < b or step <= 0:
        raise ValueError("minus flux requires first_interface == basin_a < basin_b and dt>0")
    with np.errstate(over="ignore", invalid="ignore"):
        times = np.column_stack((minus - 2, plus - 2)).astype(float) * step
        mean_time = float(times.mean(axis=0).sum())
    if mean_time <= 0 or not math.isfinite(mean_time):
        raise ValueError("RETIS residence time must be finite and positive")
    flux = 1.0 / mean_time
    if not math.isfinite(flux / ATOMIC_TIME_S):
        raise ValueError("RETIS flux overflow")
    diagnostic = _block_diagnostics(times, lambda x: 1.0 / x.mean(axis=0).sum(), block_size)
    return {
        "method": "retis_minus_residence_time",
        "first_interface": first, "basin_a": a, "basin_b": b, "dt_au": step,
        "samples": len(minus), "mean_minus_time_au": float(times[:, 0].mean()),
        "mean_plus_time_au": float(times[:, 1].mean()),
        "flux_au": flux, "flux_s_inv": flux / ATOMIC_TIME_S,
        "flux_standard_error_block_au": diagnostic["standard_errors"][0],
        "block_diagnostics": diagnostic, "converged": False, "physical_converged": False,
    }


class _MinusInterfaceSampler(TransitionInterfaceSampler):
    """First-return A excursions with the ordinary TIS shooting density."""

    def _basin(self, value):
        return int(value > self.basin_a)

    def _member(self, path, index):
        v = path.values
        return bool(index == 0 and 3 <= len(path) <= self.max_path_steps + 1
                    and v[0] > self.basin_a and v[-1] > self.basin_a
                    and np.all(v[1:-1] <= self.basin_a))

    def _generate_path(self, shooting_state, interface_index, role):
        self._index(interface_index)
        self._check_dynamics()
        if not isinstance(shooting_state, State):
            raise TypeError("shooting_state must be a State")
        center = State(shooting_state.q, shooting_state.p)
        self.dynamics.integrator._check(center.q, center.p)
        self.dynamics._check_fixed(center.q, center.p)
        value = self.value(center.q)
        if role == "interior" and value > self.basin_a:
            raise ValueError("minus shooting configuration must lie inside A")
        back, bv, forward, fv = [], [], [], []
        if role != "start":
            back, bv = self._propagate_to_basin(
                State(center.q, -center.p), self.max_path_steps - int(role == "interior"))
        if role != "end":
            forward, fv = self._propagate_to_basin(center.copy(), self.max_path_steps - len(back))
        trial = TISPath(
            [x.q for x in reversed(back)] + [center.q] + [x.q for x in forward],
            [-x.p for x in reversed(back)] + [center.p] + [x.p for x in forward],
            bv[::-1] + [value] + fv, self.dynamics.dt,
        )
        if not self._member(trial, 0):
            raise PathGenerationError("minus_not_entered")
        return trial


@dataclass
class MinusTISResult:
    path_frames: np.ndarray
    accepted: np.ndarray
    reasons: tuple[str, ...]
    final_path: TISPath
    length_rejections: int

    def summary(self):
        return {
            "method": "retis_minus_ensemble", "samples": len(self.path_frames),
            "mean_path_frames": float(np.mean(self.path_frames)),
            "acceptance_fraction": float(np.mean(self.accepted)),
            "statistical_inefficiency": float(statistical_inefficiency(self.path_frames)),
            "length_rejections": self.length_rejections,
            "converged": False, "physical_converged": False,
        }


@dataclass
class RETISResult:
    interfaces: tuple[TISResult, ...]
    swap_attempts: np.ndarray
    swap_accepts: np.ndarray
    walker_labels: np.ndarray
    rng_state: dict
    minus: MinusTISResult | None = None
    minus_walker_labels: np.ndarray | None = None
    zero_swap_attempts: int = 0
    zero_swap_accepts: int = 0
    zero_swap_length_rejections: int = 0
    run_identity: str | None = None

    def summary(self, *, block_size=None):
        crossings = np.column_stack([r.crossing for r in self.interfaces])
        probabilities = crossings.mean(axis=0)
        product = float(np.prod(probabilities))
        result = {"method": "replica_exchange_tis" if self.minus else "replica_exchange_tis_positive_ensembles",
                "interfaces": [result.summary() for result in self.interfaces],
                "swap_attempts": self.swap_attempts.tolist(),
                "swap_accepts": self.swap_accepts.tolist(),
                "conditional_probabilities": probabilities.tolist(),
                "crossing_probability_product": product, "overall_crossing_probability": product,
                "requires_independent_flux": self.minus is None, "run_identity": self.run_identity,
                "rate_s_inv": None, "converged": False, "physical_converged": False}
        if self.minus is None:
            result["block_diagnostics"] = _block_diagnostics(
                crossings, lambda x: np.prod(x.mean(axis=0)), block_size)
            result["crossing_probability_standard_error_block"] = result["block_diagnostics"]["standard_errors"][0]
            return result
        first = self.interfaces[0]
        flux = estimate_retis_flux(
            self.minus.path_frames, first.path_frames, dt=first.final_path.dt,
            basin_a=first.basin_a, basin_b=first.basin_b, first_interface=first.interface,
            block_size=block_size)
        durations = (self.minus.path_frames + first.path_frames - 4) * first.final_path.dt

        def statistics(data):
            probability = float(np.prod(data[:, :-1].mean(axis=0)))
            f = 1.0 / float(data[:, -1].mean())
            return probability, f, probability * f

        diagnostic = _block_diagnostics(np.column_stack((crossings, durations)), statistics, block_size)
        errors = diagnostic["standard_errors"]
        rate = product * flux["flux_au"]
        result.update({
            "minus": self.minus.summary(), "flux": flux,
            "zero_swap_attempts": self.zero_swap_attempts, "zero_swap_accepts": self.zero_swap_accepts,
            "zero_swap_length_rejections": self.zero_swap_length_rejections,
            "flux_au": flux["flux_au"], "flux_s_inv": flux["flux_s_inv"],
            "rate_estimate_au": rate, "rate_estimate_s_inv": rate / ATOMIC_TIME_S,
            "crossing_probability_standard_error_block": errors[0],
            "flux_standard_error_block_au": errors[1], "rate_standard_error_block_au": errors[2],
            "block_diagnostics": diagnostic,
            "qualification": "Candidate first-order rate; establish basin, timestep, bead, path-length and independent-chain convergence.",
        })
        return result


class ReplicaExchangeTIS(TransitionInterfaceSampler):
    """Two-way shooting plus neighboring path-exchange sweeps.

    Each sweep first attempts one reversible shooting move in every ensemble,
    then attempts disjoint neighboring swaps at randomly selected parity.
    Observations are taken after swaps, including repeats after rejected moves.
    Across-interface samples are correlated and must be resampled jointly.

    ``minus=True`` requires ``interfaces[0] == basin_a``. ``identity`` is an
    explicit JSON mapping identifying the potential and CV, including opaque
    callable parameters. It is required for portable checkpoints; sampler and
    Dynamics settings are included automatically. No global workflow identity
    or legacy checkpoint is changed. A continued call without ``initial_paths``
    returns cumulative observations from the initialized or restored chain.
    """

    def __init__(self, dynamics, coordinate, *, basin_a, basin_b, interfaces,
                 max_path_steps=10000, seed=0, minus=False, identity=None):
        super().__init__(dynamics, coordinate, basin_a=basin_a, basin_b=basin_b,
                         interfaces=interfaces, max_path_steps=max_path_steps, seed=seed)
        if not isinstance(minus, (bool, np.bool_)):
            raise ValueError("minus must be boolean")
        self.minus_enabled = bool(minus)
        if self.minus_enabled and self.interfaces[0] != self.basin_a:
            raise ValueError("minus RETIS requires first interface == basin_a")
        if self.minus_enabled and self.max_path_steps < 2:
            raise ValueError("minus RETIS needs max_path_steps >= 2")
        if identity is not None and (not isinstance(identity, dict) or not identity):
            raise ValueError("identity must be a nonempty JSON mapping")
        try:
            self.identity = json.loads(json.dumps(jsonable(identity, callables=False), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("identity must contain finite JSON data") from exc
        self._minus = None
        if self.minus_enabled:
            self._minus = _MinusInterfaceSampler(
                dynamics, coordinate, basin_a=basin_a, basin_b=basin_b,
                interfaces=[basin_a], max_path_steps=max_path_steps, seed=self.rng)
        self._configuration_at_start = self._configuration()
        self.run_identity = fingerprint_payload(self._configuration_at_start)
        self._coordinate_at_start = self._coordinate
        self._paths = None

    def _configuration(self):
        dyn = self.dynamics
        return jsonable({
            "format": "PyRPMD.retis.v1", "identity": self.identity,
            "potential_type": f"{type(dyn.potential).__module__}.{type(dyn.potential).__qualname__}",
            "basin_a": self.basin_a, "basin_b": self.basin_b, "interfaces": self.interfaces,
            "max_path_steps": self.max_path_steps, "minus": self.minus_enabled,
            "beta": dyn.beta, "nbeads": dyn.nbeads, "dt": dyn.dt, "masses": dyn.masses,
            "fixed_atoms": dyn.fixed_atoms, "fixed_positions": dyn.fixed_positions,
            "native": dyn.native, "threads": dyn.integrator.threads,
            "cell": dyn.cell, "pbc": dyn.pbc,
            "integrator": dyn.integrator_name, "rng_type": type(self.rng.bit_generator).__name__,
            "shooting": "all_frames_full_maxwell_canonical_start_v1",
            "exchange": "shoot_all_then_random_disjoint_parity_v1",
            "minus_flux_time": "interior_frames_times_dt_v1",
        }, callables=False)

    def _check_configuration(self):
        self._check_dynamics()
        if (self._configuration() != self._configuration_at_start
                or self._coordinate is not self._coordinate_at_start
                or fingerprint_payload(self._configuration_at_start) != self.run_identity):
            raise ValueError("RETIS configuration changed; create a new sampler")

    def _require_minus(self):
        if self._minus is None:
            raise ValueError("this operation requires minus=True")
        self._check_configuration()

    def generate_minus_path(self, shooting_state):
        self._require_minus()
        return self._minus.generate_path(shooting_state)

    def validate_minus_path(self, path):
        self._require_minus()
        return self._minus.validate_path(path)

    def _zero_proposal(self, minus_path, plus_path):
        # Exchange outgoing crossing edges; continue into the opposite domain.
        # This deterministic mapping is its own inverse and selects no frame.
        first = plus_path.state(0)
        back, bv = self._minus._propagate_to_basin(
            State(first.q, -first.p), self.max_path_steps - 1)
        new_minus = TISPath(
            [x.q for x in reversed(back)] + list(plus_path.q[:2]),
            [-x.p for x in reversed(back)] + list(plus_path.p[:2]),
            bv[::-1] + list(plus_path.values[:2]), self.dynamics.dt)
        forward, fv = [], []
        if not self._basin(minus_path.values[-1]):
            forward, fv = self._propagate_to_basin(
                minus_path.state(len(minus_path) - 1), self.max_path_steps - 1)
        new_plus = TISPath(
            list(minus_path.q[-2:]) + [x.q for x in forward],
            list(minus_path.p[-2:]) + [x.p for x in forward],
            list(minus_path.values[-2:]) + fv, self.dynamics.dt)
        if not self._minus._member(new_minus, 0) or not self._member(new_plus, 0):
            raise PathGenerationError("zero_swap_membership")
        return new_minus, new_plus

    def _zero_log_ratio(self, old_minus, old_plus, new_minus, new_plus):
        dyn = self.dynamics
        delta_h = (dyn.energy(new_minus.state(0)) + dyn.energy(new_plus.state(0))
                   - dyn.energy(old_minus.state(0)) - dyn.energy(old_plus.state(0)))
        # Unit-Jacobian endpoint continuation has no shooting-length ratio.
        return -dyn.beta / dyn.nbeads * delta_h

    def exchange_zero(self, minus_path, plus_path):
        """Return retained minus/plus paths and the endpoint-swap decision.

        External seeds must first be validated, as for direct ``shoot`` calls.
        Only path-generation failures are rejections; potential errors escape.
        """
        self._require_minus()
        if not self._minus._member(minus_path, 0) or not self._member(plus_path, 0):
            raise ValueError("zero-swap paths are outside their ensembles")
        try:
            new_minus, new_plus = self._zero_proposal(minus_path, plus_path)
        except PathGenerationError as exc:
            return minus_path, plus_path, TISMove(False, exc.reason, -1, None, None)
        log_ratio = self._zero_log_ratio(minus_path, plus_path, new_minus, new_plus)
        if not math.isfinite(log_ratio):
            raise FloatingPointError("nonfinite zero-swap acceptance ratio")
        accepted = math.log(max(float(self.rng.random()), np.finfo(float).tiny)) < min(0., log_ratio)
        move = TISMove(accepted, "accepted" if accepted else "metropolis", -1,
                       len(new_minus) + len(new_plus), log_ratio)
        return (new_minus, new_plus, move) if accepted else (minus_path, plus_path, move)

    def exchange_paths(self, paths, i, j):
        i, j = self._index(i), self._index(j)
        if i == j:
            raise ValueError("path exchange requires two distinct ensembles")
        if len(paths) != len(self.interfaces):
            raise ValueError("one path is required per interface")
        if not self._member(paths[i], i) or not self._member(paths[j], j):
            raise ValueError("current path is outside its interface ensemble")
        accepted = self._member(paths[i], j) and self._member(paths[j], i)
        if accepted:
            paths[i], paths[j] = paths[j], paths[i]
        return bool(accepted)

    def initialize(self, initial_paths, *, minus_path=None):
        self._check_configuration()
        paths = list(initial_paths)
        if len(paths) != len(self.interfaces):
            raise ValueError("provide one seed path per interface")
        for i, path in enumerate(paths):
            self.validate_path(path, i)
        if self.minus_enabled:
            self.validate_minus_path(minus_path)
        elif minus_path is not None:
            raise ValueError("minus_path requires minus=True")
        n = len(paths)
        self._paths, self._minus_path = paths, minus_path
        self._labels = np.arange(n + int(self.minus_enabled), dtype=int)
        self._crossing, self._accepted, self._lengths, self._label_history = [], [], [], []
        self._reasons = [[] for _ in paths]
        self._minus_frames, self._minus_accepted, self._minus_reasons, self._minus_label_history = [], [], [], []
        self._length_rejections = np.zeros(n, int)
        self._swap_attempts = np.zeros(max(0, n - 1), int)
        self._swap_accepts = np.zeros_like(self._swap_attempts)
        self._minus_length_rejections = 0
        self._zero_swap_attempts = self._zero_swap_accepts = self._zero_swap_length_rejections = 0
        self._sweeps = self._burn_in = 0
        return self

    def sample_interfaces(self, initial_paths=None, *, n_samples, burn_in=0,
                          on_sample=None, minus_path=None):
        n_samples = _integer(n_samples, "n_samples", 1)
        burn_in = _integer(burn_in, "burn_in")
        self._check_configuration()
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable")
        if initial_paths is not None:
            self.initialize(initial_paths, minus_path=minus_path)
        elif self._paths is None:
            raise ValueError("initialize paths or load a checkpoint before continuing")
        elif minus_path is not None:
            raise ValueError("continued chains must not replace their minus path")
        if self._crossing and burn_in:
            raise ValueError("burn_in is only allowed at the beginning of a chain")
        n, offset = len(self._paths), int(self.minus_enabled)
        for sweep in range(burn_in + n_samples):
            moves, minus_move = self._sweep()
            self._sweeps += 1
            if sweep < burn_in:
                self._burn_in += 1
                continue
            self._crossing.append([
                bool(path.values[-1] >= self.basin_b if i == n - 1
                     else np.any(path.values > self.interfaces[i + 1]))
                for i, path in enumerate(self._paths)])
            self._accepted.append([move.accepted for move in moves])
            self._lengths.append([len(path) for path in self._paths])
            self._label_history.append(self._labels[offset:].copy())
            for reasons, move in zip(self._reasons, moves):
                reasons.append(move.reason)
            if self.minus_enabled:
                self._minus_frames.append(len(self._minus_path))
                self._minus_accepted.append(minus_move.accepted)
                self._minus_reasons.append(minus_move.reason)
                self._minus_label_history.append(int(self._labels[0]))
            if on_sample is not None:
                on_sample(len(self._crossing) - 1, tuple(self._paths), tuple(moves))
        return self.result()

    def _sweep(self):
        # Snapshot immutable path references and small counters only, never
        # copy the full path bank. A numerical exception cannot leave a
        # half-updated bank whose last observation/checkpoint describes an
        # earlier sweep. Callbacks run after this transaction has committed.
        fields = ("_paths", "_minus_path", "_labels", "_length_rejections",
                  "_swap_attempts", "_swap_accepts", "_minus_length_rejections",
                  "_zero_swap_attempts", "_zero_swap_accepts", "_zero_swap_length_rejections")
        saved = {}
        for name in fields:
            value = getattr(self, name)
            saved[name] = value.copy() if isinstance(value, (list, np.ndarray)) else value
        rng = copy.deepcopy(self.rng.bit_generator.state)
        try:
            minus_move = None
            if self.minus_enabled:
                self._minus_path, minus_move = self._minus.shoot(self._minus_path)
                self._minus_length_rejections += minus_move.reason == "max_path_steps"
            moves = []
            for i, path in enumerate(self._paths):
                self._paths[i], move = self.shoot(path, i)
                self._length_rejections[i] += move.reason == "max_path_steps"
                moves.append(move)
            offset = int(self.minus_enabled)
            total = len(self._paths) + offset
            parity = int(self.rng.integers(0, 2)) if total > 2 else 0
            for k in range(parity, total - 1, 2):
                if offset and k == 0:
                    self._minus_path, self._paths[0], exchange = self.exchange_zero(self._minus_path, self._paths[0])
                    swapped = exchange.accepted
                    self._zero_swap_attempts += 1
                    self._zero_swap_accepts += swapped
                    self._zero_swap_length_rejections += exchange.reason == "max_path_steps"
                else:
                    i = k - offset
                    swapped = self.exchange_paths(self._paths, i, i + 1)
                    self._swap_attempts[i] += 1
                    self._swap_accepts[i] += swapped
                if swapped:
                    self._labels[[k, k + 1]] = self._labels[[k + 1, k]]
        except BaseException:
            for name, value in saved.items():
                setattr(self, name, value)
            self.rng.bit_generator.state = rng
            raise
        return moves, minus_move

    def result(self):
        self._check_configuration()
        if self._paths is None or not self._crossing:
            raise ValueError("no production samples available")
        n = len(self._paths)
        crossing, accepted, lengths = map(np.asarray, (self._crossing, self._accepted, self._lengths))
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        results = tuple(TISResult(
            i, float(self.interfaces[i]), self.basin_b if i == n - 1 else float(self.interfaces[i + 1]),
            i == n - 1, self.basin_a, self.basin_b, crossing[:, i].copy(), accepted[:, i].copy(), lengths[:, i].copy(),
            tuple(self._reasons[i]), self._paths[i], copy.deepcopy(rng_state), self._burn_in, self.max_path_steps,
            int(self._length_rejections[i]), self.dynamics.beta, self.dynamics.nbeads,
        ) for i in range(n))
        minus = None
        if self.minus_enabled:
            minus = MinusTISResult(np.asarray(self._minus_frames, int), np.asarray(self._minus_accepted, bool),
                                   tuple(self._minus_reasons), self._minus_path, int(self._minus_length_rejections))
        return RETISResult(results, self._swap_attempts.copy(), self._swap_accepts.copy(),
                           np.asarray(self._label_history, int), rng_state, minus,
                           np.asarray(self._minus_label_history, int) if self.minus_enabled else None,
                           int(self._zero_swap_attempts), int(self._zero_swap_accepts),
                           int(self._zero_swap_length_rejections), self.run_identity)

    def save_checkpoint(self, path):
        """Atomically save current paths, cumulative observations and RNG."""
        self._check_configuration()
        if self.identity is None:
            raise ValueError("checkpoint requires explicit potential/coordinate identity")
        if self._paths is None:
            raise ValueError("initialize paths before saving a checkpoint")
        n, count = len(self._paths), len(self._crossing)
        metadata = {
            "format": "PyRPMD.retis.v1", "configuration": self._configuration_at_start,
            "run_identity": self.run_identity, "rng_state": self.rng.bit_generator.state,
            "sweeps": self._sweeps, "burn_in": self._burn_in, "samples": count,
            "reasons": self._reasons, "minus_reasons": self._minus_reasons,
        }
        for key in ("minus_length_rejections", "zero_swap_attempts", "zero_swap_accepts",
                    "zero_swap_length_rejections"):
            metadata[key] = int(getattr(self, "_" + key))
        arrays = {
            "crossing": np.asarray(self._crossing, bool).reshape(count, n),
            "accepted": np.asarray(self._accepted, bool).reshape(count, n),
            "path_frames": np.asarray(self._lengths, int).reshape(count, n),
            "walker_labels": np.asarray(self._label_history, int).reshape(count, n),
            "current_labels": self._labels, "length_rejections": self._length_rejections,
            "swap_attempts": self._swap_attempts, "swap_accepts": self._swap_accepts,
            "minus_frames": np.asarray(self._minus_frames, int),
            "minus_accepted": np.asarray(self._minus_accepted, bool),
            "minus_labels": np.asarray(self._minus_label_history, int),
        }
        bank = ([self._minus_path] if self.minus_enabled else []) + self._paths
        for i, current in enumerate(bank):
            for name in ("q", "p", "values"):
                arrays[f"path_{i}_{name}"] = getattr(current, name)
        atomic_savez(path, kind="PyRPMD.retis.v1",
                     metadata=np.asarray(json.dumps(jsonable(metadata), allow_nan=False, sort_keys=True)),
                     **arrays)

    def load_checkpoint(self, path):
        """Validate integrity, identity and full trajectories before restoration."""
        self._check_configuration()
        if self.identity is None:
            raise ValueError("checkpoint requires explicit potential/coordinate identity")
        with load_checkpoint(path, expected_kind="PyRPMD.retis.v1", require_integrity=True) as archive:
            data = {key: archive[key] for key in archive.files}
        try:
            metadata = json.loads(str(data.pop("metadata").item()))
            if (metadata["configuration"] != self._configuration_at_start
                    or metadata["run_identity"] != self.run_identity):
                raise CheckpointError("RETIS checkpoint scientific identity does not match")
            n, offset = len(self.interfaces), int(self.minus_enabled)
            count = _integer(metadata["samples"], "samples")
            sweeps, burn = (_integer(metadata[k], k) for k in ("sweeps", "burn_in"))
            if sweeps != count + burn:
                raise CheckpointError("inconsistent RETIS checkpoint sweep counts")

            def array(name, shape, kinds, minimum=None, maximum=None):
                value = data[name]
                if (value.shape != shape or value.dtype.kind not in kinds
                        or (minimum is not None and np.any(value < minimum))
                        or (maximum is not None and np.any(value > maximum))):
                    raise CheckpointError(f"invalid RETIS checkpoint {name}")
                return value

            crossing = array("crossing", (count, n), "b")
            accepted = array("accepted", (count, n), "b")
            lengths = array("path_frames", (count, n), "iu", 2, self.max_path_steps + 1)
            labels = array("walker_labels", (count, n), "iu", 0, n + offset - 1)
            current_labels = array("current_labels", (n + offset,), "iu", 0, n + offset - 1)
            rejects = array("length_rejections", (n,), "iu", 0, sweeps)
            attempts = array("swap_attempts", (max(0, n - 1),), "iu", 0, sweeps)
            accepts = array("swap_accepts", attempts.shape, "iu", 0, sweeps)
            if np.any(accepts > attempts):
                raise CheckpointError("invalid RETIS swap acceptance counts")
            mcount = count if offset else 0
            mf = array("minus_frames", (mcount,), "iu", 3, self.max_path_steps + 1)
            ma = array("minus_accepted", (mcount,), "b")
            ml = array("minus_labels", (mcount,), "iu", 0, n + offset - 1)
            joint_labels = np.column_stack((ml, labels)) if offset else labels
            if (not np.array_equal(np.sort(current_labels), np.arange(n + offset))
                    or (count and not np.all(np.sort(joint_labels, axis=1) == np.arange(n + offset)))
                    or (count and not np.array_equal(joint_labels[-1], current_labels))):
                raise CheckpointError("invalid RETIS walker-label permutations")
            reasons, mr = metadata["reasons"], metadata["minus_reasons"]
            if (not isinstance(reasons, list) or len(reasons) != n
                    or any(not isinstance(row, list) or len(row) != count
                           or any(not isinstance(x, str) for x in row) for row in reasons)
                    or not isinstance(mr, list) or len(mr) != mcount
                    or any(not isinstance(x, str) for x in mr)):
                raise CheckpointError("invalid RETIS rejection records")
            counters = {key: _integer(metadata[key], key) for key in (
                "minus_length_rejections", "zero_swap_attempts", "zero_swap_accepts",
                "zero_swap_length_rejections")}
            if (any(value > sweeps for value in counters.values())
                    or counters["zero_swap_accepts"] > counters["zero_swap_attempts"]
                    or counters["zero_swap_length_rejections"] > counters["zero_swap_attempts"] - counters["zero_swap_accepts"]
                    or (not offset and any(counters.values()))):
                raise CheckpointError("invalid RETIS minus move counts")
            expected = {"crossing", "accepted", "path_frames", "walker_labels", "current_labels",
                        "length_rejections", "swap_attempts", "swap_accepts", "minus_frames",
                        "minus_accepted", "minus_labels"}
            expected.update(f"path_{i}_{field}" for i in range(n + offset) for field in ("q", "p", "values"))
            if set(data) != expected:
                raise CheckpointError("invalid RETIS checkpoint payload inventory")
            bank = [TISPath(data[f"path_{i}_q"], data[f"path_{i}_p"],
                            data[f"path_{i}_values"], self.dynamics.dt) for i in range(n + offset)]
            if offset:
                self.validate_minus_path(bank[0])
            for i, current in enumerate(bank[offset:]):
                self.validate_path(current, i)
            if count:
                final_crossing = [bool(p.values[-1] >= self.basin_b if i == n - 1
                                       else np.any(p.values > self.interfaces[i + 1]))
                                  for i, p in enumerate(bank[offset:])]
                if (not np.array_equal(lengths[-1], [len(p) for p in bank[offset:]])
                        or not np.array_equal(crossing[-1], final_crossing)
                        or (offset and mf[-1] != len(bank[0]))):
                    raise CheckpointError("RETIS final paths do not match observations")
            candidate_rng = copy.deepcopy(self.rng)
            candidate_rng.bit_generator.state = metadata["rng_state"]
        except CheckpointError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise CheckpointError("invalid RETIS checkpoint payload") from exc
        self._paths, self._minus_path = bank[offset:], bank[0] if offset else None
        self._crossing, self._accepted, self._lengths = crossing.tolist(), accepted.tolist(), lengths.tolist()
        self._labels, self._label_history = current_labels.copy(), list(labels.copy())
        self._reasons = copy.deepcopy(reasons)
        self._minus_frames, self._minus_accepted = mf.tolist(), ma.tolist()
        self._minus_reasons, self._minus_label_history = list(mr), ml.tolist()
        self._length_rejections = rejects.copy()
        self._swap_attempts, self._swap_accepts = attempts.copy(), accepts.copy()
        for key, value in counters.items():
            setattr(self, "_" + key, value)
        self._sweeps, self._burn_in = sweeps, burn
        self.rng.bit_generator.state = candidate_rng.bit_generator.state
        return self
