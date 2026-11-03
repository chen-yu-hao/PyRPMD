"""Infinite swapping RETIS with deterministic asynchronous task tickets.

Plain canonical shooting has binary nested ensemble weights.  Its exact
permanent marginals therefore cost O(N**2), without enumerating permutations.
See Roet, Zhang and van Erp, J. Phys. Chem. A 126, 8878 (2022),
doi:10.1021/acs.jpca.2c06004, and Zhang et al., PNAS 121, e2318731121
(2024), doi:10.1073/pnas.2318731121.  High-acceptance subtrajectory moves
and their additional path weights are not implemented here.

Workers execute concurrently and each committed move immediately admits new
work, without a sweep barrier.  Commits follow submission tickets, not elapsed
wall time.  This allows exact checkpoint replay, at the cost of possible
head-of-line waiting for an earlier slow move.  Pending inputs and private RNG
seeds are checkpointed; uncommitted results are safely regenerated on restart.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math

import numpy as np

from .checkpoint import CheckpointError, atomic_savez, load_checkpoint
from .constants import ATOMIC_TIME_S
from .io import fingerprint_payload, jsonable
from .parallel.path_executor import PathTaskExecutor
from .retis import ReplicaExchangeTIS
from .transition_interface import TISMove, TISPath, _integer


def nested_assignment_probabilities(membership):
    """Exact path/ensemble marginals for nested binary positive ensembles.

    Rows are paths; columns must be ordered by increasing interface.  Each row
    is a prefix of ones followed by zeros.  A zero permanent is an error.
    """
    matrix = np.asarray(membership)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.dtype.kind not in "biuf" or not np.isin(matrix, [0, 1]).all()):
        raise ValueError("membership must be a square binary matrix")
    n = len(matrix)
    if not n:
        return np.empty((0, 0), dtype=float)
    counts = matrix.sum(axis=1).astype(int)
    if not np.array_equal(matrix.astype(bool), np.arange(n)[None, :] < counts[:, None]):
        raise ValueError("membership rows must be nested prefixes")
    order = np.argsort(counts, kind="stable")
    counts = counts[order]
    choices = counts - np.arange(n)
    if np.any(choices <= 0):
        raise ValueError("membership has no compatible path assignment")
    sorted_p = np.zeros((n, n), dtype=float)
    sorted_p[0, :counts[0]] = 1.0 / counts[0]
    for i in range(1, n):
        previous = counts[i - 1]
        sorted_p[i, :previous] = sorted_p[i - 1, :previous] * ((previous - i) / choices[i])
        sorted_p[i, previous:counts[i]] = 1.0 / choices[i]
    result = np.empty_like(sorted_p)
    result[order] = sorted_p
    return result


def execute_retis_move(sampler, task):
    """Execute one immutable task using a freshly constructed matching sampler.

    A workflow worker constructs its dynamics/provider locally, then calls this
    helper.  The sampler's shared minus/plus generator is reset to the private
    task seed.  No coordinator RNG or mutable sampler crosses process bounds.
    """
    sampler.rng.bit_generator.state = np.random.default_rng(task["seed"]).bit_generator.state
    kind, paths, ensembles = task["kind"], task["paths"], task["ensembles"]
    if kind == "zero" and ensembles == (-1, 0):
        minus, plus, move = sampler.exchange_zero(*paths)
        return {"paths": (minus, plus), "moves": (move,)}
    if kind == "minus" and ensembles == (-1,):
        sampler._require_minus()
        path, move = sampler._minus.shoot(paths[0])
    elif kind == "shoot" and len(ensembles) == 1 and ensembles[0] >= 0:
        path, move = sampler.shoot(paths[0], ensembles[0])
    else:
        raise ValueError("invalid infinite RETIS worker task")
    return {"paths": (path,), "moves": (move,)}


class InfiniteRETIS:
    """A coordinator for canonical RETIS moves and exact infinite exchange.

    ``worker(task)`` returns ``{'paths': tuple, 'moves': tuple}``.  The payload
    contains ``context``, ``ticket``, ``kind``, ``ensembles``, ``paths`` and
    ``seed``.  Ensemble -1 denotes minus.  A zero exchange returns two retained
    paths and one shared move.  ``on_event(index, coordinator)`` runs after each
    commit and refill (including burn-in), so it can save a replayable checkpoint.
    ``worker_identity`` may identify the scientific part of a runtime context
    that also contains output paths or resume flags; by default all context is
    fingerprinted.  Callers must include every scientific worker option there.
    """

    _FORMAT = "PyRPMD.infinite_retis.v1"

    def __init__(self, sampler, worker, worker_context=None, parallel_config=None,
                 workers=None, seed=0, worker_identity=None):
        if not isinstance(sampler, ReplicaExchangeTIS):
            raise TypeError("sampler must be ReplicaExchangeTIS")
        if not callable(worker):
            raise TypeError("worker must be callable")
        self.sampler, self.worker = sampler, worker
        self.worker_context = copy.deepcopy(worker_context)
        self.worker_identity = copy.deepcopy(worker_context if worker_identity is None else worker_identity)
        self.parallel_config = dict(parallel_config or {})
        executor = PathTaskExecutor(self.parallel_config, workers=workers)
        self.plan = executor.plan
        self.workers = self.plan.trajectory_workers
        self.ensembles = tuple(range(-int(sampler.minus_enabled), len(sampler.interfaces)))
        if self.workers >= len(self.ensembles):
            raise ValueError("infinite RETIS requires fewer workers than path ensembles")
        self.seed = _integer(seed, "seed")
        self.rng = np.random.default_rng(self.seed)
        self._configuration_at_start = self._configuration()
        self.run_identity = fingerprint_payload(self._configuration_at_start)
        self._bank = None

    def _configuration(self):
        return jsonable({
            "format": self._FORMAT, "sampler": self.sampler._configuration(),
            "worker": f"{self.worker.__module__}.{self.worker.__qualname__}",
            "worker_identity": self.worker_identity, "workers": self.workers,
            "execution": asdict(self.plan), "seed": self.seed,
            "exchange": "nested_binary_permanent_v1",
            "scheduling": "submission_ticket_commit_v1", "zero_probability": 0.5,
        }, callables=False)

    def _check_configuration(self):
        self.sampler._check_configuration()
        if self._configuration() != self._configuration_at_start:
            raise ValueError("infinite RETIS configuration changed")

    def initialize(self, initial_paths, *, minus_path=None):
        self._check_configuration()
        paths = tuple(initial_paths)
        if len(paths) != len(self.sampler.interfaces):
            raise ValueError("provide one seed path per positive ensemble")
        for i, path in enumerate(paths):
            self.sampler.validate_path(path, i)
        if self.sampler.minus_enabled:
            self.sampler.validate_minus_path(minus_path)
        elif minus_path is not None:
            raise ValueError("minus_path requires minus=True")
        self._bank = dict(enumerate(paths))
        if self.sampler.minus_enabled:
            self._bank[-1] = minus_path
        self._pending = {}
        self._next_ticket = self._events = self._burn_in = self._burn_in_target = 0
        self._counts, self._crossings, self._durations, self._move_records = [], [], [], []
        self._attempts = np.zeros(len(self.ensembles), dtype=np.int64)
        self._accepts = self._attempts.copy()
        self._length_rejections = self._attempts.copy()
        self._zero_attempts = self._zero_accepts = 0
        return self

    @property
    def events(self):
        return self._events

    @property
    def sample_count(self):
        return len(self._counts)

    @property
    def burn_in_done(self):
        return self._burn_in

    @property
    def pending_intents(self):
        return tuple(copy.deepcopy(task) for _, task in sorted(self._pending.items()))

    @property
    def observations(self):
        shape = (self.sample_count, len(self.ensembles))
        return {name: np.asarray(value, dtype=dtype).reshape(shape).copy()
                for name, value, dtype in (("counts", self._counts, bool),
                                           ("crossings", self._crossings, float),
                                           ("durations", self._durations, float))}

    def _free(self):
        locked = {e for task in self._pending.values() for e in task["ensembles"]}
        return [e for e in self.ensembles if e not in locked]

    def _payload(self, task):
        return {**task, "context": copy.deepcopy(self.worker_context),
                "paths": tuple(self._bank[e] for e in task["ensembles"])}

    def _admit(self, executor, futures):
        while len(self._pending) < self.workers:
            free = self._free()
            if not free:
                break
            chosen = free[int(self.rng.integers(len(free)))]
            pair = (chosen in (-1, 0) and -1 in free and 0 in free
                    and self.rng.random() < 0.5)
            ensembles = (-1, 0) if pair else (chosen,)
            ticket = self._next_ticket
            task = {"ticket": ticket, "ensembles": ensembles,
                    "kind": "zero" if pair else "minus" if chosen == -1 else "shoot",
                    "seed": int(self.rng.integers(0, 2**63, dtype=np.int64))}
            self._pending[ticket] = task
            self._next_ticket += 1
            futures[ticket] = executor.submit(self.worker, self._payload(task))

    def _exchange(self, bank, free, *, observe):
        positive = [e for e in free if e >= 0]
        paths = [bank[e] for e in positive]
        # Common basin/path conditions were validated on admission/commit.
        # Only the maximum progress differs between positive ensembles; scan
        # each trajectory once instead of once per path/ensemble matrix entry.
        maxima = np.asarray([np.max(path.values) for path in paths], dtype=float)
        thresholds = np.asarray([self.sampler.interfaces[e] for e in positive])
        membership = maxima[:, None] > thresholds[None, :]
        probabilities = nested_assignment_probabilities(membership)
        times = np.asarray([(len(path) - 2) * path.dt for path in paths])
        count = np.zeros(len(self.ensembles), dtype=bool)
        crossing = np.zeros(len(self.ensembles), dtype=float)
        duration = np.zeros(len(self.ensembles), dtype=float)
        offset = int(self.sampler.minus_enabled)
        if observe:
            for j, ensemble in enumerate(positive):
                col = ensemble + offset
                count[col] = True
                if ensemble == len(self.sampler.interfaces) - 1:
                    values = [path.values[-1] >= self.sampler.basin_b for path in paths]
                else:
                    target = self.sampler.interfaces[ensemble + 1]
                    values = maxima > target
                crossing[col] = probabilities[:, j] @ np.asarray(values, dtype=float)
                duration[col] = probabilities[:, j] @ times
            if -1 in free:
                count[0] = True
                duration[0] = (len(bank[-1]) - 2) * bank[-1].dt
        # All legal assignments have equal weight.  Assign the most restricted
        # paths first; the number of choices is independent of earlier draws.
        available = list(range(len(positive)))
        for row in np.argsort(membership.sum(axis=1), kind="stable"):
            eligible = [j for j in available if membership[row, j]]
            selected = eligible[int(self.rng.integers(len(eligible)))]
            bank[positive[selected]] = paths[row]
            available.remove(selected)
        return count, crossing, duration

    def _commit(self, result, *, observe):
        ticket = self._events
        task = self._pending[ticket]
        retained, moves = tuple(result["paths"]), tuple(result["moves"])
        if (len(retained) != len(task["ensembles"]) or len(moves) != 1
                or not isinstance(moves[0], TISMove)):
            raise ValueError("invalid infinite RETIS worker result")
        move = moves[0]
        bank = dict(self._bank)
        for ensemble, path in zip(task["ensembles"], retained):
            if (not isinstance(path, TISPath) or path.dt != self.sampler.dynamics.dt
                    or not (self.sampler._minus._member(path, 0) if ensemble == -1
                            else self.sampler._member(path, ensemble))):
                raise ValueError("worker returned a path outside its locked ensemble")
            bank[ensemble] = path
        free = sorted(self._free() + list(task["ensembles"]))
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            count, crossing, duration = self._exchange(bank, free, observe=observe)
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        self._bank = bank
        del self._pending[ticket]
        self._events += 1
        offset = int(self.sampler.minus_enabled)
        for ensemble in task["ensembles"]:
            col = ensemble + offset
            self._attempts[col] += 1
            self._accepts[col] += move.accepted
            self._length_rejections[col] += move.reason == "max_path_steps"
        if task["kind"] == "zero":
            self._zero_attempts += 1
            self._zero_accepts += move.accepted
        self._move_records.append({"ticket": ticket, "kind": task["kind"],
                                   "ensembles": list(task["ensembles"]),
                                   "accepted": bool(move.accepted), "reason": move.reason})
        if observe:
            self._counts.append(count)
            self._crossings.append(crossing)
            self._durations.append(duration)
        else:
            self._burn_in += 1

    def sample(self, n_events, *, burn_in=0, on_event=None):
        """Commit additional production events, with optional initial burn-in."""
        n_events = _integer(n_events, "n_events", 1)
        burn_in = _integer(burn_in, "burn_in")
        if self._bank is None:
            raise ValueError("initialize paths or load a checkpoint first")
        if self._counts and burn_in:
            raise ValueError("burn_in is only allowed before production samples")
        target = self._burn_in + burn_in
        if not burn_in and self._events:
            target = self._burn_in_target
        remaining = max(0, target - self._events)
        return self.advance(remaining + n_events, burn_in_total=target, on_event=on_event)

    def advance(self, n_events, *, burn_in_total=None, on_event=None):
        """Commit a bounded number of moves, including any remaining burn-in.

        ``burn_in_total`` is fixed once the first move commits and is stored in
        the checkpoint.  This entry point supports stops before production.
        Callback indices count all events from zero, including burn-in.
        """
        n_events = _integer(n_events, "n_events", 1)
        self._check_configuration()
        if self._bank is None:
            raise ValueError("initialize paths or load a checkpoint first")
        target = self._burn_in_target if burn_in_total is None else _integer(burn_in_total, "burn_in_total")
        if self._events and target != self._burn_in_target:
            raise ValueError("burn-in target cannot change during a chain")
        self._burn_in_target = target
        if on_event is not None and not callable(on_event):
            raise TypeError("on_event must be callable")
        futures = {}
        with PathTaskExecutor(self.parallel_config, workers=self.workers) as executor:
            for ticket, task in sorted(self._pending.items()):
                futures[ticket] = executor.submit(self.worker, self._payload(task))
            self._admit(executor, futures)
            for _ in range(n_events):
                # Physical completions can arrive in any order; only the next
                # logical ticket may change the path bank or RNG stream.
                result = futures[self._events].result()
                del futures[self._events]
                self._commit(result, observe=self._events >= target)
                self._admit(executor, futures)
                if on_event is not None:
                    on_event(self._events - 1, self)
        return self.summary()

    def _statistics(self, count, crossing, duration):
        totals = count.sum(axis=0)
        if np.any(totals == 0):
            return None
        offset = int(self.sampler.minus_enabled)
        probabilities = crossing.sum(axis=0)[offset:] / totals[offset:]
        product = float(np.prod(probabilities))
        flux = None
        if offset:
            means = duration.sum(axis=0)[:2] / totals[:2]
            cycle = float(means.sum())
            if not math.isfinite(cycle) or cycle <= 0:
                raise FloatingPointError("invalid infinite RETIS residence time")
            flux = 1.0 / cycle
        return probabilities, product, flux, None if flux is None else flux * product

    def summary(self, *, block_size=None):
        self._check_configuration()
        if self._bank is None:
            raise ValueError("initialize paths first")
        n = len(self._counts)
        shape = (n, len(self.ensembles))
        count = np.asarray(self._counts, dtype=bool).reshape(shape)
        crossing = np.asarray(self._crossings, dtype=float).reshape(shape)
        duration = np.asarray(self._durations, dtype=float).reshape(shape)
        point = self._statistics(count, crossing, duration) if n else None
        size = max(1, int(np.sqrt(n))) if block_size is None else _integer(block_size, "block_size", 1)
        blocks = n // size
        estimates = [self._statistics(count[i * size:(i + 1) * size],
                                     crossing[i * size:(i + 1) * size],
                                     duration[i * size:(i + 1) * size])
                     for i in range(blocks)]
        errors = [None, None, None]
        missing = sum(value is None for value in estimates)
        if blocks >= 2 and not missing:
            for j in range(3):
                values = [value[j + 1] for value in estimates]
                if all(value is not None for value in values) and np.ptp(values) > 0:
                    errors[j] = float(np.std(values, ddof=1) / np.sqrt(blocks))
        probabilities, product, flux, rate = point if point else (None, None, None, None)
        return {
            "method": "infinite_retis", "run_identity": self.run_identity,
            "samples": n, "events": self._events, "burn_in": self._burn_in,
            "workers": self.workers, "pending_tickets": sorted(self._pending),
            "ensemble_order": list(self.ensembles), "ensemble_samples": count.sum(axis=0).tolist(),
            "conditional_probabilities": None if probabilities is None else probabilities.tolist(),
            "overall_crossing_probability": product, "crossing_probability_product": product,
            "requires_independent_flux": not self.sampler.minus_enabled,
            "flux_au": flux, "flux_s_inv": None if flux is None else flux / ATOMIC_TIME_S,
            "rate_estimate_au": rate, "rate_estimate_s_inv": None if rate is None else rate / ATOMIC_TIME_S,
            "crossing_probability_standard_error_block": errors[0],
            "flux_standard_error_block_au": errors[1], "rate_standard_error_block_au": errors[2],
            "move_attempts": self._attempts.tolist(), "move_accepts": self._accepts.tolist(),
            "length_rejections": self._length_rejections.tolist(),
            "zero_swap_attempts": self._zero_attempts, "zero_swap_accepts": self._zero_accepts,
            "block_diagnostics": {
                "block_size": size, "blocks": blocks, "blocks_missing_ensembles": missing,
                "samples_excluded_from_block_error": n - blocks * size,
                "block_estimates": [None if value is None else list(value[1:]) for value in estimates],
                "standard_errors": errors,
                "qualification": "Joint event blocks of fractional numerators and free-ensemble counts; check block-size and independent-chain convergence.",
            },
            "converged": False, "physical_converged": False,
            "qualification": "Canonical shooting with exact infinite swaps; deterministic ticket commits can wait for an earlier slow move. No high-acceptance subtrajectory moves.",
        }

    def save_checkpoint(self, path):
        self._check_configuration()
        if self.sampler.identity is None:
            raise ValueError("checkpoint requires explicit potential/coordinate identity")
        if self._bank is None:
            raise ValueError("initialize paths before saving")
        metadata = {
            "format": self._FORMAT, "configuration": self._configuration_at_start,
            "run_identity": self.run_identity, "rng_state": self.rng.bit_generator.state,
            "events": self._events, "burn_in": self._burn_in, "burn_in_target": self._burn_in_target,
            "next_ticket": self._next_ticket,
            "pending": list(self._pending.values()), "moves": self._move_records,
            "zero_attempts": self._zero_attempts, "zero_accepts": self._zero_accepts,
        }
        shape = (len(self._counts), len(self.ensembles))
        arrays = {
            "counts": np.asarray(self._counts, dtype=bool).reshape(shape),
            "crossings": np.asarray(self._crossings, dtype=float).reshape(shape),
            "durations": np.asarray(self._durations, dtype=float).reshape(shape),
            "attempts": self._attempts, "accepts": self._accepts,
            "length_rejections": self._length_rejections,
        }
        for ensemble, current in self._bank.items():
            for name in ("q", "p", "values"):
                arrays[f"path_{ensemble}_{name}"] = getattr(current, name)
        atomic_savez(path, kind=self._FORMAT,
                     metadata=np.asarray(json.dumps(jsonable(metadata), allow_nan=False, sort_keys=True)),
                     **arrays)

    def load_checkpoint(self, path):
        """Validate a complete candidate before replacing any live state."""
        self._check_configuration()
        if self.sampler.identity is None:
            raise ValueError("checkpoint requires explicit potential/coordinate identity")
        with load_checkpoint(path, expected_kind=self._FORMAT, require_integrity=True) as archive:
            data = {key: archive[key] for key in archive.files}
        try:
            metadata = json.loads(str(data.pop("metadata").item()))
            if (metadata["configuration"] != self._configuration_at_start
                    or metadata["run_identity"] != self.run_identity):
                raise CheckpointError("infinite RETIS checkpoint scientific identity does not match")
            events, burn, ticket = (_integer(metadata[k], k) for k in ("events", "burn_in", "next_ticket"))
            burn_target = _integer(metadata["burn_in_target"], "burn_in_target")
            if burn != min(events, burn_target):
                raise ValueError("burn count disagrees with the fixed target")
            n, width = events - burn, len(self.ensembles)
            expected = {"counts", "crossings", "durations", "attempts", "accepts", "length_rejections"}
            expected.update(f"path_{e}_{name}" for e in self.ensembles for name in ("q", "p", "values"))
            if set(data) != expected:
                raise ValueError("invalid payload inventory")
            counts, crossings, durations = (data[name] for name in ("counts", "crossings", "durations"))
            if (counts.shape != (n, width) or counts.dtype.kind != "b"
                    or any(x.shape != counts.shape or x.dtype.kind != "f" or not np.isfinite(x).all()
                           for x in (crossings, durations))
                    or np.any(crossings < -1e-14) or np.any(crossings > counts + 1e-14)
                    or np.any(durations < 0) or np.any(durations[~counts] != 0)
                    or (n and np.any(counts.sum(axis=1) == 0))
                    or (self.sampler.minus_enabled and np.any(crossings[:, 0] != 0))):
                raise ValueError("invalid fractional observations")
            maximum_time = (self.sampler.max_path_steps - 1) * self.sampler.dynamics.dt
            if (np.any(durations > counts * maximum_time * (1 + 1e-13))
                    or (self.sampler.minus_enabled and np.any(
                        durations[counts[:, 0], 0] < self.sampler.dynamics.dt))):
                raise ValueError("residence observations exceed path bounds")
            attempts, accepts, rejects = (data[name] for name in ("attempts", "accepts", "length_rejections"))
            if (any(x.shape != (width,) or x.dtype.kind not in "iu" or np.any(x < 0)
                    or np.any(x > events) for x in (attempts, accepts, rejects))
                    or np.any(accepts + rejects > attempts)):
                raise ValueError("invalid move counters")
            zero_attempts, zero_accepts = (_integer(metadata[k], k) for k in ("zero_attempts", "zero_accepts"))
            if not 0 <= zero_accepts <= zero_attempts <= events:
                raise ValueError("invalid zero exchange counters")
            records = metadata["moves"]
            if (not isinstance(records, list) or len(records) != events
                    or any(not isinstance(record, dict) or record.get("ticket") != i
                           or type(record.get("accepted")) is not bool
                           or not isinstance(record.get("reason"), str)
                           for i, record in enumerate(records))):
                raise ValueError("invalid committed move records")

            def valid_ensembles(kind, ensembles):
                if any(type(e) is not int or e not in self.ensembles for e in ensembles):
                    return False
                return ((kind == "zero" and ensembles == (-1, 0))
                        or (kind == "minus" and ensembles == (-1,))
                        or (kind == "shoot" and len(ensembles) == 1 and ensembles[0] >= 0))

            derived = np.zeros((3, width), dtype=np.int64)
            derived_zero = [0, 0]
            for record in records:
                if (set(record) != {"ticket", "kind", "ensembles", "accepted", "reason"}
                        or not valid_ensembles(record["kind"], tuple(record["ensembles"]))
                        or record["accepted"] != (record["reason"] == "accepted")):
                    raise ValueError("invalid committed move contents")
                for ensemble in record["ensembles"]:
                    column = ensemble + int(self.sampler.minus_enabled)
                    derived[:, column] += (1, record["accepted"], record["reason"] == "max_path_steps")
                if record["kind"] == "zero":
                    derived_zero[0] += 1
                    derived_zero[1] += record["accepted"]
            if (not np.array_equal(derived, [attempts, accepts, rejects])
                    or derived_zero != [zero_attempts, zero_accepts]):
                raise ValueError("move records and counters disagree")
            bank = {}
            for ensemble in self.ensembles:
                current = TISPath(*(data[f"path_{ensemble}_{name}"] for name in ("q", "p", "values")),
                                  self.sampler.dynamics.dt)
                if ensemble == -1:
                    self.sampler.validate_minus_path(current)
                else:
                    self.sampler.validate_path(current, ensemble)
                bank[ensemble] = current
            raw_pending = metadata["pending"]
            if (not isinstance(raw_pending, list) or len(raw_pending) > self.workers
                    or (events and len(raw_pending) != self.workers)):
                raise ValueError("invalid pending task count")
            pending, locked = {}, set()
            for i, task in enumerate(raw_pending, start=events):
                if not isinstance(task, dict) or set(task) != {"ticket", "kind", "ensembles", "seed"}:
                    raise ValueError("invalid pending descriptor")
                if _integer(task["ticket"], "ticket") != i:
                    raise ValueError("pending tickets are not contiguous")
                ensembles = tuple(task["ensembles"])
                kind = task["kind"]
                if (not valid_ensembles(kind, ensembles) or any(e in locked for e in ensembles)):
                    raise ValueError("invalid pending ensemble locks")
                seed = _integer(task["seed"], "seed")
                if seed >= 2**63:
                    raise ValueError("pending seed outside generator range")
                locked.update(ensembles)
                pending[i] = {**task, "ensembles": ensembles}
            if ticket != events + len(pending):
                raise ValueError("next ticket disagrees with pending queue")
            all_tasks = records + list(pending.values())
            for event in range(burn, events):
                occupied = [e for task in all_tasks[event + 1:event + self.workers]
                            for e in task["ensembles"]]
                if (len(set(occupied)) != len(occupied)
                        or not np.array_equal(counts[event - burn], [e not in occupied for e in self.ensembles])):
                    raise ValueError("fractional observations include occupied ensembles")
            candidate_rng = copy.deepcopy(self.rng)
            candidate_rng.bit_generator.state = metadata["rng_state"]
        except CheckpointError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise CheckpointError("invalid infinite RETIS checkpoint payload") from exc
        self._bank, self._pending = bank, pending
        self._events, self._burn_in, self._next_ticket = events, burn, ticket
        self._burn_in_target = burn_target
        self._counts, self._crossings, self._durations = list(counts), list(crossings), list(durations)
        self._attempts, self._accepts, self._length_rejections = attempts.copy(), accepts.copy(), rejects.copy()
        self._zero_attempts, self._zero_accepts = zero_attempts, zero_accepts
        self._move_records = records
        self.rng.bit_generator.state = candidate_rng.bit_generator.state
        return self
