"""Replica exchange of canonical positive transition-interface ensembles.

All paths have the same underlying dynamics/temperature/path-length support;
only the crossing indicator differs. A swap therefore accepts exactly when
both paths belong to their proposed ensembles. The independent flux estimate
is still required; a minus-interface flux ensemble is not implemented here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .transition_interface import TISResult, TransitionInterfaceSampler, _integer


@dataclass
class RETISResult:
    interfaces: tuple[TISResult, ...]
    swap_attempts: np.ndarray
    swap_accepts: np.ndarray
    walker_labels: np.ndarray
    rng_state: dict

    def summary(self):
        return {"method": "replica_exchange_tis_positive_ensembles",
                "interfaces": [result.summary() for result in self.interfaces],
                "swap_attempts": self.swap_attempts.tolist(),
                "swap_accepts": self.swap_accepts.tolist(),
                "crossing_probability_product": float(np.prod([r.crossing.mean() for r in self.interfaces])),
                "requires_independent_flux": True, "converged": False}


class ReplicaExchangeTIS(TransitionInterfaceSampler):
    """Two-way shooting plus neighboring path-exchange sweeps.

    Each sweep first attempts one reversible shooting move in every ensemble,
    then attempts disjoint neighboring swaps at randomly selected parity.
    Observations are taken after swaps, including repeats after rejected moves.
    Across-interface samples are correlated and must be resampled jointly.
    """

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

    def sample_interfaces(self, initial_paths, *, n_samples, burn_in=0, on_sample=None):
        n_samples = _integer(n_samples, "n_samples", 1)
        burn_in = _integer(burn_in, "burn_in")
        paths = list(initial_paths)
        if len(paths) != len(self.interfaces):
            raise ValueError("provide one seed path per interface")
        for i, path in enumerate(paths):
            self.validate_path(path, i)
        if on_sample is not None and not callable(on_sample):
            raise TypeError("on_sample must be callable")
        n = len(paths)
        crossing = np.empty((n_samples, n), bool)
        accepted = np.empty_like(crossing)
        lengths = np.empty((n_samples, n), int)
        labels = np.arange(n)
        label_history = np.empty((n_samples, n), int)
        reasons = [[] for _ in paths]
        length_rejections = np.zeros(n, int)
        swap_attempts, swap_accepts = np.zeros(max(0, n - 1), int), np.zeros(max(0, n - 1), int)
        for sweep in range(burn_in + n_samples):
            moves = []
            for i, path in enumerate(paths):
                paths[i], move = self.shoot(path, i)
                length_rejections[i] += move.reason == "max_path_steps"
                moves.append(move)
            parity = int(self.rng.integers(0, 2)) if n > 2 else 0
            for i in range(parity, n - 1, 2):
                swapped = self.exchange_paths(paths, i, i + 1)
                swap_attempts[i] += 1
                swap_accepts[i] += swapped
                if swapped:
                    labels[[i, i + 1]] = labels[[i + 1, i]]
            if sweep < burn_in:
                continue
            sample = sweep - burn_in
            for i, (path, move) in enumerate(zip(paths, moves)):
                crossing[sample, i] = (path.values[-1] >= self.basin_b if i == n - 1
                                       else np.any(path.values > self.interfaces[i + 1]))
                accepted[sample, i] = move.accepted
                lengths[sample, i] = len(path)
                reasons[i].append(move.reason)
            label_history[sample] = labels
            if on_sample is not None:
                on_sample(sample, tuple(paths), tuple(moves))
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        results = tuple(TISResult(
            i, float(self.interfaces[i]), self.basin_b if i == n - 1 else float(self.interfaces[i + 1]),
            i == n - 1, self.basin_a, self.basin_b, crossing[:, i], accepted[:, i], lengths[:, i],
            tuple(reasons[i]), paths[i], copy.deepcopy(rng_state), burn_in, self.max_path_steps,
            int(length_rejections[i]), self.dynamics.beta, self.dynamics.nbeads,
        ) for i in range(n))
        return RETISResult(results, swap_attempts, swap_accepts, label_history, rng_state)
