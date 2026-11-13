"""Exact permutation, concurrency, restart and independent rate references."""

import itertools
import json
from pathlib import Path
import time

import numpy as np
import pytest
from scipy.integrate import quad

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.infinite_retis import (
    InfiniteRETIS, execute_retis_move, nested_assignment_probabilities,
)
from pyrpmd.parallel.path_executor import PathTaskExecutor
from pyrpmd.potentials import PESResult
from pyrpmd.retis import ReplicaExchangeTIS


class QuarticPotential:
    def evaluate(self, q):
        energy = np.sum((q[:, :, 0] ** 2 - 1) ** 2, axis=1)
        energy += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * q[:, :, 0] * (q[:, :, 0] ** 2 - 1)
        return PESResult(energy, force)


def _sampler():
    return ReplicaExchangeTIS(
        Dynamics(QuarticPotential(), [1.0], 1.0, 1, 0.04, native=False),
        Cartesian(0), basin_a=-1.0, basin_b=1.0, interfaces=(-1.0, -0.5),
        max_path_steps=600, seed=17, minus=True,
        identity={"potential": "quartic (x*x-1)**2 with harmonic spectators", "cv": "x"},
    )


def _worker(task):
    return execute_retis_move(_sampler(), task)


def _point(x, momentum):
    q, p = np.zeros((1, 1, 3)), np.zeros((1, 1, 3))
    q[0, 0, 0], p[0, 0, 0] = x, momentum
    return State(q, p)


def _coordinator(workers=1, seed=692):
    sampler = _sampler()
    seed_path = sampler.generate_path(_point(-0.3, 1.4), 1)
    minus = sampler.generate_minus_path(_point(-1.1, -0.7))
    return InfiniteRETIS(sampler, _worker, workers=workers, seed=seed).initialize(
        [seed_path, seed_path], minus_path=minus,
    )


def _enumerated_marginals(matrix):
    n = len(matrix)
    counts, total = np.zeros((n, n)), 0
    for assignment in itertools.permutations(range(n)):
        if all(matrix[i, assignment[i]] for i in range(n)):
            counts[np.arange(n), assignment] += 1
            total += 1
    return counts / total


def test_nested_permanent_marginals_equal_independent_enumeration():
    rng = np.random.default_rng(620)
    for n in range(1, 8):
        for _ in range(7):
            counts = np.array([rng.integers(i + 1, n + 1) for i in range(n)])
            rng.shuffle(counts)
            matrix = np.arange(n)[None, :] < counts[:, None]
            observed = nested_assignment_probabilities(matrix)
            np.testing.assert_allclose(observed, _enumerated_marginals(matrix), atol=3e-15, rtol=0)
            np.testing.assert_allclose(observed.sum(axis=0), 1, atol=3e-15, rtol=0)
            np.testing.assert_allclose(observed.sum(axis=1), 1, atol=3e-15, rtol=0)
            assert not np.any(observed[~matrix])


@pytest.mark.parametrize("matrix", [
    [[1, 0, 0], [1, 0, 0], [1, 1, 1]],
    [[1, 0], [0, 1]], [[1, 0.2], [1, 1]], [[1, 1]], [[np.nan]],
])
def test_invalid_membership_is_explicit(matrix):
    with pytest.raises(ValueError, match="membership"):
        nested_assignment_probabilities(matrix)


def test_locked_ensembles_are_absent_from_fractional_observations():
    coordinator = _coordinator(workers=2)
    coordinator.advance(32, burn_in_total=3)
    observations = coordinator.observations
    assert len(observations["counts"]) == 29
    # At the sampling instant, one other task still owns one or two ensembles.
    assert np.all(observations["counts"].sum(axis=1) < 3)
    assert np.all(observations["crossings"][~observations["counts"]] == 0)
    assert np.all(observations["durations"][~observations["counts"]] == 0)
    assert len(coordinator.pending_intents) == 2


@pytest.mark.parametrize("workers,stop", [(1, 2), (1, 13), (2, 7)])
def test_checkpoint_replays_pending_moves_bitwise_including_burn(tmp_path, workers, stop):
    uninterrupted = _coordinator(workers)
    uninterrupted.advance(35, burn_in_total=5)
    interrupted = _coordinator(workers)
    interrupted.advance(stop, burn_in_total=5)
    checkpoint = tmp_path / "infinite.npz"
    interrupted.save_checkpoint(checkpoint)
    resumed = _coordinator(workers).load_checkpoint(checkpoint)
    resumed.advance(35 - stop)
    assert resumed.summary() == uninterrupted.summary()
    assert resumed.pending_intents == uninterrupted.pending_intents
    for name in uninterrupted.observations:
        np.testing.assert_array_equal(resumed.observations[name], uninterrupted.observations[name])
    for ensemble in uninterrupted.ensembles:
        for field in ("q", "p", "values"):
            np.testing.assert_array_equal(getattr(resumed._bank[ensemble], field),
                                          getattr(uninterrupted._bank[ensemble], field))
    assert resumed.rng.bit_generator.state == uninterrupted.rng.bit_generator.state


def test_burn_callback_checkpoint_and_fixed_target(tmp_path):
    coordinator = _coordinator()
    recorded = []

    def callback(index, state):
        recorded.append((index, state.sample_count, state.burn_in_done))
        state.save_checkpoint(tmp_path / "checkpoint.npz")

    coordinator.advance(2, burn_in_total=7, on_event=callback)
    assert recorded == [(0, 0, 1), (1, 0, 2)]
    resumed = _coordinator().load_checkpoint(tmp_path / "checkpoint.npz")
    assert resumed.events == 2 and resumed.sample_count == 0
    with pytest.raises(ValueError, match="target cannot change"):
        resumed.advance(1, burn_in_total=3)
    resumed.sample(2)
    assert resumed.burn_in_done == 7 and resumed.sample_count == 2


def test_submission_failure_leaves_a_replayable_pending_intent(tmp_path, monkeypatch):
    uninterrupted = _coordinator()
    uninterrupted.advance(8, burn_in_total=2)
    interrupted = _coordinator()
    original = PathTaskExecutor.submit
    calls = 0

    def fail_after_first_commit(executor, function, task):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected submission failure")
        return original(executor, function, task)

    with monkeypatch.context() as scoped:
        scoped.setattr(PathTaskExecutor, "submit", fail_after_first_commit)
        with pytest.raises(RuntimeError, match="injected submission"):
            interrupted.advance(8, burn_in_total=2)
    assert interrupted.events == 1 and len(interrupted.pending_intents) == 1
    interrupted.save_checkpoint(tmp_path / "submission.npz")
    resumed = _coordinator().load_checkpoint(tmp_path / "submission.npz")
    resumed.advance(7)
    assert resumed.summary() == uninterrupted.summary()
    assert resumed.rng.bit_generator.state == uninterrupted.rng.bit_generator.state


def test_checkpoint_rejects_identity_and_overlapping_pending_locks(tmp_path):
    coordinator = _coordinator(workers=2)
    coordinator.advance(4)
    path = tmp_path / "valid.npz"
    coordinator.save_checkpoint(path)
    with pytest.raises(CheckpointError, match="identity"):
        _coordinator(workers=2, seed=693).load_checkpoint(path)
    with load_checkpoint(path, expected_kind=InfiniteRETIS._FORMAT) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata["pending"][1]["ensembles"] = metadata["pending"][0]["ensembles"]
    metadata["pending"][1]["kind"] = metadata["pending"][0]["kind"]
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    atomic_savez(tmp_path / "bad.npz", kind=InfiniteRETIS._FORMAT, **arrays)
    with pytest.raises(CheckpointError, match="invalid infinite"):
        _coordinator(workers=2).load_checkpoint(tmp_path / "bad.npz")


def _gated_worker(task):
    directory = Path(task["directory"])
    if task["gate"]:
        deadline = time.monotonic() + 15
        while not (directory / "release").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("test gate was not released")
            time.sleep(0.01)
    return task["gate"]


def test_executor_allows_later_task_to_finish_before_earlier(tmp_path):
    with PathTaskExecutor(workers=2) as executor:
        first = executor.submit(_gated_worker, {"directory": str(tmp_path), "gate": True})
        second = executor.submit(_gated_worker, {"directory": str(tmp_path), "gate": False})
        try:
            assert second.result(timeout=15) is False
            assert not first.done()
        finally:
            (tmp_path / "release").write_text("release", encoding="utf-8")
        assert first.result(timeout=15) is True


def _failure_and_drain_worker(task):
    directory = Path(task["directory"])
    deadline = time.monotonic() + 15
    if task["fail"]:
        while not (directory / "started").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("second worker did not start")
            time.sleep(0.01)
        (directory / "release").write_text("release", encoding="utf-8")
        raise RuntimeError("injected worker failure")
    (directory / "started").write_text("started", encoding="utf-8")
    while not (directory / "release").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("first worker did not release")
        time.sleep(0.01)
    (directory / "drained").write_text("finished", encoding="utf-8")


def test_executor_error_drains_an_already_running_worker(tmp_path):
    with pytest.raises(RuntimeError, match="injected worker failure"):
        with PathTaskExecutor(workers=2) as executor:
            first = executor.submit(_failure_and_drain_worker, {"directory": str(tmp_path), "fail": True})
            executor.submit(_failure_and_drain_worker, {"directory": str(tmp_path), "fail": False})
            first.result(timeout=15)
    assert (tmp_path / "drained").read_text(encoding="utf-8") == "finished"


@pytest.mark.parametrize("config", [
    {"backend": "mpi"}, {"mode": "gpu"}, {"bead_workers": 2}, {"scheduling": "static"},
])
def test_executor_rejects_unsupported_routing(config):
    with pytest.raises(ValueError):
        PathTaskExecutor(config)


def test_infinite_retis_quartic_rate_matches_independent_phase_space_integral():
    started = time.perf_counter()
    partition, error = quad(lambda x: np.exp(-(x*x-1)**2), -np.inf, np.inf,
                            epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-10
    exact_flux = 2 / (np.sqrt(2*np.pi) * partition)
    exact_rate = exact_flux * np.exp(-1)
    coordinator = _coordinator(seed=491)
    coordinator.sample(6000, burn_in=600)
    summary = coordinator.summary(block_size=300)
    flux_error = summary["flux_standard_error_block_au"]
    rate_error = summary["rate_standard_error_block_au"]
    assert flux_error > 0 and rate_error > 0
    assert abs(summary["flux_au"] - exact_flux) <= 5*flux_error + 0.01*exact_flux
    assert abs(summary["rate_estimate_au"] - exact_rate) <= 5*rate_error + 0.01*exact_rate
    assert not any(summary["length_rejections"])
    print("INFINITE_RETIS_QUARTIC_REFERENCE", json.dumps({
        "exact_flux_au": exact_flux, "exact_rate_au": exact_rate,
        "infinite_retis_flux_au": summary["flux_au"], "flux_block_se_au": flux_error,
        "infinite_retis_rate_au": summary["rate_estimate_au"], "rate_block_se_au": rate_error,
        "production_events": 6000, "burn_in": 600, "workers": 1,
        "dt_au": 0.04, "native": False, "wall_seconds": time.perf_counter() - started,
    }, sort_keys=True))
