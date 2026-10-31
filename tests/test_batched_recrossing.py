"""Scalar workflow and restart interoperability checks for paired batching."""

import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.dynamics import native_capabilities
from pyrpmd.checkpoint import CheckpointError
from pyrpmd.system import System
from pyrpmd.workflows.batched_recrossing import _child_batch_task
from pyrpmd.workflows.rate import _child_task


def make_tasks(directory, *, native=False, pairs=4):
    system = System(
        ("H", "H", "H"), np.array([2., 2., 2.]),
        np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]), ((0,), (1, 2)),
    )
    config = {
        "task": "rate", "beta": 10., "beads": 4, "dt_au": .01, "seed": 7,
        "native": native, "potential": {"provider": "harmonic", "params": {"k": .02}},
        "sampling": {}, "recrossing": {"child_batch_pairs": 4},
        "dynamics": {"block_steps": 7},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 10.},
    }
    tasks = []
    rng = np.random.default_rng(14)
    for pair in range(pairs):
        q = np.repeat(system.geometry[None], 4, axis=0)
        noise = .08*rng.normal(size=q.shape)
        q += noise-noise.mean(axis=0)
        tasks.append({"config": copy.deepcopy(config), "system": system,
                      "chain": pair % 2, "parent": pair//2, "pair": pair,
                      "q": q, "xi_star": 1., "child_steps": 21, "output": str(directory)})
    return tasks


def compare_pair(batch, scalar):
    for key in ("chain", "parent", "pair", "xi_star", "pmf_method", "denominator"):
        assert batch[key] == scalar[key]
    np.testing.assert_array_equal(batch["numerator"], scalar["numerator"])
    for key in ("energy_drift", "endpoint_energy_error_hartree_per_bead", "endpoint_energy_error_kbt"):
        assert batch[key] == pytest.approx(scalar[key], abs=3e-12)


@pytest.mark.parametrize("native", [False] + (["required"] if native_capabilities()["available"] else []))
def test_batched_pairs_match_scalar_rng_signed_flux_and_energy(tmp_path, native):
    tasks = make_tasks(tmp_path/"batch", native=native)
    batch = _child_batch_task(tasks)
    scalar_tasks = [dict(t, output=str(tmp_path/"scalar")) for t in tasks]
    scalar = [_child_task(t) for t in scalar_tasks]
    for batched, single in zip(batch, scalar):
        compare_pair(batched, single)
        assert batched["propagation"]["pairs_in_shared_batch"] == 4
        assert batched["propagation"]["shared_batch_pes_calls"] == 24
        assert batched["propagation"]["native_steps"] == 0
        assert batched["propagation"]["python_steps"] == 42
    # Scalar readers can consume checkpoints written by the batch workflow.
    for task, expected in zip(tasks, batch):
        compare_pair(_child_task(task), expected)


def test_mixed_cached_and_new_pairs_preserve_order_and_checkpoints(tmp_path):
    tasks = make_tasks(tmp_path)
    existing = _child_task(tasks[2])
    old_path = Path(tasks[2]["output"])/"checkpoints"/"child_0000_00001_0002.npz"
    old_bytes = old_path.read_bytes()
    results = _child_batch_task(tasks)
    assert [r["pair"] for r in results] == list(range(4))
    compare_pair(results[2], existing)
    assert old_path.read_bytes() == old_bytes
    assert results[0]["propagation"]["pairs_in_shared_batch"] == 3
    with patch("pyrpmd.batched.BatchedDynamics.propagate_block", side_effect=AssertionError("must resume")):
        cached = _child_batch_task(tasks)
    for actual, expected in zip(cached, results):
        compare_pair(actual, expected)


@pytest.mark.parametrize("batched", [False, True])
def test_completed_children_replay_without_constructing_a_potential(tmp_path, batched):
    tasks = make_tasks(tmp_path)
    expected = _child_batch_task(tasks)
    before = {path: path.read_bytes() for path in (tmp_path / "checkpoints").iterdir()}
    with patch("pyrpmd.parallel.execution.make_task_potential",
               side_effect=AssertionError("completed results must not initialize a PES")):
        replay = _child_batch_task(tasks) if batched else [_child_task(task) for task in tasks]
    for actual, original in zip(replay, expected):
        compare_pair(actual, original)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("failure", ["physics", "integrity"])
def test_invalid_child_checkpoint_is_rejected_before_potential_creation(tmp_path, failure):
    tasks = make_tasks(tmp_path, pairs=1)
    _child_task(tasks[0])
    if failure == "physics":
        tasks[0]["q"] = tasks[0]["q"] + .01
        error, message = ValueError, "checkpoint physics differs"
    else:
        path = next((tmp_path / "checkpoints").iterdir())
        with np.load(path, allow_pickle=False) as saved:
            arrays = dict(saved.items())
        arrays["numerator"][0] += 1.
        np.savez(path, **arrays)
        error, message = CheckpointError, "integrity check failed"
    with patch("pyrpmd.parallel.execution.make_task_potential",
               side_effect=AssertionError("invalid caches must fail before PES initialization")):
        with pytest.raises(error, match=message):
            _child_batch_task(tasks)


def test_reordering_or_different_packing_does_not_change_pair_streams(tmp_path):
    tasks = make_tasks(tmp_path/"original")
    original = _child_batch_task(tasks)
    order = [3, 0, 2, 1]
    reordered = _child_batch_task([dict(tasks[i], output=str(tmp_path/"reordered")) for i in order])
    for index, got in zip(order, reordered):
        compare_pair(got, original[index])
    # One remaining pair in a partial task batch uses the same numerical map.
    single = _child_batch_task([dict(tasks[1], output=str(tmp_path/"one"))])
    compare_pair(single[0], original[1])


def test_batch_size_change_accepts_existing_checkpoint(tmp_path):
    tasks = make_tasks(tmp_path)
    original = _child_batch_task(tasks)
    changed = copy.deepcopy(tasks)
    for task in changed:
        task["config"]["recrossing"]["child_batch_pairs"] = 7
    for got, expected in zip(_child_batch_task(changed), original):
        compare_pair(got, expected)


def test_corrupted_cached_identity_uses_scalar_rejection(tmp_path):
    tasks = make_tasks(tmp_path)
    _child_batch_task(tasks)
    tasks[0]["q"] = tasks[0]["q"]+.01
    with pytest.raises(ValueError, match="checkpoint physics differs"):
        _child_batch_task(tasks)


def test_rejects_mixed_physics_duplicates_and_strict_native_loop(tmp_path):
    tasks = make_tasks(tmp_path)
    changed = copy.deepcopy(tasks)
    changed[1]["config"]["beta"] = 11.
    with pytest.raises(ValueError, match="must share"):
        _child_batch_task(changed)
    with pytest.raises(ValueError, match="duplicate"):
        _child_batch_task([tasks[0], tasks[0]])
    for task in tasks:
        task["config"]["dynamics"]["loop_backend"] = "native"
    with pytest.raises(ValueError, match="Python PES orchestration"):
        _child_batch_task(tasks)
    assert not (tmp_path/"checkpoints").exists()


def test_trpmd_children_are_not_silently_replaced_by_rpmd(tmp_path):
    tasks = make_tasks(tmp_path)
    for task in tasks:
        task["config"]["recrossing"].update(dynamics="trpmd", child_batch_pairs=1)
    with pytest.raises(ValueError, match="RPMD children only"):
        _child_batch_task(tasks)


def test_unsupported_custom_pes_requires_independence_declaration(tmp_path):
    from pyrpmd.potentials import FreePotential

    class Unknown(FreePotential):
        pass

    tasks = make_tasks(tmp_path)
    with patch("pyrpmd.parallel.execution.make_task_potential", return_value=Unknown()):
        with pytest.raises(ValueError, match="trajectory_batch_safe"):
            _child_batch_task(tasks)
