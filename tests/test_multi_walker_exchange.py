"""Distributed walker records, shared histories and restart invariants."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pytest
from scipy.special import logsumexp

from pyrpmd.cv import CVStack, Cartesian
from pyrpmd.opes import SharedOPESBias
from pyrpmd.shared_metadynamics import (
    AsynchronousMetadynamicsBias, SharedMetadynamicsBias)
from pyrpmd.walker_exchange import (
    AsyncWalkerStore, ensure_manifest, make_record, read_committed_records)


def cv1():
    return CVStack([Cartesian(0, 0)])


class Gather:
    def __init__(self, values):
        self.values = [np.asarray(value, dtype=np.float64) for value in values]

    def Get_size(self):
        return len(self.values)

    def allgather(self, value):
        assert np.asarray(value).shape == self.values[0].shape
        return [item.copy() for item in self.values]


@pytest.mark.parametrize("walkers", [2, 4])
def test_mpi_metadynamics_batch_is_bitwise_serial_shared_batch(walkers):
    centers = np.linspace(-0.6, 0.9, walkers)[:, None]
    serial = SharedMetadynamicsBias(
        cv1(), [0.35], 0.08, 2.0, walkers=walkers, bias_factor=5.0)
    distributed = SharedMetadynamicsBias(
        cv1(), [0.35], 0.08, 2.0, walkers=walkers, bias_factor=5.0)
    comm = Gather(centers)
    for step in range(1, 6):
        current = centers + 0.03 * step
        serial.deposit_walkers(current, step=step, time=0.1 * step)
        comm.values = [row.copy() for row in current]
        distributed.deposit_mpi(current[0], step=step, time=0.1 * step, comm=comm)
    assert distributed.hills == serial.hills
    points = np.linspace(-1.2, 1.2, 41)[:, None]
    for actual, expected in zip(
            distributed.physical_energy_gradient(points),
            serial.physical_energy_gradient(points)):
        np.testing.assert_array_equal(actual, expected)


def test_truncated_tail_is_ignored_without_advancing_checkpoint_offset(tmp_path):
    path = tmp_path / "walker-0000.jsonl"
    first = make_record(
        "metadynamics", 0, 0, step=2, time=0.2,
        payload={"center": [0.1], "height": 0.2, "bias_before": 0.0})
    path.write_bytes(json.dumps(first, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    committed_size = path.stat().st_size
    path.write_bytes(path.read_bytes() + b'{"event":{"schema":1')
    records, offset = read_committed_records(
        path, 0, method="metadynamics", walker=0)
    assert records == [first]
    assert offset == committed_size
    again, same = read_committed_records(
        path, offset, method="metadynamics", walker=0)
    assert again == []
    assert same == offset


def test_complete_checksum_corruption_fails_explicitly(tmp_path):
    record = make_record(
        "opes", 0, 0, step=1, time=0.1,
        payload={"center": [0.0], "log_weight": -1.0, "bias_before": -0.5})
    record["sha256"] = "0" * 64
    path = tmp_path / "walker-0000.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        read_committed_records(path, method="opes", walker=0)


def test_duplicate_walker_start_is_rejected_and_crash_lock_is_reusable(tmp_path):
    first = AsyncWalkerStore(
        tmp_path, method="metadynamics", walker=0, walkers=2)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            AsyncWalkerStore(
                tmp_path, method="metadynamics", walker=0, walkers=2)
    finally:
        first.close()
    reopened = AsyncWalkerStore(
        tmp_path, method="metadynamics", walker=0, walkers=2)
    reopened.close()


def test_exchange_offsets_resume_without_duplicate_merge(tmp_path):
    with AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=0, walkers=2) as zero, \
         AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=1, walkers=2) as one:
        zero.append({"center": [0.0], "height": 0.1, "bias_before": 0.0},
                    step=1, time=0.1)
        one.append({"center": [1.0], "height": 0.1, "bias_before": 0.0},
                   step=1, time=0.1)
        first = zero.poll()
        assert [(item["event"]["walker"], item["event"]["sequence"])
                for item in first.records] == [(0, 0), (1, 0)]
        state = zero.state_dict()
    with AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=0, walkers=2,
            state=state) as resumed, \
         AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=1, walkers=2) as one:
        assert resumed.poll().records == ()
        one.append({"center": [1.2], "height": 0.08, "bias_before": 0.2},
                   step=2, time=0.2)
        new = resumed.poll().records
        assert len(new) == 1
        assert new[0]["event"]["walker"] == 1
        assert new[0]["event"]["sequence"] == 1


def test_async_metadynamics_merge_order_is_deterministic_and_idempotent():
    records = [
        make_record("metadynamics", 1, 0, step=2, time=0.2,
                    payload={"center": [0.5], "height": 0.07,
                             "bias_before": 0.1}),
        make_record("metadynamics", 0, 0, step=2, time=0.2,
                    payload={"center": [-0.4], "height": 0.08,
                             "bias_before": 0.0}),
        make_record("metadynamics", 0, 1, step=4, time=0.4,
                    payload={"center": [0.1], "height": 0.05,
                             "bias_before": 0.3}),
    ]
    first = AsynchronousMetadynamicsBias(
        cv1(), [0.3], 0.08, 2.0, walkers=2, bias_factor=5.0)
    second = AsynchronousMetadynamicsBias(
        cv1(), [0.3], 0.08, 2.0, walkers=2, bias_factor=5.0)
    assert first.merge_records(records) == 3
    assert first.merge_records(records) == 0
    second.merge_records(reversed(records))
    assert first.hills == second.hills
    np.testing.assert_array_equal(
        first.physical_energy(np.linspace(-1, 1, 101)[:, None]),
        second.physical_energy(np.linspace(-1, 1, 101)[:, None]))


@pytest.mark.parametrize("walkers", [2, 4])
def test_shared_opes_batch_matches_independent_weighted_kde_formula(walkers):
    beta, barrier, gamma, sigma = 2.0, 1.4, 5.0, 0.35
    bias = SharedOPESBias(
        cv1(), [sigma], beta, walkers=walkers,
        barrier=barrier, bias_factor=gamma)
    centers = np.linspace(-0.7, 0.8, walkers)[:, None]
    events = bias.update_walkers(centers, step=3, time=0.3)
    expected_weight = -beta * barrier
    np.testing.assert_array_equal(
        [event["payload"]["log_weight"] for event in events],
        np.full(walkers, expected_weight))
    grid = np.linspace(-1.5, 1.5, 61)
    kernel = np.exp(-0.5 * ((grid[:, None] - centers[:, 0]) / sigma) ** 2) / (
        np.sqrt(2 * np.pi) * sigma)
    probability = kernel.mean(axis=1)
    center_kernel = np.exp(
        -0.5 * ((centers[:, 0, None] - centers[:, 0]) / sigma) ** 2) / (
            np.sqrt(2 * np.pi) * sigma)
    z = center_kernel.mean(axis=1).mean()
    prefactor = (1 - 1 / gamma) / beta
    expected = prefactor * np.log(
        probability / z + np.exp(-barrier / prefactor))
    np.testing.assert_allclose(
        np.exp(bias.log_probability(grid[:, None])), probability,
        rtol=2e-14, atol=2e-16)
    np.testing.assert_allclose(
        bias.physical_energy(grid[:, None]), expected,
        rtol=2e-14, atol=2e-14)
    assert bias.log_z == pytest.approx(np.log(z), abs=5e-15)


def test_shared_opes_state_and_reverse_merge_reproduce_exact_normalization():
    original = SharedOPESBias(
        cv1(), [0.4], 2.0, walkers=2, barrier=1.1, bias_factor=4.0)
    original.update_walkers([[-0.5], [0.6]], step=2, time=0.2)
    original.update_walkers([[0.1], [0.9]], step=4, time=0.4)
    restored = SharedOPESBias(
        cv1(), [0.4], 2.0, walkers=2, barrier=1.1, bias_factor=4.0,
        state=original.state_dict())
    reversed_bias = SharedOPESBias(
        cv1(), [0.4], 2.0, walkers=2, barrier=1.1, bias_factor=4.0)
    reversed_bias.merge_records(reversed(original.records))
    assert restored.state_dict() == original.state_dict()
    assert reversed_bias.state_dict() == original.state_dict()
    grid = np.linspace(-1.2, 1.2, 51)[:, None]
    np.testing.assert_array_equal(
        restored.physical_energy(grid), original.physical_energy(grid))
    np.testing.assert_array_equal(
        reversed_bias.physical_energy(grid), original.physical_energy(grid))


def test_shared_opes_normalization_uses_all_uncompressed_kernels():
    bias = SharedOPESBias(
        cv1(), [0.25], 1.5, walkers=2, barrier=1.0, bias_factor=3.0)
    for step, centers in enumerate((
            [[-0.8], [0.8]], [[-0.2], [0.3]], [[0.0], [1.0]]), 1):
        bias.update_walkers(centers, step=step, time=0.1 * step)
    state = bias.state_dict()
    assert state["compression"] == "none"
    assert len(state["records"]) == 6
    pair_logs, _ = bias._kernel(bias._centers, bias._centers)
    center_sums = logsumexp(pair_logs, axis=0)
    expected = (logsumexp(bias._log_weights + center_sums)
                - logsumexp(bias._log_weights) - np.log(len(bias)))
    assert bias.log_z == pytest.approx(expected, abs=2e-15)


def test_writer_restart_repairs_its_unterminated_tail_before_append(tmp_path):
    with AsyncWalkerStore(
            tmp_path, method="opes", walker=0, walkers=2) as writer:
        writer.append({"center": [0.0], "log_weight": -1.0,
                       "bias_before": -0.5}, step=1, time=0.1)
    path = tmp_path / "walker-0000.jsonl"
    committed = path.stat().st_size
    with path.open("ab") as stream:
        stream.write(b'{"event":{"schema":1')
    assert path.stat().st_size > committed
    with AsyncWalkerStore(
            tmp_path, method="opes", walker=0, walkers=2) as resumed:
        assert path.stat().st_size == committed
        resumed.append({"center": [0.2], "log_weight": -0.8,
                        "bias_before": -0.4}, step=2, time=0.2)
    records, offset = read_committed_records(path, method="opes", walker=0)
    assert [record["event"]["sequence"] for record in records] == [0, 1]
    assert offset == path.stat().st_size



def test_manifest_rejects_stale_unclaimed_logs_and_identity_mismatch(tmp_path):
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "walker-0000.jsonl").write_bytes(b"")
    with pytest.raises(ValueError, match="without a manifest"):
        ensure_manifest(stale, {"method": "opes", "fingerprint": "a"})

    clean = tmp_path / "clean"
    ensure_manifest(clean, {"method": "opes", "fingerprint": "a"})
    with pytest.raises(ValueError, match="different physics"):
        ensure_manifest(clean, {"method": "opes", "fingerprint": "b"})


def test_prepared_record_publication_is_idempotent_across_restart(tmp_path):
    record = make_record(
        "metadynamics", 0, 0, step=2, time=0.2,
        payload={"center": [0.1], "height": 0.2, "bias_before": 0.0})
    with AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=0, walkers=2) as writer:
        writer.ensure_record(record)
        writer.ensure_record(record)
        assert writer.next_sequence == 1
    with AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=0, walkers=2) as resumed:
        resumed.ensure_record(record)
        assert resumed.next_sequence == 1
    records, offset = read_committed_records(
        tmp_path / "walker-0000.jsonl", method="metadynamics", walker=0)
    assert records == [record]
    assert offset == (tmp_path / "walker-0000.jsonl").stat().st_size


def test_record_sequence_gap_rejection_is_transactional_for_both_biases():
    first = make_record(
        "metadynamics", 0, 0, step=1, time=0.1,
        payload={"center": [0.0], "height": 0.1, "bias_before": 0.0})
    gap = make_record(
        "metadynamics", 0, 2, step=3, time=0.3,
        payload={"center": [0.2], "height": 0.08, "bias_before": 0.1})
    meta = AsynchronousMetadynamicsBias(
        cv1(), [0.3], 0.1, 2.0, walkers=2, bias_factor=5.0,
        records=[first])
    before_records, before_hills = meta.records, meta.hills
    with pytest.raises(ValueError, match="sequence has a gap"):
        meta.merge_records([gap])
    assert meta.records == before_records
    assert meta.hills == before_hills

    opes_first = make_record(
        "opes", 0, 0, step=1, time=0.1,
        payload={"center": [0.0], "log_weight": -2.0,
                 "bias_before": -1.0})
    opes_gap = make_record(
        "opes", 0, 2, step=3, time=0.3,
        payload={"center": [0.2], "log_weight": -1.8,
                 "bias_before": -0.9})
    opes = SharedOPESBias(
        cv1(), [0.3], 2.0, walkers=2, barrier=1.0, bias_factor=5.0)
    opes.merge_records([opes_first])
    before = opes.state_dict()
    with pytest.raises(ValueError, match="sequence has a gap"):
        opes.merge_records([opes_gap])
    assert opes.state_dict() == before


@pytest.mark.parametrize("walkers", [2, 4])
def test_mpi_shared_opes_is_bitwise_rank_ordered_batch(walkers):
    centers = np.linspace(-0.5, 0.7, walkers)[:, None]
    serial = SharedOPESBias(
        cv1(), [0.35], 2.0, walkers=walkers,
        barrier=1.2, bias_factor=5.0)
    distributed = SharedOPESBias(
        cv1(), [0.35], 2.0, walkers=walkers,
        barrier=1.2, bias_factor=5.0)
    serial.update_walkers(centers, step=2, time=0.2)
    distributed.update_mpi(
        centers[0], step=2, time=0.2, comm=Gather(centers))
    assert distributed.state_dict() == serial.state_dict()


def _crashing_writer(directory, ready):
    """Subprocess target: leave a locked writer with one partial tail."""
    store = AsyncWalkerStore(
        directory, method="metadynamics", walker=0, walkers=2)
    store.append({"center": [0.0], "height": 0.1, "bias_before": 0.0},
                 step=1, time=0.1)
    with store.log_path(0).open("ab") as stream:
        stream.write(b'{"event":{"schema":1')
        stream.flush()
    ready.set()
    time.sleep(60)


def test_killed_writer_releases_lock_and_other_walker_history_remains_valid(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_crashing_writer, args=(str(tmp_path), ready))
    process.start()
    assert ready.wait(15), "writer subprocess did not reach the injected crash point"
    with AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=1, walkers=2) as survivor:
        survivor.append(
            {"center": [1.0], "height": 0.1, "bias_before": 0.0},
            step=1, time=0.1)
    process.terminate()
    process.join(15)
    assert not process.is_alive()
    with AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=0, walkers=2) as resumed:
        resumed.append(
            {"center": [0.2], "height": 0.08, "bias_before": 0.2},
            step=2, time=0.2)
    with AsyncWalkerStore(
            tmp_path, method="metadynamics", walker=1, walkers=2) as reader:
        records = reader.poll().records
    assert [(record["event"]["walker"], record["event"]["sequence"])
            for record in records] == [(0, 0), (1, 0), (0, 1)]
    committed, offset = read_committed_records(
        tmp_path / "walker-0000.jsonl", method="metadynamics", walker=0)
    assert len(committed) == 2
    assert offset == (tmp_path / "walker-0000.jsonl").stat().st_size


def test_missing_remote_log_with_saved_offset_fails_explicitly(tmp_path):
    path = tmp_path / "missing.jsonl"
    with pytest.raises(ValueError, match="missing log"):
        read_committed_records(path, 7, method="opes", walker=1)
