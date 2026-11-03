"""Durable PES admissions and process ownership for precision execution."""

from copy import deepcopy
import errno
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.parallel import task_potential
from pyrpmd.parallel.execution import _task_potential_transform
from pyrpmd.potentials import PESResult
from pyrpmd.precision_execution import (
    ForceBudgetExceeded, JobMeter, MeteredPotential, execution_lock, read_sealed,
)
from pyrpmd.precision_ledger import PrecisionLedgerError


IDENTITY = {"plan_id": "plan-a", "execution_id": "execution-a", "job_id": "job-a"}


class _AnalyticProvider:
    metadata = {"provider": "analytic-reference", "execution_device": "cpu"}

    def __init__(self, error=None):
        self.error = error
        self.inputs = []
        self.closed = 0
        self.bindings = 0
        self.result = None

    def evaluate(self, q):
        self.inputs.append(q.copy())
        if self.error is not None:
            raise self.error
        self.result = PESResult(
            .5 * np.sum(q * q, axis=(1, 2)), -q,
            messages=("analytic harmonic reference",) * len(q),
            virial=-np.einsum("bni,bnj->bij", q, q),
        )
        return self.result

    def native_binding(self, *args):
        self.bindings += 1
        raise AssertionError("native binding would bypass the admission meter")

    def close(self):
        self.closed += 1


@pytest.fixture
def meter(tmp_path):
    with execution_lock(tmp_path):
        yield JobMeter(tmp_path / "precision" / "job.json", identity=IDENTITY, max_cost=12)


def _reopen(meter, **kwargs):
    return JobMeter(meter.path, **{"identity": IDENTITY, "max_cost": 12, **kwargs})


def test_failed_write_ahead_publication_never_calls_provider(meter, monkeypatch):
    provider = _AnalyticProvider()
    wrapped = MeteredPotential(provider, meter)
    original = meter.path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk publication failed")

    monkeypatch.setattr("pyrpmd.io.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk publication failed"):
        wrapped.evaluate(np.ones((2, 1, 3)))
    assert provider.inputs == []
    assert meter.data["charged_cost"] == meter.data["evaluation_batches"] == 0
    assert meter.path.read_bytes() == original


def test_insufficient_budget_rejects_entire_batch_without_splitting(meter):
    meter.charge(8)
    provider = _AnalyticProvider()
    with pytest.raises(ForceBudgetExceeded, match="batch of 5 frames"):
        MeteredPotential(provider, meter).evaluate(np.ones((5, 1, 3)))
    assert provider.inputs == []
    saved = read_sealed(meter.path)
    assert saved["charged_cost"] == 8
    assert saved["evaluation_batches"] == 1
    assert saved["blocked_request"] == {"frames": 5, "charged_cost": 8, "max_cost": 12}
    with pytest.raises(PrecisionLedgerError, match="budget-blocked"):
        meter.complete({"job_id": "job-a"})


def test_reopen_retains_admissions_and_replayed_computation_cost(meter):
    provider = _AnalyticProvider()
    MeteredPotential(provider, meter).evaluate(np.ones((2, 1, 3)))
    recovered = _reopen(meter)
    assert recovered.data["charged_cost"] == 2
    assert recovered.data["evaluation_batches"] == 1
    # Recomputing an uncheckpointed prefix costs work but creates no receipt.
    MeteredPotential(provider, recovered).evaluate(np.ones((3, 1, 3)))
    assert read_sealed(meter.path)["charged_cost"] == 5
    assert read_sealed(meter.path)["evaluation_batches"] == 2
    assert recovered.data["record"] is None


def test_completion_is_immutable_and_replay_does_not_write_or_charge(meter):
    meter.charge(4)
    record = {"job_id": "job-a", "source_id": "source-a", "values": [1., 2.]}
    expected = deepcopy(record)
    meter.complete(record)
    before = meter.path.read_bytes()
    record["values"].append(3.)
    recovered = _reopen(meter)
    recovered.complete(expected)
    assert recovered.data["record"] == expected
    assert recovered.data["charged_cost"] == 4
    assert meter.path.read_bytes() == before
    with pytest.raises(PrecisionLedgerError, match="Conflicting immutable"):
        recovered.complete({**expected, "source_id": "different-source"})
    with pytest.raises(PrecisionLedgerError, match="terminal precision job"):
        recovered.charge(1)
    assert meter.path.read_bytes() == before


@pytest.mark.parametrize("record", [None, [], "not a record"])
def test_invalid_completion_record_is_rejected_before_publication(meter, record):
    before = meter.path.read_bytes()
    with pytest.raises((ValueError, TypeError), match="record|dictionary|object"):
        meter.complete(record)
    assert meter.path.read_bytes() == before
    assert meter.data["state"] == "running"


@pytest.mark.parametrize("change", ["checksum", "identity", "cap"])
def test_reopen_rejects_tampered_or_conflicting_execution(meter, change):
    meter.charge(2)
    kwargs = {}
    if change == "checksum":
        value = json.loads(meter.path.read_text(encoding="utf-8"))
        value["charged_cost"] = 0
        meter.path.write_text(json.dumps(value), encoding="utf-8")
    elif change == "identity":
        kwargs["identity"] = {**IDENTITY, "execution_id": "other-execution"}
    else:
        kwargs["max_cost"] = 13
    before = meter.path.read_bytes()
    with pytest.raises(PrecisionLedgerError, match="checksum|identity or budget"):
        _reopen(meter, **kwargs)
    assert meter.path.read_bytes() == before


@pytest.mark.parametrize("available", [12.5, 24.0, float("nan"), float("inf"), True, -1])
def test_available_cost_is_validated_before_clamping_to_job_cap(tmp_path, available):
    path = tmp_path / "job.json"
    with pytest.raises(ValueError, match="available.*cost|available.*budget"):
        JobMeter(path, identity=IDENTITY, max_cost=12, available_cost=available)
    assert not path.exists()


def test_available_execution_cap_is_enforced_and_cannot_erase_charges(meter):
    recovered = _reopen(meter, available_cost=3)
    recovered.charge(3)
    with pytest.raises(ForceBudgetExceeded):
        recovered.charge(1)
    with pytest.raises(PrecisionLedgerError, match="exceeds its authorized budget"):
        _reopen(meter, available_cost=2)


def test_meter_preserves_original_pes_result_and_analytic_force_contract(meter):
    q = np.arange(18., dtype=float).reshape(3, 2, 3) / 7
    original = q.copy()
    provider = _AnalyticProvider()
    result = MeteredPotential(provider, meter).evaluate(q)
    assert result is provider.result
    np.testing.assert_array_equal(result.forces, -original)
    np.testing.assert_array_equal(result.energy, .5 * np.sum(original ** 2, axis=(1, 2)))
    np.testing.assert_array_equal(result.virial, -np.einsum("bni,bnj->bij", original, original))
    np.testing.assert_array_equal(q, original)
    assert result.messages == ("analytic harmonic reference",) * 3
    assert read_sealed(meter.path)["charged_cost"] == 3


def test_meter_preserves_provider_required_constraint_validation(meter):
    provider = _AnalyticProvider()
    provider.required_bond_constraints = [(0, 1, 1.)]
    wrapped = MeteredPotential(provider, meter)
    provider.required_bond_constraints.append((1, 2, 2.))
    assert wrapped.required_bond_constraints == [(0, 1, 1.)]
    with pytest.raises(ValueError, match="potential-required bond constraints"):
        Dynamics(wrapped, [1., 1.], beta=1., nbeads=1, dt=.01,
                 native=False, loop_backend="python")
    assert meter.data["charged_cost"] == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync contract")
def test_directory_fsync_failure_prevents_pes_admission(meter, monkeypatch):
    import stat

    provider = _AnalyticProvider()
    original = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        return original(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory fsync failed"):
        MeteredPotential(provider, meter).evaluate(np.ones((2, 1, 3)))
    assert not provider.inputs
    # Publication may already have succeeded: recovery retains the charge
    # conservatively, without claiming that the PES call actually completed.
    assert read_sealed(meter.path)["charged_cost"] == 2


def test_windows_status_reader_sharing_conflict_retries_publication(meter, monkeypatch):
    import pyrpmd.precision_execution as execution

    original = execution.write_json
    calls = []
    delays = []

    def transient_conflict(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            error = PermissionError("status reader still has the old file open")
            error.winerror = 32
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(execution, "write_json", transient_conflict)
    monkeypatch.setattr(execution.time, "sleep", delays.append)
    meter.charge(2)
    assert len(calls) == 2
    assert delays == [.01]
    assert read_sealed(meter.path)["charged_cost"] == 2


@pytest.mark.parametrize("error", [RuntimeError("provider failed"), KeyboardInterrupt()])
def test_provider_failure_keeps_write_ahead_admissions(meter, error):
    provider = _AnalyticProvider(error)
    with pytest.raises(type(error)):
        MeteredPotential(provider, meter).evaluate(np.ones((4, 1, 3)))
    assert len(provider.inputs) == 1
    recovered = _reopen(meter)
    assert recovered.data["charged_cost"] == 4
    assert recovered.data["evaluation_batches"] == 1
    assert recovered.data["state"] == "running"


@pytest.mark.parametrize("entry_point", ["evaluate_fast", "evaluate_slow", "split_force"])
def test_split_force_providers_fail_before_any_admission(meter, entry_point):
    provider = _AnalyticProvider()
    setattr(provider, entry_point, lambda q: provider.evaluate(q))
    with pytest.raises(ValueError, match="split/RPC"):
        MeteredPotential(provider, meter)
    assert meter.data["charged_cost"] == 0
    assert provider.inputs == []


def test_scoped_meter_cannot_forward_native_binding_or_bypass_admission(meter):
    source = _AnalyticProvider()
    with _task_potential_transform(lambda provider: MeteredPotential(provider, meter)):
        with task_potential(source, None, reuse=False) as provider:
            assert not hasattr(provider, "native_binding")
            dynamics = Dynamics(provider, np.ones(1), 5., 2, .001, loop_backend="auto")
            assert dynamics.can_fuse() is False
            state = State(np.ones((2, 1, 3)), np.zeros((2, 1, 3)))
            dynamics.propagate(state, 1, record_centroids=False)
            assert dynamics.loop_stats["native_steps"] == 0
            assert dynamics.loop_stats["python_steps"] == 1
    assert source.bindings == 0 and source.closed == 1
    assert meter.data["charged_cost"] == sum(len(q) for q in source.inputs) > 0


def _lock_probe_then_wait(directory, connection, proceed):
    """Prove contention without assuming when the child receives CPU time."""
    try:
        path = Path(directory) / "precision" / "execution.lock"
        fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                acquire = lambda: msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                release = lambda: msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                acquire = lambda: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                release = lambda: fcntl.flock(fd, fcntl.LOCK_UN)
            try:
                acquire()
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                connection.send("blocked")
            else:
                release()
                raise AssertionError("campaign lock did not exclude a competing process")
        finally:
            os.close(fd)
        if not proceed.wait(30):
            raise AssertionError("owner did not authorize the blocking lock attempt")
        connection.send("attempting")
        with execution_lock(directory):
            connection.send("acquired")
    except BaseException as error:
        connection.send(("error", type(error).__name__, str(error)))
        raise
    finally:
        connection.close()


def _receive(connection):
    assert connection.poll(30), "lock worker did not reach its next handshake"
    return connection.recv()


def test_campaign_execution_lock_serializes_spawned_process(tmp_path):
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    proceed = context.Event()
    worker = context.Process(target=_lock_probe_then_wait, args=(str(tmp_path), child, proceed))
    try:
        with execution_lock(tmp_path):
            worker.start()
            child.close()
            assert _receive(parent) == "blocked"
            proceed.set()
            assert _receive(parent) == "attempting"
        assert _receive(parent) == "acquired"
        worker.join(30)
        assert worker.exitcode == 0
    finally:
        proceed.set()
        if worker.pid is not None:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)
            worker.close()
        parent.close()
        child.close()
