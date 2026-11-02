"""Portable failure, provenance and inherited-lease gates for water scheduling."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("fcntl", reason="the recorded campaign uses POSIX advisory leases")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.water import native_campaign_schedule_v2 as schedule
from benchmarks.water.campaign import sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def registered(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    registry = root / "registration"
    (registry / "configs").mkdir(parents=True)
    sources = {schedule.NATIVE: "fixture native, never executed\n",
               "src/pyrpmd/constants.py": "fixture constants\n"}
    for name, text in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    initial = root / "initial-state.npz"
    initial.write_bytes(b"Checksum fixture; no molecular dynamics are run.")
    historical = root / "historical-source.txt"
    historical.write_text("historical fixture source")
    identity = lambda: {name: sha256(root / name) for name in sources}
    current = identity()
    seeds = list(range(1101, 1109))
    protocol_path = registry / "protocol.json"
    write_json(protocol_path, {"source_sha256": dict(current, **{schedule.NATIVE: sha256(historical)}),
                               "fixed_stage": {"seeds": seeds}})
    patch = registry / "checkpoint-contract-v2.patch"
    patch.write_text("fixture versioned source transition")
    revision = registry / "checkpoint-contract-v2.json"
    write_json(revision, {"from_source_sha256": sha256(historical),
                          "to_source_sha256": current[schedule.NATIVE], "patch_sha256": sha256(patch)})
    configs = {}
    for seed in seeds:
        config = schedule.Config(cells_per_axis=2, cutoff_angstrom=2.5, dt_fs=.1,
                                 equilibration_steps=8, equilibration_record_stride=2,
                                 segment_steps=16, segments=2, record_stride=2,
                                 checkpoint_steps=4, sampling_refresh_fs=.4,
                                 initial_checkpoint=initial.name, seed=seed).validate()
        configs[seed] = config
        write_json(registry / "configs" / f"classical-seed-{seed}.json", asdict(config))
    legacy_path = registry / "native-queue-registration.json"
    write_json(legacy_path, {
        "protocol_sha256": sha256(protocol_path), "checkpoint_revision_sha256": sha256(revision),
        "native_source_sha256": current, "scheduler_source_sha256": sha256(schedule.LEGACY_SCHEDULER),
        "initial_checkpoint_sha256": sha256(initial),
        "config_sha256": {str(seed): sha256(registry / "configs" / f"classical-seed-{seed}.json") for seed in seeds},
    })
    output = tmp_path / "output"
    output.mkdir()
    empty, chunks = schedule.committed_prefix(output / "missing")
    receipt_path = registry / "migration-receipt.json"
    write_json(receipt_path, {"children_normally_exited": True, "dispatchers_retired": True,
                             "old_dispatchers": [{"queue": queue, "native_exit_code": 0,
                                                  "dispatcher_pid": 5000 + index, "native_child_pid": 6000 + index}
                                                 for index, queue in enumerate(("odd", "even"))],
                             "baseline_prefixes": {str(seed): {"committed": empty, "chunks": chunks} for seed in seeds}})
    registration = registry / "four-worker-registration.json"
    write_json(registration, {
        "schema": schedule.SCHEMA, "protocol_path": str(protocol_path.relative_to(root)),
        "protocol_sha256": sha256(protocol_path), "legacy_registration_sha256": sha256(legacy_path),
        "checkpoint_revision_sha256": sha256(revision), "legacy_scheduler_sha256": sha256(schedule.LEGACY_SCHEDULER),
        "scheduler_source_sha256": sha256(schedule.SCHEDULER_FILE), "native_source_sha256": current,
        "max_concurrent_workers": 4, "propagator_protocol_changes": False,
        "queues": {"a": [1101, 1105], "b": [1102, 1106], "c": [1103, 1107], "d": [1104, 1108]},
        "migration_receipt": receipt_path.name, "migration_receipt_sha256": sha256(receipt_path),
    })
    monkeypatch.setattr(schedule, "ROOT", root)
    monkeypatch.setattr(schedule, "source_identity", identity)
    return SimpleNamespace(root=root, registration=registration, protocol=protocol_path, output=output,
                           receipt=receipt_path, configs=configs, identity=identity)


def manifest(fixture, seed, *, step=4, complete=False):
    directory = fixture.output / "runs" / f"classical-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in ("model.xml", "integrator.xml"):
        (directory / name).write_text("Mock engine identity, never deserialized.")
        artifacts[name] = {"sha256": sha256(directory / name)}
    chunk = directory / "chunk-000000.npz"
    chunk.write_bytes(f"seed={seed};step={step}".encode())
    value = {"config": asdict(fixture.configs[seed]), "source_sha256": fixture.identity(),
             "artifacts": artifacts, "committed_steps": step, "complete": complete,
             "chunks": [{"file": chunk.name, "sha256": sha256(chunk), "step": step, "observations": 2}]}
    write_json(directory / "manifest.json", value)
    return directory, value


def seed_from_command(command):
    return int(Path(command[command.index("--output") + 1]).name.split("-")[-1])


def test_inherited_lease_remains_locked_after_parent_descriptor_closes(tmp_path):
    lock = tmp_path / "lease.lock"
    with schedule.exclusive_lease(lock) as stream:
        child = subprocess.Popen([sys.executable, "-c", "import sys; print('ready',flush=True);sys.stdin.readline()"],
                                 pass_fds=(stream.fileno(),), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        assert child.stdout.readline() == "ready\n"
    try:
        with pytest.raises(RuntimeError, match="already held"):
            with schedule.exclusive_lease(lock):
                pass
    finally:
        child.communicate("finish\n", timeout=10)
    assert child.returncode == 0
    with schedule.exclusive_lease(lock):
        pass


@pytest.mark.parametrize("name", ["scheduler-odd.lock", "scheduler-even.lock", "four-worker-control/coordinator.lock",
                                   "four-worker-control/seed-1103.lock"])
def test_existing_lease_prevents_every_native_launch(registered, monkeypatch, name):
    def forbidden(*args, **kwargs):
        pytest.fail("a child was launched while an existing campaign lease was held")
    monkeypatch.setattr(schedule.subprocess, "Popen", forbidden)
    with schedule.exclusive_lease(registered.output / name):
        with pytest.raises(RuntimeError, match="already held"):
            schedule.run_campaign(registered.output, registered.registration, wall_seconds=10)


@pytest.mark.parametrize("workers", [2, 4])
def test_fixed_queues_dispatch_every_seed_once_and_inherit_all_exclusion_fds(registered, monkeypatch, workers):
    calls = []
    class Child:
        def __init__(self, command, **kwargs):
            self.seed = seed_from_command(command)
            self.pid = self.seed + 10000
            assert len(set(kwargs["pass_fds"])) == 4
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENMM_CPU_THREADS", "MKL_NUM_THREADS"):
                assert kwargs["env"][name] == "1"
            calls.append(self.seed)
        def wait(self):
            manifest(registered, self.seed, step=registered.configs[self.seed].total_steps, complete=True)
            return 0
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    schedule.run_campaign(registered.output, registered.registration, wall_seconds=10, workers=workers)
    assert sorted(calls) == list(range(1101, 1109))
    for first, second in ((1101, 1105), (1102, 1106), (1103, 1107), (1104, 1108)):
        assert calls.index(first) < calls.index(second)


def test_any_failure_stops_all_queues_without_replacement_and_is_durable(registered, monkeypatch):
    calls = []
    all_started = threading.Barrier(4)
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = seed_from_command(command)
            self.finished = False
            calls.append(self.pid)
        def wait(self):
            if not self.finished:
                all_started.wait(timeout=10)
                self.finished = True
            return 17
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="campaign stopped after an error"):
        schedule.run_campaign(registered.output, registered.registration, wall_seconds=20)
    assert sorted(calls) == [1101, 1102, 1103, 1104]
    assert (registered.output / "four-worker-control/FAILURE.json").exists()
    with pytest.raises(RuntimeError, match="prior campaign failure"):
        schedule.run_campaign(registered.output, registered.registration, wall_seconds=20)
    assert len(calls) == 4


def test_self_consistent_history_replacement_is_rejected(registered, monkeypatch):
    directory, value = manifest(registered, 1101)
    receipt = schedule.read_json(registered.receipt)
    state, chunks = schedule.committed_prefix(directory)
    receipt["baseline_prefixes"]["1101"] = {"committed": state, "chunks": chunks}
    write_json(registered.receipt, receipt)
    registration = schedule.read_json(registered.registration)
    registration["migration_receipt_sha256"] = sha256(registered.receipt)
    write_json(registered.registration, registration)
    chunk = directory / value["chunks"][0]["file"]
    chunk.write_bytes(b"replaced data and recomputed hash")
    value["chunks"][0]["sha256"] = sha256(chunk)
    write_json(directory / "manifest.json", value)
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("must reject before launch"))
    with pytest.raises(RuntimeError, match="pre-migration committed trajectory prefix"):
        schedule.run_campaign(registered.output, registered.registration, wall_seconds=10)


@pytest.mark.parametrize("change", ["duplicate_seed", "changed_source", "changed_initial", "failed_old_child",
                                    "missing_old_queue", "duplicate_old_queue"])
def test_invalid_registration_or_handoff_prevents_every_launch(registered, monkeypatch, change):
    if change == "duplicate_seed":
        value = schedule.read_json(registered.registration)
        value["queues"]["d"][1] = 1101
        write_json(registered.registration, value)
    elif change == "changed_source":
        (registered.root / schedule.NATIVE).write_text("changed model runner")
    elif change == "changed_initial":
        (registered.root / "initial-state.npz").write_bytes(b"changed initial state")
    else:
        receipt = schedule.read_json(registered.receipt)
        if change == "missing_old_queue":
            receipt["old_dispatchers"] = []
        elif change == "duplicate_old_queue":
            receipt["old_dispatchers"][1]["queue"] = "odd"
        else:
            receipt["old_dispatchers"][0]["native_exit_code"] = 9
        write_json(registered.receipt, receipt)
        registration = schedule.read_json(registered.registration)
        registration["migration_receipt_sha256"] = sha256(registered.receipt)
        write_json(registered.registration, registration)
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("must reject before launch"))
    with pytest.raises(ValueError):
        schedule.run_campaign(registered.output, registered.registration, wall_seconds=10)


def test_operator_stop_marker_keeps_all_trajectories_untouched(registered, monkeypatch):
    control = registered.output / "four-worker-control"
    control.mkdir()
    (control / "STOP_AFTER_CURRENT").write_text("scheduling-only stop request")
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("must honor stop before launch"))
    schedule.run_campaign(registered.output, registered.registration, wall_seconds=10)
    assert not (registered.output / "runs").exists()


def test_post_spawn_journal_error_latches_then_drains_the_child(registered, monkeypatch):
    waited = []
    started = []
    original = schedule.Journal.__call__
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = seed_from_command(command)
            started.append(self.pid)
        def wait(self):
            assert (registered.output / "four-worker-control/FAILURE.json").exists()
            waited.append(self.pid)
            return 0
        def terminate(self):
            pytest.fail("child must finish normally")
        kill = terminate
    def journal(self, event, **values):
        if event == "invocation_started":
            raise OSError("injected post-spawn journal failure")
        return original(self, event, **values)
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    monkeypatch.setattr(schedule.Journal, "__call__", journal)
    with pytest.raises(RuntimeError, match="campaign stopped after an error"):
        schedule.run_campaign(registered.output, registered.registration, wall_seconds=10)
    assert len(started) == 1 and waited == started


def test_validation_failure_is_latched_before_another_queue_can_launch(registered, monkeypatch):
    def fail_validation(identity):
        raise ValueError("injected changed source before dispatch")
    monkeypatch.setattr(schedule, "verify_runtime", fail_validation)
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda *a, **kw: pytest.fail("failure must be latched under launch lock"))
    with pytest.raises(RuntimeError, match="campaign stopped after an error"):
        schedule.run_campaign(registered.output, registered.registration, wall_seconds=10)
    assert (registered.output / "four-worker-control/FAILURE.json").exists()


def test_other_successful_inflight_children_are_drained_after_one_failure(registered, monkeypatch):
    all_started = threading.Barrier(4)
    failure_recorded = threading.Event()
    finished = []
    original = schedule.Journal.__call__
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = seed_from_command(command)
            self.finished = False
        def wait(self):
            if not self.finished:
                all_started.wait(timeout=10)
                if self.pid != 1101:
                    assert failure_recorded.wait(timeout=10)
                    manifest(registered, self.pid, step=registered.configs[self.pid].total_steps, complete=True)
                finished.append(self.pid)
                self.finished = True
            return 17 if self.pid == 1101 else 0
        def terminate(self):
            pytest.fail("a successful in-flight child must finish normally")
        kill = terminate
    def journal(self, event, **values):
        result = original(self, event, **values)
        if event == "campaign_stopped_on_error":
            failure_recorded.set()
        return result
    monkeypatch.setattr(schedule.subprocess, "Popen", Child)
    monkeypatch.setattr(schedule.Journal, "__call__", journal)
    with pytest.raises(RuntimeError, match="campaign stopped after an error"):
        schedule.run_campaign(registered.output, registered.registration, wall_seconds=20)
    assert sorted(finished) == [1101, 1102, 1103, 1104]
