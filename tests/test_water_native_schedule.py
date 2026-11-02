"""Self-contained fixed seed queue guards; no historical simulation data needed."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("fcntl", reason="the recorded cloud campaign uses POSIX advisory locks")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.water import native_campaign_schedule as schedule
from benchmarks.water.campaign import sha256


@pytest.fixture
def registered_queue(tmp_path, monkeypatch):
    """Real hash/registration checks around a small, completely local fixture.

    Native subprocesses are mocked by individual tests, so the initial-state
    file and source/patch records need only be checksum-bearing fixtures.
    The scheduler's prefix and manifest validators are not mocked.
    """
    root = tmp_path / "checkout"
    registry = root / "registration"
    (registry / "configs").mkdir(parents=True)
    native_name = "benchmarks/water/native_reference.py"
    sources = {
        native_name: "Fixture native source; subprocess execution is mocked.\n",
        "src/pyrpmd/constants.py": "Fixture constants identity.\n",
    }
    for name, value in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    initial = root / "initial-checkpoint.npz"
    initial.write_bytes(b"Checksum-bearing initial-state fixture; never loaded by a mocked child.")
    historical = root / "historical-native.txt"
    historical.write_text("Fixture historical native source.\n")

    def fixture_source_identity():
        return {name: sha256(root / name) for name in sources}

    current = fixture_source_identity()
    protocol = registry / "protocol.json"
    protocol.write_text(json.dumps({
        "source_sha256": dict(current, **{native_name: sha256(historical)}),
        "fixed_stage": {"seeds": [1101, 1102, 1103, 1104]},
    }, sort_keys=True))
    patch = registry / "checkpoint-contract-v2.patch"
    patch.write_text("--- a/fixture\n+++ b/fixture\n@@ -1 +1 @@\n-old\n+new\n")
    revision = registry / "checkpoint-contract-v2.json"
    revision.write_text(json.dumps({"from_source_sha256": sha256(historical),
                                   "to_source_sha256": current[native_name],
                                   "patch_sha256": sha256(patch)}, sort_keys=True))
    config_hashes = {}
    for seed in (1101, 1102, 1103, 1104):
        config = schedule.Config(cells_per_axis=2, cutoff_angstrom=2.5, dt_fs=.1,
                                 equilibration_steps=8, equilibration_record_stride=2,
                                 segment_steps=16, segments=2, record_stride=2,
                                 checkpoint_steps=4, sampling_refresh_fs=.4, seed=seed,
                                 initial_checkpoint=initial.name).validate()
        path = registry / "configs" / f"classical-seed-{seed}.json"
        path.write_text(json.dumps(asdict(config), sort_keys=True))
        config_hashes[str(seed)] = sha256(path)
    registration = {
        "protocol_sha256": sha256(protocol), "checkpoint_revision_sha256": sha256(revision),
        "native_source_sha256": current, "scheduler_source_sha256": sha256(Path(schedule.__file__)),
        "initial_checkpoint_sha256": sha256(initial), "config_sha256": config_hashes,
    }
    (registry / "native-queue-registration.json").write_text(json.dumps(registration, sort_keys=True))
    monkeypatch.setattr(schedule, "ROOT", root)
    monkeypatch.setattr(schedule, "source_identity", fixture_source_identity)
    return SimpleNamespace(root=root, protocol=protocol, output=tmp_path / "output",
                           source_identity=fixture_source_identity)


def test_native_child_failure_stops_queue_without_replacing_seed(registered_queue, monkeypatch):
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=17)
    monkeypatch.setattr(schedule.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="failed; queue stopped without replacement"):
        schedule.schedule(registered_queue.output, registered_queue.protocol, "odd", wall_seconds=120)
    assert len(calls) == 1
    assert Path(calls[0][calls[0].index("--config") + 1]).name == "classical-seed-1101.json"
    rows = [json.loads(line) for line in (registered_queue.output / "scheduler-odd.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "queue_stopped_on_error"
    assert rows[-2]["exit_code"] == 17


def test_scheduler_rejects_self_consistent_replacement_of_old_chunk(registered_queue, monkeypatch):
    directory = registered_queue.output / "runs" / "classical-1102"
    directory.mkdir(parents=True)
    config = json.loads((registered_queue.protocol.parent / "configs" / "classical-seed-1102.json").read_text())
    artifacts = {}
    for name in ("model.xml", "integrator.xml"):
        (directory / name).write_text("Mock engine artifact; no molecular dynamics executed in this test.")
        artifacts[name] = {"sha256": sha256(directory / name)}
    filename = "chunk-000000.npz"
    (directory / filename).write_bytes(b"immutable committed history")
    manifest = {"config": config, "source_sha256": registered_queue.source_identity(), "artifacts": artifacts,
                "chunks": [{"file": filename, "sha256": sha256(directory / filename), "step": 4, "observations": 2}],
                "committed_steps": 4, "complete": False}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    def replace_history(command, **kwargs):
        (directory / filename).write_bytes(b"different history with a recomputed checksum")
        manifest["chunks"][0]["sha256"] = sha256(directory / filename)
        path.write_text(json.dumps(manifest))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(schedule.subprocess, "run", replace_history)
    with pytest.raises(RuntimeError, match="committed trajectory prefix was changed"):
        schedule.schedule(registered_queue.output, registered_queue.protocol, "even", wall_seconds=120)


def test_queue_requires_serialized_engine_identity_records(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"artifacts": {}}))
    with pytest.raises(ValueError, match="identity records"):
        schedule.committed_prefix(tmp_path)
