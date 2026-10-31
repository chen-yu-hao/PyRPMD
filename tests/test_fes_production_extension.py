"""Authorized umbrella extensions preserve samples and commit bounded histories."""

import copy
import ctypes
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.system import System
from pyrpmd.trajectory import CHUNK_KIND, TrajectoryStream
from pyrpmd.workflows import fes


@pytest.fixture(scope="module")
def native_pes(tmp_path_factory):
    if not native_capabilities()["available"] or shutil.which("cc") is None:
        pytest.skip("native extension and C compiler required")
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/examples/harmonic_pes.c").read_text()
    source = source.replace("#include <stdio.h>", """#include <stdio.h>
static int extension_calls=0, extension_fail_at=-1;
void extension_reset(int fail) { extension_calls=0; extension_fail_at=fail; }
""")
    start = source.index("{", source.index("pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1")) + 1
    source = source[:start] + """
 if (++extension_calls == extension_fail_at) {
   message(e,"injected extension PES failure"); return PYRPMD_INTERNAL_ERROR_V1;
 }
""" + source[start:]
    path = tmp_path_factory.mktemp("extension-pes")
    cfile, library = path / "harmonic.c", path / "harmonic.so"
    cfile.write_text(source)
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-std=c11", "-I",
                    str(root / "native/include"), str(cfile), "-o", str(library)],
                   check=True, capture_output=True)
    controls = ctypes.CDLL(str(library))
    controls.extension_reset.argtypes = [ctypes.c_int]
    controls.extension_reset.restype = None
    return library, controls


def _task(output, library=None):
    config = {
        "task": "fes", "beta": 1.7, "beads": 4, "dt_au": .04, "seed": 9131,
        "native": library is not None,
        "potential": ({"provider": "cabi", "library": str(library)} if library else
                      {"provider": "harmonic", "k": 1.0}),
        "sampling": {"equilibration_steps": 7, "production_steps": 22,
                     "sample_interval": 4, "andersen_interval": 13},
        "dynamics": {"loop_backend": "native" if library else "python", "block_steps": 64},
        "output": {"checkpoint_steps": 17},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                for axis in ("x", "y")],
        "umbrella": {"centers": [[.2, -.1]], "stiffness": [[.4, .05], [.05, .7]]},
        "analysis": {"bins": [3, 3], "bootstrap": 0},
    }
    system = System(("H",), np.array([2.3]), np.array([[.2, 0., 0.]]))
    _, tasks = fes._make_tasks(config, system, output)
    return tasks[0]


def _payload(task):
    with load_checkpoint(fes._window_checkpoint(task)) as saved:
        return dict(saved.items())


def _history(task):
    path = fes._window_checkpoint(task)
    with load_checkpoint(path) as saved:
        return fes.read_fes_sample_history(saved, dimension=2,
                                          chunk_directory=path.with_suffix(".chunks"),
                                          include_sources=True)


def _same_result(actual, expected):
    for key in ("samples", "energies", "last_q", "last_p"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("legacy", ["sealed", "unsealed"])
def test_split_extension_is_exact_and_completed_resume_does_not_rewrite(
        tmp_path, request, monkeypatch, backend, legacy):
    library = None
    if backend == "native":
        library, controls = request.getfixturevalue("native_pes")
        controls.extension_reset(-1)
    task = _task(tmp_path / "split", library)
    original = copy.deepcopy(task["config"])
    short = fes._sample_window(task)
    checkpoint = fes._window_checkpoint(task)
    before = _payload(task)
    assert "sample_stream" not in before
    if legacy == "unsealed":
        np.savez_compressed(checkpoint, **before)
    full = _task(tmp_path / "full", library)
    full["production_steps"] = task["production_steps"] = 62
    full["config"]["dynamics"]["block_steps"] = 1
    expected = fes._sample_window(full)
    actual = fes._sample_window(task)
    _same_result(actual, expected)
    np.testing.assert_array_equal(actual["samples"][:len(short["samples"])], short["samples"])
    assert task["config"] == original
    saved, uninterrupted = _payload(task), _payload(full)
    assert str(saved["fingerprint"]) == str(before["fingerprint"]) == str(uninterrupted["fingerprint"])
    for key in ("rng", "step", "time", "elapsed"):
        np.testing.assert_array_equal(saved[key], uninterrupted[key], err_msg=key)
    assert "values" not in saved and "energies" not in saved
    assert _history(task)["samples"] == 15
    records = fes.validate_fes_checkpoints(original, task["system"], task["output"], production_steps=62)
    assert records == [{"window": 0, "replica": 0, "elapsed": 69, "samples": 15,
                        "checkpoint": str(checkpoint.resolve())}]
    committed = checkpoint.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("completed extension propagated or rewrote its checkpoint")

    monkeypatch.setattr(fes, "_atomic_npz", forbidden)
    if backend == "python":
        monkeypatch.setattr(Dynamics, "step", forbidden)
    monkeypatch.setattr(Dynamics, "propagate_block", forbidden)
    _same_result(fes._sample_window(task), expected)
    assert checkpoint.read_bytes() == committed


def test_distinct_analysis_destination_and_reanalysis_read_the_same_prefix(tmp_path):
    task = _task(tmp_path / "shared")
    cfg, system = task["config"], task["system"]
    fes.run_fes(cfg, system, task["output"])
    result = fes.run_fes(cfg, system, tmp_path / "generation", return_data=True,
                         production_steps=62, checkpoint_output=task["output"])
    assert not (tmp_path / "generation/checkpoints").exists()
    assert (tmp_path / "generation/fes.npz").is_file()
    again = fes.reanalyze_fes(cfg, system, task["output"], return_data=True)
    np.testing.assert_array_equal(result["data"]["free_energy"], again["data"]["free_energy"])
    _same_result(result["streams"][0], again["streams"][0])


@pytest.mark.parametrize("failure", ["chunk", "checkpoint"])
@pytest.mark.parametrize("initial_stream", [False, True])
def test_storage_failure_never_commits_partial_samples_and_orphan_chunks_are_ignored(
        tmp_path, monkeypatch, failure, initial_stream):
    task = _task(tmp_path / "split")
    if initial_stream:
        task["production_steps"] = 22
    fes._sample_window(task)
    checkpoint = fes._window_checkpoint(task)
    committed = checkpoint.read_bytes()
    old_files = set(checkpoint.with_suffix(".chunks").glob("*.npz"))
    task["production_steps"] = 62
    expected_task = _task(tmp_path / "full")
    expected_task["production_steps"] = 62
    expected = fes._sample_window(expected_task)

    class SmallChunks(TrajectoryStream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, chunk_frames=2, **kwargs)

    original_flush = TrajectoryStream._flush
    flushes = []
    writes = []

    def failing_flush(writer):
        if writer.buffered_frames:
            flushes.append(writer.buffered_frames)
            if len(flushes) == 2:
                raise OSError("injected chunk failure")
        return original_flush(writer)

    def failing_checkpoint(path, **arrays):
        writes.append(arrays["elapsed"])
        raise OSError("injected checkpoint failure")

    with monkeypatch.context() as patch:
        patch.setattr(fes, "TrajectoryStream", SmallChunks)
        if failure == "chunk":
            patch.setattr(TrajectoryStream, "_flush", failing_flush)
        else:
            patch.setattr(fes, "_atomic_npz", failing_checkpoint)
        with pytest.raises(OSError, match="injected"):
            fes._sample_window(task)
    # The stream case may commit its first small checkpoint before the second
    # flush failure. Its authoritative state must still describe every sample.
    if failure == "checkpoint" or not initial_stream:
        assert checkpoint.read_bytes() == committed
    if failure == "checkpoint":
        assert len(writes) == 1
    assert set(checkpoint.with_suffix(".chunks").glob("*.npz")) - old_files
    resumed = fes._sample_window(task)
    _same_result(resumed, expected)
    assert str(_payload(task)["rng"]) == str(_payload(expected_task)["rng"])


def test_later_extensions_write_only_new_samples_and_small_state_payloads(tmp_path, monkeypatch):
    from pyrpmd import trajectory

    task = _task(tmp_path)
    fes._sample_window(task)
    chunk_sizes, state_payloads = [], []
    original_chunk, original_state = trajectory.atomic_savez, fes._atomic_npz

    def chunk(path, *, kind="arrays", **arrays):
        if kind == CHUNK_KIND:
            chunk_sizes.append(len(arrays["values"]))
        return original_chunk(path, kind=kind, **arrays)

    def state(path, **arrays):
        state_payloads.append(arrays)
        return original_state(path, **arrays)

    monkeypatch.setattr(trajectory, "atomic_savez", chunk)
    monkeypatch.setattr(fes, "_atomic_npz", state)
    for target, appended in ((62, 15), (126, 16), (206, 20)):
        chunk_sizes.clear()
        state_payloads.clear()
        task["production_steps"] = target
        result = fes._sample_window(task)
        assert len(result["samples"]) == target // 4
        assert sum(chunk_sizes) == appended
        assert state_payloads
        assert all("values" not in payload and "energies" not in payload for payload in state_payloads)
        assert all(np.asarray(payload["sample_stream"]).ndim == 0 for payload in state_payloads)
        # Fixed-size coordinates plus one descriptor, regardless of history.
        assert max(sum(np.asarray(value).nbytes for value in payload.values())
                   for payload in state_payloads) < 8192
        if target > 62:
            assert max(chunk_sizes) <= 5  # ceil(checkpoint_steps / sample_interval)


@pytest.mark.parametrize("field", ["elapsed", "time", "rng", "values", "energies"])
def test_preflight_rejects_invalid_legacy_payload_before_provider_or_writes(tmp_path, monkeypatch, field):
    task = _task(tmp_path)
    fes._sample_window(task)
    checkpoint = fes._window_checkpoint(task)
    saved = _payload(task)
    if field == "elapsed":
        saved[field] = np.asarray(28)
    elif field == "time":
        saved[field] = np.asarray(-1.)
    elif field == "rng":
        saved[field] = np.asarray("{}")
    elif field == "values":
        saved[field] = saved[field][:-1]
    else:
        saved[field][0] = np.nan
    atomic_savez(checkpoint, **saved)
    committed = checkpoint.read_bytes()
    monkeypatch.setattr(fes, "task_potential", lambda *a, **k: pytest.fail("preflight loaded PES"))
    with pytest.raises(ValueError):
        fes.validate_fes_checkpoints(task["config"], task["system"], task["output"], production_steps=62)
    assert checkpoint.read_bytes() == committed


def test_preflight_checks_entire_committed_chain_but_ignores_unreferenced_files(tmp_path):
    task = _task(tmp_path)
    task["production_steps"] = 62
    fes._sample_window(task)
    checkpoint = fes._window_checkpoint(task)
    chunkdir = checkpoint.with_suffix(".chunks")
    (chunkdir / ("0" * 64 + ".npz")).write_bytes(b"orphan")
    fes.validate_fes_checkpoints(task["config"], task["system"], task["output"], production_steps=62)
    oldest = Path(_history(task)["source_files"][-1])
    oldest.write_bytes(oldest.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="digest"):
        fes.validate_fes_checkpoints(task["config"], task["system"], task["output"], production_steps=62)


def test_preflight_rejects_changed_physics_budget_and_missing_streams(tmp_path):
    task = _task(tmp_path)
    task["production_steps"] = 62
    fes._sample_window(task)
    with pytest.raises(ValueError, match="authorized production"):
        fes.validate_fes_checkpoints(task["config"], task["system"], task["output"])
    with pytest.raises(ValueError, match="reduce"):
        fes.validate_fes_checkpoints(task["config"], task["system"], task["output"], production_steps=21)
    changed = copy.deepcopy(task["config"])
    changed["beta"] *= 1.1
    with pytest.raises(ValueError, match="physics differs"):
        fes.validate_fes_checkpoints(changed, task["system"], task["output"], production_steps=62)
    fes._window_checkpoint(task).unlink()
    with pytest.raises(ValueError, match="missing umbrella checkpoint"):
        fes.validate_fes_checkpoints(task["config"], task["system"], task["output"], production_steps=62)


@pytest.mark.parametrize("backend", ["python", "native"])
def test_propagation_failure_commits_valid_prefix_and_continues_same_rng(
        tmp_path, request, monkeypatch, backend):
    library = controls = None
    if backend == "native":
        library, controls = request.getfixturevalue("native_pes")
        controls.extension_reset(-1)
    task = _task(tmp_path / "split", library)
    fes._sample_window(task)
    task["production_steps"] = 62
    if controls is not None:
        # Every native step uses two PES calls, including unsampled steps.
        controls.extension_reset(2 * 10)
    original_energy = Dynamics.energy

    def energy(dynamics, state, bias=None):
        if state.step == 39:
            raise RuntimeError("injected extension observation failure")
        return original_energy(dynamics, state, bias)

    with monkeypatch.context() as patch:
        if controls is None:
            patch.setattr(Dynamics, "energy", energy)
        with pytest.raises(RuntimeError, match="injected extension"):
            fes._sample_window(task)
    saved = _payload(task)
    assert int(saved["elapsed"]) == int(saved["step"]) == 38
    assert _history(task)["samples"] == 7
    if controls is not None:
        controls.extension_reset(-1)
    actual = fes._sample_window(task)
    full = _task(tmp_path / "full", library)
    full["production_steps"] = 62
    expected = fes._sample_window(full)
    _same_result(actual, expected)
    assert str(_payload(task)["rng"]) == str(_payload(full)["rng"])


def test_checkpoint_io_counts_serialized_bytes_without_claiming_engine_speed(
        tmp_path, native_pes, monkeypatch):
    from pyrpmd import trajectory

    library, controls = native_pes
    controls.extension_reset(-1)
    original_chunk, original_state = trajectory.atomic_savez, fes._atomic_npz
    measurements, results = {}, {}
    for storage in ("legacy", "stream"):
        task = _task(tmp_path / storage, library)
        task["config"]["sampling"].update(production_steps=20000, sample_interval=1,
                                            andersen_interval=100)
        task["config"]["output"]["checkpoint_steps"] = 1000
        if storage == "stream":
            task["production_steps"] = 20000
        counters = {"state_writes": 0, "chunk_writes": 0, "history_frames_written": 0,
                    "history_payload_bytes": 0, "npz_bytes_written": 0, "largest_state_npz": 0}

        def chunk(path, *, kind="arrays", **arrays):
            result = original_chunk(path, kind=kind, **arrays)
            if kind == CHUNK_KIND:
                counters["chunk_writes"] += 1
                counters["history_frames_written"] += len(arrays["values"])
                counters["history_payload_bytes"] += arrays["values"].nbytes + arrays["energies"].nbytes
                counters["npz_bytes_written"] += Path(path).stat().st_size
            return result

        def state(path, **arrays):
            result = original_state(path, **arrays)
            size = Path(path).stat().st_size
            counters["state_writes"] += 1
            counters["npz_bytes_written"] += size
            counters["largest_state_npz"] = max(counters["largest_state_npz"], size)
            if "values" in arrays:
                counters["history_frames_written"] += len(arrays["values"])
                counters["history_payload_bytes"] += arrays["values"].nbytes + arrays["energies"].nbytes
            return result

        with monkeypatch.context() as patch:
            patch.setattr(trajectory, "atomic_savez", chunk)
            patch.setattr(fes, "_atomic_npz", state)
            results[storage] = fes._sample_window(task)
        measurements[storage] = counters
    _same_result(results["stream"], results["legacy"])
    assert measurements["stream"]["history_frames_written"] == 20000
    assert measurements["legacy"]["history_frames_written"] > 200000
    assert measurements["stream"]["largest_state_npz"] < 8192
    assert measurements["stream"]["npz_bytes_written"] < measurements["legacy"]["npz_bytes_written"]
    print("checkpoint_io=" + json.dumps(measurements, sort_keys=True))
