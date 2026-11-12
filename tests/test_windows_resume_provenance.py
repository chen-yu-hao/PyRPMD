"""Old Windows omissions must fail before state or authenticated files change.

Windows CI uses real drive paths. POSIX runs map the same drive spellings to
temporary files and monkeypatch ``os.path.isabs``; no host-wide platform spoofing
or native Windows dependency is needed.
"""

import copy
import hashlib
import json
import ntpath
import os
from pathlib import Path

import numpy as np
import pytest

from pyrpmd import io, runner
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.input import InputError, JobSpec

from test_checkpoint import collision_job


@pytest.fixture(params=["backslash", "slash"])
def drive_inputs(tmp_path, monkeypatch, request):
    """Use native drive files on Windows and equivalent aliases on POSIX."""
    assets = tmp_path / "assets"
    assets.mkdir()
    plugin = assets / "pes.py"
    model = assets / "weights.json"
    plugin.write_bytes(b"def build(system):\n    raise AssertionError('not loaded')\n")
    model.write_bytes(b'{"stiffness": 0.25}\n')
    if os.name == "nt":
        references = {"plugin": str(plugin), "model": str(model),
                      "directory": str(assets)}
    else:
        references = {"plugin": r"C:\pyrpmd-inputs\pes.py",
                      "model": r"C:\pyrpmd-inputs\weights.json",
                      "directory": r"C:\pyrpmd-inputs"}
    if request.param == "slash":
        references = {key: value.replace("\\", "/")
                      for key, value in references.items()}
    if os.name != "nt":
        aliases = {references["plugin"]: plugin, references["model"]: model,
                   references["directory"]: assets}

        original_resolve = Path.resolve
        original_isabs = os.path.isabs
        monkeypatch.setattr(
            io.Path, "resolve",
            lambda path, *args, **kwargs: original_resolve(
                aliases.get(str(path), path), *args, **kwargs,
            ),
        )
        monkeypatch.setattr(
            io.os.path, "isabs",
            lambda value: original_isabs(value) or ntpath.isabs(value),
        )
    return {**references, "plugin_file": plugin, "model_file": model}


def _pre_f9_explicit_files(config):
    """Independent old explicit-file rules, frozen before f9e85b4.

    The fixtures contain no importable module names or link manifests. The
    historical first-colon split and POSIX-only absolute-path test are the
    relevant behavior, independent of today's plugin/path helpers.
    """
    result = {}

    def add(value):
        path = io.Path(value).resolve()
        if path.is_file():
            result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def visit(value, key=None):
        if isinstance(value, dict):
            for name, item in value.items():
                if name != "directory":
                    visit(item, name)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            if key in ("factory", "module", "function", "initializer"):
                module = value.split(":", 1)[0]
                if module.endswith((".py", ".so")):
                    add(module)
            elif value.startswith("/"):
                add(value)

    visit(config)
    return result


def _config(inputs, kind="both", task="drpmd"):
    config, system = JobSpec.from_mapping(collision_job()).resolve()
    config["task"] = task
    config["potential"] = {"provider": "python"}
    if kind in ("plugin", "both"):
        config["potential"]["factory"] = inputs["plugin"] + ":build"
    if kind in ("model", "weights", "both"):
        config["potential"]["params"] = {
            "weights" if kind == "weights" else "model": inputs["model"],
        }
    return config, system


def _saved_run(tmp_path, inputs, *, schema=2, kind="both", task="drpmd",
               omitted=True, platform="Windows-10-10.0.20348-SP0"):
    config, system = _config(inputs, kind=kind, task=task)
    directory = tmp_path / "run"
    runner._initialize_output(config, system, directory)
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["platform"] = platform
    if omitted:
        manifest["files"] = _pre_f9_explicit_files(config)
    if schema == 1:
        manifest.pop("schema")
        manifest["schema_version"] = 1
        manifest["physics_fingerprint"] = io.fingerprint(config, system)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    atomic_savez(
        directory / "checkpoints" / "saved-state.npz",
        kind="test.windows-provenance.v1", q=np.arange(6).reshape(1, 2, 3),
        rng_state=json.dumps(np.random.default_rng(17).bit_generator.state),
    )
    return directory, config, system


def _bytes(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file() and path.relative_to(directory).as_posix()
            not in {"progress.json", "run.log.jsonl"}}


def _assert_diagnostic(error, paths):
    message = str(error.value)
    for fragment in ("Cannot resume this Windows run", "source/model hashes",
                     "f9e85b4", "original contents", "new, empty output directory"):
        assert fragment in message
    for path in paths:
        assert str(path.resolve()) in message


def test_drive_paths_are_hashed_against_independent_bytes(drive_inputs):
    config, _ = _config(drive_inputs)
    assert _pre_f9_explicit_files(config) == {}
    assert io.referenced_files(config) == {
        str(drive_inputs[name].resolve()):
            hashlib.sha256(drive_inputs[name].read_bytes()).hexdigest()
        for name in ("plugin_file", "model_file")
    }


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("kind", ["plugin", "model"])
def test_public_resume_refuses_omitted_hashes_before_migration_or_dispatch(
    tmp_path, monkeypatch, drive_inputs, schema, kind,
):
    from pyrpmd.workflows import drpmd

    directory, _, _ = _saved_run(tmp_path, drive_inputs, schema=schema, kind=kind)
    before = _bytes(directory)
    monkeypatch.setattr(
        drpmd, "run_drpmd", lambda *a, **k: pytest.fail("workflow dispatched"),
    )
    # The checkpoints-directory alias must reach the same early validation.
    with pytest.raises(RuntimeError, match="Root stage failed: InputError:") as error:
        runner.resume_run(directory / "checkpoints")
    _assert_diagnostic(error, [drive_inputs[kind + "_file"]])
    assert _bytes(directory) == before


def test_complete_windows_provenance_allows_execution_overrides(
    tmp_path, monkeypatch, drive_inputs,
):
    from pyrpmd.workflows import drpmd

    directory, _, _ = _saved_run(tmp_path, drive_inputs, omitted=False)
    before = _bytes(directory)
    calls = []
    monkeypatch.setattr(drpmd, "run_drpmd", lambda *args: calls.append(args))
    assert runner.resume_run(
        directory, execution={"loop_backend": "auto", "block_steps": 7},
    ) is None
    assert len(calls) == 1
    assert calls[0][0]["dynamics"]["block_steps"] == 7
    assert _bytes(directory) == before


@pytest.mark.parametrize("change", ["science", "recorded-content", "recorded-missing"])
def test_existing_mismatch_errors_take_precedence(
    tmp_path, drive_inputs, change,
):
    directory, config, system = _saved_run(tmp_path, drive_inputs)
    if change == "science":
        config = copy.deepcopy(config)
        config["temperature_K"] *= 2
        expected = "changed scientific settings"
    else:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        plugin = drive_inputs["plugin_file"]
        manifest["files"][str(plugin.resolve())] = hashlib.sha256(plugin.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if change == "recorded-content":
            plugin.write_bytes(b"# changed after recording\n")
        else:
            plugin.unlink()
        expected = "Input/model file changed or is missing"
    before = _bytes(directory)
    with pytest.raises(InputError, match=expected) as error:
        runner._initialize_output(config, system, directory, resume=True)
    assert "source/model hashes" not in str(error.value)
    assert _bytes(directory) == before


@pytest.mark.parametrize("task", ["rate", "drpmd"])
@pytest.mark.parametrize("entry", ["extend", "pending-resume"])
def test_extensions_refuse_before_lock_or_ledger_mutation(
    tmp_path, monkeypatch, drive_inputs, task, entry,
):
    from pyrpmd.workflows import extend, rate_extend

    directory, config, _ = _saved_run(tmp_path, drive_inputs, schema=1, task=task)
    budget = ({"production_steps": 8} if task == "rate" else
              {"max_steps": config["collision"]["max_steps"] + 1})
    ledger = {"format": f"PyRPMD.{task}.extensions.v1",
              "requests": [{"status": "running", **budget}]}
    (directory / "results" / "extensions.json").write_text(
        json.dumps(ledger), encoding="utf-8",
    )
    before = _bytes(directory)
    monkeypatch.setattr(extend, "_lock", lambda *a: pytest.fail("extension lock created"))
    monkeypatch.setattr(rate_extend, "_lock", lambda *a: pytest.fail("extension lock created"))
    with pytest.raises(RuntimeError, match="Root stage failed: InputError:") as error:
        if entry == "pending-resume":
            runner.resume_run(directory)
        elif task == "rate":
            runner.extend_run(directory / "checkpoints", production_steps=8)
        else:
            runner.extend_run(directory / "checkpoints", max_time=30.0)
    _assert_diagnostic(error, [drive_inputs["plugin_file"], drive_inputs["model_file"]])
    assert _bytes(directory) == before


def test_drive_directories_alone_do_not_require_source_hashes(tmp_path, drive_inputs):
    config, system = JobSpec.from_mapping(collision_job()).resolve()
    config["output"]["directory"] = drive_inputs["directory"]
    config["_base_directory"] = drive_inputs["directory"]
    directory = tmp_path / "run"
    runner._initialize_output(config, system, directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["platform"] = "Windows-10"
    assert manifest["files"] == {}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _bytes(directory)
    runner._initialize_output(config, system, directory, resume=True)
    assert _bytes(directory) == before


def test_saved_platform_scopes_the_new_diagnostic(tmp_path, drive_inputs):
    # Current-host platform is deliberately irrelevant: a POSIX-origin manifest
    # keeps its historical behavior, even when unusual drive-shaped names occur.
    directory, config, system = _saved_run(tmp_path, drive_inputs, platform="Linux-6.8")
    before = _bytes(directory)
    runner._initialize_output(config, system, directory, resume=True)
    assert _bytes(directory) == before


def test_new_execution_metadata_is_not_missing_historical_source(tmp_path, drive_inputs):
    directory, config, system = _saved_run(
        tmp_path, drive_inputs, kind="plugin", omitted=False,
    )
    before = _bytes(directory)
    # This is a newly supplied, excluded input-location hint. It was never a
    # source of the saved run and must not be diagnosed from current config only.
    config["_input_path"] = drive_inputs["model"]
    runner._initialize_output(config, system, directory, resume=True)
    assert _bytes(directory) == before


@pytest.mark.parametrize("kind", ["plugin", "model", "weights"])
def test_deleted_unrecorded_source_still_refuses_before_dispatch(
    tmp_path, monkeypatch, drive_inputs, kind,
):
    from pyrpmd.workflows import drpmd

    directory, _, _ = _saved_run(tmp_path, drive_inputs, schema=1, kind=kind)
    source = drive_inputs["plugin_file" if kind == "plugin" else "model_file"]
    source.unlink()
    before = _bytes(directory)
    monkeypatch.setattr(
        drpmd, "run_drpmd", lambda *a, **k: pytest.fail("workflow dispatched"),
    )
    with pytest.raises(RuntimeError, match="Root stage failed: InputError:") as error:
        runner.resume_run(directory)
    _assert_diagnostic(error, [source])
    assert _bytes(directory) == before


def test_nonexistent_cache_and_model_directory_are_not_missing_sources(
    tmp_path, drive_inputs,
):
    drive_inputs["model_file"].unlink()
    config, system = JobSpec.from_mapping(collision_job()).resolve()
    config["potential"]["params"] = {
        "cache": drive_inputs["model"], "model": drive_inputs["directory"],
    }
    directory = tmp_path / "run"
    runner._initialize_output(config, system, directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["platform"] = "Windows-10"
    assert manifest["files"] == {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before = _bytes(directory)
    runner._initialize_output(config, system, directory, resume=True)
    assert _bytes(directory) == before


@pytest.mark.parametrize("task", ["rate", "drpmd"])
def test_verified_legacy_windows_extension_preserves_authenticated_bytes(
    tmp_path, monkeypatch, drive_inputs, task,
):
    from pyrpmd.workflows import extend, rate_extend

    directory, config, _ = _saved_run(
        tmp_path, drive_inputs, schema=1, task=task, omitted=False,
    )
    budget = ({"production_steps": 8} if task == "rate" else
              {"max_steps": config["collision"]["max_steps"] + 1})
    ledger = {"format": f"PyRPMD.{task}.extensions.v1",
              "requests": [{"status": "completed", **budget}]}
    (directory / "results" / "extensions.json").write_text(
        json.dumps(ledger), encoding="utf-8",
    )
    before = _bytes(directory)
    calls = []

    def dispatch(path, **kwargs):
        calls.append((path, kwargs))
        return {"continued": True}

    if task == "rate":
        monkeypatch.setattr(rate_extend, "extend_rate", dispatch)
        result = runner.extend_run(directory / "checkpoints", production_steps=8)
        expected = {"production_steps": 8}
    else:
        monkeypatch.setattr(extend, "extend_drpmd", dispatch)
        result = runner.extend_run(directory / "checkpoints", max_time=30.0)
        expected = {"max_time": 30.0}
    assert result == {"continued": True}
    assert calls == [(directory / "checkpoints", expected)]
    assert _bytes(directory) == before
    assert json.loads((directory / "manifest.json").read_text())["schema_version"] == 1
