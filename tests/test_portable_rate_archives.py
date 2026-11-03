"""Authenticated legacy Windows rate generations remain readable and resumable."""

import copy
import json
import sys

import pytest

from pyrpmd._archive_paths import archive_path, archive_path_mapping
from pyrpmd.analysis.report_sampling import _checkpoint_chunk_directory
from pyrpmd.analysis.reporting import collect_run
from pyrpmd.io import file_sha256
from pyrpmd.rate_artifacts import read_generation
from pyrpmd.runner import extend_run, run_resolved
from pyrpmd.workflows import rate_extend

from test_rate_extend import LONG, configuration, generation


@pytest.mark.parametrize("name", [
    "umbrella/checkpoints/window_0000.npz",
    r"umbrella\checkpoints\window_0000.npz",
    "umbrella\\checkpoints/window_0000.npz",
])
def test_archive_paths_have_platform_independent_spelling(name):
    assert archive_path(name) == "umbrella/checkpoints/window_0000.npz"


@pytest.mark.parametrize("name", [
    "", None, 7, "/absolute", r"\absolute", r"\\server\share\file",
    "C:/file", r"C:\file", "C:file", "a/../b", r"a\..\b", "a//b",
    "a/./b", "a/", "a:stream", "a\x00b",
])
def test_archive_paths_reject_absolute_traversal_and_ambiguous_names(name):
    with pytest.raises(ValueError, match="[Aa]rchive"):
        archive_path(name)


def test_archive_mapping_rejects_duplicate_aliases_even_for_identical_hashes():
    with pytest.raises(ValueError, match="duplicate"):
        archive_path_mapping({"a/b": "same digest", r"a\b": "same digest"})


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_chunk_lookup_preserves_authenticated_metadata(tmp_path, legacy):
    artifacts = tmp_path / "extensions/rate/production-000000000074"
    checkpoint = artifacts / "umbrella/checkpoints/window_0000_replica_0000.npz"
    relative = "umbrella/checkpoints/" + checkpoint.name
    chunks = relative.removesuffix(".npz") + ".chunks"
    key = relative.replace("/", "\\") if legacy else relative
    value = chunks.replace("/", "\\") if legacy else chunks
    run = {"path": str(tmp_path), "_rate_artifact_directory": str(artifacts),
           "_rate_generation": {"source_checkpoints": {key: "digest"},
                                "sample_stream_sources": {key: value}},
           "_sources": {str(checkpoint.resolve()): {"sha256": "digest"}}}
    before = copy.deepcopy(run)
    assert _checkpoint_chunk_directory(run, checkpoint) == tmp_path / chunks
    assert run == before
    run["_rate_generation"]["sample_stream_sources"][key] = r"..\outside"
    with pytest.raises(ValueError, match="inconsistent generation sources"):
        _checkpoint_chunk_directory(run, checkpoint)


def _legacy_generation(path):
    file = path / "generation.json"
    value = json.loads(file.read_text(encoding="utf-8"))
    value["source_checkpoints"] = {
        key.replace("/", "\\"): digest for key, digest in value["source_checkpoints"].items()}
    value["sample_stream_sources"] = {
        key.replace("/", "\\"): name.replace("/", "\\")
        for key, name in value["sample_stream_sources"].items()}
    file.write_text(json.dumps(value), encoding="utf-8")


def _legacy_completed_archive(directory, artifacts):
    _legacy_generation(artifacts)
    receipt_file = artifacts / "receipt.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    receipt["artifacts"] = {
        key.replace("/", "\\") if key.startswith("umbrella/checkpoints/") else key: digest
        for key, digest in receipt["artifacts"].items()}
    receipt["artifacts"]["generation.json"] = file_sha256(artifacts / "generation.json")
    receipt_file.write_text(json.dumps(receipt), encoding="utf-8")
    digest = file_sha256(receipt_file)
    ledger_file = directory / "results/extensions.json"
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    ledger["requests"][-1]["receipt_sha256"] = digest
    ledger_file.write_text(json.dumps(ledger), encoding="utf-8")
    summary_file = directory / "results/summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["rate_extension"]["receipt_sha256"] = digest
    summary["extensions"] = ledger
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    return summary["rate_extension"]


def test_completed_legacy_generation_reports_and_replays_without_rewriting(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    config, system = configuration()
    directory = tmp_path / "run"
    run_resolved(config, system, output=directory)
    extend_run(directory, production_steps=LONG)
    expected = collect_run(directory)["sampling_diagnostics"]["umbrella"]
    artifacts = generation(directory)
    pointer = _legacy_completed_archive(directory, artifacts)
    before = {p.relative_to(artifacts): p.read_bytes() for p in artifacts.rglob("*") if p.is_file()}
    _, _, receipt = read_generation(directory, pointer)
    assert all("\\" not in key for key in receipt["artifacts"])
    original_receipt = json.loads((artifacts / "receipt.json").read_text(encoding="utf-8"))
    assert any("\\" in key for key in original_receipt["artifacts"])
    actual = collect_run(directory)["sampling_diagnostics"]["umbrella"]
    assert actual == expected
    resumed = extend_run(directory, production_steps=LONG)
    assert resumed["rate_extension"] == pointer
    assert before == {p.relative_to(artifacts): p.read_bytes() for p in artifacts.rglob("*") if p.is_file()}


def test_pending_legacy_generation_resumes_without_changing_bound_identity(tmp_path, monkeypatch):
    config, system = configuration()
    directory = tmp_path / "run"
    run_resolved(config, system, output=directory)

    def interrupt(*args):
        raise RuntimeError("interrupted before receipt publication")

    with monkeypatch.context() as patch:
        patch.setattr(rate_extend, "_finish_extension", interrupt)
        with pytest.raises(RuntimeError, match="before receipt publication"):
            extend_run(directory, production_steps=LONG)
    artifacts = generation(directory)
    assert not (artifacts / "receipt.json").exists()
    _legacy_generation(artifacts)
    bound = (artifacts / "generation.json").read_bytes()
    checkpoints = {p: p.read_bytes() for p in (artifacts / "umbrella/checkpoints").glob("*.npz")}
    result = extend_run(directory, production_steps=LONG)
    assert result["effective_production_steps"] == LONG
    assert (artifacts / "generation.json").read_bytes() == bound
    assert all(p.read_bytes() == value for p, value in checkpoints.items())
    read_generation(directory, result["rate_extension"])
