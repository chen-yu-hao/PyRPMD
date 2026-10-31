"""Archive integrity tests use tiny fixtures and never evaluate a PES."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from benchmarks.h3 import package_preparation_study as archive


class FixturePlan:
    def __init__(self, root):
        self.baseline = self.study = root

    def add(self, path, *args, **kwargs):
        assert path.is_file()
        return path

    def verify_record(self, *args, **kwargs):
        pass


def _case(root, *, audit=False, failed=False):
    root.mkdir()
    config = dict(v=0, total_energy_ev=.55, method="ne_rpmd", beads=32,
                  samples=2, id_offset=10, seed=20261031, bmin=0., bmax=3.5,
                  dt_fs=.02, time_cap_fs=250., extended_cap_fs=1000.)
    preparation = dict(kind="direct", conditioning="none", switch_time_fs=0., hold_time_fs=100.,
                       preparation_dt_fs=.02, trace_stride_fs=.2, energy_convention="fixed_collision")
    columns = {"primitive_internal_per_bead_hartree", "kinetic_per_bead_hartree",
               "full_potential_per_bead_hartree", "spring_per_bead_hartree",
               "centroid_harmonic_energy_hartree", "bond_spread_bohr"}
    summary = {"config": config}
    rows = [{"id": 10, "status": "nonreactive"}, {"id": 11, "status": "product"}]
    if audit:
        rows = []
        for identity, value in ((10, .1), (11, .3)):
            windows = {name: {"source": source, "start_fs": start, "end_fs": 100.,
                              "duration_fs": 100.-start, "values": {column: value for column in columns}}
                       for name, source, start in (("hold", "native_every_step_trapezoid", 0.),
                                                   ("hold_trace_check", "dense_trace_trapezoid", 0.),
                                                   ("final_50fs", "dense_trace_trapezoid", 50.))}
            windows["hold"].update(native_dt_fs=.02, native_steps=5000)
            study = {key: preparation[key] for key in ("kind", "conditioning", "energy_convention")}
            study.update(time_averages=windows, before={"primitive_full_ev": value}, after={"primitive_full_ev": value},
                         switch={"status": "completed", "switch_time_fs": 0., "hold_time_fs": 100.,
                                 "dt_fs": .02, "accepted_hold_duration_fs": 100.,
                                 "hold_time_average": dict(windows["hold"]["values"])})
            rows.append({"id": identity, "status": "failed" if failed else "prepared", "preparation_study": study})
        counts = dict(attempts=2, prepared=0 if failed else 2, failed=2 if failed else 0)
        summary.update(preparation=preparation, **counts)
        reductions = {}
        if not failed:
            for name, window in rows[0]["preparation_study"]["time_averages"].items():
                reductions[name] = {key: window[key] for key in ("source", "start_fs", "end_fs", "duration_fs")}
                reductions[name]["values"] = {column: {"n": 2, "mean": .2, "sem": .1} for column in columns}
        means = {**counts, "windows": reductions,
                 "native_minus_trace_hold": {} if failed else {column: {"n": 2, "mean": 0., "sem": 0.} for column in columns}}
        for name, data in (("preparation.json", preparation), ("audit_time_averages.json", means),
                           ("provenance.json", {"study_spec": preparation, "files": [{"path": "snapshot", "sha256": "f"*64}]})):
            (root/name).write_text(json.dumps(data))
    else:
        summary.update(launched_attempts=2, counts={"product": 1, "nonreactive": 1, "unresolved": 0, "failed": 0},
                       provenance={"script_sha256": "f"*64, "git_commit": "historical", "pes": {"name": "BKMP2"}})
    (root/"config.json").write_text(json.dumps(config))
    (root/"summary.json").write_text(json.dumps(summary))
    (root/"trajectories.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return root


def _validate(path, *, audit=False):
    return archive._validate_case(FixturePlan(path.parent), path, v=0, energy=.55, method="ne_rpmd",
                                  beads=32, samples=2, offset=10, audit=audit, baseline=not audit)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999", '{"n": 1, "n": 2}'])
def test_archive_json_rejects_nonfinite_and_duplicate_keys(literal):
    with pytest.raises(archive.ArchiveError, match="Nonfinite|Duplicate"):
        archive._decode_json(literal, "fixture")


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "1e999"])
def test_archive_rejects_nonfinite_jsonl_instead_of_trusting_outcome_counts(tmp_path, literal):
    path = _case(tmp_path/"baseline")
    ledger = path/"trajectories.jsonl"
    rows = ledger.read_text().splitlines()
    rows[0] = rows[0][:-1]+', "diagnostic": '+literal+'}'
    ledger.write_text("\n".join(rows)+"\n")
    with pytest.raises(archive.ArchiveError, match="Nonfinite"):
        _validate(path)


@pytest.mark.parametrize("key,value", [("n", 1), ("mean", 100.), ("sem", None)])
def test_archive_audit_reduction_must_agree_with_ledger(tmp_path, key, value):
    path = _case(tmp_path/"audit", audit=True)
    _validate(path, audit=True)
    output = path/"audit_time_averages.json"
    data = json.loads(output.read_text())
    data["windows"]["hold"]["values"]["bond_spread_bohr"][key] = value
    output.write_text(json.dumps(data))
    with pytest.raises(archive.ArchiveError, match="reduction|diagnostic"):
        _validate(path, audit=True)


def test_archive_all_failed_audit_needs_empty_reductions_and_keeps_counts(tmp_path):
    path = _case(tmp_path/"audit", audit=True, failed=True)
    _validate(path, audit=True)
    output = path/"audit_time_averages.json"
    data = json.loads(output.read_text())
    data["prepared"], data["failed"] = 1, 1
    output.write_text(json.dumps(data))
    with pytest.raises(archive.ArchiveError, match="count mismatch"):
        _validate(path, audit=True)


def test_archive_rejects_a_successful_audit_missing_native_mean(tmp_path):
    path = _case(tmp_path/"audit", audit=True)
    output = path/"trajectories.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    del rows[0]["preparation_study"]["time_averages"]["hold"]["values"]["bond_spread_bohr"]
    output.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(archive.ArchiveError, match="columns mismatch"):
        _validate(path, audit=True)


@pytest.mark.parametrize("key,value", [("native_dt_fs", .3), ("native_steps", 17), ("native_steps", True)])
def test_archive_rejects_native_hold_timestep_metadata_mismatch(tmp_path, key, value):
    path = _case(tmp_path/"audit", audit=True)
    output = path/"trajectories.jsonl"
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    rows[0]["preparation_study"]["time_averages"]["hold"][key] = value
    output.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(archive.ArchiveError, match="timestep/count"):
        _validate(path, audit=True)


def _small_plan(tmp_path):
    data = tmp_path/"relocated"/"state_preparation_study"
    data.mkdir(parents=True)
    payload = data/"fixture.jsonl"
    payload.write_text('{"id": 10, "status": "unresolved"}\n')
    plan = object.__new__(archive.Plan)
    plan.baseline, plan.study = tmp_path/"baseline", data
    plan.code_commit = "1"*40
    plan.code_hashes, plan.members, plan.watched, plan.analysis_inputs = {}, {}, {}, set()
    plan.add(payload, data, "state_preparation_study", analysis_input=True)
    return plan, payload


def test_zip_roundtrip_checksums_and_relocated_paths(tmp_path):
    plan, payload = _small_plan(tmp_path)
    result = archive.write_archive(plan, tmp_path/"result.zip")
    assert result["sha256"] == hashlib.sha256((tmp_path/"result.zip").read_bytes()).hexdigest()
    with zipfile.ZipFile(result["output"]) as contents:
        hashes = dict(line.split("  ", 1)[::-1] for line in contents.read("SHA256SUMS").decode().splitlines())
        assert set(hashes) == set(contents.namelist())-{"SHA256SUMS"}
        assert all(hashlib.sha256(contents.read(name)).hexdigest() == digest for name, digest in hashes.items())
        assert contents.read("data/state_preparation_study/fixture.jsonl") == payload.read_bytes()
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in contents.namelist())
    old = "/another/host/runs/state_preparation_study/fixture.jsonl"
    assert plan.recorded_path(old) == payload.resolve()
    with pytest.raises(archive.ArchiveError, match="already exists"):
        archive.write_archive(plan, tmp_path/"result.zip")


def test_changed_input_never_publishes_an_archive(tmp_path):
    plan, payload = _small_plan(tmp_path)
    payload.write_text("changed after validation\n")
    destination = tmp_path/"result.zip"
    with pytest.raises(archive.ArchiveError, match="changed"):
        archive.write_archive(plan, destination)
    assert not destination.exists()
    assert not Path(str(destination)+".sha256").exists()


def test_partial_publication_is_rolled_back(tmp_path, monkeypatch):
    plan, _ = _small_plan(tmp_path)
    original = archive.os.link
    def link(source, destination):
        if str(destination).endswith(".sha256"):
            raise OSError("synthetic sidecar publication failure")
        return original(source, destination)
    monkeypatch.setattr(archive.os, "link", link)
    destination = tmp_path/"result.zip"
    with pytest.raises(OSError, match="sidecar"):
        archive.write_archive(plan, destination)
    assert not destination.exists()
    assert not Path(str(destination)+".sha256").exists()
