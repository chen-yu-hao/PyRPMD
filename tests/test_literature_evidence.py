"""Synthetic fixtures exercise evidence gates; none are physical benchmark data."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.literature.evidence import audit_bundle, check_convergence, load_campaign


IDENTITY = {"method": "RPMD", "model_sha256": "test-only", "temperature_K": 298.,
            "observable": "self_diffusion"}


def record(control="beads", identity=None):
    return {"schema_version": 1, "control": control, "unit": "angstrom2 ps-1",
            "identity": copy.deepcopy(IDENTITY if identity is None else identity),
            "baseline": {"parameter": 16, "replica_ids": ["a1", "a2", "a3", "a4"],
                         "values": [.220, .221, .222, .221]},
            "refined": {"parameter": 32, "replica_ids": ["b1", "b2", "b3", "b4"],
                        "values": [.221, .222, .223, .222]}}


def check(value, **kwargs):
    return check_convergence(value, control="beads", unit="angstrom2 ps-1", identity=IDENTITY, **kwargs)


def test_equivalence_requires_whole_interval_inside_declared_band():
    result = check(record())
    assert result["passed"]
    assert result["relative_equivalence_upper_bound"] < .05
    assert result["difference_confidence_interval_95"][0] < .001 < result["difference_confidence_interval_95"][1]
    assert not result["replica_independence_independently_verified"]


def test_noisy_equal_means_do_not_establish_equivalence():
    r = record()
    r["baseline"]["values"] = [.01, .2, .3, .374]
    r["refined"]["values"] = [.01, .2, .3, .374]
    assert not check(r)["passed"]


def test_clear_different_mean_fails():
    r = record()
    r["refined"]["values"] = [.30, .31, .32, .33]
    assert not check(r)["passed"]


def test_duplicate_or_zero_variance_is_not_precision():
    r = record()
    r["refined"]["replica_ids"][0] = "a1"
    with pytest.raises(ValueError, match="disjoint"):
        check(r)
    r = record()
    r["baseline"]["values"] = r["refined"]["values"] = [.22] * 4
    assert not check(r)["passed"]


@pytest.mark.parametrize("field,value", [("method", "TRPMD"), ("temperature_K", 300),
                                          ("model_sha256", "another-model")])
def test_same_control_different_physics_is_rejected(field, value):
    r = record()
    r["identity"][field] = value
    with pytest.raises(ValueError, match="identity mismatch"):
        check(r)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0., -.2])
def test_nonfinite_and_nonphysical_replica_values_rejected(value):
    r = record()
    r["refined"]["values"][0] = value
    with pytest.raises(ValueError):
        check(r)


def test_refinement_direction_and_factor_are_checked():
    r = record()
    r["refined"]["parameter"] = 18
    with pytest.raises(ValueError, match="double"):
        check(r)
    r = record("timestep")
    r["baseline"]["parameter"], r["refined"]["parameter"] = .1, .05
    assert check_convergence(r, control="timestep", unit="angstrom2 ps-1", identity=IDENTITY)["passed"]


def write_artifact(folder, name, value):
    content = value if isinstance(value, bytes) else json.dumps(value).encode()
    (folder / name).write_bytes(content)
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}


def bundle(folder, *, complete=True):
    model = write_artifact(folder, "model.txt", b"SYNTHETIC MODEL, NOT A WATER POTENTIAL")
    identity = dict(IDENTITY, model_sha256=model["sha256"])
    result = write_artifact(folder, "result.json", {"identity": identity, "note": "synthetic test"})
    controls = ["beads", "timestep", "sampling", "electrostatic_accuracy", "correlation_length"]
    specs = {}
    for control in controls if complete else controls[:1]:
        r = record(control, identity)
        if control == "timestep":
            r["baseline"]["parameter"], r["refined"]["parameter"] = .1, .05
        specs[control] = write_artifact(folder, control + ".json", r)
    manifest = {"schema_version": 1, "case_id": "qtip4pf_h2o_diffusion", "identity": identity,
                "model": model, "result": result, "convergence": specs}
    path = folder / "bundle.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_artifact_hashes_and_numeric_checks_are_independent_of_scientific_validation(tmp_path):
    path, _ = bundle(tmp_path)
    report = audit_bundle(path)
    assert report["artifact_integrity_passed"]
    assert report["numeric_convergence_checks_passed"]
    assert report["literature_agreement"] is None
    assert report["physical_validated"] is False
    assert report["requires_scientific_review"]


def test_missing_evidence_withholds_numeric_gate(tmp_path):
    path, _ = bundle(tmp_path, complete=False)
    report = audit_bundle(path)
    assert not report["numeric_convergence_checks_passed"]
    assert not report["convergence_checks"]["timestep"]["passed"]


def test_tampering_is_detected(tmp_path):
    path, _ = bundle(tmp_path)
    (tmp_path / "beads.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        audit_bundle(path)


def test_bundle_cannot_reference_outside_files_or_symlinks(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    path, data = bundle(folder)
    (tmp_path / "outside.json").write_text("{}")
    data["result"]["path"] = "../outside.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside"):
        audit_bundle(path)
    (folder / "link.json").symlink_to(tmp_path / "outside.json")
    data["result"]["path"] = "link.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside"):
        audit_bundle(path)


def test_catalog_covers_every_requested_family_without_invented_successes():
    campaign = load_campaign()
    ids = {c["id"] for c in campaign["cases"]}
    assert ids == {"h_ch4", "oh_h2", "f_h2", "h_h2", "d_h2", "mu_h2", "h_ni100", "h_pd",
                   "qtip4pf_h2o_diffusion", "qtip4pf_d2o_diffusion", "qtip4pf_ir"}
    assert all(c["status"] == "not_reproduced" and c["blockers"] for c in campaign["cases"])
    for case in campaign["cases"]:
        if "reference_registry" in case:
            registry = json.loads((ROOT / case["reference_registry"]).read_text())
            known = {x["id"] for x in registry["cases"]}
            assert set(case["reference_ids"]) <= known
