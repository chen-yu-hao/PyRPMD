"""Lifecycle-only fixtures: no PES, numerical parent read or physical run."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
    pytest.skip("The historical parent audit uses the pinned Linux verifier", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/validation/literature-convergence-2026-11-01/h3-quantum/p256-parent-stage/archive_parents.py"
SPEC = importlib.util.spec_from_file_location("p256_parent_archive_review", PATH)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


@pytest.fixture
def ready_fixture(tmp_path):
    parent = tmp_path / "parents"; parent.mkdir()
    registration = {"schema": archive.verifier.REGISTRATION_SCHEMA, "driver_sha256": archive.DRIVER_SHA,
        "design_sha256": archive.verifier.DESIGN_SHA256, "fixed_seeds": list(archive.verifier.SEEDS)}
    write(parent / "registration.json", registration)
    manifest = {"schema": "pyrpmd.h3.p256-eight-parent-completion.v1",
        "all_eight_counts_complete_and_verified": True, "eligible_for_separately_authorized_children": True,
        "all_owned_workers_drained": True, "seed_order": list(archive.verifier.SEEDS),
        "launched_seeds": list(archive.verifier.SEEDS), "unstarted_seeds": [], "errors": [],
        "parents": [{"seed": seed} for seed in archive.verifier.SEEDS],
        "registration_sha256": archive.sha(parent / "registration.json"),
        "design_sha256": archive.verifier.DESIGN_SHA256}
    receipt = {"exit_code": 0, "all_workers_drained": True, "exec_session_id": 97158,
               "observed_utc": "synthetic fixture", "registration_sha256": archive.sha(parent / "registration.json")}
    write(tmp_path / "completion.json", {"synthetic_completed_wrapper": True})
    write(tmp_path / "exec-observation.json", {"synthetic_observed_exit": 0})
    receipt["root_completion_sha256"] = archive.sha(tmp_path / "completion.json")
    receipt["actual_tool_observation_sha256"] = archive.sha(tmp_path / "exec-observation.json")
    receipt_path = tmp_path / "root-exec-exit.json"
    def save():
        write(parent / "parents-complete.json", manifest)
        receipt["parents_complete_sha256"] = archive.sha(parent / "parents-complete.json")
        write(receipt_path, receipt)
    save()
    return parent, receipt_path, receipt, manifest, save


def test_ready_metadata_requires_full_root_observation_before_any_parent_read(ready_fixture, monkeypatch):
    parent, receipt, _, manifest, _ = ready_fixture
    assert archive.ready(parent, receipt)[1] == manifest
    receipt.unlink()
    monkeypatch.setattr(archive.verifier, "preflight", lambda: pytest.fail("numerical source/parent inputs read before exit proof"))
    with pytest.raises(ValueError, match="completion observation"):
        archive.audit_complete(parent, receipt)


@pytest.mark.parametrize("filename", ["completion.json", "exec-observation.json"])
def test_observed_exit_also_binds_actual_wrapper_and_tool_receipts(ready_fixture, monkeypatch, filename):
    parent, receipt, _, _, _ = ready_fixture
    (receipt.parent / filename).write_text("changed lifecycle bytes")
    monkeypatch.setattr(archive.verifier, "preflight", lambda: pytest.fail("unbound lifecycle reached parent reads"))
    with pytest.raises(ValueError, match="lifecycle observation"):
        archive.audit_complete(parent, receipt)


@pytest.mark.parametrize("change", ["nonzero_exit", "boolean_exit", "undrained", "session", "missing_global",
                                   "ineligible", "four_chains", "unstarted", "failure", "changed_hash"])
def test_partial_or_unbound_results_block_numerical_inspection(ready_fixture, monkeypatch, change):
    parent, receipt_path, receipt, manifest, save = ready_fixture
    if change == "nonzero_exit": receipt["exit_code"] = 2
    elif change == "boolean_exit": receipt["exit_code"] = False
    elif change == "undrained": receipt["all_workers_drained"] = False
    elif change == "session": receipt["exec_session_id"] = 12
    elif change == "ineligible": manifest["eligible_for_separately_authorized_children"] = False
    elif change == "four_chains": manifest["parents"] = manifest["parents"][:4]
    elif change == "unstarted": manifest["unstarted_seeds"] = [825623]
    elif change == "failure": manifest["errors"] = ["preserved fixture failure"]
    save()
    if change == "missing_global": (parent / "parents-complete.json").unlink()
    elif change == "changed_hash": (parent / "parents-complete.json").write_text("changed")
    monkeypatch.setattr(archive.verifier, "preflight", lambda: pytest.fail("partial result reached numerical reads"))
    with pytest.raises(ValueError): archive.audit_complete(parent, receipt_path)


def valid_events():
    events = []
    for index, seed in enumerate(archive.verifier.SEEDS):
        events.extend([{"event": "worker_started", "seed": seed, "pid": index + 10, "cpu": 0},
                       {"event": "worker_reaped", "seed": seed, "pid": index + 10, "returncode": 0},
                       {"event": "seed_verified", "seed": seed}])
    return events + [{"event": "all_owned_workers_drained"}]


def test_complete_journal_matches_all_fixed_worker_lifecycles():
    _, result = archive.audit_journal(valid_events(), archive.verifier.SEEDS, 2)
    assert result == {"starts": 8, "successful_reaps": 8, "verified_seeds": 8,
                      "maximum_active_workers": 1, "all_drained": True}


@pytest.mark.parametrize("change", ["failed_exit", "wrong_PID", "missing_reap", "duplicate_start", "duplicate_drain", "missing_seed"])
def test_journal_cannot_claim_success_with_unmatched_workers(change):
    events = valid_events()
    if change == "failed_exit": events[1]["returncode"] = 3
    elif change == "wrong_PID": events[1]["pid"] = -1
    elif change == "missing_reap": del events[1]
    elif change == "duplicate_start": events.insert(1, dict(events[0]))
    elif change == "duplicate_drain": events.append({"event": "all_owned_workers_drained"})
    elif change == "missing_seed": del events[-4:-1]
    with pytest.raises(ValueError): archive.audit_journal(events, archive.verifier.SEEDS, 2)
