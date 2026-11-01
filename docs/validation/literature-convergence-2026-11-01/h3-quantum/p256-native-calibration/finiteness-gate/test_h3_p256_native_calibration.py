"""Bounded authorization, resource and state-envelope tests; no PES evaluations."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

if sys.platform != "linux":
    pytest.skip("one-shot calibration has Linux affinity/resource contracts", allow_module_level=True)

from benchmarks.h3 import p256_native_calibration as check


def test_once_claim_retains_failed_attempt(tmp_path):
    output = tmp_path / "once"
    check.claim(output, {"attempt": 1})
    original = (output / "RUN_ONCE.json").read_bytes()
    with pytest.raises(FileExistsError):
        check.claim(output, {"attempt": 2})
    assert (output / "RUN_ONCE.json").read_bytes() == original


def test_content_guard_rejects_same_length_change(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"abc")
    guards = {str(path): check.sha(path)}
    check.check_guards(guards)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="identity changed"):
        check.check_guards(guards)


def test_checkpoint_preserves_every_state_and_observation_bit(tmp_path):
    arrays = {"q": np.arange(54, dtype=float).reshape(2, 3, 3, 3),
              "p": -np.arange(54, dtype=float).reshape(2, 3, 3, 3),
              "step": np.asarray([0, 20], dtype=np.int64), "time": np.asarray([0.0, 82.682746]),
              "surface": np.asarray([-0.0, 1.1e-12]), "energy": np.asarray([1.25, 1.25001])}
    path = tmp_path / "state.npz"
    check.checkpoint_write(path, arrays, "fixed-id", 2)
    result = check.checkpoint_read(path, "fixed-id", 2)
    assert result.keys() == arrays.keys()
    for name in arrays:
        assert result[name].dtype == arrays[name].dtype
        assert result[name].shape == arrays[name].shape
        assert result[name].tobytes() == arrays[name].tobytes()
    with pytest.raises(ValueError, match="already exists"):
        check.checkpoint_write(path, arrays, "fixed-id", 2)


@pytest.mark.parametrize("identity,frame,message", [("other", 2, "fingerprint"), ("fixed-id", 3, "frame")])
def test_checkpoint_rejects_other_registration_or_split(tmp_path, identity, frame, message):
    path = tmp_path / "state.npz"
    check.checkpoint_write(path, {"q": np.zeros((2, 3, 3))}, "fixed-id", 2)
    with pytest.raises(ValueError, match=message):
        check.checkpoint_read(path, identity, frame)


def test_checkpoint_integrity_is_required(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, q=np.zeros((2, 3, 3)), fingerprint="fixed-id", completed_frame=2)
    with pytest.raises(ValueError, match="integrity envelope"):
        check.checkpoint_read(path, "fixed-id", 2)


@pytest.mark.parametrize("cpu,wall", [(55.0, 1.0), (1.0, 115.0)])
def test_budget_is_checked_before_admission(monkeypatch, cpu, wall):
    monkeypatch.setattr(check, "cpu_seconds", lambda: cpu)
    monkeypatch.setattr(check.time, "monotonic", lambda: check.ENTRY_WALL + wall)
    with pytest.raises(TimeoutError, match="admission"):
        check.admission()


def test_resource_caps_installed_in_isolated_process_before_scientific_imports():
    if 6 not in os.sched_getaffinity(0):
        pytest.skip("fixed adopted CPU6 is unavailable on this machine")
    script = """from benchmarks.h3 import p256_native_calibration as m
import sys,json,resource,os
assert 'numpy' not in sys.modules and 'scipy' not in sys.modules
m.install_limits()
print(json.dumps({'limit':resource.getrlimit(resource.RLIMIT_CPU),'affinity':list(os.sched_getaffinity(0)), 'threads':[os.environ[x] for x in m.THREADS]}))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=check.ROOT,
                            capture_output=True, text=True, timeout=10, check=True)
    data = json.loads(result.stdout)
    assert data == {"limit": [60, 60], "affinity": [6], "threads": ["1"] * 4}


@pytest.fixture
def authorization_files(tmp_path, monkeypatch):
    proposal = tmp_path / "proposal.json"
    check.write_new(proposal, {"registered": True})
    monkeypatch.setattr(check, "PROPOSAL", proposal)
    monkeypatch.setattr(check, "PROPOSAL_SHA", check.sha(proposal))
    monkeypatch.setattr(check, "EVIDENCE", tmp_path)
    adoption = {"adopted": True, "proposal_sha256": check.PROPOSAL_SHA, "resources": {
        "CPU_affinity": [6], "workers": 1, "threads": 1, "whole_process_hard_CPU_seconds": 60,
        "admission_CPU_seconds": 55, "whole_process_wall_seconds": 120}}
    review = {"approved_for_once_execution": True, "proposal_sha256": check.PROPOSAL_SHA,
              "source_sha256": check.sha(check.__file__)}
    check.write_new(tmp_path / "root-adoption.json", adoption)
    check.write_new(tmp_path / "review.json", review)
    return tmp_path, adoption, review


def test_exact_adoption_and_review_authorize(authorization_files):
    directory, _a, _r = authorization_files
    assert check.authorization(directory / "review.json") == {"registered": True}


@pytest.mark.parametrize("which,key,value", [("review", "source_sha256", "0" * 64),
    ("review", "approved_for_once_execution", False), ("review", "proposal_sha256", "0" * 64),
    ("root-adoption", "adopted", False)])
def test_unreviewed_or_unadopted_source_is_rejected(authorization_files, which, key, value):
    directory, adoption, review = authorization_files
    data = adoption if which == "root-adoption" else review
    data[key] = value
    (directory / f"{which}.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        check.authorization(directory / "review.json")


def test_restart_bitwise_gate_detects_signed_zero():
    native = {"q": np.asarray([-0.0]), "step": np.asarray([1])}
    changed = {"q": np.asarray([0.0]), "step": np.asarray([1])}
    proposal = {"gates": {"map_comparison": {"absolute_tolerance_q_bohr": 5e-10,
        "absolute_tolerance_p_au": 5e-9, "absolute_tolerance_surface_bohr": 2e-10,
        "absolute_tolerance_energy_hartree": 2e-10, "relative_tolerance": 2e-8}}}
    with pytest.raises(ValueError, match="not bitwise"):
        check.comparisons(np, native, native, changed, proposal)


def test_identical_nan_initial_observation_cannot_pass_comparison():
    trace = {"energy": np.asarray([np.nan, 1.0, 1.0, 1.0, 1.0])}
    with pytest.raises(ValueError, match="including initial frame"):
        check.comparisons(np, trace, trace, trace, {})
