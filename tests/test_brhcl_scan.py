"""Actual BrHCl native streams: P>1 analysis, strict identity, and exact resume."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from benchmarks.brhcl import scan
from benchmarks.brhcl.pes import DEFAULT_LIBRARY
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import ANGSTROM_TO_AU, FS_TO_AU
from pyrpmd.dynamics import Dynamics, native_capabilities

pytestmark = [
    pytest.mark.skipif(not DEFAULT_LIBRARY.is_file(), reason="build optional pinned BrHCl PES first"),
    pytest.mark.skipif(not native_capabilities()["available"], reason="native PyRPMD build required"),
]


def _short_plan():
    settings = scan.plan(profile="smoke", beads=2, equilibration_fs=2., production_fs=12.,
                         bootstrap=0, orientation_mc_trials=32)
    # Close windows test the full real-model estimator without making a claim
    # about the sparse pilot's channel mixing or physical convergence.
    settings.update(centers_angstrom=[[1.8, 1.7], [1.9, 1.8]], streams=4,
                    bins=[3, 4], ranges_angstrom=[[.9, 3.], [.8, 3.2]],
                    min_count=1, min_weighted_bin_ess=1., checkpoint_steps=40,
                    planned_total_samples=48, single_dense_matrix_lower_bound_bytes=1536,
                    aggregate_bead_steps=1120)
    return settings


def _payload(path):
    with load_checkpoint(path, expected_kind="arrays", require_integrity=True) as saved:
        return {name: saved[name].copy() for name in saved}


@pytest.fixture(scope="module")
def real_scan(tmp_path_factory):
    output = tmp_path_factory.mktemp("brhcl-real-scan")
    settings = _short_plan()
    scan.run(output, settings, library=DEFAULT_LIBRARY, workers=1)
    library_hash = hashlib.sha256(DEFAULT_LIBRARY.read_bytes()).hexdigest()
    return output, settings, library_hash


def test_real_two_bead_native_sampling_and_non_square_analysis(real_scan):
    output, settings, library_hash = real_scan
    for window in range(2):
        for family in scan.FAMILIES:
            saved = scan.checked_stream(scan._path(output, window, family), settings,
                                        library_hash, window, family, complete=True)
            assert saved["q"].shape == (2, 3, 3)
            assert saved["samples"].shape == (12, 2)
            assert saved["centroids"].shape == (12, 3, 3)
            assert np.max(saved["ring_radii"]) > 0
            np.testing.assert_array_equal(json.loads(str(saved["propagation"])),
                {"native_steps": 140, "python_steps": 0, "fallback_reasons": {}})
            moves = json.loads(str(saved["orientation_mc"]))
            assert moves["attempted"] == 2*settings["orientation_mc_trials"]
            assert 0 <= moves["accepted"] <= moves["attempted"]-moves["invalid_domain"]
            assert float(saved["time"]) == pytest.approx(14. * FS_TO_AU, rel=1e-12)
    summary = scan.analyze(output, bootstrap=0)
    assert summary["beads"] == 2 and summary["samples"] == 48
    assert summary["streams"] == 4 and summary["total_bins"] == 12
    assert summary["supported_bins"] > 0
    assert not summary["physical_converged"]
    assert summary["common_wall_retained"]
    assert summary["ring_radius_median_bohr"] > 0
    arrays = _payload(output / "fes.npz")
    assert arrays["free_energy_kcal_mol"].shape == (3, 4)
    supported = ~arrays["mask"]
    np.testing.assert_allclose(arrays["channel_fractions"][:, supported].sum(axis=0), 1., atol=1e-13)
    assert np.isfinite(arrays["free_energy_kcal_mol"][supported]).all()
    assert np.isnan(arrays["standard_error_kcal_mol"]).all()
    with pytest.raises(ValueError, match="only a P=1 reference"):
        scan.analyze(output, bootstrap=0, reference=True)


def test_completed_stream_reuse_does_not_propagate(real_scan, monkeypatch):
    output, settings, library_hash = real_scan
    path = scan._path(output, 0, "compact")
    before = _payload(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("completed checkpoint must not be propagated again")

    monkeypatch.setattr(Dynamics, "propagate", forbidden)
    result = scan.sample_stream((settings, str(DEFAULT_LIBRARY), library_hash, str(output), 0, "compact"))
    assert result["samples"] == 12 and result["native_steps"] == 140
    after = _payload(path)
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)


def test_interrupt_resume_preserves_native_thermostat_clock_and_exact_stream(tmp_path, monkeypatch):
    settings = _short_plan()
    library_hash = hashlib.sha256(DEFAULT_LIBRARY.read_bytes()).hexdigest()
    direct, resumed = tmp_path / "direct", tmp_path / "resumed"
    original = Dynamics.propagate
    clock = []

    def tracing(self, state, count, **kwargs):
        clock.append((state.step, count))
        return original(self, state, count, **kwargs)

    monkeypatch.setattr(Dynamics, "propagate", tracing)
    scan.sample_stream((settings, str(DEFAULT_LIBRARY), library_hash, str(direct), 0, "open"))
    assert clock == [(0, 20), (20, 20), (40, 40), (80, 20), (100, 20), (120, 20)]
    # In particular, no native block straddles the Andersen refresh at step 100.
    assert all(begin // 100 == (begin + count - 1) // 100 for begin, count in clock)

    def interrupted(self, state, count, **kwargs):
        if state.step >= 40:
            raise RuntimeError("intentional interruption after an atomic checkpoint")
        return original(self, state, count, **kwargs)

    monkeypatch.setattr(Dynamics, "propagate", interrupted)
    task = (settings, str(DEFAULT_LIBRARY), library_hash, str(resumed), 0, "open")
    with pytest.raises(RuntimeError, match="intentional interruption"):
        scan.sample_stream(task)
    partial = scan.checked_stream(scan._path(resumed, 0, "open"), settings, library_hash, 0, "open")
    assert int(partial["elapsed"]) == 40
    assert len(partial["samples"]) == 2
    assert json.loads(str(partial["orientation_mc"]))["attempted"] == settings["orientation_mc_trials"]
    with pytest.raises(ValueError, match="incomplete"):
        scan.checked_stream(scan._path(resumed, 0, "open"), settings, library_hash, 0, "open", complete=True)
    monkeypatch.setattr(Dynamics, "propagate", original)
    scan.sample_stream(task)
    expected, actual = _payload(scan._path(direct, 0, "open")), _payload(scan._path(resumed, 0, "open"))
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


@pytest.mark.parametrize("field,value,match", [
    ("window", 1, "umbrella metadata"),
    ("family", "open", "umbrella metadata"),
    ("center", np.array([1.8, 1.7]) * ANGSTROM_TO_AU + .01, "umbrella metadata"),
    ("stiffness", np.eye(2), "umbrella metadata"),
    ("elapsed", 140.5, "scalar integer"),
    ("step", 140.5, "scalar integer"),
    ("step", True, "scalar integer"),
    ("time", float("nan"), "finite real scalar"),
    ("propagation", json.dumps({"native_steps": 139, "python_steps": 0, "fallback_reasons": {}}), "native propagation"),
    ("propagation", json.dumps({"native_steps": 140, "python_steps": 1, "fallback_reasons": {}}), "native propagation"),
    ("propagation", json.dumps({"native_steps": 140., "python_steps": 0, "fallback_reasons": {}}), "native propagation"),
    ("propagation", json.dumps({"native_steps": 140, "python_steps": 0, "fallback_reasons": {"unavailable": 1}}), "native propagation"),
    ("propagation", "[]", "native propagation"),
    ("ring_radii", np.array([np.nan]), "nonfinite checkpoint"),
    ("maximum_block_energy_drift_kbt", -1., "negative"),
])
def test_integrity_valid_but_semantically_corrupt_stream_is_rejected(real_scan, tmp_path, field, value, match):
    output, settings, library_hash = real_scan
    saved = _payload(scan._path(output, 0, "compact"))
    saved[field] = value
    path = tmp_path / "altered.npz"
    atomic_savez(path, kind="arrays", **saved)
    with pytest.raises(ValueError, match=match):
        scan.checked_stream(path, settings, library_hash, 0, "compact", complete=True)


def test_resume_scientific_identity_and_foreign_collection_are_rejected(real_scan, tmp_path):
    output, settings, library_hash = real_scan
    path = scan._path(output, 0, "compact")
    changed = copy.deepcopy(settings)
    changed["wall_angstrom"] = 7.
    with pytest.raises(ValueError, match="scientific identity"):
        scan.checked_stream(path, changed, library_hash, 0, "compact")
    with pytest.raises(ValueError, match="scientific identity"):
        scan.checked_stream(path, settings, "0" * 64, 0, "compact")
    with pytest.raises(ValueError, match="PES library changed"):
        scan.sample_stream((settings, str(DEFAULT_LIBRARY), "0" * 64, str(tmp_path), 0, "compact"))
    foreign = output / "streams" / "foreign.npz"
    try:
        atomic_savez(foreign, **_payload(path))
        with pytest.raises(ValueError, match="incomplete or contains foreign"):
            scan.analyze(output)
    finally:
        foreign.unlink(missing_ok=True)


@pytest.mark.parametrize("counts", [
    None,
    {"attempted": 32, "accepted": 0, "invalid_domain": 0},
    {"attempted": 64, "accepted": 65, "invalid_domain": 0},
    {"attempted": 64, "accepted": 32, "invalid_domain": 33},
    {"attempted": 64, "accepted": -1, "invalid_domain": 0},
    {"attempted": 64., "accepted": 0, "invalid_domain": 0},
    {"attempted": 64, "accepted": True, "invalid_domain": 0},
    {"attempted": 64, "accepted": 0},
])
def test_mc_refresh_attempt_counts_and_subcounts_are_validated(real_scan, tmp_path, counts):
    output, settings, library_hash = real_scan
    saved = _payload(scan._path(output, 0, "compact"))
    if counts is None:
        saved.pop("orientation_mc")
    else:
        saved["orientation_mc"] = json.dumps(counts)
    path = tmp_path / "bad-mc.npz"
    atomic_savez(path, **saved)
    with pytest.raises(ValueError, match="orientation MC counts"):
        scan.checked_stream(path, settings, library_hash, 0, "compact")


def test_historical_plan_without_mc_field_and_checkpoint_remains_readable(tmp_path):
    settings = _short_plan()
    settings.pop("orientation_mc_trials")
    library_hash = hashlib.sha256(DEFAULT_LIBRARY.read_bytes()).hexdigest()
    scan.sample_stream((settings, str(DEFAULT_LIBRARY), library_hash, str(tmp_path), 0, "compact"))
    path = scan._path(tmp_path, 0, "compact")
    saved = _payload(path)
    assert json.loads(str(saved.pop("orientation_mc")))["attempted"] == 0
    atomic_savez(path, **saved)
    checked = scan.checked_stream(path, settings, library_hash, 0, "compact", complete=True)
    assert int(checked["elapsed"]) == 140
