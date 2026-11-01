"""Independent time-origin and preregistered selection checks for water cadence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water.velocity_cadence import compare_cadences, select_checkpoint, vacf_trace


def test_block_fft_matches_direct_all_origin_velocity_dot_without_demeaning():
    values = np.random.default_rng(38).normal(size=(53, 7, 3)) + [.8, -.1, .3]
    expected = np.asarray([np.mean(np.sum(values[:len(values)-lag] * values[lag:], axis=2))
                           for lag in range(21)])
    for block in (1, 4, 100):
        np.testing.assert_allclose(vacf_trace(values, 20, component_block=block), expected,
                                   atol=2e-15, rtol=2e-15)


def test_cadence_comparison_preserves_constant_velocity_and_origins(monkeypatch):
    monkeypatch.delattr(np, "trapezoid", raising=False)  # NumPy 1.26 compatibility.
    values = np.zeros((1601, 2, 3))
    values[:, 0] = [1, 2, 3]
    values[:, 1] = [-2, 1, 1]
    rows, curves = compare_cadences(values, dt_fs=.25, cadences_fs=[.25, .5, 1, 2.5, 5], max_lag_ps=.2)
    expected = .2 * (14 + 6) / 2 / 3
    for row in rows:
        assert row["integral_trace_over_3_angstrom2_ps"] == pytest.approx(expected, abs=2e-15)
        assert row["difference_from_dense_angstrom2_ps"] == pytest.approx(0, abs=2e-15)
        assert row["dense_origins_quadrature_difference_angstrom2_ps"] == pytest.approx(0, abs=2e-15)
    assert rows[-1]["frames"] == 81 and rows[-1]["origins_at_maximum_lag"] == 41
    assert len(curves["dense_lag_ps"]) == 801


def test_checkpoint_selection_is_fixed_not_latest_and_rejects_ambiguous_prefix():
    registration = {"selected_seed": 1101, "selected_start_checkpoint": {"state_step": 400000}}
    selected = {"step": 400000, "file": "chunk-000099.npz", "sha256": "expected"}
    manifest = {"config": {"seed": 1101}, "chunks": [selected, {"step": 404000, "file": "later.npz"}]}
    assert select_checkpoint(manifest, registration) == selected
    invalid = deepcopy(manifest)
    invalid["chunks"].append(selected)
    with pytest.raises(ValueError, match="exactly one"):
        select_checkpoint(invalid, registration)
    invalid["chunks"] = manifest["chunks"][1:]
    with pytest.raises(ValueError, match="exactly one"):
        select_checkpoint(invalid, registration)
    invalid = deepcopy(manifest)
    invalid["config"]["seed"] = 1102
    with pytest.raises(ValueError, match="seed"):
        select_checkpoint(invalid, registration)


@pytest.mark.parametrize("cadences", [[.3], [0], [float("nan")]])
def test_invalid_record_cadence_is_rejected(cadences):
    with pytest.raises(ValueError, match="cadence"):
        compare_cadences(np.ones((41, 1, 3)), dt_fs=.25, cadences_fs=cadences, max_lag_ps=.005)


def test_dense_replay_resume_keeps_one_registered_momentum_draw_and_native_map(tmp_path, monkeypatch):
    pytest.importorskip("openmm")
    from benchmarks.water import velocity_cadence as control
    from benchmarks.water.campaign import sha256
    from benchmarks.water.native_reference import Config, run
    from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU

    config = Config(cells_per_axis=2, cutoff_angstrom=2.5, platform="Reference",
                    dt_fs=.1, equilibration_steps=8, equilibration_record_stride=2,
                    sampling_refresh_fs=.4, segment_steps=16, segments=1,
                    checkpoint_steps=4, record_stride=1, trajectory_precision="float64")
    baseline = tmp_path / "baseline"
    original = run(baseline, config, max_steps=8)
    registration_path = tmp_path / "registration.json"
    registration_path.write_text(json.dumps({"purpose": "small deterministic regression fixture"}))
    selection = {
        "checkpoint": original["chunks"][-1],
        "baseline_manifest_snapshot_sha256": sha256(baseline / "manifest.json"),
        "registration_sha256": sha256(registration_path),
    }
    monkeypatch.setattr(control, "prepare_selection", lambda *_: ({}, original, config, selection))
    full, split = tmp_path / "full", tmp_path / "split"
    complete = control.replay(baseline, registration_path, tmp_path / "unused.json", full)
    partial = control.replay(baseline, registration_path, tmp_path / "unused.json", split, wall_seconds=1e-12)
    assert partial["committed_steps"] == 4 and not partial["complete"]
    resumed = control.replay(baseline, registration_path, tmp_path / "unused.json", split, resume=True)
    assert resumed["complete"] and not resumed["additional_diffusion_replica"]

    def velocities(directory, manifest):
        pieces = []
        for record in manifest["chunks"]:
            with np.load(directory / record["file"], allow_pickle=False) as data:
                pieces.append(data["molecular_velocity_angstrom_ps"])
        return np.concatenate(pieces)

    reference = velocities(full, complete)
    np.testing.assert_allclose(velocities(split, resumed), reference, atol=2e-12, rtol=0)
    assert len(reference) == 17
    continued = run(baseline, config, resume=True)
    pieces = []
    for record in continued["chunks"]:
        with np.load(baseline / record["file"], allow_pickle=False) as data:
            pieces.append(data["molecular_velocity_au"][data["segment"] == 0])
    native_record = np.concatenate(pieces) * BOHR_ANGSTROM * 1000 * FS_TO_AU
    np.testing.assert_allclose(reference, native_record, atol=2e-12, rtol=0)
