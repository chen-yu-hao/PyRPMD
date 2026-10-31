"""Adaptive production provenance must survive the common FES file contract."""

import json

import numpy as np
import pytest

from pyrpmd.analysis.fes_result import FESResult
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.workflows.common import _legacy_fingerprint
from pyrpmd.workflows.fes import run_fes, reanalyze_fes
from pyrpmd.workflows import adaptive_umbrella as adaptive
from test_adaptive_umbrella import _config, _system


def test_three_dimensional_adaptive_annotations_are_in_every_canonical_file(tmp_path):
    cfg = _config(2)
    cfg["fes"] = {"method": "umbrella"}
    cfg["cvs"].append({"name": "z", "kind": "cartesian", "atom": 0, "axis": "z"})
    cfg["umbrella"]["centers"] = np.stack(
        np.meshgrid(*[[-0.3, 0.3]] * 3, indexing="ij"), axis=-1).reshape(-1, 3).tolist()
    cfg["analysis"]["bins"] = [3, 3, 3]
    cfg["adaptive_umbrella"].update(
        coverage_bins=[2, 2, 2], coverage_ranges=[[-0.6, 0.6]] * 3,
        min_bin_effective_samples=0.1, target_coverage=1.0,
    )
    sampled = run_fes(cfg, _system(), tmp_path, return_data=True)
    summary = sampled["summary"]
    assert summary["adaptive_umbrella"]["overlap_metric"] == adaptive.NORMALIZED_OVERLAP_METRIC
    assert "production_coverage" in summary["adaptive_umbrella"]
    assert json.loads((tmp_path / "adaptive-plan.json").read_text())["schema_version"] == 3

    def verify(saved_summary):
        # An empty supplied summary reads only authoritative archive metadata.
        # Otherwise a sibling JSON could conceal a stale embedded annotation.
        standalone = FESResult.read(tmp_path / "fes.npz", summary={})
        assert standalone.dimension == 3
        assert standalone.metadata["adaptive_umbrella"] == saved_summary["adaptive_umbrella"]
        assert standalone.metadata["warnings"] == saved_summary["warnings"]
        assert standalone.metadata["supported_draw_condition"] == sampled["data"]["supported_draw_condition"]
        for name in ("summary.json", "fes.json"):
            metadata = json.loads((tmp_path / name).read_text())
            assert metadata["adaptive_umbrella"] == saved_summary["adaptive_umbrella"]
            assert metadata["warnings"] == saved_summary["warnings"]

    verify(summary)
    reanalyzed = reanalyze_fes(cfg, _system(), tmp_path)
    assert reanalyzed["adaptive_umbrella"] == summary["adaptive_umbrella"]
    verify(reanalyzed)


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_adaptive_plan_and_stream_hashes_resume_with_the_original_metric(tmp_path, schema):
    cfg = _config(2)
    cfg["fes"] = {"method": "umbrella"}
    sampled = run_fes(cfg, _system(), tmp_path, return_data=True)
    plan = json.loads((tmp_path / "adaptive-plan.json").read_text())
    legacy = {key: value for key, value in cfg.items() if key != "fes"}
    plan["schema_version"] = schema
    plan.pop("overlap_metric")
    if schema == 1:
        plan.pop("next_pilot_steps")
    plan["fingerprint"] = _legacy_fingerprint(legacy, _system(), {"adaptive_umbrella_schema": 1})
    adaptive._save_plan(tmp_path, plan)
    frozen = adaptive._stage_config(legacy, plan["centers"], plan["stiffness"], plan["production_seed"])
    unchanged = {}
    for stream in sampled["streams"]:
        from pathlib import Path

        path = Path(stream["checkpoint"])
        with load_checkpoint(path) as saved:
            payload = {key: saved[key].copy() for key in saved.files if key != "_pyrpmd_checkpoint"}
        identity = {key: stream[key] for key in ("window", "replica", "center", "stiffness")}
        payload["fingerprint"] = _legacy_fingerprint(frozen, _system(), identity)
        atomic_savez(path, kind="arrays", **payload)
        unchanged[path] = path.read_bytes()
    for result in (run_fes(cfg, _system(), tmp_path, return_data=True),
                   reanalyze_fes(cfg, _system(), tmp_path, return_data=True)):
        assert result["summary"]["adaptive_umbrella"]["overlap_metric"] == adaptive.LEGACY_OVERLAP_METRIC
        np.testing.assert_array_equal(result["data"]["free_energy"], sampled["data"]["free_energy"])
        assert all(path.read_bytes() == payload for path, payload in unchanged.items())
        saved = FESResult.read(tmp_path / "fes.npz", summary={})
        assert saved.metadata["adaptive_umbrella"]["overlap_metric"] == adaptive.LEGACY_OVERLAP_METRIC
