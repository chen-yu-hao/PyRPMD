"""Checkpoint-only rate reports use the saved independent PMF clock."""

import sys

import numpy as np
import pytest

from pyrpmd.analysis.reporting import collect_run
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.input import JobSpec
from pyrpmd.runner import extend_run, run_resolved
from pyrpmd.workflows import fes, rate_extend

from test_rate_pmf_extend import LONG, SHORT, configuration


def _without_hdf5(path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    raw = configuration()
    raw["sampling"]["interval_steps"] = 5
    config, system = JobSpec.from_mapping(raw).resolve()
    assert config["schema_version"] == 1
    stage = config["rate"]["pmf_stage"]["config"]
    assert config["dt_au"] == .003 and stage["dt_au"] == .006
    assert config["sampling"]["production_steps"] == 200
    assert stage["sampling"]["production_steps"] == SHORT
    result = run_resolved(config, system, output=path)
    assert result["pmf"]["ready"]
    assert not (path / "umbrella/fes.h5").exists()
    assert not (path / "umbrella/umbrella-plan.json").exists()
    for checkpoint in (path / "umbrella/checkpoints").glob("window_*.npz"):
        with load_checkpoint(checkpoint) as saved:
            assert "window_id" not in saved
    return config


def _offline_report(path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline report constructed a CV, PES, or dynamics object")

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "__init__", forbidden)
        patch.setattr("pyrpmd.cv.make_cvs", forbidden)
        patch.setattr("pyrpmd.potentials.make_potential", forbidden)
        patch.setattr(fes, "resolve_window_plan", forbidden)
        return collect_run(path, energy_unit="hartree")


def _assert_samples(report, production):
    sampling = report["sampling_diagnostics"]["umbrella"]
    rows = sampling["windows"]
    assert len(rows) == 5
    assert all(row["samples"] == production // 3 for row in rows)
    np.testing.assert_array_equal([row["center_atomic"] for row in rows],
                                  np.linspace(0, 1, 5)[:, None])
    assert all(row["stiffness_atomic"] == [[10.]] for row in rows)
    return sampling


def test_schema_one_independent_pmf_checkpoint_report_uses_stage_clock_and_bias(tmp_path, monkeypatch):
    _without_hdf5(tmp_path, monkeypatch)
    report = _offline_report(tmp_path, monkeypatch)
    _assert_samples(report, SHORT)
    assert not any("checkpoint clock" in item["reason"] for item in report["skipped"])


def _interrupt_before_recrossing(*args):
    raise RuntimeError("interrupt after umbrella sampling")


def test_independent_pmf_report_withholds_actually_advanced_live_checkpoints(tmp_path, monkeypatch):
    _without_hdf5(tmp_path, monkeypatch)
    monkeypatch.setattr(rate_extend, "_bind_generation", _interrupt_before_recrossing)
    with pytest.raises(RuntimeError, match="interrupt after umbrella sampling"):
        extend_run(tmp_path, production_steps=LONG)
    report = _offline_report(tmp_path, monkeypatch)
    assert "umbrella" not in report.get("sampling_diagnostics", {})
    assert any("advanced beyond the published rate result" in item["message"]
               for item in report["findings"])
    assert any(plot["id"].endswith("-pmf") for plot in report["_plots"])


def test_independent_pmf_report_reads_published_generation_prefix_after_later_extension(tmp_path, monkeypatch):
    _without_hdf5(tmp_path, monkeypatch)
    extend_run(tmp_path, production_steps=LONG)
    published = _offline_report(tmp_path, monkeypatch)
    sampling = _assert_samples(published, LONG)
    assert any(".chunks/" in path for path in published["_sources"])
    monkeypatch.setattr(rate_extend, "_bind_generation", _interrupt_before_recrossing)
    with pytest.raises(RuntimeError, match="interrupt after umbrella sampling"):
        extend_run(tmp_path, production_steps=LONG + 17)
    current = _offline_report(tmp_path, monkeypatch)
    assert _assert_samples(current, LONG) == sampling
    assert current["metadata"]["effective_production_steps"] == LONG
