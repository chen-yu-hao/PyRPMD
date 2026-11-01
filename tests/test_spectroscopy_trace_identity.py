"""Spectral reanalysis must preserve the sampled physics and raw observables."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.system import System
from pyrpmd.workflows.spectroscopy import run_spectroscopy, reanalyze_spectroscopy
from test_spectroscopy_workflow import job, raw_job


@pytest.mark.parametrize("change", ["beta", "beads", "method", "target_frequency", "mass"])
def test_reanalysis_rejects_changed_trajectory_physics_without_writes(tmp_path, change):
    config, system = job(tmp_path)
    output = tmp_path / "run"
    run_spectroscopy(config, system, output)
    before = {p.name: p.read_bytes() for p in (output / "results").iterdir()}
    altered = copy.deepcopy(config)
    if change == "beta":
        altered["beta"] *= 2
    elif change == "beads":
        altered["beads"] *= 2
    elif change == "method":
        altered["spectroscopy"]["method"] = "trpmd"
        del altered["spectroscopy"]["target_frequency"]
    elif change == "target_frequency":
        altered["spectroscopy"]["target_frequency"] *= 2
    else:
        system = System(system.symbols, system.masses * 2, system.geometry)
    with pytest.raises(ValueError, match="physics|identity"):
        reanalyze_spectroscopy(altered, system, output)
    assert {p.name: p.read_bytes() for p in (output / "results").iterdir()} == before


def test_reanalysis_can_change_processing_without_discarding_raw_properties(tmp_path):
    config, system = job(tmp_path)
    output = tmp_path / "run"
    run_spectroscopy(config, system, output)
    trace = output / "results" / "spectroscopy-traces.npz"
    original = trace.read_bytes()
    altered = copy.deepcopy(config)
    altered["spectroscopy"].update(spectra=["ir"], window="boxcar", normalize=False)
    reanalyze_spectroscopy(altered, system, output)
    assert trace.read_bytes() == original
    altered["spectroscopy"]["spectra"] = ["raman", "sfg"]
    reanalyze_spectroscopy(altered, system, output)
    assert trace.read_bytes() == original


def test_offline_reanalysis_retains_original_model_provenance(tmp_path):
    config, system = job(tmp_path)
    output = tmp_path / "run"
    initial = run_spectroscopy(config, system, output)
    (tmp_path / "properties_model.py").unlink()
    config["potential"] = {"provider": "not_installed"}
    result = reanalyze_spectroscopy(config, system, output)
    assert result["trajectory_provenance"] == initial["trajectory_provenance"]
    assert result["trajectory_provenance"]["potential"]["provider"] == "harmonic"
    assert len(result["trajectory_provenance"]["source_files"]) == 1


def test_legacy_trace_without_physics_identity_is_not_silently_authenticated(tmp_path):
    config, system = job(tmp_path)
    output = tmp_path / "run"
    run_spectroscopy(config, system, output)
    path = output / "results" / "spectroscopy-traces.npz"
    with load_checkpoint(path) as data:
        arrays = {name: data[name] for name in data.files if name != "metadata"}
    metadata = {"method": "pa-cmd", "axes": "replica,time,property_components"}
    atomic_savez(path, kind="PyRPMD.spectroscopy-traces.v1", metadata=json.dumps(metadata), **arrays)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        reanalyze_spectroscopy(config, system, output)
    assert path.read_bytes() == before


def test_cli_reanalysis_detects_saved_input_temperature_change(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import analyze_run, run

    output = tmp_path / "public-run"
    run(JobSpec(raw_job(tmp_path), tmp_path), output=output)
    resolved_path = output / "input" / "resolved.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["beta"] *= 2
    resolved_path.write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="physics|identity"):
        analyze_run(output, recompute=True)
