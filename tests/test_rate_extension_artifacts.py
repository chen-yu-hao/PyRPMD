"""Published rate generations remain readable without optional HDF5 copies."""

import json
import sys

import pytest

from pyrpmd.analysis.rate_io import postprocess_rate
from pyrpmd.analysis.reporting import collect_run
from pyrpmd.runner import extend_run, run_resolved
from pyrpmd.workflows import rate_extend

from test_rate_extend import LONG, configuration, generation


def _without_hdf5(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "h5py", None)
    config, system = configuration()
    directory = tmp_path / "run"
    run_resolved(config, system, output=directory)
    assert not (directory / "umbrella" / "fes.h5").exists()
    return directory, config


def _interrupt_before_recrossing(*args):
    raise RuntimeError("interrupt after umbrella sampling")


def test_archived_sample_prefix_survives_a_later_pending_generation(tmp_path, monkeypatch):
    directory, config = _without_hdf5(tmp_path, monkeypatch)
    extend_run(directory, production_steps=LONG)
    artifacts = generation(directory)
    assert not (artifacts / "umbrella" / "fes.h5").exists()
    previous = collect_run(directory, energy_unit="hartree")
    sampling = previous["sampling_diagnostics"]["umbrella"]
    assert len(sampling["windows"]) == len(config["umbrella"]["centers"])
    for window in sampling["windows"]:
        assert window["samples"] == LONG // config["sampling"]["sample_interval"] * window["replicas"]
    assert any(".chunks/" in name for name in previous["_sources"])
    assert any(str(artifacts / "umbrella" / "checkpoints") in name for name in previous["_sources"])
    assert not any(str(directory / "umbrella" / "checkpoints") + "/window_" in name
                   and name.endswith(".npz") and ".chunks/" not in name
                   for name in previous["_sources"])

    monkeypatch.setattr(rate_extend, "_bind_generation", _interrupt_before_recrossing)
    with pytest.raises(RuntimeError, match="interrupt after umbrella sampling"):
        extend_run(directory, production_steps=LONG + 17)
    current = collect_run(directory, energy_unit="hartree")
    assert current["sampling_diagnostics"]["umbrella"] == sampling
    assert current["metadata"]["effective_production_steps"] == LONG


def test_original_result_withholds_advanced_live_samples_without_hdf5(tmp_path, monkeypatch):
    directory, _ = _without_hdf5(tmp_path, monkeypatch)
    monkeypatch.setattr(rate_extend, "_bind_generation", _interrupt_before_recrossing)
    with pytest.raises(RuntimeError, match="interrupt after umbrella sampling"):
        extend_run(directory, production_steps=LONG)
    report = collect_run(directory, energy_unit="hartree")
    assert any(plot["id"].endswith("-pmf") for plot in report["_plots"])
    assert "umbrella" not in report.get("sampling_diagnostics", {})
    assert any("advanced beyond the published rate result" in finding["message"]
               for finding in report["findings"])


def test_postprocessing_preserves_generation_receipt_and_identity(tmp_path, monkeypatch):
    directory, _ = _without_hdf5(tmp_path, monkeypatch)
    extend_run(directory, production_steps=LONG)
    artifacts = generation(directory)
    result = postprocess_rate(directory, bootstrap=0)
    sources = {source["path"] for source in result["source_files"]}
    for name in ("receipt.json", "generation.json"):
        path = artifacts / name
        assert str(path) in sources
        before = path.read_bytes()
        with pytest.raises(ValueError, match="must not overwrite a source scientific file"):
            postprocess_rate(directory, output=path, bootstrap=0)
        assert path.read_bytes() == before


def test_explicit_null_generation_pointer_never_falls_back(tmp_path, monkeypatch):
    directory, _ = _without_hdf5(tmp_path, monkeypatch)
    extend_run(directory, production_steps=LONG)
    path = directory / "results" / "summary.json"
    summary = json.loads(path.read_text())
    summary["rate_extension"] = None
    path.write_text(json.dumps(summary))
    for reader in (postprocess_rate, collect_run):
        with pytest.raises(ValueError, match="Unknown published rate extension format"):
            reader(directory)
