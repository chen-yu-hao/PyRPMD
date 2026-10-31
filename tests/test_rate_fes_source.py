"""External PMFs preserve the rate measure and require explicit reweighting."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.analysis.fes_result import FESResult, write_fes_result
from pyrpmd.input import InputError, JobSpec, rate_pmf_options
from pyrpmd.io import fingerprint, write_json
from pyrpmd.propagation import scientific_config
from pyrpmd.runner import run_resolved, resume_run
from pyrpmd.workflows.rate import _prepare
from pyrpmd.workflows.rate_source import read_source, source_estimate, source_paths


def saved_source(tmp_path, method="umbrella", *, reweighted=False, measure="rate_flux"):
    spec = JobSpec.from_file(Path(__file__).resolve().parents[1] / "examples/rate/pilot.inp")
    spec.data["native"] = False
    spec.data["beads"] = 1
    spec.data["recrossing"] = {
        "dividing_surface": 1.0, "parent_replicas": 1, "parent_samples": 2,
        "equilibration_steps": 2, "parent_interval_steps": 2,
        "child_pairs": 1, "child_steps": 4, "bootstrap": 0,
    }
    config, system = spec.resolve()
    config = _prepare(config, system, tmp_path / "preparation")
    source = copy.deepcopy(config)
    source.update(task="fes", fes={"method": method})
    source.pop("rate", None)
    source.pop("recrossing", None)
    if method != "umbrella":
        source.pop("umbrella")
        source[method] = {"cvs": ["xi"]}
    directory = tmp_path / "source"
    write_json(directory / "input/resolved.json", source)
    write_json(directory / "input/system.json", system.to_dict())
    write_json(directory / "manifest.json", {
        "physics_fingerprint": fingerprint(scientific_config(source), system), "files": {},
    })
    edges = np.linspace(-0.15, 1.25, 15)
    grid = (edges[:-1] + edges[1:]) / 2
    free = 0.001 * (grid - grid[0])
    result = FESResult((edges,), free, np.full_like(free, 0.0001),
                       np.zeros_like(free, bool), ("dimensionless",), method,
                       "block_bootstrap" if method == "umbrella" else "replica_spread",
                       metadata={"beta": config["beta"], "measure": measure, "reweighted": reweighted})
    write_fes_result(directory, result)
    config.setdefault("rate", {})["pmf_source"] = source_paths(str(directory))
    return config, system, result, directory


def test_default_source_remains_umbrella_and_raw_bias_selectors_fail():
    assert rate_pmf_options({})["source"] is None
    assert rate_pmf_options({"rate": {"pmf_source": "umbrella"}})["source"] is None
    for method in ("metadynamics", "opes"):
        with pytest.raises(InputError, match="reweighted"):
            rate_pmf_options({"rate": {"pmf_source": method}})
        with pytest.raises(InputError, match="reweighted"):
            source_paths(method)


def test_saved_umbrella_barrier_uses_supported_interpolation(tmp_path):
    config, system, result, _ = saved_source(tmp_path)
    estimate = source_estimate(config, system)
    assert estimate["ready"]
    assert estimate["free_energy_barrier_hartree"] == pytest.approx(0.001)
    assert estimate["barrier_standard_error_hartree"] is None
    assert estimate["pmf_bootstrap_barriers"] == []
    assert estimate["source_method"] == "umbrella"


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_nonumbrella_source_requires_both_optin_and_reweighting(method, tmp_path):
    config, system, result, directory = saved_source(tmp_path, method)
    with pytest.raises(ValueError, match="must be reweighted"):
        read_source(config, system)
    config["rate"]["pmf_source"]["allow_non_umbrella"] = True
    with pytest.raises(ValueError, match="must be reweighted"):
        read_source(config, system)
    result.metadata["reweighted"] = True
    write_fes_result(directory, result)
    estimate = source_estimate(config, system)
    assert estimate["ready"]
    assert any("requires reweighting" in text for text in estimate["warnings"])
    assert np.isnan(estimate["standard_error"]).all()


@pytest.mark.parametrize("method, estimator", [("metadynamics", "bias_estimate"), ("opes", "kde")])
def test_reweighting_flag_cannot_override_raw_estimator_provenance(method, estimator, tmp_path):
    config, system, result, directory = saved_source(tmp_path, method, reweighted=True)
    config["rate"]["pmf_source"]["allow_non_umbrella"] = True
    result.metadata["estimator"] = estimator
    write_fes_result(directory, result)
    with pytest.raises(ValueError, match="Raw bias estimates"):
        read_source(config, system)


@pytest.mark.parametrize("field, value", [
    ("schema", "foreign.run"), ("schema_version", True), ("schema_version", 999),
])
def test_source_manifest_schema_must_be_supported(field, value, tmp_path):
    config, system, _, directory = saved_source(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError, match="schema"):
        read_source(config, system)


def test_source_manifest_rejects_changed_scientific_input(tmp_path):
    config, system, _, directory = saved_source(tmp_path)
    path = directory / "input/resolved.json"
    source = json.loads(path.read_text())
    source["beta"] *= 1.1
    write_json(path, source)
    with pytest.raises(ValueError, match="does not match its manifest"):
        read_source(config, system)


def test_source_measure_change_needs_reweighting_provenance(tmp_path):
    config, system, _, directory = saved_source(tmp_path)
    path = directory / "input/resolved.json"
    source = json.loads(path.read_text())
    source["analysis"]["measure"] = "cv_density"
    write_json(path, source)
    write_json(directory / "manifest.json", {
        "physics_fingerprint": fingerprint(scientific_config(source), system), "files": {},
    })
    with pytest.raises(ValueError, match="measure differs"):
        read_source(config, system)


def test_source_search_domain_is_not_silently_clipped(tmp_path):
    config, system, _, _ = saved_source(tmp_path)
    config["rate"]["pmf_search_range"] = [0.0, 2.0]
    with pytest.raises(ValueError, match="extends beyond"):
        source_estimate(config, system)


def test_source_does_not_override_an_incompatible_requested_rate_measure(tmp_path):
    config, system, _, _ = saved_source(tmp_path)
    config["analysis"]["measure"] = "cv_density"
    with pytest.raises(ValueError, match="analysis.measure='rate_flux'"):
        _prepare(config, system, tmp_path / "bad-rate")


def test_density_source_is_rejected_and_masked_barrier_is_withheld(tmp_path):
    config, system, result, directory = saved_source(tmp_path, measure="cv_density")
    with pytest.raises(ValueError, match="flux metric"):
        read_source(config, system)
    result.metadata["measure"] = "rate_flux"
    result.mask[6] = True
    write_fes_result(directory, result)
    estimate = source_estimate(config, system)
    assert not estimate["ready"]
    assert estimate["free_energy_barrier_hartree"] is None


def test_rate_executes_from_saved_source_and_hashes_it_for_resume(tmp_path, monkeypatch):
    config, system, result, directory = saved_source(tmp_path)
    from pyrpmd.workflows import rate

    monkeypatch.setattr(rate, "run_fes", lambda *args, **kwargs: pytest.fail("external source must not resample umbrellas"))
    output = tmp_path / "rate"
    summary = run_resolved(config, system, output=output)
    assert summary["pmf"]["method"] == "source"
    assert summary["pmf"]["source_method"] == "umbrella"
    assert not (output / "umbrella").exists()
    assert resume_run(output)["pmf"]["source_method"] == "umbrella"
    result.free_energy += 0.2
    write_fes_result(directory, result)
    with pytest.raises(RuntimeError, match="changed or is missing"):
        resume_run(output)
