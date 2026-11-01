"""Public FES namespaces, method dispatch and authenticated legacy migration."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.input import InputError, JobSpec
from pyrpmd.io import fingerprint, write_json
from pyrpmd.propagation import scientific_config
from pyrpmd.runner import analyze_run, prepare, resume_run, run, run_resolved
from pyrpmd.workflows.common import _legacy_fingerprint, _fingerprint
from pyrpmd.system import System


def job(method):
    method_options = {
        "umbrella": {"centers": [-0.2, 0.2], "stiffness": 0.3},
        "metadynamics": {"sigma": 0.3, "height": 0.01, "pace": 3, "bias_factor": 5},
        "opes": {"sigma": 0.3, "barrier": 0.1, "pace": 3, "bias_factor": 5},
    }
    return {
        "task": "fes", "temperature": 300, "beads": 1, "native": False, "seed": 19,
        "system": {"symbols": ["H"], "masses": [1.0], "geometry": [[0.1, 0, 0]]},
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.1}},
        "dynamics": {"timestep": 0.02},
        "output": {"checkpoint_steps": 5},
        "fes": {
            "method": method,
            "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
            "sampling": {"equilibration_steps": 3, "production_steps": 30,
                         "interval_steps": 2, "andersen_interval": 3, "replicas": 2},
            "analysis": {"bins": 4, "bootstrap": 0},
            method: method_options[method],
        },
    }


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
def test_nested_and_legacy_namespaces_have_same_scientific_identity(method):
    nested = job(method)
    legacy = copy.deepcopy(nested)
    table = legacy.pop("fes")
    table.pop("method")
    legacy.update(table)
    legacy["task"] = "fes" if method == "umbrella" else method
    current, system = JobSpec.from_mapping(nested).resolve()
    historical, _ = JobSpec.from_mapping(legacy).resolve()
    assert current["task"] == historical["task"] == "fes"
    assert current["fes"] == {"method": method}
    assert _fingerprint(current, system, {}) == _fingerprint(historical, system, {})
    if method != "umbrella":
        assert any("deprecated" in warning for warning in historical["_warnings"])


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
def test_entry_dispatches_prepare_run_analyze_and_recompute(method, tmp_path):
    data = job(method)
    assert f"{method}_preparation" in prepare(data)
    result = run(data, output=tmp_path / method)
    assert result["method"] == method
    assert analyze_run(tmp_path / method)["method"] == method
    assert analyze_run(tmp_path / method, bins=[5])["method"] == method
    resumed = resume_run(tmp_path / method)
    assert resumed["method"] == method


@pytest.mark.parametrize("change", [
    {"method": "unknown"},
    {"umbrella": {"stiffness": 2}},
    {"sigma": 1.0},
    {"metadynamics": {"barrier": 1.0}},
])
def test_cross_method_and_misplaced_fields_are_rejected(change):
    data = job("metadynamics")
    data["fes"].update(change)
    with pytest.raises(InputError):
        JobSpec.from_mapping(data).resolve()


def test_duplicate_common_and_method_tables_are_rejected():
    for field in ("sampling", "metadynamics"):
        data = job("metadynamics")
        data[field] = copy.deepcopy(data["fes"][field])
        with pytest.raises(InputError, match="not both"):
            JobSpec.from_mapping(data).resolve()


@pytest.mark.parametrize("extra", [
    {"umbrella": {"stiffness": 1.0}},
    {"height": 0.1},
    {"metadynamics": {"height": 0.1}},
])
def test_direct_resolved_entry_does_not_drop_fes_parameters(extra, tmp_path):
    config, system = JobSpec.from_mapping(job("metadynamics")).resolve()
    config["fes"].update(extra)
    with pytest.raises(InputError):
        run_resolved(config, system, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("section, value", [
    ("umbrella", {"centers": [0.0]}),
    ("metadynamics", {"barrier": 3.0}),
    ("sampling", {"replcias": 5}),
    ("analysis", {"bootstrap_sample": 3}),
    ("adaptive_umbrella", {}),
    ("replica_exchange", {}),
])
def test_direct_resolved_entry_rejects_cross_method_and_unknown_fields(section, value, tmp_path):
    config, system = JobSpec.from_mapping(job("metadynamics")).resolve()
    config.setdefault(section, {}).update(value)
    with pytest.raises(InputError):
        run_resolved(config, system, output=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_direct_resolved_entry_can_flatten_one_unambiguous_method_subtable(tmp_path):
    config, system = JobSpec.from_mapping(job("metadynamics")).resolve()
    config["fes"]["metadynamics"] = config.pop("metadynamics")
    result = run_resolved(config, system, output=tmp_path / "run")
    assert result["method"] == "metadynamics"


@pytest.mark.parametrize("section, typo, correct", [
    (None, "methd", "method"),
    ("sampling", "production_step", "production_steps"),
    ("metadynamics", "heigt", "height"),
])
def test_nested_fes_fields_preserve_actionable_typo_suggestions(section, typo, correct):
    data = job("metadynamics")
    table = data["fes"] if section is None else data["fes"][section]
    table[typo] = table.pop(correct)
    with pytest.raises(InputError, match=f"Did you mean.*'{correct}'"):
        JobSpec.from_mapping(data).resolve()
    assert typo in table and correct not in table


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
@pytest.mark.parametrize("field", ["thermostat_params", "native"])
def test_nested_fes_preserves_provider_preflight(method, field, tmp_path, monkeypatch):
    data = job(method)
    if field == "thermostat_params":
        data["fes"]["sampling"][field] = []
        message = "sampling.thermostat_params must be a table"
    else:
        data["dynamics"][field] = "auto"
        message = "dynamics.native must be"
    before = copy.deepcopy(data)
    monkeypatch.setattr(
        "pyrpmd.parallel.execution.make_task_potential",
        lambda *args, **kwargs: pytest.fail("invalid input must fail before potential creation"),
    )
    with pytest.raises(InputError, match=message):
        run(data, output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    assert data == before


def test_direct_resolved_umbrella_uses_full_analysis_preflight(tmp_path, monkeypatch):
    config, system = JobSpec.from_mapping(job("umbrella")).resolve()
    config["analysis"]["bins"] = 1
    monkeypatch.setattr("pyrpmd.potentials.make_potential", lambda *args, **kwargs: pytest.fail("invalid analysis must fail before potential creation"))
    with pytest.raises(RuntimeError, match="analysis.bins"):
        run_resolved(config, system, output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
def test_old_run_manifest_and_streams_migrate_without_changing_samples(method, tmp_path):
    directory = tmp_path / method
    run(job(method), output=directory)
    config_path = directory / "input/resolved.json"
    config = json.loads(config_path.read_text())
    config.pop("fes")
    config["task"] = "fes" if method == "umbrella" else method
    system = System.from_dict(
        json.loads((directory / "input/system.json").read_text())
    )
    write_json(config_path, config)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["physics_fingerprint"] = fingerprint(scientific_config(config), system)
    write_json(manifest_path, manifest)
    before = {}
    for path in (directory / "checkpoints").glob("*.npz"):
        with load_checkpoint(path) as saved:
            arrays = {name: saved[name].copy() for name in saved.files if not name.startswith("_")}
        if method == "umbrella":
            window, replica = int(path.stem.split("_")[1]), int(path.stem.split("_")[3])
            extras = {"window": window, "replica": replica}
            # Umbrella stream identity also includes center/stiffness; read the
            # exact historical identity through the original stream validator.
            from pyrpmd.workflows.fes import resolve_umbrellas
            _, _, centers, stiffness = resolve_umbrellas(config, system)
            extras.update(center=centers[window], stiffness=stiffness)
        else:
            replica = int(path.stem.split("_")[-1])
            extras = {f"{method}_replica": replica}
        arrays["fingerprint"] = _legacy_fingerprint(config, system, extras)
        before[path.name] = arrays["values" if method == "umbrella" else "samples"].copy()
        atomic_savez(path, kind="arrays", **arrays)
    resumed = resume_run(directory)
    assert resumed["method"] == method
    assert json.loads(config_path.read_text())["fes"]["method"] == method
    for path in (directory / "checkpoints").glob("*.npz"):
        with load_checkpoint(path) as saved:
            np.testing.assert_array_equal(saved["values" if method == "umbrella" else "samples"], before[path.name])
    changed = copy.deepcopy(job(method))
    changed["seed"] += 1
    with pytest.raises(RuntimeError, match="changed scientific"):
        run(changed, output=directory, resume=True)
