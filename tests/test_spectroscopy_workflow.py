"""Actual serial propagation, properties, phase clocks, restart and reanalysis."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.system import System
from pyrpmd.workflows.spectroscopy import (resolve_spectroscopy, run_spectroscopy,
                                           reanalyze_spectroscopy)


def job(tmp_path, method="pa-cmd"):
    module = tmp_path / "properties_model.py"
    module.write_text('''import numpy as np
def build():
    def evaluate(q):
        mu = q.sum(axis=1)
        alpha = np.einsum("bi,bj->bij", mu, mu) + np.eye(3)[None]
        return {"dipole": mu, "polarizability": alpha}
    return evaluate
''')
    config = {
        "task": "spectroscopy", "beta": 3.1, "beads": 8, "seed": 57, "dt_au": .01,
        "potential": {"provider": "harmonic", "params": {"k": .3}},
        "properties": {"provider": "python", "factory": str(module) + ":build"},
        "spectroscopy": {"method": method, "spectra": ["ir", "raman", "sfg"], "max_lag": 10},
        "sampling": {"equilibration_steps": 10, "production_steps": 80, "sample_interval": 2,
                     "replicas": 2, "gamma_centroid": .7, "dt_au": .015},
        "output": {"checkpoint_steps": 7},
    }
    if method == "pa-cmd":
        config["spectroscopy"]["target_frequency"] = 20.
    system = System(("H",), np.array([2.]), np.array([[.2, -.1, .3]]))
    return config, system


@pytest.mark.parametrize("method", ["pa-cmd", "rpmd", "trpmd"])
def test_spectroscopy_workflow_outputs_resume_and_provider_free_reanalysis(tmp_path, method):
    config, system = job(tmp_path, method)
    out = tmp_path / "run"
    result = run_spectroscopy(config, system, out)
    assert result["status"] == "complete"
    assert result["method"] == method
    assert result["replicas"] == 2
    assert result["saved_frames_per_replica"] == 41
    assert not result["production_centroid_thermostat"]
    with load_checkpoint(result["files"]["traces"], require_integrity=True) as trace:
        assert trace["dipole"].shape == (2, 41, 3)
        assert trace["polarizability"].shape == (2, 41, 3, 3)
        original = trace["dipole"].copy()
    with load_checkpoint(result["files"]["checkpoints"][0], require_integrity=True) as checkpoint:
        assert checkpoint["step"] == 90
        assert float(checkpoint["time"]) == pytest.approx(10 * .015 + 80 * .01)
        np.testing.assert_array_equal(checkpoint["sample_steps"], np.arange(0, 81, 2))
    resumed = run_spectroscopy(config, system, out)
    with load_checkpoint(resumed["files"]["traces"], require_integrity=True) as trace:
        np.testing.assert_array_equal(trace["dipole"], original)
    # Offline analysis must need neither importable property/PES code nor a
    # network/model service. The saved observable history is the input.
    (tmp_path / "properties_model.py").unlink()
    config["potential"] = {"provider": "not_installed"}
    assert reanalyze_spectroscopy(config, system, out)["saved_frames_per_replica"] == 41


def test_interrupted_spectroscopy_matches_uninterrupted_checkpoint_and_trace(tmp_path, monkeypatch):
    import pyrpmd.workflows.spectroscopy as workflow
    config, system = job(tmp_path)
    ordinary_factory = workflow.make_potential

    class Failure:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 39:
                raise RuntimeError("interrupt spectroscopy")
            return self.wrapped.evaluate(q)

        def close(self):
            self.wrapped.close()

    monkeypatch.setattr(workflow, "make_potential", lambda *a: Failure(ordinary_factory(*a)))
    with pytest.raises(RuntimeError, match="interrupt spectroscopy"):
        run_spectroscopy(config, system, tmp_path / "interrupted")
    monkeypatch.setattr(workflow, "make_potential", ordinary_factory)
    actual = run_spectroscopy(config, system, tmp_path / "interrupted")
    reference = run_spectroscopy(config, system, tmp_path / "reference")
    for key in ("traces", "ir", "raman", "sfg"):
        with load_checkpoint(actual["files"][key]) as a, load_checkpoint(reference["files"][key]) as b:
            for field in ("dipole", "polarizability") if key == "traces" else ("per_replica_susceptibility",) if key == "sfg" else ("per_replica",):
                np.testing.assert_array_equal(a[field], b[field])


def test_property_model_content_change_invalidates_restart(tmp_path):
    config, system = job(tmp_path)
    out = tmp_path / "run"
    run_spectroscopy(config, system, out)
    with (tmp_path / "properties_model.py").open("a") as f:
        f.write("\n# new property model version\n")
    with pytest.raises(ValueError, match="property-model content changed"):
        run_spectroscopy(config, system, out)


def test_fc_cmd_requires_a_matching_fitted_pmf(tmp_path):
    config, system = job(tmp_path, "f-cmd")
    with pytest.raises(ValueError, match="beads=1"):
        resolve_spectroscopy(config, system)
    config["beads"] = 1
    with pytest.raises(ValueError, match="centroid_pmf_beta_au"):
        run_spectroscopy(config, system, tmp_path / "bad")


def test_fc_cmd_runs_temperature_tagged_conservative_pmf(tmp_path):
    config, system = job(tmp_path, "f-cmd")
    config["beads"] = 1
    factory = tmp_path / "centroid_pmf.py"
    factory.write_text('''import numpy as np
from pyrpmd.cmd import QuadraticCentroidPotential
def build():
    return QuadraticCentroidPotential(np.zeros((1,3)), np.zeros(3), np.eye(3)*.3, beta=3.1)
''')
    config["potential"] = {"provider": "python", "factory": str(factory) + ":build"}
    result = run_spectroscopy(config, system, tmp_path / "fcmd")
    assert result["method"] == "f-cmd"
    assert result["beads"] == 1


@pytest.mark.parametrize("section,key,value", [
    ("spectroscopy", "method", "qcmd"), ("spectroscopy", "spectra", ["sfg", "sfg"]),
    ("spectroscopy", "max_lag", 41), ("spectroscopy", "normalize", "yes"),
    ("dynamics", "loop_backend", "native"), ("dynamics", "integrator", "cayley-bcb"),
    ("parallel", "backend", "process"), ("properties", "provider", "unknown"),
    ("sampling", "thermostat", "andersen"),
])
def test_workflow_rejects_unsupported_or_inconsistent_options(tmp_path, section, key, value):
    config, system = job(tmp_path)
    config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        resolve_spectroscopy(config, system)


def raw_job(tmp_path):
    config, system = job(tmp_path)
    return {
        "task": "spectroscopy", "beads": 8, "temperature": 300, "seed": 7,
        "system": {"symbols": ["H"], "geometry": [[.2, 0., 0.]]},
        "potential": config["potential"],
        "properties": {"provider": "python", "factory": "properties_model.py:build"},
        "spectroscopy": {"method": "pa-cmd", "spectra": ["ir", "raman", "sfg"],
                         "target_frequency": "14000 cm^-1", "max_lag": 5},
        "dynamics": {"timestep": "0.01 fs"},
        "sampling": {"equilibration_steps": 4, "production_steps": 20, "sample_interval": 1},
        "output": {"checkpoint_steps": 3},
    }


def test_spectroscopy_frontend_resolves_units_defaults_and_property_sources(tmp_path):
    from pyrpmd.spectroscopy_input import resolve_spectroscopy_job
    from pyrpmd.constants import WAVENUMBER_TO_AU
    from pyrpmd.io import referenced_files

    raw = raw_job(tmp_path)
    raw["properties"]["params"] = {"model_file": "model.json", "label": "keep-this-label"}
    (tmp_path / "model.json").write_text("{}")
    resolved, system = resolve_spectroscopy_job(raw, base_directory=tmp_path)
    assert resolved["task"] == "spectroscopy"
    assert resolved["properties"]["factory"] == str(tmp_path / "properties_model.py") + ":build"
    assert resolved["properties"]["params"]["model_file"] == str(tmp_path / "model.json")
    assert resolved["properties"]["params"]["label"] == "keep-this-label"
    assert resolved["dynamics"]["integrator"] == "normal-mode-bab"
    assert resolved["sampling"]["integrator"] == "baoab"
    assert resolved["sampling"]["thermostat"] == "pile-l"
    assert resolve_spectroscopy(resolved, system)["target_frequency"] == pytest.approx(14000 * WAVENUMBER_TO_AU)
    assert str(tmp_path / "properties_model.py") in referenced_files(resolved)
    assert str(tmp_path / "model.json") in referenced_files(resolved)


@pytest.mark.parametrize("field,value", [
    ("path_integral", {"method": "suzuki-chin"}), ("active_learning", {}),
    ("barostat", {}), ("analysis", {}), ("typographical_error", 3),
])
def test_spectroscopy_frontend_does_not_strip_incompatible_science(tmp_path, field, value):
    from pyrpmd.spectroscopy_input import resolve_spectroscopy_job
    from pyrpmd.input import InputError
    data = raw_job(tmp_path)
    data[field] = value
    with pytest.raises(InputError):
        resolve_spectroscopy_job(data, base_directory=tmp_path)


def test_spectroscopy_public_cli_validate_run_resume_reanalyze(tmp_path, capsys):
    from pyrpmd.cli import main
    data = raw_job(tmp_path)
    path = tmp_path / "job.py"
    path.write_text("def build():\n    return " + repr(data) + "\n")
    output = tmp_path / "cli-run"
    assert main(["validate", str(path), "--check-pes", "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["run", str(path), "--output", str(output)]) == 0
    result = json.loads((output / "results" / "summary.json").read_text())
    assert result["task"] == "spectroscopy"
    assert result["method"] == "pa-cmd"
    assert main(["resume", str(output)]) == 0
    capsys.readouterr()
    assert main(["analyze", str(output), "--recompute"]) == 0
    capsys.readouterr()


def test_prepare_checks_full_ring_properties_without_propagation(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import prepare

    raw = raw_job(tmp_path)
    spec = JobSpec(raw, tmp_path)
    structural = prepare(spec)
    assert "property_check" not in structural["spectroscopy_preparation"]
    checked = prepare(spec, numerical=True)
    plan = checked["spectroscopy_preparation"]
    assert plan["property_check"]["input_shape"] == [8, 1, 3]
    assert plan["property_check"]["output_shapes"] == {
        "dipole": [8, 3], "polarizability": [8, 3, 3],
    }
    assert len(checked["pes_check"]["energy_hartree"]) == 8
    assert plan["production_centroid_thermostat"] is False
    assert not (tmp_path / "runs").exists()


def test_numerical_prepare_rejects_property_contract_and_closes_model(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import prepare

    raw = raw_job(tmp_path)
    marker = tmp_path / "closed.txt"
    (tmp_path / "properties_model.py").write_text('''import numpy as np
from pathlib import Path
class Model:
    def evaluate(self, q):
        return {"dipole": np.zeros((len(q), 2)), "polarizability": np.zeros((len(q), 3, 3))}
    def close(self):
        Path(__file__).with_name("closed.txt").write_text("closed")
def build():
    return Model()
''')
    spec = JobSpec(raw, tmp_path)
    assert prepare(spec)["ready"]
    assert not marker.exists()
    with pytest.raises(ValueError, match="dipole"):
        prepare(spec, numerical=True)
    assert marker.read_text() == "closed"


def test_numerical_prepare_rejects_unfitted_fcmd_potential(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import prepare

    raw = raw_job(tmp_path)
    raw["beads"] = 1
    raw["spectroscopy"]["method"] = "f-cmd"
    del raw["spectroscopy"]["target_frequency"]
    with pytest.raises(ValueError, match="centroid_pmf_beta_au"):
        prepare(JobSpec(raw, tmp_path), numerical=True)


def test_property_model_binding_failure_closes_allocated_provider(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import prepare

    raw = raw_job(tmp_path)
    (tmp_path / "properties_model.py").write_text('''from pathlib import Path
class Model:
    def bind_system(self, system):
        raise RuntimeError("incompatible model composition")
    def close(self):
        Path(__file__).with_name("closed.txt").write_text("closed")
def build():
    return Model()
''')
    with pytest.raises(RuntimeError, match="incompatible model composition"):
        prepare(JobSpec(raw, tmp_path), numerical=True)
    assert (tmp_path / "closed.txt").read_text() == "closed"
