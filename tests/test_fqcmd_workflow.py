"""f-QCMD production, input identity, exact restart and offline reanalysis."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.io import file_sha256, fingerprint, referenced_files
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.qcmd import FQCMDModel, TriatomicQuasiCentroid, fqcmd_base_identity
from pyrpmd.system import System
from pyrpmd.workflows.spectroscopy import (
    _trace_physics, prepare_spectroscopy, reanalyze_spectroscopy,
    resolve_spectroscopy, run_spectroscopy,
)
from test_spectroscopy_workflow import job


def write_model(config, system, *, shift=0., provenance=None):
    geometry = TriatomicQuasiCentroid(((0, 1, 2),), system.masses)
    edges = (np.linspace(.5, 5., 17), np.linspace(.1, 3.1, 17))
    centers = [(x[1:] + x[:-1]) / 2 for x in edges]
    model = FQCMDModel(
        geometry, [[0, 0, 1]], edges,
        [shift * (centers[0] - 1.8) ** 2, np.zeros_like(centers[1])],
        beta=config["beta"], base_identity=fqcmd_base_identity(config["potential"], system),
        provenance={"purpose": "workflow regression", "training_beads": 8,
                    **(provenance or {})},
    )
    model.save(config["spectroscopy"]["fqcmd_model_file"])
    return model


def fqcmd_job(tmp_path):
    config, _ = job(tmp_path, "f-qcmd")
    config["beads"] = 1
    config["spectroscopy"]["fqcmd_model_file"] = str(tmp_path / "fqcmd.npz")
    system = System(
        ("O", "H", "H"), np.array([16., 1., 1.]),
        np.array([[0., 0., 0.], [1.8, 0., 0.], [-.45, 1.74, 0.]]),
    )
    write_model(config, system)
    return config, system


def test_fqcmd_zero_correction_matches_classical_trajectory_and_records_model(tmp_path):
    config, system = fqcmd_job(tmp_path)
    result = run_spectroscopy(config, system, tmp_path / "fqcmd")
    assert result["method"] == "f-qcmd"
    assert result["beads"] == 1
    assert result["converged"] is False
    assert result["production_centroid_thermostat"] is False
    assert "quasi-centroid" in result["property_estimator"]
    digest = file_sha256(config["spectroscopy"]["fqcmd_model_file"])
    assert result["fqcmd_model_sha256"] == digest
    settings = result["trajectory_provenance"]["settings"]["spectroscopy_trajectory"]
    assert settings["fqcmd_model_sha256"] == digest
    assert result["trajectory_provenance"]["source_files"][str(tmp_path / "fqcmd.npz")] == digest
    classical = copy.deepcopy(config)
    classical["spectroscopy"]["method"] = "rpmd"
    del classical["spectroscopy"]["fqcmd_model_file"]
    reference = run_spectroscopy(classical, system, tmp_path / "classical")
    resumed = run_spectroscopy(config, system, tmp_path / "fqcmd")
    with load_checkpoint(result["files"]["traces"]) as actual, \
            load_checkpoint(reference["files"]["traces"]) as expected, \
            load_checkpoint(resumed["files"]["traces"]) as replay:
        for field in ("dipole", "polarizability"):
            np.testing.assert_array_equal(actual[field], expected[field])
            np.testing.assert_array_equal(actual[field], replay[field])
        metadata = json.loads(str(actual["metadata"]))
        assert metadata["estimator"] == "property surface at quasi-centroid geometry"


def test_fqcmd_interrupted_restart_restores_full_phase_clock_and_rng(tmp_path, monkeypatch):
    import pyrpmd.workflows.spectroscopy as workflow

    config, system = fqcmd_job(tmp_path)
    config["sampling"]["replicas"] = 1
    original = workflow.make_potential

    class Failure:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.metadata = wrapped.metadata
            self.calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 39:
                raise RuntimeError("interrupted f-QCMD")
            return self.wrapped.evaluate(q)

        def close(self):
            self.wrapped.close()

    monkeypatch.setattr(workflow, "make_potential", lambda *args: Failure(original(*args)))
    with pytest.raises(RuntimeError, match="interrupted f-QCMD"):
        run_spectroscopy(config, system, tmp_path / "interrupted")
    monkeypatch.setattr(workflow, "make_potential", original)
    resumed = run_spectroscopy(config, system, tmp_path / "interrupted")
    reference = run_spectroscopy(config, system, tmp_path / "reference")
    with load_checkpoint(resumed["files"]["traces"]) as actual, \
            load_checkpoint(reference["files"]["traces"]) as expected:
        for field in ("dipole", "polarizability"):
            np.testing.assert_array_equal(actual[field], expected[field])
    with load_checkpoint(resumed["files"]["checkpoints"][0]) as actual, \
            load_checkpoint(reference["files"]["checkpoints"][0]) as expected:
        for field in ("q", "p", "rng", "sample_steps", "time", "step"):
            np.testing.assert_array_equal(actual[field], expected[field])


@pytest.mark.parametrize("change", ["correction", "training_provenance"])
def test_fqcmd_model_content_changes_invalidate_even_completed_checkpoint(tmp_path, change):
    config, system = fqcmd_job(tmp_path)
    output = tmp_path / "run"
    run_spectroscopy(config, system, output)
    if change == "correction":
        write_model(config, system, shift=.01)
    else:
        write_model(config, system, provenance={"training_beads": 16})
    with pytest.raises(ValueError, match="checkpoint physics or property-model content changed"):
        run_spectroscopy(config, system, output)


def test_fqcmd_model_changed_during_run_is_not_published_with_wrong_provenance(tmp_path, monkeypatch):
    import pyrpmd.workflows.spectroscopy as workflow

    config, system = fqcmd_job(tmp_path)
    config["sampling"]["replicas"] = 1
    original = workflow._sample_replica

    def replace_model_after_sampling(*args):
        result = original(*args)
        write_model(config, system, shift=.01)
        return result

    monkeypatch.setattr(workflow, "_sample_replica", replace_model_after_sampling)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="model content changed during"):
        run_spectroscopy(config, system, output)
    assert not (output / "results").exists()


def test_fqcmd_offline_reanalysis_uses_saved_model_and_observable_provenance(tmp_path):
    config, system = fqcmd_job(tmp_path)
    output = tmp_path / "run"
    initial = run_spectroscopy(config, system, output)
    trace = Path(initial["files"]["traces"])
    original_bytes = trace.read_bytes()
    Path(config["spectroscopy"]["fqcmd_model_file"]).unlink()
    (tmp_path / "properties_model.py").unlink()
    config["potential"] = {"provider": "not_installed"}
    config["spectroscopy"].update(spectra=["ir"], window="boxcar", normalize=False)
    result = reanalyze_spectroscopy(config, system, output)
    assert result["trajectory_provenance"] == initial["trajectory_provenance"]
    assert result["fqcmd_model_sha256"] == initial["fqcmd_model_sha256"]
    assert trace.read_bytes() == original_bytes
    config["spectroscopy"]["fqcmd_model_file"] = str(tmp_path / "another-model.npz")
    with pytest.raises(ValueError, match="physics or system identity"):
        reanalyze_spectroscopy(config, system, output)


@pytest.mark.parametrize("change", ["beta", "mass", "base_potential"])
def test_fqcmd_prepare_rejects_wrong_temperature_mass_and_base_pes(tmp_path, change):
    config, system = fqcmd_job(tmp_path)
    if change == "beta":
        config["beta"] *= 2
    elif change == "mass":
        system = System(system.symbols, system.masses * 2, system.geometry)
    else:
        config["potential"]["params"]["k"] = .7
    with pytest.raises(ValueError):
        prepare_spectroscopy(config, system, potential=HarmonicPotential(.3))


def test_fqcmd_numerical_prepare_checks_fitted_support(tmp_path):
    config, system = fqcmd_job(tmp_path)
    structural = prepare_spectroscopy(config, system)
    assert "fqcmd_model_sha256" not in structural
    checked = prepare_spectroscopy(config, system, potential=HarmonicPotential(.3))
    assert checked["fqcmd_model_sha256"] == file_sha256(tmp_path / "fqcmd.npz")
    assert checked["observable_approximation"] is True
    outside = system.geometry.copy()
    outside[1] *= 4
    invalid_system = System(system.symbols, system.masses, outside)
    with pytest.raises(ValueError, match="support|domain|range|outside"):
        prepare_spectroscopy(config, invalid_system, potential=HarmonicPotential(.3))
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("section,key,value", [
    ("spectroscopy", "method", "qcmd"),
    ("spectroscopy", "method", "h-cmd"),
    ("spectroscopy", "method", "rpmd"),
    ("spectroscopy", "fqcmd_model_file", ""),
    ("spectroscopy", "fqcmd_model_file", True),
    ("spectroscopy", "target_frequency", 10.),
    ("spectroscopy", "friction_scale", .5),
    ("spectroscopy", "integrator", "baoab"),
    ("dynamics", "loop_backend", "native"),
    ("sampling", "engine", "native"),
])
def test_fqcmd_rejects_unsupported_options(tmp_path, section, key, value):
    config, system = fqcmd_job(tmp_path)
    config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        resolve_spectroscopy(config, system)


@pytest.mark.parametrize("change", ["beads", "periodicity", "nonperiodic_cell", "fixed", "constraint", "missing_model"])
def test_fqcmd_rejects_incompatible_systems_and_missing_model(tmp_path, change):
    config, system = fqcmd_job(tmp_path)
    if change == "beads":
        config["beads"] = 8
    elif change == "periodicity":
        system = System(system.symbols, system.masses, system.geometry,
                        cell=np.eye(3) * 20., pbc=(True, True, True))
    elif change == "nonperiodic_cell":
        system = System(system.symbols, system.masses, system.geometry, cell=np.eye(3) * 20.)
    elif change == "fixed":
        system = System(system.symbols, system.masses, system.geometry, fixed_atoms=(0,))
    elif change == "constraint":
        system = System(system.symbols, system.masses, system.geometry,
                        bond_constraints=((0, 1, 1.8),))
    else:
        del config["spectroscopy"]["fqcmd_model_file"]
    with pytest.raises(ValueError):
        resolve_spectroscopy(config, system)


def test_fqcmd_rejects_malformed_model_before_writing_a_trajectory(tmp_path):
    config, system = fqcmd_job(tmp_path)
    Path(config["spectroscopy"]["fqcmd_model_file"]).write_bytes(b"not a valid model")
    with pytest.raises(ValueError):
        run_spectroscopy(config, system, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def raw_fqcmd_job(tmp_path):
    # Build only the property source; the model beta/masses come from JobSpec.
    job(tmp_path)
    return {
        "task": "spectroscopy", "beads": 1, "temperature": 300, "seed": 7,
        "system": {"symbols": ["O", "H", "H"], "geometry_unit": "bohr",
                   "geometry": [[0., 0., 0.], [1.8, 0., 0.], [-.45, 1.74, 0.]]},
        "potential": {"provider": "harmonic", "params": {"k": .0001}},
        "properties": {"provider": "python", "factory": "properties_model.py:build"},
        "spectroscopy": {"method": "f-qcmd", "fqcmd_model_file": "fqcmd.npz", "spectra": ["ir"]},
        "dynamics": {"timestep": "0.001 fs"},
        "sampling": {"equilibration_steps": 4, "production_steps": 20, "sample_interval": 1},
        "output": {"checkpoint_steps": 3},
    }


def test_fqcmd_input_and_cli_validate_run_resume_reanalysis(tmp_path, capsys):
    from pyrpmd.cli import main
    from pyrpmd.spectroscopy_input import resolve_spectroscopy_job

    raw = raw_fqcmd_job(tmp_path)
    resolved, system = resolve_spectroscopy_job(raw, base_directory=tmp_path)
    assert resolved["spectroscopy"]["fqcmd_model_file"] == str(tmp_path / "fqcmd.npz")
    write_model(resolved, system)
    assert str(tmp_path / "fqcmd.npz") in referenced_files(resolved)
    path = tmp_path / "job.py"
    path.write_text("def build():\n    return " + repr(raw) + "\n")
    output = tmp_path / "cli"
    assert main(["validate", str(path), "--check-pes", "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["run", str(path), "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["resume", str(output)]) == 0
    capsys.readouterr()
    assert main(["analyze", str(output), "--recompute"]) == 0
    capsys.readouterr()
    summary = json.loads((output / "results" / "summary.json").read_text())
    assert summary["method"] == "f-qcmd"


@pytest.mark.parametrize("method,expected", [
    ("pa-cmd", "4735d86ff6815cfcb801ea927ddcb74f745da2f7387360afe8c0efa9509cadbf"),
    ("f-cmd", "a5f681266db0106181698446ae89ee98de38c20e5ec1a7cd71e98dc034cf5165"),
    ("rpmd", "cf5cfd0236475f63dadf1edb1098ce3d6bd2838fac1bf30aa498d7cbec875d4d"),
    ("trpmd", "42084d37840dd0baf5aeafde70293460583b6f184f232fb0671494c55c9aee1c"),
])
def test_pre_t07_trace_identity_remains_byte_compatible(method, expected):
    # These hashes were captured from main 8cc402f before this extension.
    config = {
        "task": "spectroscopy", "beta": 3.1, "beads": 1 if method == "f-cmd" else 8,
        "seed": 57, "dt_au": .01, "potential": {"provider": "harmonic", "params": {"k": .3}},
        "properties": {"provider": "python", "factory": "unloaded_module:build"},
        "spectroscopy": {"method": method},
        "sampling": {"equilibration_steps": 10, "production_steps": 80, "sample_interval": 2,
                     "replicas": 2, "gamma_centroid": .7, "dt_au": .015},
        "output": {"checkpoint_steps": 7},
    }
    system = System(("H",), np.array([2.]), np.array([[.2, -.1, .3]]))
    assert fingerprint(_trace_physics(config, resolve_spectroscopy(config, system)), system) == expected
