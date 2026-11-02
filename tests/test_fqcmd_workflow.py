"""Fitted quasi-centroid workflow identity, exact continuation and refusals."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from examples.qcmd.water_monomer import MONOMER_MASSES, geometry
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.qcmd import QCMDModel, qcmd_base_identity
from pyrpmd.spectroscopy_input import resolve_spectroscopy_job
from pyrpmd.system import System
from pyrpmd.workflows.spectroscopy import (
    prepare_spectroscopy, reanalyze_spectroscopy, resolve_spectroscopy, run_spectroscopy,
)


def job(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "qcmd" / "water_monomer.py"
    module = tmp_path / "monomer.py"
    module.write_text(source.read_text())
    system = System(("O", "H", "H"), MONOMER_MASSES, geometry())
    config = {
        "task": "spectroscopy", "beta": 1 / (KB_AU * 150.), "beads": 1, "seed": 79, "dt_au": 1.,
        "potential": {"provider": "python", "factory": str(module) + ":build"},
        "properties": {"provider": "python", "factory": str(module) + ":build_properties"},
        "spectroscopy": {"method": "f-qcmd", "spectra": ["ir"],
                         "qcmd_model_file": str(tmp_path / "model.npz")},
        "sampling": {"equilibration_steps": 10, "production_steps": 80, "sample_interval": 2,
                     "replicas": 2, "gamma_centroid": .01, "dt_au": 1.5},
        "output": {"checkpoint_steps": 7},
    }
    edges = (np.linspace(1.3, 2.5, 41), np.linspace(1., 2.8, 37))
    r, theta = ((e[1:] + e[:-1]) / 2 for e in edges)
    model = QCMDModel(
        [(0, 1, 2)], system.masses, config["beta"], edges,
        (.003 * (r - 1.8)**2, .001 * (theta - 1.9)**2),
        base_identity=qcmd_base_identity(config["potential"], system), training_beads=8,
        training={"test_fixture": "synthetic correction, not a fitted physical result"},
    )
    model.save(config["spectroscopy"]["qcmd_model_file"])
    return config, system, model


def test_fqcmd_propagates_records_full_fit_provenance_and_offline_reanalyzes(tmp_path):
    config, system, model = job(tmp_path)
    out = tmp_path / "run"
    result = run_spectroscopy(config, system, out)
    assert result["method"] == "f-qcmd"
    assert result["beads"] == 1 and not result["physical_converged"]
    assert not result["converged"]
    assert result["trajectory_provenance"]["quasicentroid_model"] == model.to_dict()
    assert "quasi-centroid" in result["property_estimator"]
    resumed = run_spectroscopy(config, system, out)
    with load_checkpoint(result["files"]["checkpoints"][0]) as data:
        assert data["step"] == 90
        assert data["time"] == pytest.approx(10 * 1.5 + 80.)
        np.testing.assert_array_equal(data["sample_steps"], np.arange(0, 81, 2))
    trace_path = Path(resumed["files"]["traces"])
    original = trace_path.read_bytes()
    Path(config["spectroscopy"]["qcmd_model_file"]).unlink()
    (tmp_path / "monomer.py").unlink()
    config["potential"] = {"provider": "not_installed"}
    config["spectroscopy"]["window"] = "boxcar"
    reanalyzed = reanalyze_spectroscopy(config, system, out)
    assert reanalyzed["trajectory_provenance"] == result["trajectory_provenance"]
    assert trace_path.read_bytes() == original


@pytest.mark.parametrize("fail_after", [11, 39, 130])
def test_fqcmd_interruption_replays_rng_phase_and_trace_exactly(tmp_path, monkeypatch, fail_after):
    import pyrpmd.workflows.spectroscopy as workflow
    config, system, _ = job(tmp_path)
    factory = workflow.make_potential

    class Failure:
        def __init__(self, base):
            self.base = base
            self.metadata = base.metadata
            self.calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == fail_after:
                raise RuntimeError("injected f-QCMD interruption")
            return self.base.evaluate(q)

        def close(self):
            self.base.close()

    monkeypatch.setattr(workflow, "make_potential", lambda *a: Failure(factory(*a)))
    with pytest.raises(RuntimeError, match="injected"):
        run_spectroscopy(config, system, tmp_path / "interrupted")
    monkeypatch.setattr(workflow, "make_potential", factory)
    actual = run_spectroscopy(config, system, tmp_path / "interrupted")
    reference = run_spectroscopy(config, system, tmp_path / "reference")
    for suffix in ("traces", "ir"):
        with load_checkpoint(actual["files"][suffix]) as a, load_checkpoint(reference["files"][suffix]) as b:
            for field in ("dipole",) if suffix == "traces" else ("per_replica", "relative_ir"):
                np.testing.assert_array_equal(a[field], b[field])
    for ap, bp in zip(actual["files"]["checkpoints"], reference["files"]["checkpoints"]):
        with load_checkpoint(ap) as a, load_checkpoint(bp) as b:
            for field in ("q", "p", "rng", "time", "step", "sample_steps"):
                np.testing.assert_array_equal(a[field], b[field])


@pytest.mark.parametrize("change", ["correction", "training", "base_source", "mass", "beta"])
def test_fqcmd_restart_rejects_scientific_changes_before_rewriting_trace(tmp_path, change):
    config, system, model = job(tmp_path)
    output = tmp_path / "run"
    result = run_spectroscopy(config, system, output)
    path = Path(result["files"]["traces"])
    original = path.read_bytes()
    if change in ("correction", "training"):
        correction = [x.copy() for x in model.corrections]
        training = model.training
        if change == "correction":
            correction[0][2] += .0001
        else:
            training["regularization"] = .2
        replacement = QCMDModel(
            model.molecules, model.masses, model.beta, model.grids, correction,
            base_identity=model.base_identity, training_beads=model.training_beads,
            training=training, target_counts=model.target_counts, diagnostics=model.diagnostics,
        )
        replacement.save(config["spectroscopy"]["qcmd_model_file"])
    elif change == "base_source":
        with (tmp_path / "monomer.py").open("a") as stream:
            stream.write("\n# changed physical model source\n")
    elif change == "mass":
        system = System(system.symbols, system.masses * 1.01, system.geometry)
    else:
        config["beta"] *= 1.01
    with pytest.raises(ValueError, match="identity|content|temperature|masses"):
        run_spectroscopy(config, system, output)
    assert path.read_bytes() == original


@pytest.mark.parametrize("section,key,value", [
    ("top", "beads", 8), ("spectroscopy", "target_frequency", 10.),
    ("spectroscopy", "friction_scale", 1.), ("spectroscopy", "integrator", "baoab"),
    ("spectroscopy", "qcmd_model_file", ""), ("spectroscopy", "method", "h-cmd"),
    ("spectroscopy", "method", "qcmd"), ("sampling", "thermostat", "gle"),
    ("dynamics", "loop_backend", "native"),
])
def test_fqcmd_unsupported_combinations_fail_explicitly(tmp_path, section, key, value):
    config, system, _ = job(tmp_path)
    (config if section == "top" else config.setdefault(section, {}))[key] = value
    with pytest.raises(ValueError):
        resolve_spectroscopy(config, system)


def test_model_field_cannot_change_other_spectroscopy_methods(tmp_path):
    config, system, _ = job(tmp_path)
    for method in ("rpmd", "trpmd", "pa-cmd", "f-cmd"):
        config["spectroscopy"]["method"] = method
        with pytest.raises(ValueError, match="only to f-QCMD"):
            resolve_spectroscopy(config, system)


def test_fqcmd_frontend_units_model_file_and_public_cli(tmp_path, capsys):
    from pyrpmd.cli import main
    config, system, _ = job(tmp_path)
    raw = {
        "task": "spectroscopy", "temperature": "150 K", "beads": 1, "seed": 79,
        "system": {"symbols": list(system.symbols), "geometry": system.geometry.tolist(),
                   "geometry_unit": "bohr"},
        "potential": {"provider": "python", "factory": "monomer.py:build"},
        "properties": {"provider": "python", "factory": "monomer.py:build_properties"},
        "spectroscopy": {"method": "f-qcmd", "qcmd_model_file": "model.npz"},
        "dynamics": {"timestep": "0.02 fs"},
        "sampling": {"equilibration_steps": 5, "production_steps": 16, "sample_interval": 2},
        "output": {"checkpoint_steps": 3},
    }
    resolved, resolved_system = resolve_spectroscopy_job(raw, base_directory=tmp_path)
    assert resolved["spectroscopy"]["qcmd_model_file"] == str(tmp_path / "model.npz")
    np.testing.assert_array_equal(resolved_system.masses, system.masses)
    source = tmp_path / "job.py"
    source.write_text("def build():\n    return " + repr(raw) + "\n")
    out = tmp_path / "cli"
    assert main(["validate", str(source), "--check-pes", "--output", str(out)]) == 0
    capsys.readouterr()
    assert main(["run", str(source), "--output", str(out)]) == 0
    capsys.readouterr()
    assert main(["resume", str(out)]) == 0
    capsys.readouterr()
    assert main(["analyze", str(out), "--recompute"]) == 0
    capsys.readouterr()
    summary = json.loads((out / "results" / "summary.json").read_text())
    assert summary["method"] == "f-qcmd"
    assert not summary["physical_converged"]


def test_fqcmd_prepare_checks_model_without_propagating_or_closing_base(tmp_path):
    from pyrpmd.potentials import make_potential
    config, system, model = job(tmp_path)
    structural = prepare_spectroscopy(config, system)
    assert "quasicentroid_model" not in structural
    with make_potential(config["potential"], system) as potential:
        checked = prepare_spectroscopy(config, system, potential=potential)
        assert checked["quasicentroid_model"] == model.to_dict()
        assert checked["property_check"]["input_shape"] == [1, 3, 3]
        assert np.isfinite(potential.evaluate(system.geometry[None]).energy).all()
    assert not (tmp_path / "checkpoints").exists()
