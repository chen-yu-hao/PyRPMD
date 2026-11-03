"""Hybrid IBI production, explicit molecular identity and exact continuation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.io import file_sha256, fingerprint, referenced_files
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.qcmd import FQCMDModel, HCMDModel, TriatomicQuasiCentroid, fqcmd_base_identity
from pyrpmd.system import System
from pyrpmd.workflows.spectroscopy import (
    _trace_physics, prepare_spectroscopy, reanalyze_spectroscopy,
    resolve_spectroscopy, run_spectroscopy,
)
from test_spectroscopy_workflow import job


def write_model(config, system, *, strength=.001, partitions=None, provenance=None):
    geometry = TriatomicQuasiCentroid(((0, 1, 2), (3, 4, 5)), system.masses)
    bond, angle = np.linspace(.5, 5., 17), np.linspace(.1, 3.1, 17)
    edges = (bond, angle, bond, angle)
    centers = [(.5 * (grid[1:] + grid[:-1])) for grid in edges]
    corrections = [strength * (center - (1.8 if i % 2 == 0 else 1.82)) ** 2
                   for i, center in enumerate(centers)]
    fit = FQCMDModel(
        geometry, [[0, 0, 1], [2, 2, 3]], edges, corrections,
        beta=config["beta"], base_identity=fqcmd_base_identity(config["potential"], system),
        provenance={"purpose": "h-CMD workflow regression", "training_beads": 8,
                    **(provenance or {})},
    )
    model = HCMDModel(fit, partitions or {"f-qcmd": [0], "f-cmd": [1]})
    model.save(config["spectroscopy"]["hcmd_model_file"])
    return model


def hcmd_job(tmp_path, *, strength=.001):
    config, _ = job(tmp_path, "h-cmd")
    config["beads"] = 1
    config["spectroscopy"]["hcmd_model_file"] = str(tmp_path / "hcmd.npz")
    molecule = np.array([[0., 0., 0.], [1.8, 0., 0.], [-.45, 1.74, 0.]])
    system = System(("O", "H", "H") * 2, np.tile([16., 1., 1.], 2),
                    np.concatenate((molecule, molecule + [0., 0., 4.])))
    write_model(config, system, strength=strength)
    return config, system


def test_zero_correction_matches_classical_and_records_explicit_partitions(tmp_path):
    config, system = hcmd_job(tmp_path, strength=0.)
    result = run_spectroscopy(config, system, tmp_path / "hybrid")
    assert result["method"] == "h-cmd"
    assert result["beads"] == 1
    assert result["converged"] is result["physical_converged"] is False
    assert result["production_centroid_thermostat"] is False
    assert "hybrid" in result["property_estimator"]
    assert result["hcmd_partitions"] == {"f-qcmd": [0], "f-cmd": [1]}
    assert result["correction_scope"] == "separable-intramolecular-IBI"
    digest = file_sha256(config["spectroscopy"]["hcmd_model_file"])
    assert result["hcmd_model_sha256"] == digest
    settings = result["trajectory_provenance"]["settings"]["spectroscopy_trajectory"]
    assert settings["hcmd_model_sha256"] == digest
    assert settings["hcmd_partitions"] == result["hcmd_partitions"]
    assert result["trajectory_provenance"]["source_files"][str(tmp_path / "hcmd.npz")] == digest
    classical = copy.deepcopy(config)
    classical["spectroscopy"]["method"] = "rpmd"
    del classical["spectroscopy"]["hcmd_model_file"]
    reference = run_spectroscopy(classical, system, tmp_path / "classical")
    replay = run_spectroscopy(config, system, tmp_path / "hybrid")
    with load_checkpoint(result["files"]["traces"]) as actual, \
            load_checkpoint(reference["files"]["traces"]) as expected, \
            load_checkpoint(replay["files"]["traces"]) as resumed:
        for key in ("dipole", "polarizability"):
            np.testing.assert_array_equal(actual[key], expected[key])
            np.testing.assert_array_equal(actual[key], resumed[key])


@pytest.mark.parametrize("failure_call", [12, 39])
def test_interrupted_sampling_and_production_restore_exact_state_rng_and_traces(
        tmp_path, monkeypatch, failure_call):
    import pyrpmd.workflows.spectroscopy as workflow

    config, system = hcmd_job(tmp_path)
    config["sampling"]["replicas"] = 1
    config["output"]["checkpoint_steps"] = 2
    original = workflow.make_potential

    class Failure:
        def __init__(self, wrapped):
            self.wrapped, self.calls = wrapped, 0
            self.metadata = wrapped.metadata

        def evaluate(self, q):
            self.calls += 1
            if self.calls == failure_call:
                raise RuntimeError("interrupted h-CMD")
            return self.wrapped.evaluate(q)

        def close(self):
            self.wrapped.close()

    monkeypatch.setattr(workflow, "make_potential", lambda *args: Failure(original(*args)))
    output = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="interrupted h-CMD"):
        run_spectroscopy(config, system, output)
    checkpoint = output / "checkpoints" / "spectroscopy_0000.npz"
    assert checkpoint.exists()
    with load_checkpoint(checkpoint) as saved:
        if failure_call == 12:
            assert int(saved["step"]) < config["sampling"]["equilibration_steps"]
        else:
            assert int(saved["step"]) > config["sampling"]["equilibration_steps"]
    monkeypatch.setattr(workflow, "make_potential", original)
    actual = run_spectroscopy(config, system, output)
    reference = run_spectroscopy(config, system, tmp_path / "reference")
    for name, keys in (("traces", ("dipole", "polarizability")),
                       ("ir", ("per_replica",)), ("raman", ("per_replica",)),
                       ("sfg", ("per_replica_susceptibility",))):
        with load_checkpoint(actual["files"][name]) as a, load_checkpoint(reference["files"][name]) as b:
            for key in keys:
                np.testing.assert_array_equal(a[key], b[key])
    with load_checkpoint(actual["files"]["checkpoints"][0]) as a, \
            load_checkpoint(reference["files"]["checkpoints"][0]) as b:
        for key in ("q", "p", "rng", "sample_steps", "step", "time"):
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("change", ["partition", "correction", "provenance", "beta", "mass", "pes"])
def test_changed_science_rejects_even_completed_checkpoint(tmp_path, change):
    config, system = hcmd_job(tmp_path)
    output = tmp_path / "run"
    run_spectroscopy(config, system, output)
    if change == "partition":
        write_model(config, system, partitions={"f-qcmd": [1], "f-cmd": [0]})
    elif change == "correction":
        write_model(config, system, strength=.02)
    elif change == "provenance":
        write_model(config, system, provenance={"training_beads": 16})
    elif change == "beta":
        config["beta"] *= 2
    elif change == "mass":
        system = System(system.symbols, system.masses * 2, system.geometry)
    else:
        config["potential"]["params"]["k"] = .7
    with pytest.raises(ValueError, match="checkpoint physics or property-model content changed"):
        run_spectroscopy(config, system, output)


def test_model_changed_during_run_cannot_publish_misattributed_spectra(tmp_path, monkeypatch):
    import pyrpmd.workflows.spectroscopy as workflow

    config, system = hcmd_job(tmp_path)
    config["sampling"]["replicas"] = 1
    original = workflow._sample_replica

    def replace_after_sampling(*args):
        streams = original(*args)
        write_model(config, system, partitions={"f-qcmd": [1], "f-cmd": [0]})
        return streams

    monkeypatch.setattr(workflow, "_sample_replica", replace_after_sampling)
    output = tmp_path / "changed"
    with pytest.raises(ValueError, match="model content changed during"):
        run_spectroscopy(config, system, output)
    assert not (output / "results").exists()


def test_offline_reanalysis_preserves_partition_and_trace_without_model_providers(tmp_path):
    config, system = hcmd_job(tmp_path)
    output = tmp_path / "run"
    initial = run_spectroscopy(config, system, output)
    trace = Path(initial["files"]["traces"])
    original = trace.read_bytes()
    Path(config["spectroscopy"]["hcmd_model_file"]).unlink()
    (tmp_path / "properties_model.py").unlink()
    config["potential"] = {"provider": "not_installed"}
    config["spectroscopy"].update(spectra=["ir"], window="boxcar", normalize=False)
    result = reanalyze_spectroscopy(config, system, output)
    assert result["trajectory_provenance"] == initial["trajectory_provenance"]
    assert result["hcmd_model_sha256"] == initial["hcmd_model_sha256"]
    assert result["hcmd_partitions"] == initial["hcmd_partitions"]
    assert trace.read_bytes() == original
    config["spectroscopy"]["hcmd_model_file"] = str(tmp_path / "other.npz")
    with pytest.raises(ValueError, match="physics or system identity"):
        reanalyze_spectroscopy(config, system, output)


@pytest.mark.parametrize("method", ["f-qcmd", "h-cmd"])
@pytest.mark.parametrize("replace_in", ["pes", "properties"])
def test_numerical_validation_cannot_report_an_unvalidated_model_digest(
        tmp_path, monkeypatch, method, replace_in):
    import pyrpmd.workflows.spectroscopy as workflow
    from test_fqcmd_workflow import fqcmd_job, write_model as write_fqcmd

    if method == "h-cmd":
        config, system = hcmd_job(tmp_path)

        def replace():
            write_model(config, system, partitions={"f-qcmd": [1], "f-cmd": [0]})
    else:
        config, system = fqcmd_job(tmp_path)

        def replace():
            write_fqcmd(config, system, shift=.02)

    class ReplacingPotential(HarmonicPotential):
        def evaluate(self, q):
            result = super().evaluate(q)
            replace()
            return result

    if replace_in == "properties":
        original = workflow._make_properties

        class ReplacingProperties:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def evaluate(self, q):
                result = self.wrapped.evaluate(q)
                replace()
                return result

            def close(self):
                self.wrapped.close()

        monkeypatch.setattr(workflow, "_make_properties", lambda *args: ReplacingProperties(original(*args)))
    potential = ReplacingPotential(.3) if replace_in == "pes" else HarmonicPotential(.3)
    with pytest.raises(ValueError, match="model content changed during numerical"):
        prepare_spectroscopy(config, system, potential=potential)


@pytest.mark.parametrize("change", ["beta", "mass", "pes", "outside"])
def test_numerical_prepare_checks_model_compatibility_and_support(tmp_path, change):
    config, system = hcmd_job(tmp_path)
    structural = prepare_spectroscopy(config, system)
    assert "hcmd_model_sha256" not in structural
    checked = prepare_spectroscopy(config, system, potential=HarmonicPotential(.3))
    assert checked["hcmd_model_sha256"] == file_sha256(tmp_path / "hcmd.npz")
    assert checked["hcmd_partitions"] == {"f-qcmd": [0], "f-cmd": [1]}
    assert checked["observable_approximation"] is True
    if change == "beta":
        config["beta"] *= 2
    elif change == "mass":
        system = System(system.symbols, system.masses * 2, system.geometry)
    elif change == "pes":
        config["potential"]["params"]["k"] = .7
    else:
        outside = system.geometry.copy()
        outside[1] *= 4
        system = System(system.symbols, system.masses, outside)
    with pytest.raises(ValueError):
        prepare_spectroscopy(config, system, potential=HarmonicPotential(.3))
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("section,key,value", [
    ("spectroscopy", "method", "qcmd"),
    ("spectroscopy", "method", "f-qcmd"),
    ("spectroscopy", "method", "f-cmd"),
    ("spectroscopy", "method", "rpmd"),
    ("spectroscopy", "hcmd_model_file", ""),
    ("spectroscopy", "hcmd_model_file", True),
    ("spectroscopy", "fqcmd_model_file", "other.npz"),
    ("spectroscopy", "target_frequency", 10.),
    ("spectroscopy", "friction_scale", .5),
    ("spectroscopy", "integrator", "baoab"),
    ("dynamics", "loop_backend", "native"),
    ("sampling", "engine", "native"),
    ("sampling", "engine", "torch"),
    ("parallel", "trajectory_workers", 2),
])
def test_unsupported_combinations_fail_explicitly(tmp_path, section, key, value):
    config, system = hcmd_job(tmp_path)
    config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        resolve_spectroscopy(config, system)


@pytest.mark.parametrize("change", ["beads", "periodic", "cell", "fixed", "constraint", "missing"])
def test_unsupported_systems_and_missing_model(tmp_path, change):
    config, system = hcmd_job(tmp_path)
    if change == "beads":
        config["beads"] = 8
    elif change in ("periodic", "cell"):
        system = System(system.symbols, system.masses, system.geometry, cell=np.eye(3) * 20.,
                        pbc=(True, True, True) if change == "periodic" else (False, False, False))
    elif change == "fixed":
        system = System(system.symbols, system.masses, system.geometry, fixed_atoms=(0,))
    elif change == "constraint":
        system = System(system.symbols, system.masses, system.geometry, bond_constraints=((0, 1, 1.8),))
    else:
        del config["spectroscopy"]["hcmd_model_file"]
    with pytest.raises(ValueError):
        resolve_spectroscopy(config, system)


def test_corrupt_model_fails_before_output(tmp_path):
    config, system = hcmd_job(tmp_path)
    Path(config["spectroscopy"]["hcmd_model_file"]).write_bytes(b"not an h-CMD model")
    with pytest.raises(ValueError):
        run_spectroscopy(config, system, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_input_and_cli_validate_run_resume_offline_analysis(tmp_path, capsys):
    from pyrpmd.cli import main
    from pyrpmd.spectroscopy_input import resolve_spectroscopy_job
    from test_fqcmd_workflow import raw_fqcmd_job

    raw = raw_fqcmd_job(tmp_path)
    molecule = np.asarray(raw["system"]["geometry"])
    raw["system"]["symbols"] *= 2
    raw["system"]["geometry"] = np.concatenate((molecule, molecule + [0., 0., 4.])).tolist()
    raw["spectroscopy"] = {"method": "h-cmd", "hcmd_model_file": "hcmd.npz", "spectra": ["ir"]}
    resolved, system = resolve_spectroscopy_job(raw, base_directory=tmp_path)
    assert resolved["spectroscopy"]["hcmd_model_file"] == str(tmp_path / "hcmd.npz")
    write_model(resolved, system)
    assert str(tmp_path / "hcmd.npz") in referenced_files(resolved)
    path, output = tmp_path / "job.py", tmp_path / "cli"
    path.write_text("def build():\n    return " + repr(raw) + "\n")
    for args in (["validate", str(path), "--check-pes", "--output", str(output)],
                 ["run", str(path), "--output", str(output)], ["resume", str(output)],
                 ["analyze", str(output), "--recompute"]):
        assert main(args) == 0
        capsys.readouterr()
    summary = json.loads((output / "results" / "summary.json").read_text())
    assert summary["method"] == "h-cmd"
    assert summary["hcmd_partitions"] == {"f-qcmd": [0], "f-cmd": [1]}
    assert summary["physical_converged"] is False


def test_existing_fqcmd_trace_identity_stays_byte_compatible():
    # Captured from unmodified main ceee59c before the T20 implementation.
    config = {
        "task": "spectroscopy", "beta": 3.1, "beads": 1, "seed": 57, "dt_au": .01,
        "potential": {"provider": "harmonic", "params": {"k": .3}},
        "properties": {"provider": "python", "factory": "unloaded_module:build"},
        "spectroscopy": {"method": "f-qcmd", "fqcmd_model_file": "/fixed/fqcmd-model.npz"},
        "sampling": {"equilibration_steps": 10, "production_steps": 80, "sample_interval": 2,
                     "replicas": 2, "gamma_centroid": .7, "dt_au": .015},
        "output": {"checkpoint_steps": 7},
    }
    system = System(("O", "H", "H"), np.array([16., 1., 1.]),
                    np.array([[0., 0., 0.], [1.8, 0., 0.], [-.45, 1.74, 0.]]))
    options = resolve_spectroscopy(config, system)
    # The path is already normalized in a saved old input. Keep this fixture
    # independent of the host's drive prefix when running on Windows.
    options["fqcmd_model_file"] = "/fixed/fqcmd-model.npz"
    options["fqcmd_model_sha256"] = "f" * 64
    assert fingerprint(_trace_physics(config, options), system) == (
        "a6bb4ef8559ec025706d22310f814e16af81201620a3376e029b285cf361ebc1"
    )


def test_cli_info_advertises_available_hcmd_without_claiming_realtime_qcmd(capsys):
    from pyrpmd.cli import main

    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert {"f-qcmd", "h-cmd"} <= set(info["spectroscopy_methods"])
    assert "h-cmd" not in info["unimplemented_methods"]
    assert "qcmd" in info["unimplemented_methods"]
