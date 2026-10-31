"""Sampling contracts across public input, dispatch, restart and optional imports."""

import copy
import importlib
import json

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import analyze_run, prepare, resume_run, run


def opes_input():
    return {
        "task": "opes", "temperature": 1 / KB_AU, "beads": 2, "seed": 102,
        "system": {"symbols": ["H"], "masses": [1.0],
                   "geometry": [[0.1, 0.0, 0.0]], "geometry_unit": "bohr"},
        "potential": {"provider": "harmonic", "params": {"k": 0.2}},
        "dynamics": {"timestep": 0.02, "native": False},
        "sampling": {"equilibration_steps": 4, "production_steps": 24,
                     "interval_steps": 2, "andersen_interval": 4},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "opes": {"sigma": ["0.3 bohr"], "barrier": "1 hartree", "pace": 3,
                 "freeze_after_steps": 12, "freeze_equilibration_steps": 2},
        "output": {"checkpoint_steps": 5},
    }


def test_opes_public_default_and_frozen_run_resume_offline(tmp_path, monkeypatch):
    job = opes_input()
    with pytest.warns(FutureWarning, match="deprecated"):
        config, _ = JobSpec.from_mapping(job).resolve()
    assert config["task"] == "fes" and config["fes"]["method"] == "opes"
    assert config["analysis"]["method"] == "kde"
    assert config["analysis"]["measure"] == "cv_density"
    assert config["analysis"]["bootstrap"] == 0
    job["analysis"] = {"method": "reweighted_histogram", "bins": 4}
    assert "opes_preparation" in prepare(job)
    output = tmp_path / "run"
    result = run(job, output=output)
    assert result["task"] == "fes" and result["method"] == "opes"
    assert result["samples"] == 5
    assert result["converged"] is False
    assert result["estimator"] == "reweighted_histogram"
    checkpoint = output / "checkpoints" / "opes_0000.npz"
    before = checkpoint.read_bytes()
    with np.load(output / "fes.npz") as archive:
        original = {name: archive[name].copy() for name in archive.files}

    def unexpected_potential(*args, **kwargs):
        raise AssertionError("completed resume/offline analysis loaded a potential")

    monkeypatch.setattr("pyrpmd.workflows.opes.make_task_potential", unexpected_potential)
    assert resume_run(output)["samples"] == 5
    analyzed = analyze_run(output, recompute=True, bins=[4], bootstrap=0)
    assert analyzed["estimator"] == result["estimator"]
    assert checkpoint.read_bytes() == before
    with np.load(output / "fes.npz") as archive:
        for name, array in original.items():
            np.testing.assert_array_equal(archive[name], array)
    with pytest.raises((ValueError, RuntimeError), match="bootstrap"):
        analyze_run(output, recompute=True, bootstrap=3)
    changed = copy.deepcopy(job)
    changed["opes"]["barrier"] = "2 hartree"
    with pytest.raises((ValueError, RuntimeError), match="scientific"):
        run(changed, output=output, resume=True)


@pytest.mark.parametrize("section", ["umbrella", "reaction", "rate", "recrossing", "collision", "outcomes"])
def test_opes_rejects_irrelevant_bias_and_rate_inputs(section):
    job = opes_input()
    job[section] = {}
    message = "does not accept umbrella parameters" if section == "umbrella" else "OPES uses"
    with pytest.raises(InputError, match=message):
        JobSpec.from_mapping(job).resolve()


def test_opes_options_cannot_be_silently_attached_to_fes():
    job = opes_input()
    job["task"] = "fes"
    with pytest.raises(InputError, match="fes.method='umbrella' does not accept opes parameters"):
        JobSpec.from_mapping(job).resolve()


def test_sampling_exports_do_not_import_optional_plumed_runtime(monkeypatch):
    original = importlib.import_module

    def no_plumed(name, *args, **kwargs):
        if name == "plumed":
            raise AssertionError("importing PyRPMD must not load PLUMED")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", no_plumed)
    from pyrpmd import (OPESBias, PlumedBias, PlumedBiasResult,
                        TransitionInterfaceSampler, TISPath,
                        estimate_effective_flux, combine_tis_rate)

    assert all(callable(value) for value in (
        OPESBias, PlumedBias, PlumedBiasResult, TransitionInterfaceSampler,
        TISPath, estimate_effective_flux, combine_tis_rate,
    ))


def test_cli_reports_workflows_and_python_only_sampling_apis(capsys):
    from pyrpmd.cli import main

    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "opes" in info["workflows"]
    assert "replica_exchange_umbrella" in info["enhanced_sampling"]
    assert "transition_interface_sampling" in info["sampling_python_apis"]
    assert "plumed_centroid_bias" in info["sampling_python_apis"]
