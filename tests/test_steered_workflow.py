"""Public steering registration, units and authoritative trajectory reanalysis."""

import copy

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.input import JobSpec
from pyrpmd.runner import analyze_run, prepare, resume_run, run
from pyrpmd.system import System
from pyrpmd.workflows.steered import resolve_steered, run_steered, reanalyze_steered


def direct_config():
    return {
        "task": "fes", "fes": {"method": "steered"}, "beta": 1.3, "beads": 1,
        "dt_au": 0.03, "seed": 812, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 1.0}},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "steered": {"start": 0.0, "end": 0.8, "stiffness": 2.0, "direction": "both", "gamma": 1.0},
        "sampling": {"equilibration_steps": 5, "production_steps": 12, "replicas": 2,
                     "sample_interval": 3, "thermostat": "langevin", "integrator": "steered-symmetric"},
        "analysis": {"bootstrap": 10}, "output": {"checkpoint_steps": 4},
    }


def system():
    return System(("H",), np.ones(1), np.zeros((1, 3)))


def public_config():
    config = direct_config()
    config.pop("beta")
    config.pop("dt_au")
    config["temperature"] = 300.0
    config["dynamics"] = {"timestep": 0.03}
    config["system"] = {"symbols": ["H"], "masses": [1.0], "geometry": [[0, 0, 0]]}
    return config


def test_workflow_endpoint_artifact_resume_and_reanalysis_without_pes(tmp_path, monkeypatch):
    config = direct_config()
    result = run_steered(config, system(), tmp_path, return_data=True)
    summary = result["summary"]
    assert summary["method"] == "steered"
    assert summary["target"] == "restrained_endpoint_free_energy_difference"
    assert summary["status"] == "complete"
    assert len(summary["files"]["checkpoints"]) == 4
    assert set(summary["estimates"]) == {"forward", "reverse", "crooks"}
    assert not (tmp_path / "fes.npz").exists()
    with load_checkpoint(tmp_path / "steered.npz") as checkpoint:
        assert checkpoint["stream_0000_time"].shape == (5,)
        work = checkpoint["forward_work"]
    repeated = run_steered(config, system(), tmp_path)
    assert repeated == summary
    monkeypatch.setattr("pyrpmd.workflows.steered.make_task_potential",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("PES constructed during reanalysis")))
    recomputed = reanalyze_steered(config, system(), tmp_path, return_data=True)
    np.testing.assert_array_equal(recomputed["data"]["forward_work"], work)
    assert recomputed["summary"] == summary


def test_public_prepare_run_analyze_and_resume(tmp_path):
    config = public_config()
    assert "steered_preparation" in prepare(config)
    summary = run(config, output=tmp_path)
    assert summary["method"] == "steered"
    assert analyze_run(tmp_path)["delta_free_energy"] == summary["delta_free_energy"]
    assert analyze_run(tmp_path, recompute=True)["delta_free_energy"] == summary["delta_free_energy"]
    assert resume_run(tmp_path)["delta_free_energy"] == summary["delta_free_energy"]
    with pytest.raises((ValueError, RuntimeError), match="histogram|endpoint"):
        analyze_run(tmp_path, bins=[5])


def test_units_and_explicit_protocol_are_canonical():
    config = direct_config()
    config["cvs"][0]["unit"] = "angstrom"
    config["steered"].update(start=0.2, end=0.8, stiffness=2.0)
    cv, selected, options = resolve_steered(config, system())
    np.testing.assert_allclose(options["centers"][[0, -1], 0], np.array([0.2, 0.8]) * ANGSTROM_TO_AU)
    assert options["stiffness"][0, 0] == pytest.approx(2 / ANGSTROM_TO_AU**2)
    config["steered"] = {"centers_au": options["centers"].tolist(),
                         "stiffness_au": options["stiffness"].tolist(), "gamma_au": 1.0}
    _, _, alternate = resolve_steered(config, system())
    np.testing.assert_array_equal(alternate["centers"], options["centers"])
    np.testing.assert_array_equal(alternate["stiffness"], options["stiffness"])


@pytest.mark.parametrize("section,change,match", [
    ("steered", {"start_au": 0}, "not both"),
    ("steered", {"centers": [[0], [1]]}, "not both"),
    ("steered", {"direction": "backward"}, "direction"),
    ("steered", {"gamma": 0}, "gamma"),
    ("steered", {"thermostat": "none"}, "conflicts|equilibration"),
    ("steered", {"initial_equilibrium": "yes"}, "boolean"),
    ("analysis", {"measure": "cv_density"}, "endpoint"),
    ("analysis", {"method": "mbar"}, "analysis.method"),
    ("analysis", {"bins": 3}, "histogram"),
    ("sampling", {"thermostat_params": {}}, "owns|accept"),
    ("sampling", {"thermostat": "andersen"}, "conflicts"),
    ("sampling", {"integrator": "baoab"}, "steered-symmetric"),
    ("sampling", {"engine": "native"}, "does not accept"),
    ("sampling", {"andersen_interval": 5}, "does not accept"),
    ("output", {"trajectory_stride": 1}, "trajectory_stride"),
    ("dynamics", {"loop_backend": "native"}, "loop_backend"),
])
def test_unsupported_options_fail_before_output(tmp_path, section, change, match):
    config = direct_config()
    config.setdefault(section, {}).update(change)
    with pytest.raises(ValueError, match=match):
        run_steered(config, system(), tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()


def test_new_parameters_enter_scientific_identity(tmp_path):
    config = direct_config()
    run_steered(config, system(), tmp_path)
    changed = copy.deepcopy(config)
    changed["steered"]["gamma"] = 2.0
    with pytest.raises((ValueError, RuntimeError), match="identity"):
        run_steered(changed, system(), tmp_path)


@pytest.mark.parametrize("option", [{"andersen_interval": 5}, {"andersen_schedule": "fixed"},
                                   {"gamma_centroid": 0.1}, {"thermostat_params": {"gamma": 1.0}}])
def test_frontend_rejects_generic_bath_options(option):
    config = public_config()
    config["sampling"].update(option)
    with pytest.raises(ValueError, match="steered|steering"):
        JobSpec.from_mapping(config).resolve()


def test_public_prepare_rejects_ignored_trajectory_stride():
    config = public_config()
    config["output"]["trajectory_stride"] = 1
    with pytest.raises(ValueError, match="trajectory_stride"):
        prepare(config)
