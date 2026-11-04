"""Strict path-sampling input grammar and preservation of scientific identity."""

import numpy as np
import pytest

from pyrpmd.constants import BOHR_ANGSTROM, KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.io import fingerprint, referenced_files
from pyrpmd.propagation import scientific_config


def raw_input():
    return {
        "task": "path_sampling", "beads": 1,
        "temperature": 1 / KB_AU, "seed": 71,
        "system": {"symbols": ["H"], "geometry": [[-0.3, 0., 0.]],
                   "geometry_unit": "bohr", "masses": [1.]},
        "potential": {"provider": "harmonic", "params": {"k": 1.}},
        "dynamics": {"timestep": "0.04 au", "native": False},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "path_sampling": {
            "method": "retis", "cv": "x", "basin_a": -1., "basin_b": 1.,
            "interfaces": [-1., -0.5], "samples": 20, "burn_in": 3,
            "max_path_steps": 600,
            "seeds": [{"q": [[-0.3, 0., 0.]], "p": [[1.4, 0., 0.]]}] * 2,
            "minus_seed": {"q": [[-1.1, 0., 0.]], "p": [[-0.7, 0., 0.]]},
        },
        "output": {"checkpoint_steps": 2},
    }


def test_path_sampling_resolves_shared_molecular_units_and_paths(tmp_path):
    data = raw_input()
    data["potential"] = {"provider": "python", "factory": "model.py:make_potential"}
    (tmp_path / "model.py").write_text("def make_potential():\n    raise AssertionError('not constructed at resolution')\n")
    resolved, system = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    assert resolved["task"] == "path_sampling"
    assert resolved["beta"] == pytest.approx(1.)
    assert resolved["dt_au"] == pytest.approx(.04)
    np.testing.assert_array_equal(system.masses, [1.])
    np.testing.assert_array_equal(system.geometry, [[-.3, 0., 0.]])
    assert resolved["potential"]["factory"] == str(tmp_path / "model.py") + ":make_potential"
    assert str(tmp_path / "model.py") in referenced_files(resolved)
    assert resolved["output"]["directory"] == str(tmp_path / "runs" / "path_sampling")
    assert resolved["output"]["checkpoint_steps"] == 2
    assert "sampling" not in resolved
    # Input normalization is idempotent, including unit conversion.
    from pyrpmd.workflows.path_sampling import resolve_path_sampling
    assert resolve_path_sampling(resolved, system) == resolved["path_sampling"]
    assert data["potential"]["factory"] == "model.py:make_potential"


def test_path_sampling_cv_quantities_are_resolved_once(tmp_path):
    data = raw_input()
    data["path_sampling"]["basin_a"] = f"{-BOHR_ANGSTROM} angstrom"
    data["path_sampling"]["basin_b"] = f"{BOHR_ANGSTROM} angstrom"
    data["path_sampling"]["interfaces"] = [
        f"{-BOHR_ANGSTROM} angstrom", f"{-0.5 * BOHR_ANGSTROM} angstrom",
    ]
    config, _ = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    options = config["path_sampling"]
    assert options["basin_a"] == pytest.approx(-1.)
    assert options["basin_b"] == pytest.approx(1.)
    assert options["interfaces"] == pytest.approx([-1., -.5])


def test_path_sampling_production_budget_does_not_require_pimd_sampling(tmp_path):
    data = raw_input()
    data["preset"] = "production"
    config, _ = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    assert config["preset"] == "production"
    assert config["path_sampling"]["samples"] == 20
    assert "sampling" not in config


def test_infinite_path_sampling_defaults_to_dynamic_scheduling(tmp_path):
    data = raw_input()
    data["path_sampling"]["exchange"] = "infinite"
    config, _ = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    assert config["parallel"]["scheduling"] == "dynamic"
    assert config["path_sampling"]["workers"] == 1


def test_infinite_path_sampling_does_not_override_explicit_static_scheduling(tmp_path):
    data = raw_input()
    data["path_sampling"]["exchange"] = "infinite"
    data["parallel"] = {"scheduling": "static"}
    with pytest.raises(InputError, match="dynamic"):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


def test_repptis_frontend_validates_matching_external_flux_metadata(tmp_path):
    data = raw_input()
    options = data["path_sampling"]
    options["method"] = "repptis"
    options.pop("minus_seed")
    options["interfaces"] = [-.8, -.4]
    options["external_flux"] = {
        "basin_a": -1., "basin_b": 1., "first_interface": -.8,
        "dt_au": .04, "flux_au": .2, "provenance": "independent A-history trace",
    }
    config, _ = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    assert config["path_sampling"]["external_flux"] == options["external_flux"]
    options["external_flux"]["dt_au"] = .03
    with pytest.raises(InputError, match="external_flux.dt_au"):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


def test_ffs_frontend_keeps_forward_process_explicit_and_normalizes_trial_counts(tmp_path):
    data = raw_input()
    data["path_sampling"] = {
        "method": "ffs", "cv": "x", "basin_a": -1., "basin_b": 1.,
        "interfaces": [-1., -.5], "flux_steps": 100, "trials_per_interface": 12,
        "dynamics": "basin-andersen", "collision_rate": "0.4 au",
        "initial_state": {"q": [[-1.1, 0., 0.]], "p": [[.3, 0., 0.]]},
    }
    config, _ = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    options = config["path_sampling"]
    assert options["trials_per_interface"] == [12, 12]
    assert options["collision_rate"] == .4
    assert options["initial_state"]["q"] == [[[-1.1, 0., 0.]]]
    data["path_sampling"].pop("dynamics")
    with pytest.raises(InputError, match="dynamics"):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


@pytest.mark.parametrize("field,value", [("initial_state", {}), ("collision_rate", 1.), ("flux_steps", 10)])
def test_retis_frontend_rejects_ffs_controls_instead_of_ignoring_them(tmp_path, field, value):
    data = raw_input()
    data["path_sampling"][field] = value
    with pytest.raises(InputError, match=field):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


@pytest.mark.parametrize("field,value", [
    ("q", [[-.3, True, 0.]]), ("p", [[1.4, False, 0.]]),
    ("q", [[-.3, float("nan"), 0.]]), ("p", [[1.4, float("inf"), 0.]]),
    ("q", [[-.3, 0.]]), ("p", [1.4, 0., 0.]),
])
def test_path_sampling_seed_arrays_are_strict_real_phase_points(tmp_path, field, value):
    data = raw_input()
    data["path_sampling"]["seeds"][0][field] = value
    with pytest.raises(InputError, match=field):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


def test_infinite_logical_worker_count_is_saved_in_scientific_identity(tmp_path):
    data = raw_input()
    data["path_sampling"]["exchange"] = "infinite"
    first, system = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    data["parallel"] = {"trajectory_workers": 2}
    second, _ = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    assert first["path_sampling"]["workers"] == 1
    assert second["path_sampling"]["workers"] == 2
    assert fingerprint(scientific_config(first), system) != fingerprint(scientific_config(second), system)
    data["path_sampling"]["workers"] = 1
    with pytest.raises(InputError, match="workers"):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


@pytest.mark.parametrize("section,value", [
    ("sampling", {}), ("sampling", {"thermostat": "andersen"}),
    ("barostat", {}), ("path_integral", {}), ("reaction", {}),
    ("analysis", {}), ("properties", {}), ("nonadiabatic", {}),
    ("misspelled_option", 1),
])
def test_path_sampling_never_drops_unrelated_sections(tmp_path, section, value):
    data = raw_input()
    data[section] = value
    with pytest.raises(InputError, match=section):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("section,value", [
    ("system", []), ("potential", []), ("dynamics", []), ("parallel", []),
    ("output", []), ("path_sampling", []), ("cvs", "x"),
])
def test_path_sampling_rejects_malformed_tables(tmp_path, section, value):
    data = raw_input()
    data[section] = value
    with pytest.raises(InputError, match=section):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


@pytest.mark.parametrize("name,value", [
    ("checkpoint_interval", "1 fs"), ("trajectory_stride", 2),
    ("trajectory_storage", "hdf5"), ("diagnostics", {}),
])
def test_path_sampling_rejects_unused_md_output_controls(tmp_path, name, value):
    data = raw_input()
    data["output"][name] = value
    with pytest.raises(InputError, match=name):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


@pytest.mark.parametrize("section,name,value", [
    ("path_sampling", "samplez", 20),
    ("system", "geometery", [[0, 0, 0]]),
    ("dynamics", "timestpe", .1),
    ("parallel", "workerz", 2),
])
def test_path_sampling_rejects_misspellings(tmp_path, section, name, value):
    data = raw_input()
    data.setdefault(section, {})[name] = value
    with pytest.raises(InputError, match=name):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()


@pytest.mark.parametrize("field,value", [
    ("samples", 21), ("burn_in", 4), ("max_path_steps", 601),
    ("interfaces", [-1., -.4]),
])
def test_path_sampling_new_scientific_options_enter_identity(tmp_path, field, value):
    data = raw_input()
    old, system = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    data["path_sampling"][field] = value
    new, new_system = JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
    assert fingerprint(scientific_config(old), system) != fingerprint(scientific_config(new), new_system)


def test_path_sampling_resolved_runner_validates_before_output_creation(tmp_path):
    from pyrpmd.runner import run_resolved

    config, system = JobSpec.from_mapping(raw_input(), base_directory=tmp_path).resolve()
    config["path_sampling"]["method"] = "invented-method"
    output = tmp_path / "invalid"
    with pytest.raises((InputError, ValueError, RuntimeError), match="method"):
        run_resolved(config, system, output=output)
    assert not output.exists()


def test_path_sampling_frontend_does_not_register_fields_for_other_tasks(tmp_path):
    data = raw_input()
    data["task"] = "pimd"
    with pytest.raises(InputError, match="path_sampling"):
        JobSpec.from_mapping(data, base_directory=tmp_path).resolve()
