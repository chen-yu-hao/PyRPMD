"""Rate PMF stages preserve raw time budgets and independent sampling choices."""

import copy
import json

import pytest

from pyrpmd.input import InputError, JobSpec, rate_pmf_options, rate_pmf_stage
from pyrpmd.io import referenced_files
from pyrpmd.units import parse_quantity
from pyrpmd.workflows.sampling import fes_sampler


def configuration():
    return {
        "task": "rate", "temperature": "500 K", "beads": 4, "seed": 23,
        "native": False,
        "system": {
            "symbols": ["H", "H", "H"], "masses": [1837., 1837., 1837.],
            "geometry": [[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]],
            "fragments": [[0], [1, 2]],
        },
        "potential": {"provider": "free"},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": "8 bohr"},
        "dynamics": {"timestep": "0.1 fs", "integrator": "cayley-bcb"},
        "sampling": {"equilibration": "2 fs", "production": "10 fs",
                     "interval": "0.2 fs", "andersen_interval": 10, "replicas": 2},
        "analysis": {"bins": 40, "bootstrap": 0},
        "output": {"checkpoint_interval": "1 fs"},
        "recrossing": {"equilibration_steps": 2, "parent_interval_steps": 2,
                       "child_steps": 4, "bootstrap": 0},
    }


def independent_pmf():
    return {
        "method": "umbrella", "estimator": "mbar",
        "dynamics": {"integrator": "normal-mode-bab", "timestep": "0.2 fs"},
        "sampling": {"thermostat": "pile-l", "gamma_centroid": "0.02 fs^-1"},
        "umbrella": {"centers": [-0.05, 0., 0.5, 1.], "kappa": "0.1 hartree"},
        "analysis": {"bins": 24, "ranges": [[-0.2, 1.2]]},
    }


def frozen_pmf(method):
    return {
        "method": method,
        method: {"sigma": 0.1, "pace": 10, "freeze_after_steps": 20,
                 "freeze_equilibration_steps": 10,
                 **({"height": "0.001 hartree", "mode": "standard"}
                    if method == "metadynamics" else {"barrier": "0.01 hartree"})},
    }


def test_legacy_input_does_not_acquire_stage_or_new_defaults():
    raw = configuration()
    before = copy.deepcopy(raw)
    resolved, system = JobSpec.from_mapping(raw).resolve()
    assert raw == before
    assert "rate" not in resolved
    assert rate_pmf_stage(resolved, system) is None
    options = rate_pmf_options(resolved)
    assert (options["sampler"], options["method"], options["source"]) == ("umbrella", "ui", None)
    assert resolved["sampling"]["integrator"] == "cayley-bcb-andersen"


def test_pmf_uses_its_own_clock_and_operator_without_mutating_parent():
    raw = configuration()
    raw["rate"] = {"pmf": independent_pmf()}
    before = copy.deepcopy(raw)
    resolved, system = JobSpec.from_mapping(raw).resolve()
    stage = rate_pmf_stage(resolved, system)
    assert raw == before
    assert resolved["dt_au"] == parse_quantity("0.1 fs", "time")
    assert stage["dt_au"] == parse_quantity("0.2 fs", "time")
    assert resolved["sampling"]["production_steps"] == 100
    assert stage["sampling"]["production_steps"] == 50
    assert stage["sampling"]["equilibration_steps"] == 10
    assert stage["sampling"]["interval_steps"] == 1
    assert stage["output"]["checkpoint_steps"] == 5
    assert resolved["output"]["checkpoint_steps"] == 10
    assert resolved["sampling"]["thermostat"] == "andersen"
    assert stage["sampling"]["thermostat"] == "pile-l"
    assert fes_sampler(stage).integrator == "baoab"
    assert stage["analysis"] == {"bins": 24, "bootstrap": 0,
                                 "ranges": [[-0.2, 1.2]], "method": "mbar", "measure": "rate_flux"}
    assert rate_pmf_options(resolved)["bins"] == 24
    assert rate_pmf_options(resolved)["method"] == "mbar"
    # A caller's modifications must not change the persisted stage.
    stage["sampling"]["production_steps"] = 123
    assert rate_pmf_stage(resolved, system)["sampling"]["production_steps"] == 50


def test_local_duration_overrides_inherited_steps_and_saved_steps_survive_json():
    raw = configuration()
    raw["sampling"].update(production_steps=200, interval_steps=4)
    pmf = independent_pmf()
    pmf["sampling"].update(production="4 fs", interval="0.4 fs")
    raw["rate"] = {"pmf": pmf}
    resolved, system = JobSpec.from_mapping(raw).resolve()
    saved = json.loads(json.dumps(resolved))
    stage = rate_pmf_stage(saved, system)
    assert resolved["sampling"]["production_steps"] == 200
    assert stage["sampling"]["production_steps"] == 20
    assert stage["sampling"]["interval_steps"] == 2
    assert stage["sampling"]["dt_au"] == stage["dt_au"]


@pytest.mark.parametrize("mode,params,expected", [
    ("nhc", {"tau": "2 fs", "chain_length": 3}, "obabo"),
    ("langevin", {"gamma": "0.02 fs^-1"}, "baoab"),
    ("andersen", {}, "normal-mode-bab-andersen"),
])
def test_local_thermostat_replaces_inherited_splitting_and_parameters(mode, params, expected):
    raw = configuration()
    raw["sampling"].update(integrator="cayley-bcb-andersen", thermostat_params={"interval_steps": 10})
    pmf = independent_pmf()
    pmf["sampling"] = {"thermostat": mode, "thermostat_params": params}
    raw["rate"] = {"pmf": pmf}
    resolved, system = JobSpec.from_mapping(raw).resolve()
    stage = rate_pmf_stage(resolved, system)
    assert stage["sampling"]["thermostat_params"] == params
    assert fes_sampler(stage).integrator == expected


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_frozen_methods_get_flux_measure_and_zero_bootstrap(method):
    raw = configuration()
    raw.pop("analysis")
    raw["rate"] = {"pmf": frozen_pmf(method)}
    resolved, system = JobSpec.from_mapping(raw).resolve()
    stage = rate_pmf_stage(resolved, system)
    options = rate_pmf_options(resolved)
    assert (options["sampler"], options["method"], options["bootstrap"]) == (method, "reweighted_histogram", 0)
    assert stage["analysis"] == {"bootstrap": 0, "method": "reweighted_histogram", "measure": "rate_flux"}
    assert stage[method]["freeze_after_steps"] == 20
    assert stage[method]["cvs"] == ["xi"]


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_selected_frozen_pmf_accepts_equivalent_top_level_method_table(method):
    nested = configuration()
    nested["rate"] = {"pmf": frozen_pmf(method)}
    top = copy.deepcopy(nested)
    top[method] = top["rate"]["pmf"].pop(method)
    expected, expected_system = JobSpec.from_mapping(nested).resolve()
    actual, actual_system = JobSpec.from_mapping(top).resolve()
    assert rate_pmf_stage(actual, actual_system) == rate_pmf_stage(expected, expected_system)
    assert actual["analysis"] == expected["analysis"]
    assert rate_pmf_options(actual) == rate_pmf_options(expected)


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
@pytest.mark.parametrize("selector", ["absent", "umbrella", "other", "duplicate"])
def test_top_level_frozen_table_requires_its_explicit_unique_pmf_selection(method, selector):
    raw = configuration()
    raw[method] = frozen_pmf(method)[method]
    if selector == "umbrella":
        raw["rate"] = {"pmf": {"method": "umbrella"}}
    elif selector == "other":
        other = "opes" if method == "metadynamics" else "metadynamics"
        raw["rate"] = {"pmf": frozen_pmf(other)}
    elif selector == "duplicate":
        raw["rate"] = {"pmf": frozen_pmf(method)}
    with pytest.raises(InputError, match="requires|does not accept|not both"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
def test_explicit_nonzero_bootstrap_requires_local_override(method):
    raw = configuration()
    raw["analysis"]["bootstrap"] = 20
    raw["rate"] = {"pmf": frozen_pmf(method)}
    with pytest.raises(InputError, match="bootstrap=0"):
        JobSpec.from_mapping(raw).resolve()
    raw["rate"]["pmf"]["analysis"] = {"bootstrap_replicates": 0}
    resolved, _ = JobSpec.from_mapping(raw).resolve()
    assert rate_pmf_options(resolved)["bootstrap"] == 0


@pytest.mark.parametrize("change,match", [
    ({"estimator": "wham"}, "estimator"),
    ({"method": "missing"}, "method"),
    ({"directory": "run"}, "external"),
    ({"dynamics": {"temprature": 300}}, "Unknown"),
    ({"sampling": {"thermostat": "none"}}, "canonical"),
    ({"sampling": {"thermostat": "piglet"}}, "canonical"),
    ({"sampling": {"thermostat_params": []}}, "thermostat_params"),
    ({"sampling": {"engine": "native"}}, "engine"),
    ({"analysis": {"measure": "cv_density"}}, "rate_flux"),
    ({"umbrella": {"cvs": ["distance"]}}, "prepared RPMDrate coordinate"),
    ({"adaptive_umbrella": {"target_overlap": 1.1}}, "target_overlap"),
    ({"dynamics": {"integrator": "velocity-verlet"}, "sampling": {"thermostat": "pile-l"}}, "PILE-L"),
])
def test_invalid_local_combinations_fail_during_input(change, match):
    raw = configuration()
    raw["rate"] = {"pmf": {"method": "umbrella", **change}}
    with pytest.raises(InputError, match=match):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method", ["metadynamics", "opes"])
@pytest.mark.parametrize("change,match", [
    ({"freeze_after_steps": 21}, "multiple of pace"),
    ({"freeze_after_steps": 100}, "at least two saved samples"),
    ({"freeze_equilibration_steps": 80}, "at least two saved samples"),
    ({"freeze_equilibration_steps": -1}, "freeze_equilibration_steps"),
])
def test_freeze_schedule_is_validated_against_pmf_production_clock(method, change, match):
    raw = configuration()
    pmf = frozen_pmf(method)
    pmf[method].update(change)
    raw["rate"] = {"pmf": pmf}
    with pytest.raises(InputError, match=match):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("legacy", ["pmf_method", "pmf_source"])
def test_ambiguous_legacy_selection_is_rejected(legacy):
    raw = configuration()
    raw["rate"] = {"pmf": {"method": "umbrella"}, legacy: "ui" if legacy == "pmf_method" else "umbrella"}
    with pytest.raises(InputError, match="not both"):
        JobSpec.from_mapping(raw).resolve()


def test_user_cannot_supply_internal_stage_or_duplicate_umbrella_tables():
    raw = configuration()
    raw["rate"] = {"pmf_stage": {"config": {}}}
    with pytest.raises(InputError, match="pmf_stage"):
        JobSpec.from_mapping(raw).resolve()
    raw["rate"] = {"pmf": independent_pmf()}
    raw["umbrella"] = {"kappa": 0.1}
    with pytest.raises(InputError, match="not both"):
        JobSpec.from_mapping(raw).resolve()


def test_external_source_is_resolved_to_hashable_paths(tmp_path):
    directory = tmp_path / "source"
    (directory / "input").mkdir(parents=True)
    for name in ("fes.npz", "input/resolved.json", "input/system.json", "manifest.json", "fes.json"):
        (directory / name).write_text("placeholder")
    raw = configuration()
    raw["rate"] = {"pmf": {"method": "external", "directory": "source", "allow_non_umbrella": True}}
    resolved, system = JobSpec.from_mapping(raw, base_directory=tmp_path).resolve()
    options = rate_pmf_options(resolved)
    assert options["sampler"] == "external"
    assert options["source"]["directory"] == str(directory)
    assert options["source"]["allow_non_umbrella"] is True
    assert str(directory / "fes.npz") in referenced_files(resolved)
    assert rate_pmf_stage(resolved, system) is None


@pytest.mark.parametrize("extra", [{"estimator": "mbar"}, {"sampling": {}}, {"analysis": {}}])
def test_external_rejects_ignored_local_controls(extra):
    raw = configuration()
    raw["rate"] = {"pmf": {"method": "external", "directory": "source", **extra}}
    with pytest.raises(InputError, match="estimator|local dynamics"):
        JobSpec.from_mapping(raw).resolve()
