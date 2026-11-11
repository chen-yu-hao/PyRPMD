"""Automatic sampling defaults are explicit, checked and opt-in."""

from copy import deepcopy
from pathlib import Path

import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.units import time_steps


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "rate"


def automatic_spec():
    return JobSpec.from_file(EXAMPLES / "automatic.inp")


def test_minimal_auto_rate_fills_all_sampling_stages_without_evaluating_pes(monkeypatch):
    monkeypatch.setattr("pyrpmd.potentials.make_potential",
                        lambda *a, **k: pytest.fail("input resolution evaluated PES"))
    spec = automatic_spec()
    original = deepcopy(spec.data)
    config, _ = spec.resolve()
    assert spec.data == original
    assert config["rate"]["auto"] == {"target_relative_error": 0.1, "max_rounds": 8}
    assert config["reaction"]["definition"] == "infer"
    assert config["umbrella"]["start"] < 0
    assert config["umbrella"]["stop"] > config["rate"]["pmf_search_range"][1]
    assert config["umbrella"]["stiffness"] == pytest.approx(KB_AU * 1000 / .05**2)
    assert config["sampling"]["replicas"] == 2
    assert config["sampling"]["production_steps"] == time_steps("1000 fs", config["dt_au"])
    assert config["analysis"]["bootstrap"] == 200
    assert config["rate"]["pmf_method"] == "mbar"
    assert config["sampling"]["andersen_interval"] == time_steps("20 fs", config["dt_au"])
    assert config["adaptive_umbrella"]["enabled"]
    assert config["adaptive_umbrella"]["adapt_stiffness"]
    assert config["adaptive_umbrella"]["target_coverage"] == 1
    assert config["recrossing"]["parent_replicas"] == 4
    assert config["recrossing"]["parent_samples"] == 16
    assert config["recrossing"]["child_pairs"] == 2
    assert config["recrossing"]["bootstrap"] == 200


def test_explicit_estimator_and_momentum_refresh_are_preserved():
    spec = automatic_spec()
    spec.data["rate"]["pmf_method"] = "ui"
    spec.data["sampling"] = {"andersen_interval": 17}
    config, _ = spec.resolve()
    assert config["rate"]["pmf_method"] == "ui"
    assert config["sampling"]["andersen_interval"] == 17


def test_production_auto_supplies_only_sampling_defaults_and_preserves_physical_choices():
    spec = automatic_spec()
    spec.data.update(preset="production", beads=8)
    spec.data["dynamics"] = {"timestep": "0.05 fs", "integrator": "normal-mode-bab"}
    spec.data["rate"]["auto"] = {"target_relative_error": .05, "max_rounds": 3}
    config, _ = spec.resolve()
    assert config["beads"] == 8
    assert config["dynamics"] == spec.data["dynamics"]
    assert config["rate"]["auto"]["max_rounds"] == 3
    assert config["adaptive_umbrella"]["pilot_steps"] == 4000


def test_explicit_operation_starts_and_reaction_are_preserved():
    spec = automatic_spec()
    spec.data["reaction"] = {
        "definition": "explicit", "fragments": [[0], [1, 2]],
        "forming_bonds": [[0, 1]], "breaking_bonds": [[1, 2]],
    }
    spec.data["umbrella"] = {
        "centers": [-.1, 0, .5, 1, 1.1], "stiffness": .25,
        "sampling": {"production": "200 fs", "chains_per_window": 3},
    }
    spec.data["recrossing"] = {
        "parent": {"chains": 3, "configurations_per_chain": 7},
        "children": {"pairs_per_configuration": 5, "duration": "30 fs"},
    }
    config, _ = spec.resolve()
    assert config["reaction"]["definition"] == "explicit"
    assert config["umbrella"]["stiffness"] == .25
    assert config["sampling"]["replicas"] == 3
    assert config["sampling"]["production_steps"] == 2000
    assert config["recrossing"]["parent_samples"] == 7
    assert config["recrossing"]["parent_replicas"] == 3
    assert config["recrossing"]["child_pairs"] == 5
    assert config["recrossing"]["child_steps"] == 300


def test_independent_pmf_clock_controls_pilot_and_inherited_starts_are_preserved():
    spec = automatic_spec()
    spec.data["sampling"] = {"replicas": 3, "production": "400 fs"}
    spec.data["rate"]["pmf"] = {
        "method": "umbrella", "estimator": "mbar",
        "dynamics": {"timestep": "0.05 fs"},
    }
    config, _ = spec.resolve()
    stage = config["rate"]["pmf_stage"]["config"]
    assert stage["sampling"]["replicas"] == 3
    assert stage["sampling"]["production_steps"] == 8000
    assert stage["adaptive_umbrella"]["pilot_steps"] == 4000
    assert stage["adaptive_umbrella"]["adapt_stiffness"]
    assert "umbrella" not in config


def test_schema_one_nested_recrossing_keeps_explicit_parent_count():
    spec = automatic_spec()
    spec.data["schema_version"] = 1
    spec.data["rate"]["recrossing"] = {"parent_samples": 9, "child_pairs": 3}
    config, _ = spec.resolve()
    assert "recrossing" not in config["rate"]
    assert config["recrossing"]["parent_samples"] == 9
    assert config["recrossing"]["child_pairs"] == 3


def test_schema_one_legacy_sampling_parent_count_is_preserved():
    spec = automatic_spec()
    spec.data["schema_version"] = 1
    spec.data["sampling"] = {"parents": 9}
    config, _ = spec.resolve()
    assert config["recrossing"]["parent_samples"] == 9


def test_generated_default_budgets_follow_arbitrary_stage_clocks():
    spec = automatic_spec()
    spec.data["dynamics"] = {"timestep": .003}
    spec.data["rate"]["pmf"] = {"method": "umbrella", "dynamics": {"timestep": .007}}
    config, _ = spec.resolve()
    stage = config["rate"]["pmf_stage"]["config"]
    assert config["dt_au"] == .003 and stage["dt_au"] == .007
    assert stage["sampling"]["production_steps"] < config["sampling"]["production_steps"]
    assert stage["adaptive_umbrella"]["pilot_steps"] > 0
    assert config["recrossing"]["child_steps"] > 0


@pytest.mark.parametrize("auto", [False, {"enabled": False}])
def test_disabled_auto_preserves_legacy_sampling(auto):
    spec = JobSpec.from_file(EXAMPLES / "minimal.inp")
    expected, _ = spec.resolve()
    spec.data["rate"] = {"auto": auto}
    actual, _ = spec.resolve()
    actual.pop("rate")
    assert actual == expected


@pytest.mark.parametrize("auto, message", [
    ("yes", "boolean or a table"),
    ({"enabled": 1}, "enabled must be boolean"),
    ({"target_relative_error": 0}, "between zero and one"),
    ({"target_relative_error": float("nan")}, "between zero and one"),
    ({"target_relative_error": True}, "between zero and one"),
    ({"max_rounds": 0}, "max_rounds"),
    ({"max_round": 2}, "Unknown rate.auto"),
])
def test_invalid_auto_request_fails_before_pes(auto, message, monkeypatch):
    spec = automatic_spec()
    spec.data["rate"]["auto"] = auto
    monkeypatch.setattr("pyrpmd.potentials.make_potential",
                        lambda *a, **k: pytest.fail("invalid request evaluated PES"))
    with pytest.raises(InputError, match=message):
        spec.resolve()


@pytest.mark.parametrize("change, message", [
    (lambda data: data.pop("temperature"), "explicit temperature"),
    (lambda data: data.update(task="fes"), "requires task='rate'"),
    (lambda data: data["rate"].update(pmf_source="old-run"), "locally sampled AUS"),
    (lambda data: data["rate"].update(pmf={"method": "opes"}), "method='umbrella'"),
    (lambda data: data.update(sampling={"replicas": 1}), "replicas"),
    (lambda data: data.update(analysis={"bootstrap": 0}), "analysis.bootstrap"),
    (lambda data: data.update(recrossing={"bootstrap": 0}), "recrossing.bootstrap"),
    (lambda data: data.update(recrossing={"children": {"method": "trpmd"}}), "dynamics='rpmd'"),
    (lambda data: data.update(adaptive_umbrella={"enabled": False}), "enabled=true"),
    (lambda data: data.update(adaptive_umbrella={"adapt_stiffness": False}), "adapt_stiffness=true"),
])
def test_incompatible_auto_inputs_are_rejected(change, message):
    spec = automatic_spec()
    change(spec.data)
    with pytest.raises(InputError, match=message):
        spec.resolve()
