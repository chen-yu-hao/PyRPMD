"""Stage-local clocks, budgets and actual constrained-parent replay."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.input_stages import normalize_sampling_stage, normalize_stage_input
from pyrpmd.system import System
from pyrpmd.units import parse_quantity
from pyrpmd.workflows import rate_recrossing
from pyrpmd.workflows.rate_options import _parent_thermostat, _recrossing_options
from pyrpmd.workflows.sampling import AndersenSchedule


def _new_job():
    return {
        "schema_version": 2, "task": "rate", "temperature": "300 K", "beads": 2,
        "system": {"symbols": ["H", "H", "H"],
                   "geometry": [[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]]},
        "potential": {"provider": "free"},
        "reaction": {"definition": "explicit", "fragments": [[0], [1, 2]],
                     "forming_bonds": [[0, 1]], "breaking_bonds": [[1, 2]]},
        "preparation": {"ts_action": "none"},
        "dynamics": {"timestep": "0.1 fs", "native": False},
        "umbrella": {
            "centers": [0, 0.5, 1], "stiffness": "0.16 hartree",
            "sampling": {"equilibration": "5 fs", "production": "20 fs",
                         "sample_every": "0.2 fs", "chains_per_window": 1,
                         "thermostat": {"type": "andersen", "schedule": "fixed",
                                        "refresh_every": "2 fs"}},
            "analysis": {"estimator": "ui", "integration_points": 101,
                         "bins": 20, "bootstrap_replicates": 10},
        },
        "recrossing": {
            "parent": {"chains": 2, "equilibration": {"steps": 10},
                       "draw_every": "1 fs", "configurations_per_chain": 4,
                       "thermostat": {"type": "andersen", "schedule": "geometric",
                                      "mean_refresh_interval": "0.5 fs"}},
            "children": {"method": "rpmd", "pairs_per_configuration": 2, "duration": "10 fs"},
            "analysis": {"bootstrap_replicates": 30},
        },
    }


def test_operation_named_input_resolves_all_stage_units_and_budgets():
    job = _new_job()
    original = deepcopy(job)
    config, _ = JobSpec.from_mapping(job).resolve()
    assert job == original
    sample = config["sampling"]
    assert (sample["equilibration_steps"], sample["production_steps"], sample["sample_interval"]) == (50, 200, 2)
    assert sample["andersen_interval"] == 20 and sample["replicas"] == 1
    assert _parent_thermostat(config) == (5, "geometric")
    options = _recrossing_options(config)
    assert options["parent_replicas"] == 2 and options["parent_samples"] == 4
    assert options["equilibration_steps"] == 10 and options["parent_interval_steps"] == 10
    assert options["child_pairs"] == 2 and options["child_steps"] == 100
    assert options["bootstrap"] == 30
    assert config["analysis"]["bootstrap"] == 10
    assert config["rate"] == {"pmf_method": "ui", "pmf_points": 101}
    assert "sampling" not in config["umbrella"] and "parent" not in config["recrossing"]


def test_production_completeness_accepts_the_new_sampling_location():
    job = _new_job()
    job["preset"] = "production"
    JobSpec.from_mapping(job).resolve()


def test_new_parent_default_is_independent_of_umbrella_and_legacy_parent_count():
    job = _new_job()
    job["recrossing"] = {"children": {"duration": "1 fs"}}
    job["umbrella"]["sampling"]["thermostat"]["refresh_every"] = "9 fs"
    config, _ = JobSpec.from_mapping(job).resolve()
    assert config["sampling"]["andersen_interval"] == 90
    assert _parent_thermostat(config) == (10, "fixed")
    assert _recrossing_options(config)["parent_samples"] == 4


def test_legacy_normalization_does_not_add_defaults_or_change_alias_precedence():
    legacy = {"task": "rate", "sampling": {"interval": "0.2 fs", "sample_interval": 3,
                                              "andersen_interval": 7, "parents": 9},
              "recrossing": {"child_steps": 4}}
    original = deepcopy(legacy)
    normalize_stage_input(legacy)
    assert legacy == original
    assert _parent_thermostat(legacy) == (7, "fixed")
    assert _recrossing_options({**legacy, "dt_au": parse_quantity("0.1 fs", "time")})["parent_samples"] == 9


def test_new_reaction_controls_also_opt_into_independent_parent_defaults():
    data = {"task": "rate", "reaction": {"definition": "explicit"},
            "sampling": {"andersen_interval": 91, "parents": 13}}
    normalize_stage_input(data)
    assert _parent_thermostat(data) == (10, "fixed")
    assert data["recrossing"]["parent_samples"] == 4


@pytest.mark.parametrize("value", [2, 0.2, True, {"steps": 2, "time": "0.2 fs"}, {"steps": 2.1}, "0.15 fs"])
def test_operation_times_cannot_be_ambiguous_or_fractional_steps(value):
    with pytest.raises(InputError, match="sample_every"):
        normalize_sampling_stage({"sample_every": value}, parse_quantity("0.1 fs", "time"))


@pytest.mark.parametrize("thermostat", [
    {"schedule": "fixed", "mean_refresh_interval": "2 fs"},
    {"schedule": "geometric", "refresh_every": "2 fs"},
    {"schedule": "fixed", "refresh_every": "2 fs", "mean_refresh_interval": "2 fs"},
    {"scheme": "per_atom", "refresh_every": "2 fs"},
    {"type": "pile-l", "refresh_every": "2 fs"},
])
def test_thermostat_schedule_must_match_its_named_parameter(thermostat):
    with pytest.raises(InputError):
        normalize_sampling_stage({"thermostat": thermostat}, parse_quantity("0.1 fs", "time"))


@pytest.mark.parametrize("mutate", [
    lambda job: job.update(sampling={"production": "1 fs"}),
    lambda job: job["recrossing"].update(parent_interval="1 fs"),
    lambda job: job["recrossing"].update(child_pairs_per_parent=2),
    lambda job: job.update(analysis={"bootstrap": 5}),
    lambda job: job["recrossing"].update(parent=None),
    lambda job: job["recrossing"]["children"].update(thermostat={"type": "andersen"}),
])
def test_stage_conflicts_and_malformed_tables_fail_before_resolving_system(mutate):
    job = _new_job()
    mutate(job)
    with pytest.raises(InputError):
        normalize_stage_input(job)


def test_explicit_trpmd_child_method_is_preserved():
    job = _new_job()
    job["recrossing"]["children"].update(method="trpmd", friction_scale=0.7)
    config, _ = JobSpec.from_mapping(job).resolve()
    assert config["recrossing"]["dynamics"] == "trpmd"
    assert config["recrossing"]["friction_scale"] == 0.7


def _parent_task(path, *, umbrella_interval=2, parent_interval=5, schedule="fixed", block_steps=7):
    system = System(("H", "H", "H"), np.full(3, 2.0),
                    np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]), ((0,), (1, 2)))
    config = {
        "task": "rate", "beta": 10., "beads": 4, "dt_au": 1e-4, "seed": 9131,
        "potential": {"provider": "free"}, "native": False,
        "sampling": {"andersen_interval": umbrella_interval},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 10.},
        "recrossing": {"parent_thermostat": {"thermostat": "andersen",
                       "andersen_interval": parent_interval, "andersen_schedule": schedule}},
        "dynamics": {"loop_backend": "python", "block_steps": block_steps},
        "output": {"checkpoint_steps": 11},
    }
    return {"config": config, "system": system, "chain": 0, "xi_star": 1., "output": str(path),
            "initial_q": np.repeat(system.geometry[None], 4, axis=0),
            "options": {"equilibration_steps": 7, "parent_samples": 4,
                        "parent_interval_steps": 4, "constraint_tolerance": 1e-10}}


def _saved(task):
    with np.load(Path(task["output"]) / "checkpoints/parent_0000.npz", allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def test_actual_parent_clock_and_trajectory_ignore_umbrella_thermostat(tmp_path, monkeypatch):
    events = []
    original = AndersenSchedule.refreshed

    def record(clock, step, rng):
        events.append(step)
        return original(clock, step, rng)

    first = _parent_task(tmp_path / "first", umbrella_interval=2)
    other = _parent_task(tmp_path / "other", umbrella_interval=17)
    with monkeypatch.context() as patch:
        patch.setattr(AndersenSchedule, "refreshed", record)
        rate_recrossing._parent_task(first)
    assert events == [0, 5, 10, 15, 20]
    rate_recrossing._parent_task(other)
    for key in ("q", "p", "snapshots", "rng"):
        np.testing.assert_array_equal(_saved(first)[key], _saved(other)[key])
    changed = _parent_task(tmp_path / "changed", parent_interval=3)
    rate_recrossing._parent_task(changed)
    assert not np.array_equal(_saved(first)["q"], _saved(changed)["q"])


def test_independent_geometric_parent_clock_resumes_with_different_execution_block(tmp_path, monkeypatch):
    complete = _parent_task(tmp_path / "complete", schedule="geometric", block_steps=1)
    resumed = _parent_task(tmp_path / "resumed", schedule="geometric")
    rate_recrossing._parent_task(complete)
    original = rate_recrossing._atomic_npz

    def interrupt(path, **arrays):
        original(path, **arrays)
        if arrays["elapsed"] == 11:
            raise KeyboardInterrupt("after checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(rate_recrossing, "_atomic_npz", interrupt)
        with pytest.raises(KeyboardInterrupt):
            rate_recrossing._parent_task(resumed)
    resumed["config"]["dynamics"]["block_steps"] = 3
    rate_recrossing._parent_task(resumed)
    actual, expected = _saved(resumed), _saved(complete)
    for key in ("q", "p", "snapshots"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-13)
    for key in ("rng", "andersen_schedule", "andersen_next_collision_step", "elapsed"):
        np.testing.assert_array_equal(actual[key], expected[key])
    resumed["config"]["recrossing"]["parent_thermostat"]["andersen_interval"] += 1
    with pytest.raises(ValueError, match="physics differs"):
        rate_recrossing._parent_task(resumed)


def test_nested_budget_generates_actual_parent_configurations_and_child_pairs(tmp_path):
    tasks = [_parent_task(tmp_path / "parents") for _ in range(2)]
    parents = []
    for chain, task in enumerate(tasks):
        task["chain"] = chain
        parents.append(rate_recrossing._parent_task(task))
    assert [len(parent["snapshots"]) for parent in parents] == [4, 4]
    children = rate_recrossing._child_tasks(parents, tasks[0]["config"], tasks[0]["system"],
                tmp_path / "children", {"child_pairs": 2, "child_steps": 3}, {"xi_star": 1.})
    assert len(children) == 16  # Each job produces the correlated +p/-p pair.
