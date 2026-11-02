"""Concrete frozen-job contracts for the existing rate executor."""

from copy import deepcopy
import json

import numpy as np
import pytest

from pyrpmd.input import rate_pmf_stage
from pyrpmd.precision import (
    CheckSchedule, NumericalRequirement, PrecisionSpec, PrecisionTarget,
    SamplingDesign, evidence_from_replicas, freeze_production,
)
from pyrpmd.precision_jobs import identity_from_config, materialize_rate_job, record_from_rate_job
from pyrpmd.runner import run_resolved
from pyrpmd.system import System


def configuration(*, scoped=False):
    system = System(("H", "H", "H"), np.array([2., 2., 2.]),
                    np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]),
                    ((0,), (1, 2)))
    config = {
        "task": "rate", "beta": 5., "temperature_K": 300., "beads": 2,
        "dt_au": .003, "seed": 17, "native": False, "potential": {"provider": "free"},
        "sampling": {"equilibration_steps": 8, "production_steps": 40,
                     "sample_interval": 2, "interval_steps": 2, "andersen_interval": 4},
        "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                     "breaking_bonds": [[1, 2]], "rinf": 4.},
        "umbrella": {"centers": np.linspace(0., 1., 5).tolist(), "kappa": 10.},
        "analysis": {"bins": 12, "bootstrap": 0},
        "parallel": {"backend": "serial"}, "dynamics": {"loop_backend": "python"},
        "output": {"checkpoint_steps": 12},
        "recrossing": {"parent_replicas": 2, "parent_samples": 2, "equilibration_steps": 6,
                       "parent_interval_steps": 4, "child_pairs": 1, "child_steps": 12,
                       "bootstrap": 0, "dividing_surface": 1.},
    }
    if scoped:
        stage = deepcopy(config)
        stage.update(task="fes", fes={"method": "umbrella"}, dt_au=.006)
        stage["sampling"].update(dt_au=.006, sample_interval=2, interval_steps=2)
        stage.pop("recrossing")
        config["rate"] = {
            "pmf": {"method": "umbrella", "estimator": "ui",
                    "umbrella": config.pop("umbrella"), "dynamics": {"timestep": .006}},
            "pmf_stage": {"schema_version": 1, "sampler": "umbrella", "config": stage},
        }
    return config, system


def frozen_jobs(config, system, *, numerical=()):
    identity = identity_from_config(config, system)
    spec = PrecisionSpec(identity, PrecisionTarget(.2), CheckSchedule((2,)), 1e8,
                         numerical=numerical, seed=41)
    sampling = SamplingDesign(1, 40, 2, 2, 1, 12, 8, 6, 4, .25)
    return spec, freeze_production(spec, sampling, pilot_source_ids=("pilot-source",))


@pytest.mark.parametrize("scoped", [False, True])
def test_materializes_fixed_per_window_and_parent_sampling_without_mutation(scoped):
    config, system = configuration(scoped=scoped)
    original = deepcopy(config)
    _, frozen = frozen_jobs(config, system)
    job = materialize_rate_job(config, system, frozen, frozen.jobs[0]["job_id"])
    assert config == original
    stage = rate_pmf_stage(job["config"], system) or job["config"]
    assert stage["sampling"]["replicas"] == frozen.sampling.umbrella_streams
    assert stage["sampling"]["production_steps"] == 40
    assert stage["sampling"]["equilibration_steps"] == 8
    assert job["config"]["recrossing"]["parent_replicas"] == 2
    assert job["config"]["seed"] == frozen.jobs[0]["seed"] != config["seed"]
    assert job["run_kwargs"] == {"resume": False}
    assert job["provenance"]["independent_initialization"] is True
    assert job["config"]["_precision_provenance"] == job["provenance"]
    assert identity_from_config(job["config"], system).identity_id == job["scientific_identity"]


@pytest.mark.parametrize("scoped", [False, True])
def test_timestep_comparison_preserves_all_physical_clocks(scoped):
    config, system = configuration(scoped=scoped)
    _, frozen = frozen_jobs(config, system, numerical=(NumericalRequirement("timestep", .0015, .1),))
    baseline, candidate = frozen.jobs[0], next(j for j in frozen.jobs if j["group"] != "baseline")
    first = materialize_rate_job(config, system, frozen, baseline["job_id"])
    second = materialize_rate_job(config, system, frozen, candidate["job_id"])
    a, b = first["config"], second["config"]
    sa, sb = rate_pmf_stage(a, system) or a, rate_pmf_stage(b, system) or b
    for key in ("equilibration_steps", "production_steps", "sample_interval", "andersen_interval"):
        assert sa["sampling"][key] * sa["dt_au"] == pytest.approx(sb["sampling"][key] * sb["dt_au"])
    for key in ("equilibration_steps", "parent_interval_steps", "child_steps"):
        assert a["recrossing"][key] * a["dt_au"] == pytest.approx(b["recrossing"][key] * b["dt_au"])
    assert a["output"]["checkpoint_steps"] * a["dt_au"] == pytest.approx(b["output"]["checkpoint_steps"] * b["dt_au"])
    assert first["identity"]["model_id"] == second["identity"]["model_id"]
    assert first["identity"]["bias_id"] == second["identity"]["bias_id"]
    assert first["scientific_identity"] != second["scientific_identity"]


def test_identity_excludes_sampling_budgets_but_pins_bias_temperature_and_model(tmp_path):
    config, system = configuration()
    original = identity_from_config(config, system)
    changed = deepcopy(config)
    changed["seed"] += 1
    changed["sampling"].update(production_steps=80, equilibration_steps=18, replicas=3)
    changed["recrossing"].update(parent_replicas=8, child_pairs=9, child_steps=40)
    assert identity_from_config(changed, system) == original
    changed["umbrella"]["kappa"] *= 2
    assert identity_from_config(changed, system).bias_id != original.bias_id
    changed = deepcopy(config)
    changed["beta"] *= 2
    assert identity_from_config(changed, system).model_id != original.model_id
    changed = deepcopy(config)
    changed["beads"] *= 2
    assert identity_from_config(changed, system).model_id == original.model_id
    assert identity_from_config(changed, system).identity_id != original.identity_id
    potential = tmp_path / "potential.py"
    potential.write_text("a = 1\n")
    changed = deepcopy(config)
    changed["potential"] = {"provider": "python", "factory": str(potential) + ":make"}
    first = identity_from_config(changed, system)
    potential.write_text("a = 2\n")
    assert identity_from_config(changed, system).model_id != first.model_id


@pytest.mark.parametrize("mode", ["automatic", "adaptive", "source", "trpmd", "window_override"])
def test_unsupported_production_routes_fail_before_submission(mode):
    config, system = configuration()
    if mode == "automatic":
        config["recrossing"].pop("dividing_surface")
    elif mode == "adaptive":
        config["adaptive_umbrella"] = {"enabled": True}
    elif mode == "source":
        config["rate"] = {"pmf_source": "/previous/run"}
    elif mode == "trpmd":
        config["recrossing"]["dynamics"] = "trpmd"
    else:
        config["umbrella"] = {"windows": [{"center": 0., "sampling": {"production_steps": 30}},
                                           {"center": 1.}], "force_constant": .1}
    with pytest.raises(ValueError):
        identity_from_config(config, system)


def test_exact_time_conversion_rejects_noninteger_save_cadence():
    config, system = configuration()
    config["sampling"].update(sample_interval=3, interval_steps=3)
    _, frozen = frozen_jobs(config, system, numerical=(NumericalRequirement("timestep", .006, .1),))
    candidate = next(j for j in frozen.jobs if j["group"] != "baseline")
    with pytest.raises(ValueError, match="physical duration"):
        materialize_rate_job(config, system, frozen, candidate["job_id"])


@pytest.mark.parametrize("scoped", [False, True])
def test_real_rate_executor_round_trip_and_failed_diagnostics_are_retained(tmp_path, scoped):
    config, system = configuration(scoped=scoped)
    spec, frozen = frozen_jobs(config, system)
    records = []
    for planned in frozen.jobs:
        job = materialize_rate_job(config, system, frozen, planned["job_id"])
        directory = tmp_path / planned["job_id"]
        run_resolved(job["config"], system, output=directory, **job["run_kwargs"])
        records.append(record_from_rate_job(directory, job, frozen))
    evidence = evidence_from_replicas(spec, frozen, records)
    assert evidence.phase == "production"
    assert evidence.replica_ids == tuple(j["job_id"] for j in frozen.jobs)
    assert len(evidence.source_ids) == 2
    assert evidence.covariance is not None
    assert evidence.plateau_established is False  # Deliberately tiny, no bootstrap.
    assert evidence.independent_from_pilot is True
    directory = tmp_path / frozen.jobs[0]["job_id"]
    job = materialize_rate_job(config, system, frozen, frozen.jobs[0]["job_id"])
    summary_path = directory / "summary.json"
    original_summary = summary_path.read_text()
    summary = json.loads(original_summary)
    summary["kappa"] = None
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="saved kappa"):
        record_from_rate_job(directory, job, frozen)
    summary_path.write_text(original_summary)
    summary["kappa"] = json.loads(original_summary)["kappa"]
    summary["child_pairs"] -= 1
    summary["energy_diagnostic_child_pairs"] = summary["child_pairs"]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete the frozen"):
        record_from_rate_job(directory, job, frozen)
    summary_path.write_text(original_summary)
    # Actual saved input identity, rather than a caller-provided label, binds a job.
    resolved_path = directory / "input/resolved.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["seed"] += 1
    resolved_path.write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="provenance"):
        record_from_rate_job(directory, job, frozen)
