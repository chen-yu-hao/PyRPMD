"""Sampling owns trajectories; rate analysis owns fits, diagnostics and errors."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import numpy as np
import pytest

from pyrpmd.analysis import fes as analysis
from pyrpmd.analysis.fes_result import FESResult
from pyrpmd.input import InputError, JobSpec, rate_pmf_options, rate_pmf_stage
from pyrpmd.io import fingerprint
from pyrpmd.workflows import fes, rate
from pyrpmd.workflows.rate_umbrella_analysis import mbar_umbrella_diagnostics

from test_adaptive_umbrella import _config, _system
from test_rate_pmf_extend import configuration
from test_rate_pmf_input import configuration as input_configuration


def _rate_case(tmp_path, method, bootstrap=3):
    raw = configuration()
    raw["rate"]["pmf"]["estimator"] = method
    raw["analysis"].update(bins=20, ranges=[[-.5, 1.5]], bootstrap=bootstrap)
    config, system = JobSpec.from_mapping(raw).resolve()
    config = rate._prepare(config, system, tmp_path)
    stage = rate_pmf_stage(config, system)
    rng = np.random.default_rng(951)
    streams = []
    for window, center in enumerate(stage["umbrella"]["centers"]):
        # Unequal replica counts exercise exact expansion of pooled overlaps.
        for replica, count in enumerate((250, 400)):
            samples = rng.normal(center, .17, count)[:, None]
            streams.append({
                "window": window, "replica": replica, "samples": samples,
                "center": np.array([center]), "stiffness": np.array([[10.]]),
                "last_q": np.repeat(system.geometry[None], config["beads"], axis=0),
                "energies": np.zeros(count),
            })
    return config, system, stage, streams


def _bundle(stage, streams, output):
    return {"streams": iter(streams), "sampling_config": stage,
            "output": str(output), "postprocess": None}


@pytest.mark.parametrize("threaded", [False, True])
def test_sampling_has_no_production_estimator_or_fes_artifacts(tmp_path, monkeypatch, threaded):
    config = _config()
    config.pop("adaptive_umbrella")
    config["analysis"].update(method="not-a-sampling-option", bootstrap=40)
    before = deepcopy(config)

    def forbidden(*args, **kwargs):
        pytest.fail("The sampler must not fit or reconstruct a production FES")

    monkeypatch.setattr(fes, "reconstruct_fes", forbidden)
    monkeypatch.setattr(analysis, "solve_mbar", forbidden)
    if threaded:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sampled = fes.sample_umbrellas(config, _system(), tmp_path, executor=executor)
            streams = list(sampled["streams"])
    else:
        sampled = fes.sample_umbrellas(config, _system(), tmp_path)
        streams = list(sampled["streams"])
    assert config == before
    assert "data" not in sampled and "summary" not in sampled
    assert len(streams) == 2
    assert all(len(s["samples"]) == 40 for s in streams)
    assert len(list((tmp_path / "checkpoints").glob("*.npz"))) == 2
    assert not list(tmp_path.glob("fes.*"))
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize("method", ["ui", "mbar"])
@pytest.mark.parametrize("location", ["omitted", "root", "local"])
def test_rate_estimator_is_the_only_selector_and_legacy_mbar_is_optional(method, location):
    raw = input_configuration()
    raw["rate"] = {"pmf": {"method": "umbrella", "estimator": method}}
    if location == "root":
        raw["analysis"]["method"] = "mbar"
    elif location == "local":
        raw["rate"]["pmf"]["analysis"] = {"method": "mbar"}
    before = deepcopy(raw)
    config, system = JobSpec.from_mapping(raw).resolve()
    stage = rate_pmf_stage(config, system)
    assert raw == before
    assert "method" not in stage["analysis"]
    assert stage["analysis"]["measure"] == "rate_flux"
    assert rate_pmf_options(config)["method"] == method


def test_misplaced_rate_estimator_has_a_migration_error():
    raw = input_configuration()
    raw["rate"] = {"pmf": {"method": "umbrella", "analysis": {"method": "ui"}}}
    with pytest.raises(InputError, match=r"rate.pmf.estimator, not rate.pmf.analysis.method"):
        JobSpec.from_mapping(raw).resolve()


def test_legacy_persisted_stage_keeps_stream_identity(tmp_path):
    config, system, stage, _ = _rate_case(tmp_path, "ui")
    legacy = deepcopy(config)
    legacy["rate"]["pmf_stage"]["config"]["analysis"]["method"] = "mbar"
    old_stage = rate_pmf_stage(legacy, system)
    identity = {"window": 0, "replica": 0, "center": np.array([0.]),
                "stiffness": np.array([[10.]])}
    assert fingerprint(stage, system, identity) == fingerprint(old_stage, system, identity)
    # Never rewrite the user's historical resolved input on read.
    assert old_stage["analysis"]["method"] == "mbar"
    assert legacy["rate"]["pmf_stage"]["config"]["analysis"]["method"] == "mbar"


def test_ui_keeps_numerics_and_overlap_but_only_bootstraps_the_rate(tmp_path, monkeypatch):
    config, system, stage, streams = _rate_case(tmp_path, "ui")
    options = rate_pmf_options(config)
    old_data = analysis.reconstruct_fes(
        [s["samples"] for s in streams], [s["center"] for s in streams],
        [s["stiffness"] for s in streams], config["beta"], bins=20,
        ranges=[[-.5, 1.5]], bootstrap=options["bootstrap"],
    )
    expected = rate._ui_rate_pmf(
        {"data": old_data}, config, np.array([s["center"][0] for s in streams]),
        [s["samples"][:, 0] for s in streams], np.full(len(streams), 10.), options,
    )
    calls = []
    solve = analysis.solve_mbar

    def counted(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(analysis, "solve_mbar", counted)
    pmf, tasks = rate._pmf_and_parent_tasks(
        _bundle(stage, streams, tmp_path / "umbrella"), config, system,
        str(tmp_path), rate._recrossing_options(config),
    )
    assert len(calls) == 1  # overlap diagnostic only; no FES bootstrap fits
    assert pmf["ready"] and tasks
    assert pmf["free_energy_barrier_hartree"] == expected["free_energy_barrier_hartree"]
    np.testing.assert_array_equal(pmf["pmf_bootstrap_barriers"], expected["pmf_bootstrap_barriers"])
    assert pmf["intermediate_fes_bootstrap"] == 0
    summary = json.loads((tmp_path / "umbrella/summary.json").read_text())
    assert summary["bootstrap_requested"] == summary["bootstrap_attempted"] == 0
    assert summary["analysis_owner"] == "rate" and summary["rate_estimator"] == "ui"


def test_mbar_fit_and_bootstrap_are_not_repeated_for_diagnostic_exports(tmp_path, monkeypatch):
    config, system, stage, streams = _rate_case(tmp_path, "mbar")
    calls = []
    solve = analysis.solve_mbar

    def counted(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(analysis, "solve_mbar", counted)
    pmf, _ = rate._pmf_and_parent_tasks(
        _bundle(stage, streams, tmp_path / "umbrella"), config, system,
        str(tmp_path), rate._recrossing_options(config),
    )
    assert pmf["ready"]
    assert len(calls) == 1 + pmf["bootstrap"]["attempted"] == 4
    assert pmf["bootstrap"]["requested"] == 3
    saved = FESResult.read(tmp_path / "umbrella/fes.npz", summary={})
    assert saved.metadata["analysis_owner"] == "rate"
    assert saved.metadata["rate_estimator"] == "mbar"
    assert saved.metadata["bootstrap_requested"] == 0
    assert np.isnan(saved.uncertainty).all()
    for name in ("fes.h5", "fes.npz", "fes.csv", "summary.json"):
        assert (tmp_path / "umbrella" / name).is_file()


def test_pooled_mbar_overlap_and_histogram_match_independent_generic_fit(tmp_path):
    config, _, _, streams = _rate_case(tmp_path, "mbar", bootstrap=0)
    estimates = analysis.mbar_rate_pmf(
        [s["samples"][:, 0] for s in streams], [s["center"][0] for s in streams],
        [s["stiffness"][0, 0] for s in streams], config["beta"],
        bins=20, ranges=[-.5, 1.5], bootstrap=0, xi_star=1.,
    )
    actual = mbar_umbrella_diagnostics(estimates, streams, config["beta"], 1)
    expected = analysis.reconstruct_fes(
        [s["samples"] for s in streams], [s["center"] for s in streams],
        [s["stiffness"] for s in streams], config["beta"],
        bins=20, ranges=[[-.5, 1.5]], bootstrap=0,
    )
    assert estimates["solver"]["unique_umbrella_hamiltonians"] == len(streams) // 2
    for key in ("overlap", "weights", "free_energy", "density", "probability_mass",
                "raw_count", "mask", "window_free_energies", "statistical_inefficiency"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-10, err_msg=key)
    np.testing.assert_array_equal(actual["components"], expected["components"])


@pytest.mark.parametrize("method", ["ui", "mbar"])
def test_disconnected_sampling_never_launches_parents(tmp_path, method):
    config, system, stage, streams = _rate_case(tmp_path, method, bootstrap=0)
    rng = np.random.default_rng(777)
    for stream in streams:
        stream["samples"] = rng.normal(stream["center"][0], 1e-5, (50, 1))
        stream["stiffness"][:] = 1e8
        stream["energies"] = np.zeros(50)
    pmf, tasks = rate._pmf_and_parent_tasks(
        _bundle(stage, streams, tmp_path / "umbrella"), config, system,
        str(tmp_path), rate._recrossing_options(config),
    )
    assert not pmf["ready"] and not tasks
    assert pmf["status"] == "insufficient_overlap"
    assert pmf["k_qtst_au"] is None
    surface = FESResult.read(tmp_path / "umbrella/fes.npz", summary={})
    assert surface.mask.all()


def test_generic_fes_still_owns_its_requested_bootstrap(tmp_path, monkeypatch):
    config = _config(2)
    config.pop("adaptive_umbrella")
    config["analysis"]["bootstrap"] = 3
    calls = []
    solve = analysis.solve_mbar

    def counted(*args, **kwargs):
        calls.append(1)
        return solve(*args, **kwargs)

    monkeypatch.setattr(analysis, "solve_mbar", counted)
    result = fes.run_fes(config, _system(), tmp_path, return_data=True)
    assert result["summary"]["estimator"] == "mbar"
    assert result["summary"]["bootstrap_requested"] == 3
    assert result["summary"]["bootstrap_attempted"] == 3
    assert len(calls) == 4
    assert "analysis_owner" not in result["summary"]


@pytest.mark.parametrize("method", ["ui", "mbar"])
def test_adaptive_rate_production_provenance_survives_deferred_analysis(tmp_path, method):
    from pyrpmd.runner import run_resolved, resume_run

    raw = configuration(production=120)
    raw["rate"]["pmf"]["estimator"] = method
    raw["rate"]["pmf"]["adaptive_umbrella"] = {
        "enabled": True, "pilot_steps": 100, "pilot_equilibration_steps": 10,
        "max_rounds": 2, "target_overlap": 1e-8, "min_effective_samples": .1,
    }
    raw["analysis"].update(bins=5, ranges=[[-.5, 1.5]])
    config, system = JobSpec.from_mapping(raw).resolve()
    result = run_resolved(config, system, output=tmp_path)
    production = result["pmf"]["adaptive_umbrella"]
    assert production["pilot_samples_used_in_production"] == 0
    assert production["production_steps_per_stream"] == 120
    assert "production_analysis_overlap_connected" in production
    for name in ("summary.json", "fes.json"):
        summary = json.loads((tmp_path / "umbrella" / name).read_text())
        assert summary["adaptive_umbrella"] == production
        assert summary["rate_estimator"] == method
        assert summary["bootstrap_requested"] == 0
    surface = FESResult.read(tmp_path / "umbrella/fes.npz", summary={})
    assert surface.metadata["adaptive_umbrella"] == production
    before = {path: path.read_bytes() for path in (tmp_path / "umbrella/checkpoints").glob("*.npz")}
    resumed = resume_run(tmp_path)
    assert resumed["pmf"]["adaptive_umbrella"] == production
    assert resumed["pmf"]["free_energy_barrier_hartree"] == result["pmf"]["free_energy_barrier_hartree"]
    assert all(path.read_bytes() == data for path, data in before.items())


def test_nonconverged_mbar_fit_masks_exports_and_withholds_rate(tmp_path, monkeypatch):
    from dataclasses import replace

    config, system, stage, streams = _rate_case(tmp_path, "mbar")
    solve = analysis.solve_mbar

    def nonconverged(*args, **kwargs):
        return replace(solve(*args, **kwargs), converged=False, residual=1.)

    monkeypatch.setattr(analysis, "solve_mbar", nonconverged)
    pmf, tasks = rate._pmf_and_parent_tasks(
        _bundle(stage, streams, tmp_path / "umbrella"), config, system,
        str(tmp_path), rate._recrossing_options(config),
    )
    assert not pmf["ready"] and not tasks
    assert pmf["status"] == "pmf_solver_failed"
    assert pmf["failure_reason"] == "mbar_not_converged"
    assert pmf["bootstrap"]["attempted"] == 0
    saved = FESResult.read(tmp_path / "umbrella/fes.npz", summary={})
    assert saved.mask.all()
    assert saved.metadata["status"] == "mbar_not_converged"


def test_ui_diagnostic_solver_failure_is_not_silently_ignored(tmp_path, monkeypatch):
    from dataclasses import replace

    config, system, stage, streams = _rate_case(tmp_path, "ui")
    solve = analysis.solve_mbar

    def nonconverged(*args, **kwargs):
        return replace(solve(*args, **kwargs), converged=False, residual=1.)

    monkeypatch.setattr(analysis, "solve_mbar", nonconverged)
    with pytest.raises(RuntimeError, match="MBAR did not converge"):
        rate._pmf_and_parent_tasks(
            _bundle(stage, streams, tmp_path / "umbrella"), config, system,
            str(tmp_path), rate._recrossing_options(config),
        )
    assert not (tmp_path / "rate_pmf.json").exists()


@pytest.mark.parametrize("method", ["ui", "wham", "misspelled"])
def test_legacy_invalid_analysis_method_fails_before_sampling(method):
    with pytest.raises(InputError, match="select the rate estimator"):
        rate_pmf_options({"analysis": {"method": method}})


@pytest.mark.parametrize("legacy_method", ["mbar", "umbrella_integration"])
@pytest.mark.parametrize("estimator", ["ui", "mbar"])
def test_legacy_analysis_field_never_overrides_rate_estimator(legacy_method, estimator):
    options = rate_pmf_options({"analysis": {"method": legacy_method},
                                "rate": {"pmf_method": estimator}})
    assert options["method"] == estimator
