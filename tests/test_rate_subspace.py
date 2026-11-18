"""Native .inp rate/subspace orchestration, source integrity and cache contracts.

Numerical smoke cases are deliberately small; no test asserts physical
convergence. Existing subspace solver tests cover the underlying estimators.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.input_subspace import normalize_rate_subspace, rate_subspace_options
from pyrpmd.io import file_sha256, fingerprint, write_json
from pyrpmd.propagation import scientific_config
from pyrpmd.runner import _initialize_output, run, run_resolved
from pyrpmd.workflows import rate_subspace as stage

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "rate"


def spec():
    result = JobSpec.from_file(EXAMPLES / "research.inp")
    result.data["rate"]["subspace_correction"] = {"enabled": True}
    return result


def test_native_input_enables_moments_and_preserves_physics():
    job = spec()
    original = deepcopy(job.data)
    resolved, system = job.resolve()
    assert job.data == original
    assert resolved["output"]["centroid_path"] is True
    assert resolved["rate"]["subspace_correction"]["mode"] == "apply"
    del job.data["rate"]["subspace_correction"]
    old, old_system = job.resolve()
    assert fingerprint(scientific_config(resolved), system) == fingerprint(scientific_config(old), old_system)
    # The independent PMF configuration must not gain a physical fingerprint.
    new_stage = resolved["rate"]["pmf_stage"]["config"]
    old_stage = old["rate"]["pmf_stage"]["config"]
    assert fingerprint(scientific_config(new_stage), system) == fingerprint(scientific_config(old_stage), old_system)


@pytest.mark.parametrize("options, message", [
    (True, "must be a table"),
    ({"enabled": "true"}, "boolean"),
    ({"reuse": 1}, "boolean"),
    ({"dimension": 3}, "dimension"),
    ({"dimension": True}, "dimension"),
    ({"dimension": []}, "empty"),
    ({"dimension": [1, 1]}, "duplicate"),
    ({"mode": "additive"}, "mode"),
    ({"environment": "relaxed"}, "environment"),
    ({"threshold": float("nan")}, "threshold"),
    ({"path": {"max_steps": 2.5}}, "max_steps"),
    ({"quantum": {"spacing_2d": [1]}}, "two spacings"),
    ({"quantum": {"dense_limit": False}}, "dense_limit"),
    ({"quantum": {"absorber_fraction": .5}}, "absorber_fraction"),
    ({"quantum": {"truncation_min": -1}}, "truncation_min"),
    ({"rpmd": {"windows": 3}}, "windows"),
    ({"rpmd": {"walkers": 1}}, "walkers"),
    ({"rpmd": {"nbeads": 16}}, "Unknown"),
    ({"rpmd": {"plateau_fraction": 2}}, "plateau_fraction"),
    ({"source": {"rate_run": "elsewhere"}}, "Unknown"),
    ({"quantum": {"refinment": 3}}, "Did you mean"),
    ({"execution": {"cpus": 0}}, "cpus"),
    ({"execution": {"cpus": "all"}}, "cpus"),
    ({"execution": {"workers": 1.5}}, "workers"),
    ({"execution": {"memory_limit_mb": -1}}, "memory_limit_mb"),
    ({"execution": {"threads": 4}}, "Unknown"),
])
def test_bad_options_fail_during_resolution(options, message, monkeypatch):
    job = spec()
    job.data["rate"]["subspace_correction"] = options
    monkeypatch.setattr("pyrpmd.potentials.make_potential", lambda *a, **k: pytest.fail("PES evaluated"))
    with pytest.raises(InputError, match=message):
        job.resolve()


@pytest.mark.parametrize("dimension", [1, 2, [1, 2], "auto"])
def test_supported_dimensions(dimension):
    c = {"task": "rate", "rate": {"subspace_correction": {"enabled": True, "dimension": dimension}}}
    assert rate_subspace_options(c)["dimension"] == dimension


def test_non_umbrella_and_other_tasks_rejected():
    for pmf in ({"method": "metadynamics"}, {"method": "opes"}, {"method": "external"}):
        c = {"task": "rate", "rate": {"pmf": pmf, "subspace_correction": {"enabled": True}}}
        with pytest.raises(InputError, match="locally sampled umbrella"):
            normalize_rate_subspace(c)
    c = {"task": "pimd", "rate": {"subspace_correction": {"enabled": True}}}
    with pytest.raises(InputError, match="requires task='rate'"):
        normalize_rate_subspace(c)


def test_disabled_and_absent_do_not_enable_statistics():
    for raw in ({}, {"enabled": False}):
        config = {"task": "rate", "rate": {"subspace_correction": raw}}
        normalize_rate_subspace(config)
        assert not config.get("output", {}).get("centroid_path")
    old = {"task": "rate"}
    normalize_rate_subspace(old)
    assert old == {"task": "rate"}


def saved_run(tmp_path, *, mode="apply"):
    job = spec().with_overrides(output=tmp_path / "run")
    job.data["rate"]["subspace_correction"]["mode"] = mode
    config, system = job.resolve()
    root = Path(config["output"]["directory"])
    _initialize_output(config, system, root)
    summary = {"task": "rate", "status": "pilot_requires_convergence", "rate_cm3_s": None,
               "rate_estimate_cm3_s": 2e-12, "rate_standard_error_cm3_s": 1e-13}
    write_json(root / "summary.json", summary)
    write_json(root / "results" / "summary.json", summary)
    write_json(root / "preparation.json", {"reaction": config["reaction"]})
    write_json(root / "rate_pmf.json", {"xi_star": 1.01})
    return config, system, root


def source_hashes(root):
    return {str(p.relative_to(root)): file_sha256(p) for p in root.rglob("*")
            if p.is_file() and "subspace" not in p.relative_to(root).parts
            and p.relative_to(root).as_posix() not in {"progress.json", "run.log.jsonl"}}


def fake_report(apply=True):
    return {"dimensions_computed": [1], "applied_dimension": 1,
            "corrections": {"1": {"factor": 1.5}},
            "corrected_rate": {"corrected_rate_cm3_s": None,
                               "corrected_rate_estimate_cm3_s": 3e-12,
                               "corrected_rate_standard_error_cm3_s": 2e-13} if apply else None}


@pytest.mark.parametrize("mode", ["apply", "factor_only"])
def test_stage_routes_shared_pipeline_and_preserves_sources(tmp_path, monkeypatch, mode):
    config, system, root = saved_run(tmp_path, mode=mode)
    before = source_hashes(root)
    monkeypatch.setattr(stage, "_check_centroid_path", lambda *a: None)
    calls = []

    def pipeline(specification, **kwargs):
        calls.append((specification, kwargs))
        return fake_report(kwargs["apply"])

    monkeypatch.setattr("pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline", pipeline)
    monkeypatch.setattr("pyrpmd.runner._initialize_output", lambda *a, **k: pytest.fail("initialized output"))
    monkeypatch.setattr("pyrpmd.workflows.rate.run_rate", lambda *a, **k: pytest.fail("re-ran RPMD"))
    result = run_resolved(config, system, stage="subspace")
    assert result["subspace_correction"]["status"] == "complete"
    assert ("corrected_rate_cm3_s" in result["subspace_correction"]) == (mode == "apply")
    assert source_hashes(root) == before
    progress = json.loads((root / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["stage"] == "subspace"
    assert progress["total_steps"] is None
    assert calls[0][0]["source"]["rate_run"] == str(root)
    assert calls[0][1]["apply"] == (mode == "apply")
    assert len(calls[0][1]["reuse_key"]) == 64


def test_failure_is_recorded_without_losing_rate_and_explicit_stage_raises(tmp_path, monkeypatch):
    config, _, root = saved_run(tmp_path)
    config["rate"]["subspace_correction"]["path"] = {"source": "centroid"}
    before = source_hashes(root)
    summary = json.loads((root / "summary.json").read_text())
    with pytest.warns(RuntimeWarning, match="uncorrected RPMD result is intact"):
        result = stage.finish_rate(config, root, summary)
    assert result["rate_estimate_cm3_s"] == 2e-12
    assert result["subspace_correction"]["status"] == "failed"
    assert "centroid_path.npz" in result["subspace_correction"]["error"]["message"]
    with pytest.raises(ValueError, match="centroid_path.npz"):
        stage.postprocess_rate_subspace(config, root, strict=True)
    assert source_hashes(root) == before
    assert json.loads((root / "subspace" / "status.json").read_text())["status"] == "failed"


def test_stage_rejects_changed_physics_before_pipeline_or_writes(tmp_path, monkeypatch):
    config, system, root = saved_run(tmp_path)
    before = source_hashes(root)
    monkeypatch.setattr(stage, "postprocess_rate_subspace", lambda *a, **k: pytest.fail("ran pipeline"))
    for key in ("beads", "beta", "seed"):
        changed = deepcopy(config)
        changed[key] += 1
        with pytest.raises(InputError, match="scientific settings"):
            stage.run_subspace_stage(changed, system, root)
    assert source_hashes(root) == before
    assert not (root / "subspace").exists()


def test_input_can_change_correction_but_pes_bytes_still_checked(tmp_path):
    config, system, root = saved_run(tmp_path)
    changed = deepcopy(config)
    changed["rate"]["subspace_correction"]["quantum"]["truncation_min"] = 0.12
    assert stage.validate_saved_rate(changed, system, root) == root
    model = tmp_path / "model.dat"
    model.write_text("original")
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][str(model)] = file_sha256(model)
    write_json(path, manifest)
    with pytest.raises(InputError, match="Input/model file"):
        stage.validate_saved_rate(changed, system, root)


@pytest.mark.parametrize("kwargs", [{"resume": True}, {"init_from": "old"}, {"init_index": 1}, {"init_bias": True}])
def test_stage_rejects_resume_and_warm_start(kwargs, tmp_path):
    config, system, _ = saved_run(tmp_path)
    with pytest.raises(InputError, match="cannot be combined"):
        run_resolved(config, system, stage="subspace", **kwargs)


def test_stage_requires_enabled_and_saved_rate(tmp_path):
    config, system, root = saved_run(tmp_path)
    config["rate"]["subspace_correction"]["enabled"] = False
    with pytest.raises(InputError, match="enabled=true"):
        stage.run_subspace_stage(config, system, root)
    config["rate"]["subspace_correction"]["enabled"] = True
    (root / "results" / "extensions.json").write_text("{}")
    with pytest.raises(InputError, match="extension"):
        stage.run_subspace_stage(config, system, root)


def test_centroid_metadata_and_complete_moments_are_checked(tmp_path, monkeypatch):
    config, _, root = saved_run(tmp_path)
    mean = SimpleNamespace(beta=config["beta"], nbeads=config["beads"],
                           complete=np.ones(5, bool), count=np.full(5, 10), locked=lambda: np.ones(5, bool))
    monkeypatch.setattr("pyrpmd.analysis.centroid_path.mean_path_from_rate_run", lambda *a: mean)
    stage._check_centroid_path(root, config)
    mean.complete[0] = False
    with pytest.raises(InputError, match="incomplete"):
        stage._check_centroid_path(root, config)
    mean.complete[:] = True
    mean.nbeads += 1
    with pytest.raises(InputError, match="bead count"):
        stage._check_centroid_path(root, config)


def test_identity_invalidates_on_numerics_or_samples_not_apply_mode(tmp_path):
    config, _, root = saved_run(tmp_path)
    options = rate_subspace_options(config)
    original = stage._source_identity(root, options)
    options.update(mode="factor_only", reuse=False)
    assert stage._source_identity(root, options) == original
    options["rpmd"]["sweeps"] = 123
    assert stage._source_identity(root, options) != original
    options["rpmd"].clear()
    (root / "centroid_path.npz").write_bytes(b"changed samples")
    assert stage._source_identity(root, options) != original


def test_native_run_rate_invokes_postprocess_after_aggregation(tmp_path, monkeypatch):
    import pyrpmd.workflows.rate as rate

    config, system, root = saved_run(tmp_path)
    # Exercise the real native rate control flow, replacing only the costly
    # sampling/estimation boundaries with observable fixtures.
    calls = []
    monkeypatch.setattr(rate, "_prepare", lambda c, s, *a: (c, s))
    monkeypatch.setattr(rate, "sample_umbrellas", lambda *a, **k: calls.append("sampling") or {})
    monkeypatch.setattr(rate, "_pmf_and_parent_tasks", lambda *a: ({}, []))
    monkeypatch.setattr(rate, "_aggregate", lambda *a: calls.append("aggregate") or {"task": "rate"})
    monkeypatch.setattr(stage, "postprocess_rate_subspace", lambda *a, **k: calls.append("subspace") or {"status": "complete"})
    result = rate.run_rate(config, system, root)
    assert calls == ["sampling", "aggregate", "subspace"]
    assert result["subspace_correction"]["status"] == "complete"


def test_cli_stage_uses_same_runner_and_input(monkeypatch, capsys):
    from pyrpmd.cli import main, parser

    assert parser().parse_args(["run", "job.inp", "--stage", "subspace"]).stage == "subspace"
    calls = []
    monkeypatch.setattr("pyrpmd.runner.run", lambda job, **kwargs: calls.append((job, kwargs)) or
                        {"task": "rate", "output": "saved", "stage": "subspace"})
    assert main(["run", str(EXAMPLES / "research.inp"), "--stage", "subspace"]) == 0
    assert calls[0][1]["stage"] == "subspace"
    assert isinstance(calls[0][0], JobSpec)
    assert json.loads(capsys.readouterr().out)["stage"] == "subspace"


def test_small_real_factor_pipeline_and_verified_reuse(tmp_path, monkeypatch):
    """Actual DVR + P=1 RPMD solve; only full-system projection is substituted."""
    from pyrpmd.analysis import subspace_pipeline as pipe
    from pyrpmd.analysis import model_from_function

    mass, v0, width = 1060., .425 / 27.211386, .66
    model = model_from_function(lambda p: v0 / np.cosh(p[..., 0] / width) ** 2,
                                [np.linspace(-8., 8., 161)], masses=mass, dividing_surface=0., reactant_reference=-4.)
    closed = []
    fake_run = SimpleNamespace(root=tmp_path, provider=SimpleNamespace(close=lambda: closed.append(True)),
                               beta=1 / (3.166811563e-6 * 1000), nbeads=1, xi_star=1.)
    monkeypatch.setattr(pipe, "load_rate_run", lambda *a: fake_run)
    mean = SimpleNamespace(curvature_width=lambda *a, **k: {"maximum": 0.1})
    monkeypatch.setattr(pipe, "build_path", lambda *a: (mean, None, np.zeros((5, 3)), {}))
    monkeypatch.setattr(pipe, "_static_curvature_width", lambda *a: {"maximum": 0.1})
    monkeypatch.setattr(pipe, "path_model", lambda *a, **k: (model, {}, {}))
    monkeypatch.setattr(pipe, "correct_rate_run", lambda *a, **k: pytest.fail("factor_only applied a rate"))
    request = {"source": {"rate_run": str(tmp_path)}, "subspace": {"dimension": 1},
               "rpmd": {"windows": 12, "walkers": 2, "burn_in": 0, "sweeps": 2, "trajectories": 8},
               "output": {"reuse": True}}
    report = pipe.run_subspace_pipeline(request, apply=False, reuse_key="same")
    assert report["corrections"]["1"]["factor"] > 0
    assert report["corrected_rate"] is None
    assert closed == [True]
    original = pipe.subspace_correction
    monkeypatch.setattr(pipe, "subspace_correction", lambda *a, **k: pytest.fail("recomputed valid cache"))
    assert pipe.run_subspace_pipeline(request, apply=False, reuse_key="same")["corrections"] == report["corrections"]
    # A changed key OR modified factor file must trigger a new computation.
    for key, tamper in (("same", True), ("different", False)):
        if tamper:
            record = tmp_path / "subspace" / "C_S_1d.json"
            data = json.loads(record.read_text()); data["factor"] *= 2
            write_json(record, data)
        invoked = []
        monkeypatch.setattr(pipe, "subspace_correction", lambda *a, **k: invoked.append(True) or original(*a, **k))
        pipe.run_subspace_pipeline(request, apply=False, reuse_key=key)
        assert invoked == [True]
    assert len(closed) == 4


def test_real_umbrella_run_collects_moments_and_publishes_stage_status(tmp_path, monkeypatch):
    job = spec().with_overrides(output=tmp_path / "native-rate")
    job.data["rate"]["subspace_correction"]["mode"] = "factor_only"
    job.data["umbrella"] = {
        "coordinate": "xi", "centers": [0., .5, .9, 1., 1.05], "force_constant": .00016,
        "sampling": {"equilibration": "0.2 fs", "production": "2 fs", "sample_every": "0.2 fs",
                     "chains_per_window": 1},
    }
    seen = []

    def pipeline(specification, **kwargs):
        source = Path(specification["source"]["rate_run"])
        assert (source / "centroid_path.npz").is_file()
        assert (source / "summary.json").is_file()
        assert kwargs["apply"] is False
        seen.append(source)
        return fake_report(False)

    monkeypatch.setattr("pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline", pipeline)
    result = run(job)
    assert seen == [Path(job.data["output"]["directory"])]
    assert result["subspace_correction"]["status"] == "complete"
    published = json.loads((seen[0] / "results" / "summary.json").read_text())
    raw = json.loads((seen[0] / "summary.json").read_text())
    assert published["subspace_correction"]["correction_factors"] == {"1": 1.5}
    assert published.get("rate_cm3_s") == raw.get("rate_cm3_s")
    assert "subspace_correction" not in raw


def test_modified_inp_correction_settings_can_resume_without_rewriting_manifest(tmp_path):
    job = spec().with_overrides(output=tmp_path / "run")
    inp = tmp_path / "calculation.inp"
    inp.write_text("# source input retained for provenance\n")
    job.input_path = str(inp)
    config, system = job.resolve()
    root = Path(config["output"]["directory"])
    _initialize_output(config, system, root)
    original = source_hashes(root)
    inp.write_text("# only correction numerical settings changed\n")
    config["rate"]["subspace_correction"]["quantum"]["truncation_min"] = 0.1
    _initialize_output(config, system, root, resume=True)
    stage.validate_saved_rate(config, system, root)
    assert source_hashes(root) == original
    changed = deepcopy(config)
    changed["beta"] *= 2
    with pytest.raises(InputError, match="scientific settings"):
        _initialize_output(changed, system, root, resume=True)


def test_standalone_reuse_identity_tracks_path_and_source(tmp_path, monkeypatch):
    from pyrpmd.analysis import subspace_pipeline as pipe

    config, _, root = saved_run(tmp_path)
    closed, identities = [], []
    fake_run = SimpleNamespace(root=root, provider=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(pipe, "load_rate_run", lambda *a: fake_run)
    # The pipeline prepares cases, solves them and writes a report; replace
    # the model building and the report so only the identity binding runs.
    monkeypatch.setattr(pipe, "_build_cases", lambda run, out, options, base, key: (
        identities.append(key) or {"execution": None, "cases": {}}))
    monkeypatch.setattr(pipe, "_finish_pipeline", lambda state, computed, apply: {})
    request = {"source": {"rate_run": str(root)}, "output": {"reuse": True}}
    pipe.run_subspace_pipeline(request, apply=False)
    pipe.run_subspace_pipeline(request, apply=False)
    assert identities[0] == identities[1]
    request["path"] = {"smoothing": 0.2}
    pipe.run_subspace_pipeline(request, apply=False)
    assert identities[-1] != identities[0]
    request.pop("path")
    (root / "centroid_path.npz").write_bytes(b"new samples")
    pipe.run_subspace_pipeline(request, apply=False)
    assert identities[-1] != identities[0]
    assert len(closed) == 4


@pytest.mark.parametrize("extra, message", [
    ({"curvature": "large"}, "curvature"),
    ({"zero_point": "quartic"}, "zero_point"),
    ({"dvr_points": 7}, "dvr_points"),
    ({"dvr_half_width": -1}, "dvr_half_width"),
    ({"path": {"source": "unknown"}}, "source"),
    ({"path": {"least_action_points": 1}}, "least_action_points"),
    ({"path": {"instanton_beads": True}}, "instanton_beads"),
])
def test_projection_controls_reject_invalid_native_input(extra, message):
    job = spec()
    job.data["rate"]["subspace_correction"].update(extra)
    with pytest.raises(InputError, match=message):
        job.resolve()


def test_native_stage_forwards_path_curvature_and_zero_point(tmp_path, monkeypatch):
    config, _, root = saved_run(tmp_path)
    config["rate"]["subspace_correction"].update(
        curvature="small", zero_point="dvr", dvr_points=41, dvr_half_width=2.0,
        path={"source": "least_action", "least_action_points": 5})
    monkeypatch.setattr(stage, "_check_centroid_path", lambda *a: None)
    calls = []

    def pipeline(request, **kwargs):
        calls.append(request)
        return fake_report()

    monkeypatch.setattr("pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline", pipeline)
    result = stage.postprocess_rate_subspace(config, root, strict=True)
    assert result["status"] == "complete"
    assert calls[0]["subspace"]["curvature"] == "small"
    assert calls[0]["subspace"]["zero_point"] == "dvr"
    assert calls[0]["subspace"]["dvr_points"] == 41
    assert calls[0]["subspace"]["dvr_half_width"] == 2.0
    assert calls[0]["path"] == {"source": "least_action", "least_action_points": 5}


@pytest.mark.parametrize("tampered", [False, True])
def test_stage_authenticates_legacy_manifest_without_migrating_source(tmp_path, tampered):
    config, system, root = saved_run(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(schema_version=1, physics_fingerprint=fingerprint(config, system))
    write_json(manifest_path, manifest)
    before = manifest_path.read_bytes()
    if tampered:
        stored = deepcopy(config)
        stored["beta"] *= 2
        write_json(root / "input" / "resolved.json", stored)
        with pytest.raises(InputError, match="Saved rate inputs"):
            stage.validate_saved_rate(config, system, root)
    else:
        assert stage.validate_saved_rate(config, system, root) == root
    assert manifest_path.read_bytes() == before



def test_parent_counters_survive_enabled_postprocessing(tmp_path, monkeypatch):
    from pyrpmd.workflows import rate

    config, system, root = saved_run(tmp_path)
    counters = {"native_steps": 12, "python_steps": 0, "fallback_reasons": {}}
    monkeypatch.setattr(rate, "_prepare", lambda c, s, *a: (c, s))
    monkeypatch.setattr(rate, "sample_umbrellas", lambda *a, **k: {})
    monkeypatch.setattr(rate, "_pmf_and_parent_tasks", lambda *a: ({}, [object()]))
    monkeypatch.setattr(rate, "run_tasks", lambda *a, **k: [{"propagation": counters}])
    monkeypatch.setattr(rate, "_child_tasks", lambda *a: [])
    monkeypatch.setattr(rate, "_dispatch_children", lambda *a, **k: [])

    def aggregate(children, pmf, *args):
        assert pmf["propagation_parents"] == [counters]
        return {"task": "rate", "propagation": counters}

    monkeypatch.setattr(rate, "_aggregate", aggregate)
    monkeypatch.setattr(stage, "postprocess_rate_subspace", lambda *a, **k: {"status": "complete"})
    result = rate.run_rate(config, system, root)
    assert result["propagation"] == counters
    assert result["subspace_correction"]["status"] == "complete"


@pytest.mark.parametrize("parallel, user, expected", [
    ({}, {}, None),
    ({"trajectory_workers": 1, "threads_per_rank": 1}, {}, None),
    ({"backend": "process", "trajectory_workers": 4, "threads_per_rank": 2}, {},
     {"backend": "process", "cpus": 8, "blas_threads": 8}),
    ({"backend": "process", "trajectory_workers": 2, "bead_workers": 3}, {},
     {"backend": "process", "cpus": 6, "blas_threads": 6}),
    # MPI: the stage runs on rank zero only, with that rank's threads.
    ({"backend": "mpi", "trajectory_workers": 16, "threads_per_rank": 4}, {},
     {"backend": "serial", "cpus": 4, "blas_threads": 4}),
    # Never more than this process's affinity.
    ({"backend": "process", "trajectory_workers": 64, "threads_per_rank": 4}, {},
     {"backend": "process", "cpus": 32, "blas_threads": 8}),
    ({"trajectory_workers": 8}, {"cpus": 1}, None),
    ({}, {"cpus": "auto"}, {"backend": "process", "cpus": 32, "blas_threads": 8}),
    ({"backend": "process", "trajectory_workers": 4, "threads_per_rank": 2}, {"cpus": "auto"},
     {"backend": "process", "cpus": 8, "blas_threads": 8}),
    ({"backend": "process", "workers": 4}, {},
     {"backend": "process", "cpus": 4, "blas_threads": 4}),
    ({}, {"cpus": 6, "workers": 2, "blas_threads": 3, "memory_limit_mb": 1000.},
     {"backend": "process", "cpus": 6, "blas_threads": 3, "memory_limit_mb": 1000., "workers": 2}),
])
def test_execution_budget_follows_global_parallel_table(parallel, user, expected, monkeypatch):
    monkeypatch.setattr(stage, "_affinity_cpus", lambda: 32)
    monkeypatch.setattr(stage, "_memory_budget_mb", lambda: 2048.)
    result = stage.subspace_execution({"parallel": parallel}, {"execution": user})
    if expected is None:
        assert result is None
    else:
        assert result == {"memory_limit_mb": 2048., **expected}


def test_resolved_input_parallel_reaches_the_pipeline(tmp_path, monkeypatch):
    job = spec().with_overrides(output=tmp_path / "run")
    job.data["parallel"] = {"trajectory_workers": 2, "threads_per_rank": 2}
    config, system = job.resolve()
    assert config["rate"]["subspace_correction"]["execution"] == {}
    monkeypatch.setattr(stage, "_affinity_cpus", lambda: 64)
    monkeypatch.setattr(stage, "_check_centroid_path", lambda *a: None)
    root = Path(config["output"]["directory"])
    _initialize_output(config, system, root)
    write_json(root / "summary.json", {"task": "rate", "status": "pilot_requires_convergence",
                                       "rate_cm3_s": None, "rate_estimate_cm3_s": 2e-12})
    calls = []
    monkeypatch.setattr("pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline",
                        lambda specification, **k: calls.append(specification) or fake_report())
    status = stage.postprocess_rate_subspace(config, root, strict=True)
    assert calls[0]["execution"]["backend"] == "process"
    assert calls[0]["execution"]["cpus"] == 4
    assert status["execution"]["cpus"] == 4
    # The default single-CPU allocation keeps the sequential route unchanged.
    config["parallel"].update(trajectory_workers=1, trajectory_groups=1, threads_per_rank=1)
    stage.postprocess_rate_subspace(config, root, strict=True)
    assert "execution" not in calls[1]


def test_execution_settings_do_not_change_factor_cache_identity(tmp_path):
    config, _, root = saved_run(tmp_path)
    options = rate_subspace_options(config)
    original = stage._source_identity(root, options)
    options["execution"] = {"cpus": 16, "workers": 2, "blas_threads": 4, "memory_limit_mb": 1e4}
    assert stage._source_identity(root, options) == original


def test_spawned_workers_inherit_passive_openmp_only_while_pool_lives(monkeypatch):
    from pyrpmd.analysis.subspace_execution import _passive_openmp

    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    with _passive_openmp():
        assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
    assert "OMP_WAIT_POLICY" not in os.environ
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    with _passive_openmp():
        assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"


def test_subspace_inp_example_configures_low_temperature_tunnelling_and_parallelism():
    from pyrpmd.analysis.subspace_paths import refine_saddle
    from pyrpmd.constants import KB_AU
    from pyrpmd.instanton import crossover_temperature
    from pyrpmd.potentials import make_potential

    job = JobSpec.from_file(EXAMPLES / "subspace.inp")
    config, system = job.resolve()
    assert config["output"]["centroid_path"] is True
    assert config["beads"] >= 16
    assert config["parallel"]["backend"] == "process"
    assert config["parallel"]["trajectory_workers"] >= 2

    sub = config["rate"]["subspace_correction"]
    assert sub["enabled"] is True
    assert sub["mode"] == "apply"
    assert sub["model_2d"] == "rph"
    assert sub["anchor"] == "skeleton"
    assert sub["quantum"]["truncation_min"] >= 0.06
    assert sub["execution"].get("cpus", "auto") == "auto"

    exec_budget = stage.subspace_execution(config, sub)
    assert exec_budget is not None and exec_budget["backend"] == "process"

    pes = make_potential(config["potential"], system)
    masses_cart = np.repeat(np.asarray(system.masses, dtype=float), 3)
    grad_fn = lambda x: -np.asarray(pes.evaluate(np.asarray(x, dtype=float).reshape(1, -1, 3)).forces[0], dtype=float).ravel()
    saddle_flat, rec = refine_saddle(grad_fn, np.asarray(system.geometry, dtype=float).ravel(), masses_cart,
                                     rigid_body=True, gradient_tolerance=1e-9)
    assert rec["saddle"] is True
    tc_k = crossover_temperature(pes, saddle_flat.reshape(-1, 3), system.masses)
    temp_k = 1.0 / (KB_AU * float(config["beta"]))
    assert temp_k < 0.75 * tc_k


def test_molecular_ts_to_cs_example_script_runs_end_to_end(tmp_path):
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "examples" / "subspace_correction" / "molecular_ts_to_cs.py"
    spec_mod = importlib.util.spec_from_file_location("molecular_ts_to_cs_example", script_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)

    out_json = tmp_path / "cs_9d.json"
    rc = mod.main([
        "--cpus", "1",
        "--beads", "16",
        "--sweeps", "30",
        "--trajectories", "48",
        "--rpmd-rate", "2.5e-16",
        "--rpmd-se", "2.0e-17",
        "--out", str(out_json),
    ])
    assert rc == 0
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["temperature_K"] < summary["crossover_temperature_K"]
    assert summary["saddle"]["rigid_modes_removed"] == 5
    assert summary["validity"]["valid"] is True
    assert summary["validity"]["valid_dimensions"] == [1, 2]
    c1 = summary["corrections"]["1d"]
    c2 = summary["corrections"]["2d_rph"]
    assert c2["anchored_factor"] > 1.5
    assert c1["anchored_factor"] > 1.05
    assert c1["anchor_log_shift"] > abs(c2["anchor_log_shift"]) + 0.03
    assert c1["bead_factor_B_P"] != pytest.approx(c2["bead_factor_B_P"], rel=1e-2)
    assert summary["applied_rate"]["corrected_rate"] > 2.5e-16


def test_apply_mode_fails_fast_before_pipeline_when_run_lacks_rate_estimate(tmp_path, monkeypatch, capsys):
    from pyrpmd.cli import main

    config, system, root = saved_run(tmp_path, mode="apply")
    monkeypatch.setattr(stage, "_check_centroid_path", lambda *a: pytest.fail("checked centroid before rate preflight"))
    monkeypatch.setattr("pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline",
                        lambda *a, **k: pytest.fail("ran expensive subspace pipeline without rate estimate"))

    # Case 1: summary.json exists, but rate_estimate_cm3_s is None.
    no_est = {"task": "rate", "status": "incomplete", "rate_cm3_s": None, "rate_estimate_cm3_s": None}
    write_json(root / "summary.json", no_est)
    write_json(root / "results" / "summary.json", no_est)
    before = source_hashes(root)
    with pytest.raises(ValueError, match="no rate estimate to correct"):
        stage.run_subspace_stage(config, system, root)
    assert json.loads((root / "subspace" / "status.json").read_text())["status"] == "failed"
    assert source_hashes(root) == before

    # CLI subspace-correct also fails immediately before calling run_subspace_pipeline.
    capsys.readouterr()
    toml_path = tmp_path / "correct.toml"
    toml_path.write_text(f'[source]\nrate_run = "{root.as_posix()}"\n', encoding="utf-8")
    assert main(["subspace-correct", str(toml_path)]) != 0
    assert "no rate estimate to correct" in capsys.readouterr().err

    # Case 2: summary.json missing altogether.
    (root / "summary.json").unlink()
    with pytest.raises(ValueError, match="summary.json"):
        stage.postprocess_rate_subspace(config, root, strict=True)

    # Positive counterpart: mode = "factor_only" succeeds without any rate estimate.
    config_fo = deepcopy(config)
    config_fo["rate"]["subspace_correction"]["mode"] = "factor_only"
    monkeypatch.setattr("pyrpmd.analysis.subspace_pipeline.run_subspace_pipeline",
                        lambda spec, **kw: fake_report(apply=False))
    status = stage.postprocess_rate_subspace(config_fo, root, strict=True)
    assert status["status"] == "complete" and "corrected_rate_cm3_s" not in status


def test_mep_source_runs_without_centroid_path_npz(tmp_path):
    # Input normalization allows non-umbrella PMF when path.source == "mep" and mode == "factor_only".
    cfg = {
        "task": "rate",
        "rate": {
            "pmf": {"method": "external"},
            "subspace_correction": {"enabled": True, "mode": "factor_only", "path": {"source": "mep"}},
        },
    }
    opts = normalize_rate_subspace(cfg)
    assert opts["mode"] == "factor_only" and opts["path"]["source"] == "mep"
    assert not cfg.get("output", {}).get("centroid_path")

    # End-to-end --stage subspace with path.source = "mep" and NO centroid_path.npz or rate_pmf.json!
    job = JobSpec.from_file(EXAMPLES / "subspace.inp").with_overrides(output=tmp_path / "mep-ts-only")
    job.data["beads"] = 8
    job.data["parallel"] = {"backend": "serial"}
    job.data["rate"]["subspace_correction"] = {
        "enabled": True,
        "mode": "factor_only",
        "dimension": 1,
        "path": {"source": "mep", "step": 10.0},
        "quantum": {"product_drop_kt": 22.0},
        "rpmd": {
            "windows": 16,
            "walkers": 8,
            "burn_in": 30,
            "sweeps": 60,
            "trajectories": 32,
            "md_time": 600.0,
            "seed": 3,
        },
    }
    config, system = job.resolve()
    root = Path(config["output"]["directory"])
    _initialize_output(config, system, root)
    assert not (root / "centroid_path.npz").exists()
    assert not (root / "rate_pmf.json").exists()
    res = run_resolved(config, system, stage="subspace")
    assert res["subspace_correction"]["status"] == "complete"
    assert res["subspace_correction"]["correction_factors"]["1"] > 0.0
    assert (root / "subspace" / "C_S_1d.json").is_file()


def test_analyze_run_surfaces_posthoc_subspace_results_without_mutating_summary(tmp_path, capsys):
    from pyrpmd.analysis.subspace_correction import CORRECTED_SCHEMA
    from pyrpmd.cli import main
    from pyrpmd.runner import analyze_run

    config, _, root = saved_run(tmp_path, mode="apply")
    raw_summary_bytes = (root / "summary.json").read_bytes()
    pub_summary_bytes = (root / "results" / "summary.json").read_bytes()

    # Without subspace files, analyze_run returns the saved summary as-is.
    plain = analyze_run(root)
    assert "subspace_correction" not in plain and "quantum_corrected_rate" not in plain

    # 1. Post-hoc --stage subspace (status.json + rate_quantum_corrected.json).
    corrected_payload = {
        "schema": CORRECTED_SCHEMA,
        "schema_version": 1,
        "task": "rate",
        "rate_unit": "cm^3 molecule^-1 s^-1",
        "source_directory": str(root),
        "source_status": "pilot_requires_convergence",
        "source_rate_publishable": False,
        "rpmd_rate_estimate_cm3_s": 2e-12,
        "rpmd_rate_standard_error_cm3_s": 1e-13,
        "corrected_rate_estimate_cm3_s": 5.8e-12,
        "corrected_rate_cm3_s": None,
        "corrected_rate_standard_error_cm3_s": 3e-13,
        "correction": {"dimension": 2, "factor": 2.9, "log_standard_error": 0.02, "valid": True},
        "approximation": "multiplicative subspace defect",
    }
    corr_file = root / "subspace" / "rate_quantum_corrected.json"
    write_json(corr_file, corrected_payload)
    status_payload = {
        "schema": "pyrpmd.rate_subspace",
        "schema_version": 1,
        "status": "complete",
        "mode": "apply",
        "directory": str(root / "subspace"),
        "selected_dimension": 2,
        "correction_factors": {"1": 1.1, "2": 2.9},
        "corrected_rate_file": str(corr_file),
        "corrected_rate_cm3_s": None,
        "corrected_rate_estimate_cm3_s": 5.8e-12,
        "corrected_rate_standard_error_cm3_s": 3e-13,
    }
    write_json(root / "subspace" / "status.json", status_payload)

    assert main(["analyze", str(root)]) == 0
    cli_out = json.loads(capsys.readouterr().out)
    assert cli_out["subspace_correction"]["status"] == "complete"
    assert cli_out["subspace_correction"]["selected_dimension"] == 2
    assert cli_out["quantum_corrected_rate"]["corrected_rate_estimate_cm3_s"] == pytest.approx(5.8e-12)
    assert cli_out["quantum_corrected_rate"]["correction_factor"] == pytest.approx(2.9)
    assert (root / "summary.json").read_bytes() == raw_summary_bytes
    assert (root / "results" / "summary.json").read_bytes() == pub_summary_bytes

    # 2. If a later stage retry fails (or switches to factor_only), stale corrected_rate is not surfaced as active.
    failed_status = {**status_payload, "status": "invalid_model"}
    failed_status.pop("corrected_rate_file", None)
    write_json(root / "subspace" / "status.json", failed_status)
    after_fail = analyze_run(root)
    assert after_fail["subspace_correction"]["status"] == "invalid_model"
    assert "quantum_corrected_rate" not in after_fail

    # 3. Standalone pyrpmd subspace-correct (subspace_pipeline.json + rate_quantum_corrected.json, no status.json).
    (root / "subspace" / "status.json").unlink()
    write_json(corr_file, corrected_payload)
    write_json(root / "subspace" / "subspace_pipeline.json", {
        "schema": "pyrpmd.subspace_pipeline",
        "schema_version": 1,
        "status": "complete",
        "mode": "apply",
        "dimensions_computed": [1, 2],
        "applied_dimension": 2,
        "corrections": {"1": {"factor": 1.1}, "2": {"factor": 2.9}},
        "corrected_rate": {
            "task": "rate",
            "corrected_rate_cm3_s": None,
            "corrected_rate_estimate_cm3_s": 5.8e-12,
            "corrected_rate_standard_error_cm3_s": 3e-13,
        },
        "validity": {"valid": True},
        "solver_plans": {},
        "warnings": [],
    })
    standalone = analyze_run(root)
    assert standalone["subspace_correction"]["status"] == "complete"
    assert standalone["subspace_correction"]["selected_dimension"] == 2
    assert standalone["quantum_corrected_rate"]["corrected_rate_estimate_cm3_s"] == pytest.approx(5.8e-12)
    assert (root / "summary.json").read_bytes() == raw_summary_bytes
    assert (root / "results" / "summary.json").read_bytes() == pub_summary_bytes
