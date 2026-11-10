"""Native .inp rate/subspace orchestration, source integrity and cache contracts.

Numerical smoke cases are deliberately small; no test asserts physical
convergence. Existing subspace solver tests cover the underlying estimators.
"""

from copy import deepcopy
import json
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
            if p.is_file() and "subspace" not in p.relative_to(root).parts}


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
    assert calls[0][0]["source"]["rate_run"] == str(root)
    assert calls[0][1]["apply"] == (mode == "apply")
    assert len(calls[0][1]["reuse_key"]) == 64


def test_failure_is_recorded_without_losing_rate_and_explicit_stage_raises(tmp_path, monkeypatch):
    config, _, root = saved_run(tmp_path)
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
    monkeypatch.setattr(pipe, "path_model", lambda *a: (model, {}, {}))
    monkeypatch.setattr(pipe, "correct_rate_run", lambda *a, **k: pytest.fail("factor_only applied a rate"))
    request = {"source": {"rate_run": str(tmp_path)}, "subspace": {"dimension": 1},
               "rpmd": {"windows": 12, "walkers": 2, "burn_in": 0, "sweeps": 2, "trajectories": 8},
               "output": {"reuse": True}}
    report = pipe.run_subspace_pipeline(request, apply=False, reuse_key="same")
    assert report["corrections"]["1"]["factor"] > 0
    assert report["corrected_rate"] is None
    assert report["physical_converged"] is False
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
    monkeypatch.setattr(pipe, "_run_pipeline", lambda *a, **k: identities.append(k["reuse_key"]) or {})
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
