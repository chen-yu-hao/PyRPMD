"""Independent PMF stages compose with explicit reactions and regional windows."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.input import InputError, JobSpec, rate_pmf_options, rate_pmf_stage
from pyrpmd.runner import extend_run, prepare, run_resolved
from pyrpmd.workflows.rate_options import _parent_thermostat
from pyrpmd.workflows.rate_plan import rate_pmf_plan
from pyrpmd.workflows import rate_extend

from test_rate_input_v2 import research_spec
from test_rate_pmf_extend import configuration as extension_configuration
from test_rate_pmf_input import configuration as input_configuration


EXAMPLES = Path(__file__).resolve().parents[1] / "examples/rate"


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
def test_explicit_reaction_prepares_the_selected_independent_sampler(method):
    spec = JobSpec.from_file(EXAMPLES / f"pmf_{method}.inp")
    research = research_spec().data
    spec.data.update(schema_version=2, reaction=research["reaction"],
                     preparation=research["preparation"], native=False)
    before = deepcopy(spec.data)
    resolved, system = spec.resolve()
    stage = rate_pmf_stage(resolved, system)
    result = prepare(spec, numerical=True)
    assert spec.data == before
    assert result["ready"] and result["reaction_preparation"]["saddle_valid"]
    assert stage["schema_version"] == 2
    assert result["pmf_plan"]["sampler"] == method
    assert result["pmf_plan"]["sampling"]["timestep_fs"] == pytest.approx(.2)
    assert result["pmf_plan"]["recrossing"]["timestep_fs"] == pytest.approx(.1)
    if method == "umbrella":
        assert result["umbrella_preparation"]["windows"] == 24
        assert all(w["production_steps"] == 500 for w in result["umbrella_preparation"]["window_plan"])
    else:
        assert "umbrella_preparation" not in result
        assert "umbrella" not in stage


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
def test_regional_operation_controls_use_pmf_clock_and_preserve_parent_clock(nested, thermostat):
    spec = research_spec()
    spec.data["native"] = False
    spec.data["rate"]["pmf"].update({
        "method": "umbrella",
        "dynamics": {"timestep": "0.2 fs", "integrator": "normal-mode-bab"},
    })
    if thermostat == "pile-l":
        spec.data["umbrella"]["sampling"]["thermostat"] = {
            "type": "pile-l", "gamma_centroid": "0.02 fs^-1",
        }
    if nested:
        spec.data["rate"]["pmf"]["umbrella"] = spec.data.pop("umbrella")
    before = deepcopy(spec.data)
    resolved, system = spec.resolve()
    stage = rate_pmf_stage(resolved, system)
    result = prepare(spec, numerical=False)
    assert spec.data == before
    assert "umbrella" not in resolved
    assert "sampling" not in stage["umbrella"]
    assert "analysis" not in stage["umbrella"]
    assert "start" not in stage["umbrella"] and "kappa" not in stage["umbrella"]
    assert resolved["rate"]["pmf"]["estimator"] == "ui"
    assert resolved["rate"]["pmf_points"] == 401
    assert stage["analysis"]["bootstrap_replicates"] == 10
    assert "bootstrap" not in stage["analysis"]
    assert rate_pmf_options(resolved)["bootstrap"] == 10
    assert _parent_thermostat(resolved) == (20, "fixed")
    assert rate_pmf_plan(resolved)["recrossing"]["parent_thermostat"] == "andersen"
    assert stage["sampling"]["thermostat"] == thermostat
    if thermostat == "andersen":
        assert stage["sampling"]["andersen_interval"] == 10
    plan = result["umbrella_preparation"]["window_plan"]
    assert len(plan) == 42
    assert all((w["equilibration_steps"], w["production_steps"], w["sample_stride_steps"])
               == (25, 100, 1) for w in plan)
    np.testing.assert_allclose([w["stiffness_au"][0][0] for w in plan], [.16] * 16 + [.32] * 26)


@pytest.mark.parametrize("field,value", [
    ("sampling", {}), ("analysis", {}), ("estimator", "ui"),
])
def test_duplicate_nested_pmf_stage_declarations_are_rejected(field, value):
    spec = research_spec()
    umbrella = spec.data.pop("umbrella")
    if field == "analysis":
        umbrella["analysis"] = {"bootstrap_replicates": 10}
    elif field == "estimator":
        umbrella["analysis"] = {"estimator": "ui"}
    spec.data["rate"]["pmf"] = {"method": "umbrella", field: value,
                                 "umbrella": umbrella}
    with pytest.raises(InputError, match="not both"):
        spec.resolve()


def test_nested_windows_with_force_matrix_keep_their_only_window_selector():
    spec = research_spec()
    spec.data["rate"]["pmf"] = {
        "method": "umbrella", "dynamics": {"timestep": "0.2 fs"},
        "umbrella": spec.data.pop("umbrella"),
    }
    umbrella = spec.data["rate"]["pmf"]["umbrella"]
    umbrella.pop("regions")
    umbrella.pop("force_constant")
    umbrella.update(force_matrix=[[.00016]], windows=[{"center": x} for x in (0, .5, 1.05)])
    result = prepare(spec)
    assert result["umbrella_preparation"]["windows"] == 3
    assert [w["stiffness_au"] for w in result["umbrella_preparation"]["window_plan"]] == [[[.16]]] * 3


def test_operation_named_pmf_controls_keep_independent_parent_defaults_in_schema_one():
    raw = input_configuration()
    raw["sampling"].update(andersen_interval=99, parents=13)
    raw["rate"] = {"pmf": {"method": "umbrella", "umbrella": {
        "centers": [0, .5, 1.05], "kappa": .1,
        "sampling": {"production": "4 fs"},
    }}}
    resolved, _ = JobSpec.from_mapping(raw).resolve()
    assert _parent_thermostat(resolved) == (10, "fixed")
    assert resolved["recrossing"]["parent_samples"] == 4


def test_window_local_duration_and_refresh_controls_use_the_pmf_timestep():
    spec = research_spec()
    umbrella = spec.data.pop("umbrella")
    umbrella["regions"][1]["sampling"] = {
        "equilibration": "2 fs", "production": "4 fs", "sample_every": "0.4 fs",
        "thermostat": {"type": "andersen", "refresh_every": "0.6 fs"},
    }
    spec.data["rate"]["pmf"] = {
        "method": "umbrella", "dynamics": {"timestep": "0.2 fs"}, "umbrella": umbrella,
    }
    config, system = spec.resolve()
    from pyrpmd.workflows.rate_pmf import configure_rate_umbrella
    from pyrpmd.workflows.prepare import prepare_rate
    from pyrpmd.workflows.umbrella_plan import resolve_plan, window_sampling

    reaction = prepare_rate(config, system, numerical=False)["reaction"]
    prepared = configure_rate_umbrella(config, system, reaction)
    stage = rate_pmf_stage(prepared, system)
    _, _, windows = resolve_plan(stage, system)
    local = window_sampling(stage, windows[-1])
    assert (local["equilibration_steps"], local["production_steps"],
            local["sample_interval"], local["andersen_interval"]) == (10, 20, 2, 3)
    assert _parent_thermostat(config) == (20, "fixed")


@pytest.mark.parametrize("scope", ["request", "saved"])
def test_nested_window_extension_rejected_before_lock_creation(scope, tmp_path, monkeypatch):
    config = {"task": "rate", "rate": {"pmf": {"method": "umbrella"}}}
    umbrella = {"regions": [{"sampling": {"production": {"steps": 20}}}]}
    if scope == "request":
        config["rate"]["pmf"]["umbrella"] = umbrella
    else:
        config["rate"]["pmf_stage"] = {"config": {"umbrella": umbrella}}
    (tmp_path / "input").mkdir()
    source = tmp_path / "input/resolved.json"
    source.write_text(json.dumps(config))
    before = source.read_bytes()
    monkeypatch.setattr(rate_extend, "_extension_lock", lambda *a: pytest.fail("unsupported stage acquired a lock"))
    with pytest.raises(InputError, match="per-window or per-region"):
        rate_extend._lock(tmp_path)
    assert source.read_bytes() == before
    assert list(tmp_path.iterdir()) == [tmp_path / "input"]


def _hashes(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("local_budget", [False, True])
def test_independent_regional_umbrella_run_resume_and_extension(tmp_path, local_budget):
    raw = extension_configuration()
    raw.update(schema_version=2, preparation={"ts_action": "none"})
    raw["reaction"]["definition"] = "explicit"
    raw["rate"]["pmf"]["umbrella"] = {
        "force_constant": "10 hartree",
        "windows": [{"id": f"slot-{i}", "center": float(x),
                     **({"force_constant": "15 hartree"} if i == 2 else {})}
                    for i, x in enumerate(np.linspace(0, 1, 5))],
    }
    if local_budget:
        raw["rate"]["pmf"]["umbrella"]["windows"][2]["sampling"] = {"production": {"steps": 18}}
    config, system = JobSpec.from_mapping(raw).resolve()
    output = tmp_path / "run"
    result = run_resolved(config, system, output=output)
    assert result["pmf"]["ready"]
    checkpoints = sorted((output / "umbrella/checkpoints").glob("window_*.npz"))
    assert len(checkpoints) == 5
    for path in checkpoints:
        with load_checkpoint(path) as saved:
            production = 18 if local_budget and str(saved["window_id"]) == "slot-2" else 31
            assert int(saved["elapsed"]) == 7 + production
            assert float(saved["time"]) == pytest.approx((7 + production) * .006)
    with load_checkpoint(output / "checkpoints/parent_0000.npz") as saved:
        assert float(saved["time"]) == pytest.approx(int(saved["step"]) * .003)
    before = {p.name: p.read_bytes() for p in checkpoints}
    resumed = run_resolved(config, system, output=output, resume=True)
    assert resumed["pmf"] == result["pmf"]
    assert {p.name: p.read_bytes() for p in checkpoints} == before
    if local_budget:
        all_before = _hashes(output)
        with pytest.raises(RuntimeError, match="per-window or per-region"):
            extend_run(output, production_steps=74)
        assert _hashes(output) == all_before
    else:
        extended = extend_run(output, production_steps=74)
        assert extended["effective_production_steps"] == 74
        for path in checkpoints:
            with load_checkpoint(path) as saved:
                assert int(saved["elapsed"]) == 7 + 74
                assert float(saved["time"]) == pytest.approx((7 + 74) * .006)
