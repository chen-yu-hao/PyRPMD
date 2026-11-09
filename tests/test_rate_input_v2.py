"""The research template exercises the public schema, preflight and preview."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import prepare, run


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "rate"


def research_spec():
    return JobSpec.from_file(EXAMPLES / "research.inp")


def test_explicit_reaction_previews_windows_without_certifying_or_evaluating_ts(monkeypatch):
    spec = research_spec()
    original = deepcopy(spec.data)
    monkeypatch.setattr("pyrpmd.potentials.make_potential",
                        lambda *a, **k: pytest.fail("static preview evaluated the PES"))
    result = prepare(spec, numerical=False)
    assert spec.data == original
    assert not result["ready"]
    reaction = result["reaction_preparation"]
    assert reaction["reaction_definition"] == "explicit"
    assert reaction["saddle_valid"] is None and reaction["channel_consistent"] is None
    assert reaction["requires_numerical_preparation"]
    plan = result["umbrella_preparation"]
    assert plan["windows"] == 42
    windows = plan["window_plan"]
    np.testing.assert_allclose([w["center_au"][0] for w in windows[:16]], np.arange(16) * .05)
    np.testing.assert_allclose([w["center_au"][0] for w in windows[16:]], .8 + np.arange(26) * .01)
    np.testing.assert_allclose([w["stiffness_au"][0][0] for w in windows], [.16] * 16 + [.32] * 26)
    assert len({w["id"] for w in windows}) == 42
    assert all((w["chains"], w["equilibration_steps"], w["production_steps"],
                w["sample_stride_steps"], w["raw_samples_per_chain"]) == (1, 50, 200, 2, 100)
               for w in windows)
    recrossing = result["recrossing_preparation"]
    assert (recrossing["planned_configurations"], recrossing["planned_pairs"],
            recrossing["planned_trajectories"]) == (8, 16, 32)
    assert recrossing["parent_refresh_interval_steps"] == 20
    assert recrossing["child_steps"] == 100


def test_research_template_checks_saddle_and_declared_channel_without_refining():
    spec = research_spec()
    _, initial = spec.resolve()
    result = prepare(spec, numerical=True)
    assert result["ready"]
    reaction = result["reaction_preparation"]
    assert reaction["saddle_valid"] is True
    assert reaction["channel_consistent"] is True
    assert reaction["refinement_performed"] is False
    np.testing.assert_array_equal(result["system"]["geometry"], initial.geometry)
    assert result["umbrella_preparation"]["windows"] == 42


@pytest.mark.parametrize("change, message", [
    (lambda d: d["umbrella"]["regions"][0].update(include_endpoints="both"), "duplicate umbrella center"),
    (lambda d: d["umbrella"]["regions"][1].update(spacing=.03), "integer multiple"),
    (lambda d: d["umbrella"].update(centers=[0, .5, 1]), "exactly one"),
    (lambda d: d["umbrella"].update(stiffness=.16), "conflicting force"),
    (lambda d: d["umbrella"].update(cvs=["other"]), r"cvs must be \['xi'\]"),
    (lambda d: d.update(umbrella={"stop": 1.05, "force_constant": .00016}), "specify umbrella"),
    (lambda d: d["umbrella"]["regions"][1].update(force_constant=-.001), "positive"),
    (lambda d: d["umbrella"]["regions"][1].update(sampling={"production": "0.2 fs"}), "at least two"),
    (lambda d: d["reaction"].update(product_fragments=[[0, 1]]), "partition all atoms"),
])
def test_bad_research_plan_fails_before_pes_or_output(change, message, tmp_path, monkeypatch):
    spec = research_spec()
    change(spec.data)
    monkeypatch.setattr("pyrpmd.potentials.make_potential",
                        lambda *a, **k: pytest.fail("invalid plan evaluated the PES"))
    output = tmp_path / "invalid-run"
    with pytest.raises(InputError, match=message):
        run(spec, output=output)
    assert not output.exists()


def test_schema_two_requires_explicit_inference_choice_and_minimal_opts_in():
    spec = research_spec()
    del spec.data["reaction"]
    with pytest.raises(InputError, match="definition='explicit'"):
        spec.resolve()
    minimal = JobSpec.from_file(EXAMPLES / "minimal.inp")
    result = prepare(minimal, numerical=True)
    assert result["ready"]
    assert result["reaction_preparation"]["reaction_definition"] == "infer"
    assert result["reaction_preparation"]["channel_consistent"] is True


def test_schema_two_does_not_reinterpret_legacy_physical_stiffness():
    spec = research_spec()
    spec.data["umbrella"] = {"centers": [0, .5, 1.05], "stiffness": .16}
    result = prepare(spec)
    assert [w["stiffness_au"] for w in result["umbrella_preparation"]["window_plan"]] == [[[.16]]] * 3


@pytest.mark.parametrize("operation", ["prepare", "run"])
def test_inference_checks_the_sampling_plan_after_learning_the_channel(operation, tmp_path):
    spec = research_spec()
    spec.data["reaction"] = {"definition": "infer", "rinf": "5 angstrom"}
    spec.data["umbrella"]["sampling"]["production"] = "0.2 fs"
    output = tmp_path / "too-short-inferred-run"
    with pytest.raises((ValueError, RuntimeError), match="at least two"):
        if operation == "prepare":
            prepare(spec, numerical=True)
        else:
            run(spec, output=output)
    # Run initialization may retain diagnostic input/preparation files after
    # numerical inference, but no biased or recrossing trajectory may start.
    assert not list(output.rglob("*.npz"))
