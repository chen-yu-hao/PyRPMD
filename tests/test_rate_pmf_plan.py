"""The CLI exposes the independently resolved PMF clock before propagation."""

import json
from pathlib import Path

import pytest

from pyrpmd.cli import main
from pyrpmd.input import JobSpec
from pyrpmd.runner import prepare
from pyrpmd.workflows.rate_plan import rate_pmf_plan


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "rate"


@pytest.mark.parametrize("method", ["umbrella", "metadynamics", "opes"])
def test_cli_validation_shows_effective_pmf_and_recrossing_stages(method, capsys):
    status = main(["validate", str(EXAMPLES / f"pmf_{method}.inp")])
    assert status in (None, 0)
    result = json.loads(capsys.readouterr().out)
    plan = result["pmf_plan"]
    assert plan["sampler"] == method
    assert plan["sampling"]["timestep_fs"] == pytest.approx(0.2)
    assert plan["recrossing"]["timestep_fs"] == pytest.approx(0.1)
    assert plan["sampling"]["thermostat"] == "pile-l"
    assert plan["recrossing"]["parent_thermostat"] == "andersen"
    if method != "umbrella":
        assert plan["sampling"]["retained_samples_per_replica"] == 140
        assert [phase["steps"] for phase in plan["sampling"]["phases"]] == [50, 200, 100, 700]
        assert [phase["duration_fs"] for phase in plan["sampling"]["phases"]] == pytest.approx([10, 40, 20, 140])
        assert "unavailable" in plan["barrier_uncertainty"]


def test_numerical_preparation_preserves_the_same_visible_plan():
    source = EXAMPLES / "pmf_opes.inp"
    config, _ = JobSpec.from_file(source).resolve()
    result = prepare(source, numerical=True)
    assert result["ready"]
    assert result["pmf_plan"] == rate_pmf_plan(config)


@pytest.mark.parametrize("adaptive, message", [
    ({"max_windows": 2}, "initial umbrella count exceeds adaptive_umbrella.max_windows"),
    ({"coverage_bins": [10, 10], "coverage_ranges": [[-0.05, 1.10], [-0.05, 1.10]]},
     "coverage_bins dimension must match selected CVs"),
])
def test_numerical_preparation_rejects_invalid_adaptive_window_geometry(adaptive, message):
    source = JobSpec.from_file(EXAMPLES / "pmf_umbrella.inp")
    source.data["rate"]["pmf"]["adaptive_umbrella"] = {"enabled": True, **adaptive}
    with pytest.raises(ValueError, match=message):
        prepare(source, numerical=True)


def test_numerical_preparation_skips_disabled_adaptive_window_geometry():
    source = JobSpec.from_file(EXAMPLES / "pmf_umbrella.inp")
    source.data["rate"]["pmf"]["adaptive_umbrella"] = {
        "enabled": False, "max_windows": 2,
        "coverage_bins": [10, 10], "coverage_ranges": [[-0.05, 1.10], [-0.05, 1.10]],
    }
    assert prepare(source, numerical=True)["ready"]
