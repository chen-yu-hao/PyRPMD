"""Equal-cost adaptive umbrella accounting and BKMP2 campaign input guards."""
from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from pyrpmd.workflows.adaptive_umbrella import (
    adaptive_options,
    equal_cost_production_budget,
)


HERE = Path(__file__).resolve().parents[1]
INPUTS = HERE / "benchmarks" / "h3" / "equal_cost_inputs"


def _frozen_plan(target=2_496_000):
    # Six pilot windows at (3000+1000) steps and ten at (3000+2000),
    # with two replicas: 48,000+100,000 = 148,000 pilot steps.
    return {
        "status": "frozen",
        "centers": [[float(i)] for i in range(10)],
        "options": {"total_integration_steps": target},
        "rounds": [
            {"centers": [[float(i)] for i in range(6)],
             "pilot": {"equilibration_steps": 3000, "production_steps": 1000}},
            {"centers": [[float(i)] for i in range(10)],
             "pilot": {"equilibration_steps": 3000, "production_steps": 2000}},
        ],
    }


def _config():
    return {"sampling": {"replicas": 2, "equilibration_steps": 3000, "interval_steps": 5}}


def test_exact_matched_budget_counts_all_discarded_pilots():
    result = equal_cost_production_budget(_config(), _frozen_plan())
    assert result["pilot_integration_steps"] == 148_000
    assert result["frozen_streams"] == 20
    assert result["production_steps_per_stream"] == 114_400
    assert result["used_integration_steps"] == 2_496_000
    assert result["unspent_integration_steps"] == 0


def test_partial_stride_is_accounted_without_exceeding_budget():
    result = equal_cost_production_budget(_config(), _frozen_plan(2_496_003))
    assert result["used_integration_steps"] == 2_496_000
    assert result["unspent_integration_steps"] == 3


def test_budget_shortfall_blocks_production():
    with pytest.raises(ValueError, match="exhausted"):
        equal_cost_production_budget(_config(), _frozen_plan(200_000))


def test_ordinary_adaptive_run_does_not_acquire_new_budget():
    plan = _frozen_plan()
    plan["options"] = {}
    assert equal_cost_production_budget(_config(), plan) is None


def test_invalid_budget_is_rejected():
    with pytest.raises((ValueError, TypeError)):
        adaptive_options({"adaptive_umbrella": {"enabled": True, "total_integration_steps": 0}})


def test_paired_campaign_inputs_use_same_physics_and_auto_surface():
    with (INPUTS / "fixed96.inp").open("rb") as stream:
        fixed = tomllib.load(stream)
    with (INPUTS / "adaptive.inp").open("rb") as stream:
        adaptive = tomllib.load(stream)
    for case in (fixed, adaptive):
        assert case["potential"]["factory"] == "../pes.py:build"
        assert case["reaction"]["equivalent_paths"] == [[0, 2, 1]]
        assert case["reaction"]["symmetry_factor"] == 1.0
        assert "dividing_surface" not in case["rate"]
        assert "dividing_surface" not in case["recrossing"]
        assert case["rate"]["pmf_method"] == "mbar"
        assert case["rate"]["compare_ui"] is True
        assert case["rate"]["pmf_search_range"] == [0.75, 1.08]
        assert case["sampling"]["replicas"] == 2
    for key in ("temperature", "beads", "dynamics", "sampling", "reaction", "analysis", "recrossing"):
        assert fixed[key] == adaptive[key]
    assert fixed["umbrella"]["count"] == 96
    assert adaptive["umbrella"]["count"] == 6
    assert adaptive["adaptive_umbrella"]["total_integration_steps"] == 2_496_000
