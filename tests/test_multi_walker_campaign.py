"""Longer toy-model evidence kept out of the default unit-test budget."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.campaign
def test_two_dimensional_double_well_estimators_are_statistically_compatible():
    script = (Path(__file__).parents[1] / "docs" / "validation" /
              "multi-walkers-2026-11-03" / "double_well_campaign.py")
    spec = importlib.util.spec_from_file_location("t38_double_well_campaign", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.run_campaign(replicates=4)
    assert result["metadynamics"]["within_statistical_error"]
    assert result["opes_explore"]["within_statistical_error"]
    assert result["opes_explore"]["same_center_formula_max_abs_difference"] < 1e-11
