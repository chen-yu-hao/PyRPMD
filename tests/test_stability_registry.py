"""Maturity labels follow selected methods without touching run readiness."""
from copy import deepcopy
import json

import pytest

from pyrpmd.stability import LEVELS, TASK_STABILITY, classify_stability, stability_registry


def test_registry_is_complete_json_and_returns_independent_snapshots():
    expected = {"rate", "drpmd", "fes", "network", "metadynamics", "opes", "pcet", "pimd",
                "diffusion", "site_rate", "surface_rate", "nonadiabatic", "nonadiabatic_rate",
                "spectroscopy", "plumed", "path_sampling"}
    assert set(TASK_STABILITY) == expected
    snapshot = stability_registry()
    json.dumps(snapshot, allow_nan=False)
    snapshot["tasks"]["rate"]["level"] = "changed"
    assert classify_stability({"task": "rate"})["level"] == "stable"
    assert all(item["level"] in LEVELS and item["scope"] for item in TASK_STABILITY.values())


@pytest.mark.parametrize("config,level", [
    ({"task": "spectroscopy", "spectroscopy": {"method": "trpmd"}}, "stable"),
    ({"task": "spectroscopy", "spectroscopy": {"method": "f-qcmd"}}, "research"),
    ({"task": "spectroscopy", "method": "f-qcmd"}, "research"),
    ({"task": "nonadiabatic", "method": "rp-iesh"}, "research"),
    ({"task": "nonadiabatic_rate"}, "research"),
    ({"task": "rate", "method": "subspace-correction"}, "research"),
    ({"task": "fes", "fes": {"method": "opes"}}, "beta"),
    ({"task": "opes"}, "beta"),
    ({"task": "fes", "fes": {"method": "metadynamics", "metadynamics": {"multiple_walkers": True}}}, "beta"),
    ({"task": "pimd", "path_integral": {"method": "fermionic"}}, "research"),
    ({"task": "drpmd", "nonadiabatic": {"method": "nrpmd"}}, "research"),
])
def test_method_overrides_and_input_are_preserved(config, level):
    config["ready"] = False
    before = deepcopy(config)
    result = classify_stability(config)
    assert result["level"] == level
    assert config == before
    assert "ready" not in result


def test_unregistered_task_is_not_silently_certified():
    with pytest.raises(ValueError, match="No stability classification"):
        classify_stability({"task": "new_unreviewed_method"})
