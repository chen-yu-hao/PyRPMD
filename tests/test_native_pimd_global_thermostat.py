"""Native PIMD flags single global baths, which do not thermalize ring modes."""

import pytest

from pyrpmd.input import JobSpec
from pyrpmd.workflows import native_pimd

from test_native_pimd_preflight import job

_MARKER = "one global bath"


def resolve(thermostat, beads):
    raw = job()
    raw["beads"] = beads
    raw["sampling"]["thermostat"] = thermostat
    if thermostat in ("nhc", "svr", "pile-g"):
        raw["sampling"]["thermostat_params"] = {"tau": "20 fs"}
    return JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("thermostat", ["nhc", "svr"])
def test_global_bath_with_beads_is_reported_at_resolve_and_in_summary(tmp_path, thermostat):
    pytest.importorskip("pyrpmd._thermal")
    config, system = resolve(thermostat, 4)
    assert sum(_MARKER in note and thermostat in note for note in config["_warnings"]) == 1
    summary = native_pimd.run_native_pimd(config, system, tmp_path)
    assert sum(_MARKER in note for note in summary["warnings"]) == 1


@pytest.mark.parametrize("thermostat,beads", [("nhc", 1), ("svr", 1), ("pile-l", 4), ("pile-g", 4)])
def test_local_baths_and_classical_global_baths_are_not_flagged(thermostat, beads):
    config, _ = resolve(thermostat, beads)
    assert not any(_MARKER in note for note in config["_warnings"])
    assert native_pimd.global_thermostat_warning(thermostat, beads) is None
