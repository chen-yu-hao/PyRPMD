"""Resolved native NHC inputs and outputs identify the map actually executed."""

import pytest

from pyrpmd.input import JobSpec
from pyrpmd.potentials import make_potential
from pyrpmd.workflows import native_pimd

from test_native_pimd_preflight import job


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("npt", [False, True])
def test_native_nhc_resolved_scheme_matches_executor_and_summary(tmp_path, integrator, npt):
    pytest.importorskip("pyrpmd._thermal")
    raw = job(npt=npt)
    raw["dynamics"]["integrator"] = integrator
    raw["sampling"].update(thermostat="nhc", thermostat_params={"tdamp": "20 fs"})
    config, system = JobSpec.from_mapping(raw).resolve()
    expected = "OBABO" if npt or integrator == "normal-mode-bab" else "OBCBO"
    assert config["sampling"]["integrator"] == expected.lower()
    provider = make_potential(config["potential"], system)
    engine = native_pimd._engine(config, system, 0, provider)
    assert engine.splitting == expected
    summary = native_pimd.run_native_pimd(config, system, tmp_path)
    assert summary["sampling_splitting"] == expected
    assert summary["sampling_thermostat"] == "nhc"


def test_existing_native_pile_input_keeps_its_particle_integrator_name():
    raw = job(npt=True)
    config, _ = JobSpec.from_mapping(raw).resolve()
    assert config["sampling"]["integrator"] == "bcocb"
    assert native_pimd.sampling_splitting("cayley-bcb", "pile-l", "bzp") == "obabo"
