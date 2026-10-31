"""Pressure-engine selection after integrating legacy and native PIMD."""
import copy

import numpy as np
import pytest

from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import resume_run, run


def _config():
    return {
        "task": "pimd", "temperature": "300 K", "beads": 4, "seed": 91,
        "system": {"symbols": ["H", "H"], "geometry": [[0., 0., 0.], [.1, .2, .3]],
                   "geometry_unit": "bohr", "cell": (np.eye(3)*4).tolist(), "cell_unit": "bohr"},
        "potential": {"provider": "free"},
        "dynamics": {"timestep": "0.1 autime", "loop_backend": "python"},
        "sampling": {"thermostat": "pile-l", "equilibration_steps": 2,
                     "production_steps": 6, "sample_interval": 2, "replicas": 1},
        "barostat": {"method": "bzp", "pressure": "1 bar", "time_constant": "10 fs", "friction": "0.01 fs^-1"},
        "output": {"checkpoint_steps": 3},
    }


@pytest.mark.parametrize("engine", [None, "legacy"])
def test_existing_bzp_input_retains_legacy_pressure_workflow(tmp_path, engine):
    config = _config()
    if engine is not None:
        config["sampling"]["engine"] = engine
    summary = run(config, output=tmp_path/"legacy")
    assert summary["ensemble"] == "npt"
    assert summary["barostat"]["method"] == "bzp"
    assert summary["barostat"]["target_pressure_bar"] == pytest.approx(1.)
    assert resume_run(tmp_path/"legacy")["estimators"] == summary["estimators"]


@pytest.mark.parametrize("barostat", [
    {"method": "bzp", "pressure": "1 bar", "time_constant": "10 fs"},
    {"pressure": "1 bar"},
    {"mode": "bzp", "pressure": "1 bar", "friction": "0.01 fs^-1"},
])
def test_native_engine_rejects_legacy_or_ambiguous_pressure_controls(barostat):
    config = _config()
    config["sampling"]["engine"] = "native"
    config["dynamics"]["loop_backend"] = "native"
    config["barostat"] = copy.deepcopy(barostat)
    with pytest.raises(InputError, match="native PIMD barostat requires"):
        JobSpec.from_mapping(config).resolve()


def test_legacy_engine_rejects_native_pressure_fields():
    config = _config()
    config["barostat"] = {"mode": "bzp", "pressure": "1 bar", "tau": "10 fs"}
    with pytest.raises(InputError, match="barostat"):
        JobSpec.from_mapping(config).resolve()
