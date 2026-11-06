"""NP6 native materials diffusion route and preflight checks."""
from __future__ import annotations

import json

import numpy as np
import pytest

from pyrpmd.runner import run, resume_run


def diffusion_config(**overrides):
    config = {
        "task": "diffusion",
        "temperature": "300 K",
        "beads": 2,
        "seed": 17,
        "system": {
            "symbols": ["H"],
            "geometry": [[0.0, 0.0, 0.0]],
            "geometry_unit": "bohr",
        },
        "potential": {"provider": "builtin", "name": "free"},
        "dynamics": {
            "integrator": "cayley-bcb",
            "loop_backend": "native",
            "timestep": "0.1 fs",
        },
        "sampling": {
            "engine": "native",
            "thermostat": "pile-l",
            "equilibration_steps": 2,
            "production_steps": 12,
            "sample_interval": 2,
            "replicas": 1,
        },
        "diffusion": {"atom_indices": [0], "axes": [0, 1, 2], "max_lag": 2},
        "output": {"checkpoint_steps": 5},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def test_native_diffusion_writes_legacy_fields_and_resumes_read_only(tmp_path):
    result = run(diffusion_config(), output=tmp_path / "run")
    assert result["task"] == "diffusion"
    assert result["propagation"][0]["python_steps"] == 0
    assert result["propagation"][0]["native_blocks"] > 0
    trajectory = np.load(result["files"]["trajectories"][0], allow_pickle=False)
    assert set(("positions", "velocities", "times", "sample_steps",
                "observables", "radii_gyration_squared")) <= set(trajectory.files)
    assert trajectory["positions"].shape[0] == result["samples_per_replica"][0]
    journal = json.loads((tmp_path / "run" / "samples" / "native_pimd_0000" / "journal.json").read_text())
    before = json.dumps(journal, sort_keys=True)
    resumed = resume_run(tmp_path / "run")
    assert resumed["propagation"] == result["propagation"]
    assert json.dumps(json.loads((tmp_path / "run" / "samples" / "native_pimd_0000" / "journal.json").read_text()),
                      sort_keys=True) == before


@pytest.mark.parametrize("change, message", [
    ({"sampling": {"thermostat": "andersen"}}, "PILE-L"),
    ({"barostat": {"mode": "bzp", "pressure": "1 bar", "tau": "10 fs", "gamma": "0.01 fs^-1"}},
     "fixed-cell"),
    ({"parallel": {"mode": "bead", "bead_workers": 2}}, "trajectory parallelism"),
])
def test_native_diffusion_rejects_unsupported_route_before_output(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        run(diffusion_config(**change), output=tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
