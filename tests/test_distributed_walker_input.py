"""Strict input registration for distributed multiple walkers."""
from __future__ import annotations

import numpy as np
import pytest

from pyrpmd.input_schema import _SECTIONS
from pyrpmd.system import System
from pyrpmd.workflows.metadynamics import resolve_metadynamics
from pyrpmd.workflows.opes import resolve_opes


def system():
    return System(("H",), np.array([1.0]), np.array([[0.1, 0.0, 0.0]]))


def metadynamics_config():
    return {
        "task": "metadynamics", "beta": 2.0, "beads": 1, "dt_au": 0.02,
        "potential": {"provider": "builtin", "name": "harmonic",
                      "params": {"k": 0.2}},
        "sampling": {"production_steps": 4, "sample_interval": 1,
                     "replicas": 2},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0,
                 "axis": "x"}],
        "metadynamics": {"shared_walkers": True,
                         "walkers_mode": "mpi-sync", "sigma": [0.3],
                         "height": 0.04, "pace": 2, "bias_factor": 5.0},
        "analysis": {"bins": [5], "bootstrap": 0},
    }


def opes_config():
    return {
        "task": "opes", "beta": 2.0, "beads": 1, "dt_au": 0.02,
        "potential": {"provider": "builtin", "name": "harmonic",
                      "params": {"k": 0.2}},
        "sampling": {"production_steps": 4, "sample_interval": 1,
                     "replicas": 2},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0,
                 "axis": "x"}],
        "opes": {"walkers_mode": "async-files", "sigma": [0.3],
                 "barrier": 1.2, "pace": 2},
        "analysis": {"method": "kde", "measure": "cv_density",
                     "bins": [5], "bootstrap": 0},
    }


def test_walker_mode_fields_are_append_only_schema_entries():
    assert "walkers_mode" in _SECTIONS["metadynamics"]
    assert "walkers_mode" in _SECTIONS["opes"]


def test_distributed_metadynamics_requires_explicit_shared_history():
    config = metadynamics_config()
    config["metadynamics"]["shared_walkers"] = False
    with pytest.raises(ValueError, match="shared_walkers=true"):
        resolve_metadynamics(config, system())


def test_unknown_mode_and_unsupported_distributed_opes_are_rejected():
    config = opes_config()
    config["opes"]["walkers_mode"] = "threads"
    with pytest.raises(ValueError, match="serial, mpi-sync or async-files"):
        resolve_opes(config, system())
    config = opes_config()
    config["opes"]["adaptive_sigma"] = True
    with pytest.raises(ValueError, match="fixed sigma"):
        resolve_opes(config, system())


def test_nonserial_mode_requires_two_walkers():
    config = opes_config()
    config["sampling"]["replicas"] = 1
    with pytest.raises(ValueError, match="replicas >= 2"):
        resolve_opes(config, system())
