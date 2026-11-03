"""Strict input registration for distributed multiple walkers."""
from __future__ import annotations

import pytest

from pyrpmd.input_schema import InputError, _SECTIONS
from pyrpmd.input_tasks import _validate_walker_mode


def test_walker_mode_fields_are_registered_without_injecting_defaults():
    assert "walkers_mode" in _SECTIONS["metadynamics"]
    assert "walkers_mode" in _SECTIONS["opes"]


def test_distributed_metadynamics_requires_explicit_shared_history():
    data = {"task": "fes", "sampling": {"replicas": 2},
            "metadynamics": {"walkers_mode": "mpi-sync"}}
    with pytest.raises(InputError, match="shared_walkers=true"):
        _validate_walker_mode(data, "metadynamics", "metadynamics")
    data["metadynamics"]["shared_walkers"] = True
    _validate_walker_mode(data, "metadynamics", "metadynamics")


def test_unknown_mode_and_unsupported_distributed_opes_are_rejected():
    data = {"task": "fes", "sampling": {"replicas": 2},
            "opes": {"walkers_mode": "threads"}}
    with pytest.raises(InputError, match="serial, mpi-sync or async-files"):
        _validate_walker_mode(data, "opes", "opes")
    data["opes"] = {"walkers_mode": "async-files", "adaptive_sigma": True}
    with pytest.raises(InputError, match="fixed sigma"):
        _validate_walker_mode(data, "opes", "opes")


def test_nonserial_mode_requires_two_walkers_and_selected_method():
    data = {"task": "fes", "sampling": {"replicas": 1},
            "opes": {"walkers_mode": "mpi-sync"}}
    with pytest.raises(InputError, match="replicas >= 2"):
        _validate_walker_mode(data, "opes", "opes")
    data["sampling"]["replicas"] = 2
    with pytest.raises(InputError, match="requires the opes FES workflow"):
        _validate_walker_mode(data, "opes", "metadynamics")
