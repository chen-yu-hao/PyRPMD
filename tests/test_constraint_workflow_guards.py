"""Direct workflow APIs must not silently discard System bond constraints."""

import importlib
from unittest.mock import Mock

import numpy as np
import pytest

from pyrpmd.system import System
from pyrpmd.workflows import validate_bond_constraint_workflow


def _system():
    return System(("H", "H"), np.array([1837., 1837.]),
                  np.array([[0., 0, 0], [1., 0, 0]]),
                  bond_constraints=[(0, 1, 1.)])


UNSUPPORTED_RUNNERS = [
    ("fes", "run_fes"), ("rate", "run_rate"), ("drpmd", "run_drpmd"),
    ("surface", "run_surface_rate"), ("site_rate", "run_site_rate"),
    ("network", "run_network"), ("metadynamics", "run_metadynamics"),
    ("opes", "run_opes"), ("adaptive_umbrella", "run_adaptive_fes"),
    ("replica_exchange", "run_replica_exchange_fes"),
    ("native_pimd", "run_native_pimd"), ("pcet", "run_pcet"),
]


@pytest.mark.parametrize("module_name,function_name", UNSUPPORTED_RUNNERS)
def test_direct_unsupported_workflow_rejects_before_dispatch_or_output(tmp_path, monkeypatch, module_name, function_name):
    module = importlib.import_module("pyrpmd.workflows."+module_name)
    function = getattr(module, function_name)
    system = _system()
    before = system.to_dict()
    target = tmp_path / "untouched"
    dispatched = Mock(side_effect=AssertionError("work dispatched before constraint rejection"))
    for name in ("run_tasks", "root_call"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, dispatched)
    # Even a misleading allowed task name cannot change the implementation's
    # actual capabilities, or route through adaptive/replica-exchange branches.
    config = {"task": "pimd", "adaptive_umbrella": {"enabled": True},
              "replica_exchange": {"enabled": True}}
    with pytest.raises(ValueError, match="bond constraints"):
        function(config, system, target)
    dispatched.assert_not_called()
    assert not target.exists()
    assert system.to_dict() == before


@pytest.mark.parametrize("task", ["fes", "rate", "drpmd", "surface_rate", "site_rate",
                                  "network", "metadynamics", "opes", "pcet"])
def test_run_resolved_rejects_before_creating_run_manifest(tmp_path, monkeypatch, task):
    from pyrpmd import runner

    initializer = Mock(side_effect=AssertionError("run files created before constraint rejection"))
    planner = Mock(side_effect=AssertionError("backend planned before constraint rejection"))
    monkeypatch.setattr(runner, "_initialize_output", initializer)
    monkeypatch.setattr(runner, "propagation_backend_plan", planner)
    config = {"task": task, "output": {"directory": str(tmp_path / "untouched")}}
    with pytest.raises(ValueError, match="bond constraints"):
        runner.run_resolved(config, _system())
    initializer.assert_not_called()
    planner.assert_not_called()
    assert not (tmp_path / "untouched").exists()


def test_resolved_materials_rejects_unsupported_engine_before_run_files(tmp_path, monkeypatch):
    from pyrpmd import runner

    initializer = Mock(side_effect=AssertionError("run files created before engine rejection"))
    planner = Mock(side_effect=AssertionError("backend planned before engine rejection"))
    monkeypatch.setattr(runner, "_initialize_output", initializer)
    monkeypatch.setattr(runner, "propagation_backend_plan", planner)
    config = {"task": "pimd", "sampling": {"engine": "native"},
              "dynamics": {"integrator": "velocity-verlet"},
              "output": {"directory": str(tmp_path / "untouched")}}
    with pytest.raises(ValueError, match="bond constraints"):
        runner.run_resolved(config, _system())
    initializer.assert_not_called()
    planner.assert_not_called()
    assert not (tmp_path / "untouched").exists()


@pytest.mark.parametrize("module_name,function_name", [("prepare", "prepare_rate"), ("network", "prepare_network")])
def test_direct_rate_preparation_rejects_before_numerical_pes_work(monkeypatch, module_name, function_name):
    from pyrpmd import potentials

    constructor = Mock(side_effect=AssertionError("PES constructed before constraint rejection"))
    monkeypatch.setattr(potentials, "make_potential", constructor)
    function = getattr(importlib.import_module("pyrpmd.workflows."+module_name), function_name)
    with pytest.raises(ValueError, match="bond constraints"):
        function({}, _system(), numerical=True)
    constructor.assert_not_called()


def test_constraint_guard_accepts_only_materials_and_catches_serialized_systems():
    system = _system()
    for task in ("pimd", "diffusion"):
        validate_bond_constraint_workflow(system, task)
        validate_bond_constraint_workflow(system.to_dict(), task)
    with pytest.raises(ValueError, match="bond constraints"):
        validate_bond_constraint_workflow(system.to_dict(), "fes")
    plain = System(system.symbols, system.masses, system.geometry)
    for module, _ in UNSUPPORTED_RUNNERS:
        validate_bond_constraint_workflow(plain, module)
