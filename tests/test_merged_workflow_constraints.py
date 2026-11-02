"""Specialized workflows must not silently discard newly shared system constraints."""
import importlib
import numpy as np
import pytest
from pyrpmd.system import System


@pytest.mark.parametrize("module, entry", [
    ("nonadiabatic", "run_nonadiabatic"),
    ("nonadiabatic", "analyze_nonadiabatic"),
    ("nonadiabatic_extensions", "run_extended_nonadiabatic"),
    ("nonadiabatic_extensions", "analyze_extended_nonadiabatic"),
    ("plumed", "run_plumed"),
    ("spectroscopy", "run_spectroscopy"),
    ("spectroscopy", "reanalyze_spectroscopy"),
])
def test_constraints_rejected_before_workflow_dispatch_and_output(module, entry, tmp_path):
    system = System(symbols=("H", "H"), masses=np.array([1836., 1836.]),
                    geometry=np.array([[0., 0., 0.], [1., 0., 0.]]),
                    bond_constraints=((0, 1, 1.),))
    output = tmp_path / "must-not-exist"
    function = getattr(importlib.import_module("pyrpmd.workflows." + module), entry)
    # The unsupported system must be rejected before task-specific config,
    # provider construction, checkpoint access, or directory creation.
    with pytest.raises(ValueError, match="bond constraints are unsupported"):
        function({}, system, output)
    assert not output.exists()
