"""Opt-in native interface guards and paper-input checks; no implicit fetch."""

import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks.hch4.pes import CBE2008Potential
from benchmarks.hch4.reproduction_input import PROTOCOL, resolve_pilot
from pyrpmd.propagation import NativeUnavailableError
from pyrpmd.workflows.rate import rate_coordinate


@pytest.fixture
def library():
    value = os.environ.get("PYRPMD_CBE2008_LIBRARY")
    if not value:
        pytest.skip("Set PYRPMD_CBE2008_LIBRARY; this test never downloads a PES")
    return Path(value)


def test_native_binding_shares_lock_and_retains_owner(library):
    a, b = CBE2008Potential(library), CBE2008Potential(library)
    binding = a.native_binding()
    assert binding.kind == "fortran-batch-v1" and binding.natoms == 6
    assert binding.owner is a and binding.lock is a._lock is b._lock
    with binding.acquire() as (kind, address, context, natoms):
        assert kind == "fortran-batch-v1" and address and context == 0 and natoms == 6


def test_native_binding_does_not_bypass_overridden_evaluate(library, monkeypatch):
    class Derived(CBE2008Potential):
        pass
    with pytest.raises(NativeUnavailableError, match="overridden"):
        Derived(library).native_binding()
    potential = CBE2008Potential(library)
    potential.evaluate = lambda q: None
    with pytest.raises(NativeUnavailableError, match="overridden"):
        potential.native_binding()
    del potential.evaluate
    monkeypatch.setattr(CBE2008Potential, "evaluate", lambda self, q: None)
    with pytest.raises(NativeUnavailableError, match="overridden"):
        potential.native_binding()


@pytest.mark.parametrize("beads", [1, 64, 128, 256])
def test_paper_mapping_is_explicit_and_uses_all_four_paths(library, beads):
    raw, config, system = resolve_pilot(library, beads=beads)
    cv = rate_coordinate(config, system)
    assert len(cv.paths) == 4 and config["reaction"]["symmetry_factor"] == 1
    assert config["reaction"]["electronic_weight"] == 1
    assert cv.rinf == 30 and len(system.masses) == 6
    assert raw["system"]["masses_amu"] == PROTOCOL["planned_physics"]["masses_amu"]
    assert config["rate"]["dividing_surface"] == (0.999 if beads == 1 else 1.007)
    assert config["sampling"]["andersen_schedule"] == "geometric"
    assert config["umbrella"]["count"] == 111
    np.testing.assert_allclose(cv.centroid_value_gradient(system.geometry)[0], 1, atol=1e-15)


def test_reference_numbers_have_separate_units_and_rounding():
    reference = PROTOCOL["reference"]
    assert reference["rate_units"] == "cm^3 molecule^-1 s^-1"
    for row in reference["rows"]:
        assert abs(row["k_qtst"] * row["kappa"] / row["k_full"] - 1) < 0.005
    assert PROTOCOL["official_distribution"]["input_grid_window_count"] == 110
    assert PROTOCOL["reference_protocol"]["umbrella_windows"] == 111
