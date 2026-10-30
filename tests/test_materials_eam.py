"""Opt-in real-PES integration; ordinary tests never download model data."""

import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def pilot():
    path = Path(__file__).resolve().parents[1] / "benchmarks/materials/ni_h_eam.py"
    spec = importlib.util.spec_from_file_location("materials_eam_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_model_requires_explicit_download_and_bad_bytes_are_rejected(tmp_path, pilot):
    path = tmp_path / "model.eam.alloy"
    with pytest.raises(FileNotFoundError, match="explicitly"):
        pilot.obtain_model(path)
    path.write_bytes(b"incorrect model")
    with pytest.raises(ValueError, match="SHA256"):
        pilot.obtain_model(path)
    assert path.read_bytes() == b"incorrect model"


def test_real_nickel_hydrogen_eam_adapter_and_periodic_workflow(tmp_path, pilot):
    model = os.environ.get("PYRPMD_NIALH_POTENTIAL")
    if not model:
        pytest.skip("Set PYRPMD_NIALH_POTENTIAL to opt into the source-pinned real EAM check")
    pytest.importorskip("ase")
    from pyrpmd import run

    model = pilot.obtain_model(model)
    assert pilot.verify_adapter(model)["passed"]
    result = run(pilot.job(model, isotope="D", beads=2, quick=True), output=tmp_path / "run")
    assert result["status"] == "complete"
    assert result["pbc"] == [True, True, False]
    assert result["phase"] == "unthermostatted_rpmd_production"
    assert result["physical_converged"] is False
    assert result["diffusion"]["diffusion_m2_s"] is None
