"""Contracts shared by the saved-FES provenance and sampling-report adapters."""

import json

import numpy as np
import pytest

from pyrpmd.analysis.reporting import collect_run
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import BOHR_ANGSTROM
from test_report_review import fes_measure_run


@pytest.mark.parametrize("unit,measure,accepted", [
    ("bohr", "cv_density", True),
    ("angstrom", "cv_density", False),
    ("bohr", "rate_flux", False),
])
def test_hdf5_sampling_uses_authoritative_npz_units_and_measure(tmp_path, unit, measure, accepted):
    h5py = pytest.importorskip("h5py")
    # The summary intentionally lacks units/measure; the checksummed FES has both.
    fes_measure_run(tmp_path)
    with h5py.File(tmp_path / "fes.h5", "w") as saved:
        saved.attrs.update(energy_unit="hartree", cv_units=json.dumps([unit]), measure=measure)
        stream = saved.create_group("streams/0")
        stream.attrs.update(window=0, replica=0)
        stream["samples"] = [[10.], [20.]]
        stream["center"] = [15.]
        stream["stiffness"] = [[1.]]
    run = collect_run(tmp_path)
    assert any(plot["id"].endswith("-fes") for plot in run["_plots"])
    traces = [plot for plot in run["_plots"] if "-trace-" in plot["id"]]
    if accepted:
        assert len(traces) == 1
        np.testing.assert_allclose(traces[0]["series"][0]["y"], np.array([10., 20.]) * BOHR_ANGSTROM)
    else:
        assert not traces
        assert any(item["figure"] == "raw HDF5 sampling" and "contradict" in item["reason"]
                   for item in run["skipped"])


def test_checkpoint_center_decoding_uses_saved_npz_units_without_summary_units(tmp_path):
    fes_measure_run(tmp_path)
    source = tmp_path / "input/resolved.json"
    config = json.loads(source.read_text())
    config["umbrella"] = {"centers_au": [[.5]], "stiffness_au": [[1.]]}
    source.write_text(json.dumps(config))
    (tmp_path / "checkpoints").mkdir()
    atomic_savez(tmp_path / "checkpoints/window_00000_replica_0000.npz", values=[[0.], [1.]])
    run = collect_run(tmp_path)
    window, = run["sampling_diagnostics"]["sampling"]["windows"]
    assert window["center_atomic"] == [.5]
    assert window["stiffness_atomic"] == [[1.]]
