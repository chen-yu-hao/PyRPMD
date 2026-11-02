"""The serial/MPI oracle ignores only a change of output-directory root."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint


compare = runpy.run_path(str(Path(__file__).with_name("mpi_exchange_smoke.py")))["compare"]


def write_pair(root):
    for execution in ("serial", "mpi"):
        for thermostat in ("andersen", "pile-l"):
            directory = root / execution / thermostat
            atomic_savez(directory / "checkpoints/exchange_0000.npz", q=np.arange(12).reshape(4, 3))
            metadata = {"beta": 1.7, "measure": "cv_density", "uncertainty_kind": "block_bootstrap",
                        "files": {"data": str(directory / "fes.npz"), "table": str(directory / "fes.csv")}}
            atomic_savez(directory / "fes.npz", kind="fes_result",
                         free_energy=np.array([0.0, 1.0]), fes_metadata=json.dumps(metadata))
            (directory / "exchange.json").write_text("{}\n")


def test_different_output_roots_keep_equivalent_physics(tmp_path):
    write_pair(tmp_path)
    compare(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["exact_serial_equivalence"]


@pytest.mark.parametrize("changed", ["beta", "measure", "uncertainty_kind", "filename", "array"])
def test_execution_normalization_never_hides_physical_or_output_changes(tmp_path, changed):
    write_pair(tmp_path)
    path = tmp_path / "mpi/andersen/fes.npz"
    with load_checkpoint(path) as saved:
        payload = {key: saved[key] for key in saved.files}
    metadata = json.loads(str(payload["fes_metadata"]))
    if changed == "filename":
        metadata["files"]["data"] = str(path.parent / "other.npz")
    elif changed == "array":
        payload["free_energy"][1] += 0.1
    else:
        metadata[changed] = {"beta": 2.0, "measure": "rate_flux", "uncertainty_kind": "replica_spread"}[changed]
    payload["fes_metadata"] = json.dumps(metadata)
    atomic_savez(path, kind="fes_result", **payload)
    with pytest.raises(AssertionError):
        compare(tmp_path)
