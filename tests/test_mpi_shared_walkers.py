"""Optional real-MPI trajectory parity for synchronous shared walkers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.system import System
from pyrpmd.workflows.metadynamics import run_metadynamics


def config(walkers):
    return {
        "task": "metadynamics", "beta": 2.0, "beads": 2, "dt_au": 0.02,
        "seed": 73, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic",
                      "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 1, "production_steps": 6,
                     "sample_interval": 2, "andersen_interval": 3,
                     "replicas": walkers},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0,
                 "axis": "x"}],
        "metadynamics": {"shared_walkers": True,
                         "mode": "well_tempered", "sigma": [0.3],
                         "height": 0.04, "pace": 2, "bias_factor": 5.0},
        "analysis": {"bins": [5], "bootstrap": 0},
        "output": {"checkpoint_steps": 3, "store_configurations": True},
    }


def read(path, name, kind):
    with load_checkpoint(path / "checkpoints" / name,
                         expected_kind=kind, require_integrity=True) as data:
        return {key: data[key].copy() for key in data.files}


@pytest.mark.parametrize("walkers", [2, 4])
def test_real_mpi_shared_walker_history_and_trajectory_are_bitwise_serial(
        tmp_path, walkers):
    pytest.importorskip("mpi4py")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPI launcher is not installed")
    if any(name in os.environ for name in (
            "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKID")):
        pytest.skip("do not nest an MPI launcher")
    system = System(("H",), np.array([1.0]), np.array([[0.1, 0.0, 0.0]]))
    serial = tmp_path / "serial"
    distributed = tmp_path / "mpi"
    run_metadynamics(config(walkers), system, serial)
    helper = Path(__file__).with_name("mpi_shared_metadynamics_worker.py")
    environment = os.environ.copy()
    environment.update(OMPI_ALLOW_RUN_AS_ROOT="1",
                       OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    completed = subprocess.run(
        [launcher, "-n", str(walkers), sys.executable, str(helper),
         str(distributed), str(walkers)],
        text=True, capture_output=True, env=environment, timeout=120)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    reference = read(serial, "shared_metadynamics.npz",
                     "PyRPMD.shared-metadynamics.v1")
    actual = read(distributed, "shared_metadynamics_mpi.npz",
                  "PyRPMD.shared-metadynamics.mpi.v1")
    for key in ("elapsed", "time", "q", "p", "sample_steps", "samples",
                "sample_bias", "configurations"):
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
    assert json.loads(str(actual["hills"])) == json.loads(str(reference["hills"]))
    assert json.loads(str(actual["rng"])) == json.loads(str(reference["rng"]))
