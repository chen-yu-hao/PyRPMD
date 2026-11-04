"""Optional real-MPI equality check for the tempering workflow."""

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
from pyrpmd.runner import run


def _mapping():
    return {
        "task": "pimd",
        "temperature": "300 K",
        "beads": 3,
        "seed": 114,
        "native": False,
        "system": {
            "symbols": ["H"],
            "masses": [1.0],
            "geometry": [[0.12, -0.03, 0.05]],
            "geometry_unit": "bohr",
        },
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.35}},
        "dynamics": {"integrator": "cayley-bcb", "timestep": 0.02, "loop_backend": "python"},
        "sampling": {
            "thermostat": "pile-l",
            "gamma_centroid": 0.6,
            "equilibration_steps": 2,
            "production_steps": 8,
            "sample_interval": 2,
            "replicas": 1,
        },
        "tempering": {
            "temperatures": ["150 K", "250 K", "450 K", "900 K"],
            "exchange_interval_steps": 2,
            "pairing": "odd_even",
            "seed": 51,
        },
        "parallel": {"backend": "serial", "trajectory_workers": 1},
        "output": {"checkpoint_steps": 3},
    }


def _compare_npz(left, right):
    with load_checkpoint(left, require_integrity=True) as a, load_checkpoint(right, require_integrity=True) as b:
        assert a.files == b.files
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_four_rank_mpi_matches_serial_bitwise(tmp_path, mpi_launch):
    pytest.importorskip("mpi4py")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPI launcher is unavailable")
    mapping = _mapping()
    serial = tmp_path / "serial"
    mpi = tmp_path / "mpi"
    run(mapping, output=serial)
    spec = tmp_path / "input.json"
    spec.write_text(json.dumps(mapping))
    script = Path(__file__).with_name("mpi_tempering_smoke.py")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    command = mpi_launch(launcher, 4, env=env) + [sys.executable, str(script), str(spec), str(mpi)]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout
    assert "tempering: PASS" in completed.stdout
    _compare_npz(serial / "checkpoints/tempering.npz", mpi / "checkpoints/tempering.npz")
    for source in sorted((serial / "samples").glob("*.npz")):
        _compare_npz(source, mpi / "samples" / source.name)
