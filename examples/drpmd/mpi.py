"""MPI example: defaults to mpiexec -n 4, with G=2, B=2, Q=1."""

import os
from pathlib import Path
import tomllib


def build():
    config = tomllib.loads(Path(__file__).with_name("free_scattering.inp").read_text())
    groups = int(os.environ.get("PYRPMD_TRAJECTORY_GROUPS", "2"))
    bead_workers = int(os.environ.get("PYRPMD_BEAD_WORKERS", "2"))
    config["sampling"]["parents"] = groups
    config["collision"]["trajectories"] = max(5, 2 * groups)
    config["parallel"] = {
        "backend": "mpi",
        "mode": "bead",
        "trajectory_groups": groups,
        "bead_workers": bead_workers,
        "threads_per_rank": 1,
        "pes_ranks": 1,
        "coordinator_ranks": 0,
        "scheduling": "static",
    }
    config["output"]["directory"] = "runs/mpi"
    return config
