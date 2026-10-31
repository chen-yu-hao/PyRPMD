"""Dynamic MPI: mpiexec -n 5 python -m pyrpmd run examples/drpmd/mpi_dynamic.py.

One coordinator plus two trajectory groups, each with two bead ranks. This
free-scattering mechanics example is not a chemical rate or scaling benchmark.
"""

import os
from pathlib import Path
import tomllib


def build():
    config = tomllib.loads(Path(__file__).with_name("free_scattering.inp").read_text())
    groups = int(os.environ.get("PYRPMD_TRAJECTORY_GROUPS", "2"))
    bead_workers = int(os.environ.get("PYRPMD_BEAD_WORKERS", "2"))
    # Keep physical parent allocation independent of the number of worker groups.
    config["sampling"]["parents"] = int(os.environ.get("PYRPMD_PARENTS", "2"))
    config["parallel"] = {
        "backend": "mpi",
        "mode": "bead",
        "trajectory_groups": groups,
        "bead_workers": bead_workers,
        "threads_per_rank": 1,
        "pes_ranks": 1,
        "coordinator_ranks": 1,
        "scheduling": "dynamic",
    }
    config["output"]["directory"] = "runs/mpi_dynamic"
    return config
