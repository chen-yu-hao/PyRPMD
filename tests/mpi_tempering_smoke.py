"""Opt-in four-rank tempering smoke test.

Run with ``mpiexec -n 4 python tests/mpi_tempering_smoke.py INPUT.json OUTPUT``.
Each replica owns one MPI trajectory group; only scalar reduced energies return
through the scheduler while full states are committed through the shared atomic
ensemble checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mpi4py import MPI

from pyrpmd.parallel import root_call
from pyrpmd.runner import run


def main():
    if MPI.COMM_WORLD.size != 4:
        raise ValueError("This tempering smoke test requires exactly four MPI ranks")
    mapping = json.loads(Path(sys.argv[1]).read_text())
    mapping["parallel"] = {
        "backend": "mpi",
        "mode": "serial",
        "trajectory_groups": 4,
        "trajectory_workers": 4,
        "bead_workers": 1,
        "threads_per_rank": 1,
        "scheduling": "static",
        "coordinator_ranks": 0,
        "pes_ranks": 1,
    }
    result = run(mapping, output=Path(sys.argv[2]))

    def check():
        assert result["status"] == "complete"
        assert result["replicas"] == 4
        assert result["converged"] is result["physical_converged"] is False
        print("T37 four-rank one-group-per-replica tempering: PASS", flush=True)

    root_call(check)


if __name__ == "__main__":
    main()
