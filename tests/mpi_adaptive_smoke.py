"""Frozen adaptive diagnostics and exact reanalysis under four MPI ranks.

Run: mpiexec -n 4 python tests/mpi_adaptive_smoke.py /fresh/shared/output
This tests two trajectory groups with two bead ranks each, not scaling.
"""

import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from pyrpmd.parallel import root_call
from pyrpmd.system import System
from pyrpmd.workflows.fes import jsonable, reanalyze_fes, run_fes


def main(output):
    if MPI.COMM_WORLD.size != 4:
        raise ValueError("This smoke test requires exactly four MPI ranks")
    system = System(("H",), np.array([2.0]), np.zeros((1, 3)))
    config = {
        "task": "fes", "beta": 2.0, "beads": 4, "dt_au": 0.03,
        "seed": 913, "native": False,
        "potential": {"provider": "harmonic", "params": {"k": 0.2}},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}],
        "umbrella": {"centers": [-0.3, 0.3], "stiffness": 1.0},
        "sampling": {
            "equilibration_steps": 40, "production_steps": 120,
            "sample_interval": 3, "andersen_interval": 20, "replicas": 2,
        },
        "adaptive_umbrella": {
            "enabled": True, "pilot_steps": 120, "pilot_equilibration_steps": 40,
            "max_rounds": 3, "max_windows": 8,
            "target_overlap": 0.001, "min_effective_samples": 1.0,
            "coverage_bins": [1], "coverage_ranges": [[-100., 100.]],
            "min_bin_effective_samples": 1.0, "target_coverage": 1.0,
        },
        "analysis": {"bins": 8, "bootstrap": 0},
        "parallel": {
            "backend": "mpi", "trajectory_groups": 2, "trajectory_workers": 2,
            "mode": "bead", "bead_workers": 2, "threads_per_rank": 1,
        },
        "output": {"checkpoint_steps": 80},
    }
    first = run_fes(config, system, output)
    replay = reanalyze_fes(config, system, output)
    assert isinstance(first["streams"], int) and "data" not in first
    assert jsonable(first) == jsonable(replay)
    adaptive = first["adaptive_umbrella"]
    assert adaptive["pilot_target_met"]
    assert adaptive["production_target_met"]
    assert adaptive["production_coverage"]["scope"] == "frozen_production"
    assert adaptive["overlap_metric"] == "normalized_mbar_cosine"
    assert first["samples"] == adaptive["frozen_windows"] * 2 * 40
    assert adaptive["pilot_samples_used_in_production"] == 0
    assert all(value == adaptive for value in MPI.COMM_WORLD.allgather(adaptive))

    def check_saved():
        plan = json.loads((output / "adaptive-plan.json").read_text())
        assert plan["schema_version"] == 3
        assert plan["production_seed"] not in [row["seed"] for row in plan["rounds"]]
        assert json.loads((output / "summary.json").read_text()) == jsonable(first)
        print("Adaptive FES G2 B2 MPI production diagnostics and exact reanalysis: PASS", flush=True)

    root_call(check_saved)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
