"""Run explicitly under mpiexec -n 4; not part of unittest discovery.

Usage: python tests/mpi_workflow_smoke.py /fresh/shared/output
Checks G=2 trajectory groups x B=2 bead ranks, both workflows, and resume.
"""

import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from pyrpmd.system import System
from pyrpmd.runner import run_resolved, resume_run


def main():
    root = Path(sys.argv[1])
    parallel = {
        "backend": "mpi",
        "trajectory_groups": 2,
        "trajectory_workers": 2,
        "mode": "bead",
        "bead_workers": 2,
        "threads_per_rank": 1,
    }
    system = System(("H",), np.array([2.0]), np.zeros((1, 3)))
    config = {
        "task": "fes",
        "beta": 2.0,
        "temperature_K": 150.0,
        "beads": 4,
        "dt_au": 0.03,
        "seed": 2,
        "native": False,
        "potential": {"provider": "harmonic", "params": {"k": 0.2}},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": 0}],
        "umbrella": {"centers": [-0.7, 0.0, 0.7], "stiffness": 2.0},
        "sampling": {
            "equilibration_steps": 40,
            "production_steps": 120,
            "sample_interval": 3,
            "andersen_interval": 20,
        },
        "analysis": {"bins": 12, "bootstrap": 2},
        "parallel": parallel,
        "_base_directory": str(Path.cwd()),
        "output": {"directory": str(root / "fes"), "checkpoint_steps": 80},
    }
    first = run_resolved(config, system)
    second = resume_run(root / "fes")
    assert first["samples"] == second["samples"] == 120
    system = System(
        ("H", "H", "H"),
        np.array([2.0, 2.0, 2.0]),
        np.array([[-1.5, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        ((0,), (1, 2)),
    )
    config = {
        "task": "rate",
        "beta": 5.0,
        "temperature_K": 300.0,
        "beads": 4,
        "dt_au": 0.003,
        "seed": 17,
        "native": False,
        "potential": {"provider": "free"},
        "sampling": {
            "equilibration_steps": 50,
            "production_steps": 200,
            "sample_interval": 2,
            "andersen_interval": 20,
        },
        "reaction": {
            "fragments": [[0], [1, 2]],
            "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]],
            "rinf": 4.0,
        },
        "umbrella": {"centers": np.linspace(0, 1, 11).tolist(), "kappa": 10.0},
        "analysis": {"bins": 30, "bootstrap": 2},
        "parallel": parallel,
        "_base_directory": str(Path.cwd()),
        "output": {"directory": str(root / "rate"), "checkpoint_steps": 200},
        "recrossing": {
            "dividing_surface": 1.0,
            "parent_replicas": 2,
            "parent_samples": 2,
            "equilibration_steps": 20,
            "parent_interval_steps": 5,
            "child_pairs": 1,
            "child_steps": 20,
            "bootstrap": 10,
        },
    }
    first = run_resolved(config, system)
    second = resume_run(root / "rate")
    assert first["child_trajectories"] == second["child_trajectories"] == 8
    assert first["kappa"] == second["kappa"]
    if MPI.COMM_WORLD.rank == 0:
        print("FES + rate G2 B2 MPI workflows and exact resume: PASS", flush=True)


if __name__ == "__main__":
    main()
    from mpi_exchange_smoke import run_mpi_exchange
    run_mpi_exchange(Path(sys.argv[1]) / "replica-exchange")
