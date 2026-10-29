"""Opt-in MPI integration: mpiexec -n 4 python tests/mpi_integration.py.

Exercises actual MPI collectives with two trajectories, each with two bead
ranks. Four ranks on one host verify communication, not multi-node scaling.
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
from mpi4py import MPI

from pyrpmd.parallel import make_task_potential, root_call, run_tasks
from pyrpmd.workflows.drpmd import KB, run_drpmd

PARALLEL = {
    "backend": "mpi",
    "mode": "bead",
    "trajectory_groups": 2,
    "trajectory_workers": 2,
    "bead_workers": 2,
}
SYSTEM = {
    "symbols": ["H", "H"],
    "masses": np.array([1.0, 1.0]),
    "geometry": np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]),
    "fragments": [[0], [1]],
}


def evaluate_task(task):
    potential = make_task_potential({"provider": "harmonic", "k": 2.0}, SYSTEM)
    try:
        q = np.full((3, 2, 3), task, dtype=float)
        result = potential.evaluate(q)
        np.testing.assert_allclose(result.forces, -2.0 * q)
        return float(result.energy.sum())
    finally:
        potential.close()


def bad_initialization(task):
    make_task_potential({"provider": "missing-provider"}, SYSTEM)


def mock_gpu_route(task):
    # Routing-only test. This does not exercise CUDA or claim GPU correctness.
    class Provider:
        def __init__(self, device):
            self.metadata = {"execution_device": device}

    with patch(
        "pyrpmd.potentials.make_potential",
        side_effect=lambda spec, system: Provider(spec["device"]),
    ):
        return make_task_potential({"provider": "torch"}, SYSTEM).metadata[
            "execution_device"
        ]


def config():
    return {
        "beta": 1.0,
        "temperature_K": 1 / KB,
        "beads": 3,
        "dt_au": 0.1,
        "seed": 17,
        "native": False,
        "system": {"separation_au": 4.0},
        "potential": {"provider": "free"},
        "sampling": {
            "equilibration_steps": 4,
            "interval_steps": 2,
            "andersen_interval": 2,
        },
        "collision": {"trajectories": 5, "bmax_au": 0.5, "max_steps": 200},
        "outcomes": {
            "bond_cutoffs_au": {"H-H": 0.3},
            "channels": {},
            "exit_distance_au": 4.0,
            "persistence_steps": 2,
            "check_interval": 1,
        },
        "output": {"trajectory_stride": 5, "checkpoint_steps": 10},
        "parallel": PARALLEL,
    }


def main():
    try:
        run_tasks(evaluate_task, [0.0], {"backend": "serial"})
    except ValueError as exc:
        assert "requires parallel.backend='mpi'" in str(exc), str(exc)
    else:
        raise AssertionError(
            "A local backend under multi-rank MPI would duplicate tasks"
        )
    results = run_tasks(evaluate_task, [0.0, 1.0, 2.0, 3.0, 4.0], PARALLEL)

    def check_results():
        np.testing.assert_allclose(results, 18 * np.arange(5.0) ** 2)

    root_call(check_results)
    try:
        run_tasks(bad_initialization, [0, 1], PARALLEL)
    except RuntimeError as exc:
        assert "PES initialization failed" in str(exc), str(exc)
    else:
        raise AssertionError("MPI PES initialization failure was swallowed")
    routing = run_tasks(
        mock_gpu_route,
        range(4),
        {
            "backend": "mpi",
            "mode": "gpu",
            "trajectory_groups": 4,
            "bead_workers": 1,
            "devices": [0, 1],
            "trajectories_per_gpu": 2,
        },
    )
    root_call(
        lambda: np.testing.assert_array_equal(
            routing, ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
        )
    )
    directory = root_call(tempfile.mkdtemp, prefix="pyrpmd-mpi-test-")
    try:
        c = config()
        first = run_drpmd(c, SYSTEM, directory)
        assert first["trajectories"] == 5
        assert first["parent_thermal_steps"] == {"0": 10, "1": 8}
        assert first["counts"]["nonreactive"] == 5
        c["resume"] = True
        second = run_drpmd(c, SYSTEM, directory)
        assert first["records"] == second["records"]
        root_call(
            lambda: print(
                "MPI G=2 B=2: ordered tasks, collective failure, continuous parents and restart passed; MPI GPU node-local routing passed with a mock provider (no CUDA test)",
                flush=True,
            )
        )
    finally:
        root_call(shutil.rmtree, directory)


if __name__ == "__main__":
    main()
