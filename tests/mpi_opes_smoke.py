"""Actual G2 x B2 OPES parity, frozen histogram, resume and offline analysis.

Run with ``mpiexec -n 4 python tests/mpi_opes_smoke.py /fresh/shared/output``.
``--serial /fresh/output`` exercises the same public workflow without MPI.
This is an execution/ownership check, not a scientific convergence study.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.dynamics import Dynamics
from pyrpmd.input import JobSpec
from pyrpmd.parallel import root_call
from pyrpmd.runner import run_resolved, resume_run, analyze_run


PARALLEL = dict(backend="mpi", mode="bead", trajectory_groups=2,
                trajectory_workers=2, bead_workers=2, threads_per_rank=1)


def checkpoint_hashes(directory):
    paths = sorted((directory / "checkpoints").glob("opes_*.npz"))
    assert len(paths) == 2, "one authoritative checkpoint per independent history"
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_case(directory, parallel):
    job = {
        "task": "opes", "temperature": 1 / (1.7 * KB_AU), "beads": 4,
        "seed": 811, "native": False,
        "system": {"symbols": ["H"], "masses": [2.3], "geometry": [[0.1, -0.05, 0.0]]},
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": 0.4}},
        "dynamics": {"timestep": 0.03},
        "sampling": {"equilibration_steps": 10, "production_steps": 60,
                     "sample_interval": 2, "andersen_interval": 7, "replicas": 2},
        "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                for axis in ("x", "y")],
        "opes": {"sigma": [0.3, 0.3], "barrier": 2.0, "bias_factor": 5.0,
                 "pace": 5, "freeze_after_steps": 30, "freeze_equilibration_steps": 10},
        "analysis": {"method": "reweighted_histogram", "bins": [3, 3],
                     "ranges": [[-2.0, 2.0], [-2.0, 2.0]], "bootstrap": 0},
        "output": {"checkpoint_steps": 11},
    }
    config, system = JobSpec.from_mapping(job).resolve()
    config["parallel"] = dict(parallel)
    first = run_resolved(config, system, output=directory)
    assert first["samples"] == 20
    assert first["kernels_per_replica"] == [6, 6]
    assert first["estimator"] == "reweighted_histogram"
    assert first["common_support"] is True
    assert first["converged"] is False
    hashes = root_call(checkpoint_hashes, directory)
    with patch.object(Dynamics, "step", side_effect=AssertionError("completed resume propagated")), \
         patch("pyrpmd.workflows.opes.make_task_potential", side_effect=AssertionError("completed resume loaded a potential")), \
         patch("pyrpmd.potentials.make_potential", side_effect=AssertionError("completed resume started a bead potential")):
        second = resume_run(directory)
        third = analyze_run(directory, recompute=True)
    for result in (second, third):
        for key in ("samples", "kernels_per_replica", "estimator", "freeze_after_steps",
                    "freeze_equilibration_steps", "log_z_per_replica",
                    "weight_effective_sample_size", "common_support", "converged"):
            assert result[key] == first[key], key
    assert root_call(checkpoint_hashes, directory) == hashes


def serial_reference(directory):
    prefixes = ("OMPI_", "PMI_", "PMIX_", "I_MPI_", "HYDRA_", "MPICH_", "MPI_LOCAL")
    env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--serial", str(directory)],
                   env=env, check=True, timeout=120)


def compare(root):
    left, right = root / "serial", root / "mpi"
    arrays = files = 0
    for pattern in ("checkpoints/opes_*.npz", "fes.npz"):
        paths = sorted(left.glob(pattern))
        assert paths and len(paths) == len(list(right.glob(pattern)))
        for path in paths:
            with np.load(path, allow_pickle=False) as a, np.load(right / path.relative_to(left), allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f"{path.name}:{key}")
                    arrays += 1
            files += 1
    assert (left / "fes.csv").read_bytes() == (right / "fes.csv").read_bytes()
    assert files == 3  # Two histories, with no duplicate writes by bead ranks.
    result = dict(files=files, arrays=arrays, ranks=4, trajectory_groups=2, bead_ranks=2,
                  histories=2, beads=4, retained_frozen_samples=20, kernels_per_history=6,
                  exact_serial_equivalence=True, completed_resume=True, offline_analysis=True,
                  unchanged_completed_checkpoints=True, converged=False)
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def run_mpi_opes(root):
    from mpi4py import MPI
    assert MPI.COMM_WORLD.size == 4, "launch exactly four ranks"
    root = Path(root).resolve()
    root_call(serial_reference, root / "serial")
    run_case(root / "mpi", PARALLEL)
    root_call(compare, root)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serial":
        run_case(Path(sys.argv[2]).resolve(), {"backend": "serial"})
        print("OPES serial reference, frozen histogram, completed resume and offline analysis: PASS", flush=True)
    elif len(sys.argv) == 2:
        run_mpi_opes(sys.argv[1])
    else:
        raise SystemExit("usage: mpi_opes_smoke.py [--serial] /fresh/shared/output")
