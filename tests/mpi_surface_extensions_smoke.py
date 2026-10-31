"""Four-rank G2/B2 checks of NHC MetaD and source-FES conditional rates.

Usage: mpiexec -n 4 python tests/mpi_surface_extensions_smoke.py NEW_DIRECTORY
Compares all numerical checkpoints with an independent serial subprocess.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np

from pyrpmd.dynamics import Dynamics
from pyrpmd.io import write_json
from pyrpmd.parallel import root_call
from pyrpmd.runner import run, resume_run
from test_surface_source import source_job, analytic_source

PARALLEL = {"backend": "mpi", "mode": "bead", "trajectory_groups": 2,
            "trajectory_workers": 2, "bead_workers": 2, "threads_per_rank": 1}


def meta_job():
    job = source_job()
    job["beads"] = 4
    job["sampling"].update(thermostat="nose-hoover-chain", replicas=2,
                            thermostat_params={"tdamp": .4, "reset_after_equilibration": True})
    job["metadynamics"]["walls_during_equilibration"] = False
    return job


def run_case(root, parallel):
    for name, job in (("meta", meta_job()), ("rate", json.loads((root / "rate-job.json").read_text()))):
        job["parallel"] = parallel
        target = root / ("mpi" if parallel["backend"] == "mpi" else "serial") / name
        result = run(job, output=target)
        if name == "rate":
            assert result["kappa"] == 1.
            assert result["rate_estimate_s_inv"] > 0
        else:
            assert result["dimension"] == 3 and result["hills_per_replica"] == [3, 3]
        with patch.object(Dynamics, "step", side_effect=AssertionError("completed resume propagated")), \
             patch.object(Dynamics, "propagate_block", side_effect=AssertionError("completed resume propagated")):
            second = resume_run(target)
        assert result["status"] == second["status"]


def serial(root):
    prefixes = ("OMPI_", "PMI_", "PMIX_", "I_MPI_", "HYDRA_", "MPICH_", "MPI_LOCAL")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    result = subprocess.run([sys.executable, __file__, "--serial", str(root)], env=env,
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def compare(root):
    count = 0
    patterns = ("meta/checkpoints/*.npz", "meta/fes.npz", "rate/checkpoints/*.npz",
                "rate/surface_pmf.npz", "rate/recrossing.npz")
    for pattern in patterns:
        paths = sorted((root / "serial").glob(pattern))
        assert paths
        for first in paths:
            second = root / "mpi" / first.relative_to(root / "serial")
            with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f"{first}:{key}")
            count += 1
    print(f"PASS: periodic 3D NHC MetaD + conditional surface rate; {count} NPZ files match serial exactly; completed resumes pass.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serial":
        root = Path(sys.argv[2]).resolve()
        job = analytic_source(root / "fixture")
        write_json(root / "rate-job.json", job)
        run_case(root, {"backend": "serial"})
    else:
        from mpi4py import MPI
        if len(sys.argv) != 2 or MPI.COMM_WORLD.size != 4:
            raise SystemExit(__doc__)
        root = Path(sys.argv[1]).resolve()
        root_call(serial, root)
        run_case(root, PARALLEL)
        root_call(compare, root)
