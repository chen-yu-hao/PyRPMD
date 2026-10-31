"""Run actual G2 x B2 REUS and compare every saved array against serial."""
from pathlib import Path
import json
import os
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from pyrpmd.constants import KB_AU
from pyrpmd.dynamics import Dynamics
from pyrpmd.input import JobSpec
from pyrpmd.parallel import root_call
from pyrpmd.runner import run_resolved, resume_run, analyze_run
from pyrpmd.workflows.fes import jsonable

PARALLEL = dict(backend="mpi", mode="bead", trajectory_groups=2,
                trajectory_workers=2, bead_workers=2, threads_per_rank=1)


def run_case(root, parallel):
    for thermostat in ("andersen", "pile-l"):
        job = {
            "task": "fes", "temperature": 1 / (1.7 * KB_AU), "beads": 4,
            "seed": 676, "native": False,
            "system": {"symbols": ["H"], "masses": [2.3], "geometry": [[0, 0, 0]]},
            "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .4}},
            "dynamics": {"timestep": .03},
            "sampling": {"thermostat": thermostat, "equilibration_steps": 20,
                         "production_steps": 120, "interval_steps": 3,
                         "andersen_interval": 7, "replicas": 2},
            "cvs": [{"name": a, "kind": "cartesian", "atom": 0, "axis": a} for a in ("x", "y")],
            "umbrella": {"centers": [[-.2,-.2], [-.2,.2], [.2,-.2], [.2,.2]],
                         "stiffness": [[1.,.2], [.2,.8]]},
            "replica_exchange": {"enabled": True, "interval_steps": 9},
            "analysis": {"bins": [4, 5], "bootstrap": 4},
            "output": {"checkpoint_steps": 11},
        }
        if thermostat == "pile-l":
            job["sampling"]["gamma_centroid"] = .5
        config, system = JobSpec.from_mapping(job).resolve()
        config["parallel"] = dict(parallel)
        directory = root / thermostat
        first = run_resolved(config, system, output=directory)
        assert first["samples"] == 320
        with patch.object(Dynamics, "step", side_effect=AssertionError("resume propagated")), \
             patch.object(Dynamics, "sample_step", side_effect=AssertionError("resume sampled")), \
             patch.object(Dynamics, "propagate_block", side_effect=AssertionError("resume propagated")):
            second = resume_run(directory)
            third = analyze_run(directory, recompute=True)
        for result in (second, third):
            assert result["samples"] == first["samples"]
            assert jsonable(result["replica_exchange"]) == jsonable(first["replica_exchange"])


def serial_reference(directory):
    prefixes = ("OMPI_", "PMI_", "PMIX_", "I_MPI_", "HYDRA_", "MPICH_", "MPI_LOCAL")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--serial", str(directory)],
                   env=env, check=True, timeout=120)


def compare(root):
    arrays = files = 0
    for thermostat in ("andersen", "pile-l"):
        left, right = root / "serial" / thermostat, root / "mpi" / thermostat
        for pattern in ("checkpoints/exchange_*.npz", "fes.npz"):
            paths = sorted(left.glob(pattern))
            assert paths and len(paths) == len(list(right.glob(pattern)))
            for path in paths:
                with np.load(path, allow_pickle=False) as a, np.load(right / path.relative_to(left), allow_pickle=False) as b:
                    assert set(a.files) == set(b.files)
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key], err_msg=f"{thermostat}:{path.name}:{key}")
                        arrays += 1
                files += 1
        assert json.loads((left / "exchange.json").read_text()) == json.loads((right / "exchange.json").read_text())
    result = dict(files=files, arrays=arrays, ranks=4, trajectory_groups=2, bead_ranks=2,
                  thermostats=["andersen", "pile-l"], exact_serial_equivalence=True,
                  completed_resume=True, offline_analysis=True)
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def run_mpi_exchange(root):
    from mpi4py import MPI
    assert MPI.COMM_WORLD.size == 4, "launch exactly four ranks"
    root = Path(root).resolve()
    root_call(serial_reference, root / "serial")
    run_case(root / "mpi", PARALLEL)
    root_call(compare, root)


if __name__ == "__main__":
    if sys.argv[1] == "--serial":
        run_case(Path(sys.argv[2]), {"backend": "serial"})
    else:
        run_mpi_exchange(sys.argv[1])
