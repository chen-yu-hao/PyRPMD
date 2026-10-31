"""Actual surface-rate MPI integration with a separate serial reference.

Usage: mpiexec -n 4 python tests/mpi_surface_smoke.py /fresh/shared/output
All four ranks use G=2 trajectory groups x B=2 bead ranks. Rank zero first
runs a fresh ordinary Python process for the serial reference; that process
does not initialize MPI. Both runs execute the analytic periodic surface PES,
fixed substrate, density PMF, constrained parents and paired recrossing.
"""

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
from pyrpmd.runner import run_resolved, resume_run


PARALLEL = {
    "backend": "mpi", "mode": "bead", "trajectory_groups": 2,
    "trajectory_workers": 2, "bead_workers": 2, "threads_per_rank": 1,
}


def job():
    factory = Path(__file__).resolve().parents[1] / "examples" / "surface" / "toy_surface.py"
    return {
        "task": "surface_rate", "temperature": 1 / (2 * KB_AU),
        "beads": 4, "seed": 211, "native": False,
        "system": {
            "symbols": ["H", "Cu"], "masses": [2.0, 20.0],
            "geometry": [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            "cell": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 12.0]],
            "pbc": [True, True, False], "fixed_atoms": [1],
        },
        "potential": {
            "provider": "python", "factory": f"{factory}:build",
            "params": {"barrier": 0.2, "transverse_barrier": 0.2,
                       "height": 1.0, "vertical_k": 1.0},
        },
        "dynamics": {"timestep": 0.05},
        "sampling": {
            "equilibration_steps": 50, "production_steps": 600,
            "interval_steps": 5, "andersen_interval": 10, "replicas": 1,
        },
        "cvs": [{"name": "migration", "kind": "surface_lateral", "atoms": [0],
                 "reference_atoms": [1], "direction": [1.0, 0.0, 0.0]}],
        "umbrella": {"centers": np.linspace(-2.5, 2.5, 11).tolist(), "stiffness": 2.0},
        "surface": {"reactant_basin": [-2.0, 2.0], "dividing_surface": 2.0,
                    "product_side": "higher"},
        "recrossing": {
            "parent_replicas": 2, "parent_samples": 3, "equilibration_steps": 7,
            "parent_interval_steps": 5, "child_pairs": 2, "child_steps": 24,
            "plateau_fraction": 0.5, "bootstrap": 4,
        },
        "analysis": {"bins": 12, "ranges": [[-3.0, 3.0]], "min_count": 1,
                     "bootstrap": 3},
        "output": {"checkpoint_steps": 50},
    }


def run_case(directory, parallel):
    config, system = JobSpec.from_mapping(job()).resolve()
    config["parallel"] = dict(parallel)
    first = run_resolved(config, system, output=directory)
    assert first["pmf"]["ready"], first["pmf"]["status"]
    assert first["child_trajectories"] == 24
    assert first["parent_configurations"] == 6
    assert first["rate_estimate_s_inv"] > 0
    assert first["rate_unit"] == "s^-1"
    assert not any("cm3" in key for key in first)
    # This runs on every MPI rank; any accidental post-completion propagation
    # fails collectively through the existing executor error handling.
    with patch.object(Dynamics, "step", side_effect=AssertionError("completed resume propagated")), \
         patch.object(Dynamics, "propagate_block", side_effect=AssertionError("completed resume propagated")):
        second = resume_run(directory)
    for key in ("rate_estimate_s_inv", "rate_standard_error_s_inv", "kappa", "status",
                "child_trajectories", "rate_quality", "plateau_diagnostics"):
        assert first[key] == second[key], key
    assert first["pmf"]["k_qtst_s_inv"] == second["pmf"]["k_qtst_s_inv"]
    return first


def serial_reference(directory):
    # Remove launcher state from the subprocess. The reference must exercise
    # the actual serial executor instead of hiding an initialized MPI world.
    prefixes = ("OMPI_", "PMI_", "PMIX_", "I_MPI_", "HYDRA_", "MPICH_", "MPI_LOCAL")
    env = {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--serial", str(directory)],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120,
    )
    if process.returncode:
        raise RuntimeError("serial reference failed:\n" + process.stdout[-6000:])
    return process.stdout.strip()


def compare_outputs(root):
    serial, mpi = root / "serial", root / "mpi"
    patterns = (
        "umbrella/checkpoints/window_*.npz", "checkpoints/parent_*.npz",
        "checkpoints/child_*.npz", "surface_pmf.npz", "recrossing.npz",
    )
    files = 0
    for pattern in patterns:
        left = sorted(serial.glob(pattern))
        right = sorted(mpi.glob(pattern))
        assert len(left) == len(right) > 0, pattern
        for a_path, b_path in zip(left, right):
            assert a_path.relative_to(serial) == b_path.relative_to(mpi)
            with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
                assert set(a.files) == set(b.files), str(a_path)
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key], err_msg=f"{a_path.name}:{key}")
                if "q" in a.files:
                    np.testing.assert_array_equal(a["q"][:, 1], 0.0)
                    np.testing.assert_array_equal(a["p"][:, 1], 0.0)
                if "snapshots" in a.files:
                    np.testing.assert_array_equal(a["snapshots"][:, :, 1], 0.0)
            files += 1
    a = json.loads((serial / "results" / "summary.json").read_text())
    b = json.loads((mpi / "results" / "summary.json").read_text())
    # Each result points to its own output directory. Check the reference and
    # retained weights before comparing the directory-independent PMF data;
    # all NPZ arrays, including weights, were compared exactly above.
    for result, directory in ((a, serial), (b, mpi)):
        pmf = result["pmf"]
        weights_file = Path(pmf["sample_weights_file"])
        assert weights_file.resolve() == (directory / "surface_pmf.npz").resolve()
        with np.load(weights_file, allow_pickle=False) as saved:
            assert pmf["sample_weights_key"] in saved.files
        pmf["sample_weights_file"] = str(weights_file.resolve().relative_to(directory.resolve()))
    for key in ("task", "status", "rate_unit", "rate_s_inv", "rate_estimate_s_inv",
                "rate_standard_error_s_inv", "rate_percentile_95_ci_s_inv", "rate_uncertainty",
                "kappa", "kappa_standard_error", "plateau_detected", "child_trajectories",
                "parent_configurations", "pmf", "surface", "converged",
                "rate_quality", "plateau_diagnostics"):
        assert a[key] == b[key], key
    print(
        f"Surface rate serial versus G2 B2 MPI: PASS; {files} array/checkpoint files "
        "match exactly, fixed substrate remains exact, and both completed resumes pass.",
        flush=True,
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--serial":
        run_case(Path(sys.argv[2]).resolve(), {"backend": "serial"})
        print("Surface rate serial reference and completed resume: PASS", flush=True)
        return
    if len(sys.argv) != 2:
        raise SystemExit("usage: mpiexec -n 4 python tests/mpi_surface_smoke.py /fresh/shared/output")
    from mpi4py import MPI

    if MPI.COMM_WORLD.size != 4:
        raise ValueError("surface MPI smoke test requires exactly four ranks (G=2, B=2)")
    root = Path(sys.argv[1]).resolve()
    root_call(serial_reference, root / "serial")
    run_case(root / "mpi", PARALLEL)
    root_call(compare_outputs, root)


if __name__ == "__main__":
    main()
