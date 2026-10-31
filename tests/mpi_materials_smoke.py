"""Opt-in four-rank test of materials streams, site rates and offline analysis.

mpiexec -n 4 python tests/mpi_materials_smoke.py /fresh/shared/output
Uses two independent replica groups, each with two bead potential ranks.
This checks single-host collectives and ownership, not cross-node scaling.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from pyrpmd import analyze_run, resume_run, run
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.parallel import root_call
from pyrpmd.trajectory import materialize_stream, validate_stream


def main():
    if MPI.COMM_WORLD.size != 4:
        raise ValueError("This smoke test requires exactly four MPI ranks")
    output = Path(sys.argv[1]).resolve()
    mapping = {
        "task": "pimd", "temperature": "300 K", "beads": 4, "seed": 739,
        "system": {
            "symbols": ["H"], "geometry": [[0.1, 0.2, 0.3]],
            "cell": [[3.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.0, 0.0, 8.0]],
            "pbc": [True, True, False],
        },
        "potential": {"provider": "builtin", "name": "periodic_cosine",
                      "params": {"height": 0.002, "confinement": 0.01}},
        "dynamics": {"timestep": "0.1 fs", "block_steps": 11},
        "sampling": {"thermostat": "pile-l", "equilibration_steps": 20,
                     "production_steps": 160, "interval_steps": 2, "replicas": 3},
        "parallel": {"backend": "mpi", "trajectory_groups": 2,
                     "trajectory_workers": 2, "mode": "bead", "bead_workers": 2,
                     "threads_per_rank": 1},
        "output": {"checkpoint_steps": 17},
    }
    for task in ("pimd", "diffusion"):
        config = copy.deepcopy(mapping)
        config["task"] = task
        if task == "diffusion":
            # Force-free periodic transport has an exact ballistic diagnostic.
            config["potential"] = {"provider": "free"}
            config["diffusion"] = {"axes": [0, 1], "bootstrap": 20}
        directory = output / task
        first = run(config, output=directory)
        second = resume_run(directory, execution={"block_steps": 7})

        def check():
            assert first["samples_per_replica"] == [81, 81, 81]
            assert first["samples_per_replica"] == second["samples_per_replica"]
            assert first["coordinate_convention"] == "continuous_unwrapped_ring_polymer_centroids"
            assert first["pbc"] == [True, True, False]
            if task == "pimd":
                assert first["estimators"] == second["estimators"]
            else:
                assert first["diffusion"] == second["diffusion"]
                assert first["analysis_status"] == "ballistic_or_drifting"
                assert first["diffusion_m2_s"] is None
            paths = sorted((directory / "samples").glob("materials_*/manifest.json"))
            assert len(paths) == 3  # No bead worker writes a duplicate stream.
            assert first["trajectory_storage"] == ["stream"] * 3
            for replica, path in enumerate(paths):
                manifest = json.loads(path.read_text())
                assert manifest["format"] == "PyRPMD.materials-trajectory-stream.v1"
                with load_checkpoint(directory / "checkpoints" / f"materials_{replica:04d}.npz",
                                     require_integrity=True) as saved:
                    descriptor = json.loads(str(saved["trajectory"]))
                assert manifest["trajectory"] == descriptor
                assert validate_stream(path.parent, descriptor)["frames"] == 81
                frames = materialize_stream(path.parent, descriptor, fields=["sample_steps"])
                np.testing.assert_array_equal(frames["sample_steps"], np.arange(20, 181, 2))

        root_call(check)
        if task == "diffusion":
            report = analyze_run(directory, recompute=True,
                                 diffusion={"axes": [0], "fit_range": [None, None]})

            def check_analysis():
                assert report["diffusion"]["input"]["axes"] == [0]
                assert report["analysis_status"] == "ballistic_or_drifting"

            root_call(check_analysis)
    # A small thermal-mass validation model tests every site-rate collective:
    # FES windows, root MBAR analysis, constrained parents, paired children and
    # final aggregation. It is not a physical hydrogen benchmark.
    site_config = {
        "task": "site_rate", "temperature": 1 / (10 * KB_AU), "beads": 4,
        "seed": 13, "native": False,
        "system": {"symbols": ["H"], "masses": [2.0], "geometry": [[0., 0., 0.]],
                   "cell": [[4., 0., 0.], [0., 8., 0.], [0., 0., 8.]],
                   "pbc": [True, False, False]},
        "potential": {"provider": "builtin", "name": "periodic_cosine",
                      "params": {"height": [.1, 0., 0.], "confinement": .1}},
        "dynamics": {"timestep": .05, "loop_backend": "python", "block_steps": 11},
        "sampling": {"thermostat": "andersen", "equilibration_steps": 200,
                     "production_steps": 1000, "sample_interval": 2,
                     "andersen_interval": 20, "replicas": 1},
        "cvs": [{"kind": "fractional", "name": "x", "atom": 0, "axis": 0}],
        "umbrella": {"cvs": ["x"], "centers": [-.6, -.3, 0., .3, .6], "stiffness": 2.},
        "analysis": {"bins": 10, "ranges": [[-.8, .8]], "bootstrap": 0},
        "site_rate": {"cv": "x", "basin": [-.5, .5], "dividing_surface": .5, "direction": 1},
        "recrossing": {"parent_replicas": 2, "parent_samples": 3,
                      "equilibration_steps": 20, "parent_interval_steps": 5,
                      "child_pairs": 2, "child_steps": 40, "plateau_fraction": .5,
                      "bootstrap": 8, "constraint_tolerance": 1e-11},
        "parallel": copy.deepcopy(mapping["parallel"]),
        "output": {"checkpoint_steps": 250},
    }
    directory = output / "site_rate"
    first = run(site_config, output=directory)
    second = resume_run(directory, execution={"block_steps": 7})

    def check_site_rate():
        assert first["pmf"]["ready"] and first["k_qtst_s"] > 0
        assert first["child_pairs"] == second["child_pairs"] == 12
        assert first["parent_configurations"] == second["parent_configurations"] == 6
        assert first["rate_estimate_s"] == second["rate_estimate_s"]
        assert first["kappa"] == second["kappa"]
        assert first["rate_unit"] == "s^-1" and not first["converged"]
        assert first["rate_standard_error_s"] is None
        assert len(list((directory / "pmf" / "checkpoints").glob("window_*.npz"))) == 5
        assert len(list((directory / "checkpoints").glob("site_parent_*.npz"))) == 2
        assert len(list((directory / "checkpoints").glob("site_child_*.npz"))) == 12

    root_call(check_site_rate)
    root_call(lambda: print("Materials PIMD + RPMD diffusion + site rate, PBC, G2 B2, exact resume and offline analysis: PASS", flush=True))


if __name__ == "__main__":
    main()
