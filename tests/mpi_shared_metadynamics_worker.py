"""MPI subprocess helper for distributed shared-metadynamics tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pyrpmd.system import System
from pyrpmd.workflows.metadynamics import run_metadynamics


def config(walkers):
    return {
        "task": "metadynamics", "beta": 2.0, "beads": 2, "dt_au": 0.02,
        "seed": 73, "native": False,
        "potential": {"provider": "builtin", "name": "harmonic",
                      "params": {"k": 0.2}},
        "sampling": {"equilibration_steps": 1, "production_steps": 6,
                     "sample_interval": 2, "andersen_interval": 3,
                     "replicas": walkers},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0,
                 "axis": "x"}],
        "metadynamics": {"shared_walkers": True,
                         "walkers_mode": "mpi-sync",
                         "mode": "well_tempered", "sigma": [0.3],
                         "height": 0.04, "pace": 2, "bias_factor": 5.0},
        "analysis": {"bins": [5], "bootstrap": 0},
        "output": {"checkpoint_steps": 3, "store_configurations": True},
    }


if __name__ == "__main__":
    output, walkers = Path(sys.argv[1]), int(sys.argv[2])
    system = System(("H",), np.array([1.0]), np.array([[0.1, 0.0, 0.0]]))
    run_metadynamics(config(walkers), system, output)
