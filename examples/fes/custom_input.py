"""Ordinary trusted Python input; create arbitrary static differentiable CVs."""

from pathlib import Path
from pyrpmd.input import JobSpec


def build():
    here = Path(__file__).resolve().parent
    return JobSpec.from_mapping(
        {
            "task": "fes",
            "preset": "pilot",
            "temperature": "300 K",
            "beads": 4,
            "seed": 74,
            "system": {
                "symbols": ["H"],
                "masses": [1837.152647],
                "geometry": [[0.0, 0.0, 0.0]],
            },
            "potential": {"provider": "harmonic", "params": {"k": 0.002}},
            "dynamics": {"timestep": "0.1 fs"},
            "sampling": {
                "equilibration": "10 fs",
                "production": "40 fs",
                "interval": "0.2 fs",
                "andersen_interval": 40,
                "replicas": 2,
            },
            "cvs": [
                {
                    "name": "rotated",
                    "kind": "python",
                    "factory": str(here / "custom_cv.py") + ":build",
                    "estimator": "full_ring",
                    "params": {"atom": 0},
                    "units": ["bohr", "bohr"],
                }
            ],
            "umbrella": {
                "cvs": ["rotated"],
                "grid": [{"start": -1.0, "stop": 1.0, "count": 3}] * 2,
                "stiffness": ["0.01 hartree/bohr^2"] * 2,
            },
            "analysis": {"bins": [16, 16], "bootstrap": 20, "min_count": 2},
            "output": {
                "directory": str(here / "runs/custom"),
                "checkpoint_interval": "10 fs",
            },
        },
        base_directory=here,
    )
