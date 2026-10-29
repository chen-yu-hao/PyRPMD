"""Two simultaneous trajectories, each with two persistent PES processes."""

from pathlib import Path
import tomllib


def build():
    config = tomllib.loads(Path(__file__).with_name("free_scattering.inp").read_text())
    config["parallel"] = {
        "backend": "process",
        "mode": "bead",
        "trajectory_workers": 2,
        "bead_workers": 2,
        "threads_per_rank": 1,
    }
    config["output"]["directory"] = "runs/bead"
    return config
