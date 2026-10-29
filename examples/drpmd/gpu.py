"""Two GPUs, two concurrent trajectory processes per GPU (four model copies)."""

import os
from pathlib import Path
import tomllib

DEVICES = [0, 1]  # Indices after CUDA_VISIBLE_DEVICES remapping.
TRAJECTORIES_PER_GPU = 2  # Set 1 for one trajectory process/model per GPU.


def build():
    directory = Path(__file__).parent
    config = tomllib.loads((directory / "free_scattering.inp").read_text())
    backend = os.environ.get("PYRPMD_BACKEND", "process")
    workers = len(DEVICES) * TRAJECTORIES_PER_GPU
    groups = int(os.environ.get("PYRPMD_TRAJECTORY_GROUPS", str(workers)))
    config["sampling"]["parents"] = groups if backend == "mpi" else workers
    config["collision"]["trajectories"] = max(8, 2 * config["sampling"]["parents"])
    config["potential"] = {
        "provider": "torch",
        "device": "cuda:0",
        "factory": str(directory / "torch_pes.py") + ":build_model",
        "params": {"height_hartree": 0.002, "width_bohr": 1.5},
    }
    config["parallel"] = {
        "backend": backend,
        "mode": "gpu",
        "devices": DEVICES,
        "trajectories_per_gpu": TRAJECTORIES_PER_GPU,
        "bead_workers": 1,
        "threads_per_rank": 1,
    }
    if backend == "mpi":
        config["parallel"]["trajectory_groups"] = groups
    config["output"]["directory"] = "runs/gpu"
    return config
