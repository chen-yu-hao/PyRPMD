"""Four-rank PLUMED multi-sim communicator smoke test with an analytic kernel."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from pyrpmd.constants import KB_AU
from pyrpmd.runner import run


class MPIKernel:
    def __init__(self, **options):
        self.values = {}
        self.replica_comm = None
        self.multi_comm = None

    def cmd(self, name, value=None):
        if name == "setMPIComm":
            self.replica_comm = value
            self.values[name] = value
        elif name == "setMPImultiSimComm":
            self.multi_comm = value
            self.values[name] = value
        elif name == "getApiVersion":
            value[0] = 10
        elif name == "isEnergyNeeded":
            value[0] = 0
        elif name == "performCalcNoUpdate":
            positions = self.values["setPositions"]
            self.values["setForces"][:] = -0.2 * positions
            self.bias = 0.1 * float(np.sum(positions * positions))
        elif name == "getBias":
            value[0] = self.bias
        elif name == "readInputLines":
            self.values[name] = value
            Path("CONFIGURED").write_text(value)
        elif name == "update":
            assert self.replica_comm.Get_size() == 1
            step = int(self.values["setStep"])
            clocks = self.multi_comm.allgather(step)
            assert clocks == [step] * self.multi_comm.Get_size()
            with Path("STATE").open("a") as stream:
                stream.write(f"{step}\n")
        elif name.startswith("set"):
            self.values[name] = value

    def finalize(self):
        pass


def main(output):
    sys.modules["plumed"] = SimpleNamespace(Plumed=MPIKernel)
    job = {
        "task": "plumed",
        "temperature": 1 / (3.1 * KB_AU),
        "beads": 1,
        "native": False,
        "seed": 31,
        "system": {
            "symbols": ["H"],
            "masses": [1.0],
            "geometry": [[0.1, 0.0, 0.0]],
        },
        "potential": {"provider": "harmonic", "params": {"k": 0.1}},
        "dynamics": {"timestep": 0.02},
        "sampling": {
            "replicas": 4,
            "equilibration_steps": 0,
            "production_steps": 2,
            "sample_interval": 1,
        },
        "plumed": {
            "input": (
                "x: POSITION ATOM=1\n"
                "m: METAD ARG=x.x SIGMA=0.1 HEIGHT=0.01 PACE=1 FILE=HILLS\n"
            ),
            "mode": "centroid",
            "walkers": "mpi",
            "restart": "files",
        },
        "parallel": {
            "backend": "mpi",
            "mode": "serial",
            "trajectory_workers": 4,
            "trajectory_groups": 4,
            "bead_workers": 1,
            "pes_ranks": 1,
            "coordinator_ranks": 0,
            "scheduling": "static",
        },
        "output": {"checkpoint_steps": 1},
    }
    result = run(job, output=output)
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert result["status"] == "complete"
        assert result["replicas"] == 4
        for replica in range(4):
            work = Path(output) / "plumed" / f"replica-{replica:04d}"
            assert "WALKERS_MPI" in (work / "CONFIGURED").read_text()
            assert (work / "STATE").read_text().splitlines() == ["1", "2"]
        summary = json.loads(
            (Path(output) / "results" / "summary.json").read_text()
        )
        assert summary["walkers"] == "mpi"


if __name__ == "__main__":
    main(sys.argv[1])
