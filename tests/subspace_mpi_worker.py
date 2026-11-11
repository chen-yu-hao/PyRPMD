"""MPI integration fixture, launched by test_subspace_execution.py."""

import json
from pathlib import Path
import sys

from mpi4py import MPI

from test_subspace_execution import cases, pipeline_fixture
from pyrpmd.analysis import subspace_corrections


def main():
    comm = MPI.COMM_WORLD
    directory, mode = Path(sys.argv[1]), sys.argv[2]
    if mode == "pipeline":
        import pytest
        from pyrpmd.analysis import run_subspace_pipeline

        with pytest.MonkeyPatch.context() as patch:
            spec = pipeline_fixture(patch, directory)
            spec["execution"] = dict(backend="mpi", cpus=2)
            result = run_subspace_pipeline(spec)
        if comm.rank == 0:
            (directory / "mpi.json").write_text(json.dumps({"ranks": comm.size,
                "dimensions": result["dimensions_computed"],
                "root_wrote_report": (directory / "subspace_pipeline.json").is_file()}))
        return
    inputs = cases() if comm.rank == 0 else None
    if comm.rank == 0 and mode == "failure":
        inputs["one"]["rpmd_options"]["backend"] = "invalid"
    if comm.rank == 0 and mode == "preparation_failure":
        inputs["one"]["model"] = None
    options = dict(backend="mpi", cpus=2, checkpoint_dir=str(directory / "tasks"))
    failed = False
    try:
        first = subspace_corrections(inputs, execution=options)["one"]
    except RuntimeError:
        failed = True
    if mode != "normal":
        report = {"ranks": comm.size, "collective_failure": all(comm.allgather(failed))}
    else:
        resumed = subspace_corrections(inputs, execution=options)["one"]
        report = {"ranks": comm.size, "factor": first.factor,
                  "pids": [r.diagnostics["execution"]["pid"] for r in (first.quantum, first.rpmd)],
                  "all_reused": all(r.diagnostics["checkpoint_reused"] for r in (resumed.quantum, resumed.rpmd))}
    if comm.rank == 0:
        (directory / "mpi.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
