"""Real two-rank static / three-rank dynamic subspace task regression.

mpiexec -n 2 python tests/mpi_subspace.py --scheduling static
mpiexec -n 3 python tests/mpi_subspace.py --scheduling dynamic
"""
from __future__ import annotations

import argparse
import shutil
import tempfile

from mpi4py import MPI
import numpy as np

from pyrpmd.analysis.subspace_execution import execute_rate_tasks
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate
from pyrpmd.analysis.subspace_rpmd import ring_polymer_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduling", choices=("static", "dynamic"), required=True)
    args = parser.parse_args()
    world = MPI.COMM_WORLD
    expected = 3 if args.scheduling == "dynamic" else 2
    if world.size != expected:
        raise ValueError(f"this smoke test needs {expected} ranks")
    x = np.linspace(-4., 4., 33)
    model = SubspaceModel((x,), .2 / np.cosh(x)**2, [1.], 0., "asymptotic", -3., {})
    rpmd = dict(windows=6, walkers=3, burn_in=2, sweeps=4, trajectories=6,
                md_dt=.05, md_time=.2, bootstrap=4, seed=17)
    request = dict(model=model, beta=3., nbeads=2, rpmd_options=rpmd)
    reference = None
    if world.rank == 0:
        reference = (exact_quantum_rate(model, 3.), ring_polymer_rate(model, 3., 2, **rpmd))
    reference = world.bcast(reference)
    directory = world.bcast(tempfile.mkdtemp(prefix="subspace-mpi-") if world.rank == 0 else None)
    settings = dict(backend="mpi", scheduling=args.scheduling, cache_directory=directory, reuse=True)
    try:
        first = execute_rate_tasks([request], settings)[0]
        second = execute_rate_tasks([request], settings)[0]
        for actual, target in zip(first, reference):
            np.testing.assert_allclose(actual.rate, target.rate, rtol=1e-12)
        np.testing.assert_array_equal(first[1].kappa, reference[1].kappa)
        assert all(r.diagnostics["stage_cache_hit"] for r in second)
        bad = {**request, "rpmd_options": {**rpmd, "windows": 2}}
        try:
            execute_rate_tasks([bad], {**settings, "cache_directory": None})
        except (ValueError, RuntimeError):
            failed = 1
        else:
            failed = 0
        assert world.allreduce(failed) == world.size, "worker failure did not reach every rank"
        if world.rank == 0:
            print(f"subspace MPI {args.scheduling}: parity, reuse and collective failure passed", flush=True)
    finally:
        world.Barrier()
        if world.rank == 0:
            shutil.rmtree(directory)


if __name__ == "__main__":
    main()
