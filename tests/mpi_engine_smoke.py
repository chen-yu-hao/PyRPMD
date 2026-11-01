"""Require independent one-rank LAMMPS contexts inside a PyRPMD MPI launch.

Run with ``mpiexec -n 2 python tests/mpi_engine_smoke.py`` after installing
LAMMPS and a matching mpi4py/MPI runtime.
"""
from pathlib import Path

from mpi4py import MPI
import numpy as np

from pyrpmd.input import JobSpec
from pyrpmd.potentials import make_potential


def main():
    comm = MPI.COMM_WORLD
    root = Path(__file__).resolve().parents[1]
    config, system = JobSpec.from_file(root / "examples/potentials/lammps_lj.inp").resolve()
    with make_potential(config["potential"], system) as provider:
        assert provider.engine.extract_setting("world_size") == 1
        q = np.repeat(system.geometry[None], 2, axis=0)
        q[1, 0, 1] += .01 * (comm.Get_rank()+1)
        out = provider.evaluate(q)
        assert np.isfinite(out.energy).all() and np.isfinite(out.forces).all()
        values = comm.allgather(float(out.energy[0]))
        assert all(value == values[0] for value in values)
    # Closing one force provider must not finalize the host MPI communicator.
    assert comm.allreduce(1) == comm.Get_size()
    if comm.Get_rank() == 0:
        print(f"LAMMPS COMM_SELF isolation and post-close MPI passed on {comm.Get_size()} ranks")


if __name__ == "__main__":
    main()
