"""Reproducible H + H2/BKMP2 hybrid-parallel benchmark (not rate convergence).

Build BKMP2 with benchmarks/h3/build_pes.py, then run e.g.:
  OPENBLAS_NUM_THREADS=1 python benchmarks/h3_subspace/parallel.py --mode native --cpus 32 --output native32.json
  mpiexec -n 4 python benchmarks/h3_subspace/parallel.py --mode mpi --cpus 32 --memory-limit-mb 16384 --output mpi32.json

The fixed 300 K/P32 budget exercises the same gridded 1D adiabatic path and
2D collinear Hamiltonians in every mode. ``python`` uses the reference time
step loops; ``native`` runs stages sequentially; ``tasks`` overlaps all four
QM/RPMD stages; ``mpi`` distributes those stages with a per-node CPU budget.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Python sampling/analysis alternates with short OpenMP regions. Sleeping
# workers avoid competing with the coordinator between HMC proposals.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np
import scipy
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collinear import Collinear  # noqa: E402
from run import BARRIER, KB, h2_ground_energy, absorber  # noqa: E402

from pyrpmd.analysis import subspace_corrections  # noqa: E402
from pyrpmd.analysis.subspace_execution import _implementation_digest  # noqa: E402
from pyrpmd.analysis.subspace_model import SubspaceModel, _arc_length_spline, project_path_subspace  # noqa: E402
from pyrpmd.analysis.subspace_quantum import _json_safe  # noqa: E402


def make_cases(backend):
    beta = 1. / (KB * 300.)
    potential = Collinear()
    s, t = np.linspace(-110., 110., 221), np.linspace(70., 290., 221)
    grid = np.stack(np.meshgrid(s, t, indexing="ij"), axis=-1)
    values = potential(grid)
    fine2 = SubspaceModel((s, t), values, [1., 1.], 0., reactant_reference=-63.,
                          provenance={"system": "collinear H + H2", "pes": "BKMP2", "grid_spacing": 1.})
    coarse2 = replace(fine2, axes=(s[::2], t[::2]), values=values[::2, ::2],
                      provenance={**fine2.provenance, "grid_spacing": 2.})
    path = []
    for row in range(0, len(s), 4):
        center = t[int(np.argmin(values[row]))]
        result = minimize_scalar(lambda y: float(potential([s[row], y])),
                                 bounds=(max(t[0], center - 5.), min(t[-1], center + 5.)),
                                 method="bounded", options={"xatol": 1e-8})
        if not result.success:
            raise RuntimeError("BKMP2 path minimization failed")
        path.append([s[row], result.x])
    path = np.asarray(path)
    length = float(_arc_length_spline(path)[2][-1])
    fine1 = project_path_subspace(potential, potential.gradient, path, [1., 1.], np.linspace(0., length, 321),
                                  rigid_body=False, reactant_reference=.2 * length)
    coarse1 = replace(fine1, axes=(fine1.axes[0][::2],), values=fine1.values[::2])
    e0 = h2_ground_energy()
    eps = absorber(30. * np.sqrt(2. / (np.pi * beta)) / 35.)(coarse2.grid_points())
    rpmd = dict(windows=24, walkers=32, burn_in=40, sweeps=80, leapfrog_steps=8,
                trajectories=256, bootstrap=100, seed=20261110, backend=backend)
    return {
        "1d": dict(model=coarse1, rpmd_model=fine1, beta=beta, nbeads=32, rpmd_options=rpmd),
        "2d": dict(model=coarse2, rpmd_model=fine2, beta=beta, nbeads=32, rpmd_options=rpmd,
                   quantum_options=dict(method="sylvester", dense_limit=8000, absorber=eps,
                                        normalization="slice", energy_floor=e0 - 1. / beta,
                                        truncation_energy=e0 + max(.04, BARRIER + 25. / beta))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("python", "native", "tasks", "mpi"), required=True)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--blas-threads", type=int, default=8)
    parser.add_argument("--memory-limit-mb", type=float, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    comm = None
    if args.mode == "mpi":
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    root = comm is None or comm.rank == 0
    cases = None
    error = None
    if root:
        try:
            cases = make_cases("python" if args.mode == "python" else "native")
        except Exception as exc:
            error = str(exc)
    if comm is not None:
        error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    options = dict(backend={"python": "serial", "native": "serial", "tasks": "process", "mpi": "mpi"}[args.mode],
                   cpus=args.cpus, workers=4, blas_threads=args.blas_threads, memory_limit_mb=args.memory_limit_mb)
    if args.checkpoint_dir is not None:
        options["checkpoint_dir"] = str(args.checkpoint_dir)
    implementation = _implementation_digest().hex()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    started = time.perf_counter()
    results = subspace_corrections(cases, execution=options)
    elapsed = time.perf_counter() - started
    if root:
        from pyrpmd import _subspace

        record = dict(schema="pyrpmd.subspace.parallel_benchmark.v1", mode=args.mode, execution=options,
                      wall_seconds=elapsed, physical_converged=False, temperature_K=300., nbeads=32,
                      rpmd_options=cases["1d"]["rpmd_options"],
                      platform=platform.platform(), python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                      affinity=sorted(os.sched_getaffinity(0)), native_extension=_subspace.__file__,
                      openmp=bool(_subspace.openmp), mpi_ranks=1 if comm is None else comm.size,
                      omp_wait_policy=os.environ.get("OMP_WAIT_POLICY"),
                      git_revision=revision, implementation_sha256=implementation,
                      corrections={key: result.to_dict() for key, result in results.items()})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_safe(record), indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(args.output), "wall_seconds": elapsed,
                          "factors": {k: v.factor for k, v in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
