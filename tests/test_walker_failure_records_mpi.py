"""Real MPI recovery after one peer fails; optional MPI runtime required."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("method", ["opes", "metadynamics"])
def test_one_failed_mpi_peer_records_every_rolled_back_walker_and_resumes(
        tmp_path, method, mpi_launch):
    pytest.importorskip("mpi4py.MPI")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPI launcher is not installed")
    if any(name in os.environ for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MPI_LOCALNRANKID")):
        pytest.skip("do not nest an MPI launcher")
    helper = tmp_path / "worker.py"
    helper.write_text('''
import importlib
from pathlib import Path
import sys
from mpi4py import MPI
sys.path.insert(0, sys.argv[3])
from test_walker_failure_records import _config, _system
from pyrpmd.dynamics import Dynamics
from pyrpmd.workflows.biased_sampling import history_failures
method, output = sys.argv[1], Path(sys.argv[2])
module = "_distributed_opes_mpi" if method == "opes" else "_distributed_meta_mpi"
workflow = importlib.import_module("pyrpmd.workflows." + module)
comm = MPI.COMM_WORLD
original = Dynamics.step

def fail(self, state, *args, **kwargs):
    result = original(self, state, *args, **kwargs)
    if comm.rank == 1 and state.step == 3:
        raise RuntimeError("one peer failed")
    return result

Dynamics.step = fail
config = _config(method, "mpi-sync")
try:
    workflow._run_mpi(config, _system(), output)
except RuntimeError as error:
    assert "one peer failed" in str(error)
else:
    raise AssertionError("peer failure did not propagate")
comm.Barrier()
if comm.rank == 0:
    records = history_failures(output)
    assert len(records) == 2
    assert {record["history"]["walker"] for record in records} == {0, 1}
    assert all(record["state"]["elapsed_steps"] == 2 for record in records)
Dynamics.step = original
workflow._run_mpi(config, _system(), output)
comm.Barrier()
if comm.rank == 0:
    assert history_failures(output) == []
''')
    environment = os.environ.copy()
    environment.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        mpi_launch(launcher, 2, env=environment) + [sys.executable, str(helper), method,
            str(tmp_path / "run"), str(Path(__file__).parent)],
        capture_output=True, text=True, timeout=60, env=environment)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
