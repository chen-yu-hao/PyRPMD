"""Portable oversubscription flags for tests that spawn mpiexec."""

import shutil

import pytest


def _accepting(*accepted):
    accepted = [list(flags) for flags in accepted]
    tried = []

    def accepts(flags):
        tried.append(list(flags))
        return list(flags) in accepted
    return accepts, tried


@pytest.mark.parametrize("accepted, flags", [
    # Open MPI 5 / PRRTE and MPICH Hydra accept the --map-by modifier.
    ((["--map-by", ":OVERSUBSCRIBE"], ["--oversubscribe"]), ["--map-by", ":OVERSUBSCRIBE"]),
    ((["--map-by", ":OVERSUBSCRIBE"],), ["--map-by", ":OVERSUBSCRIBE"]),
    # Open MPI 4 falls back to --oversubscribe.
    ((["--oversubscribe"],), ["--oversubscribe"]),
    # An unknown launcher that accepts neither gets no flag.
    ((), []),
])
def test_first_accepted_flag_set_is_used(mpi_launch, accepted, flags):
    accepts, tried = _accepting(*accepted)
    assert mpi_launch("/usr/bin/mpiexec", 4, accepts=accepts) == ["/usr/bin/mpiexec", *flags, "-n", "4"]
    assert tried[0] == ["--map-by", ":OVERSUBSCRIBE"]


def test_real_launcher_prefix_starts_more_ranks_than_cores(mpi_launch):
    pytest.importorskip("mpi4py")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPI launcher is unavailable")
    import os
    import subprocess
    import sys
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    ranks = 2 * (os.cpu_count() or 1) + 1
    command = mpi_launch(launcher, ranks, env=env) + [sys.executable, "-c", "pass"]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
