"""Private force-engine communicator routing and actual MPI regression."""

import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

from pyrpmd.parallel.execution import _make_mpi_potential, execution_plan


def test_engine_rank_topology_is_explicit_and_preserves_defaults():
    assert execution_plan().pes_ranks == 1
    plan = execution_plan({"backend": "mpi", "trajectory_groups": 2,
                           "bead_workers": 3, "pes_ranks": 4})
    assert (plan.trajectory_groups, plan.bead_workers, plan.pes_ranks) == (2, 3, 4)
    assert plan.mode == "bead"
    dynamic = execution_plan({"backend": "mpi", "scheduling": "dynamic",
                              "bead_workers": 2, "pes_ranks": 3})
    assert dynamic.coordinator_ranks == 1


@pytest.mark.parametrize("config", [
    {"pes_ranks": 2},
    {"backend": "process", "pes_ranks": 2},
    {"backend": "mpi", "mode": "gpu", "pes_ranks": 2},
    {"backend": "mpi", "pes_ranks": True},
    {"backend": "mpi", "pes_ranks": 0},
    {"backend": "mpi", "pes_ranks": 1.5},
])
def test_unsupported_engine_rank_topologies_fail_before_launch(config):
    with pytest.raises(ValueError, match="pes_ranks"):
        execution_plan(config)


def test_live_communicator_does_not_mutate_serializable_potential_spec():
    spec = {"provider": "lammps", "model": "model.in",
            "params": {"type_map": ["Ar"], "batch_mode": "persistent"}}
    original = copy.deepcopy(spec)
    communicator = object()
    with patch("pyrpmd.potentials.make_potential", return_value=object()) as create:
        _make_mpi_potential(spec, None, communicator)
        assert create.call_args.args[0]["params"]["communicator"] is communicator
    assert spec == original


@pytest.mark.parametrize("spec", [
    {"provider": "harmonic"},
    {"provider": "openmm"},
    {"provider": "lammps", "params": {"communicator": "unowned"}},
])
def test_engine_group_rejects_provider_or_communicator_mismatch(spec):
    with patch("pyrpmd.potentials.make_potential") as create:
        with pytest.raises(ValueError, match="LAMMPS|communicator"):
            _make_mpi_potential(spec, None, object())
        create.assert_not_called()


@pytest.mark.parametrize("scheduling,ranks", [("static", 4), ("dynamic", 5)])
def test_actual_lammps_private_engine_groups(scheduling, ranks):
    # Do not initialize MPI in pytest's process. Ordinary installations without
    # these optional engines still run all routing/validation unit tests above.
    pytest.importorskip("lammps")
    pytest.importorskip("mpi4py")
    launcher = Path(sys.executable).with_name("mpiexec")
    if not launcher.is_file():
        executable = shutil.which("mpiexec")
        if executable is None:
            pytest.skip("optional MPI launcher unavailable")
        launcher = Path(executable)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # Open MPI requires explicit root opt-in in containerized test environments.
    # MPICH ignores these variables; no engine or numerical option is changed.
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    script = Path(__file__).with_name("mpi_engine_subcommunicator_smoke.py")
    run = subprocess.run(
        [str(launcher), "-n", str(ranks), sys.executable, str(script),
         "--scheduling", scheduling],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=180, check=False,
    )
    assert run.returncode == 0, run.stdout
    assert "private engine groups passed" in run.stdout
