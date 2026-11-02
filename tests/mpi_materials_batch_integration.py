"""Collective materials-batch restart regression with an analytic potential.

Run both layouts explicitly (the output directories must be fresh):

    mpiexec -n 2 python tests/mpi_materials_batch_integration.py OUTPUT/static
    mpiexec -n 3 python tests/mpi_materials_batch_integration.py OUTPUT/dynamic --scheduling dynamic

Two trajectory groups each use one rank. Dynamic scheduling adds one coordinator.
This checks checkpoint ownership and MPI collectives, not material accuracy or
cross-node scaling. It deliberately corrupts one disposable test checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path
import traceback

import numpy as np
from mpi4py import MPI

from pyrpmd.dynamics import Dynamics
from pyrpmd.parallel import root_call
from pyrpmd.system import System
from pyrpmd.workflows.materials import _read_stream, _sample_replica, run_materials


def model():
    """Small, deterministic frame-local PES with independent replica RNGs."""
    atoms = System(
        ("H", "Ni"), np.array([2., 20.]),
        np.array([[.1, .2, .3], [1.1, .2, .3]]),
        cell=np.array([[6., 0., 0.], [.5, 7., 0.], [0., 0., 12.]]),
        pbc=(True, True, False),
    )
    config = {
        "task": "pimd", "beta": 2., "beads": 4, "dt_au": .015,
        "seed": 1739, "native": False,
        "potential": {"provider": "builtin", "name": "periodic_cosine",
                      "params": {"height": [.2, .1, 0.], "confinement": .05}},
        "sampling": {"equilibration_steps": 3, "production_steps": 18,
                     "sample_interval": 2, "replicas": 5,
                     "gamma_centroid": .6, "thermostat": "pile-l"},
        "dynamics": {"loop_backend": "python"},
        "parallel": {"trajectory_workers": 1, "trajectory_batch_size": 1},
        "output": {"checkpoint_steps": 4, "chunk_frames": 3,
                   "trajectory_storage": "stream"},
    }
    return config, atoms


def assert_same_stream(expected, actual):
    for key in ("positions", "velocities", "observables", "radii_gyration_squared",
                "sample_steps", "times"):
        np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)
    for key in ("q", "p"):
        np.testing.assert_array_equal(getattr(expected["state"], key),
                                      getattr(actual["state"], key), err_msg=key)
    assert expected["replica"] == actual["replica"]
    assert expected["state"].step == actual["state"].step
    assert expected["state"].time == actual["state"].time
    assert expected["rng"] == actual["rng"]
    assert expected["propagation"] == actual["propagation"]


def checkpoint_hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((directory / "checkpoints").glob("materials_*.npz"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scheduling", choices=("static", "dynamic"), default="static")
    args = parser.parse_args()
    dynamic = args.scheduling == "dynamic"
    expected_ranks = 3 if dynamic else 2
    if MPI.COMM_WORLD.size != expected_ranks:
        raise ValueError(f"{args.scheduling} requires exactly {expected_ranks} MPI ranks")

    output = args.output.resolve()
    config, atoms = model()

    def sample_replica(directory, replica):
        return _sample_replica(dict(config=config, system=atoms,
                                    output=str(directory), replica=replica))

    def prepare():
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"test output must be a fresh directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
        # Only root creates scalar reference and checkpoint fixtures. Calling
        # the single-replica worker directly avoids starting an MPI executor.
        for replica in range(5):
            sample_replica(output / "reference", replica)
        for replica in (0, 2):
            sample_replica(output / "resumed", replica)
        original = Dynamics.sample_step

        def interrupted(dynamics, state, *args, **kwargs):
            result = original(dynamics, state, *args, **kwargs)
            if state.step == 7:
                raise RuntimeError("deliberate partial checkpoint")
            return result

        Dynamics.sample_step = interrupted
        try:
            sample_replica(output / "resumed", 1)
        except RuntimeError as error:
            assert str(error) == "deliberate partial checkpoint", str(error)
        else:
            raise AssertionError("partial checkpoint injection did not execute")
        finally:
            Dynamics.sample_step = original
        prefixes = {replica: _read_stream(config, atoms, output / "resumed", replica,
                                         materialize=False)["elapsed"]
                    for replica in (0, 1, 2)}
        assert prefixes == {0: 21, 1: 6, 2: 21}, prefixes
        hashes = checkpoint_hashes(output / "resumed")
        assert len(hashes) == 3, hashes
        completed = {name: value for name, value in hashes.items()
                     if name in ("materials_0000.npz", "materials_0002.npz")}
        return prefixes, completed

    prefixes, completed = root_call(prepare)
    config["parallel"].update(
        backend="mpi", trajectory_groups=2, trajectory_workers=2,
        bead_workers=1, trajectory_batch_size=2, scheduling=args.scheduling,
        coordinator_ranks=int(dynamic),
    )
    # Pending IDs [1, 3, 4] must be identical on all ranks after root filters
    # the completed IDs [0, 2]. The single-lane tail is a genuine second task.
    result = run_materials(config, atoms, output / "resumed", return_data=True)

    def compare():
        assert [stream["replica"] for stream in result["streams"]] == list(range(5))
        for replica, actual in enumerate(result["streams"]):
            expected = _read_stream(config, atoms, output / "reference", replica)
            assert_same_stream(expected, actual)
        hashes = checkpoint_hashes(output / "resumed")
        assert len(hashes) == 5, hashes
        assert all(hashes[name] == value for name, value in completed.items())
        return hashes

    committed = root_call(compare)
    import pyrpmd.parallel as parallel
    import pyrpmd.parallel.execution as execution
    import pyrpmd.workflows.materials as materials

    def unavailable(*args, **kwargs):
        raise AssertionError("completed MPI resume opened a potential")

    # Patch every worker factory on every rank: zero pending tasks must still
    # complete MPI collectives but must never construct an analytic provider.
    parallel.make_task_potential = execution.make_task_potential = unavailable
    materials.make_task_potential = unavailable
    config["parallel"]["trajectory_batch_size"] = 3
    again = run_materials(config, atoms, output / "resumed", return_data=True)

    def check_terminal():
        assert committed == checkpoint_hashes(output / "resumed")
        assert len(again["streams"]) == 5
        for previous, actual in zip(result["streams"], again["streams"]):
            assert_same_stream(previous, actual)

    root_call(check_terminal)

    # A root-side preflight error must be broadcast before any rank enters
    # a task collective. The launcher/CI timeout also detects a stuck peer.
    root_call(lambda: (output / "resumed/checkpoints/materials_0001.npz").write_bytes(
        b"deliberately corrupted disposable test checkpoint"))
    try:
        run_materials(config, atoms, output / "resumed")
    except RuntimeError as error:
        message = str(error)
    else:
        message = ""
    messages = MPI.COMM_WORLD.allgather(message)
    assert all(messages) and len(set(messages)) == 1, messages
    assert "Root stage failed:" in messages[0], messages
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps({
            "scheduling": args.scheduling, "ranks": MPI.COMM_WORLD.size,
            "prefix_elapsed": prefixes, "mixed_resume_exact": True,
            "completed_checkpoint_immutable": True,
            "complete_resume_no_provider": True,
            "corrupt_checkpoint_collective_error": True,
            "scope": "analytic MPI and checkpoint regression",
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
