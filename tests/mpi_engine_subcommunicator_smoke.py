"""Actual G x B x R LAMMPS evaluation, independent LJ oracle and failures.

Run with four MPI ranks (static) or five (``--scheduling dynamic``). This tests
communication and spatial decomposition on one host, not multi-node scaling.
"""

import argparse
import copy
from pathlib import Path

from mpi4py import MPI
import numpy as np

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU
from pyrpmd.input import JobSpec
from pyrpmd.parallel import make_task_potential, root_call, run_tasks, task_potential
from pyrpmd.potentials import PotentialError
from pyrpmd.potentials.lammps import LAMMPSPotential


ROOT = Path(__file__).resolve().parents[1]
CONFIG, SYSTEM = JobSpec.from_file(ROOT / "examples/potentials/lammps_lj.inp").resolve()
SPEC = copy.deepcopy(CONFIG["potential"])
SPEC["params"]["batch_mode"] = "persistent"


def _coordinates(value, frames=3):
    x = np.repeat(SYSTEM.geometry[None], frames, axis=0)
    x[:, 1, 0] += (np.arange(frames) * .02 + value * .01) / BOHR_ANGSTROM
    return x


def _check_lj(result, q):
    delta = (q[:, 1] - q[:, 0]) * BOHR_ANGSTROM
    distance = np.linalg.norm(delta, axis=1)
    s6 = (3.4 / distance)**6
    energy = 4*.0103*(s6*s6-s6)*EV_TO_AU
    first = 24*.0103*(s6-2*s6*s6)[:, None]*delta/distance[:, None]**2
    forces = np.stack([first, -first], axis=1)*EV_TO_AU*BOHR_ANGSTROM
    virial = -np.einsum("bi,bj->bij", delta, first)*EV_TO_AU
    np.testing.assert_allclose(result.energy, energy, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(result.forces, forces, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(result.virial, virial, rtol=1e-10, atol=1e-15)


def evaluate_task(value):
    with task_potential(SPEC, SYSTEM) as provider:
        assert provider.local.engine.extract_setting("world_size") == provider.metadata["pes_ranks"]
        # Identical membership is acceptable for a whole-trajectory engine:
        # its context must still be private, never MPI.COMM_WORLD itself.
        assert MPI.Comm.Compare(provider.engine_comm, MPI.COMM_WORLD) != MPI.IDENT
        q = _coordinates(value)
        for offset in (0, .001, 8.0, 9.0):
            shifted = q + np.array([offset, 0, 0])/BOHR_ANGSTROM
            result = provider.evaluate(shifted)
            _check_lj(result, shifted)
        # Fewer frames than bead groups must not enter a partial collective.
        _check_lj(provider.evaluate(q[:1]), q[:1])
        assert "communicator" not in SPEC["params"]
        return provider.identifier, result.energy.tolist()


def bad_evaluation(_):
    with task_potential(SPEC, SYSTEM) as provider:
        q = _coordinates(0)
        q[-1, 0, 0] = np.nan  # Only one bead group sees this invalid frame.
        provider.evaluate(q)


def reject_one_group_factory(*, comm, fail_world_rank):
    if comm.allreduce(int(MPI.COMM_WORLD.Get_rank() == fail_world_rank)):
        raise ValueError("injected private group initialization failure")
    from lammps import lammps
    engine = lammps(comm=comm, cmdargs=["-log", "none", "-screen", "none"])
    engine.file(str(ROOT / "examples/potentials/lammps_lj.in"))
    return engine


def bad_initialization(_):
    spec = copy.deepcopy(SPEC)
    spec.pop("model")
    spec["factory"] = str(Path(__file__).resolve()) + ":reject_one_group_factory"
    spec["params"]["factory_kwargs"] = {"fail_world_rank": 2}
    make_task_potential(spec, SYSTEM)


def unsupported_provider(_):
    make_task_potential({"provider": "harmonic"}, SYSTEM)


def _failure(function, plan, message):
    try:
        run_tasks(function, [0], plan)
    except RuntimeError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("private engine group failure was swallowed")
    assert MPI.COMM_WORLD.allreduce(1) == MPI.COMM_WORLD.Get_size()


def _direct_cell_contract():
    world = MPI.COMM_WORLD
    kwargs = dict(symbols=SYSTEM.symbols, model=SPEC["model"],
                  type_map=["Ar"], cell=SYSTEM.cell, pbc=SYSTEM.pbc,
                  batch_mode="persistent")
    try:
        LAMMPSPotential(communicator=world, **kwargs)
    except ValueError as exc:
        assert "COMM_WORLD" in str(exc)
    else:
        raise AssertionError("direct adapter accepted the shared world communicator")
    private = world.Split(world.Get_rank() // 2, world.Get_rank())
    try:
        with LAMMPSPotential(communicator=private, **kwargs) as provider:
            q = _coordinates(0)
            _check_lj(provider.evaluate(q), q)
            if private.Get_size() > 1:
                def rejected_everywhere(action):
                    caught = False
                    try:
                        action()
                    except PotentialError:
                        caught = True
                    assert private.allreduce(int(caught)) == private.Get_size()
                    assert not provider.closed
                    np.testing.assert_array_equal(provider.cell, SYSTEM.cell)

                invalid_cell = SYSTEM.cell.copy()
                if private.Get_rank() == 0:
                    invalid_cell[0, 0] = -1.0
                rejected_everywhere(lambda: provider.set_cell(invalid_cell))
                different_cell = SYSTEM.cell * (1.05 if private.Get_rank() == 0 else 1.06)
                rejected_everywhere(lambda: provider.set_cell(different_cell))
                binding = {"symbols": SYSTEM.symbols, "cell": SYSTEM.cell,
                           "pbc": SYSTEM.pbc}
                if private.Get_rank() == 0:
                    binding["symbols"] = ("Xe", "Ar")
                rejected_everywhere(lambda: provider.bind_system(binding))
                different_q = q.copy()
                if private.Get_rank() == 0:
                    different_q[0, 0, 0] += .001
                rejected_everywhere(lambda: provider.evaluate(different_q))
                invalid_q = q.copy()
                if private.Get_rank() == 0:
                    invalid_q[0, 0, 0] = np.nan
                rejected_everywhere(lambda: provider.evaluate(invalid_q))
                _check_lj(provider.evaluate(q), q)
            provider.set_cell(SYSTEM.cell * 1.1)
            _check_lj(provider.evaluate(q * 1.1), q * 1.1)
            _check_lj(provider.evaluate(q * 1.1 + .0001), q * 1.1 + .0001)
        # The provider owns a duplicate, not its caller's communicator.
        assert private.allreduce(1) == private.Get_size()
    finally:
        private.Free()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduling", choices=("static", "dynamic"), default="static")
    options = parser.parse_args()
    plan = {"backend": "mpi", "trajectory_groups": 1, "bead_workers": 2,
            "pes_ranks": 2, "scheduling": options.scheduling}
    assert MPI.COMM_WORLD.Get_size() == 4 + int(options.scheduling == "dynamic")
    _direct_cell_contract()
    results = run_tasks(evaluate_task, [0, 1, 2], plan)

    def check():
        assert [identifier for identifier, _ in results] == [0, 0, 0]
        assert len(results[0][1]) == 3

    root_call(check)
    _failure(bad_evaluation, plan, "NaN/Inf")
    _failure(bad_initialization, plan, "injected private group initialization failure")
    _failure(unsupported_provider, plan, "requires the LAMMPS provider")
    run_tasks(evaluate_task, [1, 2], plan)
    # Vary both trajectory-group and engine-group widths at fixed total ranks.
    for groups, beads, ranks in ((2, 1, 2), (1, 1, 4)):
        changed = dict(plan, trajectory_groups=groups, bead_workers=beads, pes_ranks=ranks)
        run_tasks(evaluate_task, [0, 1, 2], changed)
    assert MPI.COMM_WORLD.allreduce(1) == MPI.COMM_WORLD.Get_size()
    root_call(lambda: print(
        f"LAMMPS {options.scheduling} private engine groups passed: GxBxR topology, "
        "LJ energy/force/virial oracle, migration, empty chunks, reuse, collective "
        "failures, direct set_cell, COMM_WORLD rejection and post-close MPI", flush=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # An unexpected rank-local assertion must not strand peers in their
        # next collective or leave test workers alive after the launcher exits.
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
