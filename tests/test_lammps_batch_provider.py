"""Independent force references and lifecycle contracts for persistent LAMMPS."""

from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU, KCAL_MOL_TO_AU
from pyrpmd.potentials import PotentialError, check_potential
from pyrpmd.potentials.lammps import LAMMPSPotential


def _model(tmp_path, *, units="metal", periodic=True, charged=False, triclinic=False):
    pytest.importorskip("lammps")
    path = tmp_path / "persistent.in"
    path.write_text(f"""units {units}
atom_style {'charge' if charged else 'atomic'}
boundary {'p p p' if periodic else 'f f f'}
region box {'prism 0 20 0 20 0 20 1 0.5 0.2' if triclinic else 'block 0 20 0 20 0 20'}
create_box 1 box
create_atoms 1 single 2 2 2
create_atoms 1 single 3.3 2 2
mass 1 1
{'set atom 1 charge 1' if charged else ''}
{'set atom 2 charge -1' if charged else ''}
pair_style {'lj/cut/coul/long' if charged else 'lj/cut'} 5
pair_coeff 1 1 0.2 1
{'kspace_style ewald 1e-8' if charged else ''}
neighbor 0.4 bin
""")
    cell = np.array([[20., 0, 0], [1., 20, 0], [.5, .2, 20]]) if triclinic else np.eye(3)*20
    return path, cell / BOHR_ANGSTROM, [periodic]*3


def _provider(path, cell, pbc, **kwargs):
    return LAMMPSPotential(["H", "H"], path, type_map=["H"], cell=cell,
                           pbc=pbc, **kwargs)


def _frames(count=3):
    q = np.repeat(np.array([[[2., 2, 2], [3.3, 2, 2]]]), count, axis=0)
    q[:, 1, 0] += .07 * np.arange(count)
    return q / BOHR_ANGSTROM


@pytest.mark.parametrize("units,factor", [("metal", EV_TO_AU), ("real", KCAL_MOL_TO_AU)])
def test_persistent_batches_refresh_forces_against_analytic_lj(tmp_path, units, factor):
    args = _model(tmp_path, units=units)
    with _provider(*args, batch_mode="persistent", neighbor_skin=.4/BOHR_ANGSTROM) as pes:
        q = _frames()
        first = pes.evaluate(q)
        engines = pes.engines
        q[:, 1, 0] += .03/BOHR_ANGSTROM
        before = q.copy()
        result = pes.evaluate(q)
        r = (q[:, 1, 0]-q[:, 0, 0])*BOHR_ANGSTROM
        s6 = r**-6
        energy = 4*.2*(s6*s6-s6)*factor
        force = 24*.2/r*(s6-2*s6*s6)*factor*BOHR_ANGSTROM
        np.testing.assert_allclose(result.energy, energy, rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(result.forces[:, 0, 0], force, rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(result.forces[:, 1, 0], -force, rtol=1e-13, atol=1e-16)
        np.testing.assert_allclose(result.virial[:, 0, 0], -r*force/BOHR_ANGSTROM, rtol=1e-13, atol=1e-16)
        assert not np.array_equal(first.energy, result.energy)  # run 0 pre no would be stale.
        np.testing.assert_array_equal(q, before)
        assert pes.engines == engines
        assert pes.batch_stats == {"instances": 3, "neighbor_rebuilds": 3, "neighbor_reuses": 3}
        for engine, frame in zip(engines, q):
            ids = engine.numpy.extract_atom("id")[:2].astype(int)
            np.testing.assert_allclose(engine.numpy.extract_atom("x")[:2], frame[ids-1]*BOHR_ANGSTROM, atol=1e-15)
            np.testing.assert_array_equal(engine.numpy.extract_atom("v")[:2], 0)
            assert engine.available_ids("fix") == []
        assert check_potential(pes, q)["max_absolute_force_error"] < 2e-11


def test_skin_displacement_is_accumulated_from_last_rebuild(tmp_path):
    args = _model(tmp_path)
    with _provider(*args, batch_mode="persistent", neighbor_skin=.4/BOHR_ANGSTROM) as pes:
        q = _frames(1)
        pes.evaluate(q)
        for _ in range(3):
            q[:, :, 1] += .06/BOHR_ANGSTROM
            pes.evaluate(q)
        assert pes.batch_stats["neighbor_rebuilds"] == 1
        assert pes.batch_stats["neighbor_reuses"] == 3
        q[:, :, 1] += .03/BOHR_ANGSTROM  # 0.21 A since the original list.
        pes.evaluate(q)
        assert pes.batch_stats["neighbor_rebuilds"] == 2


@pytest.mark.parametrize("triclinic", [False, True])
def test_persistent_reuse_refreshes_periodic_ghosts_and_rebuilds_at_wrap(tmp_path, triclinic):
    args = _model(tmp_path, triclinic=triclinic)
    with _provider(*args) as reference, _provider(*args, batch_mode="persistent") as pes:
        q = _frames(2)
        q[:, 0] = np.array([.002, .1, .1]) @ args[1]
        q[:, 1] = np.array([.935, .1, .1]) @ args[1]
        pes.evaluate(q)
        for displacement in (.01, .02, -.06):
            q[:, 0, 0] += displacement/BOHR_ANGSTROM
            got, expected = pes.evaluate(q), reference.evaluate(q)
            np.testing.assert_allclose(got.energy, expected.energy, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(got.forces, expected.forces, rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(got.virial, expected.virial, rtol=1e-12, atol=1e-15)
        assert pes.batch_stats["neighbor_reuses"] >= 2
        q[:, 0, 0] -= .02/BOHR_ANGSTROM  # Wrapped position jumps to the opposite face.
        before = pes.batch_stats["neighbor_rebuilds"]
        got, expected = pes.evaluate(q), reference.evaluate(q)
        assert pes.batch_stats["neighbor_rebuilds"] == before+2
        np.testing.assert_allclose(got.forces, expected.forces, rtol=1e-12, atol=1e-15)


def test_large_jump_discovers_pairs_absent_from_cached_neighbor_list(tmp_path):
    args = _model(tmp_path)
    with _provider(*args) as reference, _provider(*args, batch_mode="persistent", neighbor_skin=.4/BOHR_ANGSTROM) as pes:
        q = _frames(1)
        q[:, 1, 0] = 9/BOHR_ANGSTROM
        np.testing.assert_array_equal(pes.evaluate(q).forces, 0)
        q[:, 1, 0] = 3.3/BOHR_ANGSTROM
        got, expected = pes.evaluate(q), reference.evaluate(q)
        np.testing.assert_allclose(got.energy, expected.energy, rtol=1e-14)
        np.testing.assert_allclose(got.forces, expected.forces, rtol=1e-14)
        assert pes.batch_stats["neighbor_rebuilds"] == 2


def test_batch_growth_and_cell_changes_rebuild_all_slots_with_ewald(tmp_path):
    args = _model(tmp_path, charged=True, triclinic=True)
    with _provider(*args) as reference, _provider(*args, batch_mode="persistent") as pes:
        q = _frames(2)
        pes.evaluate(q)
        first = pes.engine
        cell = args[1] * 1.03
        pes.set_cell(cell)
        reference.set_cell(cell)
        q = _frames(4)*1.03
        got, expected = pes.evaluate(q), reference.evaluate(q)
        assert pes.engine is first
        assert len(pes.engines) == 4
        assert pes.batch_stats["neighbor_rebuilds"] == 6
        np.testing.assert_allclose(got.energy, expected.energy, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(got.forces, expected.forces, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(got.virial, expected.virial, rtol=1e-12, atol=1e-15)
        q[:, 1, 1] += .02/BOHR_ANGSTROM
        got, expected = pes.evaluate(q), reference.evaluate(q)
        np.testing.assert_allclose(got.forces, expected.forces, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(got.virial, expected.virial, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(pes.evaluate(q[:1]).energy, got.energy[:1], rtol=1e-14)
        assert len(pes.engines) == 4  # Smaller transient batches retain slot affinity.


def test_reordered_atoms_invalidate_cached_list(tmp_path):
    args = _model(tmp_path)
    with _provider(*args, batch_mode="persistent") as pes:
        q = _frames(1)
        expected = pes.evaluate(q)
        for name in ("id", "type", "x", "v", "f", "mask", "image"):
            values = pes.engine.numpy.extract_atom(name)
            values[:2] = values[:2][::-1].copy()
        got = pes.evaluate(q)
        np.testing.assert_allclose(got.forces, expected.forces, rtol=1e-14)
        assert pes.batch_stats["neighbor_rebuilds"] == 2


def test_invalid_batch_or_cell_is_rejected_before_mutation(tmp_path):
    args = _model(tmp_path, periodic=False)
    with _provider(*args, batch_mode="persistent") as pes:
        q = _frames(2)
        expected = pes.evaluate(q)
        stats = pes.batch_stats.copy()
        bad = q.copy()
        bad[1, 1, 0] = -1
        with pytest.raises(PotentialError, match="nonperiodic box"):
            pes.evaluate(bad)
        assert not pes.closed and pes.batch_stats == stats
        np.testing.assert_allclose(pes.evaluate(q).energy, expected.energy, rtol=1e-14)
    args = _model(tmp_path)
    with _provider(*args, batch_mode="persistent") as pes:
        expected = pes.evaluate(q)
        bad_cell = args[1].copy()
        bad_cell[1, 0] = 1
        with pytest.raises(PotentialError, match="cannot add tilt"):
            pes.set_cell(bad_cell)
        assert not pes.closed
        np.testing.assert_allclose(pes.evaluate(q).energy, expected.energy, rtol=1e-14)


@pytest.mark.parametrize("operation", ["evaluate", "set_cell"])
def test_backend_failure_closes_every_replica(tmp_path, operation):
    args = _model(tmp_path)
    pes = _provider(*args, batch_mode="persistent")
    q = _frames(3)
    pes.evaluate(q)
    engines = pes.engines
    with patch.object(engines[1], "command", side_effect=RuntimeError("injected backend failure")):
        with pytest.raises((PotentialError, RuntimeError), match="injected backend failure"):
            if operation == "evaluate":
                pes.evaluate(q)
            else:
                pes.set_cell(args[1]*1.02)
    assert pes.closed and pes.engine is None and pes.engines == ()
    assert all(engine.lmp is None for engine in engines)
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(q)
    pes.close()


def test_changed_model_cannot_create_mixed_batch_instances(tmp_path):
    args = _model(tmp_path)
    pes = _provider(*args, batch_mode="persistent")
    pes.evaluate(_frames(1))
    args[0].write_text(args[0].read_text().replace("0.2 1", "0.3 1"))
    with pytest.raises(PotentialError, match="model/assets changed"):
        pes.evaluate(_frames(2))
    assert pes.closed


def test_persistent_factory_kwargs_are_frozen_before_factory_mutation(tmp_path):
    path, cell, pbc = _model(tmp_path)
    factory = tmp_path / "factory.py"
    factory.write_text('''from lammps import lammps

def make(*, comm, model, parameters):
    epsilon = parameters.pop("epsilon")
    engine = lammps(comm=comm, cmdargs=["-log", "none", "-screen", "none"])
    engine.file(model)
    engine.command(f"pair_coeff 1 1 {epsilon} 1")
    return engine
''')
    kwargs = {"model": str(path), "parameters": {"epsilon": .2}}
    original = deepcopy(kwargs)
    with LAMMPSPotential(["H", "H"], factory=str(factory)+":make",
                         factory_kwargs=kwargs, type_map=["H"], cell=cell,
                         pbc=pbc, batch_mode="persistent") as pes:
        q = np.repeat(_frames(1), 3, axis=0)
        result = pes.evaluate(q)
        np.testing.assert_array_equal(result.energy, np.repeat(result.energy[:1], 3))
        np.testing.assert_array_equal(result.forces, np.repeat(result.forces[:1], 3, axis=0))
        assert len(pes.engines) == 3
    assert kwargs == original


@pytest.mark.parametrize("skin", [0, -1, np.inf, np.nan, True, np.bool_(True), "2", [2], 1j])
def test_invalid_neighbor_skin_is_explicit(skin):
    with pytest.raises(ValueError, match="neighbor_skin"):
        LAMMPSPotential(["H"], "unused.in", type_map=["H"], batch_mode="persistent", neighbor_skin=skin)


def test_serial_mode_preserves_metadata_and_rejects_unused_cache_option(tmp_path):
    args = _model(tmp_path)
    with _provider(*args) as pes:
        assert "batch_mode" not in pes.metadata
        assert "neighbor_skin_bohr" not in pes.metadata
        assert "ranks_per_instance" not in pes.metadata
        pes.evaluate(_frames(3))
        assert len(pes.engines) == 1
    with pytest.raises(ValueError, match="requires batch_mode"):
        _provider(*args, neighbor_skin=2)
    with pytest.raises(ValueError, match="batch_mode"):
        _provider(*args, batch_mode="auto")


def test_private_communicator_is_owned_and_world_is_rejected(tmp_path):
    MPI = pytest.importorskip("mpi4py.MPI")
    args = _model(tmp_path)
    with pytest.raises(ValueError, match="COMM_WORLD"):
        _provider(*args, communicator=MPI.COMM_WORLD)
    with pytest.raises(ValueError, match="non-null"):
        _provider(*args, communicator=MPI.COMM_NULL)
    comm = MPI.COMM_SELF.Dup()
    try:
        with _provider(*args, batch_mode="persistent", communicator=comm) as pes:
            assert pes.metadata["ranks_per_instance"] == 1
            pes.evaluate(_frames(2))
        assert comm.allreduce(1) == 1
    finally:
        comm.Free()


def test_persistent_mode_rejects_new_engine_side_fix(tmp_path):
    args = _model(tmp_path)
    pes = _provider(*args, batch_mode="persistent")
    pes.evaluate(_frames(2))
    pes.engines[1].command("fix forbidden all nve")
    with pytest.raises(PotentialError, match="cannot contain fixes"):
        pes.evaluate(_frames(2))
    assert pes.closed


def test_persistent_options_enter_existing_fingerprints():
    from pyrpmd.io import fingerprint
    from pyrpmd.system import System

    system = System(("H", "H"), np.full(2, 1837.), _frames(1)[0])
    config = {"task": "pimd", "beta": 1000., "beads": 2,
              "potential": {"provider": "lammps", "model": "unchanged.in",
                            "params": {"type_map": ["H"]}}}
    old = deepcopy(config)
    original = fingerprint(config, system)
    original_stream = fingerprint(config, system, {"replica": 0})
    persistent = deepcopy(config)
    persistent["potential"]["params"].update(batch_mode="persistent", neighbor_skin=3.)
    assert fingerprint(persistent, system) != original
    assert fingerprint(persistent, system, {"replica": 0}) != original_stream
    changed_skin = deepcopy(persistent)
    changed_skin["potential"]["params"]["neighbor_skin"] = 4.
    assert fingerprint(changed_skin, system) != fingerprint(persistent, system)
    assert config == old
    assert fingerprint(config, system) == original


def test_recreated_persistent_cache_resumes_lj_trajectory_and_rng_exactly(tmp_path):
    from pyrpmd.checkpoint import atomic_savez, load_checkpoint
    from pyrpmd.dynamics import Dynamics, State

    args = _model(tmp_path)
    options = {"batch_mode": "persistent", "neighbor_skin": .4/BOHR_ANGSTROM}
    parameters = {"masses": np.full(2, 1837.), "beta": 1000., "nbeads": 2,
                  "dt": .1, "native": False, "potential_identity": "lammps-lj-restart"}
    path = tmp_path / "restart.npz"
    with _provider(*args, **options) as original:
        dynamics = Dynamics(original, seed=85, **parameters)
        state = State(_frames(2), dynamics.sample_momenta())
        for _ in range(3):
            dynamics.sample_step(state)
        atomic_savez(path, kind="test.lammps.batch.restart.v1",
                     q=state.q, p=state.p, step=state.step, time=state.time,
                     operators=json.dumps(dynamics.operator_state_dict()))
        for _ in range(3):
            dynamics.sample_step(state)
        expected = state.copy()
        expected_operators = dynamics.operator_state_dict()
        assert original.batch_stats["neighbor_reuses"] > 0
    with _provider(*args, **options) as recreated:
        dynamics = Dynamics(recreated, seed=999, **parameters)
        with load_checkpoint(path, require_integrity=True) as saved:
            state = State(saved["q"], saved["p"], saved["step"].item(), saved["time"].item())
            dynamics.load_operator_state_dict(json.loads(saved["operators"].item()))
        assert recreated.batch_stats["neighbor_rebuilds"] == 0
        for _ in range(3):
            dynamics.sample_step(state)
        np.testing.assert_array_equal(state.q, expected.q)
        np.testing.assert_array_equal(state.p, expected.p)
        assert (state.step, state.time) == (expected.step, expected.time)
        assert dynamics.operator_state_dict() == expected_operators
