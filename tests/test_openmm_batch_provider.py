"""Independent pure-force checks for resident OpenMM Context batches."""

from threading import Event
import copy
import hashlib
import json

import numpy as np
import pytest

from pyrpmd.potentials import PotentialError, check_potential, make_potential
from pyrpmd.potentials.openmm import BOHR_NM, KJ_MOL_TO_AU, OpenMMPotential


def _bond(constraint=False):
    mm = pytest.importorskip("openmm")
    model = mm.System()
    model.addParticle(1.)
    model.addParticle(1.)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, .15, 100.)
    model.addForce(bond)
    if constraint:
        model.addConstraint(0, 1, .15)
    return model


def _coordinates(count=4):
    q = np.zeros((count, 2, 3))
    q[:, 0, 0] = np.arange(count) * .2
    q[:, 1, 0] = q[:, 0, 0] + np.linspace(2.5, 3.5, count)
    return q


def _periodic_model():
    mm = pytest.importorskip("openmm")
    model = mm.System()
    model.addParticle(1.)
    model.addParticle(1.)
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(.8)
    nonbonded.setEwaldErrorTolerance(1e-6)
    nonbonded.addParticle(1., .1, 0.)
    nonbonded.addParticle(-1., .1, 0.)
    model.addForce(nonbonded)
    model.setDefaultPeriodicBoxVectors(mm.Vec3(2, 0, 0), mm.Vec3(0, 2, 0), mm.Vec3(0, 0, 2))
    return model


@pytest.mark.parametrize("kwargs", [
    {"batch_mode": "automatic"}, {"max_workers": 0}, {"max_workers": -1},
    {"max_workers": True}, {"max_workers": 1.5}, {"max_workers": "2"},
    {"batch_mode": "serial", "max_workers": 2},
])
def test_batch_configuration_is_explicit_and_strict(kwargs):
    with pytest.raises(ValueError, match="batch_mode|max_workers"):
        OpenMMPotential("unused.xml", ["H"], **kwargs)


def test_serial_metadata_and_run_identity_inputs_are_unchanged():
    model = _bond()
    with OpenMMPotential(model, ["H", "H"], platform="Reference") as legacy:
        with OpenMMPotential(model, ["H", "H"], platform="Reference",
                             batch_mode="serial", max_workers=1) as explicit:
            assert explicit.metadata == legacy.metadata
            assert "batch_mode" not in legacy.metadata
            assert "max_workers" not in legacy.metadata
        with OpenMMPotential(model, ["H", "H"], platform="Reference",
                             batch_mode="contexts", max_workers=2) as batched:
            assert batched.metadata["batch_mode"] == "contexts"
            assert batched.metadata["max_workers"] == 2
            assert batched.metadata["native_batch"] is False
            assert batched.metadata["model_sha256"] == legacy.metadata["model_sha256"]


def test_context_options_enter_manifest_and_checkpoint_fingerprints():
    from pyrpmd.io import fingerprint, jsonable
    from pyrpmd.system import System

    system = System(("H", "H"), np.full(2, 1837.), _coordinates(1)[0])
    config = {"task": "pimd", "beta": 1000., "beads": 2,
              "potential": {"provider": "openmm", "model": "unchanged-system.xml",
                            "params": {"platform": "Reference"}}}
    stream = {"replica": 0}
    # Independent historical encodings: no provider construction and no new
    # defaults are allowed to alter an existing input or stream identity.
    legacy_manifest = hashlib.sha256(json.dumps(jsonable(
        {"config": config, "system": system.to_dict()}),
        sort_keys=True, allow_nan=False).encode()).hexdigest()
    legacy_stream = hashlib.sha256(json.dumps(jsonable({
        "schema": 1, "physics": config, "sampling_measure": "cv_density",
        "system": system.to_dict(), "stream": stream}),
        sort_keys=True, separators=(",", ":"), default=repr).encode()).hexdigest()
    assert fingerprint(config, system) == legacy_manifest
    assert fingerprint(config, system, stream) == legacy_stream
    contexts = copy.deepcopy(config)
    contexts["potential"]["params"].update(batch_mode="contexts", max_workers=2)
    assert fingerprint(contexts, system) != legacy_manifest
    assert fingerprint(contexts, system, stream) != legacy_stream
    more_workers = copy.deepcopy(contexts)
    more_workers["potential"]["params"]["max_workers"] = 4
    assert fingerprint(more_workers, system) != fingerprint(contexts, system)
    assert fingerprint(more_workers, system, stream) != fingerprint(contexts, system, stream)


@pytest.mark.parametrize("workers", [1, 2])
def test_context_batch_matches_analytic_energy_force_and_serial(workers):
    model = _bond()
    q = _coordinates()
    with OpenMMPotential(model, ["H", "H"], platform="Reference",
                         batch_mode="contexts", max_workers=workers) as pes:
        result = pes.evaluate(q)
        r = (q[:, 1, 0] - q[:, 0, 0]) * BOHR_NM
        np.testing.assert_allclose(result.energy, 50*(r-.15)**2*KJ_MOL_TO_AU, atol=1e-17)
        expected = 100*(r-.15)*KJ_MOL_TO_AU*BOHR_NM
        np.testing.assert_allclose(result.forces[:, 0, 0], expected, atol=1e-17)
        np.testing.assert_allclose(result.forces[:, 1, 0], -expected, atol=1e-17)
        np.testing.assert_array_equal(result.forces[:, :, 1:], 0)
        with OpenMMPotential(model, ["H", "H"], platform="Reference") as serial:
            reference = serial.evaluate(q)
            np.testing.assert_array_equal(result.energy, reference.energy)
            np.testing.assert_array_equal(result.forces, reference.forces)
        assert check_potential(pes, q)["max_absolute_force_error"] < 1e-12
        context_ids = [id(context) for context in pes._contexts]
        pes.evaluate(q[:1])
        pes.evaluate(q[::-1])
        assert [id(context) for context in pes._contexts] == context_ids
        for context in pes._contexts:
            state = context.getState(getPositions=True)
            assert state.getStepCount() == 0
            assert state.getTime().value_in_unit(pes.unit.picosecond) == 0
        assert pes.metadata["execution_device"] == "cpu"
    assert pes.context is None and pes.integrator is None
    assert pes._contexts == [] and pes._integrators == [] and pes._executor is None
    pes.close()
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(q)


def test_batch_virtual_sites_have_independent_parent_gradients():
    mm = pytest.importorskip("openmm")
    model = mm.System()
    for mass in (1., 0., 1.):
        model.addParticle(mass)
    model.setVirtualSite(1, mm.TwoParticleAverageSite(0, 2, .25, .75))
    force = mm.CustomExternalForce("6*(x*x+y*y+z*z)")
    force.addParticle(1, [])
    model.addForce(force)
    q = np.array([[[1., 2, 3], [2., 0, -1]], [[-1., 0, 2], [0., 3, 1]]])
    with OpenMMPotential(model, ["H", "H"], platform="Reference",
                         batch_mode="contexts", max_workers=2) as pes:
        result = pes.evaluate(q)
        site = (.25*q[:, 0] + .75*q[:, 1]) * BOHR_NM
        np.testing.assert_allclose(result.energy, 6*np.sum(site**2, axis=1)*KJ_MOL_TO_AU)
        for atom, weight in enumerate((.25, .75)):
            np.testing.assert_allclose(result.forces[:, atom],
                                       -12*weight*site*KJ_MOL_TO_AU*BOHR_NM)
        assert check_potential(pes, q)["max_absolute_force_error"] < 1e-12
        np.testing.assert_array_equal(pes.evaluate(q[::-1]).energy, result.energy[::-1])


def test_cpu_qtip4pf_pme_batch_matches_serial_at_platform_precision():
    pytest.importorskip("openmm")
    from benchmarks.water.qtip4pf import initial_grid, make_system
    from pyrpmd.constants import ANGSTROM_TO_AU

    q0, box = initial_grid(cells_per_axis=2)
    model = make_system(molecules=8, box_angstrom=box, cutoff_angstrom=.45*box)
    q = q0[None] + np.random.default_rng(472).normal(scale=.01, size=(4, len(q0), 3))
    options = {"cell": np.eye(3)*box*ANGSTROM_TO_AU,
               "platform": "CPU", "properties": {"Threads": "1"}}
    with OpenMMPotential(model, ["O", "H", "H"]*8, **options) as serial:
        reference = serial.evaluate(q)
    with OpenMMPotential(model, ["O", "H", "H"]*8, **options,
                         batch_mode="contexts", max_workers=2) as batched:
        result = batched.evaluate(q)
        # CPU nonbonded kernels use single precision internally. Independent
        # Contexts can reorder particles differently; Reference-only tests
        # above separately require double precision analytic agreement.
        np.testing.assert_allclose(result.energy, reference.energy, rtol=3e-6, atol=1e-7)
        np.testing.assert_allclose(result.forces, reference.forces, rtol=3e-6, atol=1e-7)
        reverse = batched.evaluate(q[::-1])
        np.testing.assert_allclose(reverse.energy, reference.energy[::-1], rtol=3e-6, atol=1e-7)
        np.testing.assert_allclose(reverse.forces, reference.forces[::-1], rtol=3e-6, atol=1e-7)


def test_context_pool_recreation_resumes_rng_and_trajectory_exactly(tmp_path):
    from pyrpmd.checkpoint import atomic_savez, load_checkpoint
    from pyrpmd.dynamics import Dynamics, State

    model = _bond()
    options = {"platform": "Reference", "batch_mode": "contexts", "max_workers": 2}
    parameters = {"masses": np.full(2, 1837.), "beta": 1000., "nbeads": 2,
                  "dt": .1, "native": False, "potential_identity": "openmm-reference-bond"}
    path = tmp_path / "restart.npz"
    with OpenMMPotential(model, ["H", "H"], **options) as original:
        dynamics = Dynamics(original, seed=85, **parameters)
        state = State(_coordinates(2), dynamics.sample_momenta())
        for _ in range(3):
            dynamics.sample_step(state)
        atomic_savez(path, kind="test.openmm.batch.restart.v1",
                     q=state.q, p=state.p, step=state.step, time=state.time,
                     operators=json.dumps(dynamics.operator_state_dict()))
        for _ in range(3):
            dynamics.sample_step(state)
        expected = state.copy()
        expected_operators = dynamics.operator_state_dict()
    with OpenMMPotential(model, ["H", "H"], **options) as recreated:
        dynamics = Dynamics(recreated, seed=999, **parameters)
        with load_checkpoint(path, require_integrity=True) as saved:
            state = State(saved["q"], saved["p"], saved["step"].item(), saved["time"].item())
            dynamics.load_operator_state_dict(json.loads(saved["operators"].item()))
        assert len(recreated._contexts) == 1  # No engine cache was checkpointed.
        for _ in range(3):
            dynamics.sample_step(state)
        np.testing.assert_array_equal(state.q, expected.q)
        np.testing.assert_array_equal(state.p, expected.p)
        assert (state.step, state.time) == (expected.step, expected.time)
        assert dynamics.operator_state_dict() == expected_operators


def test_context_mode_requires_declared_constraints_but_never_projects():
    model = _bond(constraint=True)
    q = _coordinates(2)  # Intentionally off the constraint surface.
    with OpenMMPotential(model, ["H", "H"], platform="Reference",
                         batch_mode="contexts", max_workers=2) as pes:
        with pytest.raises(PotentialError, match="Bind OpenMM"):
            pes.evaluate(q)
        with pytest.raises(PotentialError, match="must also be declared"):
            pes.bind_system({"symbols": ["H", "H"]})
        pes.bind_system({"symbols": ["H", "H"],
                         "bond_constraints": [(0, 1, .15/BOHR_NM)]})
        result = pes.evaluate(q)
        r = (q[:, 1, 0]-q[:, 0, 0])*BOHR_NM
        np.testing.assert_allclose(result.energy, 50*(r-.15)**2*KJ_MOL_TO_AU)
        for frame, context in zip(q, pes._contexts):
            actual = context.getState(getPositions=True).getPositions(asNumpy=True)
            np.testing.assert_allclose(actual.value_in_unit(pes.unit.nanometer), frame*BOHR_NM)


def test_factory_routes_new_execution_options(tmp_path):
    mm = pytest.importorskip("openmm")
    path = tmp_path / "bond.xml"
    path.write_text(mm.XmlSerializer.serialize(_bond()))
    spec = {"provider": "openmm", "model": str(path),
            "params": {"batch_mode": "contexts", "max_workers": 2, "platform": "Reference"}}
    with make_potential(spec, {"symbols": ["H", "H"]}) as pes:
        assert pes.metadata["max_workers"] == 2
        assert pes.evaluate(_coordinates()).energy.shape == (4,)


def test_periodic_cell_updates_all_resident_copies_and_future_copies():
    model = _periodic_model()
    cell = np.eye(3)*2/BOHR_NM
    q = np.array([[[.2, .3, .4], [.7, .6, .5]],
                  [[.1, .2, .3], [.8, .4, .4]]])/BOHR_NM
    with OpenMMPotential(model, ["H", "H"], cell=cell, platform="Reference",
                         batch_mode="contexts", max_workers=2) as pes:
        pes.evaluate(q)
        pes.set_cell(cell*1.1)
        # Growth after a cell change must use the new default box and PME grid.
        q3 = np.concatenate((q, q[:1]+.1))
        result = pes.evaluate(q3)
        with OpenMMPotential(model, ["H", "H"], cell=cell*1.1,
                             platform="Reference") as reference:
            expected = reference.evaluate(q3)
        np.testing.assert_allclose(result.energy, expected.energy, atol=2e-10)
        np.testing.assert_allclose(result.forces, expected.forces, atol=3e-8)
        for context in pes._contexts:
            actual = context.getState().getPeriodicBoxVectors(asNumpy=True)
            np.testing.assert_allclose(actual.value_in_unit(pes.unit.nanometer), cell*1.1*BOHR_NM)


def test_failed_context_growth_leaves_existing_copies_usable(monkeypatch):
    with OpenMMPotential(_bond(), ["H", "H"], platform="Reference",
                         batch_mode="contexts", max_workers=2) as pes:
        q = _coordinates(1)
        expected = pes.evaluate(q)
        original = pes._new_context
        attempts = []

        def fail_after_one():
            attempts.append(1)
            if len(attempts) == 2:
                raise RuntimeError("injected allocation failure")
            return original()

        monkeypatch.setattr(pes, "_new_context", fail_after_one)
        with pytest.raises(PotentialError, match="Cannot allocate OpenMM batch"):
            pes.evaluate(_coordinates(3))
        assert len(pes._contexts) == len(pes._integrators) == 1
        np.testing.assert_array_equal(pes.evaluate(q).energy, expected.energy)
        monkeypatch.setattr(pes, "_new_context", original)
        assert pes.evaluate(_coordinates(3)).energy.shape == (3,)


class _ContextProxy:
    def __init__(self, context, **overrides):
        self.context, self.overrides = context, overrides

    def __getattr__(self, name):
        return self.overrides.get(name, getattr(self.context, name))


def test_failed_cell_update_rolls_back_every_context(monkeypatch):
    cell = np.eye(3)*2/BOHR_NM
    with OpenMMPotential(_periodic_model(), ["H", "H"], cell=cell,
                         platform="Reference", batch_mode="contexts") as pes:
        q = _coordinates(3)
        expected = pes.evaluate(q)
        actual_context = pes._contexts[1]
        attempts = []

        def fail_once(*vectors):
            actual_context.setPeriodicBoxVectors(*vectors)
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("injected box failure after mutation")

        pes._contexts[1] = _ContextProxy(actual_context, setPeriodicBoxVectors=fail_once)
        with pytest.raises(PotentialError, match="previous cell restored"):
            pes.set_cell(cell*1.1)
        np.testing.assert_array_equal(pes.cell, cell)
        for context in pes._contexts:
            vectors = context.getState().getPeriodicBoxVectors(asNumpy=True)
            np.testing.assert_allclose(vectors.value_in_unit(pes.unit.nanometer), cell*BOHR_NM)
        np.testing.assert_allclose(pes.evaluate(q).energy, expected.energy, atol=1e-14)
        assert not pes.closed


def test_failed_parallel_frame_drains_workers_before_retry():
    with OpenMMPotential(_bond(), ["H", "H"], platform="Reference",
                         batch_mode="contexts", max_workers=2) as pes:
        q = _coordinates(2)
        expected = pes.evaluate(q)
        contexts = list(pes._contexts)
        other_started, other_finished = Event(), Event()

        def fail_state(**kwargs):
            assert other_started.wait(10)
            raise RuntimeError("injected frame failure")

        def other_state(**kwargs):
            other_started.set()
            try:
                return contexts[1].getState(**kwargs)
            finally:
                other_finished.set()

        pes._contexts[:] = [_ContextProxy(contexts[0], getState=fail_state),
                            _ContextProxy(contexts[1], getState=other_state)]
        with pytest.raises(PotentialError, match="batch row 0 failed"):
            pes.evaluate(q)
        assert other_finished.is_set()
        pes._contexts[:] = contexts
        np.testing.assert_array_equal(pes.evaluate(q).energy, expected.energy)
