"""Exact physical forces from an unstepped OpenMM RPMD copy bank."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.potentials import PotentialError, check_potential, make_potential
from pyrpmd.potentials.openmm import BOHR_NM, KJ_MOL_TO_AU, OpenMMPotential


def _bond():
    mm = pytest.importorskip("openmm")
    model = mm.System()
    for _ in range(2):
        model.addParticle(1.)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, .15, 100.)
    model.addForce(bond)
    return model


def _coordinates(count=4):
    q = np.zeros((count, 2, 3))
    q[:, 0, 0] = np.arange(count) * .2
    q[:, 1, 0] = q[:, 0, 0] + np.linspace(2.5, 3.5, count)
    return q


def _periodic_model():
    mm = pytest.importorskip("openmm")
    model = mm.System()
    for _ in range(2):
        model.addParticle(1.)
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.PME)
    force.setEwaldErrorTolerance(1e-7)
    force.setCutoffDistance(.8)
    force.addParticle(1., .1, 0.)
    force.addParticle(-1., .1, 0.)
    model.addForce(force)
    model.setDefaultPeriodicBoxVectors(mm.Vec3(2, 0, 0), mm.Vec3(0, 2, 0), mm.Vec3(0, 0, 2))
    return model


@pytest.mark.parametrize("workers", [2, 4])
def test_rpmd_copy_bank_does_not_accept_host_workers(workers):
    with pytest.raises(ValueError, match="max_workers.*contexts"):
        OpenMMPotential("unused.xml", ["H"], batch_mode="rpmd", max_workers=workers)


def test_rpmd_rejects_constraints_explicitly():
    model = _bond()
    model.addConstraint(0, 1, .15)
    with pytest.raises(PotentialError, match="RPMD.*constraints"):
        OpenMMPotential(model, ["H", "H"], platform="Reference", batch_mode="rpmd")


@pytest.mark.parametrize("beads", [1, 8, 32, 64])
def test_rpmd_bank_returns_unscaled_physical_energy_force_without_stepping(beads):
    q = _coordinates(beads)
    with OpenMMPotential(_bond(), ["H", "H"], platform="Reference", batch_mode="rpmd") as pes:
        out = pes.evaluate(q)
        r = (q[:, 1, 0] - q[:, 0, 0]) * BOHR_NM
        np.testing.assert_allclose(out.energy, 50 * (r - .15)**2 * KJ_MOL_TO_AU, atol=1e-17)
        expected = 100 * (r - .15) * KJ_MOL_TO_AU * BOHR_NM
        np.testing.assert_allclose(out.forces[:, 0, 0], expected, atol=1e-17)
        np.testing.assert_allclose(out.forces[:, 1, 0], -expected, atol=1e-17)
        np.testing.assert_array_equal(out.forces[:, :, 1:], 0.)
        assert out.virial is None
        assert pes.integrator.getNumCopies() == beads
        assert pes.integrator.getApplyThermostat() is False
        assert dict(pes.integrator.getContractions()) == {}
        assert len(pes._contexts) == len(pes._integrators) == 1
        for index in range(beads):
            state = pes.integrator.getState(index, getPositions=True)
            assert state.getStepCount() == 0
            assert state.getTime().value_in_unit(pes.unit.picosecond) == 0.
            np.testing.assert_allclose(state.getPositions(asNumpy=True).value_in_unit(
                pes.unit.nanometer), q[index] * BOHR_NM)
        # These affect only an integration step/ring springs, not a physical PES.
        pes.integrator.setTemperature(10000.)
        pes.integrator.setStepSize(.2)
        np.testing.assert_array_equal(pes.evaluate(q).energy, out.energy)
        np.testing.assert_array_equal(pes.evaluate(q).forces, out.forces)


def test_rpmd_reuses_bank_and_refreshes_changed_and_reordered_rows(monkeypatch):
    with OpenMMPotential(_bond(), ["H", "H"], platform="Reference", batch_mode="rpmd") as pes:
        q = _coordinates(4)
        reference = pes.evaluate(q)
        context, integrator = pes.context, pes.integrator
        metadata = copy.deepcopy(pes.metadata)
        monkeypatch.setattr(integrator, "step", lambda *_: pytest.fail("nuclei were propagated"))
        np.testing.assert_array_equal(pes.evaluate(q[::-1]).energy, reference.energy[::-1])
        np.testing.assert_array_equal(pes.evaluate(q[:1]).forces, reference.forces[:1])
        changed = q.copy()
        changed[:, 1, 0] += .1
        assert not np.array_equal(pes.evaluate(changed).energy, reference.energy)
        np.testing.assert_array_equal(pes.evaluate(q).energy, reference.energy)
        assert pes.context is context and pes.integrator is integrator
        assert pes.metadata == metadata
        assert check_potential(pes, q[:1])["max_absolute_force_error"] < 1e-12
    assert pes.context is pes.integrator is None
    assert pes._contexts == pes._integrators == []
    pes.close()
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(q)


def test_rpmd_allocation_failure_preserves_the_previous_bank(monkeypatch):
    with OpenMMPotential(_bond(), ["H", "H"], platform="Reference", batch_mode="rpmd") as pes:
        q = _coordinates(2)
        reference = pes.evaluate(q)
        previous = pes.context, pes.integrator
        original = pes._new_context

        def unavailable(copies=1):
            raise RuntimeError("simulated device allocation failure")

        monkeypatch.setattr(pes, "_new_context", unavailable)
        with pytest.raises(PotentialError, match="RPMD bank of 4 copies on Reference.*allocation failure"):
            pes.evaluate(_coordinates(4))
        assert (pes.context, pes.integrator) == previous
        np.testing.assert_array_equal(pes.evaluate(q).energy, reference.energy)
        monkeypatch.setattr(pes, "_new_context", original)
        pes.evaluate(_coordinates(4))
        assert pes.context is not previous[0] and pes.integrator is not previous[1]
        assert pes.integrator.getNumCopies() == 4


@pytest.mark.parametrize("phase", ["upload", "evaluation"])
def test_rpmd_failed_row_is_recoverable_without_stale_forces(monkeypatch, phase):
    with OpenMMPotential(_bond(), ["H", "H"], platform="Reference", batch_mode="rpmd") as pes:
        q = _coordinates(4)
        reference = pes.evaluate(q)
        method = "setPositions" if phase == "upload" else "getState"
        original = getattr(pes.integrator, method)

        def fail(index, *args, **kwargs):
            if index == 1:
                raise RuntimeError("injected row failure")
            return original(index, *args, **kwargs)

        monkeypatch.setattr(pes.integrator, method, fail)
        with pytest.raises(PotentialError, match=f"RPMD {phase} row 1.*injected row failure"):
            pes.evaluate(q[::-1])
        monkeypatch.setattr(pes.integrator, method, original)
        result = pes.evaluate(q)
        np.testing.assert_array_equal(result.energy, reference.energy)
        np.testing.assert_array_equal(result.forces, reference.forces)


@pytest.mark.parametrize("kind", ["average", "out_of_plane", "local"])
@pytest.mark.parametrize("cuda", [False, True])
def test_rpmd_general_virtual_sites_recompute_and_redistribute_parent_force(kind, cuda):
    device = _cuda_device() if cuda else "cpu"
    mm = pytest.importorskip("openmm")
    model = mm.System()
    for mass in (1., 0., 1., 1.):
        model.addParticle(mass)
    if kind == "average":
        site = mm.ThreeParticleAverageSite(0, 2, 3, .2, .3, .5)
    elif kind == "out_of_plane":
        site = mm.OutOfPlaneSite(0, 2, 3, .25, .3, .1)
    else:
        site = mm.LocalCoordinatesSite([0, 2, 3], [1., 0., 0.],
                                      [-1., 1., 0.], [-1., 0., 1.], mm.Vec3(.02, .03, .04))
    model.setVirtualSite(1, site)
    force = mm.CustomExternalForce("6*(x*x+y*y+z*z)")
    force.addParticle(1, [])
    model.addForce(force)
    q = np.array([[[1., 2., 3.], [2., 1., -1.], [0., 3., 2.]],
                  [[-1., 0., 2.], [0., 3., 1.], [2., -1., 0.]]])
    with OpenMMPotential(model, ["H"] * 3, platform="Reference") as serial:
        reference = serial.evaluate(q)
    with OpenMMPotential(model, ["H"] * 3, device=device,
                         platform="CUDA" if cuda else "Reference", batch_mode="rpmd") as pes:
        out = pes.evaluate(q)
        if not cuda:
            np.testing.assert_array_equal(out.energy, reference.energy)
            np.testing.assert_array_equal(out.forces, reference.forces)
        np.testing.assert_allclose(out.energy, reference.energy, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(out.forces, reference.forces, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(pes.evaluate(q[::-1]).forces, reference.forces[::-1], atol=1e-12, rtol=1e-10)
        assert check_potential(pes, q)["max_absolute_force_error"] < (5e-10 if cuda else 2e-11)
        if kind == "average":
            weighted = (.2 * q[:, 0] + .3 * q[:, 1] + .5 * q[:, 2]) * BOHR_NM
            np.testing.assert_allclose(out.energy, 6 * np.sum(weighted**2, axis=1) * KJ_MOL_TO_AU)
            for atom, weight in enumerate((.2, .3, .5)):
                np.testing.assert_allclose(out.forces[:, atom],
                                           -12 * weight * weighted * KJ_MOL_TO_AU * BOHR_NM)


def test_rpmd_fixed_cell_pme_growth_and_recreation_preserve_physical_forces():
    cell = np.array([[2., 0, 0], [.1, 2., 0], [.2, .1, 2.]]) / BOHR_NM
    q = np.array([[[.2, .3, .4], [.7, .6, .5]],
                  [[.1, .2, .3], [.8, .4, .4]]]) / BOHR_NM
    expanded = np.concatenate([q, q + .01])
    with OpenMMPotential(_periodic_model(), ["H", "H"], cell=cell,
                         platform="Reference") as serial:
        reference = serial.evaluate(expanded)
    with OpenMMPotential(_periodic_model(), ["H", "H"], cell=cell,
                         platform="Reference", batch_mode="rpmd") as pes:
        before = pes.evaluate(q)
        # Normal system binding can supply the identical fixed box.
        pes.set_cell(cell.copy())
        pes.bind_system({"symbols": ["H", "H"], "cell": cell.copy(), "pbc": [True] * 3})
        after = pes.evaluate(expanded)
        np.testing.assert_array_equal(before.energy, after.energy[:2])
        np.testing.assert_array_equal(before.forces, after.forces[:2])
        np.testing.assert_allclose(after.energy, reference.energy, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(after.forces, reference.forces, atol=1e-14, rtol=1e-13)
        assert pes.metadata["fixed_cell"] is True
    # A checkpoint may recreate a different cache capacity. The fixed-cell
    # force oracle must be bitwise independent of previous batch-size history.
    with OpenMMPotential(_periodic_model(), ["H", "H"], cell=cell,
                         platform="Reference", batch_mode="rpmd") as recreated:
        restored = recreated.evaluate(expanded)
        np.testing.assert_array_equal(restored.energy, after.energy)
        np.testing.assert_array_equal(restored.forces, after.forces)


@pytest.mark.parametrize("method", ["set_cell", "bind_system"])
def test_rpmd_cell_change_is_rejected_before_mutating_any_state(method):
    cell = np.eye(3) * 2 / BOHR_NM
    q = _coordinates(2)
    with OpenMMPotential(_periodic_model(), ["H", "H"], cell=cell,
                         platform="Reference", batch_mode="rpmd") as pes:
        before = pes.evaluate(q)
        context, integrator = pes.context, pes.integrator
        metadata = copy.deepcopy(pes.metadata)
        with pytest.raises(PotentialError, match="RPMD.*fixed cell"):
            if method == "set_cell":
                pes.set_cell(cell * 1.2)
            else:
                pes.bind_system({"symbols": ["H", "H"], "cell": cell * 1.2, "pbc": [True] * 3})
        assert pes.context is context and pes.integrator is integrator
        assert pes.metadata == metadata
        np.testing.assert_array_equal(pes.cell, cell)
        box = pes.context.getState().getPeriodicBoxVectors(asNumpy=True)
        np.testing.assert_array_equal(box.value_in_unit(pes.unit.nanometer), cell * BOHR_NM)
        after = pes.evaluate(q)
        np.testing.assert_array_equal(before.energy, after.energy)
        np.testing.assert_array_equal(before.forces, after.forces)


def test_rpmd_virtual_site_guard_rejects_wrong_selected_copy(monkeypatch):
    mm = pytest.importorskip("openmm")
    model = _bond()
    model.addParticle(0.)
    model.setVirtualSite(2, mm.TwoParticleAverageSite(0, 1, .25, .75))
    force = mm.CustomExternalForce("6*(x*x+y*y+z*z)")
    force.addParticle(2, [])
    model.addForce(force)
    with OpenMMPotential(model, ["H", "H"], platform="Reference", batch_mode="rpmd") as pes:
        q = _coordinates(4)
        reference = pes.evaluate(q)
        original = pes.integrator.getState

        def wrong_copy(index, **kwargs):
            return original(0, **kwargs)

        monkeypatch.setattr(pes.integrator, "getState", wrong_copy)
        with pytest.raises(PotentialError, match="row 1.*did not select.*incompatible"):
            pes.evaluate(q)
        monkeypatch.setattr(pes.integrator, "getState", original)
        np.testing.assert_array_equal(pes.evaluate(q).forces, reference.forces)


def test_rpmd_factory_metadata_and_identity_are_opt_in(tmp_path):
    from pyrpmd.io import fingerprint
    from pyrpmd.system import System

    mm = pytest.importorskip("openmm")
    path = tmp_path / "bond.xml"
    path.write_text(mm.XmlSerializer.serialize(_bond()))
    original = {"provider": "openmm", "model": str(path), "params": {"platform": "Reference"}}
    resident = copy.deepcopy(original)
    resident["params"]["batch_mode"] = "rpmd"
    with make_potential(original, {"symbols": ["H", "H"]}) as legacy:
        legacy_metadata = copy.deepcopy(legacy.metadata)
        assert "batch_mode" not in legacy.metadata
        with make_potential(resident, {"symbols": ["H", "H"]}) as pes:
            assert pes.metadata["batch_mode"] == "rpmd"
            assert pes.metadata["copy_api_transfers"] == "per_row"
            assert pes.metadata["native_batch"] is False
            assert pes.metadata["model_sha256"] == legacy.metadata["model_sha256"]
            pes.evaluate(_coordinates(4))
        assert legacy.metadata == legacy_metadata
    system = System(("H", "H"), np.full(2, 1837.), _coordinates(1)[0])
    old = {"task": "pimd", "potential": original}
    new = {"task": "pimd", "potential": resident}
    assert fingerprint(old, system) != fingerprint(new, system)


@pytest.mark.parametrize("model_name", ["bond", "pme"])
def test_rpmd_disposable_copy_bank_exact_stochastic_resume(tmp_path, model_name):
    from pyrpmd.checkpoint import atomic_savez, load_checkpoint
    from pyrpmd.dynamics import Dynamics, State

    options = {"platform": "Reference", "batch_mode": "rpmd"}
    model_factory = _bond if model_name == "bond" else _periodic_model
    if model_name == "pme":
        options["cell"] = np.eye(3) * 2 / BOHR_NM
    parameters = {"masses": np.full(2, 1837.), "beta": 1000., "nbeads": 2,
                  "dt": .1, "native": False, "potential_identity": "rpmd-copy-bond"}
    path = tmp_path / "restart.npz"
    with OpenMMPotential(model_factory(), ["H", "H"], **options) as original:
        dynamics = Dynamics(original, seed=85, **parameters)
        state = State(_coordinates(2), dynamics.sample_momenta())
        for _ in range(3):
            dynamics.sample_step(state)
        atomic_savez(path, kind="test.openmm.rpmd.restart.v1", q=state.q, p=state.p,
                     step=state.step, time=state.time,
                     operators=json.dumps(dynamics.operator_state_dict()))
        for _ in range(3):
            dynamics.sample_step(state)
        expected = state.copy()
        expected_operators = dynamics.operator_state_dict()
    with OpenMMPotential(model_factory(), ["H", "H"], **options) as recreated:
        dynamics = Dynamics(recreated, seed=999, **parameters)
        with load_checkpoint(path, require_integrity=True) as saved:
            state = State(saved["q"], saved["p"], saved["step"].item(), saved["time"].item())
            dynamics.load_operator_state_dict(json.loads(saved["operators"].item()))
        assert recreated.integrator.getNumCopies() == 1
        for _ in range(3):
            dynamics.sample_step(state)
        np.testing.assert_array_equal(state.q, expected.q)
        np.testing.assert_array_equal(state.p, expected.p)
        assert (state.step, state.time) == (expected.step, expected.time)
        assert dynamics.operator_state_dict() == expected_operators


def _cuda_device():
    mm = pytest.importorskip("openmm")
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
    except Exception:
        pytest.skip("OpenMM CUDA platform is unavailable")
    integrator = mm.VerletIntegrator(.001)
    model = _bond()
    try:
        context = mm.Context(model, integrator, platform,
                             {"DeviceIndex": "0", "Precision": "double"})
    except Exception as exc:
        pytest.skip(f"OpenMM CUDA device is unavailable: {exc}")
    del context, integrator
    return "cuda:0"


@pytest.mark.parametrize("beads", [8, 32, 64])
def test_cuda_rpmd_bank_matches_cpu_float64_and_contexts(beads):
    device = _cuda_device()
    q = _coordinates(beads)
    model = _bond()
    with OpenMMPotential(model, ["H", "H"], platform="Reference") as serial:
        expected = serial.evaluate(q)
    with OpenMMPotential(model, ["H", "H"], device=device, batch_mode="contexts", max_workers=2) as contexts:
        context_result = contexts.evaluate(q)
    with OpenMMPotential(model, ["H", "H"], device=device, batch_mode="rpmd") as resident:
        result = resident.evaluate(q)
        assert resident.metadata["platform_properties"]["Precision"] == "double"
        np.testing.assert_allclose(result.energy, expected.energy, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(result.forces, expected.forces, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(result.energy, context_result.energy, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(result.forces, context_result.forces, atol=1e-12, rtol=1e-10)
