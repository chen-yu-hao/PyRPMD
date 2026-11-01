"""Actual OpenMM energies, virtual-site gradients and periodic contracts."""

import sys
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.potentials import BackendUnavailableError, PotentialError, check_potential, make_potential
from pyrpmd.potentials.openmm import BOHR_NM, KJ_MOL_TO_AU, OpenMMPotential


def _openmm():
    return pytest.importorskip("openmm")


def _bond(*, constraint=False):
    mm = _openmm()
    model = mm.System()
    model.addParticle(1.0)
    model.addParticle(1.0)
    bond = mm.HarmonicBondForce()
    bond.addBond(0, 1, 0.15, 100.0)
    model.addForce(bond)
    if constraint:
        model.addConstraint(0, 1, 0.15)
    return model


def test_missing_openmm_has_actionable_error():
    with patch.dict(sys.modules, {"openmm": None}):
        with pytest.raises(BackendUnavailableError, match=r"pyrpmd\[openmm\]"):
            OpenMMPotential("missing.xml", ["H"])


def test_openmm_input_contracts_fail_before_loading_model():
    for kwargs in ({}, {"model": "x", "factory": "x:y"}):
        with pytest.raises(ValueError, match="exactly one"):
            OpenMMPotential(symbols=["H"], **kwargs)
    with pytest.raises(PotentialError, match="partial PBC"):
        make_potential({"provider": "openmm", "model": "absent.xml"},
                       {"symbols": ["H"], "cell": np.eye(3), "pbc": [True, False, True]})
    with pytest.raises(PotentialError, match="fixed engine units"):
        make_potential({"provider": "openmm", "model": "absent.xml", "units": {}},
                       {"symbols": ["H"]})


def test_actual_harmonic_bond_atomic_units_batch_and_reuse(tmp_path):
    mm = _openmm()
    path = tmp_path / "system.xml"
    model = _bond()
    path.write_text(mm.XmlSerializer.serialize(model))
    q = np.array([[[0., 0, 0], [3.2, 0, 0]], [[1., 1, 1], [3.8, 1, 1]]])
    spec = {"provider": "openmm", "model": str(path), "params": {"platform": "Reference"}}
    with make_potential(spec, {"symbols": ["D", "H"]}) as pes:
        context = pes.context
        result = pes.evaluate(q)
        r = (q[:, 1, 0]-q[:, 0, 0]) * BOHR_NM
        np.testing.assert_allclose(result.energy, 50*(r-.15)**2*KJ_MOL_TO_AU, atol=1e-17)
        expected = 100*(r-.15)*KJ_MOL_TO_AU*BOHR_NM
        np.testing.assert_allclose(result.forces[:, 0, 0], expected, atol=1e-17)
        np.testing.assert_allclose(result.forces[:, 1, 0], -expected, atol=1e-17)
        assert result.virial is None
        assert check_potential(pes, q)["max_absolute_force_error"] < 1e-12
        np.testing.assert_allclose(pes.evaluate(q[::-1]).energy, result.energy[::-1], atol=1e-17)
        assert pes.context is context
        assert len(pes.metadata["model_sha256"]) == 64
        assert pes.metadata["native_batch"] is False
        assert pes.metadata["execution_device"] == "cpu"
    assert pes.context is None and pes.integrator is None
    with pytest.raises(PotentialError, match="closed"):
        pes.evaluate(q)


def test_actual_virtual_site_coordinates_and_parent_force_redistribution():
    mm = _openmm()
    model = mm.System()
    model.addParticle(1.0)
    model.addParticle(0.0)
    model.addParticle(1.0)
    model.setVirtualSite(1, mm.TwoParticleAverageSite(0, 2, .25, .75))
    force = mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    force.addGlobalParameter("k", 12.0)
    force.addParticle(1, [])
    model.addForce(force)
    q = np.array([[[1., 2, 3], [2., 0, -1]], [[-1., 0, 2], [0., 3, 1]]])
    with OpenMMPotential(model, ["H", "D"], platform="Reference") as pes:
        out = pes.evaluate(q)
        site = (.25*q[:, 0]+.75*q[:, 1])*BOHR_NM
        np.testing.assert_allclose(out.energy, 6*np.sum(site**2, axis=1)*KJ_MOL_TO_AU)
        for i, weight in enumerate((.25, .75)):
            np.testing.assert_allclose(out.forces[:, i], -12*weight*site*KJ_MOL_TO_AU*BOHR_NM)
        assert check_potential(pes, q)["max_absolute_force_error"] < 1e-12
        assert pes.particle_indices.tolist() == [0, 2]
        # The provider owns an immutable model copy, independent of the factory.
        model.getForce(0).setGlobalParameterDefaultValue(0, 300.)
        np.testing.assert_array_equal(pes.evaluate(q).energy, out.energy)


def _periodic_model(method="PME"):
    mm = _openmm()
    model = mm.System()
    for _ in range(2):
        model.addParticle(1.0)
    force = mm.NonbondedForce()
    force.setNonbondedMethod(getattr(mm.NonbondedForce, method))
    force.setCutoffDistance(.8)
    force.setEwaldErrorTolerance(1e-7)
    force.addParticle(1., .1, 0.)
    force.addParticle(-1., .1, 0.)
    model.addForce(force)
    model.setDefaultPeriodicBoxVectors(mm.Vec3(2, 0, 0), mm.Vec3(0, 2, 0), mm.Vec3(0, 0, 2))
    return model


@pytest.mark.parametrize("method", ["PME", "Ewald"])
def test_actual_periodic_electrostatics_forces_translation_and_cell(method):
    cell = np.array([[2., 0, 0], [.1, 2., 0], [.2, .1, 2.]])/BOHR_NM
    if method == "Ewald":
        cell = np.eye(3)*2/BOHR_NM
    q = np.array([[[.2, .3, .4], [.7, .6, .5]]])/BOHR_NM
    with OpenMMPotential(_periodic_model(method), ["H", "H"], cell=cell, platform="Reference") as pes:
        out = pes.evaluate(q)
        shifted = q.copy()
        shifted[:, 0] += 2*cell[0]-cell[2]
        shifted_out = pes.evaluate(shifted)
        np.testing.assert_allclose(shifted_out.energy, out.energy, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(shifted_out.forces, out.forces, rtol=1e-8, atol=1e-10)
        assert check_potential(pes, q, epsilon=1e-4)["max_absolute_force_error"] < 3e-7
        pes.set_cell(cell*1.1)
        with OpenMMPotential(_periodic_model(method), ["H", "H"], cell=cell*1.1, platform="Reference") as independent:
            np.testing.assert_allclose(pes.evaluate(q).energy, independent.evaluate(q).energy, atol=1e-12)


def test_openmm_constraint_model_requires_matching_pyrpmd_constraints():
    model = _bond(constraint=True)
    q = np.array([[[0., 0, 0], [.15/BOHR_NM, 0, 0]]])
    with OpenMMPotential(model, ["H", "H"], platform="Reference") as pes:
        with pytest.raises(PotentialError, match="Bind OpenMM"):
            pes.evaluate(q)
        with pytest.raises(PotentialError, match="must also be declared"):
            pes.bind_system({"symbols": ["H", "H"]})
        with pytest.raises(PotentialError, match="must also be declared"):
            pes.bind_system({"symbols": ["H", "H"], "bond_constraints": [(0, 1, 2.)]})
        pes.bind_system({"symbols": ["H", "H"], "bond_constraints": [(1, 0, .15/BOHR_NM)]})
        np.testing.assert_allclose(pes.evaluate(q).energy, 0, atol=1e-24)


def test_openmm_rejects_propagators_bad_particle_count_and_geometry_contracts():
    mm = _openmm()
    model = _bond()
    model.addForce(mm.CMMotionRemover())
    with pytest.raises(PotentialError, match="propagation operator"):
        OpenMMPotential(model, ["H", "H"], platform="Reference")
    with pytest.raises(PotentialError, match="particle count"):
        OpenMMPotential(_bond(), ["H"], platform="Reference")
    with pytest.raises(PotentialError, match="periodicity"):
        OpenMMPotential(_bond(), ["H", "H"], cell=np.eye(3)*50, platform="Reference")
    with pytest.raises(PotentialError, match="lower-triangular"):
        OpenMMPotential(_periodic_model(), ["H", "H"], cell=np.eye(3)*50+.1, platform="Reference")
    with OpenMMPotential(_bond(), ["H", "H"], platform="Reference") as pes:
        with pytest.raises(PotentialError, match="atom count"):
            pes.evaluate(np.zeros((1, 3, 3)))
        with pytest.raises(PotentialError, match="identities/order"):
            pes.bind_system({"symbols": ["H", "O"]})
        with pytest.raises(PotentialError, match="full periodicity"):
            pes.set_cell(np.eye(3)*50)


def test_openmm_gpu_requests_are_not_silently_routed_to_cpu():
    with pytest.raises(ValueError, match="CUDA platform"):
        OpenMMPotential(_bond(), ["H", "H"], device="cuda:0", platform="CPU")
    with pytest.raises(ValueError, match="DeviceIndex conflicts"):
        OpenMMPotential(_bond(), ["H", "H"], device="cuda:1", properties={"DeviceIndex": "0"})
    with pytest.raises(ValueError, match="CPU or Reference"):
        OpenMMPotential(_bond(), ["H", "H"], device="cpu", platform="CUDA")


@pytest.mark.parametrize("wrapped", [False, True])
def test_bound_openmm_constraints_cannot_be_dropped_by_direct_dynamics(wrapped):
    from pyrpmd.constants import AMU_TO_AU
    from pyrpmd.dynamics import Dynamics, State
    from pyrpmd.potentials.cache import StepEvaluationCache
    from pyrpmd.system import System

    length = .15/BOHR_NM
    system = System(("H", "H"), np.ones(2)*AMU_TO_AU,
                    np.array([[0., 0, 0], [length, 0, 0]]), bond_constraints=[(0, 1, length)])
    with OpenMMPotential(_bond(constraint=True), system.symbols, platform="Reference") as pes:
        pes.bind_system(system)
        provider = StepEvaluationCache(pes) if wrapped else pes
        with pytest.raises(ValueError, match="constraint"):
            Dynamics(provider, system.masses, 1000., 1, .1, integrator="velocity-verlet")
        with pytest.raises(ValueError, match="constraint"):
            Dynamics(provider, system.masses, 1000., 1, .1, integrator="velocity-verlet",
                     bond_constraints=[(0, 1, length*1.01)])
        dynamics = Dynamics(provider, system.masses, 1000., 1, .1, integrator="velocity-verlet",
                            bond_constraints=system.bond_constraints)
        q = system.geometry[None].copy()
        state = State(q, dynamics.sample_momenta(q))
        dynamics.step(state)
        dynamics.constraints.check(state.q, state.p)
