"""Independent conservation, tangent-Gaussian and recovery gates for RATTLE."""
import copy

import numpy as np
import pytest

from pyrpmd.bond_constraints import BondConstraints, ConstraintError, normalize_bond_constraints
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import InputError, JobSpec
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.system import System
from pyrpmd.workflows.materials import _sample_replica, run_materials, resolve_materials


def dimer(beads=3, dt=0.003, potential=None, **kwargs):
    dynamics = Dynamics(potential or FreePotential(), [2., 3.], 5., beads, dt,
                        integrator="velocity-verlet", native=False,
                        bond_constraints=[(0, 1, 1.)], **kwargs)
    q = np.zeros((beads, 2, 3))
    for bead in range(beads):
        angle = 0.07 * bead
        q[bead, 1] = [np.cos(angle), np.sin(angle), 0]
    return dynamics, State(q, dynamics.sample_momenta(q))


@pytest.mark.parametrize("beads", [1, 3, 8])
def test_bond_tangent_energy_and_time_reversal(beads):
    dynamics, state = dimer(beads=beads)
    initial = state.copy()
    energy = dynamics.energy(state)
    momentum = state.p.sum(axis=(0, 1))
    for _ in range(80):
        dynamics.step(state)
    np.testing.assert_allclose(np.linalg.norm(state.q[:, 0] - state.q[:, 1], axis=1), 1, atol=2e-12)
    assert dynamics.constraints.residuals(state.q, state.p)[1] < 1.1e-12
    np.testing.assert_allclose(state.p.sum(axis=(0, 1)), momentum, atol=3e-13)
    assert abs(dynamics.energy(state) - energy) / max(1., abs(energy)) < 1e-5
    state.p *= -1
    for _ in range(80):
        dynamics.step(state)
    np.testing.assert_allclose(state.q, initial.q, rtol=0, atol=3e-10)
    np.testing.assert_allclose(state.p, -initial.p, rtol=0, atol=3e-10)


def test_full_ring_energy_error_is_second_order_under_timestep_halving():
    errors = []
    for dt in (.01, .005, .0025):
        dynamics, state = dimer(beads=8, dt=dt)
        energy = dynamics.energy(state)
        for _ in range(round(.4 / dt)):
            dynamics.step(state)
        errors.append(abs(dynamics.energy(state) - energy))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_rigid_water_triangle_preserves_angular_momentum_and_all_three_distances():
    q = np.array([[[0., 0., 0.], [1., 0., 0.], [-0.25, np.sqrt(15)/4, 0.]]])
    bonds = [(0, 1, 1.), (0, 2, 1.), (1, 2, np.sqrt(2.5))]
    dynamics = Dynamics(FreePotential(), [16., 1., 1.], 4., 1, .002,
                        native=False, integrator="velocity-verlet", bond_constraints=bonds)
    state = State(q, dynamics.sample_momenta(q))
    angular = np.cross(state.q, state.p).sum(axis=(0, 1))
    for _ in range(100):
        dynamics.step(state)
    assert max(dynamics.constraints.residuals(state.q, state.p)) < 1.1e-12
    np.testing.assert_allclose(np.cross(state.q, state.p).sum(axis=(0, 1)), angular, atol=1e-10)


def test_velocity_projection_matches_independent_mass_metric_linear_algebra():
    masses = np.array([16., 1., 2.])
    q = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    constraints = BondConstraints(masses, [(0, 1, 1.), (0, 2, 1.), (1, 2, np.sqrt(2))])
    rng = np.random.default_rng(19)
    p = rng.normal(size=q.shape)
    gradients = []
    for i, j, _ in constraints.bonds:
        gradient = np.zeros((3, 3)); gradient[i] = q[0, i] - q[0, j]; gradient[j] = -gradient[i]
        gradients.append(gradient.ravel())
    jacobian = np.array(gradients)
    inverse_mass = np.diag(np.repeat(1 / masses, 3))
    expected = p.ravel() - jacobian.T @ np.linalg.solve(
        jacobian @ inverse_mass @ jacobian.T, jacobian @ inverse_mass @ p.ravel())
    constraints.project_momenta(q, p)
    np.testing.assert_allclose(p.ravel(), expected, atol=2e-12)


def test_projected_maxwell_draw_uses_constrained_degrees_of_freedom():
    # Vectorize independent draws as P=10000 beads and compare with the exact
    # chi-square mean (3N-1)P/(2 beta). Projection removes one bond DOF per bead.
    dynamics, state = dimer(beads=10000, dt=1e-5)
    kinetic = np.sum(state.p**2 / (2 * np.array([2., 3.])[None, :, None]))
    expected = 5 * dynamics.nbeads**2 / (2 * dynamics.beta)
    assert kinetic == pytest.approx(expected, rel=.02)


def test_periodic_cross_boundary_bond_and_atomic_failure():
    dynamics, state = dimer(beads=1, constraint_cell=np.diag([8., 9., 10.]), constraint_pbc=[True]*3)
    state.q[0, 0] = [7.5, 0, 0]; state.q[0, 1] = [0.5, 0, 0]
    state.p[:] = dynamics.sample_momenta(state.q)
    dynamics.step(state)
    assert dynamics.constraints.residuals(state.q, state.p)[0] < 1e-12
    before = state.copy()
    calls = 0
    original = dynamics.potential.evaluate
    def fail_second(q):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("physical force failure")
        return original(q)
    dynamics.potential.evaluate = fail_second
    with pytest.raises(RuntimeError, match="physical force failure"):
        dynamics.step(state)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert (state.step, state.time) == (before.step, before.time)


def test_initial_projection_and_nonconvergence_are_transactional():
    dynamics, state = dimer(beads=1)
    state.q[0, 1, 0] += .04
    with pytest.raises(ConstraintError):
        dynamics.step(state)
    dynamics.initialize_constraints(state)
    dynamics.constraints.check(state.q, state.p)
    bad = Dynamics(FreePotential(), [1., 1., 1.], 4., 1, .01, native=False,
                   integrator="velocity-verlet", constraint_max_iterations=2,
                   bond_constraints=[(0, 1, 1.), (0, 2, 1.), (1, 2, 3.)])
    initial = State(np.array([[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]]), np.zeros((1,3,3)))
    before = initial.copy()
    with pytest.raises(ConstraintError):
        bad.initialize_constraints(initial)
    np.testing.assert_array_equal(initial.q, before.q)
    np.testing.assert_array_equal(initial.p, before.p)


@pytest.mark.parametrize("bonds", [None, [{"atoms":[0,0],"length":1.}], [(False,1,1.)],
                                         [(0,1,0.)], [(0,1,float("nan"))], [(0,1,1.),(1,0,1.)]])
def test_invalid_constraint_schema(bonds):
    with pytest.raises((ValueError, TypeError)):
        normalize_bond_constraints(bonds, 2)


def test_system_serialization_and_method_rejections():
    system = System(("H", "D"), np.array([2., 3.]), np.array([[0.,0.,0.],[1.,0.,0.]]),
                    bond_constraints=[{"atoms":[0,1],"length":"1 bohr"}])
    assert System.from_dict(system.to_dict()).to_dict() == system.to_dict()
    with pytest.raises(ValueError, match="velocity-verlet"):
        Dynamics(FreePotential(), [2.,3.], 4., 1, .01, bond_constraints=system.bond_constraints)
    with pytest.raises(ValueError, match="current bead positions"):
        dimer()[0].sample_momenta()
    cfg = material_config()
    cfg["barostat"] = {"method":"bzp"}
    with pytest.raises(ValueError, match="fixed cell"):
        resolve_materials(cfg, system)


def material_config():
    return {"task":"pimd", "beta":5., "beads":3, "dt_au":.003, "seed":2,
            "potential":{"provider":"builtin", "name":"harmonic", "params":{"k":.1}},
            "dynamics":{"integrator":"velocity-verlet", "loop_backend":"auto"},
            "sampling":{"thermostat":"andersen", "equilibration_steps":2,
                        "production_steps":12, "sample_interval":2, "replicas":1},
            "output":{"checkpoint_steps":3}, "native":False}


def test_constrained_materials_resume_and_no_unconstrained_quantum_estimators(tmp_path):
    cfg = material_config()
    system = System(("H","H"), np.array([2.,3.]), np.array([[0.,0.,0.],[1.,0.,0.]]),
                    bond_constraints=[(0,1,1.)])
    result = run_materials(cfg, system, tmp_path, return_data=True)
    assert set(result["summary"]["estimators"]) == {"potential_energy", "ring_polymer_hamiltonian"}
    assert result["summary"]["quantum_kinetic_estimators_available"] is False
    path = tmp_path / "checkpoints/materials_0000.npz"
    committed = path.read_bytes()
    resumed = run_materials(cfg, system, tmp_path, return_data=True)
    assert path.read_bytes() == committed
    assert result["summary"] == resumed["summary"]
    np.testing.assert_array_equal(result["streams"][0]["state"].q, resumed["streams"][0]["state"].q)
    assert BondConstraints(system.masses, system.bond_constraints).residuals(
        resumed["streams"][0]["state"].q, resumed["streams"][0]["state"].p)[0] < 1e-12


def test_public_input_accepts_bonds_and_refuses_unimplemented_workflows():
    mapping = {"task":"pimd", "temperature":"300 K", "beads":2,
               "system":{"symbols":["H","H"], "geometry":[[0,0,0],[1,0,0]],
                         "bond_constraints":[{"atoms":[0,1],"length":"1 bohr"}]},
               "potential":{"provider":"builtin","name":"free"},
               "dynamics":{"integrator":"velocity-verlet","timestep":"0.05 fs"},
               "sampling":{"production_steps":10,"sample_interval":2}}
    resolved, system = JobSpec.from_mapping(mapping).resolve()
    assert system.bond_constraints == ((0, 1, 1.),)
    mapping["task"] = "fes"
    with pytest.raises(InputError, match="bond constraints"):
        JobSpec.from_mapping(mapping).resolve()
