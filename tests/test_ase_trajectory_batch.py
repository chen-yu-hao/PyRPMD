"""Explicit ASE frame-independence contracts and real periodic batch results."""

import copy

import numpy as np
import pytest

from pyrpmd.batched import validate_batch_potential
from pyrpmd.input import JobSpec
from pyrpmd.potentials import PotentialError, make_potential, validate_periodic_spec
from pyrpmd.potentials.optional import ASEPotential
from pyrpmd.trajectory_batch import make_trajectory_potential, run_trajectory_batch


def _system():
    return {"symbols": ["Ar", "Ar"], "masses_amu": [39.948, 39.948],
            "geometry": [[0., 0., 0.], [3.7, .2, .1]],
            "geometry_unit": "angstrom", "cell": np.eye(3).tolist(),
            "cell_unit": "nm", "pbc": [True, True, True]}


def _job():
    return {"task": "pimd", "temperature": "100 K", "beads": 2,
            "system": _system(),
            "potential": {"provider": "ase", "factory": "ase.calculators.lj:LennardJones",
                          "params": {"sigma": 3.4, "epsilon": .0103, "rc": 4.5},
                          "trajectory_batch_safe": True},
            "sampling": {"replicas": 2, "equilibration_steps": 0,
                         "production_steps": 4, "sample_interval": 1},
            "parallel": {"trajectory_batch_size": 2}}


def _request(task):
    provider = make_trajectory_potential(task)
    result = provider.evaluate(task["q"])
    provider.close()  # A lane borrows, and must not close another lane's PES.
    return result


def test_public_ase_opt_in_preserves_real_periodic_energy_force_virial():
    pytest.importorskip("ase")
    config, system = JobSpec.from_mapping(_job()).resolve()
    with make_potential(config["potential"], system) as provider:
        validate_batch_potential(provider)
        assert provider.metadata["native_batch"] is False
        q = np.repeat(system.geometry[None], 3, axis=0)
        q[1, 1, 0] += .12
        q[2, 0, 1] += .17
        expected = provider.evaluate(q)
        permutation = [2, 0, 1]
        permuted = provider.evaluate(q[permutation])
        for name in ("energy", "forces", "virial"):
            np.testing.assert_array_equal(getattr(permuted, name), getattr(expected, name)[permutation])
        stats = {}
        outputs = run_trajectory_batch(_request, [{"q": q[:2]}, {"q": q[2:]}], provider, stats=stats)
        assert stats == {"pes_calls": 1, "pes_frames": 3}
        for name in ("energy", "forces", "virial"):
            np.testing.assert_array_equal(np.concatenate([getattr(r, name) for r in outputs]),
                                          getattr(expected, name))


def test_ase_without_a_declaration_is_not_assumed_batch_safe():
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    with ASEPotential(["Ar"], LennardJones()) as provider:
        with pytest.raises(ValueError, match="trajectory_batch_safe"):
            validate_batch_potential(provider)


@pytest.mark.parametrize("declaration", [
    {"trajectory_batch_safe": False}, {"batch_independent": False},
    {"stateful": True}, {"capabilities": {"stateful": True}},
    {"coupled_frames": True}, {"ring_polymer_coupled": True},
])
def test_explicit_ase_opt_in_cannot_hide_a_calculator_veto(declaration):
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    calculator = LennardJones()
    calculator.metadata = declaration
    with pytest.raises(ValueError, match="explicitly rejects"):
        ASEPotential(["Ar"], calculator, trajectory_batch_safe=True)
    with ASEPotential(["Ar"], calculator) as provider:
        provider.metadata = {**provider.metadata, "trajectory_batch_safe": True}
        with pytest.raises(ValueError, match="independent|stateless"):
            validate_batch_potential(provider)


def test_calculator_declaration_is_instance_local_and_can_be_disabled():
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones
    from pyrpmd.potentials.adapters import ModelASEPotential

    calculator = LennardJones()
    calculator.metadata = {"trajectory_batch_safe": True, "reuse_across_tasks": False}
    original = copy.deepcopy(ASEPotential.metadata)
    with ModelASEPotential(["Ar"], calculator, provider="test_model") as provider:
        validate_batch_potential(provider)
        assert provider.metadata["reuse_across_tasks"] is False
        assert provider.metadata["provider"] == "test_model"
    assert ASEPotential.metadata == original
    with ASEPotential(["Ar"], LennardJones(), trajectory_batch_safe=False) as provider:
        with pytest.raises(ValueError, match="independent, stateless"):
            validate_batch_potential(provider)


@pytest.mark.parametrize("depth", [1, 2])
def test_transparent_python_adapters_cannot_hide_a_live_calculator_veto(depth):
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones
    from pyrpmd.potentials.python import PythonPotential

    calculator = LennardJones()
    provider = ASEPotential(["Ar"], calculator, trajectory_batch_safe=True)
    wrapped = provider
    for _ in range(depth):
        wrapped = PythonPotential(wrapped)
    calculator.metadata = {"stateful": True}
    with pytest.raises(ValueError, match="explicitly rejects"):
        validate_batch_potential(wrapped)
    provider.close()


def test_batch_contract_does_not_change_physics_or_strip_opaque_parameters():
    from pyrpmd.propagation import scientific_config

    plain = {"provider": "ase", "factory": "module:calculator",
             "params": {"trajectory_batch_safe": "a model-specific parameter"}}
    declared = {**plain, "trajectory_batch_safe": True}
    baseline = {"potential": {"provider": "delta", "reference": plain, "correction": plain}}
    accelerated = {"potential": {"provider": "delta", "reference": declared, "correction": declared},
                   "parallel": {"trajectory_batch_size": 4}}
    assert scientific_config(accelerated) == baseline
    assert accelerated["potential"]["reference"]["trajectory_batch_safe"] is True


def test_existing_scalar_ase_checkpoint_can_add_the_contract_and_resume_batched(tmp_path, monkeypatch):
    pytest.importorskip("ase")
    from pyrpmd import run
    from pyrpmd.dynamics import Dynamics
    from pyrpmd.workflows.materials import _read_stream

    config = _job()
    reference = tmp_path / "reference"
    run(config, output=reference)
    old = copy.deepcopy(config)
    old["potential"].pop("trajectory_batch_safe")
    old["parallel"]["trajectory_batch_size"] = 1
    real_step = Dynamics.sample_step

    def stop(self, state, *args, **kwargs):
        result = real_step(self, state, *args, **kwargs)
        if state.step == 2:
            raise RuntimeError("interrupt old scalar ASE run")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "sample_step", stop)
        with pytest.raises(RuntimeError, match="old scalar ASE"):
            run(old, output=tmp_path / "continued")
    run(config, output=tmp_path / "continued", resume=True)
    resolved, system = JobSpec.from_mapping(config).resolve()
    for replica in range(2):
        expected = _read_stream(resolved, system, reference, replica)
        actual = _read_stream(resolved, system, tmp_path / "continued", replica)
        for field in ("q", "p", "step", "time"):
            np.testing.assert_array_equal(getattr(actual["state"], field), getattr(expected["state"], field))
        assert actual["rng"] == expected["rng"]
        np.testing.assert_array_equal(actual["observables"], expected["observables"])


@pytest.mark.parametrize("value", [1, "true", [], None])
def test_declaration_type_is_rejected_before_constructing_an_engine(value):
    with pytest.raises(PotentialError, match="boolean"):
        validate_periodic_spec({"provider": "ase", "factory": "missing:factory",
                                "trajectory_batch_safe": value})


@pytest.mark.parametrize("kind", ["builtin", "python", "torch", "openmm", "lammps"])
def test_ase_config_declaration_is_not_silently_accepted_by_other_providers(kind):
    with pytest.raises(PotentialError, match="ASE-calculator contract"):
        validate_periodic_spec({"provider": kind, "trajectory_batch_safe": True})


def test_lane_preserves_required_constraints_without_exposing_shared_mutation():
    from pyrpmd.potentials import HarmonicPotential

    provider = HarmonicPotential(.4)
    provider.required_bond_constraints = [(0, 1, 2.)]

    def constraints(task):
        lane = make_trajectory_potential(task)
        value = copy.deepcopy(lane.required_bond_constraints)
        lane.required_bond_constraints.append((1, 2, 3.))
        return value

    result = run_trajectory_batch(constraints, [{}, {}], provider)
    assert result == [[(0, 1, 2.)], [(0, 1, 2.)]]
    assert provider.required_bond_constraints == [(0, 1, 2.)]
