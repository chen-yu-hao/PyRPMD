"""Committee ownership/provenance, deterministic acquisition, and Δ/RPC forces."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.input import _nested_potential_paths
from pyrpmd.io import referenced_files
from pyrpmd.potentials import FreePotential, HarmonicPotential, PESResult, make_potential
from pyrpmd.potentials.lifecycle import (
    ActiveLearningMonitor, CommitteePotential, DeltaLearningPotential, active_learning_options,
)
from pyrpmd.system import System
from pyrpmd.workflows.materials import _sample_replica


def committee():
    return CommitteePotential([HarmonicPotential(k=1.), HarmonicPotential(k=3.)])


def test_committee_statistics_match_independent_member_predictions():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(3, 2, 3))
    model = committee()
    result = model.evaluate_uncertainty(q)
    energies = np.array([member.evaluate(q).energy for member in model.members])
    forces = np.array([member.evaluate(q).forces for member in model.members])
    np.testing.assert_allclose(result.mean.energy, energies.mean(axis=0))
    np.testing.assert_allclose(result.mean.forces, forces.mean(axis=0))
    np.testing.assert_allclose(result.energy_std, energies.std(axis=0, ddof=0))
    expected = np.sqrt(np.sum(forces.var(axis=0, ddof=0), axis=-1))
    np.testing.assert_allclose(result.force_std, expected)
    np.testing.assert_allclose(result.max_force_std, expected.max(axis=1))


def test_all_committee_outputs_are_detached_from_cached_uncertainty():
    q = np.ones((2, 1, 3))
    model = committee()
    result = model.evaluate_uncertainty(q)
    expected = model.uncertainty_at(q)
    for array in [result.mean.energy, result.mean.forces, result.mean.status, result.energy_std,
                  result.force_std, result.max_force_std, result.member_energies]:
        array[:] = 123
    result2 = model.uncertainty_at(q)
    for name in ["energy_std", "force_std", "max_force_std", "member_energies"]:
        np.testing.assert_array_equal(getattr(result2, name), getattr(expected, name))
    np.testing.assert_array_equal(result2.mean.forces, expected.mean.forces)
    np.testing.assert_array_equal(result2.mean.status, expected.mean.status)
    model.evaluate(q).forces[:] = 456
    np.testing.assert_array_equal(model.uncertainty_at(q).mean.forces, expected.mean.forces)
    model.uncertainty_at(q).mean.forces[:] = 789
    np.testing.assert_array_equal(model.uncertainty_at(q).mean.forces, expected.mean.forces)


def test_committee_consumes_reused_member_buffers_before_next_call():
    energy, forces = np.empty(2), np.empty((2, 1, 3))
    class Member:
        def __init__(self, k):
            self.k = k
        def evaluate(self, q):
            energy[:] = self.k * np.sum(q*q, axis=(1, 2)) / 2
            forces[:] = -self.k*q
            return PESResult(energy, forces)
    q = np.ones((2, 1, 3))
    result = CommitteePotential([Member(1), Member(3)]).evaluate_uncertainty(q)
    np.testing.assert_array_equal(result.member_energies[:, 0], [1.5, 4.5])
    np.testing.assert_array_equal(result.mean.forces, -2.)
    np.testing.assert_allclose(result.force_std, np.sqrt(3))


def test_collection_is_thresholded_and_never_changes_the_force(tmp_path):
    model = committee()
    monitor = ActiveLearningMonitor(model, tmp_path, force_threshold=1., interval_steps=2,
                                    max_frames=10, model_identity="two-harmonic-models")
    low = np.full((2, 1, 3), .01)
    expected = model.evaluate(low).forces.copy()
    assert monitor.observe(low, step=1, time=.1) is None
    event = monitor.observe(low, step=2, time=.2)
    assert event["selected"] == 0
    np.testing.assert_array_equal(model.evaluate(low).forces, expected)
    high = np.array([[[2., 0., 0.]], [[0., 3., 0.]]])
    expected = model.evaluate(high).forces.copy()
    event = monitor.observe(high, step=4, time=.4)
    assert event["selected"] == 2
    with load_checkpoint(event["record"], require_integrity=True) as record:
        np.testing.assert_array_equal(record["positions"], high)
        np.testing.assert_array_equal(record["predicted_forces"], expected)
        assert not bool(record["labels_available"])
    np.testing.assert_array_equal(model.evaluate(high).forces, expected)


def test_collection_replay_is_shape_aware_idempotent_and_capped(tmp_path):
    model = committee()
    options = dict(force_threshold=.1, max_frames=3, model_identity="same-model")
    monitor = ActiveLearningMonitor(model, tmp_path, **options)
    q = np.ones((2, 1, 3))
    first = monitor.observe(q, step=1, time=.1)
    assert first["selected"] == 2 and monitor.collected == 2
    assert monitor.observe(q.copy(), step=1, time=.1) == first
    assert monitor.collected == 2
    with pytest.raises(ValueError, match="replayed"):
        monitor.observe(q.reshape(1, 2, 3), step=1, time=.1)
    resumed = ActiveLearningMonitor(model, tmp_path, **options)
    assert resumed.collected == 2
    assert resumed.observe(q, step=2, time=.2)["selected"] == 1
    assert resumed.observe(q, step=3, time=.3)["selected"] == 0
    assert resumed.collected == 3
    assert ActiveLearningMonitor.audit_records(tmp_path, expected_identity=resumed.identity,
                through_step=3, **options) == 3
    with pytest.raises(ValueError, match="different model"):
        ActiveLearningMonitor(model, tmp_path, force_threshold=.1, max_frames=3, model_identity="changed-model")


def test_owned_cleanup_attempts_every_member_and_preserves_first_error():
    closed = []
    class Member(FreePotential):
        def __init__(self, number):
            self.number = number
        def close(self):
            closed.append(self.number)
            raise RuntimeError(f"close {self.number}")
    for build in [lambda members: CommitteePotential(members, owns_members=True),
                  lambda members: DeltaLearningPotential(*members, owns_members=True)]:
        model = build([Member(1), Member(2)])
        with pytest.raises(RuntimeError, match="close 1"):
            model.close()
        assert closed[-2:] == [1, 2]
        count = len(closed)
        model.close()
        assert len(closed) == count


def test_delta_sum_and_full_bead_rpc_agree_and_contracted_force_is_chain_rule():
    reference, correction = HarmonicPotential(k=.2), HarmonicPotential(k=.7)
    model = DeltaLearningPotential(reference, correction)
    q = np.random.default_rng(47).normal(size=(4, 2, 3))
    expected = HarmonicPotential(k=.9).evaluate(q)
    direct = model.evaluate(q)
    full = model.split(4, 4).evaluate(q)
    np.testing.assert_allclose(direct.energy, expected.energy, atol=1e-14)
    np.testing.assert_allclose(direct.forces, expected.forces, atol=1e-14)
    np.testing.assert_allclose(full.energy, direct.energy, atol=1e-14)
    np.testing.assert_allclose(full.forces, direct.forces, atol=1e-14)
    split = model.split(4, 1)
    contracted = split.evaluate(q)
    energy = np.sum(reference.evaluate(q).energy) + 4 * correction.evaluate(q.mean(axis=0, keepdims=True)).energy[0]
    assert contracted.energy.sum() == pytest.approx(energy)
    assert split.metadata["approximate_hamiltonian"]
    assert not np.isclose(contracted.energy.sum(), direct.energy.sum())
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        derivative = -(split.evaluate(plus).energy.sum() - split.evaluate(minus).energy.sum()) / 2e-6
        assert contracted.forces[index] == pytest.approx(derivative, rel=2e-6, abs=2e-9)


def test_nested_relative_models_and_named_fairchem_identity(tmp_path):
    weights = tmp_path / "local.pt"
    weights.write_bytes(b"first model")
    spec = {"provider": "committee", "members": [
        {"provider": "fairchem", "model": "./local.pt"},
        {"provider": "delta", "reference": {"provider": "fairchem", "model": "uma-s-1p1"},
         "correction": {"provider": "torch", "model": "local.pt"}, "contracted_beads": 2},
    ]}
    _nested_potential_paths(spec, tmp_path, 4)
    assert spec["members"][0]["model"] == str(weights)
    assert spec["members"][1]["reference"]["model"] == "uma-s-1p1"
    assert spec["members"][1]["correction"]["model"] == str(weights)
    assert spec["members"][1]["nbeads"] == 4
    hashes = referenced_files({"potential": spec})
    assert str(weights) in hashes
    weights.write_bytes(b"changed model")
    assert referenced_files({"potential": spec})[str(weights)] != hashes[str(weights)]


def material_config(active=True):
    cfg = {"task": "pimd", "beta": 2., "beads": 2, "dt_au": .03, "seed": 41, "native": False,
           "potential": {"provider": "committee", "members": [
               {"provider": "builtin", "name": "harmonic", "params": {"k": .2}},
               {"provider": "builtin", "name": "harmonic", "params": {"k": .6}}]},
           "sampling": {"equilibration_steps": 2, "production_steps": 8, "sample_interval": 2,
                        "replicas": 1, "thermostat": "pile-l", "gamma_centroid": .6},
           "dynamics": {"loop_backend": "python"}, "output": {"checkpoint_steps": 2, "trajectory_storage": "memory"}}
    if active:
        cfg["active_learning"] = {"force_threshold": 1e-6, "interval_steps": 2, "max_frames": 5}
    return cfg


def test_materials_monitor_leaves_dynamics_identical_and_completed_resume_is_offline(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as materials

    system = System(("H",), np.array([1.]), np.array([[.2, .1, 0.]]))
    def task(active):
        return {"config": material_config(active), "system": system, "output": str(tmp_path / str(active)), "replica": 0}
    plain = _sample_replica(task(False))
    active = _sample_replica(task(True))
    for key in ("positions", "velocities", "observables", "sample_steps"):
        np.testing.assert_array_equal(plain[key], active[key])
    assert plain["rng"] == active["rng"]
    records = sorted((tmp_path / "True" / "active_learning" / "replica_0000").glob("step_*.npz"))
    assert len(records) == 5
    def unavailable(*args, **kwargs):
        raise AssertionError("completed resume must not reconstruct a model")
    monkeypatch.setattr(materials, "make_task_potential", unavailable)
    complete = _sample_replica(task(True))
    np.testing.assert_array_equal(active["positions"], complete["positions"])
    records[0].unlink()
    with pytest.raises(ValueError, match="missing committed step"):
        _sample_replica(task(True))


@pytest.mark.parametrize("section", ["path_integral", "plumed", "barostat"])
def test_active_learning_cannot_be_silently_ignored_by_another_engine(section):
    cfg = material_config()
    cfg[section] = {}
    with pytest.raises(ValueError, match="ordinary PIMD"):
        active_learning_options(cfg)


def test_factory_delta_provides_explicit_full_and_contracted_hamiltonians():
    def spec(k):
        return {"provider": "builtin", "name": "harmonic", "params": {"k": k}}
    full = make_potential({"provider": "delta", "reference": spec(.2), "correction": spec(.7)})
    contracted = make_potential({"provider": "delta", "reference": spec(.2), "correction": spec(.7),
                                 "nbeads": 4, "contracted_beads": 1})
    q = np.random.default_rng(14).normal(size=(4, 1, 3))
    np.testing.assert_allclose(full.evaluate(q).forces, -.9 * q)
    np.testing.assert_allclose(contracted.evaluate(q).forces, -.2*q - .7*q.mean(axis=0))
    full.close(); contracted.close()


def test_rpc_estimators_use_the_contracted_partition_function_and_full_ring(tmp_path):
    from pyrpmd.dynamics import State
    from pyrpmd.workflows.materials import quantum_observables, run_materials

    q = np.random.default_rng(87).normal(size=(4, 2, 3))
    beta, masses = 2.3, np.array([1.2, 2.8])
    split = DeltaLearningPotential(HarmonicPotential(k=.2), HarmonicPotential(k=.7)).split(4, 1)
    values, _ = quantum_observables(State(q, np.zeros_like(q)), split, masses, beta)
    centroid = q.mean(axis=0)
    offsets = q - centroid
    # The correction acts only on the centroid; it contributes no internal-mode
    # virial. This differs from a full-bead .9*q^2/2 target potential.
    expected_u = .1 * np.sum(q*q) / 4 + .35 * np.sum(centroid*centroid)
    expected_k = 3 * len(masses) / (2*beta) + .2 * np.sum(offsets*offsets) / 8
    assert values[0] == pytest.approx(expected_u)
    assert values[2] == pytest.approx(expected_k)
    assert values[3] == pytest.approx(expected_u + expected_k)
    assert not np.isclose(values[0], .45 * np.sum(q*q) / 4)

    cfg = material_config(False)
    cfg["potential"] = {"provider": "delta", "reference": cfg["potential"]["members"][0],
                        "correction": cfg["potential"]["members"][1], "nbeads": 2, "contracted_beads": 1}
    result = run_materials(cfg, System(("H",), np.array([1.]), np.array([[.2, .1, 0.]])), tmp_path)
    assert result["estimator_scope"] == "rpc_contracted_hamiltonian"
    assert all(estimate["scope"] == "rpc_contracted_hamiltonian" for estimate in result["estimators"].values())
    assert result["ring_polymer_contraction"]["approximate_hamiltonian"]
    assert not result["ring_polymer_contraction"]["full_target_recomputed"]
    with load_checkpoint(result["files"]["trajectories"][0]) as trajectory:
        metadata = json.loads(str(trajectory["metadata"]))
        assert metadata["estimator_scope"] == result["estimator_scope"]
        assert metadata["ring_polymer_contraction"] == result["ring_polymer_contraction"]


@pytest.mark.parametrize("kind", ["committee", "delta"])
def test_nested_unknown_potential_fields_fail_before_any_model_is_constructed(tmp_path, kind):
    from pyrpmd.input import InputError, JobSpec
    from pyrpmd.potentials import PotentialError

    bad = {"provider": "harmonic", "params": {"k": .1}, "ignored_typo": True}
    valid = {"provider": "harmonic", "params": {"k": .2}}
    nested = ({"provider": "committee", "members": [valid, bad]} if kind == "committee" else
              {"provider": "delta", "reference": valid, "correction": bad})
    spec = {"provider": "committee", "members": [
        {"provider": "python", "factory": "must_not_import_before_validation:build"}, nested]}
    with pytest.raises(PotentialError, match="Unknown.*ignored_typo"):
        make_potential(spec)
    with pytest.raises(InputError, match="Unknown.*ignored_typo"):
        JobSpec.from_mapping({"task": "pimd", "temperature": "300 K", "beads": 4,
            "system": {"symbols": ["H"], "geometry": [[0., 0., 0.]]}, "potential": spec}).resolve()


def test_nested_model_specific_params_remain_opaque(tmp_path, monkeypatch):
    import pyrpmd.potentials as potentials

    seen = []
    def factory(**params):
        seen.append(params)
        return FreePotential()
    monkeypatch.setattr(potentials, "import_object", lambda name: factory)
    member = {"provider": "python", "factory": "plugin:build", "params": {"model_specific_setting": {"a": 4}}}
    spec = {"provider": "committee", "members": [copy.deepcopy(member), copy.deepcopy(member)]}
    _nested_potential_paths(spec, tmp_path, 4)
    model = make_potential(spec)
    assert seen == [{"model_specific_setting": {"a": 4}}] * 2
    model.close()
