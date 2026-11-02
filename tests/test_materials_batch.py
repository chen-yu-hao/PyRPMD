"""Materials replicas share physical PES calls, never state or sampling clocks."""

import threading

import numpy as np
import pytest

from pyrpmd import run
from pyrpmd.bond_constraints import BondConstraints
from pyrpmd.input import InputError, JobSpec
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.builtin import FreePotential, PeriodicCosinePotential
from pyrpmd.potentials.lifecycle import DeltaLearningPotential
from pyrpmd.system import System
from pyrpmd.workflows.materials import (
    _read_stream, _sample_replica, resolve_materials, run_materials,
)


def system(*, fixed=False, constrained=False):
    return System(
        ("H", "Ni"), np.array([2., 20.]),
        np.array([[.1, .2, .3], [1.1, .2, .3]]),
        cell=np.array([[6., 0., 0.], [.5, 7., 0.], [0., 0., 12.]]),
        pbc=(True, True, False), fixed_atoms=(1,) if fixed else (),
        bond_constraints=((0, 1, 1.),) if constrained else (),
    )


def config(task="pimd", thermostat="pile-l", *, replicas=5):
    return {
        "task": task, "beta": 2., "beads": 4, "dt_au": .015,
        "seed": 1739, "native": False,
        "potential": {"provider": "builtin", "name": "periodic_cosine",
                      "params": {"height": [.2, .1, 0.], "confinement": .05}},
        "sampling": {"equilibration_steps": 3, "production_steps": 18,
                     "sample_interval": 2, "replicas": replicas,
                     "gamma_centroid": .6, "thermostat": thermostat,
                     "andersen_interval": 3},
        "dynamics": {"loop_backend": "python"},
        "diffusion": {"atom_indices": [0], "axes": [0, 1], "bootstrap": 0},
        "parallel": {"trajectory_workers": 1, "trajectory_batch_size": 1},
        "output": {"checkpoint_steps": 4, "chunk_frames": 3,
                   "trajectory_storage": "stream"},
    }


def assert_same_stream(left, right):
    for key in ("positions", "velocities", "observables", "radii_gyration_squared",
                "sample_steps", "times"):
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    for key in ("q", "p"):
        np.testing.assert_array_equal(getattr(left["state"], key), getattr(right["state"], key))
    assert left["state"].step == right["state"].step
    assert left["state"].time == right["state"].time
    assert left["rng"] == right["rng"]
    assert left["propagation"] == right["propagation"]


def assert_same_replicas(left, right):
    assert [s["replica"] for s in left["streams"]] == [s["replica"] for s in right["streams"]]
    for a, b in zip(left["streams"], right["streams"]):
        assert_same_stream(a, b)
    for key in ("estimators", "sampling_diagnostics", "diffusion", "analysis_status",
                "nve_energy_diagnostics", "electronic_friction"):
        if key in left["summary"]:
            assert left["summary"][key] == right["summary"][key]


class CountingPotential:
    """Instrument a real analytic evaluator without changing its frame semantics."""

    def __init__(self, potential, *, fail_call=None, metadata=None):
        self.potential = potential
        self.metadata = {**potential.metadata, "trajectory_batch_safe": True,
                         **(metadata or {})}
        self.shapes = []
        self.fail_call = fail_call
        self.closed = False

    def evaluate(self, q):
        assert not self.closed, "a finished lane closed the shared physical provider"
        self.shapes.append(q.shape)
        result = self.potential.evaluate(q)
        if len(self.shapes) == self.fail_call:
            result.status[0] = 7
            result.energy[0] = np.nan
            result.forces[0] = np.nan
            result.messages = ("injected frame failure",) + result.messages[1:]
        return result

    def close(self):
        self.closed = True


def instrument(monkeypatch, *, fail_call=None, metadata=None, rpc=False):
    import pyrpmd.parallel as parallel
    import pyrpmd.parallel.execution as execution
    import pyrpmd.workflows.materials as materials

    instances = []

    def factory(spec, atoms):
        physical = PeriodicCosinePotential(atoms.cell, atoms.pbc, **(
            spec["correction"]["params"] if rpc else spec["params"]))
        counted = CountingPotential(physical, fail_call=fail_call, metadata=metadata)
        instances.append(counted)
        if not rpc:
            return counted
        fast = CountingPotential(FreePotential())
        instances.append(fast)
        owner = DeltaLearningPotential(fast, counted, owns_members=True)
        split = owner.split(spec["nbeads"], spec["contracted_beads"])
        # Mirror make_potential's ownership transfer, including its instance
        # close callback: borrowed RPC lanes must never copy this ownership.
        split.close = owner.close
        return split

    monkeypatch.setattr(parallel, "make_task_potential", factory)
    monkeypatch.setattr(execution, "make_task_potential", factory)
    monkeypatch.setattr(materials, "make_task_potential", factory)
    return instances


def test_public_materials_batch_combines_calls_and_reduces_dispatch(tmp_path, monkeypatch):
    instances = instrument(monkeypatch)
    mapping = {
        "task": "pimd", "temperature": "300 K", "beads": 4, "seed": 1739,
        "native": False,
        "system": {"symbols": ["H", "Ni"], "geometry": system().geometry.tolist(),
                   "cell": system().cell.tolist(), "pbc": [True, True, False]},
        "potential": config()["potential"],
        "dynamics": {"timestep": "0.1 fs", "loop_backend": "python"},
        "sampling": {"equilibration_steps": 3, "production_steps": 18,
                     "sample_interval": 2, "replicas": 4},
        "parallel": {"trajectory_workers": 1, "trajectory_batch_size": 1},
    }
    run(mapping, output=tmp_path / "scalar")
    scalar = [shape for p in instances for shape in p.shapes]
    instances.clear()
    mapping["parallel"]["trajectory_batch_size"] = 4
    result = run(mapping, output=tmp_path / "batch")
    batched = [shape for p in instances for shape in p.shapes]
    assert result["replicas"] == 4
    assert set(scalar) == {(4, 2, 3)}
    assert set(batched) == {(16, 2, 3)}
    assert len(scalar) == 4 * len(batched)
    assert sum(s[0] for s in scalar) == sum(s[0] for s in batched)


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
@pytest.mark.parametrize("thermostat", ["pile-l", "andersen"])
@pytest.mark.parametrize("workers", [1, 2])
def test_batches_preserve_replica_rng_streams_and_process_results(tmp_path, task, thermostat, workers):
    cfg, atoms = config(task, thermostat), system()
    reference = run_materials(cfg, atoms, tmp_path / "scalar", return_data=True)
    cfg["parallel"].update(backend="process", trajectory_workers=workers, trajectory_batch_size=2)
    actual = run_materials(cfg, atoms, tmp_path / "batch", return_data=True)
    assert_same_replicas(reference, actual)
    # Five replicas include a one-lane tail; IDs and independent streams must
    # not be replaced by process slots or batch-local lane indices.
    assert len({json_rng(s["rng"]) for s in actual["streams"]}) == 5


def json_rng(value):
    import json
    return json.dumps(value, sort_keys=True)


@pytest.mark.parametrize("initial_batch,resume_batch", [(1, 3), (2, 1), (2, 3)])
def test_partial_and_completed_replicas_resume_across_batch_layouts(
        tmp_path, monkeypatch, initial_batch, resume_batch):
    import pyrpmd.workflows.materials as materials

    cfg, atoms = config("diffusion"), system()
    reference = run_materials(cfg, atoms, tmp_path / "reference", return_data=True)
    cfg["parallel"]["trajectory_batch_size"] = initial_batch
    original = materials.quantum_observables
    lock, failed = threading.Lock(), False

    def fail_one_observation(state, *args, **kwargs):
        nonlocal failed
        result = original(state, *args, **kwargs)
        with lock:
            if state.step == 7 and not failed:
                failed = True
                raise RuntimeError("injected material observation failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(materials, "quantum_observables", fail_one_observation)
        with pytest.raises(RuntimeError, match="material observation failure"):
            run_materials(cfg, atoms, tmp_path / "resume")
    prefixes = [_read_stream(cfg, atoms, tmp_path / "resume", int(p.stem.split("_")[-1]), materialize=False)
                for p in sorted((tmp_path / "resume/checkpoints").glob("materials_*.npz"))]
    assert failed and any(s["elapsed"] == 6 for s in prefixes)
    if initial_batch > 1:
        assert any(s["complete"] for s in prefixes), "the healthy lane did not finish"
    cfg["parallel"]["trajectory_batch_size"] = resume_batch
    cfg["output"].update(checkpoint_steps=7, chunk_frames=2)
    resumed = run_materials(cfg, atoms, tmp_path / "resume", return_data=True)
    assert_same_replicas(reference, resumed)


def test_one_failed_force_frame_rolls_back_only_its_lane(tmp_path, monkeypatch):
    cfg, atoms = config(replicas=3), system()
    reference = run_materials(cfg, atoms, tmp_path / "reference", return_data=True)
    cfg["parallel"]["trajectory_batch_size"] = 3
    with monkeypatch.context() as patch:
        instances = instrument(patch, fail_call=14)
        with pytest.raises(PotentialError, match="injected frame failure"):
            run_materials(cfg, atoms, tmp_path / "resume")
    assert instances[0].closed
    prefixes = [_read_stream(cfg, atoms, tmp_path / "resume", i, materialize=False) for i in range(3)]
    assert prefixes[0]["elapsed"] == 6
    assert all(s["complete"] for s in prefixes[1:])
    # Returning to scalar changes execution only, preserving the failed lane's
    # RNG draws and every healthy lane's already committed immutable prefix.
    committed = [(tmp_path / f"resume/checkpoints/materials_{i:04d}.npz").read_bytes() for i in (1, 2)]
    cfg["parallel"]["trajectory_batch_size"] = 1
    resumed = run_materials(cfg, atoms, tmp_path / "resume", return_data=True)
    assert_same_replicas(reference, resumed)
    assert committed == [(tmp_path / f"resume/checkpoints/materials_{i:04d}.npz").read_bytes() for i in (1, 2)]


@pytest.mark.parametrize("migrate", [False, True])
def test_completed_batch_resume_never_constructs_a_potential(tmp_path, monkeypatch, migrate):
    import pyrpmd.parallel as parallel
    import pyrpmd.parallel.execution as execution
    import pyrpmd.workflows.materials as materials

    cfg, atoms = config(replicas=3), system()
    cfg["parallel"]["trajectory_batch_size"] = 2
    if migrate:
        cfg["output"]["trajectory_storage"] = "memory"
    first = run_materials(cfg, atoms, tmp_path, return_data=True)
    checkpoints = {p: p.read_bytes() for p in (tmp_path / "checkpoints").glob("*.npz")}
    cfg["parallel"]["trajectory_batch_size"] = 3
    cfg["output"]["trajectory_storage"] = "stream"

    def unavailable(*args, **kwargs):
        pytest.fail("completed replica replay attempted to construct a potential")

    monkeypatch.setattr(parallel, "make_task_potential", unavailable)
    monkeypatch.setattr(execution, "make_task_potential", unavailable)
    monkeypatch.setattr(materials, "make_task_potential", unavailable)
    again = run_materials(cfg, atoms, tmp_path, return_data=True)
    assert_same_replicas(first, again)
    if not migrate:
        assert checkpoints == {p: p.read_bytes() for p in checkpoints}


@pytest.mark.parametrize("physical", ["fixed_atoms", "bond_constraints", "electronic_friction"])
def test_material_constraints_and_physical_baths_remain_lane_local(tmp_path, physical):
    cfg = config("diffusion", replicas=3)
    atoms = system(fixed=physical in ("fixed_atoms", "electronic_friction"),
                   constrained=physical == "bond_constraints")
    if physical == "bond_constraints":
        cfg["sampling"]["thermostat"] = "andersen"
        cfg["dynamics"]["integrator"] = "velocity-verlet"
        cfg["dt_au"] = .003
    if physical == "electronic_friction":
        cfg["electronic_friction"] = {"gamma": .7}
    reference = run_materials(cfg, atoms, tmp_path / "scalar", return_data=True)
    cfg["parallel"]["trajectory_batch_size"] = 2
    actual = run_materials(cfg, atoms, tmp_path / "batch", return_data=True)
    assert_same_replicas(reference, actual)
    for stream in actual["streams"]:
        state = stream["state"]
        if atoms.fixed_atoms:
            np.testing.assert_array_equal(state.p[:, 1], 0)
            np.testing.assert_array_equal(state.q[:, 1], np.broadcast_to(atoms.geometry[1], state.q[:, 1].shape))
        if atoms.bond_constraints:
            constraints = BondConstraints(atoms.masses, atoms.bond_constraints,
                                          cell=atoms.cell, pbc=atoms.pbc)
            assert max(constraints.residuals(state.q, state.p)) < 2e-12
            assert actual["summary"]["quantum_kinetic_estimators_available"] is False


def test_rpc_is_contracted_per_ring_and_does_not_close_early_finishing_leaves(tmp_path, monkeypatch):
    from pyrpmd.dynamics import Dynamics

    cfg, atoms = config(replicas=2), system()
    cfg["potential"] = {"provider": "delta", "reference": {"provider": "free"},
                        "correction": cfg["potential"], "nbeads": 4, "contracted_beads": 2}
    instances = instrument(monkeypatch, rpc=True)
    reference = run_materials(cfg, atoms, tmp_path / "reference", return_data=True)
    # A partially completed replica exits its cooperative lane before the new
    # second replica. The factory-owned RPC leaves must remain live until both
    # lanes finish, even though the split wrapper has a bound close callback.
    original = Dynamics.sample_step

    def interrupted(dynamics, state, *args, **kwargs):
        result = original(dynamics, state, *args, **kwargs)
        if state.step == 7:
            raise RuntimeError("prepare an unequal replica clock")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "sample_step", interrupted)
        with pytest.raises(RuntimeError, match="unequal replica clock"):
            _sample_replica(dict(config=cfg, system=atoms, output=str(tmp_path / "resume"), replica=0))
    instances.clear()
    cfg["parallel"]["trajectory_batch_size"] = 2
    actual = run_materials(cfg, atoms, tmp_path / "resume", return_data=True)
    assert_same_replicas(reference, actual)
    correction, fast = instances
    assert {s[0] for s in correction.shapes} == {2, 4}
    assert {s[0] for s in fast.shapes} == {4, 8}
    assert correction.closed and fast.closed
    assert actual["summary"]["estimator_scope"] == "rpc_contracted_hamiltonian"


@pytest.mark.parametrize("metadata", [
    {"trajectory_batch_safe": False}, {"stateful": True},
    {"batch_independent": False}, {"coupled_frames": True},
])
def test_unsafe_provider_is_rejected_even_with_one_lane_in_the_batch(tmp_path, monkeypatch, metadata):
    instances = instrument(monkeypatch, metadata=metadata)
    cfg = config(replicas=1)
    cfg["parallel"]["trajectory_batch_size"] = 2
    with pytest.raises(ValueError, match="independent, stateless"):
        run_materials(cfg, system(), tmp_path)
    assert instances and all(not p.shapes for p in instances)
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("change,match", [
    ({"barostat": {"method": "bzp", "pressure": "1 bar"}}, "fixed cell"),
    ({"sampling": {"engine": "native"}, "native": True,
      "dynamics": {"loop_backend": "auto"}}, "native PIMD.*trajectory PES batching"),
    ({"dynamics": {"loop_backend": "native"}}, "Python PES orchestration"),
    ({"active_learning": {"force_threshold": .1}}, "active-learning"),
    ({"parallel": {"bead_workers": 2}}, "bead_workers=1"),
])
def test_incompatible_material_modes_fail_during_preflight(change, match):
    cfg = config()
    cfg["parallel"]["trajectory_batch_size"] = 2
    for key, value in change.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    with pytest.raises(ValueError, match=match):
        resolve_materials(cfg, system())


def test_public_native_input_cannot_silently_ignore_a_batch(tmp_path):
    mapping = {
        "task": "pimd", "temperature": "300 K", "beads": 4,
        "system": {"symbols": ["H"], "geometry": [[0., 0., 0.]]},
        "potential": {"provider": "free"},
        "sampling": {"engine": "native", "production_steps": 20, "sample_interval": 2},
        "dynamics": {"timestep": "0.1 fs", "loop_backend": "auto"},
        "parallel": {"trajectory_batch_size": 2},
    }
    with pytest.raises(InputError, match="native PIMD.*trajectory PES batching"):
        JobSpec.from_mapping(mapping).resolve()
