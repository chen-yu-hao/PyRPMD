"""Public materials fixed-atom, NPT and streamed-workflow contracts."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd import JobSpec, prepare, run, resume_run
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import BOHR_M, HARTREE_J, KB_AU, FS_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.system import System
from pyrpmd.trajectory import materialize_stream
from pyrpmd.units import parse_quantity
from pyrpmd.workflows.materials import (
    STREAM_CHECKPOINT_KIND, _read_stream, _sample_replica, quantum_observables,
    resolve_materials, run_materials, reanalyze_diffusion,
)


def fixture(task="pimd", *, fixed=True, npt=False):
    system = System(("H", "Ni"), [2., 20.], [[.1, .2, .3], [1., .5, -.2]],
                    fixed_atoms=(1,) if fixed else (),
                    cell=np.diag([3., 4., 5.]) if npt else None,
                    pbc=(True, True, True) if npt else None)
    config = {
        "task": task, "beta": 2., "beads": 3, "dt_au": .02,
        "seed": 931, "native": False,
        "potential": {"provider": "builtin", "name": "free" if npt else "harmonic",
                      "params": {} if npt else {"k": .4}},
        "sampling": {"equilibration_steps": 5, "production_steps": 35,
                     "sample_interval": 2, "replicas": 1, "gamma_centroid": .6,
                     "thermostat": "pile-l", "andersen_interval": 3},
        "dynamics": {"loop_backend": "python"},
        "output": {"checkpoint_steps": 7, "chunk_frames": 4},
    }
    if task == "diffusion":
        config["diffusion"] = {"atom_indices": [0], "axes": [0, 1], "bootstrap": 0,
                               "reference_atoms": [1]}
    if npt:
        config["barostat"] = {"pressure": .1, "time_constant": 2., "friction": .5}
    return config, system


def task(config, system, path, **options):
    return {"config": config, "system": system, "output": str(path), "replica": 0, **options}


def same(left, right):
    for key in ("positions", "velocities", "observables", "radii_gyration_squared", "sample_steps", "times"):
        np.testing.assert_array_equal(left[key], right[key])
    for key in ("q", "p"):
        np.testing.assert_array_equal(getattr(left["state"], key), getattr(right["state"], key))
    assert left["rng"] == right["rng"]
    assert left["propagation"] == right["propagation"]
    assert left["barostat"] == right["barostat"]


@pytest.mark.parametrize("task_name", ["pimd", "diffusion"])
@pytest.mark.parametrize("integrator,thermostat", [
    ("cayley-bcb", "pile-l"), ("normal-mode-bab", "pile-l"),
    ("cayley-bcb", "andersen"), ("velocity-verlet", "andersen"),
])
def test_fixed_atoms_survive_materials_thermostats_and_production(tmp_path, task_name, integrator, thermostat):
    config, system = fixture(task_name)
    config["dynamics"]["integrator"] = integrator
    config["sampling"]["thermostat"] = thermostat
    stream = _sample_replica(task(config, system, tmp_path))
    np.testing.assert_array_equal(stream["positions"][:, 1], np.tile(system.geometry[1], (stream["frames"], 1)))
    assert np.all(stream["velocities"][:, 1] == 0)
    assert np.all(stream["radii_gyration_squared"][:, 1] == 0)
    assert np.any(stream["positions"][:, 0] != system.geometry[0])
    same(stream, _sample_replica(task(config, system, tmp_path)))


def test_fixed_quantum_estimator_matches_mobile_action_derivative():
    config, system = fixture()
    beta, beads = config["beta"], config["beads"]
    q = np.repeat(system.geometry[None], beads, axis=0)
    q[:, 0] += np.random.default_rng(32).normal(scale=.1, size=(beads, 3))
    potential = HarmonicPotential(k=.4)
    values, radius = quantum_observables(State(q, np.zeros_like(q)), potential, system.masses, beta,
                                        mobile_mask=system.mobile_mask)
    def action(b):
        spring = beads * system.masses[0] * np.sum((q[:, 0] - np.roll(q[:, 0], 1, axis=0))**2) / (2*b)
        return 1.5 * beads * np.log(b) + spring + b * np.mean(potential.evaluate(q).energy)
    derivative = (action(beta + 1e-5) - action(beta - 1e-5)) / 2e-5
    assert values[0] + values[1] == pytest.approx(derivative, abs=1e-9)
    assert radius[1] == 0
    classical, _ = quantum_observables(State(q[:1], np.zeros_like(q[:1])), potential,
                                      system.masses, beta, mobile_mask=system.mobile_mask)
    assert classical[1] == classical[2] == 1.5 / beta


@pytest.mark.parametrize("task_name", ["pimd", "diffusion"])
def test_stream_default_and_legacy_storage_produce_identical_trajectories(tmp_path, task_name):
    config, system = fixture(task_name)
    streamed = _sample_replica(task(config, system, tmp_path / "stream"))
    config["output"]["trajectory_storage"] = "memory"
    legacy = _sample_replica(task(config, system, tmp_path / "memory"))
    same(streamed, legacy)
    with load_checkpoint(streamed["checkpoint"]) as saved:
        assert saved.envelope["kind"] == STREAM_CHECKPOINT_KIND
        assert "positions" not in saved and "sample_steps" not in saved
        assert json.loads(str(saved["trajectory"]))["frames"] == streamed["frames"]
    compact = _sample_replica(task({**config, "output": {"chunk_frames": 2}}, system,
                                   tmp_path / "stream", materialize=False))
    assert "positions" not in compact and "observables" not in compact


@pytest.mark.parametrize("npt", [False, True])
def test_stream_interrupt_and_new_chunk_size_resume_exactly(tmp_path, monkeypatch, npt):
    import pyrpmd.workflows.materials as module
    config, system = fixture(fixed=not npt, npt=npt)
    expected = _sample_replica(task(config, system, tmp_path / "expected"))
    original = module.quantum_observables
    def fail(state, *args, **kwargs):
        values = original(state, *args, **kwargs)
        if state.step == 13:
            raise RuntimeError("injected sample failure")
        return values
    with monkeypatch.context() as patch:
        patch.setattr(module, "quantum_observables", fail)
        with pytest.raises(RuntimeError, match="injected sample"):
            _sample_replica(task(config, system, tmp_path / "resumed"))
    config["output"].update(chunk_frames=7, checkpoint_steps=11)
    actual = _sample_replica(task(config, system, tmp_path / "resumed"))
    same(expected, actual)
    if npt:
        for key in ("cells", "pressures", "volumes"):
            np.testing.assert_array_equal(expected[key], actual[key])
        assert not np.array_equal(actual["cells"][0], actual["cells"][-1])


def test_legacy_complete_checkpoint_migrates_without_resampling(tmp_path, monkeypatch):
    config, system = fixture()
    config["output"]["trajectory_storage"] = "memory"
    original = _sample_replica(task(config, system, tmp_path))
    config["output"]["trajectory_storage"] = "stream"
    monkeypatch.setattr(Dynamics, "sample_step", lambda *a, **k: pytest.fail("migration resampled"))
    migrated = _sample_replica(task(config, system, tmp_path))
    same(original, migrated)
    assert migrated["storage"] == "stream"
    committed = Path(migrated["checkpoint"]).read_bytes()
    _sample_replica(task(config, system, tmp_path))
    assert committed == Path(migrated["checkpoint"]).read_bytes()


def test_fixed_restart_rejects_moved_atom_even_on_completed_stream(tmp_path):
    config, system = fixture()
    stream = _sample_replica(task(config, system, tmp_path))
    with load_checkpoint(stream["checkpoint"]) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["q"][0, 1, 0] += .01
    atomic_savez(stream["checkpoint"], kind=STREAM_CHECKPOINT_KIND, **payload)
    with pytest.raises(ValueError, match="fixed atoms"):
        _sample_replica(task(config, system, tmp_path))


def test_stream_analysis_and_exports_recover_without_provider(tmp_path, monkeypatch):
    import pyrpmd.workflows.materials as module
    config, system = fixture("diffusion")
    result = run_materials(config, system, tmp_path)
    manifest = Path(result["files"]["trajectories"][0])
    payload = json.loads(manifest.read_text())
    arrays = materialize_stream(manifest.parent, payload["trajectory"])
    assert arrays["positions"].shape == (18, 2, 3)
    manifest.unlink()
    monkeypatch.setattr(module, "make_task_potential", lambda *a, **k: pytest.fail("offline provider call"))
    replay = reanalyze_diffusion(config, system, tmp_path)
    assert replay["diffusion"] == result["diffusion"]
    assert manifest.exists()
    assert replay["diffusion"]["input"]["atom_indices"] == [0]
    assert replay["fixed_atoms"] == [1]


@pytest.mark.parametrize("unit,pascals", [("Pa", 1), ("bar", 1e5), ("atm", 101325), ("GPa", 1e9)])
def test_pressure_units(unit, pascals):
    assert parse_quantity("1 " + unit, "pressure") == pytest.approx(pascals * BOHR_M**3 / HARTREE_J)
    with pytest.raises(ValueError, match="dimension"):
        parse_quantity("1 fs", "pressure")


@pytest.mark.parametrize("change,match", [
    ({"method": "mttk"}, "method"), ({"pressure": "1 fs"}, "dimension"),
    ({"pressure": float("nan")}, "finite"), ({"time_constant": 0}, "positive"),
    ({"friction": -1}, "nonnegative"), ({"unknown": 1}, "Unknown"),
])
def test_npt_rejects_bad_physical_input(change, match):
    config, system = fixture(fixed=False, npt=True)
    config["barostat"].update(change)
    with pytest.raises(ValueError, match=match):
        resolve_materials(config, system)


def test_npt_scope_and_fixed_diffusion_selection_guards():
    config, system = fixture(fixed=False, npt=True)
    config["task"] = "diffusion"
    with pytest.raises(ValueError, match="fixed-cell"):
        resolve_materials(config, system)
    config["task"] = "pimd"
    system.fixed_atoms = (1,)
    with pytest.raises(ValueError, match="all atoms mobile"):
        resolve_materials(config, system)
    config, system = fixture("diffusion")
    config["diffusion"]["atom_indices"] = [1]
    with pytest.raises(ValueError, match="mobile"):
        resolve_materials(config, system)


@pytest.mark.parametrize("change", ["method", "momentum", "anisotropic", "last_frame"])
def test_completed_npt_checkpoint_validates_piston_without_loading_pes(tmp_path, monkeypatch, change):
    import pyrpmd.workflows.materials as module
    config, system = fixture(fixed=False, npt=True)
    config["sampling"]["production_steps"] = 34
    stream = _sample_replica(task(config, system, tmp_path))
    with load_checkpoint(stream["checkpoint"]) as saved:
        payload = {key: saved[key] for key in saved.files}
    piston = json.loads(str(payload["barostat"]))
    if change == "method":
        piston["method"] = "other"
    elif change == "momentum":
        piston["momentum"] = True
    elif change == "anisotropic":
        piston["cell"][0][0] *= 1.1
    else:
        piston["cell"] = (np.asarray(piston["cell"]) * 1.1).tolist()
    payload["barostat"] = json.dumps(piston)
    atomic_savez(stream["checkpoint"], kind=STREAM_CHECKPOINT_KIND, **payload)
    monkeypatch.setattr(module, "make_task_potential", lambda *a, **k: pytest.fail("completed resume loaded PES"))
    with pytest.raises(ValueError, match="BZP|piston|cell"):
        _sample_replica(task(config, system, tmp_path))


def test_public_npt_input_prepare_run_resume_and_summary(tmp_path):
    raw = {
        "task": "pimd", "temperature": 1 / (2 * KB_AU), "beads": 2, "native": False,
        "system": {"symbols": ["H"], "masses": [2.], "geometry": [[.1, .2, .3]],
                   "geometry_unit": "bohr", "cell": np.diag([3., 4., 5.]).tolist(),
                   "cell_unit": "bohr", "pbc": [True, True, True]},
        "potential": {"provider": "builtin", "name": "free"},
        "dynamics": {"timestep": .02},
        "sampling": {"equilibration_steps": 5, "production_steps": 30, "sample_interval": 2},
        "barostat": {"method": "bzp", "pressure": "1 bar", "time_constant": "5 fs"},
        "output": {"chunk_frames": 4, "checkpoint_steps": 7},
    }
    config, system = JobSpec.from_mapping(raw).resolve()
    options = resolve_materials(config, system)
    assert options["barostat"]["tau_au"] == 5 * FS_TO_AU
    preparation = prepare(raw, numerical=True)
    assert preparation["materials_preparation"]["ensemble"] == "npt"
    result = run(raw, output=tmp_path / "public")
    assert result["ensemble"] == "npt"
    assert result["barostat"]["target_pressure_bar"] == pytest.approx(1.)
    assert result["trajectory_storage"] == ["stream"]
    assert not result["physical_converged"]
    resumed = resume_run(tmp_path / "public")
    assert result["barostat"] == resumed["barostat"]
    assert result["estimators"] == resumed["estimators"]
