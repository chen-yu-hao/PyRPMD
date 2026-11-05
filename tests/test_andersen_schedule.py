"""Random collision times preserve equilibrium sampling and exact restart state."""

import copy
import ctypes
import importlib
import json
from pathlib import Path
import tomllib
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest

from pyrpmd.checkpoint import ENVELOPE, atomic_savez, load_checkpoint
from pyrpmd.constants import AU_TIME_FS, KB_AU
from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.input import InputError, JobSpec
from pyrpmd.propagation import PropagationError
from pyrpmd.system import System
from pyrpmd.workflows import fes, rate
from pyrpmd.workflows.sampling import AndersenSchedule, validate_andersen_schedule

from test_fes_block_sampling import cabi_library, cabi_controls, _task, _checkpoint, _failing_call


@pytest.fixture(scope="module")
def parent_cabi_library(cabi_library, tmp_path_factory):
    """Extend the instrumented one-atom example to a real three-atom ABI."""
    source = cabi_library.with_suffix(".c").read_text()
    changes = {
        r'\"natoms\":1': r'\"natoms\":3',
        r'\"atom_ids\":[\"0\"]': r'\"atom_ids\":[\"0\",\"1\",\"2\"]',
        r'\"atomic_numbers\":[1]': r'\"atomic_numbers\":[1,1,1]',
        "b>INT64_MAX/3": "b>INT64_MAX/9", "x->natoms!=1": "x->natoms!=3",
        "positions_count!=3*b": "positions_count!=9*b", "forces_count!=3*b": "forces_count!=9*b",
        "c<3": "c<9", "[3*i+c]": "[9*i+c]",
    }
    for old, new in changes.items():
        assert old in source
        source = source.replace(old, new)
    directory = tmp_path_factory.mktemp("parent-clock-pes")
    cfile, library = directory / "harmonic.c", directory / "harmonic.so"
    cfile.write_text(source)
    include = Path(__file__).resolve().parents[1] / "native/include"
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-std=c11", "-I", str(include),
                    str(cfile), "-o", str(library)], check=True, capture_output=True)
    return library


@pytest.fixture
def parent_cabi_controls(parent_cabi_library):
    controls = ctypes.CDLL(str(parent_cabi_library))
    controls.audit_reset.argtypes = [ctypes.c_int, ctypes.c_int]
    controls.audit_reset.restype = None
    controls.audit_reset(-1, -1)
    yield controls
    controls.audit_reset(-1, -1)


def _payload(path):
    with load_checkpoint(path) as saved:
        return {key: saved[key].copy() for key in saved.files if key != ENVELOPE}


def _geometric_task(path, library=None, **kwargs):
    task = _task(path, library, **kwargs)
    task["config"]["sampling"]["andersen_schedule"] = "geometric"
    return task


def _same_trajectory(a, b, *, tolerance=0):
    for key in ("q", "p", "values", "energies", "snapshots"):
        if key in a:
            if tolerance:
                np.testing.assert_allclose(a[key], b[key], atol=tolerance, rtol=tolerance, err_msg=key)
            else:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    for key in ("step", "elapsed", "time", "rng", "andersen_schedule", "andersen_next_collision_step"):
        if key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_omitted_schedule_replays_independent_prechange_fixture_and_hash(tmp_path, monkeypatch):
    fixture = json.loads((Path(__file__).parent / "data/andersen_legacy_window.json").read_text())
    task = {key: fixture[key] for key in ("config", "specs", "center", "stiffness", "window", "replica")}
    task.update(system=System.from_dict(fixture["system"]), output=str(tmp_path / "new"))
    fes._sample_window(task)
    actual = _payload(_checkpoint(task))
    for key, expected in fixture["expected"].items():
        if key in ("q", "p", "values", "energies"):
            np.testing.assert_allclose(actual[key], expected, atol=3e-14, rtol=3e-14)
        elif key == "fingerprint":
            from pyrpmd.workflows.common import fingerprint_matches

            identity = {name: task[name] for name in ("window", "replica", "center", "stiffness")}
            assert actual[key].item() == fes._window_fingerprint(task)
            assert actual[key].item() != expected  # Canonical FES writes schema 2.
            assert fingerprint_matches(expected, task["config"], task["system"], identity)
        else:
            assert actual[key].item() == expected
    assert "andersen_schedule" not in actual and "andersen_next_collision_step" not in actual
    # A pre-change payload has no clock state. Its old fingerprint must still
    # authorize a completed replay without asking the integrator to propagate.
    task["output"] = str(tmp_path / "legacy")
    atomic_savez(_checkpoint(task), **fixture["expected"])
    monkeypatch.setattr(Dynamics, "step", Mock(side_effect=AssertionError("replayed a completed legacy run")))
    restored = fes._sample_window(task)
    np.testing.assert_array_equal(restored["samples"], fixture["expected"]["values"])


def test_explicit_fixed_retains_old_rng_and_samples(tmp_path):
    implicit = _task(tmp_path / "implicit", backend="python")
    explicit = _task(tmp_path / "explicit", backend="python")
    explicit["config"]["sampling"]["andersen_schedule"] = "fixed"
    fes._sample_window(implicit)
    fes._sample_window(explicit)
    a, b = _payload(_checkpoint(implicit)), _payload(_checkpoint(explicit))
    _same_trajectory(a, b)
    assert "andersen_schedule" not in b and "andersen_next_collision_step" not in b
    # Explicit input belongs to the scientific configuration; adding/changing
    # this option is not a way to bypass a saved run's fingerprint.
    assert a["fingerprint"] != b["fingerprint"]


def test_geometric_clock_has_requested_mean_and_nonperiodic_waits():
    rng = np.random.default_rng(24101)
    clock = AndersenSchedule(50, "geometric")
    waits = []
    for _ in range(20000):
        start = clock.next_collision_step
        assert clock.due(start)
        clock.refreshed(start, rng)
        waits.append(clock.next_collision_step - start)
    assert min(waits) == 1 and max(waits) > 200
    assert abs(np.mean(waits) - 50) < 1.0
    assert abs(np.mean(np.asarray(waits) == 1) - .02) < .003


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
@pytest.mark.parametrize("block_steps", [1, 9, 65536])
def test_geometric_native_blocks_match_python_and_collision_boundaries(
    tmp_path, cabi_library, cabi_controls, monkeypatch, scheme, block_steps,
):
    reference = _geometric_task(tmp_path / "python", cabi_library, backend="python", scheme=scheme)
    fes._sample_window(reference)
    task = _geometric_task(tmp_path / "native", cabi_library, block_steps=block_steps, scheme=scheme)
    original_refresh, original_block = AndersenSchedule.refreshed, Dynamics.propagate_block
    deadlines, blocks = [], []

    def refreshed(clock, step, rng):
        original_refresh(clock, step, rng)
        deadlines.append((step, clock.next_collision_step))

    def propagated(dyn, state, count, **options):
        assert state.step + count <= deadlines[-1][1]
        blocks.append((state.step, count))
        return original_block(dyn, state, count, **options)

    monkeypatch.setattr(AndersenSchedule, "refreshed", refreshed)
    monkeypatch.setattr(Dynamics, "propagate_block", propagated)
    fes._sample_window(task)
    assert blocks and len(deadlines) > 1
    assert len({stop - start for start, stop in deadlines}) > 1
    a, b = _payload(_checkpoint(task)), _payload(_checkpoint(reference))
    _same_trajectory(a, b, tolerance=4e-11)
    assert json.loads(str(a["propagation"]))["native_steps"] == 69


@pytest.mark.parametrize("backend", ["python", "native"])
def test_geometric_umbrella_interruption_restores_deadline_and_rng(
    tmp_path, cabi_library, cabi_controls, monkeypatch, backend,
):
    complete = _geometric_task(tmp_path / "complete", cabi_library, backend=backend, block_steps=1)
    task = _geometric_task(tmp_path / "resumed", cabi_library, backend=backend)
    fes._sample_window(complete)
    original = fes._atomic_npz

    def interrupted(path, **arrays):
        original(path, **arrays)
        if arrays["elapsed"] == 17:
            raise KeyboardInterrupt("checkpoint interruption")

    with monkeypatch.context() as patch:
        patch.setattr(fes, "_atomic_npz", interrupted)
        with pytest.raises(KeyboardInterrupt):
            fes._sample_window(task)
    prefix = _payload(_checkpoint(task))
    assert int(prefix["step"]) == 17
    assert int(prefix["andersen_next_collision_step"]) >= 17
    task["config"]["dynamics"]["block_steps"] = 9
    fes._sample_window(task)
    _same_trajectory(_payload(_checkpoint(task)), _payload(_checkpoint(complete)))


@pytest.mark.parametrize("failure_case", ["initial", "refresh", "partial"])
def test_geometric_fused_failure_commits_prefix_and_replays_failed_refresh_once(
    tmp_path, cabi_library, cabi_controls, monkeypatch, failure_case,
):
    complete = _geometric_task(tmp_path / "complete", cabi_library, block_steps=1)
    refreshes = []
    original = AndersenSchedule.refreshed

    def recorded(clock, step, rng):
        original(clock, step, rng)
        refreshes.append((step, clock.next_collision_step))

    with monkeypatch.context() as patch:
        patch.setattr(AndersenSchedule, "refreshed", recorded)
        fes._sample_window(complete)
    if failure_case == "initial":
        failed_step = 1
    elif failure_case == "refresh":
        failed_step = refreshes[1][0] + 1
    else:
        failed_step = next(start + 2 for start, end in refreshes if end - start > 2)
    task = _geometric_task(tmp_path / "failed", cabi_library)
    # Fail that step under the actual native schedule (block-local force reuse).
    cabi_controls.audit_reset(-1, -1)
    failing = _failing_call(
        lambda: fes._sample_window(_geometric_task(tmp_path / "probe", cabi_library)),
        cabi_controls.audit_get_calls, failed_step)
    cabi_controls.audit_reset(failing, -1)
    with pytest.raises(PropagationError, match="injected CABI failure"):
        fes._sample_window(task)
    assert cabi_controls.audit_get_calls() == failing
    prefix = _payload(_checkpoint(task))
    assert int(prefix["elapsed"]) == int(prefix["step"]) == failed_step - 1
    if failure_case != "partial":
        assert int(prefix["andersen_next_collision_step"]) == failed_step - 1
    cabi_controls.audit_reset(-1, -1)
    fes._sample_window(task)
    _same_trajectory(_payload(_checkpoint(task)), _payload(_checkpoint(complete)))


@pytest.mark.parametrize("damage", ["missing", "fractional", "past", "wrong-mode"])
def test_geometric_resume_rejects_missing_or_inconsistent_clock(tmp_path, monkeypatch, damage):
    task = _geometric_task(tmp_path, backend="python")
    fes._sample_window(task)
    saved = _payload(_checkpoint(task))
    if damage == "missing":
        saved.pop("andersen_next_collision_step")
    elif damage == "fractional":
        saved["andersen_next_collision_step"] = float(saved["step"]) + .5
    elif damage == "past":
        saved["andersen_next_collision_step"] = int(saved["step"]) - 1
    else:
        saved["andersen_schedule"] = "fixed"
    atomic_savez(_checkpoint(task), **saved)
    monkeypatch.setattr(Dynamics, "step", Mock(side_effect=AssertionError("invalid clock propagated")))
    with pytest.raises(ValueError, match="checkpoint.*collision clock"):
        fes._sample_window(task)


def test_geometric_python_observation_failure_rolls_back_before_resume(tmp_path, monkeypatch):
    task = _geometric_task(tmp_path / "failed", backend="python")
    complete = _geometric_task(tmp_path / "complete", backend="python")
    fes._sample_window(complete)
    original = Dynamics.energy

    def failed(dyn, state, bias=None):
        if state.step == 19:
            raise RuntimeError("injected observation failure")
        return original(dyn, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "energy", failed)
        with pytest.raises(RuntimeError, match="injected observation failure"):
            fes._sample_window(task)
    prefix = _payload(_checkpoint(task))
    assert int(prefix["step"]) == 18 and len(prefix["values"]) == 2
    fes._sample_window(task)
    _same_trajectory(_payload(_checkpoint(task)), _payload(_checkpoint(complete)))


def test_geometric_clock_survives_adaptive_pilot_and_production(tmp_path):
    from test_fes_pile_l import config, system

    cfg = config()
    cfg["sampling"].pop("gamma_centroid")
    cfg["sampling"].update(thermostat="andersen", andersen_schedule="geometric")
    cfg["adaptive_umbrella"] = {"enabled": True, "pilot_steps": 60, "pilot_equilibration_steps": 10,
                                "max_rounds": 2, "max_windows": 8, "target_overlap": 1e-5,
                                "min_effective_samples": 1.0}
    result = fes.run_fes(cfg, system(), tmp_path, return_data=True)
    assert result["summary"]["adaptive_umbrella"]["pilot_samples_used_in_production"] == 0
    for stage in (tmp_path / "adaptation", tmp_path / "checkpoints"):
        paths = list(stage.rglob("window_*.npz"))
        assert paths, stage
        for path in paths:
            saved = _payload(path)
            assert str(saved["andersen_schedule"]) == "geometric"
            assert int(saved["andersen_next_collision_step"]) >= int(saved["step"])


def _parent_task(path, library, backend="native", block_steps=64):
    system = System(("H", "H", "H"), np.full(3, 2.0),
                    np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]), ((0,), (1, 2)))
    config = {"task": "rate", "beta": 10., "beads": 4, "dt_au": 1e-4, "seed": 9131,
              "potential": {"provider": "cabi", "library": str(library)}, "native": True,
              "sampling": {"andersen_interval": 13, "andersen_schedule": "geometric"},
              "reaction": {"fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
                           "breaking_bonds": [[1, 2]], "rinf": 10.},
              "dynamics": {"loop_backend": backend, "block_steps": block_steps},
              "output": {"checkpoint_steps": 17}}
    return {"config": config, "system": system, "chain": 1, "xi_star": 1., "output": str(path),
            "initial_q": np.repeat(system.geometry[None], 4, axis=0),
            "options": {"equilibration_steps": 7, "parent_samples": 12,
                        "parent_interval_steps": 4, "constraint_tolerance": 1e-10}}


def _parent_saved(task):
    return _payload(Path(task["output"]) / "checkpoints/parent_0001.npz")


def test_fixed_parent_replays_historical_fixture_and_fingerprint(tmp_path, monkeypatch):
    fixture = json.loads((Path(__file__).parent / "data/andersen_legacy_parent.json").read_text())
    task = {key: fixture[key] for key in ("config", "chain", "xi_star", "options")}
    task.update(system=System.from_dict(fixture["system"]), initial_q=np.array(fixture["initial_q"]),
                output=str(tmp_path / "new"))
    rate._parent_task(task)
    actual = _parent_saved(task)
    for key, expected in fixture["expected"].items():
        if key in ("q", "p", "snapshots"):
            np.testing.assert_allclose(actual[key], expected, atol=3e-14, rtol=3e-14)
        else:
            assert actual[key].item() == expected
    assert "andersen_schedule" not in actual and "andersen_next_collision_step" not in actual
    task["output"] = str(tmp_path / "legacy")
    atomic_savez(Path(task["output"]) / "checkpoints/parent_0001.npz", **fixture["expected"])
    monkeypatch.setattr(Dynamics, "propagate_block", Mock(side_effect=AssertionError("legacy parent propagated")))
    restored = rate._parent_task(task)
    np.testing.assert_array_equal(restored["snapshots"], fixture["expected"]["snapshots"])


def test_geometric_constrained_parent_native_python_agree(tmp_path, parent_cabi_library, parent_cabi_controls):
    python = _parent_task(tmp_path / "python", parent_cabi_library, "python")
    native = _parent_task(tmp_path / "native", parent_cabi_library)
    rate._parent_task(python)
    rate._parent_task(native)
    _same_trajectory(_parent_saved(python), _parent_saved(native), tolerance=2e-11)
    surface = rate.rate_coordinate(native["config"], native["system"]).dividing_surface(1.)
    for snapshot in _parent_saved(native)["snapshots"]:
        assert abs(surface.centroid_value_gradient(snapshot.mean(0))[0]) < 1e-10


@pytest.mark.parametrize("backend", ["python", "native"])
def test_geometric_parent_interruption_preserves_chain_and_deadline(
    tmp_path, parent_cabi_library, parent_cabi_controls, monkeypatch, backend,
):
    from pyrpmd.workflows import rate_recrossing

    complete = _parent_task(tmp_path / "complete", parent_cabi_library, backend, 1)
    task = _parent_task(tmp_path / "resumed", parent_cabi_library, backend)
    rate._parent_task(complete)
    original = rate_recrossing._atomic_npz

    def interrupted(path, **arrays):
        original(path, **arrays)
        if arrays["elapsed"] == 17:
            raise KeyboardInterrupt("checkpoint interruption")

    with monkeypatch.context() as patch:
        patch.setattr(rate_recrossing, "_atomic_npz", interrupted)
        with pytest.raises(KeyboardInterrupt):
            rate._parent_task(task)
    assert int(_parent_saved(task)["step"]) == 17
    task["config"]["dynamics"]["block_steps"] = 9
    rate._parent_task(task)
    _same_trajectory(_parent_saved(task), _parent_saved(complete))


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("failure_case", ["initial", "refresh", "partial"])
def test_geometric_parent_failure_preserves_accepted_prefix_and_refresh(
    tmp_path, parent_cabi_library, parent_cabi_controls, monkeypatch, backend, failure_case,
):
    complete = _parent_task(tmp_path / "complete", parent_cabi_library, backend, 1)
    refreshes = []
    original = AndersenSchedule.refreshed

    def recorded(clock, step, rng):
        original(clock, step, rng)
        refreshes.append((step, clock.next_collision_step))

    with monkeypatch.context() as patch:
        patch.setattr(AndersenSchedule, "refreshed", recorded)
        rate._parent_task(complete)
    if failure_case == "initial":
        failed_step = 1
    elif failure_case == "refresh":
        failed_step = refreshes[1][0] + 1
    else:
        failed_step = next(start + 2 for start, end in refreshes if end - start > 2)
    task = _parent_task(tmp_path / "failed", parent_cabi_library, backend)
    parent_cabi_controls.audit_reset(2 * failed_step, -1)
    with pytest.raises(PropagationError, match="injected CABI failure"):
        rate._parent_task(task)
    saved = _parent_saved(task)
    assert int(saved["elapsed"]) == int(saved["step"]) == failed_step - 1
    assert len(saved["snapshots"]) == max(0, (failed_step - 1 - 7) // 4)
    if failure_case != "partial":
        assert int(saved["andersen_next_collision_step"]) == failed_step - 1
    parent_cabi_controls.audit_reset(-1, -1)
    rate._parent_task(task)
    _same_trajectory(_parent_saved(task), _parent_saved(complete))


UNSUPPORTED_RUNNERS = [
    ("materials", "run_materials"), ("drpmd", "run_drpmd"),
    ("surface", "run_surface_rate"), ("site_rate", "run_site_rate"),
    ("network", "run_network"), ("metadynamics", "run_metadynamics"),
    ("opes", "run_opes"), ("replica_exchange", "run_replica_exchange_fes"),
    ("native_pimd", "run_native_pimd"), ("pcet", "run_pcet"),
]


@pytest.mark.parametrize("module_name,function_name", UNSUPPORTED_RUNNERS)
def test_unsupported_direct_workflow_fails_before_resources(tmp_path, monkeypatch, module_name, function_name):
    module = importlib.import_module("pyrpmd.workflows." + module_name)
    target = tmp_path / "absent"
    dispatched = Mock(side_effect=AssertionError("unsupported schedule dispatched work"))
    for name in ("run_tasks", "root_call"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, dispatched)
    config = {"task": "fes", "sampling": {"andersen_schedule": "geometric"}}
    system = System(("H",), np.ones(1), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="andersen_schedule.*supported only"):
        getattr(module, function_name)(config, system, target)
    dispatched.assert_not_called()
    assert not target.exists()


@pytest.mark.parametrize("change", [
    {"task": "pimd"}, {"task": "drpmd"},
    {"sampling": {"thermostat": "pile-l"}},
    {"replica_exchange": {"enabled": True}},
    {"replica_exchange": {"interval_steps": 10}},
])
def test_runner_rejects_unsupported_schedule_before_manifest(tmp_path, monkeypatch, change):
    from pyrpmd import runner

    config = {"task": "fes", "sampling": {"andersen_schedule": "geometric"}}
    for key, value in change.items():
        if isinstance(value, dict):
            config.setdefault(key, {}).update(value)
        else:
            config[key] = value
    initializer = Mock(side_effect=AssertionError("created a run before schedule validation"))
    monkeypatch.setattr(runner, "_initialize_output", initializer)
    with pytest.raises(ValueError, match="andersen_schedule"):
        runner.run_resolved(config, System(("H",), np.ones(1), np.zeros((1, 3))), output=tmp_path / "absent")
    initializer.assert_not_called()
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("kind", [None, "fixed", "geometric"])
def test_input_resolves_schedule_without_inserting_a_legacy_default(kind):
    source = Path(__file__).resolve().parents[1] / "examples/fes/harmonic2d.inp"
    config = tomllib.loads(source.read_text())
    if kind is not None:
        config["sampling"]["andersen_schedule"] = kind
    before = copy.deepcopy(config)
    resolved, _ = JobSpec.from_mapping(config).resolve()
    assert config == before
    if kind is None:
        assert "andersen_schedule" not in resolved["sampling"]
    else:
        assert resolved["sampling"]["andersen_schedule"] == kind


@pytest.mark.parametrize("kind", [None, True, 1, "random", [], {}])
def test_input_rejects_invalid_schedule(kind):
    source = Path(__file__).resolve().parents[1] / "examples/fes/harmonic2d.inp"
    config = tomllib.loads(source.read_text())
    config["sampling"]["andersen_schedule"] = kind
    with pytest.raises(InputError, match="andersen_schedule"):
        JobSpec.from_mapping(config).resolve()


@pytest.mark.parametrize("task,module,resolver", [
    ("network", "pyrpmd.reactions", "resolve_network_job"),
    ("pcet", "pyrpmd.pcet_input", "resolve_pcet_job"),
])
def test_early_return_input_resolvers_reject_schedule_before_preparation(monkeypatch, task, module, resolver):
    prepare = Mock(side_effect=AssertionError("unsupported clock reached specialized preparation"))
    monkeypatch.setattr(importlib.import_module(module), resolver, prepare)
    with pytest.raises(InputError, match="andersen_schedule.*supported only"):
        JobSpec.from_mapping({"task": task, "sampling": {"andersen_schedule": "geometric"}}).resolve()
    prepare.assert_not_called()


def test_unrepresentable_geometric_clock_fails_preflight():
    with pytest.raises(ValueError, match="checkpoint clock range"):
        validate_andersen_schedule({"task": "fes", "sampling": {
            "andersen_schedule": "geometric", "andersen_interval": 2 ** 63}})


def test_geometric_free_modes_recover_from_the_periodic_resonance(tmp_path, record_property):
    if not native_capabilities()["available"]:
        pytest.skip("native integrator required for the free-ring sampling control")
    beads, atoms, mass = 128, 32, 1837.1526474396187
    beta = 1 / (KB_AU * 300.)
    system = System(("H",) * atoms, np.full(atoms, mass), np.zeros((atoms, 3)))
    result = {}
    for kind in ("fixed", "geometric"):
        task = {"system": system, "output": str(tmp_path / kind), "window": 0, "replica": 0,
                "specs": [{"kind": "cartesian", "atom": 0, "axis": "x"}],
                "center": [0.], "stiffness": [[.001]],
                "config": {"task": "fes", "beta": beta, "beads": beads,
                           "dt_au": .1 / AU_TIME_FS, "seed": 91420, "native": True,
                           "potential": {"provider": "free"},
                           "sampling": {"equilibration_steps": 0, "production_steps": 5000,
                                        "sample_interval": 2500, "andersen_interval": 50,
                                        "andersen_schedule": kind},
                           "output": {"checkpoint_steps": 5000},
                           "parallel": {"threads_per_rank": 1}}}
        sampled = fes._sample_window(task)
        # The centroid bias leaves every internal free-ring mode untouched.
        transformed = np.fft.rfft(sampled["last_q"], axis=0, norm="ortho")
        ratios = {}
        for mode in (13, 37):
            omega = 2 * beads / beta * np.sin(np.pi * mode / beads)
            expected = beads / (beta * mass * omega ** 2)
            ratios[str(mode)] = float(np.mean(abs(transformed[mode]) ** 2) / expected)
        result[kind] = ratios
    record_property("free_mode_variance_ratio_after_500fs", json.dumps(result, sort_keys=True))
    assert result["fixed"]["37"] < .25, result
    assert result["fixed"]["13"] < .65, result
    for ratio in result["geometric"].values():
        assert .65 < ratio < 1.4, result
