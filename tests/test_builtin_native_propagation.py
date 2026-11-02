"""Built-in bindings preserve physical maps, bath streams and restart identity."""

import ast
import cProfile
import copy
import json
from pathlib import Path
import pstats
import runpy

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, INTEGRATORS, State, native_capabilities
from pyrpmd.potentials import (
    DoubleWellPotential,
    FreePotential,
    HarmonicPotential,
    MorsePairPotential,
    PeriodicCosinePotential,
)
from pyrpmd.propagation import NativeUnavailableError


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("free", "harmonic", "morse", "double-well", "periodic-cosine")
THERMOSTATS = [
    pytest.param({"mode": "andersen", "interval": 3}, id="andersen"),
    pytest.param("pile-l", id="pile-l"),
    pytest.param("pile-g", id="pile-g"),
    pytest.param("svr", id="svr"),
    pytest.param("langevin", id="langevin"),
    pytest.param("ffl", id="ffl"),
    pytest.param("nhc", id="nhc"),
    pytest.param("trpmd", id="trpmd"),
    pytest.param({"mode": "gle", "gle": {"A": [[1., 2.], [-2., 1.]]}}, id="gle"),
    pytest.param({"mode": "nm-gle", "gle": {"A": [[1., 2.], [-2., 1.]]}}, id="nm-gle"),
    pytest.param({"mode": "nm-gle-g", "gle": {"A": [[1., 2.], [-2., 1.]]}}, id="nm-gle-g"),
    pytest.param({"mode": "piglet", "gle": {
        "A": [[[1., 2.], [-2., 1.]]] * 4, "C": [[1., 0.], [0., 1.]],
        "beta": 6., "beads": 4,
        "provenance": "Classical covariance for dispatch parity only; no quantum-accuracy claim",
    }}, id="piglet"),
]

pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native extension unavailable",
)


def potential(name):
    if name == "free":
        return FreePotential()
    if name == "harmonic":
        return HarmonicPotential(k=[.2, .3, .4], center=[.1, -.2, .3])
    if name == "morse":
        return MorsePairPotential(atoms=(0, 2), depth=.3, alpha=.7, equilibrium=1.4)
    if name == "double-well":
        return DoubleWellPotential(height=.2, width=1.2, atom=1, axis=2, confinement=.03)
    if name == "periodic-cosine":
        return PeriodicCosinePotential(
            [[4., 0., 0.], [.2, 5., 0.], [.4, .2, 6.]],
            pbc=[True, False, True], height=[.02, .03, .04],
            repeats=(2, 1, 3), atoms=[0, 2], confinement=.03,
        )
    raise AssertionError(name)


def initial():
    rng = np.random.default_rng(117)
    q = np.repeat([[[-.8, .2, .3], [.5, -.1, .4], [1.7, .2, -.1]]], 4, axis=0)
    q += rng.normal(scale=.015, size=q.shape)
    return State(q, rng.normal(scale=.07, size=q.shape), step=7, time=.137)


def dynamics(name, integrator="cayley-bcb", thermostat="none", **options):
    return Dynamics(
        potential(name), [2., 3., 4.], 6., 4, .004,
        integrator=integrator, thermostat=thermostat, seed=19,
        potential_identity="builtin-parity-" + name, **options,
    )


def assert_state_close(actual, expected):
    np.testing.assert_allclose(actual.q, expected.q, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(actual.p, expected.p, atol=3e-12, rtol=3e-12)
    assert (actual.step, actual.time) == (expected.step, expected.time)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
def test_fresh_builtin_auto_blocks_match_independent_numpy_map(model, integrator, api):
    # No evaluate() or explicit binding call primes the provider before Dynamics.
    native = dynamics(model, integrator, block_steps=5)
    python = dynamics(model, integrator, native=False, loop_backend="python")
    actual, expected = initial(), initial()
    cv = Cartesian(1, "z")
    options = ({"observe": cv, "sample_interval": 3} if api == "propagate" else
               {"cv": cv, "record_states": True})
    got = getattr(native, api)(actual, 13, **options)
    wanted = getattr(python, api)(expected, 13, **options)
    assert got.backend == "native" and wanted.backend == "python"
    assert_state_close(actual, expected)
    for field in ("steps", "energies", "centroids", "values"):
        np.testing.assert_allclose(getattr(got, field), getattr(wanted, field),
                                   atol=3e-12, rtol=3e-12)
    if api == "propagate_block":
        np.testing.assert_allclose(got.q, wanted.q, atol=3e-12, rtol=3e-12)
        np.testing.assert_allclose(got.p, wanted.p, atol=3e-12, rtol=3e-12)
    assert native.loop_stats == {"native_steps": 13, "python_steps": 0, "fallback_reasons": {}}


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("thermostat", THERMOSTATS)
def test_composed_builtin_blocks_match_python_scheduler(model, integrator, thermostat):
    pytest.importorskip("pyrpmd._thermal")
    native = dynamics(model, integrator, thermostat, loop_backend="auto")
    python = dynamics(model, integrator, thermostat, loop_backend="python")
    actual, expected = initial(), initial()
    diagnostics = {"energy_parts": True, "ring_radii": True}
    got = native.propagate_block(actual, 13, diagnostics=diagnostics)
    wanted = python.propagate_block(expected, 13, diagnostics=diagnostics)
    assert got.backend == "native" and wanted.backend == "python"
    assert_state_close(actual, expected)
    for field in ("energies", "centroids", "centroid_momenta"):
        np.testing.assert_allclose(getattr(got, field), getattr(wanted, field),
                                   atol=3e-12, rtol=3e-12)
    for name in diagnostics:
        np.testing.assert_allclose(got.diagnostics[name], wanted.diagnostics[name],
                                   atol=3e-12, rtol=3e-12)
    assert native.rng.bit_generator.state == python.rng.bit_generator.state
    assert native.loop_stats["native_steps"] == 13
    assert native.loop_stats["python_steps"] == 0


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("thermostat", ["none", "pile-l", "nhc"])
def test_native_block_never_calls_python_builtin_evaluate(model, thermostat):
    if thermostat != "none":
        pytest.importorskip("pyrpmd._thermal")
    dyn = dynamics(model, thermostat=thermostat)
    profiler = cProfile.Profile()
    with profiler:
        result = dyn.propagate(initial(), 11, backend="native")
    assert result.backend == "native"
    calls = sum(counts[1] for (filename, _, function), counts in pstats.Stats(profiler).stats.items()
                if Path(filename).name == "builtin.py" and function == "evaluate")
    assert calls == 0


@pytest.mark.parametrize("thermostat", ["nhc", "pile-l", {"mode": "andersen", "interval": 3}])
def test_velocity_verlet_composed_limit_stays_explicit(thermostat):
    pytest.importorskip("pyrpmd._thermal")
    auto = dynamics("harmonic", "velocity-verlet", thermostat)
    python = dynamics("harmonic", "velocity-verlet", thermostat, loop_backend="python")
    actual, expected = initial(), initial()
    assert auto.propagate_block(actual, 7).backend == "python"
    python.propagate_block(expected, 7)
    assert_state_close(actual, expected)
    assert auto.rng.bit_generator.state == python.rng.bit_generator.state
    before, operators = actual.copy(), copy.deepcopy(auto.operator_state_dict())
    with pytest.raises(NativeUnavailableError):
        auto.propagate(actual, 1, backend="native")
    np.testing.assert_array_equal(actual.q, before.q)
    np.testing.assert_array_equal(actual.p, before.p)
    assert auto.operator_state_dict() == operators


@pytest.mark.parametrize("model", ["free", "periodic-cosine"])
@pytest.mark.parametrize("thermostat", ["pile-l", "nhc"])
def test_periodic_composed_cabi_limit_stays_explicit(model, thermostat):
    pytest.importorskip("pyrpmd._thermal")
    cell = [[4., 0., 0.], [.2, 5., 0.], [.4, .2, 6.]]
    options = {"cell": cell, "pbc": [True, False, True]}
    auto = dynamics(model, thermostat=thermostat, **options)
    python = dynamics(model, thermostat=thermostat, loop_backend="python", **options)
    actual, expected = initial(), initial()
    assert auto.propagate_block(actual, 7).backend == "python"
    python.propagate_block(expected, 7)
    assert_state_close(actual, expected)
    before, operators = actual.copy(), copy.deepcopy(auto.operator_state_dict())
    with pytest.raises(NativeUnavailableError):
        auto.propagate(actual, 1, backend="native")
    np.testing.assert_array_equal(actual.q, before.q)
    np.testing.assert_array_equal(actual.p, before.p)
    assert auto.operator_state_dict() == operators


@pytest.mark.parametrize("thermostat", THERMOSTATS)
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
def test_backend_switches_preserve_one_builtin_bath_stream(thermostat, integrator):
    pytest.importorskip("pyrpmd._thermal")
    native = dynamics("harmonic", integrator, thermostat)
    mixed = dynamics("harmonic", integrator, thermostat)
    actual, expected = initial(), initial()
    for backend, count in [("python", 3), ("native", 5), ("python", 4), ("native", 2)]:
        native.propagate_block(actual, count, backend="native")
        mixed.propagate_block(expected, count, backend=backend)
    assert_state_close(actual, expected)
    assert native.rng.bit_generator.state == mixed.rng.bit_generator.state
    first = native._composition.native_engine.state()
    second = mixed._composition.native_engine.state()
    assert first["rng"] == second["rng"]
    np.testing.assert_allclose(first["aux"], second["aux"], atol=3e-12, rtol=3e-12)


def save_state(path, dyn, state):
    atomic_savez(path, kind="test.builtin-native.v1", q=state.q, p=state.p,
                 step=state.step, time=state.time,
                 operators=json.dumps(dyn.operator_state_dict(), allow_nan=False))


def restore_state(path, dyn):
    with load_checkpoint(path, expected_kind="test.builtin-native.v1", require_integrity=True) as saved:
        state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
        dyn.load_operator_state_dict(json.loads(str(saved["operators"])))
    return state


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_deterministic_builtin_checkpoint_resumes_exactly(tmp_path, model, integrator):
    uninterrupted = dynamics(model, integrator)
    actual = initial()
    uninterrupted.propagate_block(actual, 7, backend="native")
    checkpoint = tmp_path / "deterministic.npz"
    save_state(checkpoint, uninterrupted, actual)
    resumed = dynamics(model, integrator)
    restored = restore_state(checkpoint, resumed)
    uninterrupted.propagate_block(actual, 9, backend="native")
    resumed.propagate_block(restored, 9, backend="native")
    np.testing.assert_array_equal(actual.q, restored.q)
    np.testing.assert_array_equal(actual.p, restored.p)
    assert (actual.step, actual.time) == (restored.step, restored.time)
    assert uninterrupted.operator_state_dict() == resumed.operator_state_dict()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("thermostat", ["pile-l", "nhc", "svr"])
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
def test_builtin_atomic_checkpoint_resumes_exactly(tmp_path, model, thermostat, integrator):
    pytest.importorskip("pyrpmd._thermal")
    uninterrupted = dynamics(model, integrator, thermostat)
    actual = initial()
    for backend, count in [("python", 3), ("native", 4), ("python", 2)]:
        uninterrupted.propagate_block(actual, count, backend=backend)
    checkpoint = tmp_path / "state.npz"
    save_state(checkpoint, uninterrupted, actual)
    resumed = dynamics(model, integrator, thermostat)
    restored = restore_state(checkpoint, resumed)
    for backend, count in [("native", 5), ("python", 3), ("native", 2)]:
        uninterrupted.propagate_block(actual, count, backend=backend)
        resumed.propagate_block(restored, count, backend=backend)
    np.testing.assert_array_equal(actual.q, restored.q)
    np.testing.assert_array_equal(actual.p, restored.p)
    assert (actual.step, actual.time) == (restored.step, restored.time)
    assert uninterrupted.operator_state_dict() == resumed.operator_state_dict()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("thermostat", ["pile-l", "nhc", "svr"])
def test_builtin_checkpoint_without_explicit_identity_is_portable(tmp_path, model, thermostat):
    pytest.importorskip("pyrpmd._thermal")

    def make():
        return Dynamics(potential(model), [2., 3., 4.], 6., 4, .004,
                        thermostat=thermostat, seed=19)

    original, fresh = make(), make()
    actual = initial()
    assert original.propagate_block(actual, 5).backend == "native"
    checkpoint = tmp_path / "implicit-identity.npz"
    save_state(checkpoint, original, actual)
    restored = restore_state(checkpoint, fresh)
    for dyn, state in [(original, actual), (fresh, restored)]:
        assert dyn.propagate_block(state, 7).backend == "native"
    np.testing.assert_array_equal(actual.q, restored.q)
    np.testing.assert_array_equal(actual.p, restored.p)
    assert original.operator_state_dict() == fresh.operator_state_dict()


def test_builtin_implicit_checkpoint_identity_rejects_changed_parameters():
    pytest.importorskip("pyrpmd._thermal")
    original = Dynamics(HarmonicPotential(.2), [2., 3., 4.], 6., 4, .004,
                        thermostat="pile-l", seed=19)
    changed = Dynamics(HarmonicPotential(.3), [2., 3., 4.], 6., 4, .004,
                       thermostat="pile-l", seed=19)
    original.propagate_block(initial(), 5)
    previous = changed.operator_state_dict()
    with pytest.raises(ValueError, match="physics|parameters|identity"):
        changed.load_operator_state_dict(original.operator_state_dict())
    assert changed.operator_state_dict() == previous


@pytest.mark.parametrize("parameter", ["k", "center"])
@pytest.mark.parametrize("api", ["propagate", "propagate_block"])
@pytest.mark.parametrize("backends", [
    pytest.param(("native", "native"), id="repeated-native"),
    pytest.param(("native", "python", "native"), id="native-python-native"),
    pytest.param(("python", "native"), id="python-first"),
])
def test_bound_builtin_mutation_is_rejected_after_rollback_and_backend_switch(parameter, api, backends):
    pytest.importorskip("pyrpmd._thermal")
    dyn = dynamics("harmonic", thermostat="pile-l")
    state = initial()
    assert dyn.propagate(state, 3, backend="native").backend == "native"
    previous = state.copy()
    operators = dyn.operator_state_dict()
    stats = copy.deepcopy(dyn.propagation_stats)
    getattr(dyn.potential, parameter)[...] += .4
    for backend in backends:
        # A failed block must retain the executor's actual bound context key.
        # Rekeying it from the mutated factory would admit the stale context
        # on the next attempt, or after briefly choosing Python scheduling.
        with pytest.raises((ValueError, RuntimeError), match="immutable"):
            getattr(dyn, api)(state, 2, backend=backend)
        np.testing.assert_array_equal(state.q, previous.q)
        np.testing.assert_array_equal(state.p, previous.p)
        assert (state.step, state.time) == (previous.step, previous.time)
        assert dyn.operator_state_dict() == operators
        assert dyn.propagation_stats == stats


@pytest.mark.parametrize("thermostat", ["pile-l", "nhc", "svr"])
def test_checkpoint_from_python_only_builtin_provider_can_gain_native_binding(tmp_path, thermostat):
    pytest.importorskip("pyrpmd._thermal")

    class PythonOnly:
        """Reproduce a pre-binding provider using the unchanged analytic PES."""
        def evaluate(self, q):
            return reference_potential.evaluate(q)

    reference_potential = potential("harmonic")
    old = Dynamics(PythonOnly(), [2., 3., 4.], 6., 4, .004, seed=19,
                   thermostat=thermostat, potential_identity="builtin-parity-harmonic")
    reference = initial()
    old.propagate_block(reference, 7, backend="python")
    checkpoint = tmp_path / "pre-binding.npz"
    save_state(checkpoint, old, reference)
    resumed = dynamics("harmonic", thermostat=thermostat)
    actual = restore_state(checkpoint, resumed)
    old.propagate_block(reference, 9, backend="python")
    assert resumed.propagate_block(actual, 9).backend == "native"
    assert_state_close(actual, reference)
    assert old.rng.bit_generator.state == resumed.rng.bit_generator.state
    previous_bath = next(iter(old._composition.baths.values())).state()
    assert previous_bath["rng"] == resumed._composition.native_engine.state()["rng"]


def test_manual_low_level_example_reports_native_steps(capsys):
    runpy.run_path(str(ROOT / "examples/manual/low_level.py"), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    stats = ast.literal_eval(next(line.split(":", 1)[1] for line in lines
                                  if line.startswith("backend counters:")))
    assert stats == {"native_steps": 100, "python_steps": 0, "fallback_reasons": {}}


def test_numpy_morse_atom_indices_have_portable_implicit_checkpoint(tmp_path):
    pytest.importorskip("pyrpmd._thermal")

    def make():
        return Dynamics(
            MorsePairPotential(atoms=np.array([0, 2], dtype=np.int64)),
            [2., 3., 4.], 6., 4, .004, thermostat="nhc", seed=19,
        )

    uninterrupted = make()
    actual = initial()
    assert uninterrupted.propagate_block(actual, 3).backend == "native"
    checkpoint = tmp_path / "numpy-indices.npz"
    save_state(checkpoint, uninterrupted, actual)
    resumed = make()
    restored = restore_state(checkpoint, resumed)
    uninterrupted.propagate_block(actual, 3)
    resumed.propagate_block(restored, 3)
    np.testing.assert_array_equal(actual.q, restored.q)
    np.testing.assert_array_equal(actual.p, restored.p)
    assert uninterrupted.operator_state_dict() == resumed.operator_state_dict()


def test_cli_init_quickstart_uses_native_steps_and_resumes_with_same_identity(tmp_path, capsys):
    from pyrpmd.cli import main

    assert main(["init", str(tmp_path / "starter")]) == 0
    starter = json.loads(capsys.readouterr().out)
    input_path = Path(starter["input"])
    output = input_path.parent / "run"
    assert main(["run", str(input_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["streams"] == 6 and summary["samples"] == 120
    assert summary["converged"] is False
    checkpoints = sorted((output / "checkpoints").glob("*.npz"))
    assert len(checkpoints) == 6
    for path in checkpoints:
        with load_checkpoint(path) as saved:
            stats = json.loads(str(saved["propagation"]))
        assert stats == {"native_steps": 120, "python_steps": 0, "fallback_reasons": {}}
    identity = json.loads((output / "manifest.json").read_text())["physics_fingerprint"]
    # Captured from unmodified main 8cc402f, before built-ins had native bindings.
    assert identity == "9cfaad50467836411241a2190591fff0d39a65eeb5147993a5082bb3eddf0623"
    assert main(["resume", str(output), "--loop-backend", "python"]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed == summary
    assert json.loads((output / "manifest.json").read_text())["physics_fingerprint"] == identity
