"""One thermostat stream survives changes of execution backend and splitting."""
import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativeEnsemble
from test_native_propagation import native_library, CompiledPotential


THERMOSTATS = ["none", "pile-l", "pile-g", "svr", "langevin", "ffl", "nhc",
              {"mode": "gle", "gle": {"A": [[1., 2.], [-2., 1.]]}},
              {"mode": "nm-gle", "gle": {"A": [[1., 2.], [-2., 1.]]}},
              {"mode": "andersen", "interval": 3}]


def config():
    rng = np.random.default_rng(72)
    q, p = rng.normal(scale=.2, size=(2, 3, 3, 3))
    q[:, 2] = [.2, -.1, .4]
    p[:, 2] = 0.
    return q, p


def dyn(potential, thermostat, integrator):
    q, _ = config()
    return Dynamics(potential, [1., 2., 3.], 2., 3, .01, seed=61,
                    thermostat=thermostat, integrator=integrator,
                    fixed_atoms=[2], fixed_positions=q[0], potential_identity="test-harmonic-.01")


@pytest.mark.parametrize("thermostat", THERMOSTATS)
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
def test_native_and_python_operators_match(native_library, thermostat, integrator):
    a = dyn(CompiledPotential(native_library), thermostat, integrator)
    b = dyn(CompiledPotential(native_library), thermostat, integrator)
    q, p = config()
    sa, sb = State(q, p), State(q, p)
    # none is intentionally the unchanged deterministic path.
    for _ in range(13):
        a._composition.step(sa, backend="native")
        b._composition.step(sb, backend="python")
    np.testing.assert_allclose(sa.q, sb.q, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(sa.p, sb.p, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("thermostat", THERMOSTATS[1:])
def test_alternating_backends_preserves_one_bath_stream(native_library, thermostat):
    a = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    b = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    q, p = config()
    sa, sb = State(q, p), State(q, p)
    for backend in ["python"]*5 + ["native"]*7 + ["python"]*4 + ["native"]*3:
        a._composition.step(sa, backend="native")
        b._composition.step(sb, backend=backend)
    np.testing.assert_allclose(sa.q, sb.q, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(sa.p, sb.p, atol=2e-11, rtol=2e-11)
    assert a._composition.native_engine.state()["rng"] == b._composition.native_engine.state()["rng"]


@pytest.mark.parametrize("thermostat", ["svr", "nhc", {"mode": "gle", "gle": {"A": [[1., 2.], [-2., 1.]]}}])
def test_explicit_bath_transfer_preserves_stream_and_physical_cache(thermostat):
    q, p = config()
    options = dict(q=q, p=p, masses=[1., 2., 3.], beta=2., dt=.01,
                   thermostat=thermostat, fixed_atoms=[2], seed=81)
    source = NativeEnsemble(**options)
    target = NativeEnsemble(**{**options, "potential": {"kind": "harmonic", "k": .7}, "seed": 999})
    ps = p.copy()
    for i in range(6):
        source.apply_thermostat(q, ps, step=i)
    physical = target.state()
    target.copy_thermostat_from(source, step=6)
    transferred = target.state()
    for field in ("q", "p", "force", "energy", "cell", "cell_p"):
        np.testing.assert_array_equal(transferred[field], physical[field])
    pt = ps.copy()
    source.apply_thermostat(q, ps, step=6)
    target.apply_thermostat(q, pt, step=6)
    np.testing.assert_array_equal(ps, pt)
    np.testing.assert_array_equal(source.state()["aux"], target.state()["aux"])
    assert source.state()["rng"] == target.state()["rng"]


def test_bath_transfer_rejects_incompatible_parameters_atomically():
    q, p = config()
    args = dict(q=q, p=p, masses=[1., 2., 3.], beta=2., dt=.01,
                thermostat="svr", fixed_atoms=[2], seed=81)
    source = NativeEnsemble(**args)
    target = NativeEnsemble(**{**args, "thermostat": {"mode": "svr", "tau": 2.}})
    before = target.state()
    with pytest.raises(ValueError, match="parameters differ"):
        target.copy_thermostat_from(source)
    assert target.state()["digest"] == before["digest"]


@pytest.mark.parametrize("thermostat", ["pile-l", "svr", "nhc", {"mode": "nm-gle", "gle": {"A": [[1., 2.], [-2., 1.]]}}])
def test_backend_switch_checkpoint_resumes_in_fresh_dynamics(native_library, thermostat):
    a = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    q, p = config()
    state = State(q, p)
    for backend in ["python"]*3 + ["native"]*4 + ["python"]*2:
        a._composition.step(state, backend=backend)
    saved = a.operator_state_dict()
    resumed_state = state.copy()
    b = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    b.load_operator_state_dict(saved)
    for backend in ["native"]*4 + ["python"]*3:
        a._composition.step(state, backend=backend)
        b._composition.step(resumed_state, backend=backend)
    np.testing.assert_array_equal(state.q, resumed_state.q)
    np.testing.assert_array_equal(state.p, resumed_state.p)
    assert a.operator_state_dict() == b.operator_state_dict()


def test_nhc_bath_state_transfer_allows_half_duration_helper():
    q, p = config()
    args = dict(q=q, p=p, masses=[1., 2., 3.], beta=2., thermostat="nhc", fixed_atoms=[2])
    helper = NativeEnsemble(**args, dt=.005)
    full = NativeEnsemble(**args, dt=.01)
    halfp = p.copy()
    helper.apply_thermostat(q, halfp)
    full.copy_thermostat_from(helper)
    fullp = halfp.copy()
    helper.apply_thermostat(q, halfp, .005)
    full.apply_thermostat(q, fullp, .005)
    np.testing.assert_array_equal(halfp, fullp)
    np.testing.assert_array_equal(helper.state()["aux"], full.state()["aux"])


@pytest.mark.parametrize("thermostat", ["pile-l", "nhc"])
def test_composed_whole_block_uses_one_native_loop(native_library, thermostat, monkeypatch):
    a = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    b = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    q, p = config()
    state, reference = State(q, p, 7, 123.4), State(q, p, 7, 123.4)

    def forbidden(*args, **kwargs):
        raise AssertionError("the configured native block entered a Python per-step scheduler")

    monkeypatch.setattr(a._composition, "step", forbidden)
    monkeypatch.setattr(a.potential, "evaluate", forbidden)
    result = a.propagate(state, 17, backend="native", sample_interval=5)
    for _ in range(17):
        b._composition.step(reference, backend="native")
    np.testing.assert_array_equal(state.q, reference.q)
    np.testing.assert_array_equal(state.p, reference.p)
    assert state.time == reference.time and state.step == reference.step
    np.testing.assert_array_equal(result.steps, [12, 17, 22, 24])
    assert result.backend == "native"
    assert a.propagation_stats["native_steps"] == 17
    # One constructor force, one explicit nonzero-clock synchronization, and
    # one endpoint force per physical step; no per-step Python force callbacks.
    assert a._composition.native_engine.state()["force_evaluations"] == 19


@pytest.mark.parametrize("thermostat", ["pile-l", "nhc"])
def test_composed_prefix_keeps_arbitrary_mask_and_diagnostics_native(native_library, thermostat, monkeypatch):
    a = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    b = dyn(CompiledPotential(native_library), thermostat, "cayley-bcb")
    q, p = config()
    state, reference = State(q, p, 7, 123.4), State(q, p, 7, 123.4)
    mask = np.array([True, False, False, True, False, True, False, False, True,
                     False, True, False, False, False, True, False, True])

    def forbidden(*args, **kwargs):
        raise AssertionError("native prefix entered a Python per-step scheduler or force callback")

    monkeypatch.setattr(a._composition, "step", forbidden)
    monkeypatch.setattr(a.potential, "evaluate", forbidden)
    diagnostics = {"energy_parts": True, "ring_radii": True}
    actual = a.propagate_block(state, len(mask), backend="native", observe=mask, diagnostics=diagnostics)
    expected = b.propagate_block(reference, len(mask), backend="python", observe=mask, diagnostics=diagnostics)
    assert actual.backend == "native" and actual.completed == len(mask)
    np.testing.assert_allclose(state.q, reference.q, atol=2e-12)
    np.testing.assert_allclose(state.p, reference.p, atol=2e-12)
    np.testing.assert_array_equal(actual.steps, expected.steps)
    np.testing.assert_array_equal(actual.times, expected.times)
    for field in ("centroids", "centroid_momenta", "energies"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=2e-12)
    for key in diagnostics:
        np.testing.assert_allclose(actual.diagnostics[key], expected.diagnostics[key], atol=2e-12)
    assert a.propagation_stats["native_steps"] == len(mask)
