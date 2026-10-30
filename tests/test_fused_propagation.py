"""Fused loops must preserve numerical and callback semantics, not just speed."""
from pathlib import Path
import copy
import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.cv import Cartesian, CVStack, Distance
from pyrpmd.bias import UmbrellaBias
from pyrpmd.potentials.cabi import CABIPotential
from pyrpmd.propagation import NativeUnavailableError, scientific_config

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not native_capabilities()["available"], reason="native extension unavailable")


@pytest.fixture(scope="module")
def libraries(tmp_path_factory):
    if shutil.which("cc") is None:
        pytest.skip("C compiler unavailable")
    directory = tmp_path_factory.mktemp("fused_pes")
    source = (ROOT / "native/examples/harmonic_pes.c").read_text()
    paths = {}
    for name, text in (("ok", source), ("domain", source.replace('if(!isfinite(q))', 'if(!isfinite(q)||fabs(q)>1.)'))):
        c = directory / f"{name}.c"; so = directory / f"{name}.so"
        c.write_text(text)
        subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-std=c11", "-I", str(ROOT / "native/include"), str(c), "-o", str(so)], check=True)
        paths[name] = so
    return paths


@pytest.mark.parametrize("beads", [1, 2, 7, 16, 32])
def test_cabi_block_matches_reference_observations_and_clock(libraries, beads):
    with_pes = CABIPotential(libraries["ok"])
    rng = np.random.default_rng(12)
    q = rng.normal(scale=.1, size=(beads, 1, 3)); p = rng.normal(scale=.2, size=q.shape)
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 2)])
    bias = UmbrellaBias(cv, [.02, -.03], [[.1, .02], [.02, .2]])
    dyn = Dynamics(with_pes, [10.], 4., beads, .001)
    a, b = State(q, p, 7, .7), State(q, p, 7, .7)
    native = dyn.propagate(a, 23, bias=bias, observe=cv, backend="native", sample_interval=7)
    reference = dyn.propagate(b, 23, bias=bias, observe=cv, backend="python", sample_interval=7)
    np.testing.assert_array_equal(native.steps, [14, 21, 28, 30])
    np.testing.assert_allclose(a.q, b.q, atol=3e-14, rtol=1e-12)
    np.testing.assert_allclose(a.p, b.p, atol=3e-13, rtol=1e-12)
    np.testing.assert_allclose(native.values, reference.values, atol=3e-14)
    np.testing.assert_allclose(native.energies, reference.energies, atol=1e-12)
    np.testing.assert_allclose(native.centroids, reference.centroids, atol=3e-14)
    assert a.step == b.step and a.time == b.time
    with_pes.close()


def test_block_partition_and_disabled_centroid_output(libraries):
    pes = CABIPotential(libraries["ok"])
    dyn = Dynamics(pes, [5.], 3., 8, .001)
    a = State(np.ones((8, 1, 3))*.1, np.ones((8, 1, 3))*.2); b = a.copy()
    one = dyn.propagate(a, 97, backend="native", record_centroids=False, sample_interval=97)
    dyn.propagate(b, 19, backend="native")
    dyn.propagate(b, 78, backend="native")
    np.testing.assert_array_equal(a.q, b.q); np.testing.assert_array_equal(a.p, b.p)
    assert a.time == b.time and one.centroids is None
    pes.close()


def test_failed_block_is_atomic_and_closed_context_rejected(libraries):
    pes = CABIPotential(libraries["domain"])
    dyn = Dynamics(pes, [1.], 1., 1, .1)
    state = State([[[.8, 0, 0]]], [[[3., 0, 0]]], 12, 1.2)
    previous = state.copy()
    with pytest.raises(RuntimeError, match="native PES frame"):
        dyn.propagate(state, 10, backend="native")
    np.testing.assert_array_equal(state.q, previous.q); np.testing.assert_array_equal(state.p, previous.p)
    assert (state.step, state.time) == (12, 1.2)
    pes.close()
    with pytest.raises(RuntimeError, match="closed"):
        dyn.propagate(state, 1, backend="native")


def test_python_pes_and_overridden_cv_are_not_bypassed(libraries):
    class ChangedCartesian(Cartesian):
        def centroid_value_gradient(self, q):
            value, grad = super().centroid_value_gradient(q)
            return 3*value+7, 3*grad
    pes = CABIPotential(libraries["ok"])
    dyn = Dynamics(pes, [2.], 1., 2, .01)
    state = State(np.ones((2, 1, 3))*.1, np.zeros((2, 1, 3)))
    cv = ChangedCartesian(0)
    old = state.copy()
    with pytest.raises(NativeUnavailableError):
        dyn.propagate(state, 2, observe=cv, backend="native")
    np.testing.assert_array_equal(state.q, old.q)
    out = dyn.propagate(state, 2, observe=cv)
    assert out.backend == "python" and "ChangedCartesian" in out.fallback_reason
    np.testing.assert_allclose(out.values[-1], cv.value(state.q))
    class PythonPES:
        def __init__(self): self.calls = 0
        def evaluate(self, q):
            self.calls += 1
            return .5*np.sum(q*q, axis=(1, 2)), -q
    py = PythonPES(); dyn = Dynamics(py, [2.], 1., 2, .01)
    result = dyn.propagate(old, 2)
    assert result.backend == "python" and py.calls > 0
    pes.close()


def test_invalid_controls_and_fingerprints(libraries):
    dyn = Dynamics(CABIPotential(libraries["ok"]), [1.], 1., 1, .01)
    state = State([[[.1, 0, 0]]], [[[0, 0, 0]]])
    for bad in [0, -1, True, 2.5]:
        with pytest.raises(ValueError): dyn.propagate(state, bad)
    config = {"dynamics": {"timestep": "0.1 fs", "loop_backend": "native", "block_steps": 97}}
    assert scientific_config(config) == {"dynamics": {"timestep": "0.1 fs"}}
    assert config["dynamics"]["block_steps"] == 97


@pytest.mark.skipif(not (ROOT / "benchmarks/h3/build/libbkmp2.so").is_file(), reason="optional real BKMP2 not built")
def test_real_rate_metric_surface_and_andersen_rng():
    from benchmarks.h3.pes import BKMP2Potential
    from benchmarks.loop_native.benchmark import initial, MASS_H, BETA, DT_AU
    from pyrpmd.cv import RPMDrateCoordinate
    from pyrpmd.bias import CompositeBias, FluxMetricBias
    q, p = initial(16)
    coordinate = RPMDrateCoordinate([[0], [1, 2]], MASS_H,
        [[-1.757, 0, 0], [0, 0, 0], [1.757, 0, 0]], [[0, 1]], [[1, 2]], 10., equivalent_paths=[[0, 2, 1]])
    cv = CVStack([coordinate])
    center = cv.value(q)
    bias = CompositeBias(UmbrellaBias(cv, center, [[.2]]), FluxMetricBias(coordinate, MASS_H, BETA))
    dyn = Dynamics(BKMP2Potential(), MASS_H, BETA, 16, DT_AU)
    a, b = State(q, p), State(q, p)
    na = dyn.propagate(a, 100, bias=bias, observe=coordinate.dividing_surface(1.00925), backend="native")
    py = dyn.propagate(b, 100, bias=bias, observe=coordinate.dividing_surface(1.00925), backend="python")
    np.testing.assert_allclose(a.q, b.q, atol=3e-11, rtol=1e-11)
    np.testing.assert_allclose(a.p, b.p, atol=3e-9, rtol=1e-10)
    np.testing.assert_allclose(na.values, py.values, atol=3e-11)
    np.testing.assert_allclose(na.energies, py.energies, atol=2e-10)
    a, b = State(q, p), State(q, p)
    dn = Dynamics(BKMP2Potential(), MASS_H, BETA, 16, DT_AU, seed=98, loop_backend="native", block_steps=11)
    dp = Dynamics(BKMP2Potential(), MASS_H, BETA, 16, DT_AU, seed=98, loop_backend="python")
    dn.thermalize(a, 73, interval=7, fragments=[[0], [1, 2]])
    dp.thermalize(b, 73, interval=7, fragments=[[0], [1, 2]])
    np.testing.assert_allclose(a.q, b.q, atol=2e-11, rtol=1e-11)
    np.testing.assert_allclose(a.p, b.p, atol=3e-9, rtol=1e-10)
    assert dn.rng.bit_generator.state == dp.rng.bit_generator.state
    assert a.step == b.step and a.time == b.time


def test_drpmd_failure_replay_preserves_last_valid_state(libraries, tmp_path):
    from pyrpmd.workflows.drpmd import _trajectory
    config = {'beads': 1, 'dt_au': .1, 'collision': {'max_steps': 30},
              'outcomes': {'check_interval': 10},
              'output': {'trajectory_stride': 10, 'checkpoint_steps': 10}}
    outcomes = {}
    for backend in ('native', 'python'):
        dyn = Dynamics(CABIPotential(libraries['domain']), [1.], 1., 1, .1, loop_backend=backend)
        s = State([[[.5, 0, 0]]], [[[1., 0, 0]]])
        outcomes[backend] = _trajectory(dyn, s, None, config, tmp_path/backend, 0, 0, {}, 'same', False)
    a, b = outcomes['native'], outcomes['python']
    assert a['status'] == b['status'] == 'failed'
    assert a['steps'] == b['steps'] and a['steps'] > 0
    for key in ('final_q', 'final_p'):
        with np.load(tmp_path/'native/trajectories/trajectory-00000000.npz') as n, np.load(tmp_path/'python/trajectories/trajectory-00000000.npz') as p:
            np.testing.assert_array_equal(n[key], p[key])


def test_stateful_or_subclass_pes_does_not_skip_python(libraries):
    class Altered(CABIPotential):
        def evaluate(self, q):
            result = super().evaluate(q)
            result.forces *= .9
            return result
    dyn = Dynamics(Altered(libraries['ok']), [1.], 1., 1, .01)
    state = State([[[.1, 0, 0]]], [[[0., 0, 0]]])
    assert dyn.propagate(state, 2).backend == 'python'
    pes = CABIPotential(libraries['ok'])
    pes.metadata['capabilities']['stateful'] = True
    dyn = Dynamics(pes, [1.], 1., 1, .01)
    assert dyn.propagate(state, 2).backend == 'python'


def test_custom_observer_error_rolls_back_reference_block(libraries):
    class Failing(Cartesian):
        def value(self, q):
            raise RuntimeError('custom observation failed')
    dyn = Dynamics(CABIPotential(libraries['ok']), [1.], 1., 1, .01)
    state = State([[[.1, 0, 0]]], [[[0., 0, 0]]])
    saved = state.copy()
    with pytest.raises(RuntimeError, match='custom observation failed'):
        dyn.propagate(state, 7, observe=Failing(0))
    np.testing.assert_array_equal(state.q, saved.q)
    np.testing.assert_array_equal(state.p, saved.p)
    assert state.step == 0


def test_instance_stack_observation_is_not_bypassed(libraries):
    from pyrpmd.cv import CVStack
    cv = CVStack([Cartesian(0)])
    calls = []
    def custom_value(q):
        calls.append(True)
        return np.array([42.])
    cv.value = custom_value
    dyn = Dynamics(CABIPotential(libraries['ok']), [1.], 1., 1, .01)
    state = State([[[.1, 0, 0]]], [[[0., 0, 0]]])
    result = dyn.propagate(state, 3, observe=cv)
    assert result.backend == 'python' and len(calls) == 3
    np.testing.assert_array_equal(result.values, np.full((3, 1), 42.))


def test_changed_python_pes_function_is_not_bypassed(libraries):
    from pyrpmd.potentials.python import PythonPotential
    pes = PythonPotential(CABIPotential(libraries['ok']))
    original = pes.fn
    calls = []
    def altered(q):
        calls.append(True)
        result = original(q)
        result.forces *= .5
        return result
    pes.fn = altered
    dyn = Dynamics(pes, [1.], 1., 1, .01)
    state = State([[[.1, 0, 0]]], [[[0., 0, 0]]])
    result = dyn.propagate(state, 3)
    assert result.backend == 'python' and calls


def test_nested_rate_cv_override_rejects_native():
    from pyrpmd.cv import RPMDrateCoordinate
    from pyrpmd.propagation import cv_descriptor
    c = RPMDrateCoordinate([[0], [1, 2]], [1., 1., 1.],
                           [[-1., 0, 0], [0., 0, 0], [1., 0, 0]], [[0, 1]], [[1, 2]], 10.)
    surface = c.dividing_surface(1.)
    c.com.centroid_value_gradient_hessian = lambda q: None
    with pytest.raises(NativeUnavailableError, match='overridden'):
        cv_descriptor(surface, 3)
