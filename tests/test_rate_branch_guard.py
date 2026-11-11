"""An umbrella must reject the far xi branch before it contributes samples."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import CVStack, RPMDrateCoordinate, RPMDrateBranchError
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import FreePotential
from pyrpmd.workflows.biased_sampling import SamplingRuntime, failure_writer, run_biased_sampling
from pyrpmd.workflows.fes import steer_to_center
from pyrpmd.workflows.sampling import AndersenSchedule


MASSES = np.ones(3)
TS = np.array([[-1.757, 0., 0.], [0., 0., 0.], [1.757, 0., 0.]])
FAR = np.array([[-1.4, 0., 0.], [0., 0., 0.], [40., 0., 0.]])


def coordinate(guard=True):
    return RPMDrateCoordinate([[0], [1, 2]], MASSES, TS, [[0, 1]], [[1, 2]], 10.,
                             equivalent_paths=[[0, 2, 1]], regular_branch_only=guard)


@pytest.mark.parametrize('method', ['value', 'centroid_value_gradient',
                                   'centroid_value_gradient_hessian', 'centroid_value_gradient_hvp'])
def test_far_branch_rejected_by_all_python_coordinate_paths(method):
    cv = coordinate()
    args = (FAR[None],) if method == 'value' else (FAR,)
    if method.endswith('_hvp'):
        args += (np.ones_like(FAR),)
    with pytest.raises(RPMDrateBranchError, match='regular branch') as caught:
        getattr(cv, method)(*args)
    assert caught.value.branch['denominator'] < 0
    assert coordinate(False).branch(FAR)['regular'] is False
    assert np.isfinite(coordinate(False).value(FAR[None])).all()


def test_high_window_steering_remains_regular():
    cv = CVStack([coordinate()])
    q = steer_to_center(np.repeat(TS[None], 256, axis=0), cv, [1.1], MASSES)
    assert cv.cvs[0].branch(q)['regular']
    np.testing.assert_allclose(cv.value(q), [1.1], atol=1e-7)
    with pytest.raises(RPMDrateBranchError):
        steer_to_center(FAR[None], cv, [1.1], MASSES)


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_crossing_stops_sampling_and_preserves_regular_prefix(tmp_path, backend):
    if backend == 'native' and not native_capabilities()['available']:
        pytest.skip('native extension unavailable')
    cv = CVStack([coordinate()])
    q = np.array([[[-1.4, 0., 0.], [0., 0., 0.], [2., 0., 0.]]])
    p = np.zeros_like(q)
    p[0, 2, 0] = 2.0
    state = State(q.copy(), p.copy(), step=1, time=0.)
    dynamics = Dynamics(FreePotential(), MASSES, 100., 1, 1., native=backend == 'native',
                        loop_backend=backend)
    bias = UmbrellaBias(cv, [1.1], [[1e-10]])
    runtime = SamplingRuntime(state, collision_clock=AndersenSchedule(100))
    sampler = SimpleNamespace(thermostat='andersen', step=lambda d, s, b: d.step(s, b))
    path = tmp_path / 'failure.json'
    saved = []
    with pytest.raises(Exception, match='regular branch'):
        run_biased_sampling(runtime, dynamics, cv, bias, neq=0, nprod=20, interval=1,
                            checkpoint_steps=20, sampling={}, sampler=sampler,
                            save=lambda r: saved.append(r.state.copy()),
                            failure=failure_writer(path, dynamics, cv, identity={'method': 'umbrella'},
                                                   neq=0, nprod=20))
    assert 0 < runtime.elapsed < 20
    assert coordinate().branch(state.q)['regular']
    assert len(runtime.values) == runtime.elapsed
    record = json.loads(path.read_text())
    assert record['status'] == 'history_failed'
    assert record['state']['elapsed_steps'] == runtime.elapsed
    assert record['propagation'][backend + '_steps'] > 0
    np.testing.assert_array_equal(saved[-1].q, state.q)
    if backend == 'python':
        assert record['attempted_branch']['denominator'] < 0


def test_completed_far_checkpoint_is_rejected_and_recorded(tmp_path):
    cv = CVStack([coordinate()])
    dynamics = Dynamics(FreePotential(), MASSES, 100., 1, .01, native=False)
    runtime = SamplingRuntime(State(FAR[None].copy(), np.zeros((1, 3, 3)), step=10), elapsed=10)
    path = tmp_path / 'failure.json'
    with pytest.raises(RPMDrateBranchError):
        run_biased_sampling(runtime, dynamics, cv, neq=0, nprod=10, interval=1,
                            checkpoint_steps=10, sampling={}, sampler=SimpleNamespace(thermostat='pile_l'),
                            failure=failure_writer(path, dynamics, cv, identity={'method': 'umbrella'},
                                                   neq=0, nprod=10))
    assert json.loads(path.read_text())['attempted_branch']['denominator'] < 0
    assert runtime.elapsed == 10


@pytest.mark.parametrize('api', ['propagate', 'propagate_block'])
def test_native_descriptor_formats_reject_far_branch(api):
    if not native_capabilities()['available']:
        pytest.skip('native extension unavailable')
    cv = CVStack([coordinate()])
    dynamics = Dynamics(FreePotential(), MASSES, 100., 1, .01, loop_backend='native')
    state = State(FAR[None].copy(), np.zeros((1, 3, 3)))
    bias = UmbrellaBias(cv, [1.1], [[.1]])
    with pytest.raises(Exception, match='regular branch'):
        getattr(dynamics, api)(state, 2, bias=bias)
    np.testing.assert_array_equal(state.q, FAR[None])
    assert state.step == 0
