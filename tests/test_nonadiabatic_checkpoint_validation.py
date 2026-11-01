"""Logical and physical checkpoint checks beyond the NPZ integrity envelope."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.surface_hopping import RPSHDynamics, RPSHState
from pyrpmd.workflows.nonadiabatic import _append, _load, _observe, _save


KIND = "PyRPMD.nonadiabatic_trajectory.v1"


class ConstantMatrix:
    def evaluate(self, q):
        matrix = np.broadcast_to(np.array([[.2, .1], [.1, -.2]]), (len(q), 2, 2)).copy()
        return DiabaticResult(matrix, np.zeros((*matrix.shape, *q.shape[1:])))


def bundle(tmp_path, method="nrpmd", *, step=0, target=0, stride=1, signed=False):
    rng = np.random.default_rng(531)
    potential = ConstantMatrix()
    nuclear = State(np.zeros((2, 1, 3)), np.full((2, 1, 3), .2))
    if method == "nrpmd":
        dynamics = NRPMDDynamics(potential, [1.], 3., 2, .1, 2, native=False)
        state = NRPMDState.from_nuclear(nuclear, 2, 0, 81)
        if signed:
            state.mapping_q[:] = [2., 0.]
            state.mapping_p[:] = 0.
    else:
        dynamics = RPSHDynamics(potential, [1.], 3., 2, .1, 2, native=False, seed=rng)
        state = RPSHState(nuclear.q, nuclear.p,
                          coefficients=[1/np.sqrt(2), 1j/np.sqrt(2)], active_surface=0)
    config = {"method": method, "beads": 2, "dt_au": .1,
              "ensemble": {"steps": target, "sample_stride": stride}}
    history = {"steps": [0], **{key: [value] for key, value in _observe(dynamics, state).items()}}
    for _ in range(step):
        dynamics.step(state)
        if state.step % stride == 0 or state.step == target:
            _append(history, dynamics, state)
    path = tmp_path/f"{method}.npz"
    _save(path, config, "identity", 0, state, rng, history)
    return path, config, dynamics, state, rng


def rewrite(path, change):
    with load_checkpoint(path, require_integrity=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    change(payload)
    # Recompute the storage hashes to exercise logical schema validation,
    # rather than merely triggering the container's checksum rejection.
    atomic_savez(path, kind=KIND, **payload)


@pytest.mark.parametrize("method,key,dtype", [
    ("nrpmd", "history_density", np.float64),
    ("nrpmd", "history_density", np.complex64),
    ("nrpmd", "history_energy", np.complex128),
    ("nrpmd", "history_norm", np.complex128),
    ("nrpmd", "history_centroid_q", np.complex128),
    ("nrpmd", "history_centroid_p", np.float32),
    ("nrpmd", "history_steps", np.float64),
    ("nrpmd", "mapping_q", np.float32),
    ("nrpmd", "q", np.int64),
    ("nrpmd", "time", np.complex128),
    ("rpsh", "coefficients", np.float64),
    ("rpsh", "coefficients", np.complex64),
    ("rpsh", "history_amplitude", np.complex128),
    ("rpsh", "history_active", np.float64),
])
def test_repacked_lossy_or_wrong_observation_and_state_dtypes_are_rejected(tmp_path, method, key, dtype):
    path, config, dynamics, _, rng = bundle(tmp_path, method)
    def change(payload):
        value = payload[key]
        # Explicit real extraction simulates an exporter silently losing phase.
        if np.iscomplexobj(value) and not np.issubdtype(dtype, np.complexfloating):
            value = value.real
        payload[key] = value.astype(dtype)
    rewrite(path, change)
    with pytest.raises(CheckpointError):
        _load(path, config, "identity", 0, dynamics, rng)


@pytest.mark.parametrize("method", ["nrpmd", "rpsh"])
def test_current_state_phase_must_match_its_final_saved_observation(tmp_path, method):
    path, config, dynamics, _, rng = bundle(tmp_path, method)
    def change(payload):
        if method == "nrpmd":
            payload["mapping_p"] *= -1.
        else:
            payload["coefficients"] = payload["coefficients"].conj()
    rewrite(path, change)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(CheckpointError, match="state/final observation mismatch for density"):
        _load(path, config, "identity", 0, dynamics, rng)
    assert rng.bit_generator.state == before


@pytest.mark.parametrize("key", ["history_energy", "history_centroid_q", "history_centroid_p"])
def test_final_real_observations_must_match_the_state(tmp_path, key):
    path, config, dynamics, _, rng = bundle(tmp_path)
    rewrite(path, lambda data: data[key].__setitem__(-1, data[key][-1]+.2))
    with pytest.raises(CheckpointError, match="state/final observation mismatch"):
        _load(path, config, "identity", 0, dynamics, rng)


def test_focused_current_mapping_norm_and_historical_norm_are_validated(tmp_path):
    path, config, dynamics, _, rng = bundle(tmp_path)
    rewrite(path, lambda data: data["mapping_q"].__imul__(1.1))
    with pytest.raises(CheckpointError, match="focused NRPMD mapping norm"):
        _load(path, config, "identity", 0, dynamics, rng)
    path, config, dynamics, _, rng = bundle(tmp_path)
    rewrite(path, lambda data: data["history_norm"].__iadd__(.2))
    with pytest.raises(CheckpointError, match="electronic norm history"):
        _load(path, config, "identity", 0, dynamics, rng)


def test_trace_and_rpsh_purity_do_not_impose_psd_on_signed_mmst_estimators(tmp_path):
    path, config, dynamics, state, rng = bundle(tmp_path, signed=True)
    restored, history = _load(path, config, "identity", 0, dynamics, rng)
    np.testing.assert_array_equal(np.diag(history["density"][0]).real, [1.5, -.5])
    np.testing.assert_array_equal(restored.mapping_q, state.mapping_q)
    rewrite(path, lambda data: data["history_density"].__imul__(2.))
    with pytest.raises(CheckpointError, match="unit trace"):
        _load(path, config, "identity", 0, dynamics, rng)

    path, config, dynamics, _, rng = bundle(tmp_path, "rpsh")
    rewrite(path, lambda data: data["history_density"].__setitem__(0, .5*np.eye(2)))
    with pytest.raises(CheckpointError, match="pure amplitude density"):
        _load(path, config, "identity", 0, dynamics, rng)


@pytest.mark.parametrize("population", [[-.1, 1.1], [.4, .8]])
def test_rpsh_historical_amplitude_populations_are_normalized_and_nonnegative(tmp_path, population):
    path, config, dynamics, _, rng = bundle(tmp_path, "rpsh")
    rewrite(path, lambda data: data["history_amplitude"].__setitem__(0, population))
    with pytest.raises(CheckpointError, match="amplitude populations"):
        _load(path, config, "identity", 0, dynamics, rng)


@pytest.mark.parametrize("method", ["nrpmd", "rpsh"])
def test_checkpoint_between_observation_frames_keeps_earlier_frame_and_restores_rng(tmp_path, method):
    path, config, dynamics, state, rng = bundle(tmp_path, method, step=3, target=5, stride=2)
    expected_rng = copy.deepcopy(rng.bit_generator.state)
    rng.random(13)
    restored, history = _load(path, config, "identity", 0, dynamics, rng)
    np.testing.assert_array_equal(history["steps"], [0, 2])
    np.testing.assert_array_equal(restored.q, state.q)
    assert not np.array_equal(history["centroid_q"][-1], restored.q.mean(axis=0))
    assert rng.bit_generator.state == expected_rng
    if method == "rpsh":
        assert dynamics.rng is rng


@pytest.mark.parametrize("extra", [False, True])
def test_full_logical_inventory_is_required_even_with_valid_storage_hashes(tmp_path, extra):
    path, config, dynamics, _, rng = bundle(tmp_path)
    def change(payload):
        if extra:
            payload["unused_state"] = np.array([0.])
        else:
            del payload["mapping_p"]
    rewrite(path, change)
    with pytest.raises(CheckpointError, match="inventory"):
        _load(path, config, "identity", 0, dynamics, rng)


def test_invalid_rng_does_not_replace_the_running_generator(tmp_path):
    path, config, dynamics, _, rng = bundle(tmp_path, "rpsh")
    def change(payload):
        metadata = json.loads(payload["metadata"].item())
        metadata["rng"] = {"bit_generator": "not-a-generator"}
        payload["metadata"] = np.asarray(json.dumps(metadata))
    rewrite(path, change)
    before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(CheckpointError, match="random generator"):
        _load(path, config, "identity", 0, dynamics, rng)
    assert rng.bit_generator.state == before
