"""Streamed FES extensions preserve independent thermostat and batching state."""

import json
import threading

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.dynamics import Dynamics
from pyrpmd.workflows import fes

from test_sampling_composition import _config, _payload, _system


@pytest.mark.parametrize("mode", ["nhc", "gle", "svr"])
def test_streamed_bath_extension_restarts_exactly_after_batched_observation_failure(
        tmp_path, monkeypatch, mode):
    pytest.importorskip("pyrpmd._thermal")
    config, system = _config(mode), _system()
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    expected = fes.run_fes(config, system, reference, production_steps=52, return_data=True)
    fes.run_fes(config, system, resumed)
    assert all("sample_stream" not in _payload(path)
               for path in (resumed / "checkpoints").glob("*.npz"))

    config["parallel"] = {"trajectory_batch_size": 2}
    fes.run_fes(config, system, resumed, production_steps=36)
    for path in (resumed / "checkpoints").glob("*.npz"):
        saved = _payload(path)
        assert "sample_stream" in saved
        assert json.loads(str(saved["operator_state"]))["sampling"]
    records = fes.validate_fes_checkpoints(config, system, resumed, production_steps=52)
    assert [record["samples"] for record in records] == [18, 18]

    original_energy = Dynamics.energy
    failed = False
    lock = threading.Lock()

    def fail_after_thermostat_step(dynamics, state, bias=None):
        nonlocal failed
        with lock:
            if state.step == 46 and not failed:
                failed = True
                raise RuntimeError("injected extended observation failure")
        return original_energy(dynamics, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "energy", fail_after_thermostat_step)
        with pytest.raises(RuntimeError, match="extended observation failure"):
            fes.run_fes(config, system, resumed, production_steps=52)
    assert failed
    partial_steps = sorted(int(_payload(path)["elapsed"])
                           for path in (resumed / "checkpoints").glob("*.npz"))
    assert partial_steps == [45, 56]

    # Regroup the same independent streams after one lane has already finished.
    config["parallel"]["trajectory_batch_size"] = 1
    actual = fes.run_fes(config, system, resumed, production_steps=52, return_data=True)
    for left, right in zip(expected["streams"], actual["streams"]):
        for name in ("samples", "energies", "last_q", "last_p"):
            np.testing.assert_array_equal(left[name], right[name], err_msg=name)
    for path in (reference / "checkpoints").glob("*.npz"):
        a, b = _payload(path), _payload(resumed / "checkpoints" / path.name)
        # Chunk boundaries legitimately differ after a split/failed extension;
        # the reconstructed arrays above and complete physical state must not.
        for key in ("q", "p", "step", "time", "elapsed", "rng", "operator_state",
                    "propagation", "fingerprint"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    np.testing.assert_array_equal(expected["data"]["free_energy"], actual["data"]["free_energy"])


@pytest.mark.parametrize("mode", ["nhc", "gle"])
@pytest.mark.parametrize("stream", [False, True], ids=["arrays", "stream"])
def test_extension_preflight_rejects_empty_bath_before_provider_or_writes(
        tmp_path, monkeypatch, mode, stream):
    pytest.importorskip("pyrpmd._thermal")
    config, system = _config(mode), _system()
    config["umbrella"]["centers"] = config["umbrella"]["centers"][:1]
    _, tasks = fes._make_tasks(config, system, tmp_path)
    task = tasks[0]
    fes._sample_window(task)
    if stream:
        task["production_steps"] = 36
        fes._sample_window(task)
    path = fes._window_checkpoint(task)
    with load_checkpoint(path) as saved:
        payload = dict(saved.items())
    operators = json.loads(str(payload["operator_state"]))
    assert operators["sampling"]
    operators["sampling"] = {}
    payload["operator_state"] = json.dumps(operators)
    atomic_savez(path, **payload)
    committed = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid checkpoint caused a PES request or checkpoint write")

    monkeypatch.setattr(fes, "task_potential", forbidden)
    monkeypatch.setattr(fes, "_atomic_npz", forbidden)
    with pytest.raises(ValueError, match="missing thermostat auxiliary state"):
        fes.validate_fes_checkpoints(config, system, tmp_path, production_steps=52)
    assert path.read_bytes() == committed
