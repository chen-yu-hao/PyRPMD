"""Unthermostatted materials production advances in event-bounded native blocks.

The reference for every comparison is the single-step path the workflow used
before blocks (``Dynamics.propagate(state, 1)``), selected here by declaring
the block path unavailable. Block lengths are read from the native entry
point itself, and PES batches are counted inside an audited C ABI library.
"""

import copy
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd.block_propagation import PropagationError
from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.system import System
import pyrpmd.workflows.materials as materials
from pyrpmd.workflows.materials import _read_stream, _sample_replica


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if not native_capabilities()["available"] or shutil.which("cc") is None:
        pytest.skip("native extension and C compiler required")
    from pyrpmd import _native

    for marker in ("BLOCK_ENERGY_CHECK_API", "DRPMD_OUTCOMES_API"):
        if getattr(_native, marker, 0) < 1:
            pytest.skip("rebuild the native extension")
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/examples/harmonic_pes.c").read_text()
    controls = """
static int audit_calls=0, audit_fail_at=-1;
void audit_reset(int fail) { audit_calls=0; audit_fail_at=fail; }
int audit_get_calls(void) { return audit_calls; }
"""
    source = source.replace("#include <stdio.h>", "#include <stdio.h>\n" + controls)
    marker = "pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1"
    opening = source.index("{", source.index(marker)) + 1
    source = source[:opening] + """
 ++audit_calls;
 if(audit_calls==audit_fail_at) { message(e,"injected CABI failure"); return PYRPMD_INTERNAL_ERROR_V1; }
""" + source[opening:]
    directory = tmp_path_factory.mktemp("materials-block-pes")
    cfile, path = directory / "harmonic.c", directory / "harmonic.so"
    cfile.write_text(source)
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-std=c11", "-I", str(root / "native/include"),
                    str(cfile), "-o", str(path)], check=True, capture_output=True)
    return path


@pytest.fixture
def audit(library):
    controls = ctypes.CDLL(str(library))
    controls.audit_reset.argtypes = [ctypes.c_int]
    controls.audit_reset.restype = None
    controls.audit_get_calls.restype = ctypes.c_int
    controls.audit_reset(-1)
    yield controls
    controls.audit_reset(-1)


@pytest.fixture
def native_calls(monkeypatch):
    """Record the step count of every fused native call, unchanged otherwise."""
    from pyrpmd import _native

    calls = {"advance": [], "run_block": [], "options": [], "hooks": {}}
    advance, run_block = _native.advance, _native.run_block

    def recorded_advance(context, binding, q, p, options, mask, records, states):
        calls["advance"].append(len(mask))
        calls["options"].append({k: options.get(k) for k in ("reuse_forces", "check_energy")})
        hook = calls["hooks"].get("before_advance")
        if hook is not None:
            hook(len(calls["advance"]))
        result = advance(context, binding, q, p, options, mask, records, states)
        hook = calls["hooks"].get("after_advance")
        if hook is not None:
            hook(len(calls["advance"]))
        return result

    def recorded_run_block(context, binding, q, p, steps, *args):
        calls["run_block"].append(int(steps))
        hook = calls["hooks"].get("before_run_block")
        if hook is not None:
            hook(len(calls["run_block"]))
        return run_block(context, binding, q, p, steps, *args)

    monkeypatch.setattr(_native, "advance", recorded_advance)
    monkeypatch.setattr(_native, "run_block", recorded_run_block)
    return calls


NEQ, NPROD, INTERVAL, EVERY = 5, 40, 6, 9


def config(library, *, block_steps, backend="native", storage="memory"):
    return {
        "task": "diffusion", "beta": 1.7, "beads": 4, "dt_au": 0.04, "seed": 2711,
        "native": "required",
        "potential": {"provider": "cabi", "library": str(library)},
        "sampling": {"equilibration_steps": NEQ, "production_steps": NPROD,
                     "sample_interval": INTERVAL, "replicas": 1,
                     "thermostat": "andersen", "andersen_interval": 3},
        "dynamics": {"loop_backend": backend, "block_steps": block_steps},
        "diffusion": {"atom_indices": [0], "axes": [0, 1, 2], "bootstrap": 0},
        # Neither the sample interval nor the checkpoint interval divides the other.
        "output": {"checkpoint_steps": EVERY, "trajectory_storage": storage},
    }


def task(cfg, output):
    system = System(("H",), np.array([2.3]), np.array([[0.4, -0.2, 0.1]]))
    return {"config": cfg, "system": system, "output": str(output), "replica": 0}


def expected_blocks(block_steps, start=NEQ):
    """Independent event schedule: samples, checkpoints and the run end."""
    total = NEQ + NPROD
    events = ({NEQ + k * INTERVAL for k in range(1, NPROD // INTERVAL + 1)}
              | set(range(EVERY, total + 1, EVERY)) | {total})
    blocks, elapsed = [], start
    while elapsed < total:
        stop = min(e for e in events if e > elapsed)
        blocks.append(min(block_steps, stop - elapsed))
        elapsed += blocks[-1]
    return blocks


def expected_sampling_blocks(block_steps):
    """Andersen equilibration blocks stop at refresh and handoff events."""
    blocks, elapsed = [], 0
    while elapsed < NEQ:
        count = min(block_steps, 4096, NEQ - elapsed,
                    3 - elapsed % 3, EVERY - elapsed % EVERY)
        blocks.append(count)
        elapsed += count
    return blocks


def single_step_reference(cfg, output, monkeypatch):
    """The pre-block production loop: one Dynamics.propagate call per step."""
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "can_fuse", lambda self, **kwargs: False)
        return _sample_replica(task(cfg, output))


def assert_same_stream(left, right, *, propagation=True):
    for key in ("positions", "velocities", "observables", "radii_gyration_squared", "sample_steps", "times"):
        np.testing.assert_array_equal(left[key], right[key])
    for key in ("q", "p"):
        np.testing.assert_array_equal(getattr(left["state"], key), getattr(right["state"], key))
    assert left["state"].step == right["state"].step
    assert left["state"].time == right["state"].time
    assert left["elapsed"] == right["elapsed"]
    assert left["rng"] == right["rng"]
    if propagation:
        assert left["propagation"] == right["propagation"]


@pytest.mark.parametrize("backend", ["native", "auto"])
@pytest.mark.parametrize("block_steps", [1, 4, 256])
def test_production_blocks_stop_at_events_and_match_single_steps(
        library, audit, native_calls, tmp_path, monkeypatch, block_steps, backend):
    cfg = config(library, block_steps=block_steps, backend=backend)
    reference = single_step_reference(cfg, tmp_path / "reference", monkeypatch)
    assert native_calls["advance"] == [] and native_calls["run_block"] == [1] * (NEQ + NPROD)
    reference_calls = audit.audit_get_calls()
    native_calls["run_block"].clear()
    audit.audit_reset(-1)

    result = _sample_replica(task(cfg, tmp_path / "blocks"))

    blocks = expected_blocks(block_steps)
    sampling_blocks = expected_sampling_blocks(block_steps)
    assert native_calls["advance"] == sampling_blocks + blocks
    assert max(sampling_blocks + blocks) <= block_steps
    assert native_calls["run_block"] == []
    assert all(o == {"reuse_forces": True, "check_energy": True} for o in native_calls["options"])
    # Single steps evaluate the PES twice per step. A block of L steps
    # evaluates it L+1 times. Saved observations are unchanged.
    assert reference_calls - audit.audit_get_calls() == NEQ + NPROD - len(sampling_blocks + blocks)
    assert_same_stream(reference, result)
    assert result["propagation"]["production"] == {
        "native_steps": NPROD, "python_steps": 0, "fallback_reasons": {}}
    assert result["propagation"]["sampling"] == {
        "native_steps": NEQ, "python_steps": 0, "fallback_reasons": {}}
    assert result["sample_steps"].tolist() == list(range(NEQ, NEQ + NPROD + 1, INTERVAL))


@pytest.mark.parametrize("storage", ["memory", "stream"])
def test_interrupted_block_resumes_from_its_start_with_a_new_block_size(
        library, native_calls, tmp_path, monkeypatch, storage):
    cfg = config(library, block_steps=256, storage=storage)
    reference = single_step_reference(cfg, tmp_path / "reference", monkeypatch)
    blocks = expected_blocks(256)

    def interrupt(call):
        if call == 3:
            raise KeyboardInterrupt  # after the native block mutated q/p
    native_calls["hooks"]["after_advance"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        _sample_replica(task(cfg, tmp_path / "resumed"))
    saved = _read_stream(cfg, task(cfg, "")["system"], tmp_path / "resumed", 0)
    start = NEQ
    assert saved["elapsed"] == saved["state"].step == start
    assert saved["propagation"]["production"]["native_steps"] == start - NEQ
    assert saved["sample_steps"].tolist() == list(range(NEQ, start + 1, INTERVAL))

    native_calls["hooks"].clear()
    native_calls["advance"].clear()
    changed = copy.deepcopy(cfg)
    changed["dynamics"]["block_steps"] = 5
    resumed = _sample_replica(task(changed, tmp_path / "resumed"))
    assert native_calls["advance"] == expected_blocks(5, start)
    assert_same_stream(reference, resumed)


def test_native_failure_keeps_the_single_step_prefix_and_resumes(
        library, audit, native_calls, tmp_path, monkeypatch):
    cfg = config(library, block_steps=256)
    clean = single_step_reference(cfg, tmp_path / "clean", monkeypatch)
    # Fail the PES at the endpoint of production step 4, the last step of the
    # first block (NEQ=5 -> checkpoint at 9).
    # The first two calls are the Andersen equilibration blocks; call 3 is the
    # first production block. Its initial force plus endpoint calls 2..4
    # complete production steps 1..3 before the injected failure.
    native_calls["hooks"]["before_advance"] = lambda call: audit.audit_reset(5 if call == 3 else -1)
    with pytest.raises(PropagationError, match="3/4 steps.*injected CABI failure"):
        _sample_replica(task(cfg, tmp_path / "blocks"))
    blocked = _read_stream(cfg, task(cfg, "")["system"], tmp_path / "blocks", 0)

    # The same physical failure on the single-step path: step NEQ+4 fails at
    # its endpoint force (call 2 of that run_block call).
    native_calls["hooks"].clear()
    native_calls["run_block"].clear()
    native_calls["hooks"]["before_run_block"] = (
        lambda call: audit.audit_reset(2 if call == NEQ + 4 else -1))
    with pytest.raises(RuntimeError, match="injected CABI failure"):
        single_step_reference(cfg, tmp_path / "steps", monkeypatch)
    stepped = _read_stream(cfg, task(cfg, "")["system"], tmp_path / "steps", 0)

    assert blocked["elapsed"] == blocked["state"].step == NEQ + 3
    assert blocked["propagation"]["production"] == {
        "native_steps": 3, "python_steps": 0, "fallback_reasons": {}}
    assert_same_stream(stepped, blocked)

    native_calls["hooks"].clear()
    audit.audit_reset(-1)
    assert_same_stream(clean, _sample_replica(task(cfg, tmp_path / "blocks")))


def test_observation_failure_discards_only_the_final_block_step(
        library, native_calls, tmp_path, monkeypatch):
    cfg = config(library, block_steps=256)
    clean = single_step_reference(cfg, tmp_path / "clean", monkeypatch)
    failing = NEQ + 2 * INTERVAL
    original = materials.quantum_observables

    def broken(state, *args, **kwargs):
        if state.step == failing:
            raise RuntimeError("injected observable failure")
        return original(state, *args, **kwargs)

    saved = {}
    for name, run in (("blocks", lambda out: _sample_replica(task(cfg, out))),
                      ("steps", lambda out: single_step_reference(cfg, out, monkeypatch))):
        with monkeypatch.context() as patch:
            patch.setattr(materials, "quantum_observables", broken)
            with pytest.raises(RuntimeError, match="observable failure"):
                run(tmp_path / name)
        saved[name] = _read_stream(cfg, task(cfg, "")["system"], tmp_path / name, 0)
    assert saved["blocks"]["elapsed"] == saved["blocks"]["state"].step == failing - 1
    assert saved["blocks"]["propagation"]["production"]["native_steps"] == failing - 1 - NEQ
    assert_same_stream(saved["steps"], saved["blocks"])
    assert_same_stream(clean, _sample_replica(task(cfg, tmp_path / "blocks")))


def test_strict_native_rejects_an_unsupported_block_and_auto_records_fallback(
        library, native_calls, tmp_path, monkeypatch):
    from pyrpmd import _native

    monkeypatch.delattr(_native, "BLOCK_ENERGY_CHECK_API")
    strict = config(library, block_steps=256, backend="native")
    with pytest.raises(RuntimeError, match="native loop required.*energy checking"):
        _sample_replica(task(strict, tmp_path / "strict"))
    saved = _read_stream(strict, task(strict, "")["system"], tmp_path / "strict", 0)
    assert saved["elapsed"] == NEQ
    assert saved["propagation"]["production"] == {
        "native_steps": 0, "python_steps": 0, "fallback_reasons": {}}
    assert native_calls["advance"] == []

    result = _sample_replica(task(config(library, block_steps=256, backend="auto"), tmp_path / "auto"))
    production = result["propagation"]["production"]
    assert production["native_steps"] == 0 and production["python_steps"] == NPROD
    (reason, count), = production["fallback_reasons"].items()
    assert "energy checking" in reason and count == NPROD
    assert native_calls["advance"] == []


def test_strict_native_capability_failure_saves_thermalized_state(library, tmp_path, monkeypatch):
    from pyrpmd.propagation import NativeUnavailableError

    def unavailable(self, **kwargs):
        raise NativeUnavailableError("PES has no explicit stateless native binding")
    cfg = config(library, block_steps=256)
    with monkeypatch.context() as patch:
        patch.setattr(Dynamics, "can_fuse", unavailable)
        with pytest.raises(NativeUnavailableError):
            _sample_replica(task(cfg, tmp_path / "strict"))
    # NEQ is not a checkpoint multiple; the completed thermalization is kept.
    saved = _read_stream(cfg, task(cfg, "")["system"], tmp_path / "strict", 0)
    assert saved["elapsed"] == saved["state"].step == NEQ
    assert saved["propagation"]["production"]["python_steps"] == 0
