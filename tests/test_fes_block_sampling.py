"""FES block boundaries preserve observations, RNG and the accepted prefix."""

import copy
import ctypes
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd.bias import UmbrellaBias
from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import CABIPotential, HarmonicPotential
from pyrpmd.propagation import PropagationError
from pyrpmd.system import System
from pyrpmd.workflows import fes


@pytest.fixture(scope="module")
def cabi_library(tmp_path_factory):
    if not native_capabilities()["available"] or shutil.which("cc") is None:
        pytest.skip("native extension and C compiler required")
    from pyrpmd import _native

    if getattr(_native, "BLOCK_ENERGY_CHECK_API", 0) < 1:
        pytest.skip("rebuild native extension with per-step block energy checking")
    root = Path(__file__).resolve().parents[1]
    source = (root / "native/examples/harmonic_pes.c").read_text()
    controls = """
static int audit_calls=0, audit_fail_at=-1, audit_overflow_at=-1;
void audit_reset(int fail, int overflow) {
    audit_calls=0; audit_fail_at=fail; audit_overflow_at=overflow;
}
int audit_get_calls(void) { return audit_calls; }
"""
    source = source.replace("#include <stdio.h>", "#include <stdio.h>\n" + controls)
    marker = "pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1"
    opening = source.index("{", source.index(marker)) + 1
    source = source[:opening] + """
 ++audit_calls;
 if(audit_calls==audit_fail_at) {
   message(e,"injected CABI failure"); return PYRPMD_INTERNAL_ERROR_V1;
 }
""" + source[opening:]
    ending = ' message(e,"");return 0;\n}\nvoid pyrpmd_pes_destroy_v1'
    assert ending in source
    source = source.replace(ending, """
 if(audit_calls==audit_overflow_at)
   for(int64_t i=0;i<b;i++) y->energies[i]=1e308;
""" + ending)
    directory = tmp_path_factory.mktemp("fes-block-pes")
    cfile, library = directory / "harmonic.c", directory / "harmonic.so"
    cfile.write_text(source)
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-std=c11", "-I",
                    str(root / "native/include"), str(cfile), "-o", str(library)],
                   check=True, capture_output=True)
    return library


@pytest.fixture
def cabi_controls(cabi_library):
    controls = ctypes.CDLL(str(cabi_library))
    controls.audit_reset.argtypes = [ctypes.c_int, ctypes.c_int]
    controls.audit_reset.restype = None
    controls.audit_get_calls.argtypes = []
    controls.audit_get_calls.restype = ctypes.c_int
    controls.audit_reset(-1, -1)
    yield controls
    controls.audit_reset(-1, -1)


def _small_state(beads=4):
    return State(np.full((beads, 1, 3), 0.1), np.full((beads, 1, 3), 0.2))


@pytest.mark.parametrize("check_energy", [False, True])
@pytest.mark.parametrize("record_energy", [False, True])
def test_python_energy_check_schedule_and_observation_reuse(check_energy, record_energy):
    dynamics = Dynamics(HarmonicPotential(), [2.3], 1.7, 4, 0.01, native=False)
    original = dynamics.energy
    calls = []

    def energy(state, bias=None):
        calls.append(state.step)
        return original(state, bias)

    dynamics.energy = energy
    result = dynamics.propagate_block(_small_state(), 4, observe=[0, 1, 0, 1],
                                      backend="python", record_energy=record_energy,
                                      check_energy=check_energy)
    assert result.completed == 4
    assert calls == ([0, 1, 2, 3, 4] if check_energy else ([2, 4] if record_energy else []))
    assert np.isfinite(result.energies).all() if record_energy else np.isnan(result.energies).all()


@pytest.mark.parametrize("failure_step", [0, 3])
def test_python_energy_failure_keeps_only_the_valid_prefix(failure_step):
    dynamics = Dynamics(HarmonicPotential(), [2.3], 1.7, 4, 0.01, native=False)
    state, reference = _small_state(), _small_state()
    for _ in range(max(0, failure_step - 1)):
        dynamics.step(reference)
    original = dynamics.energy
    calls = []

    def energy(candidate, bias=None):
        calls.append(candidate.step)
        return np.inf if candidate.step == failure_step else original(candidate, bias)

    dynamics.energy = energy
    result = dynamics.propagate_block(state, 5, observe=[1, 0, 0, 1, 1],
                                      backend="python", check_energy=True, raise_on_error=False)
    assert result.completed == state.step == max(0, failure_step - 1)
    assert result.error and "nonfinite total ring energy" in result.error
    np.testing.assert_array_equal(result.steps, [] if failure_step == 0 else [1])
    np.testing.assert_array_equal(state.q, reference.q)
    np.testing.assert_array_equal(state.p, reference.p)
    assert calls == list(range(failure_step + 1))


@pytest.mark.parametrize("invalid", [0, 1, None, "yes", []])
def test_check_energy_requires_a_boolean_before_mutating_state(invalid):
    dynamics = Dynamics(HarmonicPotential(), [2.3], 1.7, 4, 0.01, native=False)
    state = _small_state()
    with pytest.raises(ValueError, match="check_energy must be boolean"):
        dynamics.propagate_block(state, 1, check_energy=invalid)
    assert state.step == 0


def test_missing_native_energy_marker_falls_back_or_rejects_before_propagation(
    cabi_library, cabi_controls, monkeypatch,
):
    from pyrpmd import _native

    potential = CABIPotential(cabi_library)
    try:
        dynamics = Dynamics(potential, [2.3], 1.7, 4, 0.01)
        monkeypatch.delattr(_native, "BLOCK_ENERGY_CHECK_API")
        state, reference = _small_state(), _small_state()
        result = dynamics.propagate_block(state, 4, observe=[0, 1, 0, 1], check_energy=True)
        assert result.backend == "python" and "energy checking" in result.fallback_reason
        dynamics.propagate_block(reference, 4, observe=[0, 1, 0, 1],
                                 backend="python", check_energy=True)
        np.testing.assert_array_equal(state.q, reference.q)
        np.testing.assert_array_equal(state.p, reference.p)
        before = _small_state()
        calls = cabi_controls.audit_get_calls()
        with pytest.raises(RuntimeError, match="native loop required.*energy checking"):
            dynamics.propagate_block(before, 4, backend="native", check_energy=True)
        assert before.step == 0 and cabi_controls.audit_get_calls() == calls
        # The additive option does not disable the older/default block API.
        assert dynamics.propagate_block(before, 2, backend="native").backend == "native"
    finally:
        potential.close()


def test_native_checks_unobserved_total_energy_without_replaying(cabi_library, cabi_controls):
    potential = CABIPotential(cabi_library)
    try:
        dynamics = Dynamics(potential, [2.3], 1.7, 4, 0.01)
        state, reference = _small_state(), _small_state()
        dynamics.propagate_block(reference, 2, backend="native", record_energy=False)
        # All four per-bead energies remain finite. Only their total overflows
        # at the endpoint of step 3, which has no requested observation.
        cabi_controls.audit_reset(-1, 6)
        result = dynamics.propagate_block(state, 5, observe=[1, 0, 0, 1, 0],
                                          backend="native", check_energy=True,
                                          record_energy=False, raise_on_error=False)
        assert result.completed == state.step == 2
        assert "nonfinite total ring energy" in result.error
        assert cabi_controls.audit_get_calls() == 6
        np.testing.assert_array_equal(result.steps, [1])
        assert np.isnan(result.energies).all()
        np.testing.assert_array_equal(state.q, reference.q)
        np.testing.assert_array_equal(state.p, reference.p)
        cabi_controls.audit_reset(-1, 1)
        initial = _small_state()
        failed = dynamics.propagate_block(initial, 2, observe=[0, 0], backend="native",
                                          check_energy=True, raise_on_error=False)
        assert failed.completed == initial.step == 0 and failed.error
        assert cabi_controls.audit_get_calls() == 1
    finally:
        potential.close()


def _task(output, library=None, *, block_steps=64, backend="native", scheme="cayley-bcb"):
    system = System(("H",), np.array([2.3]), np.array([[0.2, 0.0, 0.0]]))
    specs = [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
             for axis in ("x", "y")]
    return {
        "config": {
            "task": "fes", "beta": 1.7, "beads": 4, "dt_au": 0.04, "seed": 9131,
            "native": library is not None,
            "potential": ({"provider": "cabi", "library": str(library)} if library is not None
                          else {"provider": "harmonic", "k": 1.0}),
            "sampling": {"equilibration_steps": 7, "production_steps": 62,
                         "sample_interval": 4, "andersen_interval": 13},
            "dynamics": {"loop_backend": backend, "block_steps": block_steps, "integrator": scheme},
            "output": {"checkpoint_steps": 17},
        },
        "system": system, "specs": specs, "center": [0.2, -0.1],
        "stiffness": [[0.4, 0.05], [0.05, 0.7]], "window": 2, "replica": 1,
        "output": str(output),
    }


def _checkpoint(task):
    return Path(task["output"]) / "checkpoints/window_00002_replica_0001.npz"


def _saved(task):
    with load_checkpoint(_checkpoint(task)) as saved:
        return {name: saved[name].copy() for name in saved.files if not name.startswith("__")}


def _reference_window(task, stop=None):
    """Direct step-by-step sampling with the same initial ensemble and RNG."""
    config, system = task["config"], task["system"]
    potential = (CABIPotential(config["potential"]["library"])
                 if config["potential"]["provider"] == "cabi" else HarmonicPotential())
    try:
        dynamics = Dynamics(potential, system.masses, config["beta"], config["beads"],
                            config["dt_au"], seed=np.random.SeedSequence(
                                [config["seed"], 101, task["window"], task["replica"]]),
                            native=config["native"], integrator=config["dynamics"]["integrator"])
        cv = CVStack([Cartesian(0, "x"), Cartesian(0, "y")])
        bias = UmbrellaBias(cv, task["center"], task["stiffness"])
        q = np.repeat(system.geometry[None], config["beads"], axis=0)
        q[:, 0, :2] = task["center"]
        state = State(q, dynamics.sample_momenta())
        sampling = config["sampling"]
        neq = sampling["equilibration_steps"]
        total = neq + sampling["production_steps"] if stop is None else stop
        values, energies = [], []
        for elapsed in range(total):
            if state.step % sampling["andersen_interval"] == 0:
                state.p[:] = dynamics.sample_momenta()
            dynamics.step(state, bias)
            production = elapsed + 1 - neq
            if production > 0 and production % sampling["sample_interval"] == 0:
                values.append(cv.value(state.q))
                energies.append(dynamics.energy(state, bias))
        return {"q": state.q.copy(), "p": state.p.copy(), "step": state.step, "time": state.time,
                "values": np.asarray(values).reshape(-1, 2), "energies": np.asarray(energies),
                "rng": copy.deepcopy(dynamics.rng.bit_generator.state)}
    finally:
        if hasattr(potential, "close"):
            potential.close()


@pytest.mark.parametrize("block_steps", [1, 9, 65536])
@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab", "velocity-verlet"])
def test_native_fes_returns_all_observations_across_unaligned_boundaries(
    tmp_path, cabi_library, cabi_controls, monkeypatch, block_steps, scheme,
):
    task = _task(tmp_path, cabi_library, block_steps=block_steps, scheme=scheme)
    expected = _reference_window(task)
    original = Dynamics.propagate_block
    calls = []

    def recorded(dynamics, state, count, **options):
        start = state.step
        assert options["check_energy"] is True and options["raise_on_error"] is False
        result = original(dynamics, state, count, **options)
        calls.append((start, count, np.array(options["observe"]), result))
        return result

    monkeypatch.setattr(Dynamics, "propagate_block", recorded)
    result = fes._sample_window(task)
    assert calls
    for start, count, mask, block in calls:
        assert count == min(block_steps, 4096, 69 - start, 13 - start % 13, 17 - start % 17)
        production = start + np.arange(1, count + 1) - 7
        np.testing.assert_array_equal(mask, (production > 0) & (production % 4 == 0))
        assert block.backend == "native" and block.completed == count
        assert block.values.shape == (int(mask.sum()), 2)
        assert block.energies.shape == (int(mask.sum()),)
    np.testing.assert_array_equal(np.concatenate([call[3].steps for call in calls]), np.arange(11, 70, 4))
    if block_steps > 4:
        assert any(len(call[3].values) > 1 for call in calls)
    np.testing.assert_allclose(result["samples"], expected["values"], atol=2e-13, rtol=1e-13)
    np.testing.assert_allclose(result["energies"], expected["energies"], atol=3e-12, rtol=1e-13)
    np.testing.assert_allclose(result["last_q"], expected["q"], atol=2e-13, rtol=1e-13)
    np.testing.assert_allclose(result["last_p"], expected["p"], atol=2e-12, rtol=1e-13)
    saved = _saved(task)
    assert int(saved["elapsed"]) == int(saved["step"]) == 69
    assert float(saved["time"]) == expected["time"]
    assert json.loads(str(saved["rng"])) == expected["rng"]
    assert result["propagation"]["native_steps"] == 69


def test_fes_caps_large_blocks_and_avoids_duplicate_terminal_checkpoint(
    tmp_path, cabi_library, cabi_controls, monkeypatch,
):
    task = _task(tmp_path, cabi_library, block_steps=65536)
    task["config"]["beads"] = 1
    task["config"]["sampling"].update(production_steps=4100, sample_interval=1000,
                                      andersen_interval=10000)
    task["config"]["output"]["checkpoint_steps"] = 4107
    calls, saves = [], []
    propagate, save = Dynamics.propagate_block, fes._atomic_npz

    def recorded(dynamics, state, count, **options):
        calls.append(count)
        return propagate(dynamics, state, count, **options)

    def saved(path, **arrays):
        saves.append(arrays["elapsed"])
        return save(path, **arrays)

    monkeypatch.setattr(Dynamics, "propagate_block", recorded)
    monkeypatch.setattr(fes, "_atomic_npz", saved)
    result = fes._sample_window(task)
    assert calls == [4096, 11]
    assert result["samples"].shape == (4, 2)
    assert saves == [4107]


def test_native_completed_resume_and_different_block_partition_preserve_samples(
    tmp_path, cabi_library, cabi_controls, monkeypatch,
):
    uninterrupted = _task(tmp_path / "complete", cabi_library, block_steps=1)
    expected = fes._sample_window(uninterrupted)
    task = _task(tmp_path / "interrupted", cabi_library, block_steps=64)
    original = fes._atomic_npz

    class Interrupted(KeyboardInterrupt):
        pass

    def interrupt_after_checkpoint(path, **arrays):
        original(path, **arrays)
        if arrays["elapsed"] == 17:
            raise Interrupted()

    with monkeypatch.context() as patch:
        patch.setattr(fes, "_atomic_npz", interrupt_after_checkpoint)
        with pytest.raises(Interrupted):
            fes._sample_window(task)
    assert int(_saved(task)["elapsed"]) == 17
    task["config"]["dynamics"]["block_steps"] = 9
    actual = fes._sample_window(task)
    for name in ("samples", "energies", "last_q", "last_p"):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert str(_saved(task)["rng"]) == str(_saved(uninterrupted)["rng"])
    before = _checkpoint(task).read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("completed resume propagated or rewrote the authoritative checkpoint")

    monkeypatch.setattr(fes, "_atomic_npz", forbidden)
    monkeypatch.setattr(Dynamics, "propagate_block", forbidden)
    resumed = fes._sample_window(task)
    np.testing.assert_array_equal(resumed["samples"], expected["samples"])
    assert _checkpoint(task).read_bytes() == before


@pytest.mark.parametrize("failed_step", [1, 14, 23])
def test_native_failure_checkpoints_valid_prefix_without_replay_and_resumes_same_rng(
    tmp_path, cabi_library, cabi_controls, failed_step,
):
    task = _task(tmp_path / "failed", cabi_library)
    cabi_controls.audit_reset(2 * failed_step, -1)
    with pytest.raises(PropagationError, match="injected CABI failure") as caught:
        fes._sample_window(task)
    assert cabi_controls.audit_get_calls() == 2 * failed_step
    saved = _saved(task)
    prefix = failed_step - 1
    assert int(saved["step"]) == int(saved["elapsed"]) == prefix
    assert saved["values"].shape == (max(0, (prefix - 7) // 4), 2)
    assert all(step <= prefix for step in caught.value.result.steps)
    assert json.loads(str(saved["propagation"]))["native_steps"] == prefix
    cabi_controls.audit_reset(-1, -1)
    reference = _reference_window(task, stop=prefix)
    for name in ("q", "p", "values", "energies"):
        np.testing.assert_allclose(saved[name], reference[name], atol=3e-12, rtol=1e-13)
    assert json.loads(str(saved["rng"])) == reference["rng"]
    resumed = fes._sample_window(task)
    complete_task = _task(tmp_path / "complete", cabi_library, block_steps=1)
    complete = fes._sample_window(complete_task)
    for name in ("samples", "energies", "last_q", "last_p"):
        np.testing.assert_array_equal(resumed[name], complete[name])
    assert str(_saved(task)["rng"]) == str(_saved(complete_task)["rng"])


def test_python_fes_retains_stateful_provider_evaluation_schedule(tmp_path, monkeypatch):
    class CountedPotential(HarmonicPotential):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def evaluate(self, q):
            self.calls += 1
            return super().evaluate(q)

    task = _task(tmp_path, backend="python")
    provider = CountedPotential()
    # The workflow now borrows from a worker-scoped lifetime manager, which
    # still calls this factory for an uncached provider outside run_tasks.
    monkeypatch.setattr("pyrpmd.parallel.execution.make_task_potential", lambda spec, system: provider)
    result = fes._sample_window(task)
    assert provider.calls == 2 * 69 + 15
    assert result["propagation"]["python_steps"] == 69
    reference = _reference_window(task)
    np.testing.assert_array_equal(result["samples"], reference["values"])
    np.testing.assert_array_equal(result["energies"], reference["energies"])


def test_python_failed_observation_is_rolled_back_before_saving_samples(tmp_path, monkeypatch):
    task = _task(tmp_path / "failed", backend="python")
    # Start at the umbrella center so the counted CV.value calls are saved
    # observations, not the geometric initialization's line-search checks.
    task["system"].geometry[0, 1] = task["center"][1]
    original = fes.make_cvs
    calls = []

    def failing_cvs(*args, **kwargs):
        cv = original(*args, **kwargs)
        value = cv.value

        def observe(q):
            calls.append(True)
            if len(calls) == 3:
                raise RuntimeError("injected observation failure")
            return value(q)

        cv.value = observe
        return cv

    with monkeypatch.context() as patch:
        patch.setattr(fes, "make_cvs", failing_cvs)
        with pytest.raises(RuntimeError, match="injected observation failure"):
            fes._sample_window(task)
    saved = _saved(task)
    assert len(calls) == 3 and int(saved["elapsed"]) == int(saved["step"]) == 18
    assert saved["values"].shape == (2, 2)
    reference = _reference_window(task, stop=18)
    for name in ("q", "p", "values", "energies"):
        np.testing.assert_array_equal(saved[name], reference[name])
    assert json.loads(str(saved["rng"])) == reference["rng"]
    resumed = fes._sample_window(task)
    complete = fes._sample_window(_task(tmp_path / "complete", backend="python"))
    for name in ("samples", "energies", "last_q", "last_p"):
        np.testing.assert_array_equal(resumed[name], complete[name])


@pytest.mark.parametrize("scheme", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
def test_reus_cabi_native_matches_numpy(tmp_path, cabi_library, cabi_controls, scheme, thermostat):
    cfg = {
        "task": "fes", "beta": 1.7, "beads": 4, "dt_au": .03, "seed": 67,
        "native": False, "dynamics": {"integrator": scheme, "loop_backend": "python"},
        "potential": {"provider": "cabi", "library": str(cabi_library)},
        "sampling": {"thermostat": thermostat, "equilibration_steps": 8,
                     "production_steps": 40, "sample_interval": 2, "replicas": 1,
                     "andersen_interval": 7},
        "cvs": [{"kind": "cartesian", "atom": 0, "axis": a} for a in ("x", "y")],
        "umbrella": {"centers": [[-.2, -.2], [-.2, .2], [.2, -.2], [.2, .2]],
                     "stiffness": [[1., .2], [.2, .8]]},
        "replica_exchange": {"enabled": True, "interval_steps": 9},
        "analysis": {"bins": [4, 5], "bootstrap": 0}, "output": {"checkpoint_steps": 11},
    }
    if thermostat == "pile-l":
        cfg["sampling"]["gamma_centroid"] = .5
    system = System(("H",), np.array([2.3]), np.zeros((1, 3)))
    fes.run_fes(cfg, system, tmp_path / "numpy")
    cfg["native"] = True
    cfg["dynamics"]["loop_backend"] = "native" if thermostat == "andersen" else "auto"
    fes.run_fes(cfg, system, tmp_path / "native")
    with load_checkpoint(tmp_path / "numpy/checkpoints/exchange_0000.npz") as a, \
         load_checkpoint(tmp_path / "native/checkpoints/exchange_0000.npz") as b:
        for key in ("q", "p", "samples", "energies"):
            np.testing.assert_allclose(a[key], b[key], rtol=3e-12, atol=3e-13, err_msg=key)
        for key in ("decisions", "walker_history", "walker_ids", "attempts", "accepts", "rngs", "exchange_rng"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        if thermostat == "andersen":
            assert all(s["native_steps"] == 48 for s in json.loads(str(b["propagation"])))
