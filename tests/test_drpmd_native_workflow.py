"""Outcome fusion preserves physical decisions, output times and restart state."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.h3.drpmd import H3ExchangeClassifier
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.workflows import drpmd
from test_native_propagation import CompiledPotential, native_library


pytestmark = pytest.mark.skipif(
    not native_capabilities()["available"], reason="native build required"
)


def classifier(kind):
    system = {"symbols": ["H"] * 3, "masses": np.full(3, 1800.),
              "fragments": [[0], [1, 2]]}
    if kind == "h3":
        return H3ExchangeClassifier(system, bond_cutoff_bohr=2.5, exit_distance_bohr=5.)
    return drpmd.ConnectivityClassifier(
        ["O", "H", "H"], system["masses"], system["fragments"],
        {"H-H": 2.5, "H-O": 2.5}, {"OH+H": ["HO", "H"]}, 5.,
    )


def initial_state(product=True):
    q = np.array([[0., 0., 0.], [1., 0., 0.], [10., 0., 0.]])
    p = np.zeros((3, 3));p[2, 0] = 1800.
    if not product:
        q = q[[2, 0, 1]];p = p[[2, 0, 1]]
    return State(np.repeat(q[None], 8, axis=0), np.repeat(p[None], 8, axis=0))


def run(directory, library, outcome, *, check=5, product=True, resume=False,
        block_steps=31, loop_backend="native", captured=None):
    dynamics = Dynamics(CompiledPotential(library), [1800.] * 3, 220., 8, .2,
                        loop_backend=loop_backend, block_steps=block_steps)
    if captured is not None:
        advance = dynamics.propagate_block

        def observe_block(state, steps, **options):
            result = advance(state, steps, **options)
            captured.append((steps, options, result))
            return result

        dynamics.propagate_block = observe_block
    config = {"beads": 8, "dt_au": .2, "collision": {"max_steps": 40},
              "outcomes": {"check_interval": check, "persistence_steps": 7},
              "output": {"checkpoint_steps": 7, "trajectory_stride": 3}}
    return drpmd._trajectory(
        dynamics, initial_state(product), outcome, config, Path(directory),
        0, 0, {}, "native-outcome-workflow", resume,
    )


def assert_same_outputs(a, b, *, exact=True):
    with (np.load(Path(a) / "trajectories/trajectory-00000000.npz") as left,
          np.load(Path(b) / "trajectories/trajectory-00000000.npz") as right):
        for key in ("final_q", "final_p", "steps", "time_au", "centroid", "energy_per_bead"):
            if exact:
                np.testing.assert_array_equal(left[key], right[key])
            else:
                np.testing.assert_allclose(left[key], right[key], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["connectivity", "h3"])
@pytest.mark.parametrize("check", [1, 5])
@pytest.mark.parametrize("product", [False, True])
def test_native_checks_match_python_callbacks_with_larger_blocks(
    native_library, tmp_path, kind, check, product,
):
    outcome = classifier(kind)
    native_blocks, callback_blocks = [], []
    actual = run(tmp_path / "native", native_library, outcome, check=check,
                 product=product, captured=native_blocks)
    expected = run(tmp_path / "callback", native_library, lambda state: outcome(state),
                   check=check, product=product, captured=callback_blocks)
    for key in ("status", "channel", "reason", "steps", "time_au", "propagation",
                "initial_energy_per_bead", "final_energy_per_bead", "max_energy_drift_per_bead"):
        assert actual[key] == expected[key]
    assert actual["status"] == ("product" if product else "nonreactive")
    assert actual["steps"] == (8 if check == 1 else 15)
    assert actual["propagation"] == {"native_steps": actual["steps"], "python_steps": 0}
    assert any(steps > check for steps, _, _ in native_blocks)
    assert all(steps <= check for steps, _, _ in callback_blocks)
    assert all(not options["record_states"] and result.q is None
               for _, options, result in native_blocks + callback_blocks)
    assert all(result.outcome is not None for _, _, result in native_blocks)
    assert sum(result.outcome["checks"] for _, _, result in native_blocks) == actual["steps"] // check
    assert all(options["record_previous"] and result.previous_q is not None
               for _, options, result in callback_blocks)
    assert_same_outputs(tmp_path / "native", tmp_path / "callback")


@pytest.mark.parametrize("kind", ["connectivity", "h3"])
@pytest.mark.parametrize("interrupt_step", [7, 15])
def test_native_outcome_restart_keeps_candidate_and_exact_stop(
    native_library, tmp_path, kind, interrupt_step,
):
    outcome = classifier(kind)
    expected = run(tmp_path / "full", native_library, outcome)
    save = drpmd._atomic_npz

    def save_then_interrupt(path, metadata, **arrays):
        save(path, metadata, **arrays)
        if (metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1"
                and metadata["done"] == interrupt_step):
            raise KeyboardInterrupt("persisted native outcome state")

    with patch.object(drpmd, "_atomic_npz", side_effect=save_then_interrupt):
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path / "resumed", native_library, outcome)
    metadata, _ = drpmd._load_npz(
        tmp_path / "resumed/trajectories/trajectory-00000000.checkpoint.npz"
    )
    assert metadata["candidate_since"] == 5
    assert metadata["candidate"][0] == "product"
    actual = run(tmp_path / "resumed", native_library, outcome, resume=True, block_steps=3)
    for key in ("status", "channel", "steps", "time_au", "propagation",
                "final_energy_per_bead", "max_energy_drift_per_bead"):
        assert actual[key] == expected[key]
    assert_same_outputs(tmp_path / "full", tmp_path / "resumed")
    assert run(tmp_path / "resumed", native_library, outcome, resume=True) == actual


def test_checkpoint_is_written_once_per_distinct_terminal_state(native_library, tmp_path):
    save = drpmd._atomic_npz
    checkpoints = []

    def record(path, metadata, **arrays):
        if metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1":
            checkpoints.append((metadata["done"], metadata["status"]))
        save(path, metadata, **arrays)

    with patch.object(drpmd, "_atomic_npz", side_effect=record):
        run(tmp_path, native_library, classifier("h3"))
    assert checkpoints == [(7, "unresolved"), (14, "unresolved"), (15, "product")]


def test_native_channel_labels_preserve_full_python_strings(native_library, tmp_path):
    outcome = classifier("connectivity")
    label = chr(0) + "产物通道"
    signature = next(iter(outcome.channels))
    outcome.channels[signature] = label
    actual = run(tmp_path / "native", native_library, outcome)
    expected = run(tmp_path / "callback", native_library, lambda state: outcome(state))
    assert actual["channel"] == expected["channel"] == label
    assert actual["steps"] == expected["steps"] == 15
    assert_same_outputs(tmp_path / "native", tmp_path / "callback")


def test_python_classifier_state_edits_invalidate_force_reuse(native_library, tmp_path):
    def editing_classifier():
        calls = []

        def classify(state):
            calls.append(state.step)
            state.q[:, 0, 1] += .1
            return None

        return classify, calls

    native, native_calls = editing_classifier()
    python, python_calls = editing_classifier()
    a = run(tmp_path / "native", native_library, native, check=4)
    b = run(tmp_path / "python", native_library, python, check=4, loop_backend="python")
    assert native_calls == python_calls == list(range(4, 41, 4))
    assert a["steps"] == b["steps"] == 40
    assert a["status"] == b["status"] == "unresolved"
    assert_same_outputs(tmp_path / "native", tmp_path / "python", exact=False)


def test_native_failure_retains_candidate_only_from_accepted_checks(native_library, tmp_path):
    from pyrpmd.block_propagation import propagate_block

    pes = CompiledPotential(native_library)
    dynamics = Dynamics(pes, [1800.] * 3, 220., 8, .2)
    state = initial_state()
    # Initial force + three endpoints; endpoint 3 fails after two accepted checks.
    pes.lib.reset(4, 0)
    block = propagate_block(
        dynamics, state, 10, backend="native", reuse_forces=True, raise_on_error=False,
        outcome={"descriptor": classifier("h3").native_outcome_descriptor(),
                 "start_step": 0, "check_interval": 1, "persistence_steps": 7,
                 "candidate": None, "candidate_since": None},
    )
    assert block.completed == state.step == 2
    assert block.outcome == {"candidate": ("product", "H2[0,1]+H"),
                             "candidate_since": 1, "resolved": None, "checks": 2,
                             "pending_check": False}
    assert block.error is not None and pes.lib.get_calls() == 4
