"""Outcome optimizations retain strict cutoffs, mutable settings and Python hooks."""
from types import MethodType

import numpy as np
import pytest

from benchmarks.h3.drpmd import H3ExchangeClassifier
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.workflows.drpmd import (
    ConnectivityClassifier, PilotConnectivityClassifier, formula,
    fragment_com, fragment_velocity,
)
from test_native_propagation import CompiledPotential, native_library


def _explicit(cls=ConnectivityClassifier):
    return cls(
        ["H", "H", "O"], [1.0, 2.0, 16.0], [[0], [1, 2]],
        {"H-H": 1.6, "H-O": 1.7}, {"H2+O": ["H2", "O"]}, 4.0,
    )


def _product_state():
    return State(
        [[[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [10.0, 0.0, 0.0]]],
        [[[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [16.0, 0.0, 0.0]]],
    )


def _reference_bonds(classifier, q):
    """Scalar pre-optimization distance arithmetic, especially at exact cutoffs."""
    center = np.asarray(q).mean(axis=0)
    return {
        (i, j) for i, symbol in enumerate(classifier.symbols)
        for j in range(i + 1, len(classifier.symbols))
        if np.linalg.norm(center[i] - center[j])
        < classifier.cutoffs[tuple(sorted((symbol, classifier.symbols[j])))]
    }


def _reference_label(classifier, state):
    # Use independent component construction and the old full-array reductions.
    groups = [{i} for i in range(len(classifier.symbols))]
    for i, j in _reference_bonds(classifier, state.q):
        left = next(group for group in groups if i in group)
        right = next(group for group in groups if j in group)
        if left is not right:
            left.update(right)
            groups.remove(right)
    components = sorted([sorted(group) for group in groups])
    if len(components) < 2:
        return None
    centers = [fragment_com(state.q, classifier.masses, f) for f in components]
    velocities = [fragment_velocity(state.p, classifier.masses, f) for f in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            r, v = centers[j] - centers[i], velocities[j] - velocities[i]
            if np.linalg.norm(r) < classifier.exit_distance or np.dot(r, v) <= 0:
                return None
    signature = tuple(sorted(formula(classifier.symbols[i] for i in f) for f in components))
    if signature == classifier.reactants:
        return {"status": "nonreactive", "channel": None}
    channel = classifier.channels.get(signature)
    return None if channel is None else {"status": "product", "channel": channel}


def test_explicit_classifier_matches_scalar_reference_on_multibead_states():
    classifier = _explicit()
    rng = np.random.default_rng(412)
    states = [_product_state()]
    for _ in range(120):
        center = rng.normal(size=(3, 3)) * 5
        states.append(State(center[None] + rng.normal(size=(7, 3, 3)),
                            rng.normal(size=(7, 3, 3))))
    for state in states:
        assert classifier.bond_graph(state.q) == _reference_bonds(classifier, state.q)
        assert classifier(state) == _reference_label(classifier, state)
    assert classifier(states[0]) == {"status": "product", "channel": "H2+O"}


def test_bond_cutoff_keeps_original_strict_floating_point_comparison():
    classifier = _explicit()
    rng = np.random.default_rng(992)
    for _ in range(30):
        delta = rng.normal(size=3)
        q = np.array([[np.zeros(3), delta, [50.0, 50.0, 50.0]]])
        distance = np.linalg.norm(delta)
        for cutoff in (np.nextafter(distance, 0.0), distance,
                       np.nextafter(distance, np.inf)):
            classifier.cutoffs[("H", "H")] = cutoff
            assert classifier.bond_graph(q) == _reference_bonds(classifier, q)
            assert ((0, 1) in classifier.bond_graph(q)) == (distance < cutoff)


def test_exit_distance_and_outward_velocity_boundaries():
    classifier = ConnectivityClassifier(["H", "O"], [1.0, 16.0], [[0], [1]],
                                        {"H-O": 1.0}, {}, 4.0)
    state = State([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
                  [[[0.0, 0.0, 0.0], [16.0, 0.0, 0.0]]])
    assert classifier(state) == {"status": "nonreactive", "channel": None}
    state.q[0, 1, 0] = np.nextafter(4.0, 0.0)
    assert classifier(state) is None
    state.q[0, 1, 0] = 4.0
    state.p[:] = 0
    assert classifier(state) is None
    state.p[0, 1, 0] = -1
    assert classifier(state) is None


def test_descriptor_and_pair_cache_follow_public_parameter_mutations():
    classifier = _explicit()
    original = classifier.native_outcome_descriptor()
    assert original["kind"] == "connectivity"
    assert classifier(_product_state()) is not None
    classifier.cutoffs[("H", "H")] = 0.2
    classifier.masses[0] = 3.0
    classifier.exit_distance = 6.0
    classifier.channels[("H2", "O")] = "renamed"
    changed = classifier.native_outcome_descriptor()
    assert changed["pair_cutoffs"][0] == 0.2
    assert changed["masses"][0] == 3.0
    assert changed["exit_distance"] == 6.0
    assert changed["channel_labels"] == ["renamed"]
    assert original["masses"][0] == 1.0
    assert classifier(_product_state()) is None


@pytest.mark.parametrize("change", ["exit", "reactants", "same_channel", "single_species", "duplicate_labels"])
def test_descriptor_declines_runtime_values_rejected_by_native_parser(change):
    classifier = _explicit()
    if change == "exit":
        classifier.exit_distance = 1.0
    elif change == "reactants":
        classifier.reactants = ("H2O",)
    elif change == "same_channel":
        classifier.channels[classifier.reactants] = "reactant_channel"
    elif change == "single_species":
        classifier.channels[("H2O",)] = "single"
    else:
        classifier.channels[("H", "H", "O")] = "H2+O"
    assert classifier.native_outcome_descriptor() is None


@pytest.mark.parametrize("hook", ["bond_graph", "components", "classify_components", "__call__"])
def test_native_descriptor_declines_instance_overrides(hook):
    classifier = _explicit()
    original = getattr(classifier, hook)
    calls = []

    def override(self, *args):
        calls.append(hook)
        return original(*args)

    setattr(classifier, hook, MethodType(override, classifier))
    assert classifier.native_outcome_descriptor() is None
    if hook == "__call__":
        classifier.__call__(_product_state())
    else:
        classifier(_product_state())
    assert calls


def test_subclass_and_class_monkeypatch_keep_extension_hooks(monkeypatch):
    class CustomClassifier(ConnectivityClassifier):
        def classify_components(self, state, components):
            return {"status": "product", "channel": "custom"}

    custom = _explicit(CustomClassifier)
    assert custom.native_outcome_descriptor() is None
    assert custom(_product_state())["channel"] == "custom"
    classifier = _explicit()
    monkeypatch.setattr(ConnectivityClassifier, "classify_components",
                        lambda self, state, components: {"status": "product", "channel": "patched"})
    assert classifier.native_outcome_descriptor() is None
    assert classifier(_product_state())["channel"] == "patched"


def test_pilot_reuses_graph_without_native_opt_in_or_exchange_regression():
    classifier = PilotConnectivityClassifier(
        ["H", "H", "H"], [1.0] * 3, [[0], [1, 2]], {"H-H": 1.6}, {}, 4.0,
        [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
    )
    assert classifier.native_outcome_descriptor() is None
    state = _product_state()
    assert classifier(state) is None  # Same formula, exchanged atom membership.
    state.q[:] = [[[-10.0, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]]
    state.p[:] = [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    assert classifier(state) == {"status": "nonreactive", "channel": None}


def _h3():
    return H3ExchangeClassifier({"symbols": ["H"] * 3, "masses": [1.0, 2.0, 3.0],
                                 "fragments": [[0], [1, 2]]}, exit_distance_bohr=4.0)


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
def test_h3_atom_identity_and_descriptor(pair):
    classifier = _h3()
    atom = 3 - sum(pair)
    q, p = np.zeros((1, 3, 3)), np.zeros((1, 3, 3))
    q[0, pair[0], 0], q[0, pair[1], 0] = -0.4, 0.4
    q[0, atom, 2], p[0, atom, 2] = 10.0, classifier.masses[atom]
    state = State(q, p)
    result = classifier(state)
    assert result == ({"status": "nonreactive", "channel": None} if pair == (1, 2)
                      else {"status": "product", "channel": f"H2[{pair[0]},{pair[1]}]+H"})
    assert classifier.diagnostics(state)["pair"] == list(pair)
    descriptor = classifier.native_outcome_descriptor()
    assert descriptor["channel_labels"] == ["H2[0,1]+H", "H2[0,2]+H"]
    classifier.bond_cutoff = 2.0
    classifier.exit_distance = 6.0
    classifier.masses[0] = 4.0
    changed = classifier.native_outcome_descriptor()
    assert (changed["bond_cutoff"], changed["exit_distance"], changed["masses"][0]) == (2.0, 6.0, 4.0)
    state.p *= -1
    assert classifier(state) is None


@pytest.mark.parametrize("hook", ["bonded_pair", "diagnostics", "__call__"])
def test_h3_descriptor_declines_overrides_and_honors_python_dispatch(hook):
    classifier = _h3()
    original = getattr(classifier, hook)
    calls = []

    def override(self, *args):
        calls.append(hook)
        return original(*args)

    setattr(classifier, hook, MethodType(override, classifier))
    assert classifier.native_outcome_descriptor() is None
    classifier.__call__(_product_state())
    assert calls


@pytest.mark.skipif(not native_capabilities()["available"], reason="native build required")
@pytest.mark.parametrize("boundary", ["bond", "exit", "outward"])
def test_nonaxis_native_boundary_preserves_python_workflow_decision(native_library, tmp_path, boundary):
    """BLAS dot rounding at a strict boundary must not change a chemical outcome."""
    from pyrpmd.workflows.drpmd import _trajectory

    delta = np.array([0.06960505283532678, -0.4097014693788604, -0.059652509543542676])
    q = np.zeros((1, 3, 3))
    p = np.zeros_like(q)
    if boundary == "bond":
        q[0, 1], q[0, 2] = delta, [10., 10., 10.]
        p[0, 2] = 1.
        cutoff, exit_distance = float(np.linalg.norm(delta)), 5.
    else:
        q[0, 0, 0], q[0, 1, 0], q[0, 2] = -.01, .01, delta
        p[0, 2] = delta if boundary == "exit" else [0., delta[2], -delta[1]]
        cutoff = .1
        exit_distance = float(np.linalg.norm(delta)) if boundary == "exit" else .2
    classifier = H3ExchangeClassifier(
        {"symbols": ["H"] * 3, "masses": [1.] * 3, "fragments": [[0], [1, 2]]},
        bond_cutoff_bohr=cutoff, exit_distance_bohr=exit_distance,
    )
    pes = CompiledPotential(native_library)
    dyn = Dynamics(pes, [1.] * 3, 100., 1, 0., loop_backend="native")
    current = State(q, p)
    pending = dyn.propagate_block(current.copy(), 6, outcome={
        "descriptor": classifier.native_outcome_descriptor(),
        "check_interval": 1, "persistence_steps": 2,
    })
    assert pending.completed == 1 and pending.outcome["pending_check"]
    assert pending.outcome["candidate"] is None
    assert pending.outcome["checks"] == 0

    config = {
        "beads": 1, "dt_au": 0., "collision": {"max_steps": 6},
        "outcomes": {"check_interval": 1, "persistence_steps": 2},
        "output": {"trajectory_stride": 1, "checkpoint_steps": 4},
    }
    native = _trajectory(dyn, current.copy(), classifier, config,
                         tmp_path / "native", 0, 0, {}, "boundary", False)
    # The callback intentionally lacks a descriptor, retaining the original
    # Python classification schedule with the same native dynamics.
    reference = _trajectory(dyn, current.copy(), lambda state: classifier(state), config,
                            tmp_path / "reference", 0, 0, {}, "boundary", False)
    for key in ("status", "channel", "reason", "steps", "time_au",
                "initial_energy_per_bead", "final_energy_per_bead", "max_energy_drift_per_bead"):
        assert native[key] == reference[key]
    with (
        np.load(tmp_path / "native/trajectories/trajectory-00000000.npz") as actual,
        np.load(tmp_path / "reference/trajectories/trajectory-00000000.npz") as expected,
    ):
        for key in ("final_q", "final_p", "steps", "centroid", "energy_per_bead"):
            np.testing.assert_array_equal(actual[key], expected[key])
