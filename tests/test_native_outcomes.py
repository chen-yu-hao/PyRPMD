"""Native outcome decisions, persistence, force reuse and compact rollback.

The compiled fixture is a real native function with a visible call counter;
no Python callback is invoked while the native loop has released the GIL.
"""
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.block_propagation import native_outcome_supported
from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import Distance
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.python import import_object
from pyrpmd.workflows.drpmd import ConnectivityClassifier
from test_native_propagation import CompiledPotential, native_library, state, coordinate


pytestmark = pytest.mark.skipif(not native_capabilities()["available"], reason="native build required")


def _connectivity():
    return ConnectivityClassifier(
        ["H", "O", "O"], [1., 16., 16.], [[0], [1, 2]],
        {"H-O": 1.5, "O-O": 1.5},
        {"hydroxyl": ["HO", "O"], "dissociation": ["H", "O", "O"]}, 10.,
    )


def _h3():
    root = Path(__file__).resolve().parents[1]
    cls = import_object(str(root / "benchmarks/h3/drpmd.py") + ":H3ExchangeClassifier")
    return cls({"symbols": ["H"] * 3, "masses": [1., 1., 1.], "fragments": [[0], [1, 2]]},
               bond_cutoff_bohr=1.5, exit_distance_bohr=10.)


def _separated(pair=(1, 2), masses=(1., 1., 1.), *, bond=1., separation=11., outward=1., beads=1):
    a, b = pair
    atom = 3 - a - b
    q = np.zeros((beads, 3, 3))
    total = masses[a] + masses[b]
    q[:, a, 0] = -bond * masses[b] / total
    q[:, b, 0] = bond * masses[a] / total
    q[:, atom, 2] = separation
    p = np.zeros_like(q)
    p[:, atom, 2] = outward * masses[atom]
    return State(q, p)


def _options(classifier, **kwargs):
    return {"descriptor": classifier.native_outcome_descriptor(),
            "check_interval": 1, "persistence_steps": 0, **kwargs}


def _label(classifier, current):
    result = classifier(current)
    return None if result is None else (result["status"], result.get("channel"))


@pytest.mark.parametrize("kind", ["connectivity", "h3"])
@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("beads", [1, 3, 16])
def test_native_classifier_matches_python_membership(native_library, kind, pair, beads):
    classifier = _h3() if kind == "h3" else _connectivity()
    current = _separated(pair, classifier.masses, beads=beads)
    if beads > 1:
        # Individual beads have different graphs; only the centroid is used.
        current.q[0, 0, 1] += 6.
        current.q[1, 0, 1] -= 6.
    expected = _label(classifier, current)
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., beads, 0.)
    got = dyn.propagate_block(current, 1, outcome=_options(classifier), backend="native")
    assert got.outcome == {"candidate": expected, "candidate_since": 1,
                           "resolved": expected, "checks": 1, "pending_check": False}
    assert got.success


@pytest.mark.parametrize("kind", ["connectivity", "h3"])
@pytest.mark.parametrize("bond,separation,outward", [
    (np.nextafter(1.5, 0.), 11., 1.),
    (1.5, 11., 1.),
    (np.nextafter(1.5, 2.), 11., 1.),
    (1., np.nextafter(10., 0.), 1.),
    (1., 10., 1.),
    (1., np.nextafter(10., 11.), 1.),
    (1., 11., 0.),
    (1., 11., -1.),
])
def test_native_classifier_thresholds_preserve_python_decisions(native_library, kind, bond, separation, outward):
    classifier = _h3() if kind == "h3" else _connectivity()
    current = _separated(masses=classifier.masses, bond=bond, separation=separation, outward=outward)
    expected = _label(classifier, current)
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, 0.)
    got = dyn.propagate_block(current, 1, outcome=_options(classifier), backend="native")
    if got.outcome["pending_check"]:
        # A roundoff ambiguity changes only where the strict test executes.
        assert got.outcome["candidate"] is None and got.outcome["checks"] == 0
        assert _label(classifier, current) == expected
        return
    assert got.outcome["resolved"] == expected
    assert got.outcome["candidate"] == expected
    assert got.outcome["candidate_since"] == (None if expected is None else 1)


@pytest.mark.parametrize("kind", ["connectivity", "h3"])
def test_connected_complex_and_three_fragments(native_library, kind):
    classifier = _h3() if kind == "h3" else _connectivity()
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, 0.)
    for coordinates in ([[0., 0., 0.], [0.5, 0., 0.], [0., 0.5, 0.]],
                        [[-20., 0., 0.], [20., 0., 0.], [0., 30., 0.]]):
        q = np.asarray(coordinates)[None]
        current = State(q, q * np.asarray(classifier.masses)[None, :, None])
        expected = _label(classifier, current)
        got = dyn.propagate_block(current, 1, outcome=_options(classifier), backend="native")
        assert got.outcome["resolved"] == expected
        if kind == "connectivity" and coordinates[0][0] == -20.:
            assert expected == ("product", "dissociation")


def test_native_persistence_exact_cadence_and_early_stop(native_library):
    classifier = _h3()
    current = _separated(pair=(0, 1))
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, 0.)
    got = dyn.propagate_block(current, 80, outcome=_options(
        classifier, check_interval=5, persistence_steps=10), backend="native")
    assert current.step == got.completed == 15
    assert got.requested == 80 and got.success
    assert got.outcome == {"candidate": ("product", "H2[0,1]+H"), "candidate_since": 5,
                           "resolved": ("product", "H2[0,1]+H"), "checks": 3, "pending_check": False}
    np.testing.assert_array_equal(got.steps, np.arange(1, 16))


def test_persistence_crosses_arbitrary_blocks_and_resets_on_unresolved(native_library):
    classifier = _h3()
    current = _separated()
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, 0.)
    first = dyn.propagate_block(current, 5, outcome=_options(
        classifier, check_interval=4, persistence_steps=7))
    assert first.outcome["candidate_since"] == 4 and first.outcome["resolved"] is None
    second = dyn.propagate_block(current, 20, outcome=_options(
        classifier, check_interval=4, persistence_steps=7,
        candidate=first.outcome["candidate"], candidate_since=4))
    assert current.step == 12 and second.completed == 7
    assert second.outcome["resolved"] == ("nonreactive", None)
    current.p *= -1
    reset = dyn.propagate_block(current, 4, outcome=_options(
        classifier, check_interval=4, persistence_steps=7,
        candidate=second.outcome["candidate"], candidate_since=4))
    assert reset.outcome == {"candidate": None, "candidate_since": None, "resolved": None,
                             "checks": 1, "pending_check": False}


@pytest.mark.parametrize("cache", [False, True])
def test_force_reuse_exact_call_counts_and_same_trajectory(native_library, cache):
    pes = CompiledPotential(native_library)
    dyn = Dynamics(pes, [1800.] * 3, 220., 8, .2)
    original = state()
    expected = original.copy()
    ref = dyn.propagate_block(expected, 17, backend="native", record_states=True)
    pes.lib.reset(-1, 0)
    current = original.copy()
    got = dyn.propagate_block(current, 17, backend="native", record_states=True, reuse_forces=cache)
    assert pes.lib.get_calls() == (18 if cache else 34)
    np.testing.assert_array_equal(got.q, ref.q)
    np.testing.assert_array_equal(got.p, ref.p)
    np.testing.assert_array_equal(got.energies, ref.energies)


def test_force_cache_invalidates_between_blocks_and_after_python_edit(native_library):
    pes = CompiledPotential(native_library)
    dyn = Dynamics(pes, [1800.] * 3, 220., 8, .2)
    current = state()
    dyn.propagate_block(current, 3, backend="native", reuse_forces=True)
    current.q += .5
    reference = current.copy()
    pes.lib.reset(-1, 0)
    dyn.propagate_block(current, 4, backend="native", reuse_forces=True)
    assert pes.lib.get_calls() == 5
    dyn.propagate_block(reference, 4, backend="native")
    np.testing.assert_array_equal(current.q, reference.q)
    np.testing.assert_array_equal(current.p, reference.p)


@pytest.mark.parametrize("kind", ["bias", "fragments", "constraint"])
def test_force_cache_keeps_reference_schedule_for_projected_or_biased_steps(native_library, kind):
    from pyrpmd.workflows.rate import project_positions, project_momenta

    pes = CompiledPotential(native_library)
    dyn = Dynamics(pes, [1800.] * 3, 220., 8, .2)
    current = state()
    options = {}
    if kind == "bias":
        options["bias"] = UmbrellaBias(Distance([0, 1]), [2.], .01)
    elif kind == "fragments":
        options["fragment_data"] = dyn._fragment_data(current, [[0], [1, 2]])
        dyn._fix_fragments(current, options["fragment_data"])
    else:
        surface = coordinate().dividing_surface(1.001)
        current.q[:] = project_positions(current.q, surface, dyn.masses)
        project_momenta(current.p, current.q, surface, dyn.masses)
        options["constraint"] = surface
    reference = current.copy()
    dyn.propagate_block(reference, 5, backend="native", **options)
    pes.lib.reset(-1, 0)
    dyn.propagate_block(current, 5, backend="native", reuse_forces=True, **options)
    assert pes.lib.get_calls() == (15 if kind == "fragments" else 10)
    np.testing.assert_array_equal(current.q, reference.q)
    np.testing.assert_array_equal(current.p, reference.p)


@pytest.mark.parametrize("nonfinite", [0, 1])
def test_cached_force_failure_retains_valid_prefix_and_does_not_retry(native_library, nonfinite):
    pes = CompiledPotential(native_library)
    dyn = Dynamics(pes, [1800.] * 3, 220., 8, .2)
    original = state()
    reference = original.copy()
    dyn.propagate_block(reference, 2, backend="native")
    pes.lib.reset(4, nonfinite)  # Initial force, then one force per step: step 3 fails.
    got = dyn.propagate_block(original, 30, backend="native", reuse_forces=True,
                              record_previous=True, raise_on_error=False)
    assert got.completed == 2 and got.error and not got.success
    assert pes.lib.get_calls() == 4
    assert got.previous_q is got.previous_p is got.previous_time is None
    np.testing.assert_array_equal(original.q, reference.q)
    np.testing.assert_array_equal(original.p, reference.p)


@pytest.mark.parametrize("backend", ["native", "python"])
@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("observe", [False, True])
def test_previous_state_is_exact_without_full_history(native_library, backend, steps, observe):
    dyn = Dynamics(CompiledPotential(native_library), [1800.] * 3, 220., 8, .2)
    initial = state()
    initial.time = .123456789
    reference = initial.copy()
    dyn.propagate_block(reference, steps - 1, backend=backend)
    got = dyn.propagate_block(initial, steps, backend=backend, record_previous=True,
                             observe=[observe] * steps)
    assert got.q is got.p is None
    assert got.previous_step == steps - 1
    assert got.previous_time == reference.time
    np.testing.assert_array_equal(got.previous_q, reference.q)
    np.testing.assert_array_equal(got.previous_p, reference.p)
    assert not np.shares_memory(got.previous_q, initial.q)


def test_outcomes_reject_unsupported_backend_before_mutation(native_library):
    classifier = _h3()
    current = _separated()
    original = current.copy()
    dyn = Dynamics(HarmonicPotential(), classifier.masses, 100., 1, .1)
    assert not native_outcome_supported(dyn)
    with pytest.raises(RuntimeError, match="native outcomes require"):
        dyn.propagate_block(current, 20, outcome=_options(classifier))
    np.testing.assert_array_equal(current.q, original.q)
    np.testing.assert_array_equal(current.p, original.p)
    assert current.step == 0
    native = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, .1)
    assert native_outcome_supported(native)
    with pytest.raises(RuntimeError, match="native outcomes require"):
        native.propagate_block(current, 20, outcome=_options(classifier), backend="python")


def test_native_outcome_failure_keeps_only_accepted_candidate_checks(native_library):
    classifier = _h3()
    pes = CompiledPotential(native_library)
    dyn = Dynamics(pes, classifier.masses, 100., 1, 0.)
    current = _separated()
    pes.lib.reset(6, 0)  # Five attempted steps with caching, fifth final force fails.
    got = dyn.propagate_block(current, 30, outcome=_options(
        classifier, check_interval=2, persistence_steps=100), reuse_forces=True,
        backend="native", raise_on_error=False)
    assert got.completed == current.step == 4
    assert pes.lib.get_calls() == 6 and got.error
    assert got.outcome == {"candidate": ("nonreactive", None), "candidate_since": 2,
                           "resolved": None, "checks": 2, "pending_check": False}


def test_nonaxis_boundary_yields_python_check_without_advancing_persistence(native_library):
    classifier = _h3()
    bond = np.array([.06960505283532678, -.4097014693788604, -.059652509543542676])
    classifier.bond_cutoff = float(np.linalg.norm(bond))
    classifier.exit_distance = 5.
    current = State(np.array([[np.zeros(3), bond, [10., 10., 10.]]]),
                    np.array([[[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]]]))
    original = current.copy()
    assert classifier(current) is None
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, 0.)
    got = dyn.propagate_block(current, 30, outcome=_options(
        classifier, check_interval=2, persistence_steps=4,
        candidate=("product", "H2[0,1]+H"), candidate_since=0), record_previous=True)
    assert got.completed == 2 and got.success
    assert got.outcome == {"candidate": ("product", "H2[0,1]+H"), "candidate_since": 0,
                           "resolved": None, "checks": 0, "pending_check": True}
    np.testing.assert_array_equal(current.q, original.q)
    np.testing.assert_array_equal(got.previous_q, original.q)
    assert got.previous_step == 1


def test_native_early_stop_matches_python_boundary_state_at_nonzero_dt(native_library):
    classifier = _h3()
    pes = CompiledPotential(native_library)
    dyn = Dynamics(pes, classifier.masses, 100., 3, .001)
    current = _separated(pair=(0, 2), beads=3)
    reference = current.copy()
    expected = dyn.propagate_block(reference, 6, backend="python", record_states=True)
    assert _label(classifier, reference) == ("product", "H2[0,2]+H")
    got = dyn.propagate_block(current, 20, backend="native", outcome=_options(
        classifier, check_interval=2, persistence_steps=4),
        record_previous=True, reuse_forces=True)
    assert got.completed == 6 and got.success
    np.testing.assert_allclose(current.q, reference.q, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(current.p, reference.p, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(got.energies, expected.energies, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(got.previous_q, expected.q[-2], atol=1e-14, rtol=1e-14)
    assert got.previous_time == expected.times[-2]


@pytest.mark.parametrize("change", [
    {"kind": "custom"}, {"masses": [1., 1.]}, {"masses": [1., 0., 1.]},
    {"bond_cutoff": 12.}, {"channel_labels": ["unexpected"]},
])
def test_invalid_native_descriptor_is_rejected_before_state_mutation(native_library, change):
    classifier = _h3()
    current = _separated()
    original = current.copy()
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, .1)
    options = _options(classifier)
    options["descriptor"].update(change)
    with pytest.raises(ValueError):
        dyn.propagate_block(current, 20, outcome=options)
    assert current.step == 0
    np.testing.assert_array_equal(current.q, original.q)
    np.testing.assert_array_equal(current.p, original.p)


@pytest.mark.parametrize("change", [
    {"check_interval": 0}, {"check_interval": True}, {"persistence_steps": -1},
    {"candidate": ("product", "missing"), "candidate_since": 0},
    {"candidate": None, "candidate_since": 0}, {"start_step": 2**63 - 1},
])
def test_invalid_outcome_state_does_not_mutate(native_library, change):
    classifier = _h3()
    current = _separated()
    original = current.copy()
    dyn = Dynamics(CompiledPotential(native_library), classifier.masses, 100., 1, .1)
    with pytest.raises(ValueError):
        dyn.propagate_block(current, 2, outcome=_options(classifier, **change))
    assert current.step == 0
    np.testing.assert_array_equal(current.q, original.q)
    np.testing.assert_array_equal(current.p, original.p)
