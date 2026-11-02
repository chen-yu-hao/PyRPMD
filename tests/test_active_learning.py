"""Transactional budgets, scientific identity and exact callback replay."""
import numpy as np
import pytest

from pyrpmd.active_learning import ActiveLearningLoop, SamplingBatch
from pyrpmd.potentials.base import PESResult
from pyrpmd.potentials.lifecycle import label_candidates


REFERENCE = {"provider": "analytic_harmonic", "k": 1.0, "version": 1}


class Harmonic:
    def __init__(self, k=1.0):
        self.k = float(k)

    def evaluate(self, q):
        return PESResult(self.k * np.sum(q*q, axis=(1, 2))/2, -self.k*q)


class Codec:
    identity = {"codec": "harmonic", "schema": 1}

    def encode(self, model):
        return {"k": np.asarray(model.k, dtype=np.float64)}

    def decode(self, payload):
        assert set(payload) == {"k"}
        return Harmonic(float(payload["k"]))


class Reference(Harmonic):
    def __init__(self, failure=None):
        super().__init__()
        self.calls, self.failure = [], failure

    def evaluate(self, q):
        self.calls.append(q.copy())
        if self.failure == "exception":
            raise RuntimeError("reference worker failed")
        result = super().evaluate(q)
        if self.failure == "partial":
            bad = q[:, 0, 0] > 0
            result.status[bad] = 4
            result.messages = tuple("SCF not converged" if b else "" for b in bad)
        return result


def sample(committee, rng, state):
    q = rng.normal(size=(6, 2, 3))
    return SamplingBatch(q, {"q": q.copy(), "count": 1 if state is None else state["count"] + 1,
                             "precise": np.uint64(2**64-1), "tuple": (np.float32(.1), 7)})


def train(dataset, previous_models):
    assert np.all(dataset.status == 0)
    # The force reference independently determines k; the finite offset keeps
    # a meaningful committee through several rounds for restart tests.
    k = -np.sum(dataset.forces*dataset.positions)/np.sum(dataset.positions**2)
    width = .3/(1+len(dataset))
    return [Harmonic(k-width), Harmonic(k+width)]


def make_loop(directory, *, resume=False, **changes):
    options = dict(reference=Reference(), reference_identity=REFERENCE,
                   sampler=sample, sampler_identity={"method": "gaussian", "shape": [6, 2, 3]},
                   trainer=train, trainer_identity="harmonic_force_fit.v1", model_codec=Codec(),
                   force_threshold=1e-8, max_rounds=3, max_labels=20,
                   labels_per_round=5, label_batch_size=2, seed=108)
    options.update(changes)
    return ActiveLearningLoop(directory, None if resume else [Harmonic(.7), Harmonic(1.3)], **options)


def test_three_rounds_keep_versions_full_reference_provenance_and_rng(tmp_path):
    reference = Reference()
    loop = make_loop(tmp_path, reference=reference)
    result = loop.run()
    assert result["stop_reason"] == "round_budget_exhausted"
    assert result["label_attempts"] == result["successful_labels"] == 15
    assert [len(q) for q in reference.calls] == [2, 2, 1]*3
    assert result["rounds_completed"] == 3 and len(result["model_versions"]) == 4
    assert not result["converged"] and not result["physical_converged"]
    for index, event in enumerate(result["history"]):
        assert event["round"] == index and event["trained"]
        assert event["committee"] == result["model_versions"][index]["sha256"]
        assert event["next_committee"] == result["model_versions"][index+1]["sha256"]
        assert event["selected_force_rmse_after"] < event["selected_force_rmse_before"] + 1e-15
        assert event["error_scope"] == "selected_training_labels_only"
    for row in loop.dataset.provenance:
        event = result["history"][row["round"]]
        assert row["sample_sha256"] == event["sample"]
        assert row["committee_sha256"] == event["committee"]
    snapshot = loop.sampler_state
    snapshot["q"][:] = 111
    assert not np.any(loop.sampler_state["q"] == 111)
    restored = make_loop(tmp_path, resume=True)
    assert restored.result == result
    assert restored.rng_state == loop.rng_state
    assert restored.sampler_state["precise"].dtype == np.uint64
    assert restored.sampler_state["tuple"][0].dtype == np.float32


@pytest.mark.parametrize("after_steps", [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15])
def test_interruption_at_every_round_phase_matches_uninterrupted(tmp_path, after_steps):
    full = make_loop(tmp_path / "full")
    full.run()
    partial = make_loop(tmp_path / "part")
    partial.run(max_steps=after_steps)
    resumed = make_loop(tmp_path / "part", resume=True)
    resumed.run()
    assert resumed.result == full.result
    assert resumed.rng_state == full.rng_state
    np.testing.assert_array_equal(resumed.sampler_state["q"], full.sampler_state["q"])
    assert resumed.dataset.sha256 == full.dataset.sha256


@pytest.mark.parametrize("phase_steps", [0, 1, 2, 4])
def test_crash_before_checkpoint_replace_replays_uncommitted_callback(tmp_path, monkeypatch, phase_steps):
    full = make_loop(tmp_path / "full", max_rounds=1)
    full.run()
    interrupted = make_loop(tmp_path / "part", max_rounds=1)
    interrupted.run(max_steps=phase_steps)
    original_result, original_rng = interrupted.result, interrupted.rng_state

    def power_loss(*args, **kwargs):
        raise KeyboardInterrupt("power loss before commit")

    monkeypatch.setattr(interrupted, "_write_checkpoint", power_loss)
    with pytest.raises(KeyboardInterrupt, match="power loss"):
        interrupted.step()
    assert interrupted.result == original_result
    assert interrupted.rng_state == original_rng
    resumed = make_loop(tmp_path / "part", resume=True, max_rounds=1)
    resumed.run()
    assert resumed.result == full.result


@pytest.mark.parametrize("stage", ["sample", "train"])
def test_interrupted_callback_keeps_rng_and_model_state_replayable(tmp_path, stage):
    full = make_loop(tmp_path / "full", max_rounds=1)
    full.run()

    def broken_sample(committee, rng, state):
        rng.random(37)
        committee.members[0].k = 900
        raise KeyboardInterrupt("callback interruption")

    def broken_train(dataset, old):
        old[0].k = 900
        raise KeyboardInterrupt("callback interruption")

    loop = make_loop(tmp_path / "part", max_rounds=1)
    if stage == "train":
        loop.run(max_steps=4)
        loop.trainer = broken_train
    else:
        loop.sampler = broken_sample
    before = loop.rng_state
    with pytest.raises(KeyboardInterrupt, match="callback interruption"):
        loop.step()
    assert loop.rng_state == before
    resumed = make_loop(tmp_path / "part", resume=True, max_rounds=1)
    resumed.run()
    assert resumed.result == full.result


def test_label_budget_counts_failures_and_last_partial_batch(tmp_path):
    reference = Reference(failure="partial")
    loop = make_loop(tmp_path, reference=reference, max_labels=7)
    result = loop.run()
    assert result["stop_reason"] == "label_budget_exhausted"
    assert result["label_attempts"] == 7
    assert result["successful_labels"] + result["failed_labels"] == 7
    assert result["failed_labels"] > 0
    assert [len(q) for q in reference.calls] == [2, 2, 1, 2]
    assert np.isnan(loop.dataset.energy[~loop.dataset.successful_mask]).all()


def test_reference_exceptions_preserve_attempts_and_stop_without_training(tmp_path):
    def forbidden(*args):
        raise AssertionError("no labels available for training")

    reference = Reference(failure="exception")
    loop = make_loop(tmp_path, reference=reference, trainer=forbidden)
    result = loop.run()
    assert result["stop_reason"] == "reference_failed"
    assert result["failed_labels"] == result["label_attempts"] == 5
    assert len(result["model_versions"]) == 1
    assert all("reference worker failed" in message for message in loop.dataset.messages)
    resumed = make_loop(tmp_path, resume=True, reference=reference, trainer=forbidden)
    assert resumed.run() == result
    assert len(reference.calls) == 3


def test_small_disagreement_only_certifies_the_sampled_batch(tmp_path):
    reference = Reference()
    loop = make_loop(tmp_path, reference=reference, force_threshold=100)
    result = loop.run()
    assert result["stop_reason"] == "uncertainty_below_threshold"
    assert result["uncertainty_converged"] is True
    assert result["uncertainty_scope"] == "current_sampled_batch_only"
    assert result["uncertainty_is_calibrated_error"] is False
    assert result["physical_converged"] is False and result["converged"] is False
    assert reference.calls == [] and loop.dataset is None


def test_initial_labels_deduplication_and_energy_only_selection(tmp_path):
    q = np.zeros((4, 2, 3))
    q[:, 0, 0] = [1, 2, 2, 3]
    initial = label_candidates(Reference(), q[-1:], reference_identity=REFERENCE)

    def fixed(*args):
        return SamplingBatch(q)

    loop = make_loop(tmp_path, sampler=fixed, initial_dataset=initial,
                     force_threshold=None, energy_threshold=.01, labels_per_round=5, max_labels=7)
    result = loop.run()
    assert result["stop_reason"] == "no_new_candidates"
    assert result["label_attempts"] == 2 and len(loop.dataset) == 3
    # Descending disagreement chooses x=2 before x=1; x=3 was already labelled.
    np.testing.assert_array_equal(loop.dataset.positions[:, 0, 0], [3, 2, 1])


def test_extreme_valid_threshold_keeps_descending_disagreement_order(tmp_path):
    q = np.zeros((3, 2, 3))
    q[:, 0, 0] = [1, 2, 3]
    loop = make_loop(tmp_path, sampler=lambda *args: SamplingBatch(q),
                     force_threshold=1e-310, labels_per_round=1, max_labels=1)
    loop.run()
    assert loop.dataset.positions[0, 0, 0] == 3


@pytest.mark.parametrize("key,value", [
    ("force_threshold", .2), ("energy_threshold", .2), ("max_rounds", 4),
    ("max_labels", 19), ("labels_per_round", 3), ("label_batch_size", 1), ("seed", 12),
    ("reference_identity", {"provider": "changed"}), ("sampler_identity", "changed"),
    ("trainer_identity", "changed"), ("sampler_state", {"q": np.zeros((2, 3))}),
])
def test_every_scientific_option_enters_new_run_identity(tmp_path, key, value):
    make_loop(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        make_loop(tmp_path, resume=True, **{key: value})


@pytest.mark.parametrize("change", ["model", "dataset", "codec"])
def test_changed_initial_content_is_rejected(tmp_path, change):
    initial = label_candidates(Reference(), np.ones((1, 2, 3)), reference_identity=REFERENCE)
    loop = make_loop(tmp_path, initial_dataset=initial)
    if change == "model":
        original = Codec.encode
        class Changed(Codec):
            def encode(self, model):
                return {"k": original(self, model)["k"]+1}
        kwargs = dict(model_codec=Changed())
    elif change == "dataset":
        kwargs = dict(initial_dataset=label_candidates(Reference(), np.ones((1, 2, 3))*2,
                                                       reference_identity=REFERENCE))
    else:
        class Changed(Codec):
            identity = {"codec": "harmonic", "schema": 2}
        kwargs = dict(model_codec=Changed())
    with pytest.raises(ValueError, match="identity|codec"):
        # Supplying the initial models again must verify their contents too.
        make_loop(tmp_path, **kwargs)
    assert loop.identity


@pytest.mark.parametrize("kind", ["datasets", "models", "samples"])
def test_completed_resume_audits_committed_artifacts_before_callbacks(tmp_path, kind):
    loop = make_loop(tmp_path, max_rounds=1)
    loop.run()
    next((tmp_path / kind).glob("*.npz")).unlink()
    with pytest.raises(ValueError, match="Cannot read"):
        make_loop(tmp_path, resume=True, max_rounds=1)


def test_stale_writer_and_live_lock_fail_explicitly(tmp_path):
    from pyrpmd.active_learning import _exclusive
    first = make_loop(tmp_path)
    second = make_loop(tmp_path, resume=True)
    first.step()
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        second.step()
    with _exclusive(tmp_path):
        with pytest.raises(RuntimeError, match="another writer"):
            first.step()


@pytest.mark.parametrize("state", [object(), {"x": np.array([np.nan])}, {"x": {1: "bad key"}}, np.array([object()])])
def test_unserialisable_sampler_state_is_rejected_before_commit(tmp_path, state):
    def broken(*args):
        return SamplingBatch(np.ones((2, 2, 3)), state)
    loop = make_loop(tmp_path, sampler=broken)
    with pytest.raises(ValueError, match="state|JSON"):
        loop.step()
    assert loop.phase == "sample" and loop.round_index == 0


def test_mutating_train_or_lossy_codec_cannot_install_a_model(tmp_path):
    def mutating(dataset, old):
        old[0].k = 80
        return old
    loop = make_loop(tmp_path / "mutation", trainer=mutating)
    loop.run(max_steps=4)
    with pytest.raises(ValueError, match="mutated previous"):
        loop.step()
    assert loop.phase == "train" and len(loop.result["model_versions"]) == 1

    class Lossy(Codec):
        def decode(self, payload):
            return Harmonic(float(payload["k"])+.01)
    loop = make_loop(tmp_path / "lossy", model_codec=Lossy())
    with pytest.raises(ValueError, match="exactly"):
        loop.step()


@pytest.mark.parametrize("identity", [{1: "invalid"}, {"nested": {1: "invalid"}}, {"x": float("nan")}])
def test_identity_cannot_alias_nonstring_keys_or_nonfinite_values(tmp_path, identity):
    with pytest.raises(ValueError, match="keys|JSON"):
        make_loop(tmp_path, sampler_identity=identity)
