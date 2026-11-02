"""Independent audit of active-learning transaction and provenance boundaries."""

import json

import numpy as np
import pytest

from pyrpmd.active_learning import ActiveLearningLoop, SamplingBatch
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.potentials.base import PESResult
from pyrpmd.potentials.lifecycle import LabelledDataset


class Quadratic:
    def __init__(self, k, mutate=False):
        self.k = float(k)
        self.mutate = bool(mutate)

    def evaluate(self, q):
        if self.mutate:
            self.k += .01
        return PESResult(.5*self.k*np.sum(q*q, axis=(1, 2)), -self.k*q)


class Codec:
    identity = "audit-quadratic-v1"

    def encode(self, model):
        return {"k": model.k, "mutate": model.mutate}

    def decode(self, arrays):
        return Quadratic(float(arrays["k"]), bool(arrays["mutate"]))


def sample(committee, rng, state):
    round_index = 0 if state is None else state["round"]
    return SamplingBatch(rng.uniform(.2, 1.2, size=(5, 1, 3)), {"round": round_index+1})


def train(dataset, previous):
    return [Quadratic(1.0), Quadratic(1.1)]


def make_loop(path, *, trainer=train, **kwargs):
    return ActiveLearningLoop(path, [Quadratic(.5), Quadratic(1.5)], reference=Quadratic(1.),
                              reference_identity={"method": "harmonic", "k": 1.},
                              sampler=sample, sampler_identity=kwargs.pop("sampler_identity", "uniform-v1"),
                              trainer=trainer, trainer_identity="fit-v1", model_codec=Codec(),
                              force_threshold=.01, labels_per_round=4, label_batch_size=2,
                              max_rounds=2, max_labels=8, seed=17, **kwargs)


def alter_metadata(path, mutate):
    with load_checkpoint(path, require_integrity=True) as data:
        payload = dict(data.items())
    metadata = json.loads(str(payload["metadata"]))
    mutate(metadata)
    payload["metadata"] = json.dumps(metadata)
    atomic_savez(path, kind="PyRPMD.active-learning-loop.v1", **payload)


def test_model_mutation_during_post_fit_diagnostics_is_rejected(tmp_path):
    def mutating_train(dataset, previous):
        return [Quadratic(.9, mutate=True), Quadratic(1.1, mutate=True)]

    loop = make_loop(tmp_path, trainer=mutating_train)
    loop.run(max_steps=3)
    assert loop.phase == "train"
    with pytest.raises(ValueError, match="model|committee|mutat"):
        loop.step()
    assert loop.phase == "train" and len(loop.result["model_versions"]) == 1


def test_identity_cannot_coerce_non_string_mapping_keys(tmp_path):
    with pytest.raises(ValueError, match="key|JSON|identity"):
        make_loop(tmp_path, sampler_identity={1: "version"})


def test_validly_hashed_replacement_cannot_break_dataset_prefix(tmp_path):
    loop = make_loop(tmp_path)
    loop.run(max_steps=3)
    assert loop.phase == "train"
    with load_checkpoint(loop.checkpoint_path, require_integrity=True) as saved:
        metadata = json.loads(str(saved["metadata"]))
    original_path = tmp_path / "datasets" / (metadata["datasets"][0]+".npz")
    original = LabelledDataset.load(original_path)
    changed = original.positions.copy()
    changed[0, 0, 0] += .123
    replacement = LabelledDataset(changed, original.energy, original.forces, original.status,
                                  original.messages, reference_identity=original.reference_identity,
                                  provenance=original.provenance)
    replacement.save(tmp_path / "datasets" / (replacement.sha256+".npz"))
    alter_metadata(loop.checkpoint_path, lambda value: value["datasets"].__setitem__(0, replacement.sha256))
    with pytest.raises(ValueError, match="prefix|dataset|provenance|geometry"):
        make_loop(tmp_path)


def test_history_metadata_must_match_the_committed_sample(tmp_path):
    loop = make_loop(tmp_path)
    loop.run()
    alter_metadata(loop.checkpoint_path, lambda value: value["history"][0].__setitem__("round", 99))
    with pytest.raises(ValueError, match="history|sample|round|provenance"):
        make_loop(tmp_path)


def test_stop_reason_cannot_claim_uncertainty_convergence_after_budget_stop(tmp_path):
    loop = make_loop(tmp_path)
    result = loop.run()
    assert result["stop_reason"] == "label_budget_exhausted"
    assert result["history"][-1]["above_threshold"] > 0
    alter_metadata(loop.checkpoint_path, lambda value: value.__setitem__("reason", "uncertainty_below_threshold"))
    with pytest.raises(ValueError, match="reason|stop|progress|uncertainty"):
        make_loop(tmp_path)


def test_uncommitted_reference_batch_replays_to_same_result_and_rng(tmp_path, monkeypatch):
    baseline = make_loop(tmp_path / "baseline")
    baseline.run()
    interrupted = make_loop(tmp_path / "interrupted")
    interrupted.step()
    write = interrupted._write_checkpoint

    def lose_checkpoint(metadata, state):
        raise OSError("simulated crash before checkpoint replace")

    monkeypatch.setattr(interrupted, "_write_checkpoint", lose_checkpoint)
    with pytest.raises(OSError, match="simulated crash"):
        interrupted.step()
    assert interrupted.phase == "label" and interrupted.result["label_attempts"] == 0
    monkeypatch.setattr(interrupted, "_write_checkpoint", write)
    resumed = make_loop(tmp_path / "interrupted")
    resumed.run()
    assert resumed.result == baseline.result
    assert resumed.rng_state == baseline.rng_state
    assert resumed.sampler_state == baseline.sampler_state
    assert resumed.dataset.sha256 == baseline.dataset.sha256


def test_uncommitted_fit_replays_without_mutating_previous_versions(tmp_path, monkeypatch):
    baseline = make_loop(tmp_path / "baseline")
    baseline.run()
    interrupted = make_loop(tmp_path / "interrupted")
    interrupted.run(max_steps=3)
    write = interrupted._write_checkpoint

    def lose_checkpoint(metadata, state):
        raise OSError("simulated crash before checkpoint replace")

    monkeypatch.setattr(interrupted, "_write_checkpoint", lose_checkpoint)
    with pytest.raises(OSError, match="simulated crash"):
        interrupted.step()
    assert interrupted.phase == "train" and len(interrupted.result["model_versions"]) == 1
    monkeypatch.setattr(interrupted, "_write_checkpoint", write)
    resumed = make_loop(tmp_path / "interrupted")
    resumed.run()
    assert resumed.result == baseline.result
    assert resumed.rng_state == baseline.rng_state
    assert resumed.dataset.sha256 == baseline.dataset.sha256
