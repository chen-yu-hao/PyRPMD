"""Transactional sampling -> labelling -> fitting with immutable model versions.

This is an explicit Python workflow, independent of the input-file monitor.
Callbacks must be deterministic functions of their inputs: the sampler receives
its RNG and all persistent state; a trainer uses a declared, reproducible seed.
An interrupted callback is replayed from its last committed boundary. A model
never changes within a sampling segment. Committee disagreement is not a
calibrated error, and stopping this loop does not certify physical convergence.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .checkpoint import atomic_savez, load_checkpoint
from .path_integral import _integer, _positive
from .potentials.base import coordinates
from .potentials.lifecycle import CommitteePotential, LabelledDataset, label_candidates


_KIND = "PyRPMD.active-learning-loop.v1"
_MODEL_KIND = "PyRPMD.active-learning-model.v1"
_SAMPLE_KIND = "PyRPMD.active-learning-sample.v1"


def _json(value):
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("active-learning JSON mapping keys must be strings")
            for val in item.values():
                check(val)
        elif isinstance(item, (list, tuple)):
            for val in item:
                check(val)
    check(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("active-learning identities must contain finite JSON data") from error


def _identity(value, name):
    if not isinstance(value, (str, Mapping)) or not value or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{name} must be an explicit nonempty string or JSON mapping")
    return json.loads(_json(dict(value) if isinstance(value, Mapping) else value))


def _json_hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("invalid active-learning content hash")
    return value


def _arrays_hash(payload):
    digest = hashlib.sha256(b"PyRPMD.active-learning-arrays.v1\0")
    for name in sorted(payload):
        array = np.asarray(payload[name])
        header = _json([name, array.dtype.str, list(array.shape)]).encode()
        for value in (header, array.tobytes(order="C")):
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    return digest.hexdigest()


def _model_arrays(codec, model):
    result = codec.encode(model)
    if not isinstance(result, Mapping) or not result:
        raise ValueError("model_codec.encode must return a nonempty mapping of arrays")
    arrays = {}
    for key, value in result.items():
        if not isinstance(key, str) or not key:
            raise ValueError("model payload keys must be nonempty strings")
        array = np.array(value, copy=True)
        if array.dtype.kind not in "biufUS" or (array.dtype.kind == "f" and not np.isfinite(array).all()):
            raise ValueError("model payloads require finite real, boolean or string arrays without pickle")
        arrays[key] = array
    return arrays


def _pack_state(value):
    """Store nested sampler state without pickle or losing array dtype/shape."""
    arrays = {}

    def pack(item):
        if isinstance(item, (np.ndarray, np.generic)):
            array = np.array(item, copy=True)
            if array.dtype.kind not in "biufUS" or (array.dtype.kind == "f" and not np.isfinite(array).all()):
                raise ValueError("sampler state arrays must be finite and have no object/complex dtype")
            name = f"state_{len(arrays):06d}"
            arrays[name] = array
            return {"kind": "scalar" if isinstance(item, np.generic) else "array", "name": name}
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("sampler state mapping keys must be strings")
            return {"kind": "dict", "items": [[key, pack(val)] for key, val in item.items()]}
        if isinstance(item, (list, tuple)):
            return {"kind": "tuple" if isinstance(item, tuple) else "list", "items": [pack(val) for val in item]}
        if item is None or type(item) in (str, bool, int, float):
            _json(item)
            return {"kind": "value", "value": item}
        raise ValueError("sampler state must contain JSON values, lists, tuples, mappings or numpy arrays")

    tree = pack(value)
    return tree, arrays


def _unpack_state(tree, arrays):
    used = set()

    def unpack(node):
        if not isinstance(node, dict) or "kind" not in node:
            raise ValueError("invalid sampler state tree")
        kind = node["kind"]
        if kind in ("array", "scalar") and set(node) == {"kind", "name"}:
            name = node["name"]
            if name not in arrays or name in used:
                raise ValueError("invalid sampler state array inventory")
            used.add(name)
            value = arrays[name].copy()
            return value[()] if kind == "scalar" else value
        if kind == "value" and set(node) == {"kind", "value"}:
            if node["value"] is not None and type(node["value"]) not in (str, bool, int, float):
                raise ValueError("invalid sampler state value")
            _json(node["value"])
            return node["value"]
        if kind in ("dict", "list", "tuple") and set(node) == {"kind", "items"}:
            if not isinstance(node["items"], list):
                raise ValueError("invalid sampler state container")
            if kind == "dict":
                result = {}
                for key, value in node["items"]:
                    if not isinstance(key, str) or key in result:
                        raise ValueError("invalid sampler state mapping")
                    result[key] = unpack(value)
                return result
            result = [unpack(value) for value in node["items"]]
            return tuple(result) if kind == "tuple" else result
        raise ValueError("invalid sampler state tree")

    result = unpack(tree)
    if used != set(arrays):
        raise ValueError("unreferenced sampler state array")
    # Apply the same finite/type policy to saved data as to fresh callback data.
    _pack_state(result)
    return result


def _copy_state(value):
    tree, arrays = _pack_state(value)
    return _unpack_state(tree, arrays)


def _state_hash(value):
    tree, arrays = _pack_state(value)
    return _arrays_hash({"tree": np.asarray(_json(tree)), **arrays})


@contextmanager
def _exclusive(directory):
    """An OS lock is released even after process death; no stale lock removal."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".active-learning.lock").open("a+b") as stream:
        if os.name == "posix":
            import fcntl
            lock, unlock = (lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB),
                            lambda: fcntl.flock(stream, fcntl.LOCK_UN))
        elif os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            lock, unlock = (lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1),
                            lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            raise RuntimeError("active-learning transactions require a supported OS file lock")
        try:
            lock()
        except OSError as error:
            raise RuntimeError("another writer is using this active-learning directory") from error
        try:
            yield
        finally:
            unlock()


@dataclass(frozen=True)
class SamplingBatch:
    """Independent candidate configurations and the sampler's complete next state.

    Positions have shape [B,N,3] in bohr. State may contain nested finite JSON
    values and numpy arrays, including momenta, clocks and thermostat state.
    The loop separately checkpoints the supplied numpy Generator's state.
    """

    positions: np.ndarray
    state: object = None


class ActiveLearningLoop:
    """Resume a deterministic active-learning loop after any committed stage.

    ``sampler(committee, rng, state) -> SamplingBatch`` owns the sampling method;
    put *all* evolving sampler state in the returned batch. ``trainer(dataset,
    previous_models) -> models`` receives successful reference labels only.
    It must return at least two separate providers and must not mutate its
    inputs. A trainer's declared seed and dataset must completely determine its
    output; hidden/global RNG, optimizer or reference state is unsupported.

    ``model_codec.encode(model)`` returns named numpy arrays (or scalar values),
    and ``decode(arrays)`` reconstructs an independent provider. Its nonempty
    ``identity`` identifies the codec schema. Arbitrary pickle is never used.
    The caller supplies complete reference/sampler/trainer scientific identities,
    including method settings, package/code versions and external model hashes.

    One ``step`` commits a sampling segment, one reference batch, or one fit.
    Callbacks interrupted before commit are rerun from that boundary, so external
    callbacks must tolerate replay. Models, dataset snapshots and sampled frames
    are content-addressed and audited on restart. This class owns reconstructed
    models, while the caller owns the reference, callbacks and initial models.
    """

    def __init__(self, directory, models=None, *, reference, reference_identity,
                 sampler, sampler_identity, trainer, trainer_identity, model_codec,
                 force_threshold=None, energy_threshold=None, max_rounds=8,
                 max_labels=1000, labels_per_round=32, label_batch_size=32,
                 seed=0, sampler_state=None, initial_dataset=None):
        self.directory = Path(directory)
        self.checkpoint_path = self.directory / "checkpoint.npz"
        self.reference = reference
        self.sampler = sampler
        self.trainer = trainer if callable(trainer) else getattr(trainer, "train", None)
        self.codec = model_codec
        if not callable(sampler) or not callable(self.trainer):
            raise TypeError("sampler and trainer must implement their callback contracts")
        if not callable(getattr(model_codec, "encode", None)) or not callable(getattr(model_codec, "decode", None)):
            raise TypeError("model_codec must implement encode(model) and decode(arrays)")
        if not callable(getattr(reference, "evaluate", None)):
            raise TypeError("reference must implement evaluate(positions)")
        if force_threshold is None and energy_threshold is None:
            raise ValueError("at least one uncertainty threshold is required")
        self.codec_identity = _identity(getattr(model_codec, "identity", None), "model_codec.identity")
        configuration = {
            "schema": _KIND,
            "reference": _identity(reference_identity, "reference_identity"),
            "sampler": _identity(sampler_identity, "sampler_identity"),
            "trainer": _identity(trainer_identity, "trainer_identity"),
            "model_codec": self.codec_identity,
            "force_threshold": None if force_threshold is None else _positive(force_threshold, "force_threshold"),
            "energy_threshold": None if energy_threshold is None else _positive(energy_threshold, "energy_threshold"),
            "max_rounds": _integer(max_rounds, "max_rounds"),
            "max_labels": _integer(max_labels, "max_labels", 0),
            "labels_per_round": _integer(labels_per_round, "labels_per_round"),
            "label_batch_size": _integer(label_batch_size, "label_batch_size"),
            "seed": _integer(seed, "seed", 0),
            "selection": "descending_max_normalized_disagreement_unique_geometry.v1",
            "units": {"positions": "bohr", "energy": "hartree", "forces": "hartree/bohr"},
        }
        # Validate stateless/batch-independent reference capabilities before any
        # sampling or filesystem commit; an empty batch performs no evaluation.
        label_candidates(reference, np.empty((0, 1, 3)),
                         reference_identity=configuration["reference"], batch_size=1)
        self._closed = False
        with _exclusive(self.directory):
            if self.checkpoint_path.exists():
                metadata, state = self._read_checkpoint()
                original = metadata["configuration"]
                configuration["initial_models"] = (self._encode_models(models)[0] if models is not None
                                                    else original["initial_models"])
                configuration["initial_dataset"] = (initial_dataset.sha256 if initial_dataset is not None
                                                     else original["initial_dataset"])
                configuration["initial_state"] = (_state_hash(sampler_state) if sampler_state is not None
                                                   else original["initial_state"])
                if configuration != original or _json_hash(configuration) != metadata["identity"]:
                    raise ValueError("active-learning run identity differs from its checkpoint")
                self._meta, self._sampler_state = metadata, state
                self._validate_metadata()
                self._audit()
            else:
                if models is None:
                    raise ValueError("a new active-learning run requires initial models")
                hashes, payloads = self._encode_models(models)
                configuration["initial_models"] = hashes
                configuration["initial_state"] = _state_hash(sampler_state)
                if initial_dataset is not None:
                    if not isinstance(initial_dataset, LabelledDataset):
                        raise TypeError("initial_dataset must be a LabelledDataset")
                    if initial_dataset.reference_identity != configuration["reference"]:
                        raise ValueError("initial dataset reference identity differs")
                configuration["initial_dataset"] = None if initial_dataset is None else initial_dataset.sha256
                self._meta = {
                    "format": _KIND, "configuration": configuration,
                    "identity": _json_hash(configuration), "sequence": 0,
                    "phase": "sample", "round": 0, "reason": None,
                    "label_attempts": 0, "dataset": configuration["initial_dataset"],
                    "datasets": [] if initial_dataset is None else [initial_dataset.sha256],
                    "versions": [{"members": hashes, "sha256": self._committee_hash(hashes),
                                  "dataset": configuration["initial_dataset"], "round": None}],
                    "history": [], "pending": None,
                    "rng": np.random.default_rng(configuration["seed"]).bit_generator.state,
                }
                self._sampler_state = _copy_state(sampler_state)
                self._save_models(hashes, payloads)
                if initial_dataset is not None:
                    self._save_dataset(initial_dataset)
                self._write_checkpoint(self._meta, self._sampler_state)

    @property
    def identity(self):
        return self._meta["identity"]

    @property
    def phase(self):
        return self._meta["phase"]

    @property
    def round_index(self):
        return self._meta["round"]

    @property
    def history(self):
        return deepcopy(self._meta["history"])

    @property
    def sampler_state(self):
        return _copy_state(self._sampler_state)

    @property
    def rng_state(self):
        return deepcopy(self._meta["rng"])

    @property
    def dataset(self):
        digest = self._meta["dataset"]
        return None if digest is None else self._load_dataset(digest)

    @property
    def committee(self):
        """Return a new owned committee; callers should close it after use."""
        return CommitteePotential(self._load_models(self._meta["versions"][-1]["members"]), owns_members=True)

    @property
    def result(self):
        dataset = self.dataset
        return {
            "identity": self.identity, "phase": self.phase, "complete": self.phase == "done",
            "stop_reason": self._meta["reason"], "rounds_completed": self.round_index,
            "label_attempts": self._meta["label_attempts"],
            "successful_labels": 0 if dataset is None else int(dataset.successful_mask.sum()),
            "failed_labels": 0 if dataset is None else int((~dataset.successful_mask).sum()),
            "dataset_sha256": self._meta["dataset"],
            "model_versions": deepcopy(self._meta["versions"]), "history": self.history,
            "uncertainty_converged": self._meta["reason"] == "uncertainty_below_threshold",
            "uncertainty_scope": "current_sampled_batch_only",
            "uncertainty_is_calibrated_error": False,
            "converged": False, "physical_converged": False,
        }

    def _committee_hash(self, hashes):
        return _json_hash({"codec": self.codec_identity, "members": hashes})

    def _encode_models(self, models):
        models = tuple(models)
        CommitteePotential(models)  # also checks separate provider objects
        payloads, hashes = [], []
        for model in models:
            values = _model_arrays(self.codec, model)
            names = sorted(values)
            payload = {"names": np.asarray(_json(names)), "codec": np.asarray(_json(self.codec_identity))}
            payload.update({f"array_{i:06d}": values[name] for i, name in enumerate(names)})
            payloads.append(payload)
            hashes.append(_arrays_hash(payload))
        return hashes, payloads

    def _model_payload(self, digest):
        with load_checkpoint(self.directory / "models" / (_digest(digest) + ".npz"),
                             expected_kind=_MODEL_KIND, require_integrity=True) as saved:
            payload = dict(saved.items())
        if _arrays_hash(payload) != digest or str(payload.get("codec")) != _json(self.codec_identity):
            raise ValueError("model content or codec hash differs")
        try:
            names = json.loads(str(payload["names"]))
        except (ValueError, KeyError) as error:
            raise ValueError("invalid model payload names") from error
        if (not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names)
                or names != sorted(set(names))
                or set(payload) != {"names", "codec"} | {f"array_{i:06d}" for i in range(len(names))}):
            raise ValueError("invalid model payload inventory")
        return {name: payload[f"array_{i:06d}"] for i, name in enumerate(names)}

    def _save_models(self, hashes, payloads):
        for digest, payload in zip(hashes, payloads):
            path = self.directory / "models" / (digest + ".npz")
            if path.exists():
                self._model_payload(digest)
            else:
                atomic_savez(path, kind=_MODEL_KIND, **payload)

    def _load_models(self, hashes):
        models = []
        try:
            for digest in hashes:
                model = self.codec.decode(self._model_payload(digest))
                models.append(model)
            # A lossy codec (e.g. an implicit dtype/device conversion) is unsafe.
            if self._encode_models(models)[0] != hashes:
                raise ValueError("model codec does not reproduce the committed model exactly")
            return tuple(models)
        except BaseException:
            self._close_models(models)
            raise

    @staticmethod
    def _close_models(models):
        from .potentials.lifecycle import _close_all
        _close_all(models)

    def _save_dataset(self, dataset):
        path = self.directory / "datasets" / (dataset.sha256 + ".npz")
        if path.exists():
            self._load_dataset(dataset.sha256)
        else:
            dataset.save(path)

    def _load_dataset(self, digest):
        dataset = LabelledDataset.load(self.directory / "datasets" / (_digest(digest) + ".npz"))
        if dataset.sha256 != digest or dataset.reference_identity != self._meta["configuration"]["reference"]:
            raise ValueError("dataset content or reference identity differs")
        return dataset

    def _write_checkpoint(self, metadata, state):
        tree, arrays = _pack_state(state)
        atomic_savez(self.checkpoint_path, kind=_KIND,
                     metadata=_json(metadata), state_tree=_json(tree), **arrays)

    def _read_checkpoint(self):
        with load_checkpoint(self.checkpoint_path, expected_kind=_KIND, require_integrity=True) as saved:
            if {"metadata", "state_tree"} - set(saved.files):
                raise ValueError("incomplete active-learning checkpoint")
            metadata = json.loads(str(saved["metadata"]))
            tree = json.loads(str(saved["state_tree"]))
            state = _unpack_state(tree, {name: saved[name] for name in saved.files
                                         if name not in ("metadata", "state_tree")})
        return metadata, state

    def _validate_metadata(self):
        value = self._meta
        required = {"format", "configuration", "identity", "sequence", "phase", "round", "reason",
                    "label_attempts", "dataset", "datasets", "versions", "history", "pending", "rng"}
        if set(value) != required or value["format"] != _KIND or _json_hash(value["configuration"]) != value["identity"]:
            raise ValueError("invalid active-learning checkpoint metadata or identity")
        if value["phase"] not in ("sample", "label", "train", "done"):
            raise ValueError("invalid active-learning phase")
        for key in ("sequence", "round", "label_attempts"):
            if type(value[key]) is not int or value[key] < 0:
                raise ValueError("invalid active-learning counter")
        cfg = value["configuration"]
        if (value["round"] > cfg["max_rounds"] or value["label_attempts"] > cfg["max_labels"]
                or len(value["history"]) != value["round"] or not value["versions"]
                or (value["phase"] in ("label", "train")) != (value["pending"] is not None)
                or (value["phase"] == "done") != (value["reason"] is not None)):
            raise ValueError("inconsistent active-learning progress")
        rng = np.random.default_rng()
        rng.bit_generator.state = value["rng"]
        if value["versions"][0]["members"] != cfg["initial_models"]:
            raise ValueError("initial models differ from scientific identity")
        for version in value["versions"]:
            if len(version["members"]) < 2 or version["sha256"] != self._committee_hash(version["members"]):
                raise ValueError("invalid active-learning model version")
        if value["dataset"] is not None and (not value["datasets"] or value["datasets"][-1] != value["dataset"]):
            raise ValueError("dataset version chain differs")

    def _audit(self):
        for digest in set(item for version in self._meta["versions"] for item in version["members"]):
            self._model_payload(digest)
        meta, cfg = self._meta, self._meta["configuration"]
        initial = cfg["initial_dataset"]
        dataset = None if initial is None else self._load_dataset(initial)
        initial_count = 0 if dataset is None else len(dataset)
        snapshot_index = 0 if initial is None else 1
        if initial is not None and (not meta["datasets"] or meta["datasets"][0] != initial):
            raise ValueError("initial dataset is missing from the version chain")
        if meta["versions"][0] != {"members": cfg["initial_models"],
                                   "sha256": self._committee_hash(cfg["initial_models"]),
                                   "dataset": initial, "round": None}:
            raise ValueError("invalid initial model version")
        version_index, attempted, sequence = 0, 0, 0
        terminal_reason = None
        entries = [(event, False) for event in meta["history"]]
        if meta["pending"] is not None:
            entries.append((meta["pending"], True))
        for round_index, (event, pending) in enumerate(entries):
            if terminal_reason is not None:
                raise ValueError("active-learning history continues after a terminal event")
            sample = self._load_sample(event["sample"])
            record = json.loads(str(sample["record"]))
            if record["round"] != round_index or record["committee"] != meta["versions"][version_index]["sha256"]:
                raise ValueError("sample round or model differs from the committed history")
            if not pending and any(event.get(key) != value for key, value in record.items()):
                raise ValueError("history metadata differs from the committed sample")
            if pending and event["dataset_before"] != (None if dataset is None else dataset.sha256):
                raise ValueError("pending dataset prefix differs from the committed history")
            selected = sample["selected"]
            expected, above = self._selection(sample["positions"], sample["energy_std"],
                                               sample["max_force_std"], dataset, attempted)
            if (not np.array_equal(selected, expected) or record["sample_count"] != len(sample["positions"])
                    or record["above_threshold"] != above):
                raise ValueError("committed candidate selection differs from its scientific policy")
            committed = event["labelled"] if pending else event["selected"]
            if (type(committed) is not int or not 0 <= committed <= len(selected)
                    or (not pending and committed != len(selected))
                    or (pending and (self.phase == "train") != (committed == len(selected)))):
                raise ValueError("invalid committed label progress")
            sequence += 1  # the sampled batch and its RNG/state commit
            for start in range(0, committed, cfg["label_batch_size"]):
                indices = selected[start:start + cfg["label_batch_size"]]
                if start + len(indices) > committed or snapshot_index >= len(meta["datasets"]):
                    raise ValueError("incomplete reference batch in the dataset version chain")
                following = self._load_dataset(meta["datasets"][snapshot_index])
                before = 0 if dataset is None else len(dataset)
                if len(following) != before + len(indices):
                    raise ValueError("dataset prefix length differs from committed labels")
                if dataset is not None:
                    for name in ("positions", "energy", "forces", "status"):
                        if not np.array_equal(getattr(following, name)[:before], getattr(dataset, name), equal_nan=True):
                            raise ValueError("dataset prefix differs from its previous committed version")
                    if (following.messages[:before] != dataset.messages or following.provenance[:before] != dataset.provenance):
                        raise ValueError("dataset prefix provenance differs from its previous version")
                provenance = tuple({"round": round_index, "candidate_index": int(index),
                                    "sample_sha256": event["sample"], "committee_sha256": record["committee"]}
                                   for index in indices)
                if (not np.array_equal(following.positions[before:], sample["positions"][indices])
                        or following.provenance[before:] != provenance):
                    raise ValueError("label provenance or geometry differs from the committed sample")
                dataset = following
                snapshot_index += 1
                attempted += len(indices)
                sequence += 1
            if not pending:
                current_hash = None if dataset is None else dataset.sha256
                good = 0 if committed == 0 else int(dataset.successful_mask[-committed:].sum())
                if (event["dataset"] != current_hash or event["successful"] != good
                        or event["failed"] != committed - good or event["trained"] != (good > 0)):
                    raise ValueError("history label counts or dataset version differ")
                if committed:
                    sequence += 1  # fit or explicit all-reference-failed stop
                if event["trained"]:
                    version_index += 1
                    if version_index >= len(meta["versions"]):
                        raise ValueError("trained model version is missing")
                    version = meta["versions"][version_index]
                    if (version["dataset"] != current_hash or version["round"] != round_index
                            or version["sha256"] != event["next_committee"]):
                        raise ValueError("trained model lineage differs from the committed dataset")
                if committed == 0:
                    terminal_reason = ("uncertainty_below_threshold" if above == 0 else
                                       "label_budget_exhausted" if attempted >= cfg["max_labels"] else
                                       "no_new_candidates")
                else:
                    terminal_reason = ("reference_failed" if good == 0 else
                                       "label_budget_exhausted" if attempted >= cfg["max_labels"] else
                                       "round_budget_exhausted" if round_index + 1 >= cfg["max_rounds"] else None)
        if (snapshot_index != len(meta["datasets"]) or version_index + 1 != len(meta["versions"])
                or attempted != meta["label_attempts"] or sequence != meta["sequence"]
                or (None if dataset is None else dataset.sha256) != meta["dataset"]
                or (0 if dataset is None else len(dataset)) != initial_count + attempted):
            raise ValueError("dataset/model chain or counters disagree with committed progress")
        if meta["reason"] != terminal_reason or (self.phase == "done") != (terminal_reason is not None):
            raise ValueError("stop reason contradicts the committed uncertainty and budget history")

    def _commit(self, metadata, state=None, *, replace_state=False):
        metadata["sequence"] = self._meta["sequence"] + 1
        value = state if replace_state else self._sampler_state
        self._write_checkpoint(metadata, value)
        self._meta, self._sampler_state = metadata, _copy_state(value)

    def _save_sample(self, arrays):
        digest = _arrays_hash(arrays)
        atomic_savez(self.directory / "samples" / (digest + ".npz"), kind=_SAMPLE_KIND, **arrays)
        return digest

    def _load_sample(self, digest):
        with load_checkpoint(self.directory / "samples" / (_digest(digest) + ".npz"),
                             expected_kind=_SAMPLE_KIND, require_integrity=True) as saved:
            arrays = dict(saved.items())
        required = {"positions", "predicted_energy", "predicted_forces", "energy_std", "max_force_std", "selected", "record"}
        if set(arrays) != required or _arrays_hash(arrays) != digest:
            raise ValueError("sample record inventory or content hash differs")
        q = coordinates(arrays["positions"])
        for name, shape in (("predicted_energy", (len(q),)), ("predicted_forces", q.shape),
                            ("energy_std", (len(q),)), ("max_force_std", (len(q),))):
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError("invalid sampled prediction or disagreement")
        if np.any(arrays["energy_std"] < 0) or np.any(arrays["max_force_std"] < 0):
            raise ValueError("negative sampled disagreement")
        selected = arrays["selected"]
        if (selected.dtype.kind not in "iu" or selected.ndim != 1 or len(set(selected.tolist())) != len(selected)
                or np.any(selected < 0) or np.any(selected >= len(q))):
            raise ValueError("invalid candidate indices")
        record = json.loads(str(arrays["record"]))
        if record["identity"] != self.identity:
            raise ValueError("sample record belongs to another run")
        return arrays

    @staticmethod
    def _geometry_key(q):
        return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes()).digest()

    @staticmethod
    def _statistics(result):
        return {"mean_energy_std": float(result.energy_std.mean()),
                "max_energy_std": float(result.energy_std.max()),
                "mean_max_force_std": float(result.max_force_std.mean()),
                "max_force_std": float(result.max_force_std.max())}

    def _selection(self, q, energy_std, max_force_std, dataset, attempted):
        cfg = self._meta["configuration"]
        # Compare normalized scores by mantissa/exponent, never forming a ratio
        # that can overflow when a valid threshold is extremely small. This also
        # retains ordering when both normalized scores would become infinity.
        exponent = np.full(len(q), np.iinfo(np.int32).min, dtype=np.int32)
        mantissa = np.zeros(len(q))
        high = np.zeros(len(q), dtype=bool)
        for values, threshold in ((max_force_std, cfg["force_threshold"]),
                                  (energy_std, cfg["energy_threshold"])):
            if threshold is None:
                continue
            high |= values >= threshold
            scale, power = np.frexp(threshold)
            fraction, powers = np.frexp(values)
            fraction, extra = np.frexp(fraction / scale)
            powers = np.where(values > 0, powers - power + extra, np.iinfo(np.int32).min)
            better = (powers > exponent) | ((powers == exponent) & (fraction > mantissa))
            exponent[better], mantissa[better] = powers[better], fraction[better]
        seen = set() if dataset is None else {self._geometry_key(row) for row in dataset.positions}
        selected = []
        allowance = min(cfg["labels_per_round"], cfg["max_labels"] - attempted)
        for index in np.lexsort((np.arange(len(q)), -mantissa, -exponent.astype(np.int64))):
            if not high[index] or len(selected) == allowance:
                break
            key = self._geometry_key(q[index])
            if key not in seen:
                seen.add(key)
                selected.append(int(index))
        return np.asarray(selected, dtype=np.int64), int(np.count_nonzero(high))

    def _sample(self):
        cfg, meta = self._meta["configuration"], deepcopy(self._meta)
        rng = np.random.default_rng()
        rng.bit_generator.state = meta["rng"]
        with self.committee as committee:
            batch = self.sampler(committee, rng, self.sampler_state)
            if not isinstance(batch, SamplingBatch):
                raise TypeError("sampler must return SamplingBatch(positions, state)")
            q, state = coordinates(batch.positions).copy(), _copy_state(batch.state)
            result = committee.evaluate_uncertainty(q)
            if self._encode_models(committee.members)[0] != meta["versions"][-1]["members"]:
                raise ValueError("sampler modified the fixed committee")
        dataset = self.dataset
        if dataset is not None and q.shape[1:] != dataset.positions.shape[1:]:
            raise ValueError("sample atom count differs from the labelled dataset")
        selected, above = self._selection(q, result.energy_std, result.max_force_std, dataset, meta["label_attempts"])
        allowance = min(cfg["labels_per_round"], cfg["max_labels"] - meta["label_attempts"])
        record = {"identity": self.identity, "round": meta["round"],
                  "committee": meta["versions"][-1]["sha256"],
                  "sample_count": len(q), "above_threshold": above,
                  "statistics": self._statistics(result)}
        sample = self._save_sample({"positions": q, "predicted_energy": result.mean.energy,
                                   "predicted_forces": result.mean.forces, "energy_std": result.energy_std,
                                   "max_force_std": result.max_force_std,
                                   "selected": np.asarray(selected, dtype=np.int64), "record": np.asarray(_json(record))})
        meta["rng"] = rng.bit_generator.state
        if len(selected):
            meta["phase"] = "label"
            meta["pending"] = {"sample": sample, "labelled": 0, "dataset_before": meta["dataset"]}
        else:
            reason = ("uncertainty_below_threshold" if record["above_threshold"] == 0 else
                      "label_budget_exhausted" if allowance == 0 else "no_new_candidates")
            meta["history"].append({**record, "sample": sample, "selected": 0,
                                    "successful": 0, "failed": 0, "dataset": meta["dataset"], "trained": False})
            meta["round"] += 1
            meta["phase"], meta["reason"] = "done", reason
        self._commit(meta, state, replace_state=True)

    def _label(self):
        cfg, meta = self._meta["configuration"], deepcopy(self._meta)
        pending = meta["pending"]
        sample = self._load_sample(pending["sample"])
        start = pending["labelled"]
        selected = sample["selected"][start:start + cfg["label_batch_size"]]
        provenance = [{"round": meta["round"], "candidate_index": int(index),
                       "sample_sha256": pending["sample"], "committee_sha256": meta["versions"][-1]["sha256"]}
                      for index in selected]
        labels = label_candidates(self.reference, sample["positions"][selected],
                                  reference_identity=cfg["reference"], batch_size=cfg["label_batch_size"],
                                  provenance=provenance)
        dataset = self.dataset
        dataset = labels if dataset is None else dataset.append(labels)
        self._save_dataset(dataset)
        meta["dataset"] = dataset.sha256
        meta["datasets"].append(dataset.sha256)
        meta["label_attempts"] += len(labels)
        pending["labelled"] += len(labels)
        if pending["labelled"] == len(sample["selected"]):
            meta["phase"] = "train"
        self._commit(meta)

    def _train(self):
        cfg, meta = self._meta["configuration"], deepcopy(self._meta)
        pending = meta["pending"]
        sample = self._load_sample(pending["sample"])
        record = json.loads(str(sample["record"]))
        dataset = self.dataset
        count = len(sample["selected"])
        good = dataset.successful_mask[-count:]
        event = {**record, "sample": pending["sample"], "selected": count,
                 "successful": int(good.sum()), "failed": int((~good).sum()),
                 "dataset": dataset.sha256, "trained": False}
        if not good.any():
            reason = "reference_failed"
        else:
            old = self._load_models(meta["versions"][-1]["members"])
            trained = []
            try:
                # Retain ownership as members are yielded: if a generator
                # raises while producing a later member, earlier providers
                # still need to be closed before retrying this transaction.
                for model in self.trainer(dataset.successful(), old):
                    trained.append(model)
                if self._encode_models(old)[0] != meta["versions"][-1]["members"]:
                    raise ValueError("trainer mutated previous models; return independent fitted providers")
                hashes, payloads = self._encode_models(trained)
                if any(model is previous for model in trained for previous in old):
                    raise ValueError("trainer must return independent new model instances")
                committee = CommitteePotential(trained)
                result = committee.evaluate_uncertainty(sample["positions"])
                if self._encode_models(trained)[0] != hashes:
                    raise ValueError("trained model evaluation mutated its committed state")
                self._save_models(hashes, payloads)
                version = {"members": hashes, "sha256": self._committee_hash(hashes),
                           "dataset": dataset.sha256, "round": meta["round"]}
                meta["versions"].append(version)
                indices = sample["selected"][good]
                energies, forces = dataset.energy[-count:][good], dataset.forces[-count:][good]
                event.update(trained=True, next_committee=version["sha256"],
                             post_training_statistics=self._statistics(result),
                             error_scope="selected_training_labels_only",
                             selected_energy_rmse_before=float(np.sqrt(np.mean((sample["predicted_energy"][indices] - energies)**2))),
                             selected_force_rmse_before=float(np.sqrt(np.mean((sample["predicted_forces"][indices] - forces)**2))),
                             selected_energy_rmse_after=float(np.sqrt(np.mean((result.mean.energy[indices] - energies)**2))),
                             selected_force_rmse_after=float(np.sqrt(np.mean((result.mean.forces[indices] - forces)**2))))
            finally:
                self._close_models(tuple(old) + tuple(trained))
            reason = ("label_budget_exhausted" if meta["label_attempts"] >= cfg["max_labels"] else
                      "round_budget_exhausted" if meta["round"] + 1 >= cfg["max_rounds"] else None)
        meta["history"].append(event)
        meta["round"] += 1
        meta["pending"] = None
        meta["phase"], meta["reason"] = ("sample", None) if reason is None else ("done", reason)
        self._commit(meta)

    def step(self):
        """Commit one stage/batch; errors leave the previous commit resumable."""
        if self._closed:
            raise RuntimeError("active-learning loop is closed")
        with _exclusive(self.directory):
            saved, _ = self._read_checkpoint()
            if saved["identity"] != self.identity or saved["sequence"] != self._meta["sequence"]:
                raise RuntimeError("active-learning checkpoint changed; reconstruct the loop before continuing")
            if self.phase == "sample":
                self._sample()
            elif self.phase == "label":
                self._label()
            elif self.phase == "train":
                self._train()
        return self.result

    def run(self, max_steps=None):
        """Run to the stopping criterion, or pause after ``max_steps`` commits."""
        if max_steps is not None:
            max_steps = _integer(max_steps, "max_steps", 0)
        count = 0
        while self.phase != "done" and (max_steps is None or count < max_steps):
            self.step()
            count += 1
        return self.result

    def close(self):
        """No borrowed references/callbacks are closed by this workflow."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = ["ActiveLearningLoop", "SamplingBatch"]
