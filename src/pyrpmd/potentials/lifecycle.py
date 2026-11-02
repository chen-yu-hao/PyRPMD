"""Committees, acquisition, immutable reference labels and delta potentials.

Disagreement is an epistemic diagnostic, not a calibrated error bound.  Candidate
files contain model predictions, explicitly not quantum-chemistry labels.
Reference labelling writes separate datasets. No provider is refitted or
replaced during a trajectory.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .base import Potential, PESResult, PotentialError, coordinates, validate_result
from .contraction import SplitPotential, _component
from ..checkpoint import atomic_savez, load_checkpoint
from ..path_integral import _integer, _positive


@dataclass(frozen=True)
class CommitteeResult:
    mean: PESResult
    energy_std: np.ndarray
    force_std: np.ndarray
    max_force_std: np.ndarray
    member_energies: np.ndarray


def _copy_committee(result):
    """Detach every array, including the PES status, from a cached result."""
    mean = result.mean
    return CommitteeResult(PESResult(mean.energy.copy(), mean.forces.copy(),
        mean.status.copy(), mean.messages, None if mean.virial is None else mean.virial.copy()),
        result.energy_std.copy(), result.force_std.copy(), result.max_force_std.copy(),
        result.member_energies.copy())


def _mean_std(values):
    """Population moments without squaring a large dimensional force first."""
    scale = np.max(np.abs(values), axis=0)
    scaled = np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)
    return scaled.mean(axis=0) * scale, scaled.std(axis=0) * scale


def _close_all(providers):
    """Attempt every owned close once, then propagate the first close error."""
    seen, first = set(), None
    for provider in providers:
        if id(provider) in seen:
            continue
        seen.add(id(provider))
        close = getattr(provider, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as error:
                if first is None:
                    first = error
    if first is not None:
        raise first


class CommitteePotential(Potential):
    """Uniform committee mean and population standard deviations in atomic units.

    Force uncertainty per atom is sqrt(sum_xyz Var_members(force)); its maximum
    over atoms is used for candidate selection. Every member must describe the
    same energy zero, atom order, geometry and units. Members are borrowed unless
    ``owns_members`` is explicitly true (the input-file factory sets it).
    """

    def __init__(self, members, *, owns_members=False):
        self.members = tuple(_component(m, "committee member") for m in members)
        if len(self.members) < 2:
            raise ValueError("a committee requires at least two members")
        if len({id(member) for member in self.members}) != len(self.members):
            raise ValueError("committee members must be separate provider instances")
        self.owns_members = bool(owns_members)
        self.metadata = {"provider": "committee", "thread_safety": "context_serial",
            "batch_independent": True, "trajectory_batch_safe": False,
            "units": {"energy": "hartree", "length": "bohr"}, "uncertainty": "committee_population_std"}
        self._last_q, self._last_result = None, None

    def evaluate_uncertainty(self, positions):
        q = coordinates(positions)
        energies, forces, virials = [], [], []
        for member in self.members:
            result = validate_result(member.evaluate(q), q.shape)
            energies.append(result.energy.copy())
            forces.append(result.forces.copy())
            virials.append(None if result.virial is None else result.virial.copy())
        energies, forces = np.asarray(energies), np.asarray(forces)
        virial = _mean_std(np.asarray(virials))[0] if all(x is not None for x in virials) else None
        energy_mean, energy_std = _mean_std(energies)
        force_mean, component_std = _mean_std(forces)
        mean = validate_result(PESResult(energy_mean, force_mean, virial=virial), q.shape)
        force_std = np.hypot(np.hypot(component_std[..., 0], component_std[..., 1]), component_std[..., 2])
        if not np.isfinite(energy_std).all() or not np.isfinite(force_std).all():
            raise ValueError("committee uncertainty is not representable")
        result = CommitteeResult(mean, energy_std, force_std, force_std.max(axis=1), energies)
        self._last_q = q.copy()
        self._last_result = _copy_committee(result)
        return result

    def evaluate(self, positions):
        return self.evaluate_uncertainty(positions).mean

    def uncertainty_at(self, positions):
        """Reuse only an exactly matching result; returned diagnostics own data."""
        q = coordinates(positions)
        if self._last_q is not None and np.array_equal(q, self._last_q):
            return _copy_committee(self._last_result)
        return self.evaluate_uncertainty(q)

    def close(self):
        if self.owns_members:
            self.owns_members = False
            _close_all(self.members)


class DeltaLearningPotential(Potential):
    """V = V_reference + learned_delta, with an explicit RPC fast/slow split."""

    def __init__(self, reference, correction, *, owns_members=False):
        self.reference = _component(reference, "reference")
        self.correction = _component(correction, "learned correction")
        self.owns_members = bool(owns_members)
        self.metadata = {"provider": "delta_learning", "batch_independent": True,
            "trajectory_batch_safe": False, "thread_safety": "context_serial",
            "units": {"energy": "hartree", "length": "bohr"}}

    def evaluate(self, positions):
        q = coordinates(positions)
        low = validate_result(self.reference.evaluate(q), q.shape)
        energy, force = low.energy.copy(), low.forces.copy()
        virial = None if low.virial is None else low.virial.copy()
        delta = validate_result(self.correction.evaluate(q), q.shape)
        energy += delta.energy
        force += delta.forces
        virial = None if virial is None or delta.virial is None else virial + delta.virial
        return validate_result(PESResult(energy, force, virial=virial), q.shape)

    def split(self, nbeads, contracted_beads=None):
        """Reference on P beads; learned additive correction on Pc beads.

        The correction model must already learn target-reference.  A full target
        model would double-count the reference and is not converted implicitly.
        Returned SplitPotential borrows both providers and works with MTSDynamics.
        """
        return SplitPotential(self.reference, self.correction, nbeads, contracted_beads)

    def close(self):
        if self.owns_members:
            self.owns_members = False
            _close_all((self.reference, self.correction))


class ActiveLearningMonitor:
    """Observe accepted states and atomically save high-disagreement candidates.

    Each observed step has one checksummed record, even if no bead crosses a
    threshold.  Replaying the same step is idempotent and must match its geometry.
    Use a separate directory per trajectory/replica. The caller owns its clock.
    """

    def __init__(self, committee, directory, *, force_threshold=None, energy_threshold=None,
                 interval_steps=1, max_frames=1000, model_identity=None):
        if not callable(getattr(committee, "uncertainty_at", None)):
            raise TypeError("active learning requires a CommitteePotential")
        self.committee = committee
        self._configure(directory, force_threshold=force_threshold, energy_threshold=energy_threshold,
                        interval_steps=interval_steps, max_frames=max_frames, model_identity=model_identity)

    def _configure(self, directory, *, force_threshold=None, energy_threshold=None,
                   interval_steps=1, max_frames=1000, model_identity=None):
        if force_threshold is None and energy_threshold is None:
            raise ValueError("at least one uncertainty threshold is required")
        self.directory = Path(directory)
        self.force_threshold = None if force_threshold is None else _positive(force_threshold, "force_threshold")
        self.energy_threshold = None if energy_threshold is None else _positive(energy_threshold, "energy_threshold")
        self.interval_steps = _integer(interval_steps, "interval_steps")
        self.max_frames = _integer(max_frames, "max_frames")
        self.model_identity = str(model_identity or "caller_managed")
        self.collected = 0
        # Count only validated records belonging to this exact model/threshold
        # identity; unrelated NPZ files are never treated as training data.
        self.identity = hashlib.sha256(json.dumps({"model": self.model_identity,
            "force_threshold": self.force_threshold, "energy_threshold": self.energy_threshold,
            "interval_steps": self.interval_steps, "max_frames": self.max_frames}, sort_keys=True).encode()).hexdigest()
        for path in sorted(self.directory.glob("step_*.npz")):
            self.collected += self._read_record(path)
        if self.collected > self.max_frames:
            raise ValueError("active-learning records exceed the configured frame cap")

    @classmethod
    def audit_records(cls, directory, *, expected_identity, through_step, **options):
        """Validate a committed collection prefix without loading any models."""
        reader = cls.__new__(cls)
        reader._configure(directory, **options)
        if reader.identity != expected_identity:
            raise ValueError("active-learning checkpoint identity differs from its configuration")
        through_step = _integer(through_step, "through_step", 0)
        for step in range(reader.interval_steps, through_step + 1, reader.interval_steps):
            if not (reader.directory / f"step_{step:012d}.npz").is_file():
                raise ValueError(f"active-learning collection is missing committed step {step}")
        return reader.collected

    def _read_record(self, path, *, q=None, time=None):
        with load_checkpoint(path, expected_kind="PyRPMD.active-learning.v1", require_integrity=True) as saved:
            required = {"identity", "model_identity", "geometry_shape", "selected_beads", "step", "time",
                        "energy_std", "max_force_std", "positions", "predicted_energy", "predicted_forces",
                        "member_energies", "labels_available", "geometry_sha256"}
            if required - set(saved.files):
                raise ValueError("active-learning record is missing required fields")
            if str(saved["identity"]) != self.identity or str(saved["model_identity"]) != self.model_identity:
                raise ValueError("active-learning record belongs to different model or thresholds")
            shape, selected, step = saved["geometry_shape"], saved["selected_beads"], saved["step"]
            if (shape.shape != (3,) or shape.dtype.kind not in "iu" or np.any(shape <= 0) or shape[-1] != 3
                    or selected.ndim != 1 or selected.dtype.kind not in "iu"
                    or np.any(selected < 0) or np.any(selected >= shape[0])
                    or np.any(selected[1:] <= selected[:-1])
                    or step.shape != () or step.dtype.kind not in "iu" or int(step) < 0
                    or int(step) % self.interval_steps or path.name != f"step_{int(step):012d}.npz"
                    or saved["time"].shape != () or saved["time"].dtype.kind not in "iuf"
                    or not np.isfinite(saved["time"]) or saved["time"] < 0):
                raise ValueError("invalid active-learning record shape, indices or clock")
            expected = {"energy_std": (int(shape[0]),), "max_force_std": (int(shape[0]),),
                        "positions": (len(selected), int(shape[1]), 3),
                        "predicted_energy": (len(selected),), "predicted_forces": (len(selected), int(shape[1]), 3)}
            for key, dimensions in expected.items():
                value = saved[key]
                if value.shape != dimensions or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                    raise ValueError("invalid active-learning numerical record: " + key)
            if np.any(saved["energy_std"] < 0) or np.any(saved["max_force_std"] < 0):
                raise ValueError("active-learning uncertainty cannot be negative")
            members = saved["member_energies"]
            if (members.ndim != 2 or members.shape[0] < 2 or members.shape[1] != len(selected)
                    or members.dtype.kind not in "iuf" or not np.isfinite(members).all()
                    or saved["labels_available"].shape != () or saved["labels_available"].dtype.kind != "b"
                    or bool(saved["labels_available"])):
                raise ValueError("invalid active-learning member predictions or label provenance")
            if q is not None and (not np.array_equal(shape, q.shape)
                    or str(saved["geometry_sha256"]) != self._geometry_digest(q) or float(saved["time"]) != time):
                raise ValueError("replayed active-learning step differs from its saved state")
            return len(selected)

    @staticmethod
    def _geometry_digest(q):
        digest = hashlib.sha256(np.asarray(q.shape, dtype="<i8").tobytes())
        digest.update(q.tobytes())
        return digest.hexdigest()

    def observe(self, positions, *, step, time):
        step = _integer(step, "step", 0)
        if (isinstance(time, (bool, np.bool_)) or not np.isscalar(time) or np.iscomplexobj(time)
                or not np.isfinite(time) or time < 0):
            raise ValueError("time must be nonnegative and finite")
        if step % self.interval_steps:
            return None
        q = coordinates(positions)
        digest = self._geometry_digest(q)
        path = self.directory / f"step_{step:012d}.npz"
        if path.exists():
            count = self._read_record(path, q=q, time=time)
            return {"step": step, "selected": count, "record": str(path)}
        result = self.committee.uncertainty_at(q)
        high = np.zeros(len(q), dtype=bool)
        if self.force_threshold is not None:
            high |= result.max_force_std >= self.force_threshold
        if self.energy_threshold is not None:
            high |= result.energy_std >= self.energy_threshold
        selected = np.flatnonzero(high)[:max(0, self.max_frames-self.collected)]
        atomic_savez(path, kind="PyRPMD.active-learning.v1", identity=self.identity,
            model_identity=self.model_identity, geometry_sha256=digest, geometry_shape=np.asarray(q.shape, dtype=np.int64), step=step, time=time,
            energy_std=result.energy_std, max_force_std=result.max_force_std, selected_beads=selected,
            positions=q[selected], predicted_energy=result.mean.energy[selected],
            predicted_forces=result.mean.forces[selected], member_energies=result.member_energies[:,selected],
            labels_available=False, units=json.dumps({"positions":"bohr","energy":"hartree","forces":"hartree/bohr"}))
        self.collected += len(selected)
        return {"step": step, "selected": len(selected), "record": str(path)}


def _label_json(value, name):
    """Canonical, finite JSON without lossy conversion of dictionary keys."""
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError(f"{name} must have string JSON object keys")
            for child in item.values():
                check(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                check(child)

    check(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be finite JSON data") from error


def _reference_json(identity):
    if isinstance(identity, Mapping):
        identity = dict(identity)
        if not identity:
            raise ValueError("reference_identity must be a nonempty mapping or string")
    elif not isinstance(identity, str) or not identity.strip():
        raise ValueError("reference_identity must be a nonempty mapping or string")
    return _label_json(identity, "reference_identity")


def _label_positions(positions):
    q = np.asarray(positions)
    if (q.dtype.kind not in "fiu" or q.ndim != 3 or q.shape[-1] != 3
            or q.shape[1] < 1 or not np.isfinite(q).all()):
        raise ValueError("label positions must be finite real [batch,atom,3] coordinates")
    with np.errstate(over="ignore"):
        result = np.array(q, dtype="<f8", order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("label positions must be representable as finite float64 coordinates")
    return result


def _frozen_array(value, dtype):
    """A bytes-backed array cannot be made writable with setflags()."""
    array = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _label_provenance(provenance, count):
    if provenance is None:
        return ("{}",) * count
    elif isinstance(provenance, Mapping):
        return (_label_json(dict(provenance), "provenance"),) * count
    else:
        try:
            provenance = list(provenance)
        except TypeError as error:
            raise ValueError("label provenance must be a mapping or one mapping per frame") from error
    if len(provenance) != count or any(not isinstance(row, Mapping) for row in provenance):
        raise ValueError("label provenance must be a mapping or one mapping per frame")
    return tuple(_label_json(dict(row), "provenance") for row in provenance)


@dataclass(frozen=True, init=False)
class LabelledDataset:
    """Immutable reference labels, including failed attempts, in atomic units.

    ``reference_identity`` is a required nonempty JSON mapping or string. The
    caller must include all scientifically relevant reference settings, software
    versions, and hashes of external parameter files; provider metadata alone
    generally does not identify a calculation. ``reference_hash`` hashes that
    declaration, while ``sha256`` also covers every geometry, result, status,
    message and provenance entry. Neither digest depends on archive timestamps.

    ``provenance`` is one JSON mapping shared by all rows, or a sequence of JSON
    mappings with one entry per row (for example source, round and model hash).
    Failed rows retain coordinates/status/messages, with canonical NaN labels.
    Only ``successful()`` is suitable for training. Empty datasets are allowed
    provided their coordinate shape still specifies the atom count.
    """

    positions: np.ndarray
    energy: np.ndarray
    forces: np.ndarray
    status: np.ndarray
    messages: tuple[str, ...]
    _reference_json: str
    _provenance_json: tuple[str, ...]
    reference_hash: str
    sha256: str

    def __init__(self, positions, energy, forces, status=None, messages=None, *,
                 reference_identity, provenance=None):
        q = _label_positions(positions)
        n = len(q)
        energy, forces = np.asarray(energy), np.asarray(forces)
        if (energy.dtype.kind not in "fiu" or energy.shape != (n,)
                or forces.dtype.kind not in "fiu" or forces.shape != q.shape):
            raise ValueError("label energy/forces must be real arrays with shape [B] and [B,N,3]")
        if status is None:
            status = np.zeros(n, dtype=np.int32)
        status = np.asarray(status)
        if (status.dtype.kind not in "iu" or status.shape != (n,)
                or np.any(status < 0) or np.any(status > np.iinfo(np.int32).max)):
            raise ValueError("label status must be nonnegative int32 values with shape [B]")
        if isinstance(messages, str):
            raise ValueError("label messages must contain one string per frame")
        messages = ("",) * n if messages is None else tuple(messages)
        if len(messages) != n or any(not isinstance(message, str) for message in messages):
            raise ValueError("label messages must contain one string per frame")
        good = status == 0
        if (not np.isfinite(energy[good]).all() or not np.isfinite(forces[good]).all()):
            raise ValueError("successful reference labels must be finite")
        with np.errstate(over="ignore"):
            energy, forces = np.array(energy, dtype="<f8", copy=True), np.array(forces, dtype="<f8", copy=True)
        if not np.isfinite(energy[good]).all() or not np.isfinite(forces[good]).all():
            raise ValueError("successful reference labels must be representable as finite float64 values")
        energy[~good], forces[~good] = np.nan, np.nan
        reference_json = _reference_json(reference_identity)
        provenance_json = _label_provenance(provenance, n)
        for name, value, dtype in (("positions", q, "<f8"), ("energy", energy, "<f8"),
                                    ("forces", forces, "<f8"), ("status", status, "<i4")):
            object.__setattr__(self, name, _frozen_array(value, dtype))
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "_reference_json", reference_json)
        object.__setattr__(self, "_provenance_json", provenance_json)
        object.__setattr__(self, "reference_hash", hashlib.sha256(reference_json.encode()).hexdigest())
        digest = hashlib.sha256(b"PyRPMD.labelled-dataset.v1\0bohr,hartree,hartree/bohr\0")

        def add(value):
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)

        add(reference_json.encode())
        for name in ("positions", "energy", "forces", "status"):
            value = getattr(self, name)
            add(name.encode())
            add(_label_json(list(value.shape), "array shape").encode())
            add(value.tobytes())
        add(_label_json(messages, "messages").encode())
        add(("[" + ",".join(provenance_json) + "]").encode())
        object.__setattr__(self, "sha256", digest.hexdigest())

    def __len__(self):
        return len(self.positions)

    @property
    def reference_identity(self):
        """Return a detached copy of the reference-method declaration."""
        return json.loads(self._reference_json)

    @property
    def provenance(self):
        """Return detached per-frame provenance mappings."""
        return tuple(json.loads(row) for row in self._provenance_json)

    @property
    def successful_mask(self):
        return self.status == 0

    def successful(self):
        """Copy the successful subset; never expose failures as training labels."""
        mask = self.successful_mask
        return type(self)(self.positions[mask], self.energy[mask], self.forces[mask], self.status[mask],
                          tuple(message for message, keep in zip(self.messages, mask) if keep),
                          reference_identity=self.reference_identity,
                          provenance=[row for row, keep in zip(self.provenance, mask) if keep])

    def append(self, other):
        """Return a new ordered dataset for the same reference and atom count."""
        if not isinstance(other, LabelledDataset):
            raise TypeError("can only append another LabelledDataset")
        if self._reference_json != other._reference_json:
            raise ValueError("cannot append labels with a different reference identity")
        if self.positions.shape[1:] != other.positions.shape[1:]:
            raise ValueError("cannot append labels with a different atom count")
        return type(self)(np.concatenate((self.positions, other.positions)),
                          np.concatenate((self.energy, other.energy)),
                          np.concatenate((self.forces, other.forces)),
                          np.concatenate((self.status, other.status)), self.messages + other.messages,
                          reference_identity=self.reference_identity,
                          provenance=self.provenance + other.provenance)

    def save(self, path):
        """Atomically commit the complete dataset with container/content hashes."""
        atomic_savez(path, kind="PyRPMD.labelled-dataset.v1", positions=self.positions,
                     energy=self.energy, forces=self.forces, status=self.status,
                     messages=np.asarray(self.messages, dtype=str),
                     reference_identity=self._reference_json, reference_hash=self.reference_hash,
                     provenance=np.asarray(self._provenance_json, dtype=str), sha256=self.sha256,
                     units='{"energy":"hartree","forces":"hartree/bohr","positions":"bohr"}')

    @classmethod
    def load(cls, path):
        """Verify both the NPZ integrity envelope and semantic dataset digest."""
        with load_checkpoint(path, expected_kind="PyRPMD.labelled-dataset.v1", require_integrity=True) as data:
            required = {"positions", "energy", "forces", "status", "messages", "reference_identity",
                        "reference_hash", "provenance", "sha256", "units"}
            if set(data.files) != required:
                raise ValueError("labelled dataset has missing or unknown fields")
            for name in ("reference_identity", "reference_hash", "sha256", "units"):
                value = data[name]
                if value.shape != () or value.dtype.kind != "U":
                    raise ValueError("invalid labelled dataset scalar: " + name)
            messages, provenance = data["messages"], data["provenance"]
            if messages.ndim != 1 or provenance.ndim != 1 or messages.dtype.kind != "U" or provenance.dtype.kind != "U":
                raise ValueError("invalid labelled dataset messages or provenance")
            if json.loads(str(data["units"])) != {"energy": "hartree", "forces": "hartree/bohr", "positions": "bohr"}:
                raise ValueError("labelled dataset units differ from atomic units")
            energy, forces = data["energy"], data["forces"]
            result = cls(data["positions"], energy, forces, data["status"], messages.tolist(),
                         reference_identity=json.loads(str(data["reference_identity"])),
                         provenance=[json.loads(row) for row in provenance])
            failed = ~result.successful_mask
            if not np.isnan(energy[failed]).all() or not np.isnan(forces[failed]).all():
                raise ValueError("failed reference labels must be canonical NaNs")
            if str(data["reference_hash"]) != result.reference_hash or str(data["sha256"]) != result.sha256:
                raise ValueError("labelled dataset content or reference hash differs")
            return result


def label_candidates(reference, positions, *, reference_identity, batch_size=64, provenance=None):
    """Evaluate reference labels in bounded batches, retaining every attempt.

    Existing PySCF, tblite and i-PI providers satisfy the same ``evaluate`` PES
    contract. The method declaration is required; it is never inferred from an
    object repr. A provider exception records its entire submitted batch as
    failed (status 1), without retrying single frames or changing method/device.
    Returned per-frame nonzero statuses are preserved. Malformed returned data
    still raise, and process interruptions are not swallowed.

    Hidden electronic warm-start state and coupled ring potentials cannot be
    reproduced by this stateless labelling API and are explicitly rejected.
    An i-PI reference additionally requires a mapping identity containing
    ``independent_frames=True``, a nonempty ``driver_identity`` mapping/string,
    and ``driver_sha256`` (64 hexadecimal digits). This is the caller's audited
    promise that the external driver has no hidden scientific state and will
    reconnect with the same method and parameters; an endpoint is not a method
    identity. It only permits i-PI's scheduling-affinity flag, never an explicit
    stateful or coupled-frame declaration.
    For interruption-safe progress, call this function for one batch at a time
    and atomically save each returned dataset before submitting the next batch.
    Providers are borrowed and remain owned by the caller.
    """
    if not callable(getattr(reference, "evaluate", None)):
        raise TypeError("reference must implement evaluate(positions)")
    metadata = getattr(reference, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("reference metadata must be a mapping")
    capabilities = metadata.get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        raise ValueError("reference capabilities must be a mapping")
    if (metadata.get("batch_independent") is False or capabilities.get("stateful") is True
            or getattr(reference, "warm_start", False)
            or any(metadata.get(key) is True for key in
                   ("stateful", "warm_start", "ring_polymer_coupled", "coupled_frames"))):
        raise ValueError("reference labelling requires independent frames without hidden warm-start state")
    if metadata.get("requires_bead_affinity") is True:
        audited_ipi = metadata.get("provider") == "ipi" and isinstance(reference_identity, Mapping)
        if audited_ipi:
            driver_hash = reference_identity.get("driver_sha256")
            audited_ipi = (reference_identity.get("independent_frames") is True
                           and isinstance(driver_hash, str) and len(driver_hash) == 64
                           and all(character in "0123456789abcdef" for character in driver_hash))
            if audited_ipi:
                try:
                    _reference_json(reference_identity.get("driver_identity"))
                except ValueError:
                    audited_ipi = False
        if not audited_ipi:
            raise ValueError("reference labelling requires independent frames; i-PI needs independent_frames=True, "
                             "driver_identity and driver_sha256 in reference_identity")
    q = _label_positions(positions)
    batch_size = _integer(batch_size, "batch_size")
    # Validate all provenance/identity before an expensive reference call.
    reference_json = _reference_json(reference_identity)
    provenance_json = _label_provenance(provenance, len(q))
    energy, forces = np.full(len(q), np.nan), np.full(q.shape, np.nan)
    status, messages = np.zeros(len(q), dtype=np.int32), [""] * len(q)
    for start in range(0, len(q), batch_size):
        stop = min(start + batch_size, len(q))
        batch = q[start:stop].copy()
        try:
            evaluated = reference.evaluate(batch)
        except Exception as error:
            status[start:stop] = 1
            messages[start:stop] = [f"{type(error).__name__}: {error}"] * (stop - start)
            continue
        result = validate_result(evaluated, batch.shape, require_success=False)
        # PESResult is mutable; a provider can invalidate its constructor's
        # status guarantee. Never truncate a mutated float/overflowed status
        # into a successful int32 while collecting the labels.
        if (result.status.dtype.kind not in "iu" or np.any(result.status < 0)
                or np.any(result.status > np.iinfo(np.int32).max)):
            raise PotentialError("reference result status must be nonnegative int32 values")
        if any(not isinstance(message, str) for message in result.messages):
            raise PotentialError("reference result messages must be strings")
        energy[start:stop], forces[start:stop] = result.energy, result.forces
        status[start:stop], messages[start:stop] = result.status, result.messages
    return LabelledDataset(q, energy, forces, status, messages,
                           reference_identity=json.loads(reference_json),
                           provenance=[json.loads(row) for row in provenance_json])


__all__ = ["CommitteePotential", "CommitteeResult", "DeltaLearningPotential", "ActiveLearningMonitor",
           "LabelledDataset", "label_candidates"]


def active_learning_options(config):
    """Validate input controls without constructing models or writing records."""
    if "active_learning" not in config:
        return None
    from ..units import parse_quantity
    table = config["active_learning"]
    allowed = {"interval_steps", "force_threshold", "energy_threshold", "directory", "max_frames"}
    if not isinstance(table, dict) or set(table) - allowed:
        raise ValueError("unknown active_learning fields or non-table options")
    if (config.get("task") not in ("pimd", "diffusion") or "barostat" in config
            or "path_integral" in config or "plumed" in config
            or config.get("sampling", {}).get("engine", "legacy") != "legacy"):
        raise ValueError("active learning currently requires fixed-cell ordinary PIMD/diffusion")
    potential = config.get("potential", {})
    if potential.get("provider", potential.get("type")) != "committee":
        raise ValueError("active learning requires potential.provider='committee'")
    result = {key: _positive(parse_quantity(table[key], dimension), key)
              for key, dimension in (("force_threshold", "force"), ("energy_threshold", "energy"))
              if key in table}
    if not result:
        raise ValueError("active learning requires a force_threshold or energy_threshold")
    result["interval_steps"] = _integer(table.get("interval_steps", 10), "active_learning.interval_steps")
    result["max_frames"] = _integer(table.get("max_frames", 1000), "active_learning.max_frames")
    if "directory" in table and (not isinstance(table["directory"], str) or not table["directory"]):
        raise ValueError("active_learning.directory must be a nonempty path")
    return result
