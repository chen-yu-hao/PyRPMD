"""Committee diagnostics, candidate collection and additive delta potentials.

Disagreement is an epistemic diagnostic, not a calibrated error bound.  Candidate
files contain model predictions, explicitly not quantum-chemistry labels.  No
provider is refitted or replaced during a trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .base import Potential, PESResult, coordinates, validate_result
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


__all__ = ["CommitteePotential", "CommitteeResult", "DeltaLearningPotential", "ActiveLearningMonitor"]


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
