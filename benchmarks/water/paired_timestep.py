#!/usr/bin/env python3
"""Preregistered all-eight paired time-step control; no paper-agreement test.

Both fresh native contexts start from one saved pending segment0 Maxwell draw.
A complete original cohort and observed normal coordinator exit are required
before live registration. The execution path is sequential and uses one CPU
force thread; no long control is launched by importing or testing this module.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water import native_preparation_coordinator as preparation
from benchmarks.water import native_preparation_extension as extension
from benchmarks.water import native_reference as native
from benchmarks.water.campaign import atomic_json, atomic_npz, sha256
from pyrpmd.analysis.diffusion import analyze_diffusion
from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU, KCAL_MOL_TO_AU

PROTOCOL_SHA256 = "d7a34a9f92604b38ef0a4ab2bc6271b20bd14b09e92b95b4312bd2efe0543747"
COORDINATOR_SHA256 = "f51c63e1ba8edbca91ef6247bcd9e10fd9eb281c3e1cec986835e9d961d1bd02"
REGISTRATION_SCHEMA = "pyrpmd.water-paired-timestep-registration.v1"
BRANCH_SCHEMA = "pyrpmd.water-paired-timestep-branch.v1"
SEEDS = tuple(range(1101, 1109))


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity():
    identity = preparation.coordinator_identity()
    name = "benchmarks/water/native_preparation_coordinator.py"
    if identity[name] != COORDINATOR_SHA256:
        raise ValueError("the reviewed original-pool handoff helpers changed")
    return {**identity, "benchmarks/water/paired_timestep.py": sha256(Path(__file__)),
            "src/pyrpmd/analysis/diffusion.py": sha256(ROOT / "src/pyrpmd/analysis/diffusion.py")}


@dataclass(frozen=True)
class Branch:
    label: str
    dt_fs: float
    duration_fs: float = 4000.0
    record_fs: float = 5.0
    checkpoint_fs: float = 1000.0

    def validate(self):
        if not isinstance(self.label, str) or not self.label or Path(self.label).name != self.label:
            raise ValueError("branch label must be a nonempty basename")
        for name in ("dt_fs", "duration_fs", "record_fs", "checkpoint_fs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("branch times must be positive finite numbers")
            ratio = value / self.dt_fs
            if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
                raise ValueError("all branch times must be exact integer multiples of dt")
        if self.steps % self.record_stride or self.checkpoint_steps % self.record_stride:
            raise ValueError("duration and checkpoints must preserve the common recording grid")
        return self

    @property
    def steps(self):
        return round(self.duration_fs / self.dt_fs)

    @property
    def record_stride(self):
        return round(self.record_fs / self.dt_fs)

    @property
    def checkpoint_steps(self):
        return round(self.checkpoint_fs / self.dt_fs)


BRANCHES = (Branch("dt025", .25), Branch("dt0125", .125))


def _protocol(path):
    path = Path(path).resolve()
    protocol = read_json(path)
    if (sha256(path) != PROTOCOL_SHA256
            or protocol.get("schema") != "pyrpmd.water-eight-chain-paired-timestep-protocol.v1"
            or protocol["fixed_seeds"] != list(SEEDS)
            or protocol["physics"]["original_frozen_source_sha256"] != native.source_identity()):
        raise ValueError("paired time-step preregistration or original physical source changed")
    return protocol


def register(original_root, output, protocol_path, assessment_path, exit_proof_path):
    """Select all eight inputs only after the complete original experiment."""
    original_root, output = Path(original_root).resolve(), Path(output).resolve()
    protocol_path, assessment_path, exit_proof_path = map(lambda p: Path(p).resolve(),
                                                        (protocol_path, assessment_path, exit_proof_path))
    identity = source_identity()
    protocol = _protocol(protocol_path)
    assessment = read_json(assessment_path)
    rows = assessment.get("chains", [])
    if (assessment.get("schema") != "pyrpmd.water-native-chain-analysis.v1"
            or assessment.get("all_predeclared_chains_complete") is not True
            or assessment.get("protocol_sha256") != protocol["original_protocol_sha256"]
            or len(rows) != len(SEEDS) or sorted(row["seed"] for row in rows) != list(SEEDS)
            or any(row.get("complete") is not True for row in rows)):
        raise ValueError("the full preserved original cohort assessment is required")
    assessed = {row["seed"]: row["manifest_sha256"] for row in rows}
    assessment_hash = sha256(assessment_path)
    exit_reference = {"path": str(exit_proof_path), "sha256": sha256(exit_proof_path)}
    if output == original_root or original_root in output.parents or output in original_root.parents:
        raise ValueError("control data must have a separate output root")
    selection = {}
    with preparation.original_exclusion_leases(original_root):
        preparation.validate_original_exit(original_root, {"original_coordinator_exit_proof": exit_reference})
        for seed in SEEDS:
            directory, raw, manifest, base, chosen = extension._original(
                str(original_root / "runs" / f"classical-{seed}"))
            if (asdict(base) != asdict(native.Config(seed=seed, initial_checkpoint=base.initial_checkpoint))
                    or base.equilibration_steps != 400000 or manifest.get("checkpoint_contract_version") != 2
                    or hashlib.sha256(raw).hexdigest() != assessed[seed]):
                raise ValueError("original configuration/checkpoint differs from the fixed assessed experiment")
            preparation._checkpoint_phase(directory / chosen["file"], expected_step=400000)
            selection[str(seed)] = {"directory": str(directory), "manifest_sha256": assessed[seed],
                                    "checkpoint": chosen, "config": asdict(base), "artifacts": manifest["artifacts"],
                                    "box_angstrom": manifest["box_angstrom"]}
        if source_identity() != identity or sha256(assessment_path) != assessment_hash:
            raise RuntimeError("source or original assessment changed during registration")
        with preparation.exclusive_lease(output.with_name(output.name + "-registration.lock")):
            if output.exists() and any(output.iterdir()):
                raise ValueError("control registration output must be new/empty")
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(output / "registration.json", {
                "schema": REGISTRATION_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "output_root": str(output), "original_root": str(original_root),
                "protocol_path": str(protocol_path), "protocol_sha256": PROTOCOL_SHA256,
                "assessment": {"path": str(assessment_path), "sha256": assessment_hash},
                "original_coordinator_exit_proof": exit_reference,
                "source_sha256": identity, "environment": native.environment_identity(),
                "selection": selection, "fixed_seeds": list(SEEDS),
                "branches": [asdict(branch) for branch in BRANCHES],
                "only_numerical_control": True, "preparation_failure_repaired": False,
                "literature_comparison": None,
            })
    return output / "registration.json"


def load_registration(path):
    path = Path(path).resolve()
    registration = read_json(path)
    output = Path(registration["output_root"]).resolve()
    if (registration.get("schema") != REGISTRATION_SCHEMA or path.parent != output
            or registration.get("source_sha256") != source_identity()
            or registration.get("environment") != native.environment_identity()
            or registration.get("fixed_seeds") != list(SEEDS)
            or registration.get("branches") != [asdict(branch) for branch in BRANCHES]
            or registration.get("only_numerical_control") is not True
            or registration.get("preparation_failure_repaired") is not False
            or registration.get("literature_comparison") is not None
            or set(registration["selection"]) != {str(seed) for seed in SEEDS}):
        raise ValueError("time-step control registration/source/phase changed")
    _protocol(registration["protocol_path"])
    if (registration["protocol_sha256"] != PROTOCOL_SHA256
            or sha256(registration["assessment"]["path"]) != registration["assessment"]["sha256"]):
        raise ValueError("registered protocol or complete original assessment changed")
    preparation.validate_original_exit(registration["original_root"],
                                       {"original_coordinator_exit_proof": registration["original_coordinator_exit_proof"]})
    for seed in SEEDS:
        row = registration["selection"][str(seed)]
        directory = Path(row["directory"]).resolve()
        expected = Path(registration["original_root"]).resolve() / "runs" / f"classical-{seed}"
        if (directory != expected or sha256(directory / "manifest.json") != row["manifest_sha256"]
                or Path(row["checkpoint"]["file"]).name != row["checkpoint"]["file"]
                or row["checkpoint"]["step"] != 400000
                or sha256(directory / row["checkpoint"]["file"]) != row["checkpoint"]["sha256"]):
            raise ValueError("exact original input checkpoint changed")
        manifest = read_json(directory / "manifest.json")
        base = native.Config(**row["config"]).validate()
        if (manifest["config"] != asdict(base) or base.seed != seed or not manifest["complete"]
                or manifest["source_sha256"] != native.source_identity()
                or manifest["environment"] != native.environment_identity()
                or row["checkpoint"] not in manifest["chunks"] or manifest["artifacts"] != row["artifacts"]
                or manifest["box_angstrom"] != row["box_angstrom"]):
            raise ValueError("original bound model/seed/source identity changed")
        for name, artifact in row["artifacts"].items():
            if Path(name).name != name or sha256(directory / name) != artifact["sha256"]:
                raise ValueError("original model or historical artifact changed")
        preparation._checkpoint_phase(directory / row["checkpoint"]["file"], expected_step=400000)
    return {"path": path, "sha256": sha256(path), "registration": registration, "output": output}


def prepare_initial(output, row):
    """Restore one pre-NVE checkpoint and save exactly one COM Maxwell draw."""
    import openmm as mm
    from openmm import unit
    output = Path(output)
    manifest_path = output / "initial.json"
    identity = source_identity()
    expected = {"source_sha256": identity, "original_manifest_sha256": row["manifest_sha256"],
                "original_checkpoint_sha256": row["checkpoint"]["sha256"], "seed": row["config"]["seed"]}
    if manifest_path.exists():
        result = read_json(manifest_path)
        if (any(result.get(key) != value for key, value in expected.items())
                or sha256(output / "initial.npz") != result["initial_sha256"]):
            raise ValueError("saved shared initial state changed")
        return result
    if output.exists() and any(output.iterdir()):
        raise ValueError("uncommitted initial-state files require review; never redraw/replace silently")
    directory = Path(row["directory"])
    if (sha256(directory / "manifest.json") != row["manifest_sha256"]
            or sha256(directory / row["checkpoint"]["file"]) != row["checkpoint"]["sha256"]):
        raise ValueError("original checkpoint changed before its one pending draw")
    base = native.Config(**row["config"]).validate()
    with np.load(directory / row["checkpoint"]["file"], allow_pickle=False) as data:
        checkpoint = {name: data[name].copy() for name in data.files}
    if int(checkpoint["state_step"]) != base.equilibration_steps or int(checkpoint["active_segment"]) != -1:
        raise ValueError("source state must precede the segment0 momentum draw")
    system, integrator, context, masses = native.create_engine(base, row["box_angstrom"])
    try:
        for name, model in (("model.xml", system), ("integrator.xml", integrator)):
            if hashlib.sha256(mm.XmlSerializer.serialize(model).encode()).hexdigest() != row["artifacts"][name]["sha256"]:
                raise ValueError("fresh frozen factory differs from the original model/integrator")
        context.loadCheckpoint(checkpoint["openmm_checkpoint"].tobytes())
        source_box = np.asarray(context.getState().getPeriodicBoxVectors(asNumpy=True) / unit.nanometer)
        if not np.allclose(source_box, np.eye(3)*row["box_angstrom"]/10, rtol=0, atol=1e-12):
            raise ValueError("restored source checkpoint periodic cell differs from its registered model")
        checked = native.validate_checkpoint_state(context, masses, base, checkpoint, allow_legacy=False)
        q, _, _ = native.get_state(context, masses)
        rng = np.random.default_rng()
        rng.bit_generator.state = json.loads(str(checkpoint["rng_json"]))
        before = json.dumps(rng.bit_generator.state, sort_keys=True)
        p = native.sample_momenta(rng, masses, base.temperature_K, remove_com=True)
        after = json.dumps(rng.bit_generator.state, sort_keys=True)
        output.mkdir(parents=True, exist_ok=True)
        atomic_npz(output / "initial.npz", q=q, p=p, masses_au=masses,
                   box_angstrom=row["box_angstrom"], rng_before_json=before, rng_after_json=after)
        if source_identity() != identity:
            raise RuntimeError("source changed before shared initial state was committed")
        result = {**expected, "schema": "pyrpmd.water-paired-timestep-shared-initial.v1",
                  "initial_sha256": sha256(output / "initial.npz"), "source_restore_validation": checked,
                  "pending_COM_Maxwell_draw_count": 1, "both_branches_use_this_same_state": True,
                  "total_momentum_max_abs_au": float(np.max(abs(p.sum(axis=1))))}
        atomic_json(manifest_path, result)
        return result
    finally:
        del context, integrator


def _branch_config(base, branch):
    branch.validate()
    return replace(base, dt_fs=branch.dt_fs, equilibration_steps=0, segment_steps=branch.steps, segments=1,
                   record_stride=branch.record_stride, checkpoint_steps=branch.checkpoint_steps,
                   # Common double-precision observations isolate dt from storage quantization.
                   trajectory_precision="float64", initial_checkpoint=None).validate()


def run_branch(output, initial_directory, row, branch, *, wall_seconds=None, max_steps=None):
    """One unthermostatted trajectory, with redundant state and no further RNG."""
    import openmm as mm
    from openmm import unit
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("positive finite wall budget required")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    output, initial_directory = Path(output), Path(initial_directory)
    identity = source_identity()
    initial = read_json(initial_directory / "initial.json")
    if (initial["source_sha256"] != identity
            or initial["original_checkpoint_sha256"] != row["checkpoint"]["sha256"]
            or initial["original_manifest_sha256"] != row["manifest_sha256"]
            or initial["pending_COM_Maxwell_draw_count"] != 1
            or initial["initial_sha256"] != sha256(initial_directory / "initial.npz")):
        raise ValueError("branch must use the verified shared one-draw initial state")
    base = native.Config(**row["config"]).validate()
    config = _branch_config(base, branch)
    expected = {"source_sha256": identity, "environment": native.environment_identity(),
                "branch": asdict(branch), "native_config": asdict(config),
                "initial_sha256": initial["initial_sha256"], "thermostat": "none",
                "additional_momentum_draws": 0}
    manifest_path = output / "manifest.json"
    resume = manifest_path.exists()
    if resume:
        manifest = read_json(manifest_path)
        if manifest.get("schema") != BRANCH_SCHEMA or any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("branch source/configuration/initial-state identity changed")
        native.verify_manifest(output, manifest)
        steps = [entry["step"] for entry in manifest["chunks"]]
        if (not steps or steps[0] != 0 or steps[-1] != manifest["committed_steps"]
                or any(right <= left for left, right in zip(steps, steps[1:]))
                or steps[-1] > branch.steps or manifest["complete"] is not (steps[-1] == branch.steps)):
            raise ValueError("branch committed checkpoint sequence is inconsistent")
        if manifest["complete"]:
            return manifest
        step = manifest["committed_steps"]
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("uncommitted branch output requires explicit review")
        manifest = {"schema": BRANCH_SCHEMA, **expected, "committed_steps": 0, "complete": False,
                    "thermostat": "none", "additional_momentum_draws": 0,
                    "chunks": [], "artifacts": {}, "restart_validations": [], "elapsed_seconds": 0.0,
                    "created_utc": datetime.now(timezone.utc).isoformat()}
        step = 0
    with np.load(initial_directory / "initial.npz", allow_pickle=False) as data:
        q0, p0 = data["q"].copy(), data["p"].copy()
        initial_masses = data["masses_au"].copy()
        box = float(data["box_angstrom"])
    system, integrator, context, masses = native.create_engine(config, box)
    started, invocation_step = time.perf_counter(), step
    try:
        if not np.array_equal(masses, initial_masses):
            raise ValueError("paired branches must preserve the exact physical masses")
        model_xml = mm.XmlSerializer.serialize(system)
        integrator_xml = mm.XmlSerializer.serialize(integrator)
        if hashlib.sha256(model_xml.encode()).hexdigest() != row["artifacts"]["model.xml"]["sha256"]:
            raise ValueError("control changed the physical force model")
        if resume:
            if (manifest["artifacts"]["model.xml"]["sha256"] != hashlib.sha256(model_xml.encode()).hexdigest()
                    or manifest["artifacts"]["integrator.xml"]["sha256"] != hashlib.sha256(integrator_xml.encode()).hexdigest()):
                raise ValueError("resumed model/integrator differs from this time-step branch")
            with np.load(output / manifest["chunks"][-1]["file"], allow_pickle=False) as data:
                checkpoint = {name: data[name].copy() for name in data.files}
            if int(checkpoint["state_step"]) != step or int(checkpoint["active_segment"]) != 0:
                raise ValueError("paired control checkpoint step/phase differs")
            context.loadCheckpoint(checkpoint["openmm_checkpoint"].tobytes())
            checked = native.validate_checkpoint_state(context, masses, config, checkpoint, allow_legacy=False)
            if context.getStepCount() != step:
                raise ValueError("native Context step count differs from committed control time")
            manifest["restart_validations"].append({"step": step, **checked})
        else:
            native.set_state(context, q0, p0, masses)
            actual_q, actual_p, _ = native.get_state(context, masses)
            if (not np.allclose(actual_q, q0, rtol=0, atol=2e-12)
                    or not np.allclose(actual_p, p0, rtol=0, atol=2e-13)):
                raise ValueError("fresh context did not preserve the shared canonical initial q,p")
            manifest["initial_q_max_error_bohr"] = float(np.max(abs(actual_q-q0)))
            manifest["initial_p_max_error_au"] = float(np.max(abs(actual_p-p0)))
            output.mkdir(parents=True, exist_ok=True)
            for name, text in (("model.xml", model_xml), ("integrator.xml", integrator_xml)):
                (output / name).write_text(text)
                manifest["artifacts"][name] = {"sha256": sha256(output / name)}
        actual_box = np.asarray(context.getState().getPeriodicBoxVectors(asNumpy=True) / unit.nanometer)
        if not np.allclose(actual_box, np.eye(3)*box/10, rtol=0, atol=1e-12):
            raise ValueError("control periodic cell differs from the shared source")
        last_commit = time.perf_counter()

        def commit(observations):
            nonlocal last_commit
            q, p, _, raw = native.get_state(context, masses, raw=True)
            if (not np.isfinite(q).all() or not np.isfinite(p).all()
                    or any(not np.isfinite(value).all() for value in raw.values())):
                raise ValueError("nonfinite control state; no invalid checkpoint is committed")
            arrays = {key: np.asarray([item[key] for item in observations]) for key in observations[0]} if observations else {}
            if any(not np.isfinite(values).all() for values in arrays.values()):
                raise ValueError("nonfinite control observation; no replacement branch")
            filename = f"chunk-{len(manifest['chunks']):06d}.npz"
            atomic_npz(output / filename, **arrays, q=q, p=p, state_step=step, active_segment=0,
                       box_angstrom=box, **raw,
                       openmm_checkpoint=np.frombuffer(context.createCheckpoint(), dtype=np.uint8))
            if (source_identity() != identity or sha256(initial_directory / "initial.npz") != initial["initial_sha256"]):
                raise RuntimeError("source/shared initial state changed; new chunk not committed")
            manifest["chunks"].append({"file": filename, "sha256": sha256(output / filename),
                                       "step": step, "observations": len(observations)})
            manifest["committed_steps"] = step
            manifest["complete"] = step == branch.steps
            manifest["elapsed_seconds"] += time.perf_counter()-last_commit
            atomic_json(manifest_path, manifest)
            last_commit = time.perf_counter()

        if not resume:
            commit([native.observe(context, masses, config, 0, 0, 0)])
        while step < branch.steps:
            stop = min(branch.steps, step+branch.checkpoint_steps)
            if max_steps is not None:
                stop = min(stop, invocation_step+max_steps)
            observations = []
            while step < stop:
                advance = min(stop-step, branch.record_stride-step % branch.record_stride)
                integrator.step(advance)
                step += advance
                if step % branch.record_stride == 0:
                    observations.append(native.observe(context, masses, config, step, 0, step))
            commit(observations)
            if max_steps is not None and step-invocation_step >= max_steps:
                break
            if wall_seconds is not None and time.perf_counter()-started >= wall_seconds:
                break
        return manifest
    finally:
        del context, integrator


def run_control(registration_path, *, wall_seconds=1800):
    """Sequential fixed seed/branch order; partial runs expose no transport test."""
    if not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 7200:
        raise ValueError("wall budget must lie in (0,7200] seconds")
    identity = load_registration(registration_path)
    output = identity["output"]
    failure = output / "FAILURE.json"
    if failure.exists():
        raise RuntimeError("a retained control failure needs review; no automatic replacement")
    started = time.perf_counter()
    seed, label = None, None
    with ExitStack() as stack:
        stack.enter_context(preparation.original_exclusion_leases(identity["registration"]["original_root"]))
        stack.enter_context(preparation.exclusive_lease(output / "control.lock"))
        if failure.exists():
            raise RuntimeError("a retained control failure needs review")
        try:
            for seed in SEEDS:
                if time.perf_counter()-started >= wall_seconds:
                    return {"complete": False, "transport_comparison": None}
                refreshed = load_registration(identity["path"])
                if refreshed["sha256"] != identity["sha256"]:
                    raise ValueError("control registration changed during execution")
                row = identity["registration"]["selection"][str(seed)]
                initial_directory = output / "runs" / f"seed-{seed}" / "initial"
                prepare_initial(initial_directory, row)
                for branch in BRANCHES:
                    label = branch.label
                    remaining = wall_seconds-(time.perf_counter()-started)
                    if remaining <= 0:
                        return {"complete": False, "transport_comparison": None}
                    result = run_branch(output / "runs" / f"seed-{seed}" / label,
                                        initial_directory, row, branch, wall_seconds=remaining)
                    if not result["complete"]:
                        return {"complete": False, "transport_comparison": None}
            return {"complete": True, "transport_comparison": "requires separate all-eight analysis"}
        except BaseException as error:
            if not failure.exists():
                atomic_json(failure, {"created_utc": datetime.now(timezone.utc).isoformat(),
                                     "seed": seed, "branch": label, "registration_sha256": identity["sha256"],
                                     "error_type": type(error).__name__, "message": str(error),
                                     "all_committed_data_retained": True, "no_seed_replacement": True})
            raise


def paired_statistics(coarse, fine, *, margin=.01):
    coarse, fine = np.asarray(coarse, dtype=float), np.asarray(fine, dtype=float)
    if coarse.shape != (8,) or fine.shape != (8,) or not np.isfinite(coarse).all() or not np.isfinite(fine).all():
        raise ValueError("all eight finite paired chain estimates are required")
    if isinstance(margin, bool) or not math.isfinite(margin) or margin <= 0:
        raise ValueError("equivalence margin must be positive and finite")
    delta = fine-coarse
    mean = float(delta.mean())
    standard_error = float(delta.std(ddof=1)/np.sqrt(8))
    halfwidth = float(student_t.ppf(.975, 7)*standard_error)
    interval = [mean-halfwidth, mean+halfwidth]
    equivalent = interval[0] >= -margin and interval[1] <= margin
    status = "equivalent_within_predeclared_margin" if equivalent else (
        "difference_exceeds_margin" if interval[0] > margin or interval[1] < -margin else "equivalence_unresolved")
    return {"chain_differences": delta.tolist(), "mean": mean, "standard_error": standard_error,
            "degrees_of_freedom": 7, "CI95": interval, "CI95_halfwidth": halfwidth,
            "equivalence_margin_angstrom2_ps": margin, "equivalent_within_margin": bool(equivalent), "status": status}


def analyze(registration_path, output):
    """Require every original pair; never select chains or compare to the paper."""
    identity = load_registration(registration_path)
    control_root = identity["output"]
    if (control_root / "FAILURE.json").exists():
        raise ValueError("retained control failure prevents an aggregate control claim")
    # Validate completion for all16 before computing even a per-branch D.
    if any(not (control_root / "runs" / f"seed-{seed}" / name).is_file()
           for seed in SEEDS for name in ("initial/initial.json", "dt025/manifest.json", "dt0125/manifest.json")):
        raise ValueError("all eight complete pairs are required before transport analysis")
    loaded = {}
    for seed in SEEDS:
        selected = identity["registration"]["selection"][str(seed)]
        initial_directory = control_root / "runs" / f"seed-{seed}" / "initial"
        initial = read_json(initial_directory / "initial.json")
        if (sha256(initial_directory / "initial.npz") != initial["initial_sha256"]
                or initial.get("source_sha256") != source_identity() or initial.get("seed") != seed
                or initial.get("original_manifest_sha256") != selected["manifest_sha256"]
                or initial.get("original_checkpoint_sha256") != selected["checkpoint"]["sha256"]
                or initial.get("pending_COM_Maxwell_draw_count") != 1
                or initial.get("both_branches_use_this_same_state") is not True):
            raise ValueError("shared initial state or its exact original seed identity changed")
        for branch in BRANCHES:
            directory = control_root / "runs" / f"seed-{seed}" / branch.label
            manifest = read_json(directory / "manifest.json")
            if (manifest.get("schema") != BRANCH_SCHEMA or manifest["complete"] is not True
                    or manifest["committed_steps"] != branch.steps or manifest["branch"] != asdict(branch)
                    or manifest["source_sha256"] != source_identity()
                    or manifest["environment"] != native.environment_identity()
                    or manifest["native_config"] != asdict(_branch_config(native.Config(**selected["config"]), branch))
                    or manifest["initial_sha256"] != initial["initial_sha256"]
                    or manifest["artifacts"].get("model.xml") != selected["artifacts"]["model.xml"]
                    or manifest.get("thermostat") != "none" or manifest.get("additional_momentum_draws") != 0):
                raise ValueError("all eight complete, source-bound, unthermostatted pairs are required")
            native.verify_manifest(directory, manifest)
            loaded[(seed, branch.label)] = (directory, manifest)
    rows, curves = [], {}
    for seed in SEEDS:
        row = {"seed": seed, "branches": {}}
        for branch in BRANCHES:
            directory, manifest = loaded[(seed, branch.label)]
            columns = ("step", "molecular_com_bohr", "molecular_velocity_au", "hamiltonian_per_bead_hartree")
            parts = {key: [] for key in columns}
            for chunk in manifest["chunks"]:
                with np.load(directory / chunk["file"], allow_pickle=False) as data:
                    if chunk["observations"]:
                        if not np.all(data["segment"] == 0):
                            raise ValueError("control observations have the wrong phase")
                        for key in columns:
                            parts[key].append(data[key].copy())
            data = {key: np.concatenate(value) for key, value in parts.items()}
            expected = np.arange(0, branch.steps+1, branch.record_stride)
            if not np.array_equal(data["step"], expected) or any(not np.isfinite(value).all() for value in data.values()):
                raise ValueError("missing/duplicate/nonfinite observations on the common 0–4ps grid")
            if (data["molecular_com_bohr"].shape != (801, 216, 3)
                    or data["molecular_velocity_au"].shape != (801, 216, 3)):
                raise ValueError("paired control must retain every molecule on the common grid")
            result = analyze_diffusion(data["molecular_com_bohr"], expected*branch.dt_fs*FS_TO_AU,
                                       velocities=data["molecular_velocity_au"], max_lag=400,
                                       fit_range=(1000*FS_TO_AU, 2000*FS_TO_AU), bootstrap=0)
            energy = data["hamiltonian_per_bead_hartree"]
            energy_delta = (energy-energy[0])/(216*KCAL_MOL_TO_AU)
            summary = {"D_VACF_2ps_angstrom2_ps": float(np.asarray(result["vacf_integral_m2_s"])[-1])*1e8,
                       "D_MSD_1_2ps_angstrom2_ps": float(result["raw_fit"]["diffusion_m2_s"])*1e8,
                       "maximum_abs_energy_change_kcal_mol_water": float(np.max(abs(energy_delta))),
                       "linear_energy_drift_kcal_mol_water_ps": float(np.polyfit(expected*branch.dt_fs/1000, energy_delta, 1)[0]),
                       "individual_MSD_regime_status": result["status"], "frames": len(expected),
                       "manifest_sha256": sha256(directory / "manifest.json")}
            row["branches"][branch.label] = summary
            prefix = f"seed_{seed}_{branch.label}_"
            curves[prefix+"D_VACF_angstrom2_ps"] = np.asarray(result["vacf_integral_m2_s"])*1e8
            curves[prefix+"msd_angstrom2"] = np.asarray(result["msd_au"])*BOHR_ANGSTROM**2
            curves["lag_ps"] = np.asarray(result["lag_time_fs"])/1000
        rows.append(row)
    comparisons = {}
    for key in ("D_VACF_2ps_angstrom2_ps", "D_MSD_1_2ps_angstrom2_ps"):
        comparisons[key] = paired_statistics([row["branches"]["dt025"][key] for row in rows],
                                             [row["branches"]["dt0125"][key] for row in rows])
    energy_pass = all(abs(branch["linear_energy_drift_kcal_mol_water_ps"]) <= .001
                      and branch["maximum_abs_energy_change_kcal_mol_water"] <= .01
                      for row in rows for branch in row["branches"].values())
    report = {"schema": "pyrpmd.water-paired-timestep-analysis.v1", "registration_sha256": identity["sha256"],
              "protocol_sha256": PROTOCOL_SHA256, "chains": rows, "paired_comparisons": comparisons,
              "all_eight_pairs_complete": True, "energy_limits_pass": energy_pass,
              "overall_numerical_control_pass": energy_pass and all(item["equivalent_within_margin"] for item in comparisons.values()),
              "independence_unit": "eight paired chain differences", "physical_converged": False,
              "preparation_gate_failure_repaired": False, "literature_reproduced": False,
              "literature_comparison": None, "external_OpenMM_reference": True,
              "PyRPMD_production_validation": False,
              "interpretation": "Time-step control on the fixed original configurations only. Passing cannot repair failed preparation or establish physical diffusion agreement."}
    if load_registration(identity["path"])["sha256"] != identity["sha256"]:
        raise RuntimeError("registration changed during analysis")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("analysis output must be new/empty")
    output.mkdir(parents=True, exist_ok=True)
    atomic_npz(output / "curves.npz", **curves)
    report["curves_sha256"] = sha256(output / "curves.npz")
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    registration = commands.add_parser("register")
    for option in ("original-root", "output", "protocol", "assessment", "exit-proof"):
        registration.add_argument("--"+option, type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--registration", type=Path, required=True)
    execute.add_argument("--wall-seconds", type=float, default=1800)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--registration", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "register":
        print(register(args.original_root, args.output, args.protocol, args.assessment, args.exit_proof))
    elif args.action == "run":
        print(json.dumps(run_control(args.registration, wall_seconds=args.wall_seconds), indent=2))
    else:
        print(json.dumps(analyze(args.registration, args.output), indent=2))


if __name__ == "__main__":
    main()

