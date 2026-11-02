#!/usr/bin/env python3
"""Separate, preparation-only continuation from a fixed native water checkpoint.

The original run must be complete and is only read. A configuration binds the
original manifest and its exact final-preparation checkpoint by SHA256. The
frozen Verlet/model helpers and periodic Maxwell refresh are reused unchanged.
No NVE phase or COM-projected momentum draw exists in this runner. Long-stage
adoption is a separate scientific decision; the current proposal is unlaunched.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water import native_reference as native
from benchmarks.water.campaign import atomic_json, atomic_npz, sha256
from pyrpmd.constants import FS_TO_AU

FROZEN_NATIVE_SHA256 = "92867b1d2ada3fb1ca789cb5a9859024ac68fee326f18a8fb910711bf3915da2"
SCHEMA = "pyrpmd.water-preparation-only-continuation.v1"


@dataclass(frozen=True)
class Config:
    source_run: str
    source_manifest_sha256: str
    source_checkpoint_sha256: str
    additional_preparation_steps: int
    checkpoint_steps: int

    def validate(self):
        if not isinstance(self.source_run, str) or not self.source_run:
            raise ValueError("source_run must identify the original completed run")
        for name in ("source_manifest_sha256", "source_checkpoint_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be an explicit lowercase SHA256")
        for name in ("additional_preparation_steps", "checkpoint_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return self


def source_identity():
    identity = native.source_identity()
    if identity["benchmarks/water/native_reference.py"] != FROZEN_NATIVE_SHA256:
        raise ValueError("continuation requires the explicitly frozen native-reference source")
    return {**identity, "benchmarks/water/native_preparation_extension.py": sha256(Path(__file__))}


def _original(source_run):
    directory = (ROOT / source_run).resolve()
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema") != "pyrpmd.water-openmm-external-reference.v1"
            or manifest.get("external_reference") is not True or manifest.get("complete") is not True):
        raise ValueError("the original fixed run must be complete before a preparation branch is constructed")
    if not {"model.xml", "integrator.xml"}.issubset(manifest.get("artifacts", {})):
        raise ValueError("original serialized model and integrator identities are required")
    if manifest["source_sha256"] != native.source_identity():
        raise ValueError("original source differs from this frozen native-reference checkout")
    if manifest["environment"] != native.environment_identity():
        raise ValueError("original engine/Python environment differs from the continuation environment")
    native.verify_manifest(directory, manifest)
    base = native.Config(**manifest["config"]).validate()
    if manifest["committed_steps"] != base.total_steps:
        raise ValueError("the original completed run has an inconsistent terminal step")
    origin = base.equilibration_steps
    if origin <= 0 or origin % base.sampling_refresh_steps:
        raise ValueError("this continuation branches at the original final-preparation refresh boundary")
    selected = [entry for entry in manifest["chunks"] if entry["step"] == origin]
    if len(selected) != 1:
        raise ValueError("an exact, unique final-preparation checkpoint is required; nearest frames are not accepted")
    return directory, raw, manifest, base, selected[0]


def prepare_config(source_run, additional_preparation_ps=400.0):
    """Read/hash the predetermined checkpoint; never create an engine or run MD."""
    source_identity()
    if (isinstance(additional_preparation_ps, bool) or not math.isfinite(additional_preparation_ps)
            or additional_preparation_ps <= 0):
        raise ValueError("additional preparation duration must be finite and positive")
    directory, raw, manifest, base, checkpoint = _original(source_run)
    steps = additional_preparation_ps * 1000 / base.dt_fs
    if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
        raise ValueError("additional duration must contain an integer number of frozen integration steps")
    config = Config(str(directory), hashlib.sha256(raw).hexdigest(), checkpoint["sha256"],
                    round(steps), base.checkpoint_steps).validate()
    _validate_duration(config, base)
    return config


def _validate_duration(config, base):
    if (config.additional_preparation_steps % base.sampling_refresh_steps
            or config.additional_preparation_steps % base.equilibration_record_stride
            or config.checkpoint_steps % base.equilibration_record_stride):
        raise ValueError("extension/checkpoint durations must preserve the registered refresh/observation boundaries")


def _load_source(config):
    config.validate()
    directory, raw, original, base, selected = _original(config.source_run)
    if (hashlib.sha256(raw).hexdigest() != config.source_manifest_sha256
            or selected["sha256"] != config.source_checkpoint_sha256):
        raise ValueError("the explicitly selected original manifest/checkpoint changed")
    _validate_duration(config, base)
    with np.load(directory / selected["file"], allow_pickle=False) as data:
        checkpoint = {name: data[name].copy() for name in data.files}
    if int(checkpoint["state_step"]) != base.equilibration_steps or int(checkpoint["active_segment"]) != -1:
        raise ValueError("the selected checkpoint must precede every NVE momentum draw")
    if not np.isclose(float(checkpoint["box_angstrom"]), original["box_angstrom"], rtol=0, atol=1e-12):
        raise ValueError("source checkpoint and registered box differ")
    return directory, original, base, selected, checkpoint


def _next_refresh(step, interval):
    return ((step + interval - 1) // interval) * interval


def _restore(context, masses, base, checkpoint, expected_step):
    from openmm import unit
    if (int(checkpoint["state_step"]) != expected_step or int(checkpoint["active_segment"]) != -1
            or not {"engine_positions_nm", "engine_half_step_velocities_nm_ps", "engine_forces_kj_mol_nm"}.issubset(checkpoint)):
        raise ValueError("preparation continuation requires an exact v2 raw checkpoint and preparation-only phase")
    context.loadCheckpoint(checkpoint["openmm_checkpoint"].tobytes())
    checked = native.validate_checkpoint_state(context, masses, base, checkpoint, allow_legacy=False)
    box = np.asarray(context.getState().getPeriodicBoxVectors(asNumpy=True) / unit.nanometer)
    if not np.allclose(box, np.eye(3) * float(checkpoint["box_angstrom"]) / 10, rtol=0, atol=1e-12):
        raise ValueError("restored Context periodic box differs from the selected checkpoint")
    return checked


def run(output, config, *, resume=False, wall_seconds=None, max_steps=None):
    """Advance only the declared preparation branch and persist pending phase."""
    import openmm as mm
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("wall budget must be finite and positive")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    identity = source_identity()
    source, original, base, selected, initial = _load_source(config)
    output = Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("the continuation output must be separate from the original run")
    origin = base.equilibration_steps
    target = origin + config.additional_preparation_steps
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if (manifest.get("schema") != SCHEMA or manifest["config"] != asdict(config)
                or manifest["source_sha256"] != identity or manifest["base_config"] != asdict(base)
                or manifest["environment"] != native.environment_identity()
                or manifest["target_preparation_step"] != target or manifest.get("production_started") is not False
                or manifest.get("preparation_only") is not True):
            raise ValueError("continuation source/configuration/environment/phase identity mismatch")
        if not {"model.xml", "integrator.xml"}.issubset(manifest.get("artifacts", {})):
            raise ValueError("continuation engine identities are required")
        native.verify_manifest(output, manifest)
        if not manifest["chunks"] or manifest["chunks"][0]["step"] != origin:
            raise ValueError("continuation must retain its pre-refresh branch checkpoint")
        recorded_steps = [entry["step"] for entry in manifest["chunks"]]
        if (any(second <= first for first, second in zip(recorded_steps, recorded_steps[1:]))
                or recorded_steps[-1] != manifest["committed_steps"]):
            raise ValueError("continuation checkpoint steps are inconsistent")
        if manifest["committed_steps"] < origin or manifest["committed_steps"] > target:
            raise ValueError("continuation step is outside its preparation-only interval")
        with np.load(output / manifest["chunks"][-1]["file"], allow_pickle=False) as data:
            checkpoint = {name: data[name].copy() for name in data.files}
        step = manifest["committed_steps"]
        if (int(checkpoint["next_refresh_step"]) != _next_refresh(step, base.sampling_refresh_steps)
                or manifest["next_refresh_step"] != int(checkpoint["next_refresh_step"])
                or manifest["refresh_count"] != (step-origin+base.sampling_refresh_steps-1)//base.sampling_refresh_steps
                or int(checkpoint["state_step"]) != step or int(checkpoint["active_segment"]) != -1):
            raise ValueError("saved periodic-refresh phase is inconsistent")
        if any(manifest["artifacts"][name]["sha256"] != original["artifacts"][name]["sha256"]
               for name in ("model.xml", "integrator.xml")):
            raise ValueError("continuation serialized engine differs from the original model")
        if manifest["complete"]:
            if step != target:
                raise ValueError("completed continuation has the wrong endpoint")
            return manifest
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("continuation output must be new/empty; resume a committed branch explicitly")
        checkpoint, step = initial, origin
        manifest = {"schema": SCHEMA, "config": asdict(config), "base_config": asdict(base),
                    "external_OpenMM_reference": True, "pyrpmd_production_trajectory": False,
                    "source_sha256": identity, "environment": native.environment_identity(),
                    "source_checkpoint_file": selected["file"], "source_preparation_step": origin,
                    "target_preparation_step": target, "committed_steps": origin,
                    "preparation_only": True, "production_started": False, "complete": False,
                    "sampling_thermostat": "frozen periodic global Maxwell refresh; remove_com=False",
                    "next_refresh_step": origin, "refresh_count": 0, "chunks": [], "artifacts": {},
                    "restart_validations": [], "elapsed_seconds": 0.0,
                    "created_utc": datetime.now(timezone.utc).isoformat()}
    system, integrator, context, masses = native.create_engine(base, original["box_angstrom"])
    rng = np.random.default_rng(base.seed)
    started = time.perf_counter()
    invocation_step = step
    try:
        for name, model in (("model.xml", system), ("integrator.xml", integrator)):
            encoded = mm.XmlSerializer.serialize(model).encode()
            if hashlib.sha256(encoded).hexdigest() != original["artifacts"][name]["sha256"]:
                raise ValueError("fresh frozen factory model/integrator differs from the original serialized engine")
            if resume and manifest["artifacts"][name]["sha256"] != original["artifacts"][name]["sha256"]:
                raise ValueError("continuation serialized engine differs from the original model")
        checked = _restore(context, masses, base, checkpoint, step)
        rng.bit_generator.state = json.loads(str(checkpoint["rng_json"]))
        manifest["restart_validations"].append({"step": step, **checked})
        if not resume:
            output.mkdir(parents=True, exist_ok=True)
            for name in ("model.xml", "integrator.xml"):
                shutil.copyfile(source / name, output / name)
                manifest["artifacts"][name] = {"sha256": sha256(output / name)}
        last_commit = time.perf_counter()

        def commit(observations, refresh_steps):
            nonlocal last_commit
            q, p, _, raw = native.get_state(context, masses, raw=True)
            arrays = {key: np.asarray([row[key] for row in observations]) for key in observations[0]} if observations else {}
            filename = f"chunk-{len(manifest['chunks']):06d}.npz"
            next_refresh = _next_refresh(step, base.sampling_refresh_steps)
            atomic_npz(output / filename, **arrays, q=q, p=p, state_step=step,
                       state_time=step*base.dt_fs*FS_TO_AU, box_angstrom=original["box_angstrom"],
                       active_segment=-1, next_refresh_step=next_refresh,
                       refresh_steps=np.asarray(refresh_steps, dtype=np.int64),
                       rng_json=json.dumps(rng.bit_generator.state, sort_keys=True), **raw,
                       openmm_checkpoint=np.frombuffer(context.createCheckpoint(), dtype=np.uint8))
            if source_identity() != identity or sha256(source / "manifest.json") != config.source_manifest_sha256:
                raise RuntimeError("source or original manifest changed; new chunk is not committed")
            manifest["chunks"].append({"file": filename, "sha256": sha256(output / filename),
                                       "step": step, "observations": len(observations)})
            manifest["committed_steps"] = step
            manifest["next_refresh_step"] = next_refresh
            manifest["complete"] = step == target
            manifest["elapsed_seconds"] += time.perf_counter() - last_commit
            atomic_json(output / "manifest.json", manifest)
            last_commit = time.perf_counter()
            print(json.dumps({"output": str(output), "absolute_step": step,
                              "absolute_preparation_ps": step*base.dt_fs/1000,
                              "complete": manifest["complete"], "production_started": False}), flush=True)

        if not resume:
            commit([], [])  # Restored source state, before the one pending refresh.
        while step < target:
            stop = min(target, step + config.checkpoint_steps)
            if max_steps is not None:
                stop = min(stop, invocation_step + max_steps)
            observations, refresh_steps = [], []
            while step < stop:
                if step % base.sampling_refresh_steps == 0:
                    q, _, _ = native.get_state(context, masses)
                    p = native.sample_momenta(rng, masses, base.temperature_K, remove_com=False)
                    native.set_state(context, q, p, masses)
                    refresh_steps.append(step)
                    manifest["refresh_count"] += 1
                boundary = (step // base.sampling_refresh_steps + 1) * base.sampling_refresh_steps
                advance = min(stop-step, boundary-step,
                              base.equilibration_record_stride-step % base.equilibration_record_stride)
                integrator.step(advance)
                step += advance
                if step % base.equilibration_record_stride == 0:
                    observations.append(native.observe(context, masses, base, step, -1, step))
            commit(observations, refresh_steps)
            if max_steps is not None and step-invocation_step >= max_steps:
                break
            if wall_seconds is not None and time.perf_counter()-started >= wall_seconds:
                break
        return manifest
    finally:
        del context, integrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--prepare-config", type=Path, help="write a checksum-bound config without creating a Context")
    parser.add_argument("--additional-preparation-ps", type=float, default=400.0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.prepare_config:
        if args.source_run is None or args.config is not None or args.resume:
            parser.error("--prepare-config requires --source-run and does not execute or resume MD")
        config = prepare_config(str(args.source_run), args.additional_preparation_ps)
        if args.prepare_config.exists():
            raise ValueError("refusing to overwrite an existing prepared continuation configuration")
        atomic_json(args.prepare_config, asdict(config))
        print(json.dumps(asdict(config), indent=2))
        return
    if args.output is None:
        parser.error("--output is required to execute a declared continuation")
    if args.config:
        config = Config(**json.loads(args.config.read_text()))
    elif args.resume:
        config = Config(**json.loads((args.output / "manifest.json").read_text())["config"])
    else:
        parser.error("use a prepared --config or explicitly --resume an existing branch")
    run(args.output, config, resume=args.resume, wall_seconds=args.wall_seconds, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
