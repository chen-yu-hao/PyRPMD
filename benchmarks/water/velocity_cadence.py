#!/usr/bin/env python3
"""Replay one preregistered NVE segment to assess velocity recording cadence.

This is a numerical sampling control, not an additional diffusion replica.
The original campaign is read-only. All compared cadences are subsamples of
one float64, every-step velocity record; no reference value selects the run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, atomic_npz, sha256
from benchmarks.water.native_reference import (
    Config, create_engine, environment_identity, get_state, sample_momenta,
    set_state, source_identity, validate_checkpoint_state,
)
from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU


def identity():
    return dict(source_identity(), **{
        "benchmarks/water/velocity_cadence.py": sha256(Path(__file__)),
    })


def vacf_trace(velocities, max_lag, *, component_block=24):
    """Uncentered all-origin, per-molecule velocity-dot autocorrelation.

    FFTs operate on small component blocks. This avoids allocating a complex
    time-by-molecule array for the complete source-sized dense record. Origins
    at each lag are normalized by their actual count; no statistical replicas
    are inferred from those origins or molecules.
    """
    values = np.asarray(velocities, dtype=np.float64)
    if (values.ndim != 3 or values.shape[2] != 3 or min(values.shape[:2]) < 1
            or not np.isfinite(values).all()):
        raise ValueError("velocities must be finite [frame,molecule,3]")
    if (isinstance(max_lag, bool) or not isinstance(max_lag, int)
            or not 0 <= max_lag < len(values)):
        raise ValueError("max_lag must be an integer smaller than frame count")
    if (isinstance(component_block, bool) or not isinstance(component_block, int)
            or component_block < 1):
        raise ValueError("component_block must be a positive integer")
    frames, molecules = values.shape[:2]
    nfft = 1 << (2 * frames - 1).bit_length()
    power = np.zeros(nfft // 2 + 1)
    components = values.reshape(frames, -1)
    for start in range(0, components.shape[1], component_block):
        spectrum = np.fft.rfft(components[:, start:start + component_block],
                               n=nfft, axis=0)
        power += np.einsum("fc,fc->f", spectrum.real, spectrum.real)
        power += np.einsum("fc,fc->f", spectrum.imag, spectrum.imag)
    numerator = np.fft.irfft(power, n=nfft)[:max_lag + 1]
    return numerator / ((frames - np.arange(max_lag + 1)) * molecules)


def compare_cadences(velocities, *, dt_fs, cadences_fs, max_lag_ps=2.):
    """Return both total recording error and isolated lag quadrature error."""
    if not math.isfinite(dt_fs) or dt_fs <= 0 or not math.isfinite(max_lag_ps) or max_lag_ps <= 0:
        raise ValueError("dt and maximum lag must be finite and positive")
    maximum = max_lag_ps * 1000 / dt_fs
    if not math.isclose(maximum, round(maximum), rel_tol=0, abs_tol=1e-10):
        raise ValueError("maximum lag must be an integer multiple of dt")
    full = vacf_trace(velocities, round(maximum))
    full_integral = float(trapezoid(full, dx=dt_fs / 1000) / 3)
    rows, curves = [], {"dense_lag_ps": np.arange(len(full)) * dt_fs / 1000,
                        "dense_vacf_trace_angstrom2_ps2": full}
    for cadence in cadences_fs:
        ratio = cadence / dt_fs
        if (not math.isfinite(ratio) or ratio < 1
                or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-12)
                or round(maximum) % round(ratio)):
            raise ValueError("each cadence must divide maximum lag and be an integer multiple of dt")
        stride = round(ratio)
        coarse = vacf_trace(np.asarray(velocities)[::stride], round(maximum) // stride)
        integral = float(trapezoid(coarse, dx=cadence / 1000) / 3)
        quadrature = float(trapezoid(full[::stride], dx=cadence / 1000) / 3)
        rows.append({
            "cadence_fs": float(cadence),
            "integral_trace_over_3_angstrom2_ps": integral,
            "difference_from_dense_angstrom2_ps": integral - full_integral,
            "dense_origins_quadrature_difference_angstrom2_ps": quadrature - full_integral,
            "frames": len(np.asarray(velocities)[::stride]),
            "origins_at_maximum_lag": len(np.asarray(velocities)[::stride]) - len(coarse) + 1,
        })
        curves[f"stride_{stride}_lag_ps"] = np.arange(len(coarse)) * cadence / 1000
        curves[f"stride_{stride}_vacf_trace_angstrom2_ps2"] = coarse
    return rows, curves


def select_checkpoint(manifest, registration):
    """Identify a predetermined checkpoint, never the newest available one."""
    expected = registration["selected_start_checkpoint"]["state_step"]
    if manifest["config"]["seed"] != registration["selected_seed"]:
        raise ValueError("selected chain seed differs from preregistration")
    selected = [record for record in manifest["chunks"] if record["step"] == expected]
    if len(selected) != 1:
        raise ValueError("require exactly one committed preregistered start checkpoint")
    if Path(selected[0]["file"]).name != selected[0]["file"]:
        raise ValueError("checkpoint filename must stay inside the selected run")
    return selected[0]


def prepare_selection(directory, registration_path, protocol_path):
    directory, registration_path, protocol_path = map(Path, (directory, registration_path, protocol_path))
    registration = json.loads(registration_path.read_text())
    if sha256(protocol_path) != registration["protocol_sha256"]:
        raise ValueError("baseline protocol checksum differs from preregistration")
    if registration["source_sha256"] != source_identity():
        raise ValueError("baseline source differs from preregistration")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    config_path = protocol_path.parent / "configs" / f"classical-seed-{registration['selected_seed']}.json"
    if manifest["config"] != json.loads(config_path.read_text()):
        raise ValueError("baseline configuration differs from its fixed registered configuration")
    if manifest["source_sha256"] != registration["source_sha256"]:
        raise ValueError("baseline checkpoint source differs from preregistration")
    if manifest["environment"] != environment_identity():
        raise ValueError("baseline checkpoint environment differs from current runtime")
    for name in ("model.xml", "integrator.xml"):
        record = manifest["artifacts"].get(name)
        if not record or sha256(directory / name) != record["sha256"]:
            raise ValueError(f"baseline serialized model identity failed: {name}")
    config = Config(**manifest["config"]).validate()
    if (registration["selected_segment"] != 0
            or config.equilibration_steps != registration["selected_start_checkpoint"]["state_step"]
            or config.dt_fs != registration["dense_dt_fs"]
            or config.segment_steps * config.dt_fs != 4000):
        raise ValueError("preregistered control requires the first 4 ps segment after preparation")
    selected = select_checkpoint(manifest, registration)
    if sha256(directory / selected["file"]) != selected["sha256"]:
        raise ValueError("selected baseline checkpoint checksum mismatch")
    selection = {
        "baseline_manifest_snapshot_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "baseline_directory": str(directory.resolve()),
        "checkpoint": selected,
        "config_sha256": sha256(config_path),
        "model_artifacts": {name: manifest["artifacts"][name] for name in ("model.xml", "integrator.xml")},
        "registration_sha256": sha256(registration_path),
        "protocol_sha256": sha256(protocol_path),
    }
    return registration, manifest, config, selection


def molecular_velocity(context, masses):
    _, p, _ = get_state(context, masses)
    # Sum physical atomic momenta and divide by molecular mass; linear virtual
    # sites are massless. Convert the coincident canonical velocity to A/ps.
    return (p[0].reshape(-1, 3, 3).sum(axis=1)
            / masses.reshape(-1, 3).sum(axis=1)[:, None]
            * BOHR_ANGSTROM * 1000 * FS_TO_AU)


def replay(directory, registration_path, protocol_path, output, *, resume=False, wall_seconds=None):
    """Checkpoint the numerical replay every original campaign checkpoint interval."""
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("wall_seconds must be finite and positive")
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or directory in output.parents or output in directory.parents:
        raise ValueError("cadence replay output must be separate from the baseline run")
    registration, baseline, config, selection = prepare_selection(directory, registration_path, protocol_path)
    source = identity()
    started, cpu_started = time.perf_counter(), time.process_time()
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        # The baseline may acquire additional immutable chunks meanwhile; its
        # live manifest SHA is informational, not a replay resume identity.
        expected = dict(selection)
        expected.pop("baseline_manifest_snapshot_sha256")
        previous = dict(manifest["selection"])
        previous.pop("baseline_manifest_snapshot_sha256")
        if (previous != expected or manifest["source_sha256"] != source
                or manifest["environment"] != environment_identity()):
            raise ValueError("cadence resume identity mismatch")
        for record in manifest["chunks"]:
            if (Path(record["file"]).name != record["file"]
                    or sha256(output / record["file"]) != record["sha256"]):
                raise ValueError("cadence committed chunk checksum mismatch")
        if manifest["complete"]:
            return manifest
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("cadence output must be empty/new; use --resume")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "pyrpmd.water-velocity-cadence-replay.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection": selection, "source_sha256": source, "environment": environment_identity(),
            "config": baseline["config"], "external_reference": True,
            "additional_diffusion_replica": False, "literature_reproduced": False,
            "baseline_modified": False, "precision": "float64",
            "committed_steps": 0, "chunks": [], "complete": False,
            "wall_seconds": 0., "process_cpu_seconds": 0.,
        }
    _, integrator, context, masses = create_engine(config, baseline["box_angstrom"])
    rng = np.random.default_rng(config.seed)
    try:
        if resume and manifest["chunks"]:
            checkpoint_path = output / manifest["chunks"][-1]["file"]
        else:
            checkpoint_path = directory / selection["checkpoint"]["file"]
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            context.loadCheckpoint(checkpoint["openmm_checkpoint"].tobytes())
            checked = validate_checkpoint_state(context, masses, config, checkpoint)
            rng.bit_generator.state = json.loads(str(checkpoint["rng_json"]))
            if resume and manifest["chunks"]:
                step = int(checkpoint["state_step"])
                if step != manifest["committed_steps"]:
                    raise ValueError("cadence checkpoint step differs from committed manifest")
            else:
                if (int(checkpoint["state_step"]) != config.equilibration_steps
                        or int(checkpoint["active_segment"]) != -1):
                    raise ValueError("selected checkpoint is not before the first NVE momentum refresh")
                q, _, _ = get_state(context, masses)
                set_state(context, q, sample_momenta(rng, masses, config.temperature_K, remove_com=True), masses)
                step = 0
        manifest.setdefault("restart_validations", []).append(checked)
        last_wall, last_cpu = time.perf_counter(), time.process_time()
        while step < config.segment_steps:
            stop = min(step + config.checkpoint_steps, config.segment_steps)
            frames, velocities = [], []
            if step == 0:
                frames.append(0)
                velocities.append(molecular_velocity(context, masses))
            while step < stop:
                integrator.step(1)
                step += 1
                frames.append(step)
                velocities.append(molecular_velocity(context, masses))
            q, p, _, raw = get_state(context, masses, raw=True)
            filename = f"chunk-{len(manifest['chunks']):06d}.npz"
            atomic_npz(output / filename, step=np.asarray(frames),
                       molecular_velocity_angstrom_ps=np.asarray(velocities, dtype=np.float64),
                       q=q, p=p, state_step=step, rng_json=json.dumps(rng.bit_generator.state, sort_keys=True),
                       **raw, openmm_checkpoint=np.frombuffer(context.createCheckpoint(), dtype=np.uint8))
            if identity() != source or sha256(registration_path) != selection["registration_sha256"]:
                raise RuntimeError("cadence source/registration changed; chunk not committed")
            manifest["chunks"].append({"file": filename, "sha256": sha256(output / filename),
                                       "step": step, "frames": len(frames)})
            manifest["committed_steps"] = step
            manifest["complete"] = step == config.segment_steps
            manifest["wall_seconds"] += time.perf_counter() - last_wall
            manifest["process_cpu_seconds"] += time.process_time() - last_cpu
            atomic_json(output / "manifest.json", manifest)
            print(json.dumps({"cadence_replay_steps": step, "complete": manifest["complete"],
                              "wall_seconds_this_invocation": time.perf_counter() - started,
                              "process_cpu_seconds_this_invocation": time.process_time() - cpu_started}), flush=True)
            last_wall, last_cpu = time.perf_counter(), time.process_time()
            if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
                break
        return manifest
    finally:
        del context, integrator


def analyze(directory, registration_path, output):
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory / "manifest.json").read_text())
    registration = json.loads(Path(registration_path).read_text())
    if not manifest["complete"]:
        raise ValueError("cadence comparison requires the complete preregistered segment")
    if sha256(registration_path) != manifest["selection"]["registration_sha256"]:
        raise ValueError("cadence analysis registration checksum mismatch")
    source = identity()
    if manifest["source_sha256"] != source:
        raise ValueError("cadence analysis source differs from replay")
    config = Config(**manifest["config"])
    velocities = np.empty((config.segment_steps + 1, config.cells_per_axis**3, 3), dtype=np.float64)
    offset = 0
    for record in manifest["chunks"]:
        path = directory / record["file"]
        if Path(record["file"]).name != record["file"] or sha256(path) != record["sha256"]:
            raise ValueError("cadence chunk checksum mismatch")
        with np.load(path, allow_pickle=False) as chunk:
            steps = chunk["step"]
            values = chunk["molecular_velocity_angstrom_ps"]
            if not np.array_equal(steps, np.arange(offset, offset + len(steps))):
                raise ValueError("cadence record has missing, duplicated or unordered steps")
            if values.shape != (len(steps), config.cells_per_axis**3, 3) or values.dtype != np.float64:
                raise ValueError("cadence record shape or precision mismatch")
            velocities[offset:offset + len(steps)] = values
            offset += len(steps)
    if offset != len(velocities):
        raise ValueError("cadence record does not cover the full registered segment")
    rows, curves = compare_cadences(velocities, dt_fs=config.dt_fs,
                                   cadences_fs=registration["record_cadences_fs"])
    margin = registration["numerical_acceptance_absolute_difference_angstrom2_ps"]
    report = {
        "schema": "pyrpmd.water-velocity-cadence-analysis.v1",
        "selection": manifest["selection"], "source_sha256": source,
        "replay_manifest_sha256": sha256(directory / "manifest.json"),
        "replay_restart_validations": manifest["restart_validations"],
        "environment": environment_identity(), "frames": offset,
        "molecules": config.cells_per_axis**3, "maximum_lag_ps": 2.,
        "cadences": rows, "numerical_margin_angstrom2_ps": margin,
        "recording_cadence_pass": all(abs(row["difference_from_dense_angstrom2_ps"]) <= margin for row in rows),
        "comparison": "All cadences subsample the same dense trajectory; coarse records also reduce available time origins.",
        "quadrature_diagnostic": "Dense-origin VACF sampled at each lag spacing isolates trapezoid quadrature from origin subsampling.",
        "no_velocity_demeaning": True, "additional_diffusion_replica": False,
        "literature_reproduced": False, "baseline_modified": False,
        "limitations": ["One predetermined segment assesses recording numerics, not long-time transport convergence.",
                        "CPU checkpoint force-cache roundoff prevents a bitwise future-trajectory replay claim."],
        "replay_wall_seconds": manifest["wall_seconds"],
        "replay_process_cpu_seconds": manifest["process_cpu_seconds"],
    }
    if identity() != source:
        raise RuntimeError("cadence analysis source changed during analysis")
    output.mkdir(parents=True, exist_ok=True)
    atomic_npz(output / "curves.npz", **curves)
    report["curves_sha256"] = sha256(output / "curves.npz")
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--baseline", type=Path, required=True)
    run_parser.add_argument("--registration", type=Path, required=True)
    run_parser.add_argument("--protocol", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--wall-seconds", type=float)
    analysis_parser = sub.add_parser("analyze")
    analysis_parser.add_argument("--input", type=Path, required=True)
    analysis_parser.add_argument("--registration", type=Path, required=True)
    analysis_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        replay(args.baseline, args.registration, args.protocol, args.output,
               resume=args.resume, wall_seconds=args.wall_seconds)
    else:
        analyze(args.input, args.registration, args.output)


if __name__ == "__main__":
    main()
