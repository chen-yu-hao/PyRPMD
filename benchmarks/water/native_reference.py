#!/usr/bin/env python3
"""Checkpointed OpenMM external classical reference for q-TIP4P/F water.

OpenMM, not the PyRPMD production propagator, advances these trajectories.
The force field is the same independently audited factory. Built-in Verlet
avoids the observed intermediate virtual-site refresh limitation of Reference/
CPU CustomIntegrator. Stored leapfrog velocities are converted to coincident
physical velocities whenever states are set or read. Sampling periodically
refreshes all physical Maxwell momenta, followed by predeclared NVE segments.
There is no reference-value stopping rule or trajectory selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, atomic_npz, sha256
from benchmarks.water.qtip4pf import (
    D_MASS_DA, H_MASS_DA, O_MASS_DA, initial_grid, make_system,
)
from pyrpmd.constants import AMU_TO_AU, BOHR_ANGSTROM, FS_TO_AU, KB_AU, KCAL_MOL_TO_AU

BOHR_NM = BOHR_ANGSTROM / 10
VELOCITY_NM_PS = BOHR_NM * 1000 * FS_TO_AU
HARTREE_KJ_MOL = 4.184 / KCAL_MOL_TO_AU
LEGACY_CHECKPOINT_SOURCE_SHA256 = "592ca5eed2e0234bf936f8808fe4877281c3b762720e68fcd4eb9caab30482f2"
CHECKPOINT_CONTRACT_VERSION = 2


@dataclass(frozen=True)
class Config:
    cells_per_axis: int = 6
    beads: int = 1
    isotope: str = "H2O"
    temperature_K: float = 298.0
    density_g_cm3: float = 0.997
    dt_fs: float = .25
    integration: str = "velocity-verlet"
    cutoff_angstrom: float = 9.0
    ewald_tolerance: float = 1e-5
    dispersion_correction: bool = False
    equilibration_steps: int = 400000
    segment_steps: int = 16000
    segments: int = 25
    record_stride: int = 20
    equilibration_record_stride: int = 400
    checkpoint_steps: int = 4000
    sampling_refresh_fs: float = 100.0
    seed: int = 1101
    platform: str = "CPU"
    threads: int = 1
    trajectory_precision: str = "float32"
    initial_checkpoint: str | None = None

    def validate(self):
        for name in ("cells_per_axis", "segment_steps", "segments", "record_stride",
                     "equilibration_record_stride", "checkpoint_steps", "threads", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.equilibration_steps, bool)
                or not isinstance(self.equilibration_steps, int) or self.equilibration_steps < 0):
            raise ValueError("equilibration_steps must be a nonnegative integer")
        for name in ("temperature_K", "density_g_cm3", "dt_fs",
                     "cutoff_angstrom", "ewald_tolerance", "sampling_refresh_fs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.beads, int) or self.beads != 1 or isinstance(self.beads, bool):
            raise ValueError("native_reference is an external classical P=1 driver")
        if self.isotope not in ("H2O", "D2O") or self.platform not in ("CPU", "Reference"):
            raise ValueError("unsupported isotope or platform")
        if self.integration != "velocity-verlet":
            raise ValueError("only built-in Verlet with synchronized velocities is validated")
        ratio = self.sampling_refresh_fs / self.dt_fs
        if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-12):
            raise ValueError("sampling refresh time must be an integer multiple of dt")
        if self.segment_steps % self.record_stride:
            raise ValueError("record_stride must divide segment_steps")
        if self.equilibration_steps % self.equilibration_record_stride:
            raise ValueError("equilibration_record_stride must divide equilibration_steps")
        if self.trajectory_precision not in ("float32", "float64"):
            raise ValueError("trajectory_precision must be float32 or float64")
        if self.ewald_tolerance >= 1 or not isinstance(self.dispersion_correction, bool):
            raise ValueError("invalid electrostatic tolerance or dispersion correction")
        if self.seed >= 2**31:
            raise ValueError("seed must fit OpenMM's signed integer seed")
        return self

    @property
    def total_steps(self):
        return self.equilibration_steps + self.segments * self.segment_steps

    @property
    def sampling_refresh_steps(self):
        return round(self.sampling_refresh_fs / self.dt_fs)


def source_identity():
    names = ("benchmarks/water/native_reference.py", "benchmarks/water/campaign.py",
             "benchmarks/water/qtip4pf.py", "src/pyrpmd/constants.py")
    return {name: sha256(ROOT / name) for name in names}


def environment_identity():
    return {"python": platform.python_version(), "machine": platform.machine(),
            "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "openmm")}}


def create_engine(config, box):
    """Create an uninitialized OpenMM leapfrog Context; set/get convert velocities."""
    import openmm as mm
    hydrogen = H_MASS_DA if config.isotope == "H2O" else D_MASS_DA
    system = make_system(molecules=config.cells_per_axis**3, box_angstrom=box,
                         cutoff_angstrom=config.cutoff_angstrom, ewald_tolerance=config.ewald_tolerance,
                         dispersion_correction=config.dispersion_correction, hydrogen_mass_da=hydrogen)
    for force in system.getForces():
        force.setForceGroup(1 if isinstance(force, mm.NonbondedForce) else 0)
    integrator = mm.VerletIntegrator(config.dt_fs / 1000)
    properties = {"Threads": str(config.threads), "DeterministicForces": "true"} if config.platform == "CPU" else {}
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(config.platform), properties)
    masses = np.tile([O_MASS_DA, hydrogen, hydrogen], config.cells_per_axis**3) * AMU_TO_AU
    return system, integrator, context, masses


def set_state(context, q, p, masses):
    """Set coincident OHH q/p and convert p to the engine's half-step velocity."""
    from openmm import unit
    q, p = np.asarray(q), np.asarray(p)
    if q.ndim == 3:
        if q.shape[0] != 1 or p.shape != q.shape:
            raise ValueError("native_reference requires exactly one classical copy")
        q, p = q[0], p[0]
    if q.shape != (len(masses), 3) or p.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError("physical state must be finite OHH coordinates/momenta")
    n = len(masses) // 3
    positions, velocities = np.zeros((n, 4, 3)), np.zeros((n, 4, 3))
    positions[:, :3] = q.reshape(n, 3, 3) * BOHR_NM
    velocities[:, :3] = (p / masses[:, None]).reshape(n, 3, 3) * VELOCITY_NM_PS
    context.setPositions(positions.reshape(-1, 3))
    context.computeVirtualSites()
    forces = np.asarray(context.getState(getForces=True).getForces(asNumpy=True)
                        / (unit.kilojoule_per_mole / unit.nanometer)).reshape(n, 4, 3)[:, :3]
    dt_ps = float(context.getIntegrator().getStepSize() / unit.picosecond)
    velocities[:, :3] -= .5 * dt_ps * forces / (masses / AMU_TO_AU).reshape(n, 3, 1)
    context.setVelocities(velocities.reshape(-1, 3))


def get_state(context, masses, *, energy=False, raw=False):
    from openmm import unit
    state = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=energy)
    n = len(masses) // 3
    positions_nm = np.asarray(state.getPositions(asNumpy=True) / unit.nanometer)
    half_step_velocities_nm_ps = np.asarray(state.getVelocities(asNumpy=True) / (unit.nanometer / unit.picosecond))
    forces_kj_mol_nm = np.asarray(state.getForces(asNumpy=True) / (unit.kilojoule_per_mole / unit.nanometer))
    q = positions_nm.reshape(n, 4, 3)[:, :3].reshape(-1, 3) / BOHR_NM
    velocity_nm_ps = half_step_velocities_nm_ps.reshape(n, 4, 3)[:, :3].reshape(-1, 3)
    forces_nm = forces_kj_mol_nm.reshape(n, 4, 3)[:, :3].reshape(-1, 3)
    dt_ps = float(context.getIntegrator().getStepSize() / unit.picosecond)
    v = (velocity_nm_ps + .5 * dt_ps * forces_nm / (masses / AMU_TO_AU)[:, None]) / VELOCITY_NM_PS
    p = v * masses[:, None]
    if not np.isfinite(q).all() or not np.isfinite(p).all():
        raise FloatingPointError("nonfinite native OpenMM state")
    potential = float(state.getPotentialEnergy() / unit.kilojoule_per_mole) / HARTREE_KJ_MOL if energy else None
    result = q[None], p[None], potential
    if raw:
        return (*result, {"engine_positions_nm": positions_nm,
                          "engine_half_step_velocities_nm_ps": half_step_velocities_nm_ps,
                          "engine_forces_kj_mol_nm": forces_kj_mol_nm})
    return result


def validate_checkpoint_state(context, masses, config, checkpoint, *, allow_legacy=False):
    """Verify the binary engine state and bounded CPU force-cache roundoff.

    CPU checkpoints preserve all positions and half-step velocities bitwise,
    but rebuilding neighbor/PME force caches can change final force bits. The
    canonical momentum difference must equal dt/2 times that measured force
    difference, after unit conversion. No physical state is reset here.
    """
    q, p, _, actual = get_state(context, masses, raw=True)
    if not np.array_equal(q, checkpoint["q"]):
        raise ValueError("loaded native checkpoint physical position mismatch")
    keys = tuple(actual)
    redundant = all(name in checkpoint for name in keys)
    if any(name in checkpoint for name in keys) and not redundant:
        raise ValueError("incomplete native checkpoint raw-state contract")
    physical = np.arange(len(actual["engine_positions_nm"])).reshape(-1, 4)[:, :3].ravel()
    new_force = actual["engine_forces_kj_mol_nm"][physical]
    if redundant:
        for name in keys[:2]:
            if not np.array_equal(actual[name], checkpoint[name]):
                raise ValueError(f"loaded native checkpoint raw state mismatch: {name}")
        old_force = np.asarray(checkpoint["engine_forces_kj_mol_nm"])[physical]
    elif allow_legacy:
        # Only the explicitly migrated, checksum-verified v1 checkpoint is
        # eligible. Its force is inferred, not claimed to have been stored.
        velocity = actual["engine_half_step_velocities_nm_ps"][physical]
        old_force = ((checkpoint["p"][0] / masses[:, None] * VELOCITY_NM_PS - velocity)
                     * (masses / AMU_TO_AU)[:, None] * 2 / (config.dt_fs / 1000))
    else:
        raise ValueError("legacy checkpoint requires explicit versioned contract migration")
    if not np.isfinite(old_force).all() or not np.isfinite(new_force).all():
        raise ValueError("nonfinite checkpoint force")
    # Independent bounds, in kJ/mol/nm, keep an altered model from being
    # accepted just because the velocity conversion remains algebraic.
    # Measured liquid CPU round trips: max 0.00174; RMS relative 2.24e-7.
    force_atol, force_rtol = (0.005, 2e-6) if config.platform == "CPU" else (1e-9, 1e-12)
    delta_force = new_force - old_force
    rms_relative = float(np.sqrt(np.mean(delta_force**2)) / max(1., np.sqrt(np.mean(old_force**2))))
    if (not np.allclose(new_force, old_force, atol=force_atol, rtol=force_rtol)
            or rms_relative > (1e-6 if config.platform == "CPU" else 1e-12)):
        raise ValueError("checkpoint recomputed force exceeds declared roundoff bound")
    predicted_delta_p = .5 * config.dt_fs / 1000 * AMU_TO_AU / VELOCITY_NM_PS * delta_force
    residual = p[0] - checkpoint["p"][0] - predicted_delta_p
    if not np.isfinite(residual).all() or np.max(abs(residual)) > 2e-13:
        raise ValueError("checkpoint canonical momentum differs beyond force-cache conversion")
    return {"contract_version": CHECKPOINT_CONTRACT_VERSION,
            "legacy_force_inferred": not redundant,
            "raw_positions_and_half_step_velocities_exact": redundant,
            "physical_q_exact": True,
            "force_max_difference_kj_mol_nm": float(np.max(abs(delta_force))),
            "force_RMS_relative_difference": rms_relative,
            "force_atol_kj_mol_nm": force_atol, "force_rtol": force_rtol,
            "canonical_p_max_difference_au": float(np.max(abs(p-checkpoint["p"]))),
            "canonical_p_conversion_residual_max_au": float(np.max(abs(residual)))}


def sample_momenta(rng, masses, temperature_K, *, remove_com):
    p = rng.normal(size=(len(masses), 3)) * np.sqrt(masses[:, None] * KB_AU * temperature_K)
    if remove_com:
        p -= masses[:, None] * p.sum(axis=0) / masses.sum()
    return p[None]


def observe(context, masses, config, step, segment, segment_step):
    q, p, potential = get_state(context, masses, energy=True)
    n = len(masses) // 3
    molecular = q.reshape(n, 3, 3)
    weights = masses.reshape(n, 3) / masses.reshape(n, 3).sum(axis=1, keepdims=True)
    velocities = (p[0] / masses[:, None]).reshape(n, 3, 3)
    bonds = (molecular[:, 1:] - molecular[:, :1]) * BOHR_ANGSTROM
    lengths = np.linalg.norm(bonds, axis=-1)
    angles = np.degrees(np.arccos(np.clip(np.sum(bonds[:, 0] * bonds[:, 1], axis=1) / lengths.prod(axis=1), -1, 1)))
    kinetic = float(np.sum(p[0]**2 / (2 * masses[:, None])))
    dtype = np.dtype(config.trajectory_precision)
    return {"step": step, "segment": segment, "segment_step": segment_step,
            "molecular_com_bohr": np.sum(molecular * weights[:, :, None], axis=1).astype(dtype),
            "molecular_velocity_au": np.sum(velocities * weights[:, :, None], axis=1).astype(dtype),
            "oxygen_centroid_bohr": molecular[:, 0].astype(dtype),
            "temperature_momenta_K": 2 * kinetic / (3 * len(masses) * KB_AU),
            "temperature_centroid_K": 2 * kinetic / ((3 * len(masses) - (3 if segment >= 0 else 0)) * KB_AU),
            "potential_per_molecule_hartree": potential / n,
            "hamiltonian_per_bead_hartree": kinetic + potential,
            "mean_OH_angstrom": float(lengths.mean()), "mean_HOH_degrees": float(angles.mean()),
            "hydrogen_radius_gyration_angstrom": 0.0}


def verify_manifest(output, manifest):
    output = Path(output)
    for entry in manifest["chunks"]:
        if Path(entry["file"]).name != entry["file"] or sha256(output / entry["file"]) != entry["sha256"]:
            raise ValueError(f"committed checkpoint checksum mismatch: {entry['file']}")
    for name, artifact in manifest["artifacts"].items():
        if Path(name).name != name or sha256(output / name) != artifact["sha256"]:
            raise ValueError(f"serialized engine checksum mismatch: {name}")


def run(output, config, *, resume=False, wall_seconds=None, max_steps=None,
        migrate_checkpoint_contract=False):
    """Run a bounded invocation, restoring OpenMM and external RNG checkpoints."""
    import openmm as mm
    config.validate()
    if migrate_checkpoint_contract and not resume:
        raise ValueError("checkpoint contract migration requires --resume")
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("wall_seconds must be positive")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    output = Path(output).resolve()
    started = time.perf_counter()
    identity, environment = source_identity(), environment_identity()
    hydrogen = H_MASS_DA if config.isotope == "H2O" else D_MASS_DA
    initial_q, box = initial_grid(cells_per_axis=config.cells_per_axis, density_g_cm3=config.density_g_cm3,
                                 seed=config.seed, hydrogen_mass_da=hydrogen)
    initial_sha = sha256(config.initial_checkpoint) if config.initial_checkpoint else None
    migrating = False
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        old_identity = manifest["source_sha256"]
        native_path = "benchmarks/water/native_reference.py"
        allowed_legacy = dict(identity, **{native_path: LEGACY_CHECKPOINT_SOURCE_SHA256})
        if (migrate_checkpoint_contract and old_identity == allowed_legacy
                and manifest.get("checkpoint_contract_version", 1) == 1):
            migrating = True
        if manifest["config"] != asdict(config) or (old_identity != identity and not migrating):
            raise ValueError("resume source/configuration identity mismatch")
        if migrate_checkpoint_contract and not migrating:
            raise ValueError("migration only accepts the specifically audited legacy source SHA")
        if manifest["environment"] != environment or manifest["initial_checkpoint_sha256"] != initial_sha:
            raise ValueError("resume environment/initial-checkpoint identity mismatch")
        verify_manifest(output, manifest)
        if manifest["complete"]:
            return manifest
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty/new; use --resume for a committed run")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": "pyrpmd.water-openmm-external-reference.v1", "external_reference": True,
                    "propagator": "OpenMM VerletIntegrator; synchronized physical velocities",
                    "sampling_thermostat": "periodic global Maxwell momentum refresh; no thermostat in NVE",
                    "pyrpmd_production_trajectory": False, "config": asdict(config), "source_sha256": identity,
                    "environment": environment, "initial_checkpoint_sha256": initial_sha,
                    "box_angstrom": box, "chunks": [], "artifacts": {}, "committed_steps": 0,
                    "complete": False,  "literature_reproduced": False,
                    "elapsed_seconds": 0.0, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "trajectory_precision": config.trajectory_precision, "checkpoint_precision": "float64",
                    "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
                    "independence_unit": "independently seeded equilibrated chain; NVE segments share a configuration chain"}
        try:
            manifest["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["git_revision"] = None
    system, integrator, context, masses = create_engine(config, box)
    rng = np.random.default_rng(config.seed)
    try:
        if not resume:
            for name, value in (("model.xml", system), ("integrator.xml", integrator)):
                (output / name).write_text(mm.XmlSerializer.serialize(value))
                manifest["artifacts"][name] = {"sha256": sha256(output / name)}
        if resume and manifest["chunks"]:
            with np.load(output / manifest["chunks"][-1]["file"], allow_pickle=False) as checkpoint:
                context.loadCheckpoint(checkpoint["openmm_checkpoint"].tobytes())
                step = int(checkpoint["state_step"])
                active_segment = int(checkpoint["active_segment"])
                rng.bit_generator.state = json.loads(str(checkpoint["rng_json"]))
                legacy_hashes = {item["historical_checkpoint_sha256"] for item in manifest.get("source_migrations", [])}
                checked = validate_checkpoint_state(
                    context, masses, config, checkpoint,
                    allow_legacy=migrating or manifest["chunks"][-1]["sha256"] in legacy_hashes)
                checked.update({"step": step, "chunk_sha256": manifest["chunks"][-1]["sha256"],
                                "validated_utc": datetime.now(timezone.utc).isoformat()})
                if migrating:
                    backup = "manifest.before-checkpoint-contract-v2.json"
                    if (output / backup).exists():
                        raise ValueError("refusing to overwrite existing migration evidence")
                    (output / backup).write_bytes((output / "manifest.json").read_bytes())
                    migration = {"reason": "Raw engine state identity replaces invalid exact canonical-p requirement on CPU force-cache rebuild.",
                                 "from_source_sha256": old_identity, "to_source_sha256": identity,
                                 "historical_manifest_file": backup, "historical_manifest_sha256": sha256(output / backup),
                                 "historical_checkpoint_sha256": manifest["chunks"][-1]["sha256"],
                                 "at_committed_step": step, "state_validation": checked,
                                 "propagator_model_protocol_unchanged": True}
                    manifest.setdefault("source_migrations", []).append(migration)
                    manifest["source_sha256"] = identity
                    manifest["checkpoint_contract_version"] = CHECKPOINT_CONTRACT_VERSION
                    manifest["artifacts"][backup] = {"sha256": sha256(output / backup)}
                    atomic_json(output / "manifest.json", manifest)
                manifest.setdefault("restart_validations", []).append(checked)
        else:
            if config.initial_checkpoint:
                with np.load(config.initial_checkpoint, allow_pickle=False) as checkpoint:
                    if not np.isclose(float(checkpoint["box_angstrom"]), box, atol=1e-10, rtol=0):
                        raise ValueError("initial checkpoint box differs from requested density")
                    initial_q = checkpoint["q"].mean(axis=0)
            set_state(context, initial_q[None], sample_momenta(rng, masses, config.temperature_K, remove_com=False), masses)
            step, active_segment = 0, -1
        invocation_step = step
        last_commit = time.perf_counter()
        while step < config.total_steps:
            stop = min(config.total_steps, step + config.checkpoint_steps)
            if max_steps is not None:
                stop = min(stop, invocation_step + max_steps)
            observations = []
            while step < stop:
                if step < config.equilibration_steps:
                    segment, segment_step = -1, step
                    boundary = min(config.equilibration_steps,
                                   (step // config.sampling_refresh_steps + 1) * config.sampling_refresh_steps)
                    stride = config.equilibration_record_stride
                    if step > 0 and step % config.sampling_refresh_steps == 0:
                        q, _, _ = get_state(context, masses)
                        set_state(context, q, sample_momenta(rng, masses, config.temperature_K, remove_com=False), masses)
                else:
                    segment, segment_step = divmod(step - config.equilibration_steps, config.segment_steps)
                    boundary = config.equilibration_steps + (segment + 1) * config.segment_steps
                    stride = config.record_stride
                    if segment_step == 0 and active_segment != segment:
                        q, _, _ = get_state(context, masses)
                        set_state(context, q, sample_momenta(rng, masses, config.temperature_K, remove_com=True), masses)
                        active_segment = segment
                        observations.append(observe(context, masses, config, step, segment, 0))
                advance = min(stop - step, boundary - step, stride - segment_step % stride)
                integrator.step(advance)
                step += advance
                segment_step += advance
                if segment_step % stride == 0:
                    observations.append(observe(context, masses, config, step, segment, segment_step))
            q, p, _, raw_checkpoint = get_state(context, masses, raw=True)
            arrays = {key: np.asarray([row[key] for row in observations]) for key in observations[0]} if observations else {}
            filename = f"chunk-{len(manifest['chunks']):06d}.npz"
            atomic_npz(output / filename, **arrays, q=q, p=p, state_step=step,
                       state_time=step * config.dt_fs * FS_TO_AU, box_angstrom=box,
                       active_segment=active_segment, rng_json=json.dumps(rng.bit_generator.state, sort_keys=True),
                       **raw_checkpoint,
                       openmm_checkpoint=np.frombuffer(context.createCheckpoint(), dtype=np.uint8))
            if source_identity() != identity:
                raise RuntimeError("native-reference source changed during propagation; chunk not committed")
            manifest["chunks"].append({"file": filename, "sha256": sha256(output / filename),
                                       "step": step, "observations": len(observations)})
            manifest["committed_steps"] = step
            manifest["complete"] = step == config.total_steps
            manifest["elapsed_seconds"] += time.perf_counter() - last_commit
            atomic_json(output / "manifest.json", manifest)
            print(json.dumps({"output": str(output), "step": step, "time_ps": step * config.dt_fs / 1000,
                              "total_steps": config.total_steps, "elapsed_seconds": manifest["elapsed_seconds"],
                              "complete": manifest["complete"], "external_reference": True}), flush=True)
            last_commit = time.perf_counter()
            if max_steps is not None and step - invocation_step >= max_steps:
                break
            if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
                break
        return manifest
    finally:
        del context, integrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--migrate-checkpoint-contract", action="store_true",
                        help="explicitly migrate only the audited original v1 source/checkpoint contract")
    args = parser.parse_args()
    if args.config:
        config = Config(**json.loads(args.config.read_text()))
    elif args.resume:
        config = Config(**json.loads((args.output / "manifest.json").read_text())["config"])
    else:
        config = Config()
    run(args.output, config, resume=args.resume, wall_seconds=args.wall_seconds, max_steps=args.max_steps,
        migrate_checkpoint_contract=args.migrate_checkpoint_contract)


if __name__ == "__main__":
    main()
