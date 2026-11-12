#!/usr/bin/env python3
"""Restartable full-PME OpenMM RPMD reference, separate from PyRPMD production.

Every force is evaluated on every bead. Preparation uses an external global
Andersen heat bath with geometric waiting times of a declared mean, not a
fixed collision period. The same NumPy RNG supplies waiting times and Maxwell
momenta. Production consists of fixed NVE segments with one new momentum draw
per segment and removal of only the total ring COM momentum.

OpenMM Context checkpoints alone do not reliably preserve RPMD copies. Every
commit therefore includes all-copy raw positions/velocities, physical q/p,
forces, the Context blob, the absolute collision deadline and the RNG/phase.
GPU selection is explicit and remains untested in the recorded CPU environment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, atomic_npz, sha256
from benchmarks.water.qtip4pf import H_MASS_DA, O_MASS_DA, PARAMETERS, initial_grid, make_system
from pyrpmd.constants import AMU_TO_AU, BOHR_ANGSTROM, FS_TO_AU, KB_AU, KCAL_MOL_TO_AU
from pyrpmd.workflows.sampling import AndersenSchedule

BOHR_NM = BOHR_ANGSTROM / 10
VELOCITY_NM_PS = BOHR_NM * 1000 * FS_TO_AU
HARTREE_KJ_MOL = 4.184 / KCAL_MOL_TO_AU
SCHEMA = "pyrpmd.water-openmm-full-rpmd-reference.v1"


@dataclass(frozen=True)
class Config:
    cells_per_axis: int = 6
    beads: int = 32
    isotope: str = "H2O"
    temperature_K: float = 298.
    density_g_cm3: float = .997
    dt_fs: float = .1
    cutoff_angstrom: float = 9.
    ewald_tolerance: float = 1e-5
    dispersion_correction: bool = False
    equilibration_steps: int = 1000000
    segment_steps: int = 40000
    segments: int = 100
    record_stride: int = 50
    equilibration_record_stride: int = 1000
    checkpoint_steps: int = 10000
    max_native_block_steps: int = 50
    andersen_mean_interval_fs: float = 100.
    andersen_schedule: str = "geometric"
    seed: int = 2101
    platform: str = "CPU"
    threads: int = 1
    platform_properties: dict[str, str] = field(default_factory=dict)
    trajectory_precision: str = "float32"
    initial_checkpoint: str | None = None

    def validate(self):
        for name in ("cells_per_axis", "beads", "segment_steps", "segments", "record_stride",
                     "equilibration_record_stride", "checkpoint_steps", "max_native_block_steps", "seed", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.equilibration_steps, bool) or not isinstance(self.equilibration_steps, int)
                or self.equilibration_steps < 0):
            raise ValueError("equilibration_steps must be a nonnegative integer")
        for name in ("temperature_K", "density_g_cm3", "dt_fs", "cutoff_angstrom",
                     "ewald_tolerance", "andersen_mean_interval_fs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.isotope != "H2O" or self.andersen_schedule != "geometric":
            raise ValueError("this declared quantum reference supports H2O and geometric Andersen sampling")
        if self.platform not in ("Reference", "CPU", "CUDA", "OpenCL", "HIP"):
            raise ValueError("select an explicit supported OpenMM platform")
        if (not isinstance(self.platform_properties, dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.platform_properties.items())):
            raise ValueError("platform_properties must map strings to strings")
        if self.platform in ("Reference", "CPU") and self.platform_properties:
            raise ValueError("Reference/CPU properties are fixed by the explicit platform and threads fields")
        if self.platform not in ("Reference", "CPU"):
            if self.platform_properties.get("Precision") not in ("mixed", "double"):
                raise ValueError("an untested GPU configuration must explicitly choose mixed or double precision")
        ratio = self.andersen_mean_interval_fs / self.dt_fs
        if (ratio < 1 or ratio > np.iinfo(np.int64).max
                or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-10)):
            raise ValueError("Andersen mean interval must be an integer number of MD steps")
        if self.segment_steps % self.record_stride or self.equilibration_steps % self.equilibration_record_stride:
            raise ValueError("phase observation strides must divide their complete phase lengths")
        if self.trajectory_precision not in ("float32", "float64"):
            raise ValueError("trajectory_precision must be float32 or float64")
        if self.initial_checkpoint is not None and (not isinstance(self.initial_checkpoint, str) or not self.initial_checkpoint):
            raise ValueError("initial_checkpoint must be a nonempty path string or null")
        if not isinstance(self.dispersion_correction, bool) or self.ewald_tolerance >= 1 or self.seed >= 2**31:
            raise ValueError("invalid correction flag, Ewald tolerance or signed engine seed")
        if self.total_steps > np.iinfo(np.int64).max:
            raise ValueError("campaign step count exceeds the checkpoint clock range")
        return self

    @property
    def total_steps(self):
        return self.equilibration_steps + self.segments * self.segment_steps

    @property
    def mean_collision_steps(self):
        return round(self.andersen_mean_interval_fs / self.dt_fs)


def source_identity():
    return {name: sha256(ROOT / name) for name in (
        "benchmarks/water/native_quantum.py", "benchmarks/water/qtip4pf.py",
        "benchmarks/water/campaign.py", "src/pyrpmd/constants.py", "src/pyrpmd/workflows/sampling.py")}


def environment_identity():
    import openmm._openmm as extension
    return {
        "python": platform.python_version(), "machine": platform.machine(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "openmm")},
        "openmm_python_extension_sha256": sha256(extension.__file__),
    }


def sample_ring_momenta(rng, masses, beads, temperature_K, *, remove_com):
    """Maxwell distribution at P*T; optionally project only total ring momentum."""
    p = rng.normal(size=(beads, len(masses), 3)) * np.sqrt(masses[None, :, None] * beads * KB_AU * temperature_K)
    if remove_com:
        p -= masses[None, :, None] * p.sum(axis=(0, 1)) / (beads * masses.sum())
    return p


class FullRPMDEngine:
    """All-copy physical/canonical and raw-state transfers, without contraction."""

    def __init__(self, config, box_angstrom):
        import openmm as mm
        from openmm import unit
        config.validate()
        self.config, self.mm, self.unit = config, mm, unit
        self.beads, self.molecules = config.beads, config.cells_per_axis**3
        self.masses = np.tile([O_MASS_DA, H_MASS_DA, H_MASS_DA], self.molecules) * AMU_TO_AU
        self.indices = np.arange(4 * self.molecules).reshape(-1, 4)[:, :3].ravel()
        self.system = make_system(molecules=self.molecules, box_angstrom=box_angstrom,
                                  cutoff_angstrom=config.cutoff_angstrom,
                                  ewald_tolerance=config.ewald_tolerance,
                                  dispersion_correction=config.dispersion_correction)
        self.integrator = mm.RPMDIntegrator(config.beads, config.temperature_K, 0., config.dt_fs / 1000)
        self.integrator.setApplyThermostat(False)
        self.integrator.setRandomNumberSeed(config.seed)
        selected = mm.Platform.getPlatformByName(config.platform)
        properties = ({"Threads": str(config.threads), "DeterministicForces": "true"}
                      if config.platform == "CPU" else dict(config.platform_properties))
        if not set(properties).issubset(set(selected.getPropertyNames())):
            raise ValueError("requested OpenMM properties are unavailable on this platform")
        self.context = mm.Context(self.system, self.integrator, selected, properties)
        tracked = {"Threads", "DeterministicForces", "Precision", "DeviceIndex", "UseCpuPme", "OpenCLPlatformIndex"}
        self.platform_identity = {"name": selected.getName(), "properties": {
            name: selected.getPropertyValue(self.context, name) for name in selected.getPropertyNames() if name in tracked}}
        if self.integrator.getContractions() or self.system.getNumConstraints() or self.integrator.getApplyThermostat():
            raise ValueError("full reference must have no RPC, bond constraints, or native thermostat")

    def set_momenta(self, p):
        p = np.asarray(p, dtype=float)
        if p.shape != (self.beads, len(self.masses), 3) or not np.isfinite(p).all():
            raise ValueError("ring momenta must be finite physical [bead,atom,3]")
        for bead in range(self.beads):
            velocities = np.zeros((4 * self.molecules, 3))
            velocities[self.indices] = p[bead] / self.masses[:, None] * VELOCITY_NM_PS
            self.integrator.setVelocities(bead, velocities)

    def set_state(self, q, p):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.beads, len(self.masses), 3) or not np.isfinite(q).all():
            raise ValueError("ring coordinates must be finite physical [bead,atom,3]")
        for bead in range(self.beads):
            positions = np.zeros((self.molecules, 4, 3))
            positions[:, :3] = q[bead].reshape(-1, 3, 3) * BOHR_NM
            gamma = PARAMETERS["gamma"]
            positions[:, 3] = gamma * positions[:, 0] + (1 - gamma) * (positions[:, 1] + positions[:, 2]) / 2
            self.integrator.setPositions(bead, positions.reshape(-1, 3))
        self.set_momenta(p)

    def raw_state(self, *, forces=False, energy=False):
        positions, velocities, all_forces, energies = [], [], [], []
        for bead in range(self.beads):
            state = self.integrator.getState(bead, getPositions=True, getVelocities=True,
                                            getForces=forces, getEnergy=energy)
            positions.append(np.asarray(state.getPositions(asNumpy=True) / self.unit.nanometer).copy())
            velocities.append(np.asarray(state.getVelocities(asNumpy=True) / (self.unit.nanometer / self.unit.picosecond)).copy())
            if forces:
                all_forces.append(np.asarray(state.getForces(asNumpy=True) / (self.unit.kilojoule_per_mole / self.unit.nanometer)).copy())
            if energy:
                energies.append(float(state.getPotentialEnergy() / self.unit.kilojoule_per_mole) / HARTREE_KJ_MOL)
        raw = {"engine_positions_nm": np.asarray(positions), "engine_velocities_nm_ps": np.asarray(velocities)}
        if forces:
            raw["engine_forces_kj_mol_nm"] = np.asarray(all_forces)
        if energy:
            raw["potential_bead_hartree"] = np.asarray(energies)
        if any(not np.isfinite(value).all() for value in raw.values()):
            raise FloatingPointError("nonfinite native full-RPMD state")
        return raw

    def physical_state(self, raw):
        return (raw["engine_positions_nm"][:, self.indices] / BOHR_NM,
                raw["engine_velocities_nm_ps"][:, self.indices] / VELOCITY_NM_PS * self.masses[None, :, None])

    def restore(self, checkpoint):
        expected_shape = (self.beads, 4 * self.molecules, 3)
        for name in ("engine_positions_nm", "engine_velocities_nm_ps", "engine_forces_kj_mol_nm"):
            if checkpoint[name].shape != expected_shape or not np.isfinite(checkpoint[name]).all():
                raise ValueError("invalid redundant all-copy checkpoint state")
        self.context.loadCheckpoint(checkpoint["openmm_checkpoint"].tobytes())
        # Context restores engine state; explicit copy transfers restore the
        # RPMD phase space. Never infer the latter from a successful blob load.
        for bead in range(self.beads):
            self.integrator.setPositions(bead, checkpoint["engine_positions_nm"][bead])
            self.integrator.setVelocities(bead, checkpoint["engine_velocities_nm_ps"][bead])
        restored = self.raw_state(forces=True)
        for name in ("engine_positions_nm", "engine_velocities_nm_ps"):
            if not np.array_equal(restored[name], checkpoint[name]):
                raise ValueError(f"all-copy raw checkpoint restoration mismatch: {name}")
        q, p = self.physical_state(restored)
        if not np.array_equal(q, checkpoint["q"]) or not np.array_equal(p, checkpoint["p"]):
            raise ValueError("physical/canonical checkpoint state disagrees with raw copies")
        previous = np.asarray(checkpoint["engine_forces_kj_mol_nm"])
        difference = restored["engine_forces_kj_mol_nm"] - previous
        rms_relative = float(np.sqrt(np.mean(difference**2)) / max(1., np.sqrt(np.mean(previous**2))))
        atol, rtol, rms_limit = ((1e-9, 1e-12, 1e-12) if self.config.platform == "Reference" else (.005, 2e-6, 1e-6))
        if (not np.allclose(restored["engine_forces_kj_mol_nm"], previous, atol=atol, rtol=rtol)
                or rms_relative > rms_limit):
            raise ValueError("recomputed full-RPMD checkpoint forces exceed the declared bound")
        return {"all_copy_positions_and_velocities_exact": True, "physical_q_and_p_exact": True,
                "force_max_difference_kj_mol_nm": float(np.max(abs(difference))),
                "force_RMS_relative_difference": rms_relative,
                "force_component_atol_kj_mol_nm": atol, "force_component_rtol": rtol,
                "force_RMS_relative_limit": rms_limit,
                "GPU_force_tolerance_validation_pending": self.config.platform not in ("Reference", "CPU"),
                "bitwise_future_trajectory_claimed": False}

    def close(self):
        self.context = None
        self.integrator = None

    def integrator_specification(self):
        """Explicit reconstruction contract: RPMD lacks an XML proxy in 8.6."""
        return {
            "schema": "pyrpmd.openmm-rpmd-integrator-specification.v1",
            "type": "RPMDIntegrator", "num_copies": int(self.integrator.getNumCopies()),
            "temperature_kelvin": float(self.integrator.getTemperature() / self.unit.kelvin),
            "friction_inverse_ps": float(self.integrator.getFriction() / (self.unit.picosecond**-1)),
            "step_size_ps": float(self.integrator.getStepSize() / self.unit.picosecond),
            "apply_native_thermostat": bool(self.integrator.getApplyThermostat()),
            "random_seed": int(self.integrator.getRandomNumberSeed()),
            "integration_force_groups": int(self.integrator.getIntegrationForceGroups()),
            "constraint_tolerance": float(self.integrator.getConstraintTolerance()),
            "contractions": {str(group): int(copies) for group, copies in self.integrator.getContractions().items()},
        }


def checkpoint_phase(config, step, active_segment):
    expected = -1 if step <= config.equilibration_steps else (step - config.equilibration_steps - 1) // config.segment_steps
    if not 0 <= step <= config.total_steps or active_segment != expected:
        raise ValueError("checkpoint step and active NVE segment are inconsistent")
    if step < config.equilibration_steps:
        return "preparation"
    if step == config.total_steps:
        return "complete"
    return "production-awaiting-momentum-refresh" if (step - config.equilibration_steps) % config.segment_steps == 0 else "production"


def observe(engine, config, step, segment, segment_step):
    raw = engine.raw_state(energy=True)
    q, p = engine.physical_state(raw)
    molecules = q.reshape(config.beads, engine.molecules, 3, 3)
    masses = engine.masses.reshape(engine.molecules, 3)
    molecular_mass = masses.sum(axis=1)
    centroid_q, centroid_p = q.mean(axis=0), p.mean(axis=0)
    com = np.sum(molecules.mean(axis=0) * masses[:, :, None], axis=1) / molecular_mass[:, None]
    velocity = centroid_p.reshape(engine.molecules, 3, 3).sum(axis=1) / molecular_mass[:, None]
    bonds = (molecules[:, :, 1:] - molecules[:, :, :1]) * BOHR_ANGSTROM
    lengths = np.linalg.norm(bonds, axis=-1)
    angles = np.degrees(np.arccos(np.clip(np.sum(bonds[:, :, 0] * bonds[:, :, 1], axis=-1) / lengths.prod(axis=-1), -1, 1)))
    kinetic = float(np.sum(p**2 / (2 * engine.masses[None, :, None])))
    centroid_kinetic = float(np.sum(centroid_p**2 / (2 * engine.masses[:, None])))
    projected = 3 if segment >= 0 else 0
    temperature = 2 * kinetic / ((3 * len(engine.masses) * config.beads - projected) * config.beads * KB_AU)
    centroid_temperature = 2 * centroid_kinetic / ((3 * len(engine.masses) - projected) * KB_AU)
    hydrogen = np.arange(len(engine.masses)) % 3 != 0
    rg = np.sqrt(np.mean(np.sum((q[:, hydrogen] - centroid_q[None, hydrogen])**2, axis=-1))) * BOHR_ANGSTROM
    total_energy = float(engine.integrator.getTotalEnergy() / engine.unit.kilojoule_per_mole) / HARTREE_KJ_MOL
    dtype = np.dtype(config.trajectory_precision)
    return {"step": step, "segment": segment, "segment_step": segment_step,
            "molecular_com_bohr": com.astype(dtype), "molecular_velocity_au": velocity.astype(dtype),
            "oxygen_centroid_bohr": centroid_q.reshape(engine.molecules, 3, 3)[:, 0].astype(dtype),
            "temperature_momenta_K": temperature, "temperature_centroid_K": centroid_temperature,
            "potential_per_molecule_hartree": float(raw["potential_bead_hartree"].mean() / engine.molecules),
            "hamiltonian_per_bead_hartree": total_energy / config.beads,
            "mean_OH_angstrom": float(lengths.mean()), "mean_HOH_degrees": float(angles.mean()),
            "hydrogen_radius_gyration_angstrom": float(rg)}


def verify_manifest(output, manifest):
    if manifest.get("schema") != SCHEMA or not {"model.xml", "integrator.json"}.issubset(manifest.get("artifacts", {})):
        raise ValueError("full-RPMD manifest requires its declared schema and serialized model/integrator")
    config = Config(**manifest["config"]).validate()
    if (not 0 <= manifest["committed_steps"] <= config.total_steps
            or manifest["complete"] != (manifest["committed_steps"] == config.total_steps)):
        raise ValueError("full-RPMD completion flag disagrees with its fixed stage")
    previous = -1
    for index, record in enumerate(manifest["chunks"]):
        if (record["file"] != f"chunk-{index:06d}.npz" or record["step"] <= previous
                or (index == 0 and record["step"] != 0)
                or (index > 0 and record["step"] - previous > config.checkpoint_steps)
                or sha256(output / record["file"]) != record["sha256"]):
            raise ValueError("full-RPMD committed checkpoint checksum/order mismatch")
        previous = record["step"]
    if not manifest["chunks"] or previous != manifest["committed_steps"]:
        raise ValueError("full-RPMD manifest endpoint differs from its committed checkpoint")
    for name, record in manifest["artifacts"].items():
        if Path(name).name != name or sha256(output / name) != record["sha256"]:
            raise ValueError("full-RPMD serialized engine checksum mismatch")


def run(output, config, *, resume=False, wall_seconds=None, max_steps=None):
    config.validate()
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("wall_seconds must be finite and positive")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    output = Path(output).resolve()
    source, environment = source_identity(), environment_identity()
    initial_sha = sha256(config.initial_checkpoint) if config.initial_checkpoint else None
    initial_q, box = initial_grid(cells_per_axis=config.cells_per_axis, density_g_cm3=config.density_g_cm3, seed=config.seed)
    started = time.perf_counter()
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if (manifest["config"] != asdict(config) or manifest["source_sha256"] != source
                or manifest["environment"] != environment or manifest["initial_checkpoint_sha256"] != initial_sha):
            raise ValueError("full-RPMD resume configuration/source/environment/initial-state identity mismatch")
        verify_manifest(output, manifest)
        if manifest["complete"]:
            return manifest
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("full-RPMD output must be empty/new; use --resume for a committed run")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": SCHEMA, "config": asdict(config), "source_sha256": source, "environment": environment,
            "initial_checkpoint_sha256": initial_sha, "box_angstrom": box,
            "created_utc": datetime.now(timezone.utc).isoformat(), "artifacts": {}, "chunks": [],
            "committed_steps": 0, "complete": False, "wall_seconds": 0., "process_cpu_seconds": 0.,
            "external_reference": True, "pyrpmd_production_trajectory": False,
             "literature_reproduced": False,
            "propagator": "OpenMM native RPMD, full PME at every bead; no RPC or MTS",
            "preparation": "external global geometric Andersen clock; mean interval exact on the discrete step grid",
            "momentum_distribution": "bead Maxwell at P*T; NVE refresh removes only total ring COM momentum",
            "checkpoint_contract": "Context blob plus explicit all-copy raw q/v, redundant canonical q/p/forces, RNG/clock/phase",
            "native_thermostat_enabled": False, "GPU_tested": False,
            "GPU_force_tolerance_validation_pending": config.platform not in ("Reference", "CPU"),
            "independence_unit": "independently prepared seed chain; adjacent NVE segments share a configuration chain",
        }
    engine = FullRPMDEngine(config, box)
    rng = np.random.default_rng(config.seed)
    clock = AndersenSchedule(config.mean_collision_steps, "geometric")
    try:
        if resume:
            if manifest["platform_identity"] != engine.platform_identity:
                raise ValueError("full-RPMD actual platform properties changed on resume")
            if json.loads((output / "integrator.json").read_text()) != engine.integrator_specification():
                raise ValueError("full-RPMD reconstructed integrator differs from the recorded specification")
            with np.load(output / manifest["chunks"][-1]["file"], allow_pickle=False) as saved:
                step, active_segment = int(saved["state_step"]), int(saved["active_segment"])
                phase = checkpoint_phase(config, step, active_segment)
                if step != manifest["committed_steps"] or str(saved["phase"]) != phase:
                    raise ValueError("full-RPMD checkpoint phase/manifest mismatch")
                clock.restore(saved, min(step, config.equilibration_steps))
                rng.bit_generator.state = json.loads(str(saved["rng_json"]))
                checked = engine.restore(saved)
                checked.update(step=step, chunk_sha256=manifest["chunks"][-1]["sha256"])
                manifest.setdefault("restart_validations", []).append(checked)
        else:
            (output / "model.xml").write_text(engine.mm.XmlSerializer.serialize(engine.system))
            atomic_json(output / "integrator.json", engine.integrator_specification())
            for name in ("model.xml", "integrator.json"):
                manifest["artifacts"][name] = {"sha256": sha256(output / name)}
            manifest["platform_identity"] = engine.platform_identity
            q = np.repeat(initial_q[None], config.beads, axis=0)
            if config.initial_checkpoint:
                with np.load(config.initial_checkpoint, allow_pickle=False) as saved:
                    if not np.isclose(float(saved["box_angstrom"]), box, atol=1e-10, rtol=0):
                        raise ValueError("initial geometry cell differs from declared density/masses")
                    saved_q = np.asarray(saved["q"])
                    if saved_q.shape == (1, len(engine.masses), 3):
                        q = np.repeat(saved_q, config.beads, axis=0)
                    elif saved_q.shape == q.shape:
                        q = saved_q.copy()
                    else:
                        raise ValueError("initial geometry must contain one copy or exactly the declared bead count")
            p = (sample_ring_momenta(rng, engine.masses, config.beads, config.temperature_K, remove_com=False)
                 if config.equilibration_steps else np.zeros_like(q))
            engine.set_state(q, p)
            if config.equilibration_steps:
                clock.refreshed(0, rng)
            step, active_segment = 0, -1

        def commit(observations, collision_steps, refreshed_segments):
            nonlocal last_wall, last_cpu
            raw = engine.raw_state(forces=True)
            q, p = engine.physical_state(raw)
            arrays = ({key: np.asarray([row[key] for row in observations]) for key in observations[0]}
                      if observations else {})
            filename = f"chunk-{len(manifest['chunks']):06d}.npz"
            atomic_npz(output / filename, **arrays, **raw, q=q, p=p, state_step=step,
                       state_time=step * config.dt_fs * FS_TO_AU, box_angstrom=box,
                       active_segment=active_segment, phase=checkpoint_phase(config, step, active_segment),
                       rng_json=json.dumps(rng.bit_generator.state, sort_keys=True), **clock.checkpoint_fields(),
                       collision_steps=np.asarray(collision_steps, dtype=np.int64),
                       refreshed_NVE_segments=np.asarray(refreshed_segments, dtype=np.int64),
                       openmm_checkpoint=np.frombuffer(engine.context.createCheckpoint(), dtype=np.uint8))
            if source_identity() != source:
                raise RuntimeError("full-RPMD source changed during propagation; chunk not committed")
            manifest["chunks"].append({"file": filename, "sha256": sha256(output / filename),
                                       "step": step, "observations": len(observations)})
            manifest["committed_steps"], manifest["complete"] = step, step == config.total_steps
            manifest["wall_seconds"] += time.perf_counter() - last_wall
            manifest["process_cpu_seconds"] += time.process_time() - last_cpu
            atomic_json(output / "manifest.json", manifest)
            last_wall, last_cpu = time.perf_counter(), time.process_time()

        last_wall, last_cpu = time.perf_counter(), time.process_time()
        if not resume:
            commit([], [0] if config.equilibration_steps else [], [])
        if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
            return manifest
        invocation_step = step
        while step < config.total_steps:
            stop = min(config.total_steps, step + config.checkpoint_steps)
            if max_steps is not None:
                stop = min(stop, invocation_step + max_steps)
            observations, collision_steps, refreshed_segments = [], [], []
            while step < stop:
                if step < config.equilibration_steps:
                    segment, segment_step = -1, step
                    if clock.due(step):
                        engine.set_momenta(sample_ring_momenta(rng, engine.masses, config.beads,
                                                             config.temperature_K, remove_com=False))
                        clock.refreshed(step, rng)
                        collision_steps.append(step)
                    boundary = min(config.equilibration_steps, clock.next_collision_step)
                    stride = config.equilibration_record_stride
                else:
                    segment, segment_step = divmod(step - config.equilibration_steps, config.segment_steps)
                    boundary = config.equilibration_steps + (segment + 1) * config.segment_steps
                    stride = config.record_stride
                    if segment_step == 0 and active_segment != segment:
                        engine.set_momenta(sample_ring_momenta(rng, engine.masses, config.beads,
                                                             config.temperature_K, remove_com=True))
                        active_segment = segment
                        refreshed_segments.append(segment)
                        observations.append(observe(engine, config, step, segment, 0))
                advance = min(stop - step, boundary - step, stride - segment_step % stride, config.max_native_block_steps)
                if advance <= 0:
                    raise RuntimeError("full-RPMD clock failed to schedule a positive native block")
                engine.integrator.step(advance)
                step += advance
                segment_step += advance
                if segment_step % stride == 0:
                    observations.append(observe(engine, config, step, segment, segment_step))
                if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
                    break
            commit(observations, collision_steps, refreshed_segments)
            print(json.dumps({"output": str(output), "step": step, "time_ps": step * config.dt_fs / 1000,
                              "total_steps": config.total_steps, "complete": manifest["complete"],
                              "wall_seconds": manifest["wall_seconds"], "process_cpu_seconds": manifest["process_cpu_seconds"],
                              "external_reference": True, "GPU_tested": False}), flush=True)
            if max_steps is not None and step - invocation_step >= max_steps:
                break
            if wall_seconds is not None and time.perf_counter() - started >= wall_seconds:
                break
        return manifest
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--describe", action="store_true", help="print the explicit job without constructing a Context or starting computation")
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text())).validate()
    if args.describe:
        print(json.dumps({"config": asdict(config), "source_sha256": source_identity(),
                          "preparation_ps": config.equilibration_steps * config.dt_fs / 1000,
                          "production_ps": config.segments * config.segment_steps * config.dt_fs / 1000,
                          "total_steps": config.total_steps, "external_reference": True, "GPU_tested": False}, indent=2))
    else:
        if args.output is None:
            parser.error("--output is required unless --describe is used")
        run(args.output, config, resume=args.resume, wall_seconds=args.wall_seconds, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
