#!/usr/bin/env python3
"""Checkpointed, fixed-cell q-TIP4P/F sampling and NVE diffusion campaign.

This is a benchmark driver, not a convergence certificate. Every committed
chunk contains its endpoint state and RNG; a manifest atomically commits the
chunk. Resume validates model, source, configuration, and all committed bytes.
Full forces are evaluated on every bead (no contraction or multiple stepping).
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.qtip4pf import (
    D_MASS_DA, H_MASS_DA, O_MASS_DA, initial_grid, make_system,
)
from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, BOHR_ANGSTROM, FS_TO_AU, KB_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.base import PESResult, coordinates, validate_result
from pyrpmd.potentials.openmm import OpenMMPotential


@dataclass(frozen=True)
class Config:
    cells_per_axis: int = 6
    beads: int = 1
    isotope: str = "H2O"
    temperature_K: float = 298.0
    density_g_cm3: float = 0.997
    dt_fs: float = 0.25
    cutoff_angstrom: float = 9.0
    ewald_tolerance: float = 1e-5
    dispersion_correction: bool = False
    equilibration_steps: int = 80000
    segment_steps: int = 16000
    segments: int = 2
    record_stride: int = 20
    checkpoint_steps: int = 1000
    centroid_friction_time_fs: float = 100.0
    seed: int = 541
    platform: str = "CPU"
    threads: int = 2
    initial_checkpoint: str | None = None

    def validate(self):
        for name in ("cells_per_axis", "beads", "segment_steps", "segments", "record_stride",
                     "checkpoint_steps", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("equilibration_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("temperature_K", "density_g_cm3", "dt_fs", "cutoff_angstrom",
                     "ewald_tolerance", "centroid_friction_time_fs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.ewald_tolerance >= 1 or self.isotope not in ("H2O", "D2O"):
            raise ValueError("invalid Ewald tolerance or isotope")
        if self.platform not in ("CPU", "Reference"):
            raise ValueError("campaign explicitly supports CPU or Reference")
        if not isinstance(self.dispersion_correction, bool):
            raise ValueError("dispersion_correction must be boolean")
        if self.segment_steps % self.record_stride or self.equilibration_steps % self.record_stride:
            raise ValueError("record_stride must divide equilibration_steps and segment_steps")
        return self

    @property
    def total_steps(self):
        return self.equilibration_steps + self.segments * self.segment_steps


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths = ["benchmarks/water/campaign.py", "benchmarks/water/qtip4pf.py",
             "src/pyrpmd/potentials/openmm.py", "src/pyrpmd/potentials/base.py",
             "src/pyrpmd/dynamics.py", "src/pyrpmd/constants.py", "src/pyrpmd/geometry.py"]
    # Native code participates in the integration map, not only Python.
    paths += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "native").rglob("*.cpp"))
    paths += sorted(str(p.relative_to(ROOT)) for p in (ROOT / "native").rglob("*.h"))
    return {p: sha256(ROOT / p) for p in paths}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class FixedWaterCache:
    """One exact-coordinate entry for this immutable, fixed-cell Hamiltonian.

    Reuses the final force of one accepted step at the next initial kick. Owns
    immutable arrays and the key; caller mutations never cause stale hits.
    Used only by this driver, whose model and cell cannot change during a run.
    """

    def __init__(self, provider):
        self.provider = provider
        self._q = self._result = None
        self.evaluations = self.hits = 0

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def evaluate(self, q):
        q = coordinates(q)
        if self._q is not None and np.array_equal(q, self._q):
            self.hits += 1
            return self._result
        self._q = self._result = None
        result = validate_result(self.provider.evaluate(q), q.shape)
        owned = PESResult(result.energy.copy(), result.forces.copy(), result.status.copy(), result.messages)
        for array in (owned.energy, owned.forces, owned.status):
            array.flags.writeable = False
        self._q, self._result = q.copy(), owned
        self.evaluations += 1
        return owned

    def set_cell(self, cell):
        raise ValueError("FixedWaterCache belongs to an immutable fixed-cell campaign")


def create_dynamics(config, box):
    hydrogen = H_MASS_DA if config.isotope == "H2O" else D_MASS_DA
    n = config.cells_per_axis ** 3
    model = make_system(molecules=n, box_angstrom=box, cutoff_angstrom=config.cutoff_angstrom,
                        ewald_tolerance=config.ewald_tolerance,
                        dispersion_correction=config.dispersion_correction, hydrogen_mass_da=hydrogen)
    properties = {"Threads": str(config.threads), "DeterministicForces": "true"} if config.platform == "CPU" else {}
    provider = OpenMMPotential(model, ["O", "H", "H"] * n,
                              cell=np.eye(3) * box * ANGSTROM_TO_AU, pbc=[True] * 3,
                              platform=config.platform, properties=properties)
    masses = np.tile([O_MASS_DA, hydrogen, hydrogen], n) * AMU_TO_AU
    cache = FixedWaterCache(provider)
    dynamics = Dynamics(cache, masses, 1 / (KB_AU * config.temperature_K), config.beads,
                        config.dt_fs * FS_TO_AU, seed=config.seed, native=True,
                        threads=1, loop_backend="auto", integrator="normal-mode-bab")
    return provider, dynamics


def observe(dynamics, state, config, segment, segment_step):
    n = config.cells_per_axis ** 3
    masses = dynamics.masses.reshape(n, 3)
    weights = masses / masses.sum(axis=1, keepdims=True)
    q = state.q.reshape(config.beads, n, 3, 3)
    p = state.p.reshape(q.shape)
    centroid = q.mean(axis=0)
    velocity = (p / masses[None, :, :, None]).mean(axis=0)
    oh = (q[:, :, 1:] - q[:, :, :1]) * BOHR_ANGSTROM
    lengths = np.linalg.norm(oh, axis=-1)
    cosine = np.sum(oh[:, :, 0] * oh[:, :, 1], axis=-1) / lengths.prod(axis=-1)
    angles = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
    potential = dynamics.potential.evaluate(state.q)
    kinetic = np.sum(state.p**2 / (2 * dynamics.masses[None, :, None]))
    spring = 0.5 * (config.beads / dynamics.beta)**2 * np.sum(
        dynamics.masses[None, :, None] * (state.q - np.roll(state.q, 1, axis=0))**2)
    total = kinetic + spring + potential.energy.sum()
    # This is a thermostat diagnostic of fictitious momenta, not quantum K.
    temperature = 2 * kinetic / (3 * len(dynamics.masses) * config.beads**2 * KB_AU)
    centroid_kinetic = np.sum(state.p.mean(axis=0)**2 / (2 * dynamics.masses[:, None]))
    centroid_temperature = 2 * centroid_kinetic / ((3 * len(dynamics.masses) - (3 if segment >= 0 else 0)) * KB_AU)
    return {
        "step": state.step, "segment": segment, "segment_step": segment_step,
        "molecular_com_bohr": np.sum(centroid * weights[:, :, None], axis=1),
        "molecular_velocity_au": np.sum(velocity * weights[:, :, None], axis=1),
        "oxygen_centroid_bohr": centroid[:, 0],
        "temperature_momenta_K": temperature,
        "temperature_centroid_K": centroid_temperature,
        "potential_per_molecule_hartree": float(potential.energy.mean() / n),
        "hamiltonian_per_bead_hartree": float(total / config.beads),
        "mean_OH_angstrom": float(lengths.mean()),
        "mean_HOH_degrees": float(angles.mean()),
        "hydrogen_radius_gyration_angstrom": float(np.sqrt(np.mean(
            np.sum((q[:, :, 1:] - centroid[None, :, 1:])**2, axis=-1))) * BOHR_ANGSTROM),
    }


def verify_manifest(output, manifest):
    for record in manifest["chunks"]:
        path = output / record["file"]
        if path.name != record["file"] or sha256(path) != record["sha256"]:
            raise ValueError(f"committed chunk checksum mismatch: {record['file']}")
    if sha256(output / "model.xml") != manifest["model_xml_sha256"]:
        raise ValueError("serialized model checksum mismatch")


def run(output, config, *, resume=False, wall_seconds=None, max_steps=None):
    """Run until complete or a checkpoint after the given invocation budget."""
    config.validate()
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("wall_seconds must be positive")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    output = Path(output).resolve()
    start = time.perf_counter()
    identity = source_identity()
    hydrogen = H_MASS_DA if config.isotope == "H2O" else D_MASS_DA
    q0, box = initial_grid(cells_per_axis=config.cells_per_axis, density_g_cm3=config.density_g_cm3,
                           seed=config.seed, hydrogen_mass_da=hydrogen)
    initial_hash = sha256(config.initial_checkpoint) if config.initial_checkpoint else None
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["config"] != asdict(config) or manifest["source_sha256"] != identity:
            raise ValueError("resume configuration/source identity mismatch")
        if manifest["initial_checkpoint_sha256"] != initial_hash:
            raise ValueError("resume initial checkpoint checksum mismatch")
        verify_manifest(output, manifest)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be new or empty; use --resume for committed runs")
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": "pyrpmd.water-campaign.v1", "config": asdict(config),
                    "source_sha256": identity, "initial_checkpoint_sha256": initial_hash,
                    "box_angstrom": box, "chunks": [], "complete": False,
                    "physical_converged": False, "literature_reproduced": False,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "platform": platform.platform(),
                    "dependencies": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "openmm")},
                    "elapsed_seconds": 0.0}
        try:
            manifest["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["git_revision"] = None
    provider, dynamics = create_dynamics(config, box)
    try:
        if resume and manifest["provider_model_sha256"] != provider.model_sha256:
            raise ValueError("resume provider model mismatch")
        if not resume:
            from openmm import XmlSerializer
            (output / "model.xml").write_text(XmlSerializer.serialize(provider.system), encoding="utf-8")
            manifest["provider"] = provider.metadata
            manifest["provider_model_sha256"] = provider.model_sha256
            manifest["model_xml_sha256"] = sha256(output / "model.xml")
        if resume and manifest["chunks"]:
            with np.load(output / manifest["chunks"][-1]["file"], allow_pickle=False) as data:
                state = State(data["q"], data["p"], int(data["state_step"]), float(data["state_time"]))
                dynamics.rng.bit_generator.state = json.loads(str(data["rng_json"]))
        else:
            if config.initial_checkpoint:
                with np.load(config.initial_checkpoint, allow_pickle=False) as data:
                    saved_box = float(data["box_angstrom"])
                    if not np.isclose(saved_box, box, atol=1e-10, rtol=0):
                        raise ValueError("initial checkpoint box does not match requested density/isotope")
                    initial = np.asarray(data["q"])
                    if initial.shape[1:] != q0.shape:
                        raise ValueError("initial checkpoint atom count mismatch")
                    q0 = initial.mean(axis=0)
            state = State(np.repeat(q0[None], config.beads, axis=0), dynamics.sample_momenta())
        invocation_start_step = state.step
        if resume and manifest["complete"]:
            return manifest
        while state.step < config.total_steps:
            observations = []
            stop = min(config.total_steps, state.step + config.checkpoint_steps)
            if max_steps is not None:
                stop = min(stop, invocation_start_step + max_steps)
            for _ in range(state.step, stop):
                if state.step < config.equilibration_steps:
                    dynamics.sample_step(state, gamma_centroid=1 / (config.centroid_friction_time_fs * FS_TO_AU))
                    segment, segment_step = -1, state.step
                else:
                    index = state.step - config.equilibration_steps
                    segment, segment_step = divmod(index, config.segment_steps)
                    if segment_step == 0:
                        # Independent Maxwell momenta, sequential canonical configurations.
                        state.p[:] = dynamics.sample_momenta()
                        velocity_com = state.p.mean(axis=0).sum(axis=0) / dynamics.masses.sum()
                        state.p -= dynamics.masses[None, :, None] * velocity_com
                        observations.append(observe(dynamics, state, config, segment, 0))
                    dynamics.step(state)
                    segment_step += 1
                if state.step % config.record_stride == 0:
                    observations.append(observe(dynamics, state, config, segment, segment_step))
            if not np.isfinite(state.q).all() or not np.isfinite(state.p).all():
                raise FloatingPointError("nonfinite state; incomplete chunk was not committed")
            arrays = {key: np.asarray([row[key] for row in observations]) for key in observations[0]} if observations else {}
            name = f"chunk-{len(manifest['chunks']):06d}.npz"
            atomic_npz(output / name, **arrays, q=state.q, p=state.p,
                       state_step=state.step, state_time=state.time, box_angstrom=box,
                       rng_json=json.dumps(dynamics.rng.bit_generator.state, sort_keys=True))
            manifest["chunks"].append({"file": name, "sha256": sha256(output / name),
                                       "step": state.step, "observations": len(observations)})
            if source_identity() != identity:
                raise RuntimeError("propagation source changed; new chunk not committed to manifest")
            manifest["committed_steps"] = state.step
            manifest["complete"] = state.step == config.total_steps
            elapsed = time.perf_counter() - start
            manifest["elapsed_seconds"] += elapsed
            manifest["last_invocation"] = {"evaluations": dynamics.potential.evaluations,
                                           "cache_hits": dynamics.potential.hits}
            atomic_json(output / "manifest.json", manifest)
            print(json.dumps({"output": str(output), "step": state.step, "total_steps": config.total_steps,
                              "elapsed_seconds": manifest["elapsed_seconds"], "complete": manifest["complete"]}), flush=True)
            start = time.perf_counter()
            # Wall budget includes all committed chunks within this invocation.
            if max_steps is not None and state.step - invocation_start_step >= max_steps:
                break
            if wall_seconds is not None:
                wall_seconds -= elapsed
                if wall_seconds <= 0:
                    break
        return manifest
    finally:
        provider.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.config:
        config = Config(**json.loads(args.config.read_text()))
    elif args.resume:
        config = Config(**json.loads((args.output / "manifest.json").read_text())["config"])
    else:
        config = Config()
    run(args.output, config, resume=args.resume, wall_seconds=args.wall_seconds, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
