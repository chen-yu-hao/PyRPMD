#!/usr/bin/env python3
"""Read-only audit of OpenMM CPU checkpoint and canonical-velocity conversion.

The original campaign chunks store canonical p and OpenMM's binary checkpoint.
This audit also measures a fresh checkpoint round trip with independently held
raw positions, half-step velocities, forces, and canonical p.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, sha256
from benchmarks.water.native_reference import (
    AMU_TO_AU, BOHR_NM, Config, VELOCITY_NM_PS, create_engine,
)


def raw_state(context, masses):
    from openmm import unit
    state = context.getState(getPositions=True, getVelocities=True, getForces=True)
    x = np.asarray(state.getPositions(asNumpy=True) / unit.nanometer).copy()
    v = np.asarray(state.getVelocities(asNumpy=True) / (unit.nanometer / unit.picosecond)).copy()
    f = np.asarray(state.getForces(asNumpy=True) / (unit.kilojoule_per_mole / unit.nanometer)).copy()
    physical = np.arange(len(x)).reshape(-1, 4)[:, :3].ravel()
    dt = float(context.getIntegrator().getStepSize() / unit.picosecond)
    q = x[physical][None] / BOHR_NM
    p = ((v[physical] + .5 * dt * f[physical] / (masses / AMU_TO_AU)[:, None])
         / VELOCITY_NM_PS * masses[:, None])[None]
    return {"positions_nm": x, "half_step_velocities_nm_ps": v,
            "forces_kj_mol_nm": f, "physical": physical, "q": q, "p": p}


def virtual_site_residual(system, positions):
    residual = []
    for index in range(system.getNumParticles()):
        if system.isVirtualSite(index):
            site = system.getVirtualSite(index)
            expected = sum(site.getWeight(j) * positions[site.getParticle(j)] for j in range(3))
            residual.append(positions[index] - expected)
    return float(np.max(np.abs(residual)))


def audit_run(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    config = Config(**manifest["config"])
    entry = manifest["chunks"][-1]
    if sha256(path / entry["file"]) != entry["sha256"]:
        raise ValueError("input checkpoint checksum mismatch")
    with np.load(path / entry["file"], allow_pickle=False) as saved:
        q, p = saved["q"].copy(), saved["p"].copy()
        checkpoint = saved["openmm_checkpoint"].tobytes()
    system, integrator, context, masses = create_engine(config, manifest["box_angstrom"])
    system2, integrator2, context2, masses2 = create_engine(config, manifest["box_angstrom"])
    try:
        context.loadCheckpoint(checkpoint)
        first = raw_state(context, masses)
        physical = first["physical"]
        # Historical chunks have no separate raw velocity/force arrays. The
        # pre-save force is reconstructed from the saved canonical p and the
        # binary checkpoint's half-step velocity, rather than called measured.
        original_force_inferred = ((p[0] / masses[:, None] * VELOCITY_NM_PS
                                    - first["half_step_velocities_nm_ps"][physical])
                                   * (masses / AMU_TO_AU)[:, None] * 2 / (config.dt_fs / 1000))
        row = {"run": path.name, "seed": config.seed,
               "input_manifest_sha256": sha256(path / "manifest.json"),
               "input_chunk": entry["file"], "input_chunk_sha256": entry["sha256"],
               "input_source_sha256": manifest["source_sha256"],
               "state_time_ps": manifest["committed_steps"] * config.dt_fs / 1000,
               "positions_exact_from_historical_canonical_q": bool(np.array_equal(q, first["q"])),
               "historical_canonical_p_max_difference_au": float(np.max(abs(first["p"] - p))),
               "historical_canonical_p_RMS_difference_au": float(np.sqrt(np.mean((first["p"] - p)**2))),
               "historical_kinetic_energy_difference_hartree": float(np.sum((first["p"]**2-p**2) / (2*masses[None,:,None]))),
               "historical_force_difference_inferred_max_kj_mol_nm": float(np.max(abs(first["forces_kj_mol_nm"][physical] - original_force_inferred))),
               "historical_force_note": "Pre-save force inferred from canonical p and binary checkpoint half-step velocity; not a separately stored force measurement."}

        # New, independently measured before/after pair at a realistic liquid
        # state: integration leaves the CPU force/neighbor cache populated.
        integrator.step(10)
        before = raw_state(context, masses)
        context2.loadCheckpoint(context.createCheckpoint())
        after = raw_state(context2, masses2)
        delta_force = after["forces_kj_mol_nm"][physical] - before["forces_kj_mol_nm"][physical]
        predicted_delta_p = .5 * config.dt_fs / 1000 * AMU_TO_AU / VELOCITY_NM_PS * delta_force
        row["measured_round_trip"] = {
            "advance_before_checkpoint_steps": 10,
            "all_positions_including_virtual_sites_exact": bool(np.array_equal(before["positions_nm"], after["positions_nm"])),
            "all_half_step_velocities_exact": bool(np.array_equal(before["half_step_velocities_nm_ps"], after["half_step_velocities_nm_ps"])),
            "physical_q_exact": bool(np.array_equal(before["q"], after["q"])),
            "before_virtual_site_residual_nm": virtual_site_residual(system, before["positions_nm"]),
            "after_virtual_site_residual_nm": virtual_site_residual(system2, after["positions_nm"]),
            "force_max_difference_kj_mol_nm": float(np.max(abs(delta_force))),
            "force_RMS_difference_kj_mol_nm": float(np.sqrt(np.mean(delta_force**2))),
            "force_relative_RMS_difference": float(np.sqrt(np.mean(delta_force**2) / np.mean(before["forces_kj_mol_nm"][physical]**2))),
            "canonical_p_max_difference_au": float(np.max(abs(after["p"] - before["p"]))),
            "canonical_p_delta_minus_force_conversion_max_au": float(np.max(abs(after["p"][0] - before["p"][0] - predicted_delta_p))),
        }
        for _ in range(20):
            integrator.step(1)
            integrator2.step(1)
        a, b = raw_state(context, masses), raw_state(context2, masses2)
        row["continued_20_step_comparison"] = {
            "max_q_difference_bohr": float(np.max(abs(a["q"]-b["q"]))),
            "max_p_difference_au": float(np.max(abs(a["p"]-b["p"]))),
            "interpretation": "CPU checkpoint continuation preserves the raw engine state; rebuilt force caches need not give bitwise identical later trajectories."}
        return row
    finally:
        del context, integrator, context2, integrator2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "pyrpmd.water-native-checkpoint-audit.v1",
              "read_only": True, "platform": "CPU", "threads": 1,
              "source_sha256": {name: sha256(ROOT / name) for name in (
                  "benchmarks/water/checkpoint_validation.py", "benchmarks/water/native_reference.py")},
              "runs": [audit_run(path) for path in args.runs]}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
