#!/usr/bin/env python3
"""Verify saved water-pilot bytes and recompute diffusion/spectrum diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pyrpmd.analysis.diffusion import analyze_diffusion
from pyrpmd.analysis.spectroscopy import relative_ir_spectrum
from pyrpmd.constants import KB_AU


def replay(directory):
    directory = Path(directory).resolve()
    report = json.loads((directory / "report.json").read_text())
    if report.get("schema") != "pyrpmd.qtip4pf-pilot.v1":
        raise ValueError("unsupported water pilot schema")
    required = {"trajectory.npz", "model.xml", "diffusion-diagnostic.json", "spectrum-diagnostic.npz"}
    if set(report.get("artifacts", {})) != required:
        raise ValueError("pilot must identify exactly its four recorded artifacts")
    for name, spec in report["artifacts"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("pilot artifact is outside the evidence directory")
        content = path.read_bytes()
        if len(content) != spec["bytes"] or hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise ValueError(f"pilot artifact hash/size mismatch: {name}")
    if report["provider"]["model_sha256"] != report["artifacts"]["model.xml"]["sha256"]:
        raise ValueError("model identity differs from recorded XML bytes")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as saved:
        times = saved["times_au"]
        dipoles = saved["bead_dipoles_e_bohr"]
        energy = saved["ring_polymer_hamiltonian_hartree"]
        analysis = analyze_diffusion(saved["molecular_centroid_positions_bohr"], times,
                                    velocities=saved["molecular_centroid_velocities_au"], bootstrap=0)
        spectrum = relative_ir_spectrum(dipoles, float(times[1] - times[0]),
            beta=1 / (KB_AU * report["conditions"]["temperature_K"]), normalize=False, window="hann")
    with np.load(directory / "spectrum-diagnostic.npz", allow_pickle=False) as saved:
        np.testing.assert_allclose(spectrum["wavenumber_cm1"], saved["wavenumber_cm1"], rtol=1e-13)
        np.testing.assert_allclose(spectrum["relative_ir"], saved["relative_ir"], rtol=1e-12, atol=1e-20)
    old = json.loads((directory / "diffusion-diagnostic.json").read_text())
    if analysis["status"] != old["status"]:
        raise AssertionError("diffusion diagnostic status changed")
    for key in ("diffusion_au", "diffusion_m2_s", "diffusion_tensor_au", "log_msd_slope"):
        np.testing.assert_allclose(analysis["raw_fit"][key], old["raw_fit"][key], rtol=1e-12, atol=1e-20)
    drift = float(np.max(abs(energy - energy[0])) / report["conditions"]["beads"])
    np.testing.assert_allclose(drift, report["max_energy_drift_per_bead_hartree"], rtol=1e-13)
    if report["physical_converged"] is not False or report["literature_reproduced"] is not False:
        raise ValueError("a software pilot cannot assert physical/literature validation")
    return {"artifact_integrity_passed": True, "estimator_replay_passed": True,
            "diffusion_status": analysis["status"], "physical_converged": False,
            "literature_reproduced": False, "max_energy_drift_per_bead_hartree": drift,
            "scope": "Local saved-array estimator replay; no new dynamics or physical convergence claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.directory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
