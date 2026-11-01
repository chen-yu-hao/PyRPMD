"""Replay the predeclared, excluded-geometry static EAM4 checks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .build_eam4 import sha256_file
from .eam4 import EAM4StaticReference

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("eam4_static_protocol.json")


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)*np.cos(angle) + (1-np.cos(angle))*np.outer(axis, axis) + np.sin(angle)*cross


def finite_difference(reference, q, mobile, lengths, periodic, steps, tolerance):
    _, gradient = reference.evaluate(q, mobile_ni=mobile, lengths_angstrom=lengths, periodic_xy=periodic)
    records = []
    for h in steps:
        finite = np.zeros_like(gradient[:mobile+1])
        for i in range(mobile+1):
            for j in range(3):
                plus = q.copy()
                minus = q.copy()
                plus[i, j] += h
                minus[i, j] -= h
                ep, _ = reference.evaluate(plus, mobile_ni=mobile, lengths_angstrom=lengths,
                                           periodic_xy=periodic, energy_only=True)
                em, _ = reference.evaluate(minus, mobile_ni=mobile, lengths_angstrom=lengths,
                                           periodic_xy=periodic, energy_only=True)
                finite[i, j] = (ep-em)/(2*h)
        error = np.abs(finite-gradient[:mobile+1])
        bound = tolerance["finite_difference_atol_eV_per_angstrom"] + tolerance["finite_difference_rtol"]*np.abs(gradient[:mobile+1])
        records.append({"step_angstrom": h, "max_abs_error_eV_per_angstrom": float(np.max(error)),
                        "max_error_over_predeclared_bound": float(np.max(error/bound)),
                        "active_coordinates": int(error.size), "passed": bool(np.all(error <= bound))})
    return records


def verify(library: Path) -> dict:
    started = time.monotonic()
    cpu_started = time.process_time()
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    reference = EAM4StaticReference(library)
    q = np.asarray(protocol["q_angstrom"], dtype=float)
    lengths = np.asarray(protocol["lengths_angstrom"], dtype=float)
    tol = protocol["tolerances"]
    rot = rotation(protocol["rotation_axis"], protocol["rotation_angle_rad"])
    shift = np.asarray(protocol["translation_angstrom"])
    _, all_mobile_gradient = reference.evaluate(q, mobile_ni=len(q)-1, lengths_angstrom=lengths)
    cases = []
    for case in protocol["cases"]:
        mobile, periodic = case["mobile_ni"], case["periodic_xy"]
        options = {"mobile_ni": mobile, "lengths_angstrom": lengths, "periodic_xy": periodic}
        energy, gradient = reference.evaluate(q, **options)
        energy_only, _ = reference.evaluate(q, energy_only=True, **options)
        translated_energy, translated_gradient = reference.evaluate(q+shift, **options)
        checks = {
            "energy_route_agreement": abs(energy-energy_only) <= tol["energy_route_agreement_eV"],
            "translation_energy": abs(energy-translated_energy) <= tol["symmetry_energy_eV"],
            "translation_gradient": float(np.max(np.abs(gradient-translated_gradient))) <= tol["symmetry_gradient_eV_per_angstrom"],
            "upstream_fixed_rows_zero": bool(np.all(gradient[mobile+1:] == 0.0)),
        }
        row = {**case, "energy_eV": energy, "gradient_eV_per_angstrom": gradient.tolist(),
               "energy_route_error_eV": abs(energy-energy_only),
               "translation_energy_error_eV": abs(energy-translated_energy),
               "translation_gradient_error_eV_per_angstrom": float(np.max(np.abs(gradient-translated_gradient)))}
        if periodic:
            wrapped = q.copy()
            wrapped[0, 0] += lengths[0]
            wrapped[3, 1] -= lengths[1]
            ep, gp = reference.evaluate(wrapped, **options)
            row["lattice_shift_energy_error_eV"] = abs(energy-ep)
            row["lattice_shift_gradient_error_eV_per_angstrom"] = float(np.max(np.abs(gradient-gp)))
            checks["lattice_shift_energy"] = abs(energy-ep) <= tol["symmetry_energy_eV"]
            checks["lattice_shift_gradient"] = row["lattice_shift_gradient_error_eV_per_angstrom"] <= tol["symmetry_gradient_eV_per_angstrom"]
        else:
            er, gr = reference.evaluate(q @ rot.T, **options)
            row["rotation_energy_error_eV"] = abs(energy-er)
            row["rotation_gradient_error_eV_per_angstrom"] = float(np.max(np.abs(gr-gradient @ rot.T)))
            checks["rotation_energy"] = abs(energy-er) <= tol["symmetry_energy_eV"]
            checks["rotation_gradient"] = row["rotation_gradient_error_eV_per_angstrom"] <= tol["symmetry_gradient_eV_per_angstrom"]
            row["active_partition_gradient_error_eV_per_angstrom"] = float(np.max(np.abs(gradient[:mobile+1]-all_mobile_gradient[:mobile+1])))
            checks["active_partition_gradient"] = row["active_partition_gradient_error_eV_per_angstrom"] <= tol["partition_active_gradient_eV_per_angstrom"]
        if mobile == len(q)-1:
            row["net_gradient_eV_per_angstrom"] = np.sum(gradient, axis=0).tolist()
            checks["all_mobile_net_gradient"] = float(np.max(np.abs(np.sum(gradient, axis=0)))) <= tol["net_gradient_eV_per_angstrom"]
        row["finite_difference"] = finite_difference(reference, q, mobile, lengths, periodic,
            protocol["finite_difference_steps_angstrom"], tol)
        checks["active_finite_differences"] = all(item["passed"] for item in row["finite_difference"])
        row["checks"] = {key: bool(value) for key, value in checks.items()}
        row["passed"] = all(row["checks"].values())
        cases.append(row)
    cutoff = []
    for distance in protocol["cutoff_probe_H_Ni_distances_angstrom"]:
        pair = np.array([[0., 0., 0.], [distance, 0., 0.]])
        energy, gradient = reference.evaluate(pair, mobile_ni=1)
        fd = finite_difference(reference, pair, 1, lengths, False,
                               protocol["finite_difference_steps_angstrom"], tol)
        cutoff.append({"H_Ni_distance_angstrom": distance, "energy_eV": energy,
                       "gradient_eV_per_angstrom": gradient.tolist(), "finite_difference": fd,
                       "passed": all(item["passed"] for item in fd)})
    sources = [Path(__file__), Path(__file__).with_name("eam4.py"), Path(__file__).with_name("build_eam4.py"),
               Path(__file__).with_name("eam4_bridge.f90"), PROTOCOL]
    return {
        "schema": "pyrpmd.materials-reference.eam4-static-validation.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": protocol["purpose"], "paper_reproduction": False, "diffusion_computed": False,
        "mixed_bead_or_dynamical_validation": False,
        "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": hashlib.sha256(protocol_bytes).hexdigest()},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "source_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}},
        "build_receipt": reference.build_receipt,
        "build_receipt_sha256": sha256_file(library.with_name("build.json")),
        "cases": cases, "cutoff_probes": cutoff,
        "all_static_checks_passed": all(row["passed"] for row in cases+cutoff),
        "wall_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-cpu_started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior validation receipts; choose a new output path")
    result = verify(args.library.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("all_static_checks_passed", "paper_reproduction", "wall_seconds", "cpu_seconds")}, indent=2))
    if not result["all_static_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
