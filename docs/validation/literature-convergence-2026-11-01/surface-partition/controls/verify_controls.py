"""Verify complete equilibrium-control evidence and immutable primary parents.

Integrity success is distinct from a scientific equivalence gate: a complete,
correctly retained experiment can legitimately fail its registered criterion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.surface_partition_controls import expected_settings
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.checkpoint import load_checkpoint


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return {"sha256": sha(path), "bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[-1]})
    directory, base = args.directory, args.directory.parent
    protocol = json.loads((directory/"protocol.json").read_text())
    comparison = json.loads((directory/"comparison.json").read_text())
    require(comparison["complete"], "control campaign is incomplete")
    require(comparison["protocol_sha256"] == sha(directory/"protocol.json"), "comparison/protocol mismatch")
    sources = dict(protocol["source_identity"]["source_sha256"])
    sources["benchmarks/h3/surface_partition_controls.py"] = protocol["controls_source_sha256"]
    for name, expected in sources.items():
        require(sha(ROOT/name) == expected, f"frozen source changed: {name}")
    provider = BKMP2Potential()
    require(provider.metadata == protocol["source_identity"]["model"], "registered PES model/binary changed")
    require(sha(base/"primary-protocol.json") == protocol["primary_protocol_sha256"], "primary protocol changed")
    require(sha(base/"primary-summary.json") == protocol["primary_summary_sha256"], "primary summary changed")
    require(sha(base.parent/"h3-quantum/diatom-transfer.json") == protocol["reactant_reference_sha256"],
            "exact reactant reference changed")

    primary = json.loads((base/"primary-manifest.json").read_text())
    require(primary["registered_protocol_sha256"] == protocol["primary_protocol_sha256"],
            "primary manifest/protocol mismatch")
    require(primary["complete_summary_sha256"] == protocol["primary_summary_sha256"],
            "primary manifest/summary mismatch")
    primary_files = 0
    for cohort in primary["cohorts"]:
        for name, expected in cohort["files"].items():
            require(record(ROOT/cohort["directory"]/name) == expected,
                    f"immutable primary file changed: {cohort['directory']}/{name}")
            primary_files += 1

    cohorts = []
    file_names = ("checkpoint.npz", "settings.json", "report.json", "progress.json",
                  "reference.json", "reference-domain.json")
    for case in protocol["cases"]:
        expected_directories = {f"seed-{seed}" for seed in case["seeds"]}
        actual_directories = {path.name for path in (directory/case["name"]).glob("seed-*") if path.is_dir()}
        require(actual_directories == expected_directories, "registered seed directory inventory differs")
        row = next(item for item in comparison["controls"] if item["name"] == case["name"])
        require(row["complete"], "individual control case is incomplete")
        for seed in case["seeds"]:
            path = directory/case["name"]/f"seed-{seed}"
            settings = json.loads((path/"settings.json").read_text())
            report = json.loads((path/"report.json").read_text())
            require(settings == expected_settings(protocol, case, seed), "control settings differ from protocol")
            require(report["complete"] and report["source_unchanged"], "control report is incomplete or changed")
            require(report["checkpoint_sha256"] == sha(path/"checkpoint.npz"), "checkpoint/report SHA differs")
            saved_row = next(item for item in row["completed_seed_results"] if item["seed"] == seed)
            require(saved_row["report_sha256"] == sha(path/"report.json"), "comparison/report SHA differs")
            fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            with load_checkpoint(path/"checkpoint.npz", expected_kind="h3.surface-partition",
                                 require_integrity=True) as saved:
                require(str(saved["fingerprint"].item()) == fingerprint, "checkpoint settings fingerprint differs")
                require(str(saved["domain_sha256"].item()) == sha(path/"reference-domain.json"),
                        "checkpoint domain normalization differs")
                require(int(saved["completed"]) == protocol["warmup"]+protocol["production"],
                        "checkpoint sweep count differs")
                require(saved["delta_energy"].shape == (protocol["production"], len(protocol["lambda_grid"])),
                        "retained BAR observation count differs")
                require(np.isfinite(saved["delta_energy"]).all(), "nonfinite retained BAR observation")
                require(saved["endpoint_q"].shape == (0, case["beads"], 3, 3),
                        "controls unexpectedly generated child parents")
            cohorts.append({"case": case["name"], "seed": seed, "beads": case["beads"],
                            "minimum_pair_bohr": case["minimum"],
                            "completed_warmup": protocol["warmup"], "completed_production": protocol["production"],
                            "checkpoint_internal_integrity_verified": True,
                            "files": {name: record(path/name) for name in file_names}})
    manifest = {
        "schema": "pyrpmd.h3.complete-equilibrium-control-manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_verification_passed": True, "completed_control_chains": len(cohorts),
        "protocol_sha256": sha(directory/"protocol.json"),
        "comparison_sha256": sha(directory/"comparison.json"),
        "verifier_source_sha256": sha(__file__), "frozen_source_sha256": sources,
        "PES_identity": provider.metadata,
        "primary_manifest_sha256": sha(base/"primary-manifest.json"),
        "immutable_primary_files_reverified": primary_files,
        "immutable_TS_parent_count": primary["total_TS_parents"],
        "primary_parent_files_unchanged": True, "cohorts": cohorts,
        "equilibrium_control_equivalence_gates_passed": comparison["equilibrium_control_equivalence_gates_passed"],
        "scientific_gate_note": "The registered coarse-P64 failure is retained; integrity success does not convert a failed equivalence result into a pass.",
        "raw_NPZ_availability": "Retained locally with hashes; this manifest does not assert external publication or upload.",
    }
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({key: manifest[key] for key in (
        "integrity_verification_passed", "completed_control_chains", "immutable_primary_files_reverified",
        "immutable_TS_parent_count", "equilibrium_control_equivalence_gates_passed")}))


if __name__ == "__main__":
    main()
