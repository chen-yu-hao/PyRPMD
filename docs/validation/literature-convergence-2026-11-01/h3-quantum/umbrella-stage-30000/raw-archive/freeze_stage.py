"""Freeze the completed 30,000-sweep umbrella stage before any continuation.

Local packaging only. No sampler, PES, recrossing output, or analysis is invoked.
The original NPZ bytes, including restart state and RNG, are never rewritten.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
WORK = ROOT.parent
DATA = WORK / "literature-convergence-quantum"
EVIDENCE = Path(__file__).resolve().parent
ARCHIVE = WORK / "pyrpmd-h3-quantum-umbrella-stage-30000-2026-11-01.tar.gz"
BASE = "docs/validation/literature-convergence-2026-11-01/h3-quantum"
PROTOCOL = BASE + "/protocol.json"
DECISION = BASE + "/umbrella-stage-30000/continuation-decision.json"
SEEDS = [211017, 211019, 211021, 211023, 211027, 211031, 211033, 211039]
STATE_KEYS = ("R", "r", "completed", "rng", "accepted", "proposed", "swaps", "lineage")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(data, path):
    return {"path": path, "bytes": len(data), "sha256": sha(data)}


def array_record(array):
    require(not array.dtype.hasobject, "no object/pickle arrays")
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": sha(np.ascontiguousarray(array).tobytes())}


def write_new(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return data


def seed_snapshot(read, seed, protocol_sha, driver_sha):
    directory = f"seed-{seed}"
    checkpoint_name = directory + "/checkpoint.npz"
    state_name = directory + "/stages/30000/state.npz"
    receipt_name = directory + "/stages/30000/stage.json"
    settings_name = directory + "/settings.json"
    report_name = directory + "/run-report.json"
    checkpoint, state = read(checkpoint_name), read(state_name)
    receipt_bytes, settings_bytes, report_bytes = read(receipt_name), read(settings_name), read(report_name)
    receipt, settings, report = map(json.loads, (receipt_bytes, settings_bytes, report_bytes))
    require(receipt["seed"] == seed == settings["seed"] == report["seed"], "seed mismatch")
    require(receipt["production_sweeps"] == 30000 and settings["production"] == 300000,
            "stage/registered final production differs")
    require(settings["warmup"] == 5000 and settings["beads"] == 128, "warmup/beads differ")
    require(receipt["protocol_sha256"] == protocol_sha and receipt["driver_sha256"] == driver_sha,
            "protocol/driver differs")
    require(receipt["checkpoint_sha256"] == sha(checkpoint) == report["checkpoint_sha256"],
            "checkpoint/report/receipt identity differs")
    require(receipt["state_sha256"] == sha(state), "stage state identity differs")
    require(report["source_unchanged"] and report["completed_sweeps"] == 35000,
            "source or stage completion differs")
    with np.load(io.BytesIO(checkpoint), allow_pickle=False) as saved, np.load(io.BytesIO(state), allow_pickle=False) as prior:
        require(len(saved.files) == len(set(saved.files)), "duplicate checkpoint arrays")
        arrays = {name: array_record(saved[name]) for name in saved.files}
        require(int(saved["completed"]) == 35000, "checkpoint completion differs")
        require(saved["samples"].shape == (30000, 48), "sample shape differs")
        require(saved["diagnostics"].shape == (3000, 48, 7), "diagnostic shape differs")
        require(saved["R"].shape == saved["r"].shape == (48, 128, 3), "ring geometry shape differs")
        require(arrays["samples"]["sha256"] == receipt["samples_prefix_sha256"], "sample prefix differs")
        require(arrays["diagnostics"]["sha256"] == receipt["diagnostics_prefix_sha256"], "diagnostic prefix differs")
        require(set(prior.files) == set(STATE_KEYS), "restart state field inventory differs")
        for name in STATE_KEYS:
            require(array_record(prior[name]) == arrays[name], f"restart state differs: {name}")
        json.loads(str(saved["rng"].item()))
    return {"seed": seed, "directory": directory, "checkpoint": record(checkpoint, checkpoint_name),
            "settings": record(settings_bytes, settings_name), "run_report": record(report_bytes, report_name),
            "arrays": arrays, "completed_sweeps": 35000,
            "stage_receipts": [{"production_sweeps": 30000,
                                "receipt": record(receipt_bytes, receipt_name), "state": record(state, state_name),
                                "samples_prefix_sha256": receipt["samples_prefix_sha256"],
                                "diagnostics_prefix_sha256": receipt["diagnostics_prefix_sha256"]}],
            "source_sha256": settings["source_sha256"], "legacy_NPZ_without_array_envelope": True}


def main():
    require((ROOT / "benchmarks/h3/quantum_sampling_campaign.py").is_file(), "repository root resolution failed")
    require(not ARCHIVE.exists(), "immutable archive already exists")
    protocol_bytes = (ROOT / PROTOCOL).read_bytes()
    protocol = json.loads(protocol_bytes)
    require(sha(protocol_bytes) == "5834471583ae8e513050dc1c22e065459e1cfa6f139cc5490fcb4eded9fe711a", "protocol changed")
    decision_bytes = (ROOT / DECISION).read_bytes()
    require(sha(decision_bytes) == "c96255a6eea63023f4eefe9450daa8fc4d67399d55a3143769bd83aa41169c16", "decision changed")
    driver = "benchmarks/h3/quantum_sampling_campaign.py"
    driver_sha = sha((ROOT / driver).read_bytes())
    require(driver_sha == "29a70de601253ba2262fd1be3a65e65d4bc7cef26652cbe2ff2d30ccccf9df50", "driver changed")
    require(protocol["independent_ladder_seeds"] == SEEDS, "registered cohort changed")
    require(sha((ROOT / protocol["schedule_file"]).read_bytes()) == protocol["schedule_sha256"], "schedule changed")
    seed_rows = [seed_snapshot(lambda name: (DATA / "campaign-run" / name).read_bytes(), seed,
                               sha(protocol_bytes), driver_sha) for seed in SEEDS]
    comparison_bytes = (DATA / "full-cohort-stage-30000/campaign-comparison.json").read_bytes()
    comparison = json.loads(comparison_bytes)
    require(comparison["completed_independent_seeds"] == comparison["expected_independent_seeds"] == SEEDS,
            "complete analysis cohort differs")
    require(comparison["production_sweeps_per_ladder"] == 30000 and
            comparison["criteria"]["statistical_precision"] is False, "stage/gate changed")
    for row in seed_rows:
        detail = json.loads((DATA / "full-cohort-stage-30000" / f"seed-{row['seed']}.json").read_text())
        require(detail["checkpoint_sha256"] == row["checkpoint"]["sha256"] and
                detail["receipt_sha256"] == row["stage_receipts"][0]["receipt"]["sha256"], "analysis/raw identity differs")
    paths = []
    # This allowlist deliberately excludes every recrossing directory.
    for folder, category in (("campaign-run", "complete_stage_30000_sampling"),
                             ("full-cohort-stage-30000", "complete_eight_chain_analysis"),
                             ("campaign", "failed_initialization_empty_logs"),
                             ("analysis", "one_chain_diagnostic_not_final_inference"),
                             ("cohort-analysis", "two_chain_diagnostic_before_stdout_fix"),
                             ("cohort-analysis-after-stdout-fix", "two_chain_diagnostic_after_stdout_fix")):
        for path in sorted((DATA / folder).rglob("*")):
            if path.is_file():
                require(not path.is_symlink() and path.suffix not in (".py", ".so", ".f", ".f90"), "unexpected raw file type")
                paths.append((path, "data/" + path.relative_to(DATA).as_posix(), category))
    paths.append((DATA / "full-cohort-stage-30000.log", "data/full-cohort-stage-30000.log", "complete_analysis_log"))
    for name in (PROTOCOL, DECISION, protocol["schedule_file"], BASE + "/analysis-prespecification.json",
                 BASE + "/initialization-failure.json", BASE + "/analysis-stdout-fix/receipt.json",
                 BASE + "/analysis-stdout-fix/two-seed-comparison-before-fix.json",
                 BASE + "/analysis-stdout-fix/two-seed-comparison-after-fix.json"):
        paths.append((ROOT / name, "reference/" + name, "immutable_protocol_decision_or_historical_receipt"))
    paths.sort(key=lambda x: x[1])
    members = [{**record(path.read_bytes(), name), "category": category} for path, name, category in paths]
    require(len({row["path"] for row in members}) == len(members), "duplicate member")
    descriptor, temporary = tempfile.mkstemp(prefix=ARCHIVE.name + ".", suffix=".partial", dir=WORK)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with gzip.GzipFile(filename="", fileobj=output, mode="wb", compresslevel=6, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for (path, name, _), expected in zip(paths, members):
                        data = path.read_bytes()
                        require(sha(data) == expected["sha256"] and len(data) == expected["bytes"], "source changed during packing")
                        info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o444; info.mtime = 0
                        info.uid = info.gid = 0; info.uname = info.gname = ""
                        archive.addfile(info, io.BytesIO(data))
            output.flush(); os.fsync(output.fileno())
        with tarfile.open(temporary, "r:gz") as archive:
            actual = archive.getmembers()
            require(len(actual) == len(members), "archive member count differs")
            for info, expected in zip(actual, members):
                data = archive.extractfile(info).read()
                require(info.name == expected["path"] and info.isfile() and
                        record(data, info.name) == {k: expected[k] for k in ("path", "bytes", "sha256")},
                        "archived member differs")
            archived_rows = [seed_snapshot(lambda name: archive.extractfile("data/campaign-run/" + name).read(),
                                           seed, sha(protocol_bytes), driver_sha) for seed in SEEDS]
            require(archived_rows == seed_rows, "archive restart/prefix reconstruction differs")
        for (path, _, _), expected in zip(paths, members):
            data = path.read_bytes()
            require(sha(data) == expected["sha256"] and len(data) == expected["bytes"], "source changed after packing")
        os.link(temporary, ARCHIVE); ARCHIVE.chmod(0o444)
    finally:
        if Path(temporary).exists(): Path(temporary).unlink()
    archive_identity = record(ARCHIVE.read_bytes(), str(ARCHIVE))
    inventory = {"schema": "pyrpmd.h3.quantum-umbrella-archive-members.v1", "archive": archive_identity,
                 "members": members, "member_count": len(members), "member_bytes": sum(row["bytes"] for row in members),
                 "category_counts": dict(sorted(Counter(row["category"] for row in members).items())),
                 "scope": "Frozen stage30000 only; all8 checkpoints, exact restart state/RNG, settings, receipts, complete and diagnostic analysis. No external model or live child/water data."}
    sidecar = ARCHIVE.with_suffix("").with_suffix(".members.json")
    sidecar_bytes = write_new(sidecar, inventory); sidecar.chmod(0o444)
    write_new(EVIDENCE / "member-inventory.json", inventory)
    source_files = {k: v for k, v in seed_rows[0]["source_sha256"].items() if k != "bkmp2_library"}
    source_files[driver] = driver_sha
    source_files.update(comparison["source_sha256"])
    for name, expected in source_files.items():
        require(sha((ROOT / name).read_bytes()) == expected, "registered source dependency differs")
    snapshot = {"schema": "pyrpmd.h3.quantum-umbrella-stage-snapshot.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "production_sweeps": 30000, "warmup_sweeps": 5000, "independent_seeds": SEEDS,
                "campaign_directory": str(DATA / "campaign-run"), "protocol": record(protocol_bytes, PROTOCOL),
                "schedule": record((ROOT / protocol["schedule_file"]).read_bytes(), protocol["schedule_file"]),
                "source_sha256": source_files, "model_sha256": seed_rows[0]["source_sha256"]["bkmp2_library"],
                "seeds": seed_rows, "archive": archive_identity,
                "member_inventory": record(sidecar_bytes, str(sidecar)),
                "complete_analysis": record(comparison_bytes, str(DATA / "full-cohort-stage-30000/campaign-comparison.json")),
                "continuation_decision": record(decision_bytes, DECISION),
                "old_full_file_and_array_identities_preserved": True,
                "raw_archive_verified_before_any_extension": True,
                "legacy_checkpoint_note": "Original NPZ has no integrity envelope; complete file hashes, ZIP CRC reads, prefix hashes and all state-array identities are preserved without rewriting it.",
                "physical_converged": False, "full_rpmd_rate_validated": False}
    write_new(EVIDENCE / "stage-snapshot.json", snapshot)
    provenance = {"schema": "pyrpmd.h3.quantum-umbrella-stage-archive-provenance.v1", "created_utc": snapshot["created_utc"],
                  "archive": archive_identity, "member_inventory": record(sidecar_bytes, str(sidecar)),
                  "git_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "builder_sha256": sha(Path(__file__).read_bytes()), "source_dependencies": source_files,
                  "checks": {"all_member_bytes_verified": True, "all_inputs_rehashed_after_pack": True,
                             "eight_checkpoint_state_RNG_and_prefixes_verified_from_archive": True,
                             "all_eight_complete_analysis_inputs_matched": True},
                  "stage_QTST_cm3_molecule_s": comparison["pooled_wham"]["qtst_cm3_molecule_s"],
                  "stage_primary_log_standard_error": comparison["statistics"]["primary_log_standard_error"],
                  "registered_standard_error_limit": protocol["gates"]["target_qtst_relative_standard_error"],
                  "precision_passed": False, "paper_point_used_for_continuation": False,
                  "local_only": True, "upload_performed": False, "simulation_or_analysis_recomputed": False,
                  "Git_mutated": False}
    write_new(EVIDENCE / "archive-provenance.json", provenance)
    print(json.dumps({"archive": archive_identity, "sidecar": record(sidecar_bytes, str(sidecar)),
                      "member_count": len(members), "member_bytes": inventory["member_bytes"],
                      "stage_snapshot": record((EVIDENCE / "stage-snapshot.json").read_bytes(), str(EVIDENCE / "stage-snapshot.json"))}, indent=2))


if __name__ == "__main__":
    main()
