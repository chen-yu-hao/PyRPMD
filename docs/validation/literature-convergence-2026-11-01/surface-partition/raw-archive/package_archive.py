"""Package or independently verify the frozen surface-partition evidence.

This tool never calls a sampler, evaluates a PES, resumes a checkpoint, accesses
the network, or modifies an input. It accepts only the frozen 243-file inventory
and its 35 raw NPZ files. Archive creation refuses to replace an existing file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import tarfile
import tempfile

import numpy as np


BASE = "docs/validation/literature-convergence-2026-11-01/surface-partition"
FROZEN = f"{BASE}/controls/publishable-files.json"
FROZEN_SHA = "f68456bddc3e340911ae91590507a2769ae0a17d67eba983325870b15ba87b68"
ENVELOPE = "_pyrpmd_checkpoint"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path, value):
    data = json_bytes(value)
    with Path(path).open("xb") as stream:
        stream.write(data)
    return data


def safe_name(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            f"unsafe archive path: {name}")
    require(path.as_posix() == name, f"noncanonical archive path: {name}")
    return name


def category(name):
    if not name.startswith(BASE + "/"):
        return "repository_source_or_test"
    part = name[len(BASE) + 1:]
    if part.startswith("primary/ts/"):
        return "primary_TS_equilibrium_and_parents"
    if part.startswith("primary/reactant/"):
        return "primary_reactant_normalization_control"
    if part.startswith("controls/short-distance"):
        return "finite_grid_short_distance_diagnostic"
    if part.startswith("controls/p") and "/seed-" in part:
        return "registered_bead_or_domain_control"
    if part.startswith(("calibration-", "calibration.", "calibrate.", "surface_partition.calibration", "test_h3_surface_partition.calibration")):
        return "excluded_initial_calibration"
    if part.startswith(("efficiency-", "efficiency.", "efficiency_", "ts-grid-efficiency", "reactant-grid-efficiency")):
        return "excluded_efficiency_calibration"
    if part.startswith(("ts-importance.", "reactant-importance.")):
        return "excluded_direct_importance_pilot"
    return "method_protocol_reference_or_aggregate_evidence"


def array_description(array):
    require(not array.dtype.hasobject, "object/pickle arrays are prohibited")
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": sha(array.tobytes(order="C"))}


def inspect_npz(name, data, text):
    """Read every ZIP member and independently check the original array envelope."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        require(len(set(archive.files)) == len(archive.files), "duplicate NPZ array names")
        arrays = {key: archive[key] for key in archive.files if key != ENVELOPE}
        descriptions = {key: array_description(value) for key, value in arrays.items()}
        envelope = json.loads(str(archive[ENVELOPE].item())) if ENVELOPE in archive else None
    if envelope is not None:
        require(envelope.get("schema") == "pyrpmd.checkpoint", "unknown checkpoint schema")
        require(envelope.get("schema_version") == 2, "unknown checkpoint version")
        require(envelope["arrays"] == descriptions, "per-array integrity envelope differs")
    row = {"path": name, "archive_member": name, "sha256": sha(data), "bytes": len(data),
           "category": category(name), "array_count": len(arrays),
           "container_integrity": "verified_v2_envelope" if envelope else "no_original_envelope; ZIP CRC and whole-file SHA verified"}
    if name.endswith("/checkpoint.npz"):
        require(envelope and envelope["kind"] == "h3.surface-partition", "unexpected checkpoint kind")
        directory = name.rsplit("/", 1)[0]
        report_name, settings_name = directory + "/report.json", directory + "/settings.json"
        domain_name = directory + "/reference-domain.json"
        report, settings = json.loads(text[report_name]), json.loads(text[settings_name])
        require(report["checkpoint_sha256"] == row["sha256"], "report/checkpoint SHA mismatch")
        require(report["complete"] and report["source_unchanged"], "checkpoint report is not complete/frozen")
        fingerprint = sha(json.dumps(settings, sort_keys=True).encode())
        require(str(arrays["fingerprint"].item()) == fingerprint, "settings fingerprint differs")
        require(str(arrays["domain_sha256"].item()) == sha(text[domain_name]), "domain normalization hash differs")
        completed = settings["warmup"] + settings["production"]
        require(int(arrays["completed"]) == completed == report["completed_sweeps"], "sweep count differs")
        samples = (settings["production"] + settings["sample_interval"] - 1) // settings["sample_interval"]
        diagnostics = (settings["production"] + settings["diagnostic_interval"] - 1) // settings["diagnostic_interval"]
        require(arrays["delta_energy"].shape == (samples, len(settings["lambdas"])), "BAR sample shape differs")
        require(arrays["diagnostics"].shape == (diagnostics, len(settings["lambdas"]), 8), "diagnostic shape differs")
        require(np.isfinite(arrays["delta_energy"]).all(), "nonfinite BAR observations")
        require(report["production_samples_per_lambda"] == samples, "report/sample count differs")
        require((report["seed"], report["beads"], report["surface"]) ==
                (settings["seed"], settings["beads"], settings["surface"]), "report/settings identity differs")
        row.update(seed=settings["seed"], beads=settings["beads"], surface=settings["surface"],
                   completed_sweeps=completed, production_samples_per_lambda=samples,
                   report={"path": report_name, "sha256": sha(text[report_name])},
                   settings={"path": settings_name, "sha256": sha(text[settings_name]), "fingerprint": fingerprint},
                   domain={"path": domain_name, "sha256": sha(text[domain_name])},
                   checkpoint_kind=envelope["kind"], rng_array=descriptions["rng"],
                   physical_converged_as_reported=report["physical_converged"],
                   literature_reproduced_as_reported=report["literature_reproduced"])
        if "endpoint_q" in arrays:
            interval = settings["snapshot_interval"]
            count = settings["production"] // interval if interval else 0
            expected_sweeps = settings["warmup"] + np.arange(1, count + 1) * interval
            require(arrays["endpoint_q"].shape == (count, settings["beads"], 3, 3), "parent shape differs")
            require(np.array_equal(arrays["endpoint_completed_sweeps"], expected_sweeps), "parent order/sweeps differ")
            require(report["endpoint_snapshot_count"] == count, "report parent count differs")
            require(np.isfinite(arrays["endpoint_q"]).all(), "nonfinite parent coordinates")
            row["parents"] = {"count": count, "coordinate_units": "bohr",
                              "endpoint_q": descriptions["endpoint_q"],
                              "endpoint_completed_sweeps": descriptions["endpoint_completed_sweeps"]}
    elif name.endswith("-importance.npz"):
        report_name = name[:-4] + ".json"
        report = json.loads(text[report_name])
        require(report["raw_sha256"] == row["sha256"], "importance report/raw SHA differs")
        row.update(report={"path": report_name, "sha256": sha(text[report_name])},
                   seed=report["seed"], beads=report["beads"], surface=report["surface"], arrays=descriptions,
                   primary_inference_inclusion=False)
    elif name.endswith("controls/short-distance-grid.npz"):
        report_name = f"{BASE}/controls/short-distance-domain-check.json"
        report = json.loads(text[report_name])
        require(report["raw"]["sha256"] == row["sha256"] and report["raw"]["bytes"] == len(data),
                "short-distance report/raw identity differs")
        row.update(report={"path": report_name, "sha256": sha(text[report_name])}, arrays=descriptions,
                   primary_inference_inclusion=False)
    else:
        raise ValueError(f"unregistered raw file: {name}")
    return row


def verify_relations(text, records, raw):
    primary = json.loads(text[f"{BASE}/primary-manifest.json"])
    controls = json.loads(text[f"{BASE}/controls/manifest.json"])
    frozen = json.loads(text[FROZEN])
    require(sha(text[FROZEN]) == FROZEN_SHA, "frozen publication inventory changed")
    require(len(frozen["files"]) == 243, "frozen publication file count differs")
    for item in frozen["files"]:
        require(records[item["path"]]["sha256"] == item["sha256"] and
                records[item["path"]]["bytes"] == item["bytes"], "frozen publication entry differs")
    require(len(raw) == 35 and sum(x["bytes"] for x in raw) == frozen["excluded_raw_NPZ_bytes"],
            "raw count/size differs from frozen inventory")
    verified_files = 0
    for cohort in primary["cohorts"]:
        for name, expected in cohort["files"].items():
            found = records[cohort["directory"] + "/" + name]
            require(found["sha256"] == expected["sha256"] and found["bytes"] == expected["bytes"],
                    "primary immutable manifest differs")
            verified_files += 1
    require(verified_files == 72 and len(primary["cohorts"]) == 12, "primary cohort inventory differs")
    primary_rows = [x for x in raw if x["category"].startswith("primary_")]
    ts = [x for x in primary_rows if x["surface"] == "ts"]
    require(len(ts) == 8 and sum(x["parents"]["count"] for x in ts) == 1600, "primary parents incomplete")
    control_files = 0
    for cohort in controls["cohorts"]:
        directory = f"{BASE}/controls/{cohort['case']}/seed-{cohort['seed']}"
        for name, expected in cohort["files"].items():
            found = records[directory + "/" + name]
            require(found["sha256"] == expected["sha256"] and found["bytes"] == expected["bytes"],
                    "control immutable manifest differs")
            control_files += 1
    require(len(controls["cohorts"]) == 16 and control_files == 96, "control cohort inventory differs")
    require(controls["primary_manifest_sha256"] == sha(text[f"{BASE}/primary-manifest.json"]),
            "control/primary manifest link differs")
    require(primary["registered_protocol_sha256"] == sha(text[f"{BASE}/primary-protocol.json"]),
            "primary protocol identity differs")
    require(primary["complete_summary_sha256"] == sha(text[f"{BASE}/primary-summary.json"]),
            "primary summary identity differs")
    require(controls["protocol_sha256"] == sha(text[f"{BASE}/controls/protocol.json"]),
            "control protocol identity differs")
    require(controls["comparison_sha256"] == sha(text[f"{BASE}/controls/comparison.json"]),
            "control comparison identity differs")
    require(frozen["individual_control_equivalence"] == {
        "p64-cutoff05": False, "p256-cutoff05": True,
        "p128-cutoff04": True, "p128-cutoff06": True}, "registered control decisions changed")
    require(controls["equilibrium_control_equivalence_gates_passed"] is False, "failed aggregate gate changed")
    for row in raw:
        if row["category"] == "registered_bead_or_domain_control":
            case = row["path"][len(BASE + "/controls/"):].split("/")[0]
            row.update(control_case=case, registered_case_equivalence_passed=frozen["individual_control_equivalence"][case])
        if row["category"].startswith("excluded_"):
            row["primary_inference_inclusion"] = False
    return {"primary_checkpoints": len(primary_rows), "primary_manifest_file_identities": verified_files,
            "TS_chains": len(ts), "registered_TS_parents": 1600,
            "control_chains": 16, "control_manifest_file_identities": control_files,
            "all_checkpoint_report_SHA_links_verified": True,
            "all_checkpoint_array_envelopes_verified": True,
            "control_decisions_retained": frozen["individual_control_equivalence"],
            "aggregate_control_equivalence_passed": False,
            "integrity_success_is_not_scientific_convergence": True}


def verify_archive(archive_path, inventory):
    expected = {row["path"]: row for row in inventory["members"]}
    require(len(expected) == 279, "archive must contain exactly 243 frozen entries, their inventory, and 35 NPZ")
    records, text, raw_data = {}, {}, []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        require(len(members) == len(expected), "archive member count differs")
        for member in members:
            safe_name(member.name)
            require(member.isfile() and member.name in expected and member.name not in records,
                    "unlisted, duplicate, or nonregular archive member")
            data = archive.extractfile(member).read()
            row = expected[member.name]
            require(len(data) == row["bytes"] == member.size and sha(data) == row["sha256"],
                    f"archive member bytes differ: {member.name}")
            records[member.name] = row
            if not member.name.endswith(".npz"):
                text[member.name] = data
        # One raw file at a time: bounded memory; NPZ integrity is checked from
        # the bytes in the completed archive, not by reopening the source path.
        for name in sorted(expected):
            if name.endswith(".npz"):
                data = archive.extractfile(name).read()
                raw_data.append(inspect_npz(name, data, text))
    checks = verify_relations(text, records, raw_data)
    checks.update(archive_member_count=len(records), archive_member_bytes_verified=True,
                  raw_NPZ_count=len(raw_data), raw_NPZ_bytes=sum(x["bytes"] for x in raw_data),
                  raw_category_counts=dict(sorted(Counter(x["category"] for x in raw_data).items())))
    return checks, raw_data


def extract_parents(archive_path, destination, inventory):
    """Extract exactly the frozen TS coordinates/order, without new dynamics/RNG."""
    checks, raw = verify_archive(archive_path, inventory)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for item in raw:
            if item["category"] != "primary_TS_equilibrium_and_parents":
                continue
            with np.load(io.BytesIO(archive.extractfile(item["path"]).read()), allow_pickle=False) as arrays:
                for key in ("endpoint_q", "endpoint_completed_sweeps"):
                    value = arrays[key]
                    path = destination / f"seed-{item['seed']}-{key}.npy"
                    with path.open("xb") as stream:
                        np.save(stream, value, allow_pickle=False)
                    loaded = np.load(path, allow_pickle=False)
                    require(array_description(loaded) == item["parents"][key], "extracted parent array differs")
                    rows.append({"path": path.name, "source_checkpoint": item["path"],
                                 "source_checkpoint_sha256": item["sha256"],
                                 "array": key, **array_description(loaded)})
    write_new(destination / "parent-array-identities.json", {"parents": 1600, "files": rows})
    return {"parent_arrays_extracted_and_reloaded": len(rows), "parents": checks["registered_TS_parents"]}


def create(root, archive_path, evidence):
    root, archive_path, evidence = root.resolve(), archive_path.resolve(), evidence.resolve()
    require(not archive_path.exists(), "archive already exists; immutable files are not overwritten")
    sidecar = archive_path.with_suffix("").with_suffix(".members.json")
    require(not sidecar.exists(), "archive inventory already exists")
    frozen_data = (root / FROZEN).read_bytes()
    require(sha(frozen_data) == FROZEN_SHA, "frozen publication manifest changed")
    frozen = json.loads(frozen_data)
    rows = []
    for item in frozen["files"]:
        name = safe_name(item["path"])
        path = root / name
        require(path.is_file() and not path.is_symlink(), "input must be a regular file")
        data = path.read_bytes()
        require(sha(data) == item["sha256"] and len(data) == item["bytes"], f"frozen source differs: {name}")
        rows.append({"path": name, "sha256": sha(data), "bytes": len(data),
                     "role": "frozen_publishable", "category": category(name)})
    require(len(rows) == 243, "frozen source count differs")
    rows.append({"path": FROZEN, "sha256": sha(frozen_data), "bytes": len(frozen_data),
                 "role": "frozen_publication_inventory", "category": "scope_and_file_identity"})
    raw_paths = sorted((root / BASE).rglob("*.npz"))
    require(len(raw_paths) == 35 and sum(p.stat().st_size for p in raw_paths) == 82902442,
            "raw scope/count/size differs")
    for path in raw_paths:
        require(path.is_file() and not path.is_symlink(), "raw input must be a regular file")
        name = safe_name(path.relative_to(root).as_posix())
        data = path.read_bytes()
        rows.append({"path": name, "sha256": sha(data), "bytes": len(data),
                     "role": "raw_NPZ", "category": category(name)})
    rows.sort(key=lambda row: row["path"])
    require(len({row["path"] for row in rows}) == 279, "duplicate scoped input")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    inventory = {"schema": "pyrpmd.h3.quantum-equilibrium-archive-members.v1",
                 "archive_basename": archive_path.name, "member_path_base": "repository root; no leading directory",
                 "frozen_inventory": {"path": FROZEN, "sha256": FROZEN_SHA, "bytes": len(frozen_data)},
                 "scope": "243 frozen source/text/image files, their original inventory, exactly 35 raw NPZ; no live child/umbrella data or external PES files",
                 "members": rows, "member_count": len(rows), "member_uncompressed_bytes": sum(row["bytes"] for row in rows)}
    evidence.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=archive_path.name + ".", suffix=".partial", dir=archive_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw_stream:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, compresslevel=6, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for row in rows:
                        data = (root / row["path"]).read_bytes()
                        require(len(data) == row["bytes"] and sha(data) == row["sha256"], "source changed while packing")
                        info = tarfile.TarInfo(row["path"])
                        info.size, info.mtime, info.mode = len(data), 0, 0o444
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        tar.addfile(info, io.BytesIO(data))
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        checks, raw_index = verify_archive(Path(temporary), inventory)
        for row in rows:
            data = (root / row["path"]).read_bytes()
            require(len(data) == row["bytes"] and sha(data) == row["sha256"], "source changed after packing")
        os.link(temporary, archive_path)  # Atomic no-replace publication of local bytes.
        archive_path.chmod(0o444)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    archive_bytes = archive_path.read_bytes()
    archive_identity = {"path": str(archive_path), "bytes": len(archive_bytes), "sha256": sha(archive_bytes)}
    member_bytes = write_new(sidecar, inventory)
    sidecar.chmod(0o444)
    write_new(evidence / "member-inventory.json", inventory)
    raw_document = {"schema": "pyrpmd.h3.quantum-equilibrium-raw-index.v1",
                    "archive": archive_identity, "member_inventory_sha256": sha(member_bytes),
                    "raw_count": len(raw_index), "raw_bytes": 82902442,
                    "files": raw_index, "category_counts": checks["raw_category_counts"],
                    "all_1600_primary_parents_retained": True,
                    "raw_NPZ_Git_publication_requested": False,
                    "external_upload_performed": False}
    write_new(evidence / "raw-data-index.json", raw_document)
    primary = json.loads((root / BASE / "primary-manifest.json").read_text())
    source_dependencies = []
    names = {row["path"] for row in rows}
    for name, expected in primary["source_sha256"].items():
        data = (root / name).read_bytes()
        require(sha(data) == expected, f"registered source dependency differs: {name}")
        source_dependencies.append({"path": name, "sha256": expected, "bytes": len(data),
                                    "included_in_archive": name in names})
    # A genuine roundtrip verifies both the exported .npy header and the exact
    # coordinate/sweep array bytes. Only this new temporary directory is removed.
    with tempfile.TemporaryDirectory(prefix="quantum-parent-extraction-", dir=archive_path.parent) as temp:
        extraction = extract_parents(archive_path, Path(temp) / "parents", inventory)
    verification = {"schema": "pyrpmd.h3.quantum-equilibrium-archive-verification.v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(), **checks,
                    "original_inputs_rehashed_after_archive": 279,
                    "frozen_243_file_manifest_unmodified": True, "parent_extraction_roundtrip": extraction,
                    "new_sampling_or_PES_evaluation": False, "external_upload_performed": False}
    write_new(evidence / "verification.json", verification)
    provenance = {"schema": "pyrpmd.h3.quantum-equilibrium-archive-provenance.v1",
                  "created_utc": verification["created_utc"], "archive": archive_identity,
                  "member_inventory": {"path": str(sidecar), "sha256": sha(member_bytes), "bytes": len(member_bytes)},
                  "source_repository": str(root), "git_HEAD_at_packaging": head,
                  "git_HEAD_at_end": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "file_identity_authority": "per-file immutable SHA256, not an assertion that all working-tree evidence was committed at HEAD",
                  "frozen_inventory_sha256": FROZEN_SHA, "source_dependencies": source_dependencies,
                  "builder": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).read_bytes())},
                  "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                  "archive_format": "sorted PAX tar, regular 0444 members, uid/gid=0, mtime=0; gzip level 6 and mtime=0",
                  "immutability": "exclusive creation; postwrite full-member verification; archive and adjacent member inventory chmod 0444; no overwrite/resume",
                  "scope_counts": {"frozen_publishable": 243, "frozen_inventory": 1, "raw_NPZ": 35},
                  "availability": "local workspace only; no upload attempted or external persistence asserted",
                  "excluded": ["external BKMP2 Fortran source", "PES libraries and compiled native binaries", "live recrossing data", "umbrella campaigns", "unlisted repository files"]}
    write_new(evidence / "archive-provenance.json", provenance)
    print(json.dumps({"archive": archive_identity, "member_inventory_sha256": sha(member_bytes),
                      "member_count": len(rows), "checks": verification}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--verify", type=Path, metavar="MEMBER_INVENTORY")
    parser.add_argument("--extract-parents", type=Path)
    args = parser.parse_args()
    if args.verify:
        inventory = json.loads(args.verify.read_text())
        if args.extract_parents:
            print(json.dumps(extract_parents(args.archive, args.extract_parents, inventory), indent=2))
        else:
            print(json.dumps(verify_archive(args.archive, inventory)[0], indent=2))
    else:
        require(args.root is not None and args.evidence is not None and args.extract_parents is None,
                "creation requires --root and --evidence; parent extraction requires --verify")
        create(args.root, args.archive, args.evidence)


if __name__ == "__main__":
    main()
