"""Local-only, explicit-whitelist four-pair increment and extraction replay."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import resource
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[6]
SCRATCH = ROOT.parent
OUT = Path(__file__).resolve().parent
REVIEW = OUT.parent / "independent-review"
EVIDENCE = ROOT / "docs/validation/literature-convergence-2026-11-01"
BASE = SCRATCH / "literature-convergence-quantum"
DATA = BASE / "recrossing-primary"
ANALYSIS = BASE / "recrossing-analysis-200-four-pairs"
INTEGRATION = SCRATCH / "pyrpmd-convergence-integration"
ARCHIVE = SCRATCH / "pyrpmd-h3-quantum-recrossing-stage-200-pairs-4-increment-2026-11-01.tar.gz"
SIDECAR = ARCHIVE.with_name(ARCHIVE.name.removesuffix(".tar.gz") + ".members.json")
BASE_ARCHIVE = SCRATCH / "pyrpmd-h3-quantum-recrossing-stage-200-pairs-2-2026-11-01.tar.gz"
BASE_SIDECAR = BASE_ARCHIVE.with_name(BASE_ARCHIVE.name.removesuffix(".tar.gz") + ".members.json")
PARENT_ARCHIVE = SCRATCH / "pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz"
PARENT_SIDECAR = PARENT_ARCHIVE.with_name(PARENT_ARCHIVE.name.removesuffix(".tar.gz") + ".members.json")
BASE_SHA = "d3205ebc4a5210af59d03ebdbf0737708278d83f1e4cb540a18818213d78cea8"
PARENT_SHA = "051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d"
REVIEW_SHA = "945c4f1385f55142fd78a6f9f32bc510ec8cf39de0705b4cb524c5a9d67cee54"
PUBLICATION_SHA = "40dc3e500115a0bbff95be18ab25d82cfb1b6459da9e692b184606d65804ab46"
ROUNDTRIP = SCRATCH / "h3-four-pair-archive-roundtrip-2026-11-01"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked_name(name):
    parsed = PurePosixPath(name)
    assert not parsed.is_absolute() and ".." not in parsed.parts and "\\" not in name
    return parsed


def extract_verified(archive_path, rows, destination, *, selected=None):
    """Hash every archive member; exclusively write only declared safe members."""
    indexed = {row.get("member", row.get("path")): row for row in rows}
    assert len(indexed) == len(rows)
    selected = set(indexed) if selected is None else set(selected)
    assert selected <= set(indexed)
    seen, extracted = set(), []
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            assert member.isfile() and member.name in indexed and member.name not in seen
            checked_name(member.name)
            raw = archive.extractfile(member).read()
            expected = indexed[member.name]
            assert len(raw) == expected["bytes"] and digest(raw) == expected["sha256"]
            seen.add(member.name)
            if member.name in selected:
                path = destination / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                assert path.resolve().is_relative_to(destination.resolve())
                with path.open("xb") as stream:
                    stream.write(raw)
                assert sha(path) == expected["sha256"]
                extracted.append({"member": member.name, "sha256": expected["sha256"], "bytes": len(raw)})
    assert seen == set(indexed) and {row["member"] for row in extracted} == selected
    return extracted


def main():
    started = time.monotonic()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(15)
    assert not ARCHIVE.exists() and not SIDECAR.exists() and not ROUNDTRIP.exists()
    assert os.statvfs(SCRATCH).f_bavail * os.statvfs(SCRATCH).f_frsize > 1_200_000_000
    source_sha = sha(__file__)
    assert sha(BASE_ARCHIVE) == BASE_SHA and sha(PARENT_ARCHIVE) == PARENT_SHA
    assert sha(REVIEW / "publication-inventory.json") == REVIEW_SHA
    review_inventory = read(REVIEW / "publication-inventory.json")
    for row in review_inventory["files"]:
        path = ROOT / row["path"]
        assert sha(path) == row["sha256"] and path.stat().st_size == row["bytes"]
    publication_path = INTEGRATION / "docs/validation/literature-convergence-2026-11-01/h3-quantum/recrossing-stage-200-pairs-4/publication-inventory.json"
    assert sha(publication_path) == PUBLICATION_SHA
    publication = read(publication_path)
    for row in publication["files"]:
        path = INTEGRATION / row["path"]
        assert sha(path) == row["sha256"] and path.stat().st_size == row["bytes"]
    raw_rows = read(REVIEW / "original/raw-chunk-identities.json")
    parent_rows = read(REVIEW / "original/parent-identities.json")
    original_replay = read(REVIEW / "original/raw-replay.json")
    assert len(raw_rows) == 800 and len(parent_rows) == 8
    increment = [row for row in raw_rows if row["pair_batch"] == 1]
    assert len(increment) == 400
    for row in read(REVIEW / "original/input-file-identities.json"):
        path = Path(row["path"])
        assert sha(path) == row["sha256"] and path.stat().st_size == row["bytes"]
    assert original_replay["summaries"]["4"]["fixed_P_primary_gates_passed"]
    assert not original_replay["summaries"]["2"]["fixed_P_primary_gates_passed"]
    assert not original_replay["summaries"]["4"]["supplementary_parent_half_equivalence_passed"]
    completed = read(DATA / "schedule-200-pairs-1-completed.json")
    exit_receipt = read(BASE / "extra-pairs-stage-completion-exec.json")
    assert exit_receipt["exit_code"] == 0 and exit_receipt["complete_manifest_written_after_ProcessPoolExecutor_shutdown"]
    assert exit_receipt["completion_sha256"] == sha(DATA / "schedule-200-pairs-1-completed.json")
    assert tuple(row["seed"] for row in completed["completed"]) == tuple(original_replay["summaries"]["4"]["per_timestep_per_seed"]["0.05"][i]["seed"] for i in range(8))
    members = {}

    def include(path, category, member=None):
        path = Path(path)
        assert path.is_file() and not path.is_symlink()
        assert path.suffix not in (".py", ".so", ".dll", ".dylib", ".f", ".f90", ".c", ".cpp")
        name = str(path.relative_to(BASE)) if member is None else member
        checked_name(name)
        assert name not in members
        members[name] = {"member": name, "source_path": str(path.resolve()), "category": category,
                         "sha256": sha(path), "bytes": path.stat().st_size}

    for row in increment:
        assert row["member"].startswith("recrossing-primary/extra-pair-batch-1/")
        include(row["path"], "increment_child_raw_NPZ", row["member"])
    run_directories = sorted({str(Path(row["path"]).parent) for row in increment})
    assert len(run_directories) == 16
    for directory in run_directories:
        for filename in ("settings.json", "report.json", "kappa.csv"):
            include(Path(directory) / filename, "increment_complete_settings_reports_curves")
    for filename in ("schedule-200-pairs-1.json", "schedule-200-pairs-1-completed.json"):
        include(DATA / filename, "increment_original_and_completed_schedule")
    for filename in ("quantum-rate-comparison.json", "kappa-curve.csv", "execution.json"):
        include(ANALYSIS / filename, "complete_combined_analysis")
    for filename, category in (("extra-pairs-stage-completion-exec.json", "observed_normal_pool_exit_receipt"),
                               ("recrossing-extra-pairs-200.log", "complete_extra_batch_stdout"),
                               ("recrossing-analysis-200-four-pairs.log", "complete_combined_analysis_stdout")):
        include(BASE / filename, category)
    include(publication_path, "root_compact_publication_identity", "reference/four-pair-publication-inventory.json")
    include(REVIEW / "publication-inventory.json", "independent_review_identity", "reference/four-pair-independent-review-inventory.json")
    for filename in ("independent-review.json", "original-execution.json", "original/raw-replay.json",
                     "original/input-file-identities.json", "original/raw-chunk-identities.json", "original/parent-identities.json"):
        include(REVIEW / filename, "independent_full_raw_replay_metadata", "reference/four-pair-review/" + filename)
    protocol = read(DATA / "protocol.json")
    sources = []
    for name, expected in protocol["source_sha256"].items():
        path = ROOT / ("src/pyrpmd/" + name.split(":", 1)[1] if name.startswith("loaded_native:") else name)
        assert sha(path) == expected
        sources.append({**identity(path), "registered_name": name, "included_in_archive": False})
    for name in ("benchmarks/h3/quantum_recrossing_analyze.py", "benchmarks/h3/quantum_recrossing_campaign.py"):
        sources.append({**identity(ROOT / name), "repository_path": name, "included_in_archive": False})
    replay_source = REVIEW / "replay_complete.py"
    model = SCRATCH / "literature-audit-assets/libbkmp2.so"
    assert sha(model) == protocol["model"]["library_sha256"]
    prerequisites = {"schema": "pyrpmd.h3.four-pair-increment-prerequisites.v1",
        "base_archive": identity(BASE_ARCHIVE), "base_member_inventory": identity(BASE_SIDECAR),
        "parent_archive": identity(PARENT_ARCHIVE), "parent_member_inventory": identity(PARENT_SIDECAR),
        "parent_files": parent_rows, "same_parent_configurations": 1600, "independent_chains": 8,
        "scientific_sources_and_runtime": sources, "replay_source": identity(replay_source),
        "packaging_source": identity(Path(__file__)), "repository_code_obtained_separately": True,
        "source_identity_authority": "Exact per-file hashes, not a claimed single executed Git commit.",
        "external_model": {"library": identity(model), "build_receipt": identity(model.with_suffix(".build.json")),
                           "registered_model": protocol["model"], "included_in_archive": False},
        "base_criteria": original_replay["summaries"]["2"]["criteria"],
        "combined_criteria": original_replay["summaries"]["4"]["criteria"],
        "supplementary_parent_half_equivalence_passed": False,
        "exit_observation": "observed_utc is the observed normal dispatcher exit; it is not launch time or exact last native step.",
        "first_stage_archives_unchanged": True, "no_umbrella_or_water_data": True,
        "all_raw_parent_and_base_arrays_are_external_references": True}
    write_new(OUT / "external-prerequisites.json", prerequisites)
    include(OUT / "external-prerequisites.json", "archive_prerequisite_metadata", "reference/increment-external-prerequisites.json")
    include(OUT / "INCREMENT-SCOPE.md", "scope_and_extraction_instructions", "INCREMENT-SCOPE.md")
    rows = [members[name] for name in sorted(members)]
    assert sum(row["member"].endswith(".npz") for row in rows) == 400
    inventory = {"schema": "pyrpmd.h3.quantum-child-increment-members.v1", "files": rows,
        "member_count": len(rows), "uncompressed_bytes": sum(row["bytes"] for row in rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "scope": "Exactly the completed pairIDs2,3 increment; base and original parents are immutable prerequisites."}
    write_new(SIDECAR, inventory)
    write_new(OUT / "member-inventory.json", inventory)
    with ARCHIVE.open("xb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    raw = Path(row["source_path"]).read_bytes()
                    assert len(raw) == row["bytes"] and digest(raw) == row["sha256"]
                    info = tarfile.TarInfo(row["member"])
                    info.size, info.mode, info.mtime, info.uid, info.gid = len(raw), 0o444, 0, 0, 0
                    archive.addfile(info, io.BytesIO(raw))
    print(f"Wrote increment archive: {ARCHIVE.stat().st_size} bytes; now extracting immutable union", flush=True)
    ROUNDTRIP.mkdir()
    campaign, repository = ROUNDTRIP / "campaign", ROUNDTRIP / "repository"
    campaign.mkdir(); repository.mkdir()
    base_extracted = extract_verified(BASE_ARCHIVE, read(BASE_SIDECAR)["files"], campaign)
    increment_extracted = extract_verified(ARCHIVE, rows, campaign)
    parent_members = read(PARENT_SIDECAR)["members"]
    selected_parents = {str(Path(row["path"]).relative_to(ROOT)) for parent in parent_rows for row in parent["files"].values()}
    selected_parents.add(str((EVIDENCE / "surface-partition/primary-protocol.json").relative_to(ROOT)))
    parent_extracted = extract_verified(PARENT_ARCHIVE, parent_members, repository, selected=selected_parents)
    assert len(parent_extracted) == 41
    command = [sys.executable, str(replay_source), "--data-root", str(campaign),
               "--parent-evidence", str(repository / "docs/validation/literature-convergence-2026-11-01/surface-partition"),
               "--reference-root", str(campaign / "reference"), "--output", str(OUT / "extracted-replay")]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    replay_started = time.monotonic()
    with (OUT / "extracted-replay.log").open("x") as stream:
        execution = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
    assert execution.returncode == 0, "Extraction replay failed; retain archive/extraction/log without retry or modification"
    extracted = read(OUT / "extracted-replay/raw-replay.json")
    assert extracted["summaries"] == original_replay["summaries"]
    assert extracted["numerical_replay_maximum_errors"] == original_replay["numerical_replay_maximum_errors"]
    expected_raw = {row["member"]: {k: v for k, v in row.items() if k != "path"} for row in raw_rows}
    extracted_raw = {row["member"]: {k: v for k, v in row.items() if k != "path"} for row in read(OUT / "extracted-replay/raw-chunk-identities.json")}
    assert expected_raw == extracted_raw and len(extracted_raw) == 800
    extracted_parents = read(OUT / "extracted-replay/parent-identities.json")
    for original, copied in zip(parent_rows, extracted_parents):
        assert original["seed"] == copied["seed"] and original["endpoint_q_shape"] == copied["endpoint_q_shape"]
        for name, row in original["files"].items():
            assert all(row[k] == copied["files"][name][k] for k in ("sha256", "bytes"))
    for row in rows:
        assert sha(Path(row["source_path"])) == row["sha256"]
    assert sha(BASE_ARCHIVE) == BASE_SHA and sha(PARENT_ARCHIVE) == PARENT_SHA
    assert sha(__file__) == source_sha and sha(replay_source) == prerequisites["replay_source"]["sha256"]
    ARCHIVE.chmod(0o444); SIDECAR.chmod(0o444)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    verification = {"schema": "pyrpmd.h3.quantum-child-increment-extraction-verification.v1",
        "archive": identity(ARCHIVE), "member_inventory": identity(SIDECAR),
        "member_count": len(rows), "increment_raw_NPZ_count": 400, "combined_raw_NPZ_count": 800,
        "base_members_verified_and_extracted": len(base_extracted),
        "increment_members_verified_and_extracted": len(increment_extracted),
        "parent_members_verified_and_selected": len(parent_extracted),
        "all_parent_archive_members_hash_verified": len(parent_members),
        "all_original_and_extracted_800_raw_identity_records_equal": True,
        "all_numerical_summaries_bitwise_equal_as_JSON_numbers": True,
        "all_8_parent_file_sets_hash_equal": True, "whole_original_files_rehashed_after_archiving": len(rows),
        "first_stage_archives_whole_hash_unchanged": True, "actual_extraction_root": str(ROUNDTRIP),
        "extracted_replay_command": command, "extracted_replay_returncode": execution.returncode,
        "extracted_replay_wall_seconds": time.monotonic() - replay_started,
        "extracted_replay_report": identity(OUT / "extracted-replay/raw-replay.json"),
        "fixed_P_primary_criteria": original_replay["summaries"]["4"]["criteria"],
        "supplementary_parent_half_equivalence_passed": False,
        "original_two_pair_precision_failure_retained": True, "independent_parent_chains": 8,
        "parent_configurations": 1600, "children_per_timestep": 12800, "new_dynamics_steps": 0, "PES_calls": 0,
        "packager_CPU_seconds": usage.ru_utime + usage.ru_stime,
        "replay_child_CPU_seconds": child_usage.ru_utime + child_usage.ru_stime,
        "packager_peak_RSS_KiB": usage.ru_maxrss, "elapsed_seconds": time.monotonic() - started}
    write_new(OUT / "verification.json", verification)
    write_new(OUT / "raw-data-index.json", {"schema": "pyrpmd.h3.four-pair-increment-raw-index.v1",
        "archive": identity(ARCHIVE), "sidecar": identity(SIDECAR), "raw_files": increment,
        "increment_raw_count": 400, "increment_raw_bytes": sum(row["bytes"] for row in increment),
        "prerequisite_archives": [identity(BASE_ARCHIVE), identity(PARENT_ARCHIVE)],
        "base_raw_duplicated": False, "parent_raw_duplicated": False,
        "actual_union_extraction_and_replay_passed": True, "verification": identity(OUT / "verification.json")})
    write_new(OUT / "archive-provenance.json", {"schema": "pyrpmd.h3.four-pair-increment-archive-provenance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "archive": identity(ARCHIVE),
        "member_inventory": identity(SIDECAR), "packager_source": identity(Path(__file__)),
        "replay_source": identity(replay_source), "external_prerequisites": identity(OUT / "external-prerequisites.json"),
        "raw_index": identity(OUT / "raw-data-index.json"), "verification": identity(OUT / "verification.json"),
        "root_publication_inventory": identity(publication_path), "independent_review_inventory": identity(REVIEW / "publication-inventory.json"),
        "scope_counts": inventory["category_counts"], "availability": "Local only. No upload attempted or external persistence asserted.",
        "immutability": "Exclusive creation; sorted regular PAX members, mode0444, uid/gid/mtime0; gzip level1/mtime0; archive/sidecar chmod0444 after verification.",
        "excluded": ["PES source/library", "native binaries", "repository source code", "base raw payload", "parent raw payload", "water", "umbrella", "live data"],
        "exit_time_scope": "Observed dispatcher normal-exit time; not exact launch or last integration time."})
    print(json.dumps({"archive": identity(ARCHIVE), "sidecar": identity(SIDECAR),
        "combined_raw_checkpoints": 800, "exact_extraction_replay": True,
        "packager_CPU_seconds": verification["packager_CPU_seconds"], "replay_CPU_seconds": verification["replay_child_CPU_seconds"],
        "wall_seconds": verification["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
