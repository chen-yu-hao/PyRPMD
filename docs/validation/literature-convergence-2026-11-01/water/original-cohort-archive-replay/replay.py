"""One bounded real extraction/reanalysis of the sealed original water archive."""
from __future__ import annotations

import time
ENTRY_WALL = time.monotonic()

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = Path(__file__).resolve().parent
WATER = EVIDENCE.parent
ORIGINAL = WATER / "original-fixed-cohort"
FORBIDDEN_RAW = ROOT.parent / "water-convergence-2026-11-01"
WORK = ROOT.parent / "water-raw-archive-replay-2026-11-01"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS")
RECONSTRUCTION_COMMIT = "c2c5ae2dce253f1407e7de25ec343ef5e1844a73"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    os.sched_setaffinity(0, {5})
    os.nice(10)
    for name in THREADS:
        os.environ[name] = "1"

    def expired(_signal, _frame):
        raise TimeoutError("bounded archive replay exceeded 600 wall seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 600 - (time.monotonic() - ENTRY_WALL)))
    WORK.mkdir(exist_ok=False)
    if sha(ORIGINAL / "raw-data-index.json") != "355e72b3fb035940693c685ef6d12dda01a7557851c5415be4fc5243fb176472":
        raise ValueError("the frozen raw-archive index changed")
    inputs = json.loads((ORIGINAL / "raw-data-index.json").read_text())
    reference = json.loads((ORIGINAL / "analysis/report.json").read_text())
    expected_report_sha = "f544d5de9202a73f7013ee5191fed4e53e330a9eeb41c87770bfaaa3f7d4d1f8"
    if sha(ORIGINAL / "analysis/report.json") != expected_report_sha:
        raise ValueError("the frozen original report changed")
    source_guards = {str(ROOT / name): expected for name, expected in reference["source_sha256"].items()}
    source_guards[str(Path(__file__).resolve())] = sha(__file__)
    for name, expected in source_guards.items():
        if sha(name) != expected:
            raise ValueError("the frozen analyzer dependency changed")
    extraction = WORK / "extracted"
    extraction.mkdir()
    archive = Path(inputs["archive"]["path"])
    if sha(archive) != inputs["archive"]["sha256"]:
        raise ValueError("the sealed archive changed")
    top = inputs["archive"]["top_level_directory"]
    members = []
    with tarfile.open(archive, mode="r|gz") as container:
        for member in container:
            relative = Path(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or relative.parts[0] != top):
                raise ValueError("unexpected archive member type or path")
            target = extraction / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with container.extractfile(member) as stream, target.open("xb") as destination:
                for block in iter(lambda: stream.read(1048576), b""):
                    destination.write(block)
            members.append(member.name)
    if len(members) != inputs["archive"]["members"] or len(set(members)) != len(members):
        raise ValueError("archive membership differs")
    data_root = extraction / top
    checksums = {}
    for line in (data_root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha(data_root / name) != expected:
            raise ValueError(f"actual extracted-file hash differs: {name}")
        checksums[name] = expected
    if len(checksums) + 1 != len(members):
        raise ValueError("extracted checksum coverage differs")

    # Existing archived definitions/configs are used directly, after checking
    # their exact Git identity. The omitted source patch is reconstructed only
    # in this replay workspace; the data archive remains unchanged.
    input_root = data_root / "inputs"
    names = ["protocol.json", "checkpoint-contract-v2.json", "paired-equivalence-preregistration.json"]
    names += [f"configs/classical-seed-{seed}.json" for seed in range(1101, 1109)]
    names += ["checkpoint-contract-v2.patch"]
    reconstruction = []
    for name in names:
        git_path = str(WATER.relative_to(ROOT) / name)
        blob = subprocess.check_output(["git", "show", f"{RECONSTRUCTION_COMMIT}:{git_path}"], cwd=ROOT, timeout=10)
        target = input_root / name
        existed = target.exists()
        if existed:
            if target.read_bytes() != blob:
                raise ValueError(f"extracted protocol/config does not match frozen Git bytes: {name}")
        else:
            if name != "checkpoint-contract-v2.patch":
                raise ValueError("unexpected missing archived input")
            with target.open("xb") as stream:
                stream.write(blob)
        reconstruction.append({"input": name, "git_commit": RECONSTRUCTION_COMMIT,
                               "git_path": git_path, "sha256": sha(target),
                               "existing_extracted_payload_used": existed,
                               "source_patch_reconstructed_outside_archive": not existed})

    phase = {"name": "imports"}
    numerical_reads, config_reads, original_attempts = set(), set(), []
    generated_output_reads = set()

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path == FORBIDDEN_RAW or FORBIDDEN_RAW in path.parents:
            original_attempts.append(str(path))
            raise RuntimeError("archive replay attempted original raw-data access")
        if phase["name"] != "analyze":
            return
        mode, flags = args[1], args[2]
        reading = (isinstance(mode, str) and "r" in mode) or (
            isinstance(flags, int) and flags & os.O_ACCMODE == os.O_RDONLY)
        if reading and path.suffix == ".npz":
            if path == WORK / "analysis/curves.npz":
                # The frozen analyzer hashes its newly generated output before
                # returning. This is not an input trajectory read.
                generated_output_reads.add(str(path))
                return
            if data_root not in path.parents:
                raise RuntimeError("numerical trajectory read escaped the extracted payload")
            numerical_reads.add(str(path.relative_to(data_root)))
        if reading and "configs" in path.parts and path.suffix == ".json":
            if input_root not in path.parents:
                raise RuntimeError("configuration read escaped the extracted payload")
            config_reads.add(str(path.relative_to(data_root)))

    sys.addaudithook(audit)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from benchmarks.water import analyze_native, native_reference
    imported_sources = {}
    for name, expected in reference["source_sha256"].items():
        module_name = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
        actual_path = Path(sys.modules[module_name].__file__).resolve()
        if actual_path != ROOT / name or sha(actual_path) != expected:
            raise ValueError("an imported analyzer dependency differs from its frozen source")
        imported_sources[module_name] = {"path": str(actual_path), "sha256": expected}
    engine_attempts = []

    def no_engine(*_args, **_kwargs):
        engine_attempts.append(True)
        raise RuntimeError("offline archive replay must not create an engine")

    original_factory = native_reference.create_engine
    native_reference.create_engine = no_engine
    phase["name"] = "analyze"
    try:
        replay = analyze_native.analyze([data_root / "raw/runs" / f"classical-{seed}" for seed in range(1101, 1109)],
                                        input_root / "protocol.json", WORK / "analysis")
    finally:
        native_reference.create_engine = original_factory
        phase["name"] = "compare_reference"
    expected_numeric = {f"raw/runs/classical-{seed}/chunk-{number:06d}.npz"
                        for seed in range(1101, 1109) for number in range(200)}
    expected_configs = {f"inputs/configs/classical-seed-{seed}.json" for seed in range(1101, 1109)}
    if numerical_reads != expected_numeric or config_reads != expected_configs or original_attempts or engine_attempts:
        raise ValueError("actual extracted trajectory/config read coverage differs")
    left, right = dict(reference), dict(replay)
    original_artifacts, replay_artifacts = left.pop("artifacts"), right.pop("artifacts")
    if left != right:
        raise AssertionError("physical/statistical/gate fields differ from the frozen complete cohort")
    curve_records = []
    with np.load(ORIGINAL / "analysis/curves.npz", allow_pickle=False) as first, np.load(WORK / "analysis/curves.npz", allow_pickle=False) as second:
        if set(first.files) != set(second.files):
            raise AssertionError("compact curve names differ")
        for name in sorted(first.files):
            a, b = first[name], second[name]
            if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
                raise AssertionError(f"compact physical curve differs: {name}")
            curve_records.append({"name": name, "shape": list(a.shape), "dtype": a.dtype.str,
                                  "contiguous_sha256": hashlib.sha256(a.tobytes()).hexdigest()})
    for name, expected in source_guards.items():
        if sha(name) != expected:
            raise AssertionError("source changed during replay")
    usage, children = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {"schema": "pyrpmd.water-original-archive-extraction-replay.v1", "completed_utc": datetime.now(timezone.utc).isoformat(),
              "all_checks_passed": True, "archive": inputs["archive"], "actual_extraction_root": str(data_root),
              "extracted_members": len(members), "extracted_files_verified_by_checksums": len(checksums),
              "source_reconstruction": reconstruction, "all_physical_statistical_and_gate_fields_exactly_equal": True,
              "only_excluded_from_JSON_field_comparison": "artifacts container hashes, independently verified as identical curve arrays below",
              "original_artifacts": original_artifacts, "replay_artifacts": replay_artifacts,
              "original_report_sha256": expected_report_sha, "replay_report_sha256": sha(WORK / "analysis/report.json"),
              "whole_report_byte_identical": sha(WORK / "analysis/report.json") == expected_report_sha,
              "original_raw_access_attempts": original_attempts, "engine_creation_attempts": len(engine_attempts),
              "physical_steps": 0, "PES_evaluations": 0, "source_guards_unchanged": source_guards,
              "actual_imported_sources": imported_sources,
              "affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0), "threads": {x: os.environ[x] for x in THREADS},
              "CPU_seconds": usage.ru_utime + usage.ru_stime, "Git_reader_CPU_seconds": children.ru_utime + children.ru_stime,
              "wall_seconds": time.monotonic() - ENTRY_WALL, "peak_RSS_KiB": usage.ru_maxrss,
              "CPU_hard_limit_seconds": 300, "wall_limit_seconds": 600}
    write_new(EVIDENCE / "read-audit.json", {"numerical_trajectory_NPZ_reads": sorted(numerical_reads),
                                            "configuration_reads": sorted(config_reads),
                                            "generated_curve_hash_reads": sorted(generated_output_reads),
                                            "curve_array_comparisons": curve_records})
    result["read_audit_sha256"] = sha(EVIDENCE / "read-audit.json")
    write_new(EVIDENCE / "report.json", result)
    print(json.dumps({name: result[name] for name in ("all_checks_passed", "whole_report_byte_identical", "CPU_seconds", "wall_seconds", "peak_RSS_KiB")}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure = EVIDENCE / "failure.json"
        if not failure.exists():
            write_new(failure, {"error_type": type(error).__name__, "message": str(error),
                                "created_utc": datetime.now(timezone.utc).isoformat(),
                                "all_partial_extracted_data_retained": True})
        raise
