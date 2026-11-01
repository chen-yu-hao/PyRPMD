"""Exact-byte local archive of one completed excluded HCH4 efficiency pilot.

Only the explicit data/receipt/reference whitelist is copied. Git-backed Python
source, native libraries and third-party PES files remain external identities.
The fresh-extraction replay reads saved values; it cannot evaluate the PES.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time

DATA_NAMES = (
    "PREPARATION_ONCE.json", "RUN_ONCE.json", "WORKER_ONCE.json", "calibration-followup-ledger.json",
    "calibration-initial-ledger.json", "calls.jsonl", "child-started.json", "control-results.json",
    "diagnostic-ledger.json", "frozen-proposal.json", "main-results.json", "registration.json",
    "report.json", "runtime.json", "supervisor.json", "worker.log",
)
EXECUTION_NAMES = (
    "execution-plan.json", "prepare-exec-observation.json", "prepare.log", "root-adoption-verification.json",
    "root-resource-adoption.json", "run-exec-observation.json", "run.log",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    assert path.is_file() and not path.is_symlink()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def new_json(path, value):
    with Path(path).open("x") as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False); out.write("\n")


def package(root, data, destination):
    start = time.monotonic(); root, data, destination = map(Path, (root, data, destination))
    base = Path(__file__).resolve().parent
    review = root/"docs/validation/literature-convergence-2026-11-01/review"
    execution = review/"hch4-reference-efficiency-execution"
    assert {p.name for p in data.iterdir()} == set(DATA_NAMES), "unexpected or missing original data member"
    assert {p.name for p in execution.iterdir()} == set(EXECUTION_NAMES), "root receipt set changed; retain and inspect"
    protocol = json.loads((base/"replay-protocol.json").read_text())
    for item in protocol["actual_data_files"]:
        assert record(item["path"]) == item
    observation = json.loads((execution/"run-exec-observation.json").read_text())
    assert observation["exec_session_id"] == 70968 and type(observation["tool_result"]["exit_code"]) is int and observation["tool_result"]["exit_code"] == 0
    assert sha(data/"report.json") == "5162a97ad5e2fd981a039898020c3798e6b0ab957cd56e5ff3153c18aa0d58fc"
    supervisor = json.loads((data/"supervisor.json").read_text())
    assert supervisor["all_owned_workers_drained"] is True and supervisor["worker_exit_code"] == 0
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination/"pyrpmd-hch4-efficiency-pilot-2026-11-01-v1.tar.gz"
    partial = archive.with_suffix(archive.suffix+".partial")
    restored = destination/"fresh-extraction"; restored.mkdir()
    pairs = [(data/name, "data/"+name, "complete_actual_pilot") for name in DATA_NAMES]
    pairs += [(execution/name, "root-execution/"+name, "actual_execution_receipt") for name in EXECUTION_NAMES]
    references = {
        "efficiency-design.json": review/"hch4-reference-efficiency-design/proposal.json",
        "physics-coarea-proposal.json": review/"hch4-reference-feasibility/pilot-design/proposal.json",
        "geometry.json": root/"benchmarks/hch4/geometries.json",
        "design-method-review.json": review/"hch4-reference-efficiency-design/independent-method-review.json",
        "implementation-inventory.json": review/"hch4-reference-efficiency-implementation/publication-inventory.json",
        "implementation-method-review.json": review/"hch4-reference-efficiency-implementation/independent-method-review.json",
        "implementation-method-identity-update.json": review/"hch4-reference-efficiency-implementation/independent-method-identity-update.json",
        "implementation-lifecycle-review.json": review/"hch4-reference-efficiency-implementation/independent-lifecycle-review.json",
        "replay-protocol.json": base/"replay-protocol.json",
    }
    pairs += [(path, "reference/"+name, "fixed_reference_metadata") for name, path in references.items()]
    pairs += [(base/"actual-replay-attempt-1"/name, "calculation-replay/"+name, "independent_calculation_receipt")
              for name in ("command.json", "execution.json", "replay-result.json", "replay.log")]
    entries = [{"source": record(path), "member": member, "category": category} for path, member, category in pairs]
    assert len({r["member"] for r in entries}) == len(entries) == 36
    assert all(not r["member"].endswith((".py", ".so", ".dll", ".dylib", ".f90")) for r in entries)
    new_json(destination/"registered-members.json", {"entries": entries, "source": record(__file__), "PES_calls": 0, "random_draws": 0})
    with partial.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for row in entries:
                    assert record(row["source"]["path"]) == row["source"]
                    info = tarfile.TarInfo(row["member"]); info.size = row["source"]["bytes"]; info.mode = 0o444; info.mtime = 0
                    with Path(row["source"]["path"]).open("rb") as stream: tar.addfile(info, stream)
        raw.flush(); os.fsync(raw.fileno())
    expected = {row["member"]: row for row in entries}; seen = set()
    with tarfile.open(partial, "r:gz") as tar:
        for item in tar:
            path = PurePosixPath(item.name)
            assert item.name in expected and item.name not in seen and item.isfile() and not item.issparse()
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in item.name and path.as_posix() == item.name
            raw = tar.extractfile(item).read(); row = expected[item.name]
            assert len(raw) == row["source"]["bytes"] and hashlib.sha256(raw).hexdigest() == row["source"]["sha256"]
            copied = restored/item.name; copied.parent.mkdir(parents=True, exist_ok=True)
            with copied.open("xb") as stream: stream.write(raw)
            assert sha(copied) == row["source"]["sha256"]
            seen.add(item.name)
    assert seen == set(expected)
    command = [sys.executable, str(base/"replay_saved.py"), "--data", str(restored/"data"),
               "--physics", str(restored/"reference/physics-coarea-proposal.json"), "--design", str(restored/"reference/efficiency-design.json"),
               "--output", str(destination/"fresh-replay-result.json")]
    new_json(destination/"fresh-replay-command.json", {"command": command, "source": record(base/"replay_saved.py"), "actual_PES_or_RNG_calls": 0})
    with (destination/"fresh-replay.log").open("x") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    assert completed.returncode == 0, "fresh extraction calculation failed; preserve partial archive and log"
    first = json.loads((base/"actual-replay-attempt-1/replay-result.json").read_text())
    second = json.loads((destination/"fresh-replay-result.json").read_text())
    keys = ("analysis_recomputed", "calibration_recomputed", "comparison_counts", "maximum_absolute_reconstruction_errors",
            "staged_geometries", "actual_saved_PES_calls", "retained_geometric_zero_draws")
    assert all(first[key] == second[key] for key in keys), "fresh extraction scientific calculation differs"
    for row in entries: assert record(row["source"]["path"]) == row["source"], "original archive input changed"
    os.link(partial, archive)
    inventory = {"schema": "pyrpmd.hch4-efficiency-raw-archive-members.v1", "archive": record(archive),
                 "entries": entries, "member_count": len(entries), "member_bytes": sum(r["source"]["bytes"] for r in entries),
                 "all_original_member_extracted_hashes_equal": True, "exact_fresh_calculation_equal_fields": list(keys),
                 "external_program_source": record(base/"replay_saved.py"), "PES_and_native_libraries_included": False,
                 "producer_source_included": False, "upload_performed": False, "new_PES_calls": 0, "new_random_draws": 0,
                 "scope": "Complete excluded4000-geometry pilot and saved-data reconstruction only; all scientific screen failures retained.",
                 "wall_seconds_before_receipt": time.monotonic()-start}
    new_json(destination/"member-inventory.json", inventory)
    print(json.dumps({"archive": inventory["archive"], "member_count": len(entries), "exact_fresh_calculation_equal": True}, indent=2))
    return inventory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "data", "destination"): parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    package(args.root, args.data, args.destination)
