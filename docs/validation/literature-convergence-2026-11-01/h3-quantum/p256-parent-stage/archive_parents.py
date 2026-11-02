"""Read-only complete P256 parent audit and local checkpoint-only archive.

Numerical parent files are inaccessible through this entry point until both a
root normal-exit observation and the complete eligible eight-chain manifest
have been verified. The scientific producer is never invoked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import resource
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from benchmarks.h3 import p256_parent_campaign as verifier

OUT = Path(__file__).resolve().parent
PARENT_ROOT = ROOT.parent / "literature-convergence-quantum/p256-fixed-eight-chain-v1/parents"
DRIVER_SHA = "7934c1d7f695ce43337d11171cbad8c46ff95311490b755bf4191a56e00c0da3"
ARCHIVE = ROOT.parent / "pyrpmd-h3-quantum-p256-parents-2026-11-01.tar.gz"
SIDECAR = ARCHIVE.with_name(ARCHIVE.name.removesuffix(".tar.gz") + ".members.json")
HISTORY_ARCHIVE = ROOT.parent / "pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz"
HISTORY_SHA = "051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d"
PRODUCER_FILES = ("settings.json", "report.json", "reference.json", "reference-domain.json", "progress.json", "checkpoint.npz")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def without_paths(value):
    if isinstance(value, dict):
        return {key: without_paths(item) for key, item in value.items() if key != "path"}
    if isinstance(value, list):
        return [without_paths(item) for item in value]
    return value


def ready(parent_root, completion_receipt):
    """Only lifecycle metadata is read before this guard succeeds."""
    parent_root, completion_receipt = Path(parent_root), Path(completion_receipt)
    require(completion_receipt.is_file(), "root completion observation is not present")
    completion = read(completion_receipt)
    require(type(completion.get("exit_code")) is int and completion["exit_code"] == 0 and completion.get("all_workers_drained") is True
            and completion.get("exec_session_id") == 97158 and isinstance(completion.get("observed_utc"), str),
            "actual normal root exit and drain observation required")
    for key, name in (("root_completion_sha256", "completion.json"),
                      ("actual_tool_observation_sha256", "exec-observation.json")):
        if key in completion:
            require(sha(completion_receipt.parent / name) == completion[key], "root lifecycle observation bytes changed")
    manifest_path, registration_path = parent_root / "parents-complete.json", parent_root / "registration.json"
    require(manifest_path.is_file() and registration_path.is_file(), "all-eight global manifest is not present")
    require(sha(manifest_path) == completion.get("parents_complete_sha256")
            and sha(registration_path) == completion.get("registration_sha256"), "root completion binds other bytes")
    manifest, registration = read(manifest_path), read(registration_path)
    require(manifest.get("schema") == "pyrpmd.h3.p256-eight-parent-completion.v1"
            and manifest.get("all_eight_counts_complete_and_verified") is True
            and manifest.get("eligible_for_separately_authorized_children") is True
            and manifest.get("all_owned_workers_drained") is True, "complete eligible drained parent cohort required")
    require(tuple(manifest.get("seed_order", ())) == verifier.SEEDS
            and tuple(manifest.get("launched_seeds", ())) == verifier.SEEDS
            and manifest.get("unstarted_seeds") == [] and manifest.get("errors") == [], "incomplete or selected cohort")
    require(len(manifest.get("parents", ())) == 8
            and tuple(row["seed"] for row in manifest["parents"]) == verifier.SEEDS,
            "all eight manifest parent rows required")
    require(manifest["registration_sha256"] == sha(registration_path)
            and manifest["design_sha256"] == verifier.DESIGN_SHA256
            and registration.get("driver_sha256") == DRIVER_SHA
            and registration.get("design_sha256") == verifier.DESIGN_SHA256
            and tuple(registration.get("fixed_seeds", ())) == verifier.SEEDS,
            "registered source/design/cohort identity differs")
    require(registration.get("schema") == verifier.REGISTRATION_SCHEMA,
            "unexpected registration schema")
    return completion, manifest, registration


def audit_journal(events, seeds, worker_limit):
    """Require one successful start/reap/verification of every fixed seed."""
    active, started, reaped, verified = {}, {}, {}, set()
    drain_seen, maximum = False, 0
    for event in events:
        kind = event["event"]
        if kind == "worker_started":
            seed = event["seed"]
            require(seed in seeds and seed not in started, "duplicate or unregistered worker admission")
            require(event["cpu"] not in [row["cpu"] for row in active.values()], "concurrent CPU reuse")
            active[seed] = started[seed] = event
            maximum = max(maximum, len(active))
            require(maximum <= worker_limit, "registered worker limit exceeded")
        elif kind == "worker_reaped":
            seed = event["seed"]
            require(seed in active and event["pid"] == active[seed]["pid"] and event["returncode"] == 0,
                    "unmatched or unsuccessful worker reap")
            reaped[seed] = event
            del active[seed]
        elif kind == "seed_verified":
            require(event["seed"] in reaped and event["seed"] not in verified, "invalid seed verification order")
            verified.add(event["seed"])
        elif kind == "all_owned_workers_drained":
            require(not active and not drain_seen, "invalid or duplicated drain event")
            drain_seen = True
        else:
            raise ValueError("unexpected completed-attempt journal event")
    require(not active and drain_seen and tuple(started) == tuple(seeds)
            and set(reaped) == set(verified) == set(seeds), "incomplete drained worker journal")
    require(events[-1]["event"] == "all_owned_workers_drained", "drain must be the final event")
    return started, {"starts": len(started), "successful_reaps": len(reaped),
                     "verified_seeds": len(verified), "maximum_active_workers": maximum, "all_drained": True}


def audit_complete(parent_root, completion_receipt):
    parent_root = Path(parent_root).resolve()
    completion, manifest, registration = ready(parent_root, completion_receipt)
    require(sha(verifier.__file__) == DRIVER_SHA, "frozen parent verifier changed")
    design, original_root, guards = verifier.preflight()
    require(Path(registration["output"]).resolve() == original_root, "registered parent root differs")

    def relocated(path):
        path = Path(path).resolve()
        require(path.is_relative_to(original_root), "manifest attempt path is outside its original campaign")
        return parent_root / path.relative_to(original_root)

    plan_path = relocated(manifest["attempt_plan"]["path"])
    require(sha(plan_path) == manifest["attempt_plan"]["sha256"], "attempt plan identity changed")
    plan, attempt = read(plan_path), plan_path.parent
    require(plan["registration_sha256"] == sha(parent_root / "registration.json")
            and tuple(plan["seed_order"]) == verifier.SEEDS and 1 <= plan["workers"] <= 2,
            "attempt registration or cohort differs")
    require(plan["authorization"] == manifest["authorization"], "authorization identity differs")
    registered_authorization_path = Path(manifest["authorization"]["path"])
    authorization_path = registered_authorization_path
    execution_root = original_root.parent / "execution"
    if authorization_path.is_relative_to(execution_root):
        authorization_path = Path(completion_receipt).parent / authorization_path.relative_to(execution_root)
    require(without_paths(identity(authorization_path)) == without_paths(manifest["authorization"]),
            "adopted execution authorization changed")
    authorization = read(authorization_path)
    require(authorization["parent_adoption"] is True and authorization["driver_sha256"] == DRIVER_SHA
            and authorization["design_sha256"] == verifier.DESIGN_SHA256,
            "root did not adopt this exact design and source")
    require(authorization["workers"] == plan["workers"] and authorization["cpu_affinity"] == plan["cpu_affinity"],
            "adopted resource allocation differs")
    guards.update(plan["source_guards"])
    verifier.verify_guards(guards)
    require((attempt / "result.json").read_bytes() == (parent_root / "parents-complete.json").read_bytes(),
            "attempt result and global manifest bytes differ")
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    started, journal = audit_journal(events, verifier.SEEDS, plan["workers"])
    rows = []
    for expected, recorded in zip(design["parent_generation"]["cohorts"], manifest["parents"]):
        seed = expected["seed"]
        directory = verifier.child_path(parent_root, expected["relative_directory"])
        require({path.name for path in directory.iterdir()} == set(PRODUCER_FILES),
                "unexpected or missing final producer file")
        actual = verifier.verify_chain(directory, expected, design["parent_generation"]["sampling_flags"])
        require(actual["complete"] and actual["identity_and_geometry_verified"] and actual["sampling_flags_passed"],
                "a complete parent failed its registered identity/geometry/sampling gate")
        for name in actual["files"]:
            require(Path(recorded["files"][name]["path"]).resolve() == (original_root / expected["relative_directory"] / name).resolve(),
                    "manifest parent path differs from the frozen design route")
        require(actual["historical_replay"] == recorded["historical_replay"], "historical file-map identity differs")
        require(without_paths(actual) == without_paths(recorded), "reloaded parent differs from the global manifest")
        require(actual["parents"] == 200 and actual["beads"] == 256 and actual["completed_sweeps"] == 22000,
                "incorrect parent observations or counts")
        if seed in verifier.HISTORICAL_SEEDS:
            require(actual["historical_replay"]["bitwise_verified_array_count"] == 16,
                    "historical replay omitted original arrays")
        progress = read(directory / "progress.json")
        require(progress["completed_sweeps"] == 22000 and progress["production_sweeps"] == 20000,
                "last progress counters are incomplete")
        claim = read(attempt / f"seed-{seed}-worker-claim.json")
        execution = read(attempt / f"seed-{seed}-execution.json")
        require(claim["seed"] == execution["seed"] == seed
                and claim["pid"] == execution["pid"] == started[seed]["pid"]
                and claim["parent_pid"] == plan["coordinator_pid"], "worker claim/PID identity differs")
        require(execution["cpu_affinity"] == [started[seed]["cpu"]]
                and set(execution["thread_environment"]) == set(verifier.THREAD_VARIABLES)
                and all(value == "1" for value in execution["thread_environment"].values()),
                "worker affinity/thread identity differs")
        require(execution["failure"] is None and without_paths(execution["result"]) == without_paths(actual),
                "worker execution result differs")
        rows.append({**actual, "relative_directory": expected["relative_directory"],
                     "progress": identity(directory / "progress.json"),
                     "worker_execution": identity(attempt / f"seed-{seed}-execution.json")})
    verifier.verify_guards(guards)
    require(sha(parent_root / "parents-complete.json") == completion["parents_complete_sha256"],
            "global manifest changed while auditing")
    return {"schema": "pyrpmd.h3.p256-complete-parent-readonly-audit.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "completion_observation": identity(completion_receipt),
        "global_manifest": identity(parent_root / "parents-complete.json"),
        "registration": identity(parent_root / "registration.json"), "attempt_plan": identity(plan_path),
        "authorization": identity(authorization_path), "journal": journal,
        "parents": rows, "source_guards": guards, "frozen_verifier_sha256": DRIVER_SHA,
        "historical_array_comparisons": 64, "complete_independent_parent_chains": 8,
        "parent_configurations": 1600, "all_parent_gates_passed": True,
        "sampler_or_propagator_called": False, "PES_evaluations": 0,
        "BAR_reestimated": False, "rate_estimate_computed": False,
        "dynamic_bead_convergence_demonstrated": False}


def package(completion_receipt):
    started = time.monotonic()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    os.nice(15)
    require(not ARCHIVE.exists() and not SIDECAR.exists(), "archive is no-overwrite")
    source_sha = sha(__file__)
    audit = audit_complete(PARENT_ROOT, completion_receipt)
    require(sha(HISTORY_ARCHIVE) == HISTORY_SHA, "external historical archive changed")
    write_new(OUT / "complete-audit.json", audit)
    members = {}

    def include(path, member, category):
        path, parsed = Path(path), PurePosixPath(member)
        require(path.is_file() and not path.is_symlink(), "regular immutable input required")
        require(not parsed.is_absolute() and ".." not in parsed.parts and "\\" not in member
                and member not in members, "unsafe or duplicate archive member")
        require(path.suffix in (".npz", ".json", ".jsonl", ".log", ".md", ".xml"), "source or binary is excluded")
        members[member] = {"member": member, "source_path": str(path.resolve()),
                           "category": category, "sha256": sha(path), "bytes": path.stat().st_size}

    for row in audit["parents"]:
        for name in PRODUCER_FILES:
            path = PARENT_ROOT / row["relative_directory"] / name
            include(path, "parents/" + str(path.relative_to(PARENT_ROOT)),
                    "current_parent_checkpoint" if name == "checkpoint.npz" else "current_parent_metadata")
    for name in ("registration.json", "parents-complete.json"):
        include(PARENT_ROOT / name, "parents/" + name, "campaign_registration_and_completion")
    # Root's observed exit and the global drain proof make all attempt files
    # immutable. Preserve failed-attempt metadata too, if a reviewed resume
    # ever occurred; do not silently choose only the successful attempt.
    for path in sorted((PARENT_ROOT / "attempts").rglob("*")):
        if path.is_file():
            require(path.suffix in (".json", ".jsonl", ".log"), "unexpected attempt payload")
            include(path, "parents/" + str(path.relative_to(PARENT_ROOT)), "all_attempt_lifecycle_metadata")
    for path in sorted(Path(completion_receipt).parent.iterdir()):
        require(path.is_file() and path.suffix in (".json", ".jsonl", ".log"), "unexpected root execution payload")
        include(path, "root-execution/" + path.name, "observed_root_execution_metadata")
    include(verifier.DESIGN, "metadata/fixed-eight-chain-design.json", "fixed_design_metadata")
    for name in ("archive-plan.json", "manifest-path-correction.json", "complete-audit.json"):
        include(OUT / name, "metadata/" + name, "archive_audit_metadata")
    authorization = Path(audit["authorization"]["path"])
    include(authorization, "metadata/execution-authorization.json", "execution_adoption")
    external = {"schema": "pyrpmd.h3.p256-parent-archive-prerequisites.v1",
        "historical_archive": identity(HISTORY_ARCHIVE),
        "historical_source_files": {str(row["seed"]): row["historical_replay"]["original_files"] for row in audit["parents"] if row["historical_replay"]},
        "scientific_sources_and_inputs": audit["source_guards"],
        "external_authorization": audit["authorization"],
        "verifier_source": identity(Path(verifier.__file__)), "archive_helper_source": identity(Path(__file__)),
        "source_and_model_files_included": False,
        "replay_requires": "Matching review checkout and registered external historical/source/model identities; source code is supplied by Git, not this raw archive.",
        "scope": "Current8P256 checkpoints and metadata only; no child trajectory or dynamic-bead claim."}
    write_new(OUT / "external-prerequisites.json", external)
    include(OUT / "external-prerequisites.json", "metadata/external-prerequisites.json", "external_identity_metadata")
    rows = [members[name] for name in sorted(members)]
    require(sum(row["category"] == "current_parent_checkpoint" for row in rows) == 8, "archive must contain exactly eight raw checkpoints")
    require(sum(row["member"].endswith(".npz") for row in rows) == 8, "unregistered raw payload")
    inventory = {"schema": "pyrpmd.h3.p256-parent-archive-members.v1", "files": rows,
        "member_count": len(rows), "raw_checkpoint_count": 8,
        "uncompressed_bytes": sum(row["bytes"] for row in rows)}
    write_new(SIDECAR, inventory)
    write_new(OUT / "member-inventory.json", inventory)
    with ARCHIVE.open("xb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", filename="", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    raw = Path(row["source_path"]).read_bytes()
                    require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "source changed during packaging")
                    info = tarfile.TarInfo(row["member"])
                    info.size, info.mode, info.mtime, info.uid, info.gid = len(raw), 0o444, 0, 0, 0
                    archive.addfile(info, io.BytesIO(raw))
    extracted = ROOT.parent / "h3-p256-parent-archive-roundtrip-2026-11-01"
    extracted.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            require(member.isfile() and member.name in members and member.name not in seen, "unexpected archive member")
            raw = archive.extractfile(member).read()
            expected = members[member.name]
            require(len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"], "archived bytes differ")
            path = extracted / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
            seen.add(member.name)
    require(seen == set(members), "missing archive member")
    replay = audit_complete(extracted / "parents", extracted / "root-execution" / Path(completion_receipt).name)
    require([without_paths(row) for row in replay["parents"]] == [without_paths(row) for row in audit["parents"]],
            "extracted complete-parent results differ")
    require(replay["journal"] == audit["journal"], "extracted lifecycle differs")
    write_new(OUT / "extracted-complete-audit.json", replay)
    for row in rows:
        require(sha(Path(row["source_path"])) == row["sha256"], "original input changed after packaging")
    require(sha(HISTORY_ARCHIVE) == HISTORY_SHA and sha(__file__) == source_sha,
            "external history or archive helper changed")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {"schema": "pyrpmd.h3.p256-parent-archive-verification.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "archive": identity(ARCHIVE),
        "member_inventory": identity(SIDECAR), "member_count": len(rows), "raw_checkpoint_count": 8,
        "all_members_extracted_and_hashed": True, "all_eight_extracted_parents_reverified": True,
        "historical_array_comparisons_per_pass": 64, "independent_parent_chains": 8, "parent_configurations": 1600,
        "exact_original_and_extracted_results": True, "original_inputs_unchanged": True,
        "source": identity(Path(__file__)), "frozen_verifier": identity(Path(verifier.__file__)),
        "actual_extraction_root": str(extracted), "CPU_seconds": usage.ru_utime + usage.ru_stime,
        "peak_RSS_KiB": usage.ru_maxrss, "wall_seconds": time.monotonic() - started,
        "PES_evaluations": 0, "new_dynamics_steps": 0, "upload_attempted": False,
        "immutability_scope": "Exclusive creation, no overwrite, frozen bytes/SHA/member inventory; no durable filesystem write-protection claim."}
    write_new(OUT / "archive-verification.json", result)
    write_new(OUT / "raw-data-index.json", {"schema": "pyrpmd.h3.p256-parent-raw-index.v1",
        "archive": identity(ARCHIVE), "member_inventory": identity(SIDECAR),
        "raw_files": [row for row in rows if row["category"] == "current_parent_checkpoint"],
        "raw_checkpoint_count": 8, "parent_chains": 8, "parent_configurations": 1600,
        "historical_raw_duplicated": False, "source_or_model_included": False,
        "complete_audit": identity(OUT / "complete-audit.json"), "verification": identity(OUT / "archive-verification.json")})
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-receipt", type=Path, required=True)
    args = parser.parse_args()
    package(args.completion_receipt)
