"""Package only the immutable base 200-fs/two-pair child cohort, locally.

No simulation, source mutation, recursive live-data discovery, Git operation,
or upload occurs. Whole input bytes and checkpoint integrity are verified
before packing and after reloading every archived chunk.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import tarfile
import time

import numpy as np

from pyrpmd.checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parents[6]
SCRATCH = ROOT.parent
EVIDENCE = ROOT / "docs/validation/literature-convergence-2026-11-01"
OUT = Path(__file__).resolve().parent
BASE = SCRATCH / "literature-convergence-quantum"
DATA = BASE / "recrossing-primary"
ANALYSIS = BASE / "recrossing-analysis-200"
SEEDS = (721301, 721303, 721307, 721309, 721313, 721317, 721321, 721323)
PROTOCOL_SHA = "53203aeca4fdf66d9fbf1a23fe7a430ab99e24d18c351e85bd2eb383a87d3589"
ARCHIVE = SCRATCH / "pyrpmd-h3-quantum-recrossing-stage-200-pairs-2-2026-11-01.tar.gz"
SIDECAR = ARCHIVE.with_name(ARCHIVE.name.removesuffix(".tar.gz") + ".members.json")
PARENT_ARCHIVE = SCRATCH / "pyrpmd-h3-quantum-equilibrium-2026-11-01.tar.gz"
PARENT_ARCHIVE_SHA = "051ca8d016c722b740bc340f07387b72cf14c3379c21394a171dd34f083e944d"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def record(path):
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def payload_description(saved):
    return {name: {"shape": list(saved[name].shape), "dtype": saved[name].dtype.str,
                   "sha256": digest(np.ascontiguousarray(saved[name]).tobytes())}
            for name in sorted(saved.files)}


def check_chunk(data, settings, left):
    """Reload exact saved bytes without evaluating a PES or integrating atoms."""
    with load_checkpoint(io.BytesIO(data), expected_kind="h3.quantum-surface-children",
                         require_integrity=True) as saved:
        indices = np.arange(left, left + 8)
        assert str(saved["fingerprint"].item()) == fingerprint(settings)
        np.testing.assert_array_equal(saved["parent_index"], indices)
        np.testing.assert_array_equal(saved["parent_completed_sweeps"], 2100 + 100 * indices)
        np.testing.assert_array_equal(saved["momentum_pair_index"], [0, 1])
        np.testing.assert_array_equal(saved["time_fs"], np.arange(1, 201))
        assert saved["surface_values"].shape == (8, 2, 2, 200)
        assert saved["minimum_pair_at_1fs"].shape == (8, 2, 2, 201)
        assert saved["initial_p"].shape == (8, 2, 128, 3, 3)
        assert saved["final_q"].shape == saved["final_p"].shape == (8, 2, 2, 128, 3, 3)
        assert saved["energy_extrema"].shape == (8, 2, 2, 3)
        assert str(saved["previous_chunk_sha256"].item()) == ""
        np.testing.assert_array_equal(saved["final_step"], np.full((8, 2, 2), round(200 / settings["timestep_fs"])))
        assert int(saved["native_steps_this_stage"]) == 8 * 2 * 2 * round(200 / settings["timestep_fs"])
        for name in saved.files:
            value = saved[name]
            if value.dtype.kind in "fc":
                assert np.isfinite(value).all(), name
        sign = ((saved["surface_values"][:, :, 0] > 0).astype(float)
                - (saved["surface_values"][:, :, 1] > 0))
        numerator = saved["velocity"][..., None] / saved["flux_metric"][:, None, None] * sign
        denominator = abs(saved["velocity"]) / saved["flux_metric"][:, None]
        np.testing.assert_array_equal(saved["numerator"], numerator)
        np.testing.assert_array_equal(saved["denominator"], denominator)
        payload_sha = fingerprint(payload_description(saved))
        return {"payload_descriptor_sha256": payload_sha,
                "checkpoint_kind": saved.envelope["kind"],
                "schema_version": saved.envelope["schema_version"],
                "integrity_envelope_verified": True,
                "native_steps": int(saved["native_steps_this_stage"]),
                "initial_p_sha256": digest(saved["initial_p"].tobytes()),
                "channel_sha256": digest(saved["channel"].tobytes())}


def main():
    started = time.monotonic()
    assert not ARCHIVE.exists() and not SIDECAR.exists(), "archive is immutable: never overwrite"
    builder_sha = sha(Path(__file__))
    protocol = json.loads((DATA / "protocol.json").read_text())
    assert sha(DATA / "protocol.json") == PROTOCOL_SHA
    assert tuple(protocol["seeds"]) == SEEDS
    assert protocol["parents_per_seed"] == 200 and protocol["beads"] == 128
    normalized_protocol = dict(protocol)
    normalized_protocol.pop("created_before_primary_children_utc")
    completed = json.loads((DATA / "schedule-200-pairs-0-completed.json").read_text())
    schedule = json.loads((DATA / "schedule-200-pairs-0.json").read_text())
    assert completed["inventory"] == schedule and schedule["pair_batch"] == 0
    assert tuple(row["seed"] for row in completed["completed"]) == SEEDS
    assert all(row["complete"] for row in completed["completed"])
    report = json.loads((ANALYSIS / "quantum-rate-comparison.json").read_text())
    assert tuple(report["expected_seeds"]) == tuple(report["completed_seeds"]) == SEEDS
    assert report["pair_batches"] == [0] and report["horizon_fs"] == 200
    assert report["protocol_sha256"] == PROTOCOL_SHA
    assert report["fixed_P_independent_rate_gates_passed"] is False
    decision = json.loads((ANALYSIS / "precision-extension-decision.json").read_text())
    assert decision["complete_report_sha256"] == sha(ANALYSIS / "quantum-rate-comparison.json")
    assert decision["complete_execution_sha256"] == sha(ANALYSIS / "execution.json")
    assert decision["trigger"]["criterion"] == "kappa_statistical_precision"
    assert sha(PARENT_ARCHIVE) == PARENT_ARCHIVE_SHA

    members = {}

    def include(path, category, member=None):
        path = Path(path)
        assert path.is_file() and not path.is_symlink()
        assert "extra-pair-batch-1" not in path.parts
        assert path.suffix not in (".py", ".so", ".f", ".f90", ".dll", ".dylib")
        name = member if member else str(path.relative_to(BASE))
        assert name not in members and not Path(name).is_absolute() and ".." not in Path(name).parts
        members[name] = {"member": name, "source_path": str(path), "category": category,
                         "sha256": sha(path), "bytes": path.stat().st_size}

    for name in ("protocol.json", "schedule-200-pairs-0.json", "schedule-200-pairs-0-completed.json"):
        include(DATA / name, "base_protocol_and_schedule")
    for name in ("quantum-rate-comparison.json", "kappa-curve.csv", "execution.json",
                 "precision-extension-decision.json", "precision-extension-launch-receipt.json"):
        include(ANALYSIS / name, "completed_base_analysis_and_continuation_metadata")
    include(BASE / "recrossing-analysis-200.log", "base_analysis_log")
    include(BASE / "recrossing-base-final-tool-output.json", "partial_truncated_base_terminal_receipt")
    references = {
        "surface-partition/primary-protocol.json": "parent-protocol.json",
        "h3-classical/transmission/transmission-comparison.json": "classical-transmission-reference.json",
        "h3-quantum/diatom-transfer.json": "finite-P-reactant-reference.json",
        "h3-quantum/recrossing-analysis-prespecification.json": "analysis-prespecification.json",
    }
    for source, target in references.items():
        include(EVIDENCE / source, "analysis_reference", "reference/" + target)

    external_parents, runs, raw = [], [], []
    pairing = {}
    for seed in SEEDS:
        parent = EVIDENCE / "surface-partition/primary/ts" / f"seed-{seed}"
        parent_files = {name: record(parent / name) for name in
                        ("checkpoint.npz", "settings.json", "report.json", "reference-domain.json")}
        parent_report = json.loads((parent / "report.json").read_text())
        assert parent_files["checkpoint.npz"]["sha256"] == parent_report["checkpoint_sha256"]
        with load_checkpoint(parent / "checkpoint.npz", expected_kind="h3.surface-partition",
                             require_integrity=True) as saved:
            assert saved["endpoint_q"].shape == (200, 128, 3, 3)
            np.testing.assert_array_equal(saved["endpoint_completed_sweeps"], np.arange(2100, 22001, 100))
        external_parents.append({"seed": seed, "files": parent_files,
                                 "available_from": str(PARENT_ARCHIVE), "included_here": False})
        for dt in (0.1, 0.05):
            run = DATA / f"seed-{seed}" / f"dt-{dt:g}" / "horizon-200"
            settings = json.loads((run / "settings.json").read_text())
            run_report = json.loads((run / "report.json").read_text())
            assert settings["seed"] == seed and settings["timestep_fs"] == dt
            assert settings["pair_batch"] == 0 and settings["momentum_pair_indices"] == [0, 1]
            assert settings["horizon_fs"] == 200 and settings["start_horizon_fs"] == 0
            assert settings["parents"] == 200 and settings["beads"] == 128 and settings["chunk_parents"] == 8
            assert settings["protocol_sha256"] == fingerprint(normalized_protocol)
            assert settings["source_sha256"] == protocol["source_sha256"]
            assert run_report["complete"] and run_report["source_unchanged"]
            assert run_report["settings_sha256"] == sha(run / "settings.json")
            for filename in ("checkpoint.npz", "settings.json", "report.json"):
                assert settings["parent_identity"][filename.split(".")[0] + "_sha256"] == parent_files[filename]["sha256"]
            matched = next(row for row in report["per_timestep_per_seed"][str(dt)] if row["seed"] == seed)
            receipt = matched["receipts"][0]
            assert len(matched["receipts"]) == 1 and receipt["batch"] == 0
            assert receipt["report_sha256"] == sha(run / "report.json")
            assert receipt["settings_sha256"] == sha(run / "settings.json")
            assert receipt["chunks"] == run_report["chunks"]
            chunk_hashes = {row["file"]: row["sha256"] for row in run_report["chunks"]}
            expected = {f"chunk-{left:04d}.npz" for left in range(0, 200, 8)}
            assert len(run_report["chunks"]) == 25 and set(chunk_hashes) == expected
            assert {path.name for path in run.iterdir()} == expected | {"settings.json", "report.json", "kappa.csv"}
            for filename in ("settings.json", "report.json", "kappa.csv"):
                include(run / filename, "base_seed_settings_reports_curves")
            for left in range(0, 200, 8):
                path = run / f"chunk-{left:04d}.npz"
                data = path.read_bytes()
                assert digest(data) == chunk_hashes[path.name]
                checked = check_chunk(data, settings, left)
                include(path, "base_child_raw_NPZ")
                row = {**members[str(path.relative_to(BASE))], "seed": seed, "timestep_fs": dt,
                       "first_parent_index": left, "parents": 8, **checked}
                raw.append(row)
                key = (seed, left)
                pair = (checked["initial_p_sha256"], checked["channel_sha256"])
                if key in pairing:
                    assert pairing[key] == pair
                else:
                    pairing[key] = pair
            runs.append({"seed": seed, "timestep_fs": dt, "directory": str(run.relative_to(BASE)),
                         "chunks": 25, "parents": 200, "children": 800,
                         "settings": record(run / "settings.json"), "report": record(run / "report.json"),
                         "physical_converged_flag_retained": run_report["physical_converged"]})
            print(f"Verified base seed {seed}, dt={dt:g}: 25 immutable chunks", flush=True)

    review_path = EVIDENCE / "review/pr68-software-review/complete-200fs-review/input-file-hashes.json"
    reviewed = json.loads(review_path.read_text())
    assert len(reviewed) == 483
    assert all("extra-pair-batch-1" not in Path(path).parts for path in reviewed)
    for path, expected in reviewed.items():
        assert sha(Path(path)) == expected, path
    assert all(reviewed[row["source_path"]] == row["sha256"] for row in raw)
    include(review_path, "independent_review_input_identities", "reference/independent-review-input-file-hashes.json")

    sources = []
    for name, expected in protocol["source_sha256"].items():
        path = ROOT / ("src/pyrpmd/" + name.split(":", 1)[1] if name.startswith("loaded_native:") else name)
        assert sha(path) == expected
        sources.append({**record(path), "registered_name": name, "included_in_archive": False})
    scheduler = ROOT / "benchmarks/h3/quantum_recrossing_campaign.py"
    assert sha(scheduler) == schedule["scheduler_source_sha256"]
    sources.append({**record(scheduler), "included_in_archive": False})
    execution = json.loads((ANALYSIS / "execution.json").read_text())
    for name, expected in execution["source_sha256"].items():
        assert sha(Path(name)) == expected
        if Path(name).suffix == ".py":
            sources.append({**record(Path(name)), "included_in_archive": False})
    model = SCRATCH / "literature-audit-assets/libbkmp2.so"
    assert sha(model) == protocol["model"]["library_sha256"]
    external = {"schema": "pyrpmd.h3.quantum-child-archive-prerequisites.v1",
                "source_identity_authority": "exact per-file SHA256; no assertion that all executed working-tree files belonged to one commit",
                "source_repository": str(ROOT), "sources_and_native_runtime": sources,
                "external_model": {**record(model), "build_receipt": record(model.with_suffix(".build.json")),
                                   "registered_model": protocol["model"], "included_in_archive": False},
                "parent_archive": record(PARENT_ARCHIVE), "external_parent_files": external_parents,
                "independent_review_input_inventory": record(review_path),
                "base_launch_evidence": "Only the completed schedule and partial/truncated final tool output survive; no exact launch timestamp or full stdout is claimed.",
                "base_launch_command_reconstructed_from_schedule_not_a_launch_receipt": [
                    str(ROOT / ".venv/bin/python"), "benchmarks/h3/quantum_recrossing_campaign.py",
                    "--output", str(DATA), "--library", str(model), "--horizon", "200", "--pair-batch", "0", "--workers", "2"]}
    write_new(OUT / "external-prerequisites.json", external)
    include(OUT / "external-prerequisites.json", "archive_prerequisite_metadata", "reference/external-prerequisites.json")
    include(OUT / "archive-scope.md", "archive_scope_and_replay_instructions", "ARCHIVE-SCOPE.md")

    rows = [members[name] for name in sorted(members)]
    assert len(raw) == 400 and len(runs) == 16
    inventory = {"schema": "pyrpmd.h3.quantum-child-archive-members.v1",
                 "scope": "Complete base pair_batch0 at horizon200 only; 8 seeds, two paired timesteps; all400 chunks.",
                 "files": rows, "member_count": len(rows), "uncompressed_bytes": sum(row["bytes"] for row in rows),
                 "category_counts": dict(Counter(row["category"] for row in rows))}
    write_new(SIDECAR, inventory)
    (OUT / "member-inventory.json").write_bytes(SIDECAR.read_bytes())
    with ARCHIVE.open("xb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", filename="", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for row in rows:
                    data = Path(row["source_path"]).read_bytes()
                    assert len(data) == row["bytes"] and digest(data) == row["sha256"]
                    info = tarfile.TarInfo(row["member"])
                    info.size, info.mode, info.mtime, info.uid, info.gid = len(data), 0o444, 0, 0, 0
                    archive.addfile(info, io.BytesIO(data))
    raw_by_name = {row["member"]: row for row in raw}
    settings_by_run = {row["directory"]: json.loads(Path(row["settings"]["path"]).read_text()) for row in runs}
    checked_names, reloaded = [], 0
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            assert member.isfile() and member.name in members and member.name not in checked_names
            data = archive.extractfile(member).read()
            row = members[member.name]
            assert len(data) == row["bytes"] and digest(data) == row["sha256"]
            checked_names.append(member.name)
            if member.name in raw_by_name:
                expected = raw_by_name[member.name]
                checked = check_chunk(data, settings_by_run[str(Path(member.name).parent)], expected["first_parent_index"])
                assert all(checked[key] == expected[key] for key in checked)
                reloaded += 1
    assert checked_names == [row["member"] for row in rows] and reloaded == 400
    for row in rows:
        assert sha(Path(row["source_path"])) == row["sha256"], "source mutated during archive creation"
    assert sha(Path(__file__)) == builder_sha
    assert sha(PARENT_ARCHIVE) == PARENT_ARCHIVE_SHA
    ARCHIVE.chmod(0o444)
    SIDECAR.chmod(0o444)
    write_new(OUT / "raw-data-index.json", {"schema": "pyrpmd.h3.quantum-child-raw-index.v1",
              "archive": record(ARCHIVE), "sidecar": record(SIDECAR), "runs": runs, "raw_files": raw,
              "raw_file_count": len(raw), "raw_bytes": sum(row["bytes"] for row in raw),
              "all_raw_files_reloaded_from_archive": True,
              "independent_parent_chains": 8, "parent_configurations": 1600,
              "children_per_timestep": 6400, "paired_timestep_runs_are_not_independent_replicas": True,
              "active_extra_pair_batch_included": False, "parent_raw_duplicated": False})
    usage = resource.getrusage(resource.RUSAGE_SELF)
    verification = {"schema": "pyrpmd.h3.quantum-child-archive-verification.v1",
                    "members_verified": len(rows), "original_files_rehashed_after_archive": len(rows),
                    "raw_checkpoints_original_and_archived_reloaded": 400,
                    "parent_checkpoint_integrity_envelopes_verified": 8,
                    "independent_review_input_hashes_verified": len(reviewed),
                    "all_16_runs_bound_to_original_complete_analysis": True,
                    "all_stored_signed_flux_numerators_and_denominators_exactly_replayed": True,
                    "paired_timestep_momenta_and_channels_identical": True,
                    "native_steps_preserved": sum(row["native_steps"] for row in raw),
                    "no_simulation_or_analysis_output_rewritten": True,
                    "all_original_registered_gate_outcomes_retained": report["criteria"],
                    "fixed_P_independent_rate_gates_passed": False,
                    "elapsed_seconds": time.monotonic() - started,
                    "CPU_seconds": usage.ru_utime + usage.ru_stime,
                    "peak_RSS_KiB": usage.ru_maxrss}
    write_new(OUT / "verification.json", verification)
    write_new(OUT / "archive-provenance.json", {
        "schema": "pyrpmd.h3.quantum-child-archive-provenance.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": record(ARCHIVE), "member_inventory": record(SIDECAR), "builder": record(Path(__file__)),
        "availability": "Local only; no upload attempted or external persistence asserted.",
        "immutability": "Exclusive creation, all member bytes verified, original input bytes rehashed, archive and sidecar chmod0444.",
        "format": "Sorted regular PAX members, mode0444, uid/gid/mtime0; gzip level1, mtime0.",
        "scope_counts": inventory["category_counts"], "member_count": len(rows),
        "external_prerequisites": record(OUT / "external-prerequisites.json"),
        "scientific_claim": "Preserves a complete first cohort whose registered kappa precision gate failed; archiving does not change that outcome.",
        "excluded": ["active extra-pair-batch-1", "its live logs and output", "water data", "umbrella data",
                     "parent raw already in equilibrium archive", "external PES source and binary", "compiled native binary", "Git-backed source code"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "git_HEAD_observed_not_source_identity_authority": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    print(json.dumps({"archive": record(ARCHIVE), "sidecar": record(SIDECAR), "verification": verification}, indent=2), flush=True)


if __name__ == "__main__":
    main()
