"""Bounded, once-only archive and extracted-data replay of the fixed dt control.

This is an offline evidence operation. It creates no Context and consumes no
random numbers. The frozen physical driver/analyzer is imported unchanged.
"""
from __future__ import annotations

import time
ENTRY_WALL = time.monotonic()

from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import subprocess
import sys
import tarfile

EVIDENCE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[6]
WATER = EVIDENCE.parents[1]
WORKSPACE = ROOT.parent
CONTROL = WORKSPACE / "water-paired-timestep-control-2026-11-01"
ORIGINAL = WORKSPACE / "water-convergence-2026-11-01"
EXECUTION = WORKSPACE / "water-paired-timestep-execution-2026-11-01"
WORK = WORKSPACE / "water-paired-timestep-archive-replay-2026-11-01"
ARCHIVE = WORKSPACE / "pyrpmd-water-paired-timestep-2026-11-01-raw.tar.gz"
REPORT_SHA = "6d657d7dc11d98dd78ac8a2659a9907c8b94eedba0aa013dcd2d3d5db54aa370"
REGISTRATION_SHA = "0d73545ac4751b84bdf64a99d693dee23a7f683bb3cd38e2cff5b7d8055e8365"
OBSERVATION_SHA = "ecf9be6d574c79159fc3a0af4c5b450e5f864c49244312ffdc287a43bcc44978"
GIT_CANDIDATES = ("36ec289981639bbeb8feecf3b30400b4cbda9afe", "3d096d5cfc5709f14a1ad5e3d184b108eef9f20e")
SEEDS = tuple(range(1101, 1109))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encoded(value))


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular source file required")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def main():
    # Reserve at most20 one-CPU-second read-only Git helpers from the300s cap.
    resource.setrlimit(resource.RLIMIT_CPU, (280, 280))
    os.sched_setaffinity(0, {5})
    os.nice(10)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True

    def expired(_signum, _frame):
        raise TimeoutError("archive/replay exceeded600 wall seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 600-(time.monotonic()-ENTRY_WALL)))
    WORK.mkdir(exist_ok=False)
    require(not ARCHIVE.exists(), "archive output must be new")
    write_new(WORK / "RUN_ONCE.json", {"source": ref(__file__), "CPU_affinity": [5],
                                      "main_CPU_hard_seconds": 280, "Git_helper_count_max": 20,
                                      "Git_helper_CPU_hard_seconds_each": 1,
                                      "combined_nominal_CPU_ceiling": 300, "wall_limit_seconds": 600,
                                      "PES_calls": 0, "RNG_draws": 0, "physical_steps": 0})
    require(sha(CONTROL / "registration.json") == REGISTRATION_SHA, "registration changed")
    require(sha(CONTROL / "analysis/report.json") == REPORT_SHA, "complete control report changed")
    require(sha(EXECUTION / "root-exec-observation.json") == OBSERVATION_SHA, "actual exit receipt changed")
    observation, completion = read(EXECUTION / "root-exec-observation.json"), read(EXECUTION / "completion.json")
    require(observation["exec_session_id"] == 91496 and type(observation["tool_result"]["exit_code"]) is int
            and observation["tool_result"]["exit_code"] == 0, "actual normal root wrapper exit required")
    require(completion["fixed_control_complete"] is True and completion["all_commands_reaped"] is True
            and completion["exit_code"] == 0 and completion["adopted_physical_steps"] == 384000
            and all(row["exit_code"] == 0 for row in completion["records"]), "all fixed control commands must complete normally")
    registration = read(CONTROL / "registration.json")
    require(len(registration["source_sha256"]) == 10, "fixed source inventory count differs")
    guards = {ROOT / name: digest for name, digest in registration["source_sha256"].items()}
    guards[Path(__file__).resolve()] = sha(__file__)
    for path, digest in guards.items():
        require(sha(path) == digest, "frozen source changed")

    # Read-only exclusion proof is taken after the observed whole-wrapper exit.
    locks = [ORIGINAL / "four-worker-control/coordinator.lock", ORIGINAL / "scheduler-odd.lock", ORIGINAL / "scheduler-even.lock"]
    locks += [ORIGINAL / "four-worker-control" / f"seed-{seed}.lock" for seed in SEEDS]
    locks += [CONTROL / "control.lock"]
    members, raw_payloads = [], {}
    with ExitStack() as leases:
        for path in locks:
            handle = leases.enter_context(path.open("rb"))
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease_started = datetime.now(timezone.utc).isoformat()
        sys.path.insert(0, str(ROOT))
        import numpy as np
        from benchmarks.water import paired_timestep as paired
        from scipy.special import stdtrit
        for name, digest in registration["source_sha256"].items():
            module = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
            imported = Path(sys.modules[module].__file__).resolve()
            require(imported == ROOT/name and sha(imported) == digest, "imported source differs")
        loaded = paired.load_registration(CONTROL / "registration.json")
        require(loaded["sha256"] == REGISTRATION_SHA, "full frozen registration check failed")
        report = read(CONTROL / "analysis/report.json")
        require(report["all_eight_pairs_complete"] and report["energy_limits_pass"]
                and not report["overall_numerical_control_pass"] and report["literature_comparison"] is None,
                "complete report flags differ")
        initial_checks = []
        for seed in SEEDS:
            directory = CONTROL / "runs" / f"seed-{seed}"
            with np.load(directory / "initial/initial.npz", allow_pickle=False) as data:
                q, p = data["q"].copy(), data["p"].copy()
            for label, expected_steps in (("dt025", 16000), ("dt0125", 32000)):
                branch = directory / label
                manifest = read(branch / "manifest.json")
                require(manifest["complete"] and manifest["committed_steps"] == expected_steps
                        and len(manifest["chunks"]) == 5, "fixed complete branch membership differs")
                paired.native.verify_manifest(branch, manifest)
                with np.load(branch / "chunk-000000.npz", allow_pickle=False) as data:
                    require(int(data["state_step"]) == 0 and int(data["active_segment"]) == 0,
                            "initial branch checkpoint phase differs")
                    dq, dp = float(np.max(abs(data["q"]-q))), float(np.max(abs(data["p"]-p)))
                    require(np.isfinite(dq) and np.isfinite(dp) and dq <= 2e-12 and dp <= 2e-13,
                            "stored branch canonical initial state differs from shared draw")
                initial_checks.append({"seed": seed, "branch": label, "q_max_difference_bohr": dq,
                                       "p_max_difference_au": dp, "committed_steps": expected_steps})
        stats = {}
        require([row["seed"] for row in report["chains"]] == list(SEEDS), "paired cohort identity differs")
        for key, old in report["paired_comparisons"].items():
            delta = [row["branches"]["dt0125"][key]-row["branches"]["dt025"][key] for row in report["chains"]]
            mean = math.fsum(delta)/8
            se = math.sqrt(math.fsum((x-mean)**2 for x in delta)/56)
            half = float(stdtrit(7, .975))*se
            values = {"chain_differences": delta, "mean": mean, "standard_error": se,
                      "CI95": [mean-half, mean+half], "degrees_of_freedom": 7,
                      "equivalent_within_margin": mean-half >= -.01 and mean+half <= .01}
            require(delta == old["chain_differences"] and abs(mean-old["mean"]) <= 2e-15
                    and abs(se-old["standard_error"]) <= 2e-15
                    and max(abs(a-b) for a,b in zip(values["CI95"], old["CI95"])) <= 2e-15
                    and values["equivalent_within_margin"] == old["equivalent_within_margin"] is False,
                    "independent whole-chain paired-statistics reconstruction differs")
            stats[key] = values
        write_new(WORK / "independent-checks.json", {"all_checks_passed": True, "initial_states": initial_checks,
                                                    "paired_statistics": stats, "physical_steps": 0})

        def add(path, name):
            require(name not in raw_payloads, "duplicate archive member")
            record = ref(path)
            members.append({"member": name, "source": record})
            raw_payloads[name] = Path(path)

        control_files = sorted(path for path in CONTROL.rglob("*") if path.is_file() and path != CONTROL / "control.lock")
        require(len(control_files) == 147, "unexpected control file set")
        for path in control_files:
            add(path, "control/"+str(path.relative_to(CONTROL)))
        for seed in SEEDS:
            row = registration["selection"][str(seed)]
            for name in ("manifest.json", row["checkpoint"]["file"], *sorted(row["artifacts"])):
                add(Path(row["directory"]) / name, f"original-selection/runs/classical-{seed}/"+name)
        for name in ("four-worker-registration.json", "four-worker-control/events.jsonl"):
            add(ORIGINAL / name, "original-selection/"+name)
        execution_names = {"RUN_ONCE.json", "complete-analysis.json", "complete-analysis.stderr.log", "complete-analysis.stdout.log",
                           "completion.json", "registration.json", "registration.stderr.log", "registration.stdout.log",
                           "root-exec-observation.json", "run-1.json", "run-1.stderr.log", "run-1.stdout.log"}
        require({p.name for p in EXECUTION.iterdir()} == execution_names, "root execution file set changed")
        for name in sorted(execution_names):
            add(EXECUTION / name, "execution/"+name)
        references = {"paired-protocol.json": Path(registration["protocol_path"]),
                      "original-report.json": Path(registration["assessment"]["path"]),
                      "original-exit.json": Path(registration["original_coordinator_exit_proof"]["path"]),
                      "root-adoption.json": WATER / "timestep-paired-control/original-cohort-adoption/root-adoption.json"}
        for name, path in references.items():
            add(path, "references/"+name)
        source_records = []
        for name, digest in registration["source_sha256"].items():
            matching = []
            for commit in GIT_CANDIDATES:
                blob = subprocess.run(["git", "show", commit+":"+name], cwd=ROOT, capture_output=True, timeout=10,
                                      preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CPU, (1, 1)))
                if blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == digest:
                    matching.append(commit)
            require(matching, "no exact Git source reconstruction reference")
            source_records.append({"path": name, "sha256": digest, "git_commit": matching[0]})
        generated = {"references/source-identities.json": encoded({"frozen_sources": source_records,
                     "environment": registration["environment"], "archive_replay_helper": ref(__file__),
                     "source_code_included": False, "third_party_code_included": False}),
                     "README.md": ("# Complete fixed eight-chain paired timestep data\n\n"
                     "All16 branches,64ps and384000 integration steps completed. Energy limits pass; both fixed +/-0.01 A^2/ps paired diffusion equivalence tests remain unresolved. No paper agreement is claimed.\n\n"
                     "control/ preserves every original control data/report byte; original-selection/ holds all8 selected100ps checkpoints, source manifests/models and original exit journal. execution/ and references/ preserve approval/normal-exit and fixed protocol metadata. SHA256SUMS covers every other member. No Python/native/PES source is included.\n\n"
                     "Reconstruct the frozen Git sources using references/source-identities.json, including each per-file commit and SHA. The offline helper identified there derives a new path-only registration for extracted data; it does not modify any archived source record. Only that new registration SHA may differ in the replay report; every scientific field and compact array must match. Original absolute paths are provenance, not authorized fallback reads.\n\n"
                     "The selected original directory is intentionally a provenance subset, not a resumable full old campaign: it contains the exact100ps checkpoints needed by the control analyzer, not all200 original chunks. The full original cohort is separately archived under SHA b3dc0faf5ddb84d06182ebf788edc73dfa1a0ec674069f01e61eaa62918d21af.\n").encode()}
        checksums = {row["member"]: row["source"]["sha256"] for row in members}
        checksums.update({name: hashlib.sha256(raw).hexdigest() for name, raw in generated.items()})
        generated["SHA256SUMS"] = "".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())).encode()
        write_new(WORK / "archive-members.json", {"original_files": members, "generated_members": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in generated.items()}})
        partial = ARCHIVE.with_suffix(ARCHIVE.suffix+".partial")
        with partial.open("xb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0, compresslevel=1) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    for name in sorted(raw_payloads.keys() | generated.keys()):
                        require(Path(name).suffix not in {".py", ".patch", ".so", ".f90"}, "source in raw-only archive")
                        info = tarfile.TarInfo(name); info.mode = 0o444; info.mtime = 0
                        if name in generated:
                            raw = generated[name]; info.size = len(raw); tar.addfile(info, io.BytesIO(raw))
                        else:
                            path = raw_payloads[name]; info.size = path.stat().st_size
                            with path.open("rb") as stream:
                                tar.addfile(info, stream)
            destination.flush(); os.fsync(destination.fileno())
        for row in members:
            require(ref(row["source"]["path"]) == row["source"], "source data changed during archive")
        lease_finished = datetime.now(timezone.utc).isoformat()
    lease_proof = {"schema": "pyrpmd.water-paired-control-post-exit-lease-proof.v1", "root_exit_observation_sha256": OBSERVATION_SHA,
                   "all12_leases_acquired_nonblocking": True, "lease_paths": list(map(str, locks)),
                   "acquired_utc": lease_started, "snapshot_finished_utc": lease_finished,
                   "all_handles_closed_and_leases_released": True, "data_file_mutations": False}
    write_new(WORK / "lease-proof.json", lease_proof)

    extracted = WORK / "extracted"; extracted.mkdir()
    expected = dict(checksums); expected["SHA256SUMS"] = hashlib.sha256(generated["SHA256SUMS"]).hexdigest()
    seen = set()
    with tarfile.open(partial, "r|gz") as container:
        for entry in container:
            relative = PurePosixPath(entry.name)
            require(entry.isfile() and not entry.issparse() and not relative.is_absolute() and ".." not in relative.parts
                    and relative.as_posix() == entry.name and entry.name in expected and entry.name not in seen,
                    "unsafe or duplicate archive member")
            path = extracted / entry.name; path.parent.mkdir(parents=True, exist_ok=True)
            with container.extractfile(entry) as stream, path.open("xb") as target:
                for block in iter(lambda: stream.read(1048576), b""):
                    target.write(block)
            require(sha(path) == expected[entry.name], "extracted member checksum differs")
            seen.add(entry.name)
    require(seen == set(expected), "extracted member coverage differs")
    archived_registration = read(extracted / "control/registration.json")
    derived = json.loads(json.dumps(archived_registration))
    derived["output_root"] = str(extracted / "control")
    derived["original_root"] = str(extracted / "original-selection")
    derived["protocol_path"] = str(extracted / "references/paired-protocol.json")
    derived["assessment"]["path"] = str(extracted / "references/original-report.json")
    derived["original_coordinator_exit_proof"]["path"] = str(extracted / "references/original-exit.json")
    for seed in SEEDS:
        derived["selection"][str(seed)]["directory"] = str(extracted / "original-selection/runs" / f"classical-{seed}")
    derived_path = extracted / "control/registration.relocated.json"
    write_new(derived_path, derived)
    writes = {"registration_paths_only_changed": True, "original_sha256": REGISTRATION_SHA,
              "derived_sha256": sha(derived_path), "derived_path": str(derived_path),
              "original_archive_bytes_preserved": True}
    write_new(WORK / "relocation.json", writes)

    # Every later trajectory read must resolve to the extracted payload.
    phase = {"value": "replay"}; numerical_reads, forbidden_attempts = set(), []
    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path == denied or denied in path.parents for denied in (CONTROL, ORIGINAL, EXECUTION)):
            forbidden_attempts.append(str(path))
            raise RuntimeError("replay attempted original data fallback")
        mode, flags = args[1], args[2]
        reading = (isinstance(mode, str) and "r" in mode) or (isinstance(flags, int) and flags & os.O_ACCMODE == os.O_RDONLY)
        if phase["value"] == "replay" and reading and path.suffix == ".npz":
            if path == WORK / "analysis/curves.npz":
                return
            require(extracted in path.parents, "numerical input escaped extracted data")
            numerical_reads.add(str(path.relative_to(extracted)))
    sys.addaudithook(audit)
    def no_engine(*_args, **_kwargs):
        raise RuntimeError("archive replay must not create an engine")
    factory = paired.native.create_engine
    paired.native.create_engine = no_engine
    try:
        replay = paired.analyze(derived_path, WORK / "analysis")
    finally:
        paired.native.create_engine = factory
        phase["value"] = "comparison"
    reference = read(extracted / "control/analysis/report.json")
    normalized = dict(replay); normalized["registration_sha256"] = reference["registration_sha256"]
    require(normalized == reference, "physical/statistical report differs after path-only relocation")
    require(sha(WORK / "analysis/curves.npz") == reference["curves_sha256"], "compact curve container differs")
    with np.load(WORK / "analysis/curves.npz", allow_pickle=False) as current, np.load(extracted / "control/analysis/curves.npz", allow_pickle=False) as previous:
        require(set(current.files) == set(previous.files), "curve names differ")
        for name in current.files:
            require(current[name].dtype == previous[name].dtype and current[name].shape == previous[name].shape
                    and current[name].tobytes() == previous[name].tobytes(), "compact curve array differs")
        curve_count = len(current.files)
    expected_reads = {f"control/runs/seed-{seed}/{branch}/chunk-{n:06d}.npz" for seed in SEEDS
                      for branch in ("dt025", "dt0125") for n in range(5)}
    expected_reads |= {f"control/runs/seed-{seed}/initial/initial.npz" for seed in SEEDS}
    expected_reads |= {f"original-selection/runs/classical-{seed}/chunk-000099.npz" for seed in SEEDS}
    require(numerical_reads == expected_reads and not forbidden_attempts, "actual extracted numerical-read coverage differs")
    for path, digest in guards.items():
        require(sha(path) == digest, "frozen source changed during evidence operation")
    os.link(partial, ARCHIVE)
    partial.unlink()
    read_audit = {"original_data_access_attempts": forbidden_attempts, "extracted_numerical_reads": sorted(numerical_reads),
                  "numerical_input_files": len(numerical_reads), "no_original_fallback": True}
    write_new(WORK / "read-audit.json", read_audit)
    result = {"schema": "pyrpmd.water-complete-paired-control-extraction-replay.v1", "all_checks_passed": True,
              "archive": ref(ARCHIVE), "archive_members": len(seen), "original_data_files": len(members),
              "complete_branches": 16, "original_integration_steps": 384000, "original_NVE_ps": 64,
              "new_PES_calls": 0, "new_RNG_draws": 0, "new_dynamics_steps": 0,
              "source_guards_unchanged": True, "original_data_unchanged": True,
              "all_scientific_and_failed_gate_fields_equal": True, "only_report_metadata_difference": "registration_sha256 for explicit path-only relocation",
              "original_report_sha256": REPORT_SHA, "replayed_report_sha256": sha(WORK / "analysis/report.json"),
              "compact_curve_container_sha256": sha(WORK / "analysis/curves.npz"), "bitwise_equal_compact_arrays": curve_count,
              "extracted_numerical_input_files": len(numerical_reads), "original_data_access_attempts": 0,
              "lease_proof": ref(WORK / "lease-proof.json"), "independent_checks": ref(WORK / "independent-checks.json"),
              "relocation": ref(WORK / "relocation.json"), "read_audit": ref(WORK / "read-audit.json"),
              "CPU_affinity": [5], "threads": 1, "nice_increment": 10,
              "CPU_seconds_before_write": time.process_time(), "wall_seconds_before_write": time.monotonic()-ENTRY_WALL,
              "Git_helpers_CPU_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
              "maximum_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "physical_converged": False, "literature_reproduced": False,
              "scope": "Exact data curation/replay and independently recomputed paired statistics only; both equivalence failures remain. No new physical execution."}
    write_new(WORK / "replay-report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
