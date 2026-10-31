"""Package the completed, fixed H3 preparation study without running dynamics.

Only explicitly named data/report files enter the ZIP.  Input ledgers, final
analysis hashes, current-run source snapshots and the supplied Git revision
are verified before and after writing. Historical baseline source identities
are preserved as historical provenance, not compared with newer live code.

The ZIP has deterministic member order, timestamps and permissions. Its
internal SHA256SUMS covers all other members; an external .members.sha256
also covers SHA256SUMS itself, and .sha256 covers the complete ZIP.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
POINTS = ((0, .55), (1, .90), (1, 1.10))
VARIANTS = ("ne-direct-none", "ne-switch200-none",
            "qct-action-fixed_collision", "qct-action-fixed_total")
AUDITS = ("direct_paper", "direct_none", "switch50_none", "switch200_none")
CASE_FILES = ("config.json", "preparation.json", "provenance.json",
              "summary.json", "trajectories.jsonl")
BASE_FILES = ("config.json", "summary.json", "trajectories.jsonl")
LITERATURE_FILES = ("marjollet_2021_rpmd_fig5b_fig6b_digitized.csv",
    "marjollet_2021_requested_energy_interpolation.csv",
    "marjollet_2021_digitization_provenance.json")
ANALYSIS_FILES = ("preparation_analysis.json", "preparation_cross_sections.csv",
    "preparation_paired_comparisons.csv", "audit_statistics.csv",
    "audit_cases.csv", "audit_paired_outcomes.csv", "audit_paired_time_means.csv", "preparation_numerics.csv",
    "internal_energy_references.csv", "unresolved_endpoint_diagnostics.json",
    "preparation_report.html", "figures/preparation_scattering.png",
    "figures/preparation_scattering.svg", "figures/preparation_energy_hold.png",
    "figures/preparation_energy_hold.svg")
OPTIONAL_ANALYSIS = ("pyrpmd_state_preparation_report.pdf",)
BASELINE_CODE = "0860a079950e09e1e97c001acd23a436ce273ec1"
REPOSITORY_URL = "https://github.com/chen-yu-hao/PyRPMD"
CHUNK = 1024 * 1024


class ArchiveError(ValueError):
    """Incomplete, changed, or inconsistent inputs; no final ZIP is published."""


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def _decode_json(text, path):
    def invalid(value):
        raise ArchiveError(f"Nonfinite JSON value in {path}: {value}")
    def number(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)
    def mapping(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ArchiveError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    try:
        return json.loads(text, parse_constant=invalid, parse_float=number, object_pairs_hook=mapping)
    except json.JSONDecodeError as exc:
        raise ArchiveError(f"Cannot read required JSON {path}: {exc}") from exc


def _read_json(path):
    try:
        return _decode_json(Path(path).read_text(), path)
    except OSError as exc:
        raise ArchiveError(f"Cannot read required JSON {path}: {exc}") from exc


def _regular_file(path, root):
    path, root = Path(path), Path(root).resolve()
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ArchiveError(f"Missing, nonregular, symlinked, or out-of-root file: {path}")
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise ArchiveError(f"Symlinked archive input directory: {parent}")
    return path.resolve()


def _hash_string(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ArchiveError(f"Invalid recorded SHA256: {value!r}")
    return value


def _number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArchiveError(f"Missing/nonfinite numeric audit diagnostic: {context}")
    return value


def _accumulate_audit(row, preparation, windows, differences, path):
    """Keep only the small time-mean arrays while streaming the full ledger."""
    study = row.get("preparation_study", {})
    native = study.get("switch", {})
    averages = study.get("time_averages", {})
    columns = {"primitive_internal_per_bead_hartree", "kinetic_per_bead_hartree",
               "full_potential_per_bead_hartree", "spring_per_bead_hartree",
               "centroid_harmonic_energy_hartree", "bond_spread_bohr"}
    sources = {"hold": "native_every_step_trapezoid", "hold_trace_check": "dense_trace_trapezoid",
               "final_50fs": "dense_trace_trapezoid"}
    if (native.get("status") != "completed" or set(averages) != set(sources) or
            any(study.get(key) != preparation[key] for key in ("kind", "conditioning", "energy_convention"))):
        raise ArchiveError(f"Invalid successful audit diagnostics: {path}, ID {row['id']}")
    start, duration = preparation["switch_time_fs"], preparation["hold_time_fs"]
    for key, expected in (("switch_time_fs", start), ("hold_time_fs", duration),
                           ("accepted_hold_duration_fs", duration), ("dt_fs", preparation["preparation_dt_fs"])):
        if not math.isclose(_number(native.get(key), key), expected, rel_tol=1e-12, abs_tol=1e-9):
            raise ArchiveError(f"Native audit timing mismatch: {path}, ID {row['id']}")
    for name, source in sources.items():
        window = averages[name]
        expected = {"start_fs": start+duration-50. if name == "final_50fs" else start,
                    "end_fs": start+duration, "duration_fs": 50. if name == "final_50fs" else duration}
        values = window.get("values", {})
        if (window.get("source") != source or not columns <= values.keys() or
                any(not math.isclose(_number(window.get(key), key), value, rel_tol=1e-12, abs_tol=1e-9)
                    for key, value in expected.items())):
            raise ArchiveError(f"Audit averaging window/columns mismatch: {path}, ID {row['id']}")
        if name == "hold" and (set(values) != columns or values != native.get("hold_time_average")):
            raise ArchiveError(f"Audit hold values disagree with native integral: {path}, ID {row['id']}")
        if name == "hold" and (window.get("native_dt_fs") != native["dt_fs"] or
                type(window.get("native_steps")) is not int or window["native_steps"] != round(duration/native["dt_fs"])):
            raise ArchiveError(f"Invalid native hold timestep/count: {path}, ID {row['id']}")
        if name not in windows:
            windows[name] = {"source": source, **expected, "values": {key: [] for key in values}}
        target = windows[name]["values"]
        if set(target) != set(values):
            raise ArchiveError(f"Audit successful mean columns changed: {path}, ID {row['id']}")
        for key, value in values.items():
            target[key].append(_number(value, f"{name}.{key}"))
    for key in columns:
        differences.setdefault(key, []).append(averages["hold"]["values"][key]-averages["hold_trace_check"]["values"][key])


def _check_reduced_values(recorded, observations, path):
    if not isinstance(recorded, dict) or set(recorded) != set(observations):
        raise ArchiveError(f"Audit reduction columns differ from ledger: {path}")
    for key, values in observations.items():
        result = recorded[key]
        n = len(values)
        mean = math.fsum(values)/n
        sem = math.sqrt(math.fsum((value-mean)**2 for value in values)/(n-1)/n) if n > 1 else None
        if type(result.get("n")) is not int or result["n"] != n:
            raise ArchiveError(f"Audit reduction count differs from ledger: {path}, {key}")
        for name, expected in (("mean", mean), ("sem", sem)):
            value = result.get(name)
            if ((expected is None and value is not None) or
                    (expected is not None and not math.isclose(_number(value, f"{key}.{name}"), expected,
                                                              rel_tol=1e-10, abs_tol=1e-12))):
                raise ArchiveError(f"Audit reduction {name} differs from ledger: {path}, {key}")


@dataclass(frozen=True)
class Member:
    name: str
    path: Path
    size: int
    sha256: str


class Plan:
    def __init__(self, baseline, study, code_commit):
        self.baseline, self.study = Path(baseline).resolve(), Path(study).resolve()
        if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
            raise ArchiveError("--code-commit must be an exact 40-character Git SHA")
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify",
            code_commit+"^{commit}"], text=True, capture_output=True)
        if result.returncode or result.stdout.strip() != code_commit:
            raise ArchiveError("--code-commit is not an available exact commit in this checkout")
        self.code_commit = code_commit
        self.members, self.watched, self.code_hashes = {}, {}, {}
        self.analysis_inputs = set()

    def watch(self, path, expected=None, *, code=False):
        path = Path(path).resolve()
        actual = self.watched.get(path)
        if actual is None:
            if not path.is_file():
                raise ArchiveError(f"Recorded source file is absent: {path}")
            actual = _digest(path)
            self.watched[path] = actual
        if expected is not None and actual != _hash_string(expected):
            raise ArchiveError(f"Recorded source hash changed: {path}; expected {expected}, found {actual}")
        if code:
            relative = path.relative_to(ROOT).as_posix()
            if relative not in self.code_hashes:
                result = subprocess.run(["git", "-C", str(ROOT), "show",
                    f"{self.code_commit}:{relative}"], capture_output=True)
                if result.returncode or hashlib.sha256(result.stdout).hexdigest() != actual:
                    raise ArchiveError(f"Live source does not match --code-commit: {relative}")
                self.code_hashes[relative] = actual
        return actual

    def add(self, path, root, prefix, *, analysis_input=False):
        path = _regular_file(path, root)
        relative = path.relative_to(Path(root).resolve()).as_posix()
        name = str(PurePosixPath("data") / prefix / relative)
        if name in self.members:
            raise ArchiveError(f"Duplicate archive member: {name}")
        self.members[name] = Member(name, path, path.stat().st_size, self.watch(path))
        if analysis_input:
            self.analysis_inputs.add(path)
        return path

    def recorded_path(self, value):
        """Map historical absolute paths to these data roots or this code checkout."""
        if not isinstance(value, str):
            raise ArchiveError("A recorded source path must be text")
        original = Path(value)
        for root in (self.baseline, self.study):
            if original.is_absolute() and original.is_relative_to(root):
                return _regular_file(original, root)
        parts = PurePosixPath(value).parts
        for marker, root in (("state_selected_reproduction", self.baseline),
                             ("state_preparation_study", self.study)):
            if marker in parts:
                suffix = parts[parts.index(marker)+1:]
                return _regular_file(root.joinpath(*suffix), root)
        if original.is_absolute() and original.is_relative_to(ROOT):
            return _regular_file(original, ROOT)
        for marker in ("benchmarks", "src", "native"):
            if marker in parts:
                return _regular_file(ROOT.joinpath(*parts[parts.index(marker):]), ROOT)
        raise ArchiveError(f"Source path is outside the allowed data/code roots: {value}")

    def verify_record(self, entry, *, source_snapshot=False):
        if not isinstance(entry, dict) or not {"path", "sha256"} <= entry.keys():
            raise ArchiveError("Invalid source-hash record")
        path = self.recorded_path(entry["path"])
        in_code = path.is_relative_to(ROOT) and not (
            path.is_relative_to(self.baseline) or path.is_relative_to(self.study))
        relative = path.relative_to(ROOT) if in_code else None
        tracked_source = in_code and "build" not in relative.parts and path.suffix not in (".so", ".pyd", ".dll", ".dylib")
        self.watch(path, entry["sha256"], code=tracked_source)
        if not source_snapshot and not in_code and path not in self.analysis_inputs:
            raise ArchiveError(f"Analysis depends on data outside the archive whitelist: {path}")
        return path

    def verify_all(self):
        for path, expected in self.watched.items():
            if not path.is_file() or _digest(path) != expected:
                raise ArchiveError(f"Input/source changed while preparing the archive: {path}")


def _validate_case(plan, path, *, v, energy, method, beads, samples, offset, audit=False,
                   baseline=False):
    root = plan.baseline if baseline else plan.study
    prefix = "state_selected_reproduction" if baseline else "state_preparation_study"
    files = BASE_FILES if baseline else CASE_FILES+("audit_time_averages.json",) if audit else CASE_FILES
    for name in files:
        plan.add(path/name, root, prefix, analysis_input=True)
    config, summary = _read_json(path/"config.json"), _read_json(path/"summary.json")
    preparation = _read_json(path/"preparation.json") if not baseline else None
    expected = dict(v=v, total_energy_ev=energy, method=method, beads=beads,
                    samples=samples, id_offset=offset, seed=20261031, bmin=0.,
                    bmax=3.5 if v == 0 else 4., dt_fs=.02,
                    time_cap_fs=250., extended_cap_fs=1000.)
    if (any(config.get(key) != value for key, value in expected.items()) or summary.get("config") != config or
            any(type(config.get(key)) is not int for key in ("v", "beads", "samples", "id_offset", "seed"))):
        raise ArchiveError(f"Case configuration does not match the fixed study: {path}")
    if not baseline and summary.get("preparation") != preparation:
        raise ArchiveError(f"Summary/preparation mismatch: {path}")
    seen, counts = set(), Counter()
    windows, differences = {}, {}
    statuses = {"prepared", "failed"} if audit else {"product", "nonreactive", "unresolved", "failed"}
    with (path/"trajectories.jsonl").open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = _decode_json(line, f"{path}/trajectories.jsonl, line {number}")
            if not isinstance(row, dict):
                raise ArchiveError(f"Ledger row is not an object: {path}, line {number}")
            identity, status = row.get("id"), row.get("status")
            if type(identity) is not int or identity in seen or status not in statuses:
                raise ArchiveError(f"Invalid or duplicate ID/status in {path}, line {number}")
            seen.add(identity); counts[status] += 1
            if audit and status == "prepared":
                _accumulate_audit(row, preparation, windows, differences, path)
    if seen != set(range(offset, offset+samples)):
        raise ArchiveError(f"Ledger does not contain the exact declared ID range: {path}")
    if audit:
        totals = {"attempts": samples, "prepared": counts["prepared"], "failed": counts["failed"]}
        means = _read_json(path/"audit_time_averages.json")
        if any(type(summary.get(key)) is not int or type(means.get(key)) is not int or
               summary[key] != value or means[key] != value for key, value in totals.items()):
            raise ArchiveError(f"Audit summary/reduction count mismatch: {path}")
        if not isinstance(means.get("windows"), dict) or set(means["windows"]) != set(windows):
            raise ArchiveError(f"Audit reduction windows differ from ledger: {path}")
        for name, expected_window in windows.items():
            window = means["windows"][name]
            if (window.get("source") != expected_window["source"] or any(
                    not math.isclose(_number(window.get(key), key), expected_window[key], rel_tol=1e-12, abs_tol=1e-9)
                    for key in ("start_fs", "end_fs", "duration_fs"))):
                raise ArchiveError(f"Audit reduction window metadata differs from ledger: {path}")
            _check_reduced_values(window.get("values"), expected_window["values"], path)
        _check_reduced_values(means.get("native_minus_trace_hold"), differences, path)
    elif (type(summary.get("launched_attempts")) is not int or summary["launched_attempts"] != samples or
          summary.get("counts") != {s: counts[s] for s in statuses} or
          any(type(value) is not int for value in summary.get("counts", {}).values())):
        raise ArchiveError(f"Collision summary/ledger count mismatch: {path}")
    if baseline:
        # The old run's Git HEAD can precede its eventual exact-source archive.
        # Its embedded script/PES identities must survive without rewriting.
        provenance = summary.get("provenance", {})
        _hash_string(provenance.get("script_sha256"))
        if not provenance.get("git_commit") or not provenance.get("pes"):
            raise ArchiveError(f"Missing historical baseline provenance: {path}")
        for optional in ("preparation.json", "provenance.json"):
            if (path/optional).exists():
                plan.add(path/optional, root, prefix, analysis_input=True)
    else:
        provenance = _read_json(path/"provenance.json")
        if provenance.get("study_spec") != preparation:
            raise ArchiveError(f"Preparation/provenance mismatch: {path}")
        records = provenance.get("files")
        if not isinstance(records, list) or not records:
            raise ArchiveError(f"Missing current-run source snapshots: {path}")
        for entry in records:
            plan.verify_record(entry, source_snapshot=True)
    return config, preparation


def prepare_plan(baseline, study, code_commit):
    plan = Plan(baseline, study, code_commit)
    analysis = _read_json(plan.study/"analysis/preparation_analysis.json")
    manifest = _read_json(plan.study/"isolated_audit/audit_manifest.json")
    if analysis.get("complete") is not True or analysis.get("pending_cases") != []:
        raise ArchiveError("Final analysis is incomplete; regenerate it after all cases finish")
    if manifest.get("status") != "completed" or manifest.get("wrapper_unchanged_after_run") is not True:
        raise ArchiveError("The isolated audit manifest is not successfully completed")
    if analysis.get("audit_manifest_status") != "completed":
        raise ArchiveError("Analysis predates the completed isolated audit manifest")
    if len(analysis.get("cross_sections", [])) != 21 or len(analysis.get("paired_comparisons", [])) != 12:
        raise ArchiveError("Final analysis lacks the planned 21 estimates or 12 paired contrasts")
    if len(analysis.get("audit_cases", [])) != 12 or len(analysis.get("audit_paired_outcomes", [])) != 9:
        raise ArchiveError("Final analysis lacks complete audit attempt/pair accounting; regenerate it")
    plan.watch(Path(__file__), code=True)
    expected_audits = {}
    for v, energy in POINTS:
        for method, beads, offset in (("ne_rpmd", 32, 10000), ("qct", 1, 20000)):
            _validate_case(plan, plan.baseline/"main"/f"{method}-v{v}-Et{energy:.3f}-P{beads}",
                v=v, energy=energy, method=method, beads=beads, samples=4096, offset=offset, baseline=True)
        for variant in VARIANTS:
            is_ne = variant.startswith("ne-")
            _, preparation = _validate_case(plan, plan.study/"collisions"/f"{variant}-v{v}-Et{energy:.3f}",
                v=v, energy=energy, method="ne_rpmd" if is_ne else "qct",
                beads=32 if is_ne else 1, samples=1024 if is_ne else 4096,
                offset=10000 if is_ne else 20000)
            expected_preparation = {
                "kind": "switch" if variant == "ne-switch200-none" else "direct" if is_ne else "action_qct",
                "conditioning": "none", "switch_time_fs": 200. if variant == "ne-switch200-none" else 0.,
                "hold_time_fs": 0., "preparation_dt_fs": .02, "trace_stride_fs": .2,
                "energy_convention": "fixed_total" if variant.endswith("fixed_total") else "fixed_collision"}
            if preparation != expected_preparation:
                raise ArchiveError(f"Unexpected preparation settings for {variant}, v={v}, E={energy}")
        for variant in AUDITS:
            name = f"v{v}_E{energy:.2f}_{variant}".replace(".", "p")
            expected_audits[name] = _validate_case(plan, plan.study/"isolated_audit"/name,
                v=v, energy=energy, method="ne_rpmd", beads=32, samples=512, offset=910000, audit=True)
            expected_preparation = {"kind": "switch" if variant.startswith("switch") else "direct",
                "conditioning": "paper" if variant == "direct_paper" else "none",
                "switch_time_fs": 50. if variant == "switch50_none" else 200. if variant == "switch200_none" else 0.,
                "hold_time_fs": 100., "preparation_dt_fs": .02,
                "trace_stride_fs": .2, "energy_convention": "fixed_collision"}
            if expected_audits[name][1] != expected_preparation:
                raise ArchiveError(f"Unexpected isolated-audit preparation: {name}")
    declared = manifest.get("cases", [])
    completed = manifest.get("completed_cases", [])
    if (len(declared) != 12 or len(completed) != 12 or set(completed) != set(expected_audits)
            or {r.get("name") for r in declared} != set(expected_audits)):
        raise ArchiveError("Audit manifest does not describe exactly the twelve planned cases")
    for case in declared:
        if (case.get("config"), case.get("preparation")) != expected_audits[case["name"]]:
            raise ArchiveError(f"Audit manifest/config mismatch: {case['name']}")
    plan.verify_record(manifest.get("wrapper_source"), source_snapshot=True)
    plan.add(plan.study/"isolated_audit/audit_manifest.json", plan.study,
             "state_preparation_study", analysis_input=True)
    plan.add(plan.baseline/"h2_reference.json", plan.baseline,
             "state_selected_reproduction", analysis_input=True)
    for filename in LITERATURE_FILES:
        plan.add(plan.baseline/"literature_reference"/filename, plan.baseline,
                 "state_selected_reproduction", analysis_input="interpolation" in filename)
    for filename in ANALYSIS_FILES:
        plan.add(plan.study/"analysis"/filename, plan.study, "state_preparation_study")
    for filename in OPTIONAL_ANALYSIS:
        if (plan.study/"analysis"/filename).exists():
            plan.add(plan.study/"analysis"/filename, plan.study, "state_preparation_study")
    if (plan.study/"planned_workload.json").exists():
        plan.add(plan.study/"planned_workload.json", plan.study, "state_preparation_study")
    sources = analysis.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ArchiveError("Final analysis lacks input hashes")
    verified = {plan.verify_record(entry) for entry in sources}
    if not plan.analysis_inputs <= verified:
        missing = sorted(str(p) for p in plan.analysis_inputs-verified)
        raise ArchiveError(f"Final analysis does not authenticate all archived inputs: {missing}")
    plan.verify_all()
    return plan


def _readme(plan):
    return f"""# BKMP2 state-preparation study: raw data and analysis

This archive contains exactly 12 new collision cases (30,720 attempts),
12 compact isolated-diatom audits (6,144 attempts), and the six original
4096-attempt baseline cases needed for the paired analysis. Original statuses,
unknown outcomes, configuration/provenance JSON and ledgers are unchanged.
The original NE paired control uses only IDs 10000–11023 of its archived full
4096-row case; the full sample is not a second independent paired control.

## Exact code

Follow-up implementation and analysis: {REPOSITORY_URL}/tree/{plan.code_commit}

Protocol: {REPOSITORY_URL}/blob/{plan.code_commit}/benchmarks/h3/STATE_PREPARATION_STUDY.md

Historical formal-baseline source archive: {REPOSITORY_URL}/tree/{BASELINE_CODE}
The baseline summary's original Git HEAD and script SHA are preserved; the
eventual exact-source archive can be later than the HEAD recorded at launch.
Historical baseline code is not asserted to equal the newer follow-up source.
Current-run source snapshots and the analysis input hashes were checked when
this ZIP was built. See ARCHIVE_MANIFEST.json for verified current code hashes.

## Verify and reanalyze

After extraction, run `sha256sum -c SHA256SUMS` from this directory. This checks
every other ZIP member, including this README and ARCHIVE_MANIFEST.json.
The external `<archive>.zip.members.sha256` also hashes SHA256SUMS itself;
the external `<archive>.zip.sha256` verifies the complete ZIP. A checksum file
cannot include its own ordinary SHA256 without a circular definition.

Check out the exact follow-up commit above in a separate PyRPMD directory.
Install NumPy, SciPy and Matplotlib (or the repository's documented environment),
then run from that checkout, replacing `/absolute/extracted-archive`:

```bash
PYTHONPATH=src:. python -m benchmarks.h3.analyze_preparation_study \\
  --baseline /absolute/extracted-archive/data/state_selected_reproduction \\
  --study /absolute/extracted-archive/data/state_preparation_study \\
  --output /absolute/new-reanalysis-directory
```

Reanalysis uses archived data; it does not require evaluating the BKMP2 PES.
The original absolute paths inside provenance records are retained as evidence
of the producing environment. The portable dataset roots are the two paths
under `data/` shown above. Use a fresh reanalysis output to preserve this copy.

## Contents and limits

The `data/state_preparation_study/analysis/` directory holds JSON/CSV reports,
standalone HTML, PNG/SVG figures and the generated PDF when it existed at
packaging time. The isolated audit manifest and per-case time-mean files are
included. `audit_cases.csv` and `audit_paired_outcomes.csv` retain completed
cases and pair counts even when no preparation succeeds. Corresponding means
and SEM are unavailable in that case; empty statistics CSVs contain no old
results and the energy figure labels the absence of successful preparations.
The baseline also supplies the independent H2 radial-level reference
and the digitized 2021 figure points/interpolation/provenance. Those approximate
figure-reading tolerances are distinct from Monte Carlo confidence intervals.

Smoke/pilot cases, execution logs, publisher PDFs, third-party PES Fortran and
binaries, compiled kernels, and duplicate repository code are excluded. Build
the pinned PES separately only when rerunning dynamics; its origin/hashes stay
in the data provenance. The full-PES j=0 action orbit is leading-order EBK,
and the smooth-switched multi-bead construction is experimental. No exact
quantum pure-state, final-state population, or universal convergence claim is
implied by this archive.

ZIP members have fixed timestamps/permissions and sorted order. Identical
inputs and code commit give reproducible bytes with the same Python/zlib
compression implementation. Dataset timestamps inside original files are
preserved verbatim.
""".encode()


def _checksums(hashes):
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())).encode()


def _zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def write_archive(plan, output):
    output = Path(output).resolve()
    if output.suffix.lower() != ".zip":
        raise ArchiveError("--output must end in .zip")
    sidecar = Path(str(output)+".sha256")
    member_sidecar = Path(str(output)+".members.sha256")
    destinations = (output, sidecar, member_sidecar)
    if any(p.exists() for p in destinations):
        raise ArchiveError("Archive or checksum destination already exists; choose a fresh output")
    payload = {"README.md": _readme(plan)}
    payload["ARCHIVE_MANIFEST.json"] = _json_bytes({
        "schema": "pyrpmd.h3.preparation-data-archive.v1",
        "code_commit": plan.code_commit, "code_url": f"{REPOSITORY_URL}/tree/{plan.code_commit}",
        "historical_baseline_code_commit": BASELINE_CODE,
        "case_counts": {"new_collisions": 12, "isolated_audits": 12, "baseline": 6},
        "attempt_counts": {"new_collisions": 30720, "isolated_audits": 6144, "baseline": 24576},
        "verified_current_code_sha256": dict(sorted(plan.code_hashes.items())),
        "files": [{"path": m.name, "bytes": m.size, "sha256": m.sha256}
                  for m in sorted(plan.members.values(), key=lambda item: item.name)],
        "integrity": "SHA256SUMS covers all other ZIP members; external .members.sha256 covers every ZIP member",
        "compression": "ZIP_DEFLATED level 6; sorted names; fixed 1980-01-01 timestamps and mode 0644"})
    hashes = {name: member.sha256 for name, member in plan.members.items()}
    hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in payload.items()})
    payload["SHA256SUMS"] = _checksums(hashes)
    hashes["SHA256SUMS"] = hashlib.sha256(payload["SHA256SUMS"]).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    published = []
    with tempfile.TemporaryDirectory(prefix="preparation-package-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary/"archive.zip"
        plan.verify_all()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True) as target:
            for name in sorted(hashes):
                if name in payload:
                    target.writestr(_zip_info(name), payload[name], compresslevel=6)
                else:
                    member, hasher = plan.members[name], hashlib.sha256()
                    with member.path.open("rb") as source, target.open(_zip_info(name), "w", force_zip64=True) as destination:
                        for block in iter(lambda: source.read(CHUNK), b""):
                            hasher.update(block); destination.write(block)
                    if hasher.hexdigest() != member.sha256:
                        raise ArchiveError(f"Input changed while writing ZIP: {member.path}")
        plan.verify_all()
        with zipfile.ZipFile(archive) as check:
            if set(check.namelist()) != set(hashes) or len(check.namelist()) != len(hashes):
                raise ArchiveError("ZIP member inventory mismatch")
            for name in sorted(hashes):
                hasher = hashlib.sha256()
                with check.open(name) as stream:
                    for block in iter(lambda: stream.read(CHUNK), b""):
                        hasher.update(block)
                if hasher.hexdigest() != hashes[name]:
                    raise ArchiveError(f"ZIP round-trip checksum mismatch: {name}")
        plan.verify_all()
        zip_hash = _digest(archive)
        checks = temporary/"archive.sha256"
        checks.write_text(f"{zip_hash}  {output.name}\n")
        members = temporary/"members.sha256"
        members.write_bytes(_checksums(hashes))
        try:
            # Same-directory hard links publish atomically without overwriting
            # an output created by another process after the initial check.
            for source, destination in ((archive, output), (checks, sidecar), (members, member_sidecar)):
                os.link(source, destination)
                published.append(destination)
        except BaseException:
            for path in published:
                path.unlink()
            raise
    return {"output": str(output), "bytes": output.stat().st_size, "sha256": zip_hash,
            "members": len(hashes), "checksum": str(sidecar), "member_checksums": str(member_sidecar)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT/"runs/state_selected_reproduction")
    parser.add_argument("--study", type=Path, default=ROOT/"runs/state_preparation_study")
    parser.add_argument("--output", type=Path, help="New ZIP path; never overwritten")
    parser.add_argument("--code-commit", required=True, help="Exact committed follow-up source SHA (40 hex characters)")
    parser.add_argument("--dry-run", action="store_true", help="Fully validate completed inputs and list payload; write nothing")
    args = parser.parse_args()
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run is used")
    try:
        plan = prepare_plan(args.baseline, args.study, args.code_commit)
        if args.dry_run:
            report = {"validated": True, "code_commit": plan.code_commit,
                "input_files": len(plan.members), "input_bytes": sum(m.size for m in plan.members.values()),
                "files": sorted(plan.members), "current_code_files_verified": len(plan.code_hashes)}
        else:
            report = write_archive(plan, args.output)
    except (ArchiveError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Archive refused: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
