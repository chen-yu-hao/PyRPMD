"""Package completed numerical evidence only, with a verified member manifest.

No PES source, native binary or Git-backed program is copied into the data
archives. Earlier CV stages are represented by the exact final observation
array plus their original prefix hashes and state snapshots.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.h3.classical_sampling_campaign import file_hash

ALLOWED = {".npz", ".json", ".csv", ".txt", ".log"}
GROUPS = {
    "equilibrium": ("pyrpmd-h3-classical-equilibrium-2026-11-01", [
        "campaign", "reference-128", "reference-256", "reference-384", "estimator-oracle",
        "previous-window-diagnosis", "analysis-20000", "analysis-65536", "analysis-65536-final",
        "analysis-262144", "replay-20000-analysis", "numpy-compatibility",
        "replay-20000/prefix-replay.json"]),
    "transmission": ("pyrpmd-h3-classical-transmission-2026-11-01", [
        "transmission", "transmission-diagnostics"]),
    "v2-calibration": ("pyrpmd-h3-classical-v2-calibration-2026-11-01", [
        "v2-calibration", "v2-calibration-analysis"]),
}


def _metadata_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True)+"\n").encode()


def _add_bytes(archive, name, contents):
    info = tarfile.TarInfo(name)
    info.size = len(contents)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(contents))


def pack(data, output, evidence, group):
    data, output, evidence = Path(data).resolve(), Path(output).resolve(), Path(evidence).resolve()
    name, folders = GROUPS[group]
    files = []
    for relative in folders:
        path = data/relative
        if not path.exists():
            raise ValueError(f"completed evidence is missing: {relative}")
        files.extend([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()))
    if any(path.suffix not in ALLOWED for path in files):
        raise ValueError("unexpected non-data member; source code and binaries are not archive payloads")
    if group == "equilibrium":
        summary = json.loads((data/"campaign/campaign-stage-262144.json").read_text())
        if not summary["all_stages_completed"]:
            raise ValueError("original campaign stage is incomplete")
    elif group == "transmission":
        if not (data/"transmission/transmission-comparison.json").is_file():
            raise ValueError("transmission campaign comparison is missing")
    else:
        if not (data/"v2-calibration-analysis/calibration-comparison.json").is_file():
            raise ValueError("calibration analysis is incomplete")
    output.mkdir(parents=True, exist_ok=True)
    target = output/(name+".tar.gz")
    manifest = {"schema": "pyrpmd.h3.classical-raw-data-manifest.v1", "group": group,
                "raw_data_only": True, "members": [],
                "code_location": "PyRPMD repository, benchmarks/h3/classical_sampling*.py; exact source hashes are in the per-run receipts and repository evidence.",
                "excluded": ["BKMP2 source", "compiled potential/library files", "Git-backed program source", "duplicate CV prefix checkpoints"],
                "initial_oracle_provenance_limit": "The initial 128/64 oracle retains its original source hash and raw arrays, but its full script snapshot was not saved. A portable endpoint replay differs by only 5.33e-15. Final 256/384 oracle source snapshots and all production/TS/V2 sampler sources are retained in the repository."}
    readme = (
        f"{name}\n\n"
        "This archive contains numerical data, configurations and execution receipts only.\n"
        "Read manifest.json for every member SHA256. No BKMP2 source or compiled binary is included.\n"
        "Use the PyRPMD repository's docs/validation/literature-convergence-2026-11-01/h3-classical/README.md\n"
        "for scientific scope, primary paper values, exact source identities and replay commands.\n"
        "The original 20000/65536/262144 stages share one preserved full CV array per seed;\n"
        "stage receipts and RNG/geometry snapshots reconstruct earlier prefixes without resampling.\n"
        "The independent classical transmission result is separate from umbrella convergence.\n"
        "V2 calibration observations must not be included in a later primary estimate.\n"
        "PES source SHA256: 6fa37803d0d1db4f43e5bc4bfccd5e26585f01116dae9dc4f6d0d28d5d71d237.\n"
        "Build the potential separately according to benchmarks/h3/README.md.\n"
    ).encode()
    with target.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0, compresslevel=1) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for path in sorted(set(files)):
                    relative = path.relative_to(data).as_posix()
                    with path.open("rb") as source:
                        digest = hashlib.sha256()
                        size = 0
                        for block in iter(lambda: source.read(1024*1024), b""):
                            digest.update(block)
                            size += len(block)
                        source.seek(0)
                        info = tarfile.TarInfo(name+"/"+relative)
                        info.size, info.mode, info.mtime = size, 0o644, 0
                        archive.addfile(info, source)
                    manifest["members"].append({"path": relative, "bytes": size, "sha256": digest.hexdigest()})
                _add_bytes(archive, name+"/README.txt", readme)
                manifest["members"].append({"path": "README.txt", "bytes": len(readme),
                                            "sha256": hashlib.sha256(readme).hexdigest()})
                _add_bytes(archive, name+"/manifest.json", _metadata_bytes(manifest))
    expected = {name+"/"+row["path"]: row for row in manifest["members"]}
    verified = set()
    manifest_verified = False
    with tarfile.open(target, "r:gz") as archive:
        for member in archive:
            if member.name == name+"/manifest.json":
                if archive.extractfile(member).read() != _metadata_bytes(manifest):
                    raise ValueError("archive manifest differs after compression")
                manifest_verified = True
                continue
            row = expected[member.name]
            digest = hashlib.sha256()
            with archive.extractfile(member) as source:
                for block in iter(lambda: source.read(1024*1024), b""):
                    digest.update(block)
            if member.size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"archive member failed round-trip verification: {member.name}")
            verified.add(member.name)
    if verified != set(expected) or not manifest_verified:
        raise ValueError("archive inventory is incomplete")
    entry = {"group": group, "filename": target.name, "bytes": target.stat().st_size,
             "sha256": file_hash(target), "data_members": len(manifest["members"]),
             "payload_bytes": sum(row["bytes"] for row in manifest["members"]),
             "all_member_hashes_verified_after_compression": True,
             "manifest_sha256": hashlib.sha256(_metadata_bytes(manifest)).hexdigest(),
             "archive_root": name, "packing_source_sha256": file_hash(Path(__file__))}
    index_path = evidence/"raw-data-index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "schema": "pyrpmd.h3.classical-raw-data-index.v1", "archives": [], "raw_data_only": True}
    index["archives"] = [row for row in index["archives"] if row["group"] != group]+[entry]
    index["archives"].sort(key=lambda row: row["group"])
    index_path.write_bytes(_metadata_bytes(index))
    (evidence/f"raw-data-manifest-{group}.json").write_bytes(_metadata_bytes(manifest))
    print(json.dumps(entry, indent=2), flush=True)
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--group", choices=GROUPS, required=True)
    args = parser.parse_args()
    pack(args.data, args.output, args.evidence, args.group)
