"""Inventory immutable tracked validation blobs without moving or deleting data.

The source is a Git commit, so sparse checkouts are supported. Emit the full
manifest outside the repository; payload candidates still require review of
test/figure dependencies and redistribution permissions before publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def manifest(root, revision="HEAD"):
    root = Path(root)
    commit = subprocess.check_output(["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=root, text=True).strip()
    listing = subprocess.check_output(["git", "ls-tree", "-r", "-l", "-z", commit,
                                      "docs/validation", "benchmarks"], cwd=root)
    rows = []
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=root,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as reader:
        for entry in listing.split(b"\0"):
            if not entry:
                continue
            header, raw_path = entry.split(b"\t", 1)
            mode, kind, oid, size = header.split()
            if kind != b"blob" or mode == b"120000":
                raise ValueError("archive inventory requires regular tracked files")
            path, count = raw_path.decode(), int(size)
            reader.stdin.write(oid + b"\n")
            reader.stdin.flush()
            returned = reader.stdout.readline().split()
            if returned != [oid, b"blob", size]:
                raise ValueError("Git blob metadata mismatch")
            digest = hashlib.sha256()
            remaining = count
            while remaining:
                block = reader.stdout.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("truncated Git blob")
                digest.update(block)
                remaining -= len(block)
            if reader.stdout.read(1) != b"\n":
                raise ValueError("invalid Git batch delimiter")
            keep = Path(path).suffix.lower() in {".md", ".py", ".sh", ".inp", ".toml", ".f", ".f90", ".c", ".cpp", ".h"}
            rows.append({"path": path, "bytes": count, "sha256": digest.hexdigest(),
                         "git_blob": oid.decode(), "classification": "keep_summary_or_source" if keep else "review_payload_dependencies"})
        reader.stdin.close()
        if reader.wait() != 0:
            raise RuntimeError("Git blob reader failed")
    return {"schema": "pyrpmd.archive-candidate.v1", "source_commit": commit,
            "published": False, "deletion_authorized": False,
            "files": rows, "file_count": len(rows), "bytes": sum(r["bytes"] for r in rows),
            "destination": None, "classification_status": "provisional_requires_dependency_and_license_review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    print(json.dumps(manifest(args.root, args.revision), indent=2))


if __name__ == "__main__":
    main()
