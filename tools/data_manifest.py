#!/usr/bin/env python3
"""Read-only SHA-256 inventory of a worktree, emitted as deterministic JSON.

Only the Python standard library is required. Paths in the output are relative
to the scanned root, use forward slashes, and never include a timestamp.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


DEFAULT_EXCLUDES = (
    ".git", ".venv", ".venv-*", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "pytest-of-*", "build", "dist", "_build",
    "*.egg-info", "*.pyc", "*.pyo", "*.so", "*.pyd", "*.o",
)
CHUNK_BYTES = 1024 * 1024


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    # Windows path stat can add executable permission bits and report creation
    # time as ctime while fstat reports metadata-change time. Compare only
    # compatible fields across these APIs (CPython issue #157671).
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode),
            info.st_size, info.st_mtime_ns)


def _signature(info: os.stat_result) -> tuple[int, ...]:
    # Full metadata remains useful before/after through the same API. Reads may
    # update atime, so it is deliberately absent from the comparison.
    return _file_identity(info) + (info.st_mode, info.st_ctime_ns,
                                  getattr(info, "st_file_attributes", 0),
                                  getattr(info, "st_reparse_tag", 0))


def _link_kind(info: os.stat_result) -> str | None:
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    reparse_tag = getattr(info, "st_reparse_tag", 0)
    if reparse_tag and reparse_tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None):
        return "junction"
    # A Windows reparse point can look like an ordinary directory or file.
    # Do not traverse unknown tags (including links to external/cloud data).
    if reparse_tag or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return "reparse point"
    return None


def _sha256(path: Path, expected: os.stat_result) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _file_identity(opened) != _file_identity(expected):
            raise RuntimeError(f"file changed before hashing: {path}")
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
        if _signature(os.fstat(stream.fileno())) != _signature(opened):
            raise RuntimeError(f"file changed while hashing: {path}")
    if _signature(path.stat(follow_symlinks=False)) != _signature(expected):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _empty_totals() -> dict[str, int]:
    return {"files": 0, "bytes": 0, "large_files": 0, "large_bytes": 0}


def collect_manifest(
    root: Path | str,
    *,
    min_bytes: int = 1024 * 1024,
    directory_depth: int = 2,
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> dict:
    """Hash included regular files; raise on unreadable or changing files.

    This scans the filesystem, including untracked files, not Git's index.
    Exclusion patterns containing '/' match a root-relative POSIX path; other
    patterns match an entry's basename at any depth. Matching directories are
    pruned. Symlinks, Windows reparse points and special files are reported and
    are never traversed.
    """
    if type(min_bytes) is not int or min_bytes < 0:
        raise ValueError("min_bytes must be a nonnegative integer")
    if type(directory_depth) is not int or directory_depth < 1:
        raise ValueError("directory_depth must be a positive integer")
    root = Path(root)
    invalid_root = f"root must be a real directory, not a symlink or reparse point: {root}"
    try:
        root_info = root.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(invalid_root) from error
    if _link_kind(root_info) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(invalid_root)
    root = root.resolve(strict=True)
    excludes = tuple(sorted(set(excludes)))
    files = []
    skipped = []

    def walk(directory: Path) -> None:
        # Sorting and explicit stat calls avoid hiding permission/stat errors.
        for path in sorted(directory.iterdir(), key=lambda value: value.name):
            relative = path.relative_to(root).as_posix()
            pattern = next((pattern for pattern in excludes if fnmatch.fnmatchcase(
                relative if "/" in pattern else path.name, pattern)), None)
            if pattern is not None:
                skipped.append({"path": relative, "reason": "excluded", "pattern": pattern})
                continue
            info = path.stat(follow_symlinks=False)
            link_kind = _link_kind(info)
            if link_kind:
                skipped.append({"path": relative, "reason": link_kind})
            elif stat.S_ISDIR(info.st_mode):
                walk(path)
            elif stat.S_ISREG(info.st_mode):
                files.append({"path": relative, "bytes": info.st_size,
                              "sha256": _sha256(path, info)})
            else:
                skipped.append({"path": relative, "reason": "not a regular file"})

    walk(root)
    files.sort(key=lambda record: record["path"])
    skipped.sort(key=lambda record: record["path"])
    totals = _empty_totals()
    by_directory = defaultdict(_empty_totals)
    by_type = defaultdict(_empty_totals)
    for record in files:
        path = Path(record["path"])
        directory = "/".join(path.parts[:-1][:directory_depth]) or "."
        suffix = path.suffix.lower() or "(no suffix)"
        for summary in (totals, by_directory[directory], by_type[suffix]):
            summary["files"] += 1
            summary["bytes"] += record["bytes"]
            if record["bytes"] >= min_bytes:
                summary["large_files"] += 1
                summary["large_bytes"] += record["bytes"]
    return {
        "schema": "pyrpmd.data-manifest.v1",
        "scope": "worktree regular files; paths relative to the requested root",
        "hash_algorithm": "sha256",
        "min_bytes": min_bytes,
        "directory_depth": directory_depth,
        "exclude_patterns": list(excludes),
        "totals": totals,
        "by_directory": dict(sorted(by_directory.items())),
        "by_type": dict(sorted(by_type.items())),
        "large_files": sorted((record for record in files if record["bytes"] >= min_bytes),
                              key=lambda record: (-record["bytes"], record["path"])),
        "files": files,
        "skipped": skipped,
    }


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _positive(value: str) -> int:
    number = _nonnegative(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="directory to scan (default: .)")
    parser.add_argument("--min-bytes", type=_nonnegative, default=1024 * 1024,
                        help="inclusive large-file threshold in bytes (default: 1048576)")
    parser.add_argument("--directory-depth", type=_positive, default=2,
                        help="group by this many leading parent directories (default: 2)")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                        help="additional case-sensitive basename or root-relative glob; repeatable")
    parser.add_argument("--no-default-excludes", action="store_true",
                        help="disable the documented build/cache/Git exclusions")
    args = parser.parse_args(argv)
    excludes = (() if args.no_default_excludes else DEFAULT_EXCLUDES) + tuple(args.exclude)
    try:
        result = collect_manifest(args.root, min_bytes=args.min_bytes,
                                  directory_depth=args.directory_depth, excludes=excludes)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"data_manifest: error: {error}", file=sys.stderr)
        return 1
    # ensure_ascii keeps redirected output valid UTF-8 even under a Windows
    # legacy stdout encoding; JSON consumers recover the original path text.
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
