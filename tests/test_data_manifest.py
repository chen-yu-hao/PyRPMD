"""Read-only inventory contracts, with published SHA-256 known-answer vectors."""

import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "data_manifest.py"
SPEC = importlib.util.spec_from_file_location("data_manifest_tool", SCRIPT)
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def test_hashes_totals_and_large_file_threshold_have_independent_references(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "c").mkdir()
    (tmp_path / "a" / "b" / "abc.JSON").write_bytes(b"abc")
    (tmp_path / "a" / "c" / "empty.json").write_bytes(b"")
    (tmp_path / "root").write_bytes(b"ab")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in tmp_path.rglob("*") if path.is_file()}

    result = manifest.collect_manifest(tmp_path, min_bytes=3, directory_depth=1)

    assert result["totals"] == {"files": 3, "bytes": 5, "large_files": 1, "large_bytes": 3}
    assert result["by_directory"] == {
        ".": {"files": 1, "bytes": 2, "large_files": 0, "large_bytes": 0},
        "a": {"files": 2, "bytes": 3, "large_files": 1, "large_bytes": 3},
    }
    assert result["by_type"][".json"] == result["by_directory"]["a"]
    files = {record["path"]: record for record in result["files"]}
    # FIPS 180-4 SHA-256 test vectors, independent of the scanner's hash helper.
    assert files["a/b/abc.JSON"]["sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert files["a/c/empty.json"]["sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert result["large_files"] == [files["a/b/abc.JSON"]]
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before} == before


def test_exclusions_are_explicit_and_can_be_disabled(tmp_path):
    for directory in (".git", "pytest-of-user", "data", "other"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "record.json").write_bytes(b"abc")
    result = manifest.collect_manifest(tmp_path, excludes=manifest.DEFAULT_EXCLUDES + ("data/*.json",))
    assert [record["path"] for record in result["files"]] == ["other/record.json"]
    assert result["skipped"] == [
        {"path": ".git", "reason": "excluded", "pattern": ".git"},
        {"path": "data/record.json", "reason": "excluded", "pattern": "data/*.json"},
        {"path": "pytest-of-user", "reason": "excluded", "pattern": "pytest-of-*"},
    ]
    assert manifest.collect_manifest(tmp_path, excludes=())["totals"]["files"] == 4


def test_symlinks_are_reported_without_following_targets(tmp_path):
    root = tmp_path / "scan"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_bytes(b"abc")
    try:
        (root / "external").symlink_to(outside, target_is_directory=True)
        (root / "broken").symlink_to(tmp_path / "missing")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"host does not permit creating symbolic links: {error}")
    result = manifest.collect_manifest(root)
    assert result["files"] == []
    assert result["skipped"] == [
        {"path": "broken", "reason": "symlink"},
        {"path": "external", "reason": "symlink"},
    ]
    with pytest.raises(ValueError, match="real directory"):
        manifest.collect_manifest(root / "external")


@pytest.mark.parametrize("tag,attributes,reason", [
    (0xA0000003, stat.FILE_ATTRIBUTE_REPARSE_POINT, "junction"),
    (0x80000001, stat.FILE_ATTRIBUTE_REPARSE_POINT, "reparse point"),
    (0, stat.FILE_ATTRIBUTE_REPARSE_POINT, "reparse point"),
])
def test_directory_reparse_metadata_is_never_traversed(tmp_path, monkeypatch, tag, attributes, reason):
    external = tmp_path / "external"
    external.mkdir()
    (external / "must-not-be-read.json").write_bytes(b"abc")
    original_stat = Path.stat
    monkeypatch.setattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003, raising=False)

    def reparse_stat(self, *args, **kwargs):
        if self == external:
            assert kwargs.get("follow_symlinks") is False
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_reparse_tag=tag,
                                   st_file_attributes=attributes)
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", reparse_stat)
    result = manifest.collect_manifest(tmp_path)
    assert result["files"] == []
    assert result["skipped"] == [{"path": "external", "reason": reason}]
    with pytest.raises(ValueError, match="reparse point"):
        manifest.collect_manifest(external)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction integration requires cmd /c mklink /J")
def test_windows_junctions_do_not_escape_root_or_create_cycles(tmp_path):
    root = tmp_path / "scan"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "must-not-be-read.json").write_bytes(b"abc")
    (root / "local.json").write_bytes(b"abc")
    junctions = []
    try:
        for name, target in (("external", outside), ("cycle", root)):
            junction = root / name
            try:
                result = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(junction), str(target)],
                                        capture_output=True, text=True)
            except OSError as error:
                pytest.skip(f"Windows junction creation command unavailable: {error}")
            if result.returncode:
                pytest.skip(f"host does not permit directory junctions: {result.stdout} {result.stderr}")
            junctions.append(junction)
        result = manifest.collect_manifest(root)
        assert [record["path"] for record in result["files"]] == ["local.json"]
        assert result["skipped"] == [
            {"path": "cycle", "reason": "junction"},
            {"path": "external", "reason": "junction"},
        ]
        with pytest.raises(ValueError, match="reparse point"):
            manifest.collect_manifest(root / "external")
    finally:
        # rmdir removes a Windows junction itself, leaving its target intact.
        for junction in junctions:
            junction.rmdir()


def test_cli_is_deterministic_including_unicode_paths_and_empty_directories(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "数据.json").write_bytes(b"abc")
    command = [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--min-bytes", "0"]
    first = subprocess.run(command, check=True, capture_output=True)
    second = subprocess.run(command, check=True, capture_output=True)
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    result = json.loads(first.stdout.decode("utf-8"))
    assert result["files"][0]["path"] == "数据.json"
    assert result["totals"] == {"files": 1, "bytes": 3, "large_files": 1, "large_bytes": 3}
    assert str(tmp_path) not in first.stdout.decode("utf-8")


@pytest.mark.parametrize("options", [
    {"min_bytes": -1}, {"min_bytes": True}, {"directory_depth": 0}, {"directory_depth": 1.5},
])
def test_invalid_summary_parameters_fail_explicitly(tmp_path, options):
    with pytest.raises(ValueError, match="integer"):
        manifest.collect_manifest(tmp_path, **options)


def test_cli_missing_root_produces_an_error_without_partial_json(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--root", str(tmp_path / "missing")],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "root must be a real directory" in result.stderr


def test_read_error_is_not_silently_omitted(tmp_path, monkeypatch, capsys):
    path = tmp_path / "unreadable.json"
    path.write_bytes(b"abc")
    original_open = Path.open

    def denied_open(self, *args, **kwargs):
        if self == path:
            raise PermissionError(f"access denied: {self}")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied_open)
    assert manifest.main(["--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "access denied" in captured.err
    assert "unreadable.json" in captured.err


def test_changed_file_rejects_stale_inventory_metadata(tmp_path):
    path = tmp_path / "changing.json"
    path.write_bytes(b"abc")
    before = path.stat()
    path.write_bytes(b"abcd")
    with pytest.raises(RuntimeError, match="changed before hashing"):
        manifest._sha256(path, before)


def _stat_copy(info, **changes):
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns",
              "st_file_attributes", "st_reparse_tag")
    return SimpleNamespace(**{**{name: getattr(info, name, 0) for name in fields}, **changes})


@pytest.mark.parametrize("different_ctime,different_permissions", [(True, False), (False, True), (True, True)])
def test_path_stat_and_fstat_metadata_differences_are_supported(tmp_path, monkeypatch,
                                                              different_ctime, different_permissions):
    path = tmp_path / "script.bat"
    path.write_bytes(b"abc")
    original_fstat = os.fstat

    def windows_fstat(fd):
        info = original_fstat(fd)
        return _stat_copy(info,
                          st_ctime_ns=info.st_ctime_ns + (1 if different_ctime else 0),
                          st_mode=info.st_mode ^ (0o111 if different_permissions else 0))

    monkeypatch.setattr(os, "fstat", windows_fstat)
    result = manifest.collect_manifest(tmp_path)
    assert result["files"] == [{"path": "script.bat", "bytes": 3, "sha256":
                                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]


def test_changed_fstat_ctime_still_rejects_a_file(tmp_path, monkeypatch):
    path = tmp_path / "changing.json"
    path.write_bytes(b"abc")
    original_fstat = os.fstat
    calls = 0

    def changed_fstat(fd):
        nonlocal calls
        calls += 1
        info = original_fstat(fd)
        return _stat_copy(info, st_ctime_ns=info.st_ctime_ns + calls)

    monkeypatch.setattr(os, "fstat", changed_fstat)
    with pytest.raises(RuntimeError, match="changed while hashing"):
        manifest.collect_manifest(tmp_path)


def test_previously_edited_executable_suffix_file_can_be_hashed(tmp_path):
    path = tmp_path / "edited.bat"
    path.write_bytes(b"before")
    path.write_bytes(b"abc")
    # Exercise actual Windows creation/change-time and executable-bit behavior.
    # An explicit timestamp update does not rely on a filesystem clock tick.
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
    result = manifest.collect_manifest(tmp_path)
    assert result["files"] == [{"path": "edited.bat", "bytes": 3, "sha256":
                                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}]
