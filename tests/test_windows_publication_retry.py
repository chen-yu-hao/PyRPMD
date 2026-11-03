"""Windows contention retries preserve atomicity and explicit failure."""

import errno
import json
from types import SimpleNamespace

import pytest

from pyrpmd import io

from test_atomic_json import WRITERS


def _windows_error(code):
    error = PermissionError(errno.EACCES, "simulated Windows replacement failure")
    error.winerror = code
    return error


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_conflict_retries_same_complete_temporary(tmp_path, monkeypatch, writer, code):
    path = tmp_path / "result.json"
    writer(path, {"accepted": 1})
    previous = path.read_bytes()
    replace = io.os.replace
    attempts, pauses = [], []

    def busy_twice(source, destination):
        assert path.read_bytes() == previous
        assert json.loads(io.Path(source).read_text(encoding="utf-8")) == {"accepted": 2}
        attempts.append(source)
        if len(attempts) <= 2:
            raise _windows_error(code)
        return replace(source, destination)

    monkeypatch.setattr(io, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(io.os, "replace", busy_twice)
    monkeypatch.setattr(io.time, "sleep", pauses.append)
    writer(path, {"accepted": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"accepted": 2}
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert len(pauses) == 2 and 0 < sum(pauses) < .2
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("platform,code,attempt_limit", [
    ("win32", 5, 6), ("win32", 32, 6), ("win32", 33, 6),
    ("win32", None, 1), ("win32", 112, 1), ("linux", 5, 1),
])
def test_permanent_or_unrelated_failure_is_bounded_and_preserves_previous_file(
    tmp_path, monkeypatch, platform, code, attempt_limit,
):
    path = tmp_path / "result.json"
    io.write_json(path, {"accepted": 1})
    previous = path.read_bytes()
    failure = (_windows_error(code) if code is not None
               else PermissionError(errno.EACCES, "permanent file permissions"))
    attempts, pauses = [], []

    def denied(source, destination):
        attempts.append(source)
        raise failure

    monkeypatch.setattr(io, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(io.os, "replace", denied)
    monkeypatch.setattr(io.time, "sleep", pauses.append)
    with pytest.raises(PermissionError) as caught:
        io.write_json(path, {"accepted": 2})
    assert caught.value is failure
    assert len(attempts) == attempt_limit
    assert len(pauses) == attempt_limit - 1 and sum(pauses) < .2
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
