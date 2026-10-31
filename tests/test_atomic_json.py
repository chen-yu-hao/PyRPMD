"""Durable JSON replacement and workflow-specific serialization contracts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from pyrpmd import io
from pyrpmd.workflows.drpmd import _atomic_json as write_collision_json
from pyrpmd.workflows.fes import write_json as write_fes_json


WRITERS = [io.write_json, write_fes_json, write_collision_json]


@pytest.mark.parametrize("writer", WRITERS)
def test_json_writers_create_parents_and_convert_scientific_values(tmp_path, writer):
    path = tmp_path / "new" / "nested" / "result.json"
    writer(path, {"array": np.arange(3), "scalar": np.float64(.5),
                  "path": Path("model.dat"), "nested": ({"value": np.int64(7)},)})
    assert json.loads(path.read_text()) == {
        "array": [0, 1, 2], "scalar": .5, "path": "model.dat",
        "nested": [{"value": 7}],
    }
    assert [p.name for p in path.parent.iterdir()] == ["result.json"]


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("failure", ["serialization", "fsync", "replace"])
def test_failed_json_replacement_preserves_previous_file_and_cleans_up(
    tmp_path, monkeypatch, writer, failure,
):
    path = tmp_path / "result.json"
    writer(path, {"accepted": 1})
    previous = path.read_bytes()
    replacement = {"accepted": 2}
    error = OSError

    def interrupted(*args, **kwargs):
        raise OSError("interrupted publication")

    if failure == "serialization":
        replacement["unsupported"] = object()
        error = TypeError
    else:
        monkeypatch.setattr(io.os, failure, interrupted)
    with pytest.raises(error):
        writer(path, replacement)
    assert path.read_bytes() == previous
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


@pytest.mark.parametrize("writer", WRITERS)
def test_independent_json_writers_do_not_share_a_temporary_file(tmp_path, monkeypatch, writer):
    path = tmp_path / "result.json"
    rendezvous = Barrier(2)
    replace = io.os.replace

    def publish_together(source, destination):
        rendezvous.wait(timeout=10)
        return replace(source, destination)

    monkeypatch.setattr(io.os, "replace", publish_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, path, {"writer": index, "values": [index] * 1000})
                   for index in (1, 2)]
        for future in futures:
            future.result()
    saved = json.loads(path.read_text())
    assert saved["writer"] in (1, 2)
    assert saved["values"] == [saved["writer"]] * 1000
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


@pytest.mark.parametrize("writer", [io.write_json, write_collision_json])
def test_strict_json_writers_reject_nonfinite_values(tmp_path, writer):
    with pytest.raises(ValueError, match="JSON"):
        writer(tmp_path / "result.json", {"estimate": np.array([np.nan, np.inf])})
    assert not (tmp_path / "result.json").exists()


def test_fes_writer_preserves_null_estimates_and_existing_text_format(tmp_path):
    path = tmp_path / "fes.json"
    write_fes_json(path, {"z": np.array([np.nan, np.inf, -np.inf]), "a": "温度"})
    assert json.loads(path.read_text()) == {"z": [None, None, None], "a": "温度"}
    raw = path.read_text()
    assert raw.index('"z"') < raw.index('"a"')
    assert "温度" not in raw
    assert raw.endswith("\n")


def test_collision_writer_preserves_dataclasses_sorted_keys_and_text_format(tmp_path):
    @dataclass
    class Result:
        samples: np.ndarray
        source: Path

    path = tmp_path / "collision.json"
    write_collision_json(path, {"z": "温度", "a": Result(np.array([2.]), Path("pes.so"))})
    raw = path.read_text()
    assert json.loads(raw) == {"a": {"samples": [2.], "source": "pes.so"}, "z": "温度"}
    assert raw.index('"a"') < raw.index('"z"')
    assert "温度" not in raw
    assert not raw.endswith("\n")


def test_generic_writer_preserves_unicode_and_newline(tmp_path):
    path = tmp_path / "manifest.json"
    io.write_json(path, {"message": "温度"})
    raw = path.read_text()
    assert "温度" in raw
    assert raw.endswith("\n")
