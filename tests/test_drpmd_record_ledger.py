"""Regression coverage for scalable dRPMD parent checkpoints and summaries."""

import csv
import json

import numpy as np
import pytest

from pyrpmd.workflows import drpmd
from pyrpmd.workflows.drpmd_records import (
    EMPTY_DIGEST, LEGACY_PARENT_KIND, PARENT_KIND, ParentRecordLedger,
)
from test_drpmd import config, system


def _short_run(tmp_path, trajectories=3):
    options = config()
    options["collision"].update(trajectories=trajectories, max_steps=2)
    result = drpmd.run_drpmd(options, system(), tmp_path)
    checkpoint = tmp_path / "checkpoints" / "parent-000000.npz"
    return options, result, checkpoint


def test_compact_parent_checkpoint_and_trajectory_table(tmp_path):
    options, result, checkpoint = _short_run(tmp_path)
    metadata, _ = drpmd._load_npz(checkpoint, expected_kind=PARENT_KIND)
    assert metadata["trajectory_start"] == 0
    assert metadata["trajectory_count"] == 3
    assert metadata["record_count"] == 3
    assert "trajectory_ids" not in metadata
    assert "records" not in metadata
    ledger = ParentRecordLedger(checkpoint)
    assert ledger.restore(metadata) == result["records"]

    saved = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert "records" not in saved
    assert saved["trajectory_table"] == "results/trajectories.csv"
    assert saved["counts"] == result["counts"]
    with (tmp_path / saved["trajectory_table"]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["trajectory_id"]) for row in rows] == [0, 1, 2]
    assert json.loads(rows[0]["launch"]) == result["records"][0]["launch"]

    options["resume"] = True
    assert drpmd.run_drpmd(options, system(), tmp_path)["records"] == result["records"]


def test_legacy_checkpoint_migrates_without_resampling(tmp_path):
    options, result, checkpoint = _short_run(tmp_path, trajectories=2)
    metadata, arrays = drpmd._load_npz(checkpoint)
    metadata.update(format=LEGACY_PARENT_KIND, trajectory_ids=[0, 1],
                    records=result["records"])
    for key in ("trajectory_start", "trajectory_count", "record_count", "record_digest"):
        metadata.pop(key)
    drpmd._atomic_npz(checkpoint, metadata, **arrays)
    ParentRecordLedger(checkpoint).path.unlink()

    options["resume"] = True
    assert drpmd.run_drpmd(options, system(), tmp_path)["records"] == result["records"]
    migrated, _ = drpmd._load_npz(checkpoint, expected_kind=PARENT_KIND)
    assert migrated["record_count"] == 2
    assert "records" not in migrated
    assert len(ParentRecordLedger(checkpoint).restore(migrated)) == 2


def test_uncheckpointed_append_is_rolled_back_on_resume(tmp_path):
    options, result, checkpoint = _short_run(tmp_path, trajectories=2)
    ledger = ParentRecordLedger(checkpoint)
    ledger.append({"trajectory_id": 2, "status": "uncommitted"})
    options["resume"] = True
    assert drpmd.run_drpmd(options, system(), tmp_path)["records"] == result["records"]
    metadata, _ = drpmd._load_npz(checkpoint)
    assert metadata["record_count"] == 2
    assert len(ledger.path.read_text().splitlines()) == 2


def test_record_ledger_detects_committed_corruption(tmp_path):
    options, _, checkpoint = _short_run(tmp_path, trajectories=2)
    ledger = ParentRecordLedger(checkpoint)
    raw = bytearray(ledger.path.read_bytes())
    raw[0] = ord("x")  # The first committed line is no longer valid JSON.
    ledger.path.write_bytes(raw)
    options["resume"] = True
    with pytest.raises(ValueError, match="record ledger"):
        drpmd.run_drpmd(options, system(), tmp_path)


def test_parent_checkpoint_size_independent_of_record_history(tmp_path):
    """5000 records alter only the fixed-width digest and an integer count."""
    tmp_path.joinpath("checkpoints").mkdir()
    checkpoint = tmp_path / "checkpoints" / "parent-000000.npz"
    ledger = ParentRecordLedger(checkpoint)
    records = [{"trajectory_id": i, "status": "nonreactive",
                "launch": {"speed_au": 1.234}, "payload": "x" * 500}
               for i in range(5000)]
    arrays = {"q": np.zeros((1, 2, 3)), "p": np.zeros((1, 2, 3)),
              "masses": np.ones(2)}
    sizes = []
    for count in (10, 5000):
        ledger.replace(records[:count])
        metadata = {"format": PARENT_KIND, "parent_id": 0,
                    "trajectory_start": 0, "trajectory_count": 5000,
                    "next_index": count, "record_count": ledger.count,
                    "record_digest": ledger.digest}
        drpmd._atomic_npz(checkpoint, metadata, **arrays)
        sizes.append(checkpoint.stat().st_size)
    assert sizes[1] <= sizes[0] + 512
    assert sizes[1] < 4096
    assert ledger.count == 5000
    assert ledger.digest != EMPTY_DIGEST
    assert len(ledger.restore(metadata)) == 5000
