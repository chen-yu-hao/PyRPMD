"""One validated chunk traversal supplies offline materials analysis."""

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.trajectory import TrajectoryStream
from pyrpmd.workflows.materials import (
    STREAM_CHECKPOINT_KIND, _sample_replica, reanalyze_diffusion, run_materials,
)
from test_materials_extensions import fixture, task
from test_materials_stream_failures import setup


@pytest.mark.parametrize("return_data", [False, True])
def test_reanalysis_loads_each_chunk_once_and_matches_memory_results(tmp_path, monkeypatch, return_data):
    import pyrpmd.trajectory as trajectory

    config, system = setup("diffusion")
    memory_config = copy.deepcopy(config)
    memory_config["output"]["trajectory_storage"] = "memory"
    reference = run_materials(memory_config, system, tmp_path / "memory", return_data=True)
    output = tmp_path / "stream"
    run_materials(config, system, output)
    with load_checkpoint(output / "checkpoints/materials_0000.npz") as saved:
        descriptor = json.loads(str(saved["trajectory"]))
    original = trajectory._load_chunk
    reads = Counter()

    def counted(directory, digest, *args, **kwargs):
        reads[digest] += 1
        return original(directory, digest, *args, **kwargs)

    monkeypatch.setattr(trajectory, "_load_chunk", counted)
    replay = reanalyze_diffusion(config, system, output, return_data=return_data)
    assert len(reads) == descriptor["chunks"]
    assert set(reads.values()) == {1}
    summary = replay["summary"] if return_data else replay
    assert summary["diffusion"] == reference["summary"]["diffusion"]
    assert summary["nve_energy_diagnostics"] == reference["summary"]["nve_energy_diagnostics"]
    if return_data:
        for name in ("sample_steps", "positions", "velocities", "observables",
                     "radii_gyration_squared", "times"):
            np.testing.assert_array_equal(replay["streams"][0][name], reference["streams"][0][name])


def _replace_frames(stream, arrays, output):
    """Publish a valid container with deliberately invalid scientific values."""
    writer = TrajectoryStream(Path(output) / "samples/materials_0000",
                              schema=stream["trajectory"]["schema"], chunk_frames=3)
    writer.extend(arrays)
    with load_checkpoint(stream["checkpoint"]) as saved:
        payload = {name: saved[name] for name in saved.files}
    payload["trajectory"] = json.dumps(writer.prepare_commit())
    atomic_savez(stream["checkpoint"], kind=STREAM_CHECKPOINT_KIND, **payload)


@pytest.mark.parametrize("field", ["sample_steps", "positions", "velocities", "radii_gyration_squared"])
def test_reanalysis_checks_old_clock_and_unselected_fixed_atoms(tmp_path, field):
    config, system = fixture("diffusion")
    config["diffusion"].pop("reference_atoms")
    stream = _sample_replica(task(config, system, tmp_path))
    arrays = {name: stream[name].copy() for name in stream["trajectory"]["schema"]}
    if field == "sample_steps":
        arrays[field][0] += 1
        message = "phase clock"
    else:
        arrays[field][0, 1] += .25
        message = "fixed-atom"
    _replace_frames(stream, arrays, tmp_path)
    with pytest.raises(RuntimeError, match=message):
        reanalyze_diffusion(config, system, tmp_path)


def test_reanalysis_rejects_old_chunk_corruption_before_replacing_results(tmp_path):
    config, system = setup("diffusion")
    run_materials(config, system, tmp_path)
    summary = tmp_path / "summary.json"
    original_summary = summary.read_bytes()
    with load_checkpoint(tmp_path / "checkpoints/materials_0000.npz") as saved:
        descriptor = json.loads(str(saved["trajectory"]))
    directory = tmp_path / "samples/materials_0000"
    digest = descriptor["head"]
    while True:
        path = directory / f"{digest}.npz"
        with np.load(path, allow_pickle=False) as saved:
            previous = json.loads(str(saved["metadata"]))["previous"]
        if previous is None:
            break
        digest = previous
    path.write_bytes(b"damaged older chunk")
    with pytest.raises(RuntimeError, match="TrajectoryError: .*digest mismatch"):
        reanalyze_diffusion(config, system, tmp_path)
    assert summary.read_bytes() == original_summary
