"""Acceptance-harness guards and an opt-in real EAM end-to-end check."""

import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.materials.batch_acceptance import configuration, read_streams, run_acceptance
from benchmarks.materials.batch_convergence import confidence_interval, equivalence, protocol


def test_convergence_controls_preserve_time_and_use_independent_streams():
    design = protocol(replicas=4, beads=4, equilibration_steps=16, production_steps=32)
    base, refined = design["arms"]["base"], design["arms"]["timestep"]
    for count in ("equilibration_steps", "production_steps", "sample_interval"):
        assert base[count] * base["dt_fs"] == refined[count] * refined["dt_fs"]
    assert len({arm["seed"] for arm in design["arms"].values()}) == len(design["arms"])
    assert design["arms"]["cell_size"]["lateral_repeat"] == 2
    with pytest.raises(ValueError, match="replicas"):
        protocol(replicas=3)


def test_equivalence_requires_precision_and_rejects_missing_evidence():
    # Identical means with huge uncertainty must not be called equivalent.
    broad = [-10., -1., 1., 10.]
    assert equivalence(broad, broad, .1)["passed"] is False
    narrow = [1., 1.0001, .9999, 1.]
    assert equivalence(narrow, narrow, .01)["passed"] is True
    assert equivalence(narrow, np.array(narrow)+.1, .01)["passed"] is False
    assert equivalence(narrow[:3], narrow, .01)["passed"] is False
    assert confidence_interval([1., 1., np.nan, 1.])["available"] is False


def test_configuration_preserves_coverage_and_public_task_contract(tmp_path):
    pytest.importorskip("ase")
    from pyrpmd.input import JobSpec

    model = tmp_path / "table.eam.alloy"
    model.write_text("configuration-only placeholder; no PES evaluation")
    primitive = configuration(model, "pimd")
    repeated = configuration(model, "pimd", lateral_repeat=2)
    assert "diffusion" not in primitive
    JobSpec.from_mapping(primitive).resolve()
    for symbol in ("Ni", "H"):
        assert repeated["system"]["symbols"].count(symbol) == 4*primitive["system"]["symbols"].count(symbol)
    transport = configuration(model, "diffusion", mode="mpi_dynamic_batch")
    JobSpec.from_mapping(transport).resolve()
    assert transport["parallel"]["coordinator_ranks"] == 1
    assert transport["potential"]["trajectory_batch_safe"] is True


def test_real_eam_batch_oracles_exact_recovery_and_missing_replica_rejection(tmp_path):
    model = os.environ.get("PYRPMD_NIALH_POTENTIAL")
    if not model:
        pytest.skip("Set PYRPMD_NIALH_POTENTIAL to explicitly opt into the real EAM acceptance")
    pytest.importorskip("ase")
    report = run_acceptance(model, tmp_path / "acceptance", modes=("serial", "batch"), tasks=("pimd",))
    assert report["implementation_checks_passed"] is True
    assert report["physical_converged"] is False
    assert report["oracle"]["all_nine_strain_components_max_error_eV"] < 1e-4
    reference, batched = report["cases"]
    assert batched["measurement"]["pes_calls"] < reference["measurement"]["pes_calls"]
    assert batched["measurement"]["pes_frames"] == reference["measurement"]["pes_frames"]
    assert any(row["shape"][0] == 8 for row in batched["measurement"]["shapes"])
    recovery = report["recovery"]["pimd"]
    assert recovery["committed_chunks_preserved"] > 0
    assert recovery["terminal_restart"]["pes_calls"] == 0
    assert recovery["terminal_restart"]["committed_replica_steps_advanced"] == 0
    directory = tmp_path / "acceptance/pimd-batch"
    _, _, streams = read_streams(directory)
    assert len(streams) == 5
    assert len({str(stream["rng"]) for stream in streams}) == 5
    checkpoint = directory / "checkpoints/materials_0004.npz"
    checkpoint.unlink()
    with pytest.raises(AssertionError, match="Missing"):
        read_streams(directory)

