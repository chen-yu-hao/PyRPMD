"""Campaign restarts, immutable-force reuse, and conservative statistics."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.water.campaign import Config, FixedWaterCache, run, verify_manifest
from benchmarks.water.analyze_campaign import analyze, literature_diffusion_reference, rdf_oxygen, replay_compact
from pyrpmd.potentials.base import PESResult


class QuadraticProvider:
    def __init__(self):
        self.calls = 0

    def evaluate(self, q):
        self.calls += 1
        return PESResult(np.sum(q*q, axis=(1, 2))/2, -q, np.zeros(len(q), dtype=int))


def test_endpoint_cache_owns_geometry_and_results_and_rejects_cell_change():
    original = QuadraticProvider()
    cache = FixedWaterCache(original)
    q = np.ones((2, 3, 3))
    first = cache.evaluate(q)
    assert cache.evaluate(q.copy()) is first and original.calls == 1
    q[0, 0, 0] = 5
    second = cache.evaluate(q)
    assert original.calls == 2 and second.energy[0] != first.energy[0]
    assert first.forces[0, 0, 0] == -1
    with pytest.raises(ValueError):
        first.forces[:] = 0
    with pytest.raises(ValueError, match="fixed-cell"):
        cache.set_cell(np.eye(3))


def small_config():
    return Config(cells_per_axis=2, beads=4, cutoff_angstrom=2.5,
                  dt_fs=.1, equilibration_steps=4, segment_steps=16, segments=2,
                  record_stride=1, checkpoint_steps=7, platform="Reference", threads=1)


def endpoint(directory, manifest):
    with np.load(directory / manifest["chunks"][-1]["file"], allow_pickle=False) as data:
        return {key: data[key].copy() for key in ("q", "p", "rng_json", "state_step", "state_time")}


def test_resume_mid_nve_preserves_state_rng_and_segment_boundaries(tmp_path):
    pytest.importorskip("openmm")
    config = small_config()
    full, split = tmp_path / "full", tmp_path / "split"
    complete = run(full, config)
    incomplete = run(split, config, max_steps=11)
    assert not incomplete["complete"] and incomplete["committed_steps"] == 11
    resumed = run(split, config, resume=True)
    assert resumed["complete"] and resumed["committed_steps"] == config.total_steps
    first, second = endpoint(full, complete), endpoint(split, resumed)
    for key in ("q", "p"):
        np.testing.assert_allclose(first[key], second[key], atol=5e-13, rtol=1e-13)
    for key in ("rng_json", "state_step", "state_time"):
        np.testing.assert_array_equal(first[key], second[key])
    evidence = analyze(split, tmp_path / "summary", make_plot=False)
    assert len(evidence["segments"]) == 2
    assert not evidence["physical_converged"] and not evidence["literature_reproduced"]
    assert evidence["validated_diffusion_angstrom2_ps"] is None
    assert evidence["validated_standard_error_angstrom2_ps"] is None
    assert evidence["eligibility"]["this_independent_batches"] == 0
    assert all(not row["independent_replica"] for row in evidence["segments"])
    replay = replay_compact(tmp_path / "summary")
    assert replay["artifact_integrity_passed"] and replay["correlation_estimator_replay_passed"]
    (tmp_path / "summary" / "curves.npz").write_bytes(b"changed correlations")
    with pytest.raises(ValueError, match="checksum"):
        replay_compact(tmp_path / "summary")


def test_resume_identity_and_corrupt_checkpoint_are_rejected_before_propagation(tmp_path):
    pytest.importorskip("openmm")
    config = small_config()
    manifest = run(tmp_path, config, max_steps=7)
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="configuration/source"):
        run(tmp_path, replace(config, dt_fs=.2), resume=True)
    assert (tmp_path / "manifest.json").read_bytes() == before
    chunk = tmp_path / manifest["chunks"][-1]["file"]
    chunk.write_bytes(b"corrupted data")
    with pytest.raises(ValueError, match="checksum"):
        run(tmp_path, config, resume=True)
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_periodic_rdf_invariant_to_molecular_images():
    rng = np.random.default_rng(7)
    positions = rng.uniform(0, 15, size=(8, 32, 3))
    translated = positions.copy()
    from pyrpmd.constants import ANGSTROM_TO_AU
    translated[:, ::2, 1] += 15
    r1, g1 = rdf_oxygen(positions * ANGSTROM_TO_AU, 15)
    r2, g2 = rdf_oxygen(translated * ANGSTROM_TO_AU, 15)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(g1, g2)


def test_numerical_reference_evidence_has_distinct_precision_and_physics_claims():
    path = Path(__file__).resolve().parents[1] / "docs/validation/literature-audit-2026-11-01/water/numerics/report.json"
    evidence = json.loads(path.read_text())
    assert evidence["cluster_oracle"]["passed"]
    assert evidence["reference"]["method"] == "direct Ewald"
    pme = {(row["platform"], row["tolerance"]): row for row in evidence["electrostatics"]}
    assert pme["Reference", 1e-7]["force_rms_error_hartree_per_bohr"] < pme["Reference", 1e-4]["force_rms_error_hartree_per_bohr"] / 100
    assert pme["CPU", 1e-5]["force_relative_rms_error"] < 1e-4
    assert evidence["dispersion_tail"]["force_max_error_hartree_per_bohr"] == 0
    assert evidence["whole_molecule_image"]["force_max_error_hartree_per_bohr"] < 1e-12
    assert evidence["author_fortran"]["intramolecular_source_parameters"]["force_max_error_hartree_per_bohr"] < 1e-12
    assert not evidence["physical_converged"] and not evidence["literature_reproduced"]


def test_model_defaults_match_the_independent_paper_parameter_transcription():
    from benchmarks.water.qtip4pf import PARAMETERS
    path = Path(__file__).resolve().parents[1] / "docs/validation/literature-audit-2026-11-01/water/source-audit.json"
    source = json.loads(path.read_text())
    assert PARAMETERS == source["force_field"]["parameters_from_printed_table_I"]
    assert source["force_field"]["constraints"] == []
    assert source["source_protocol"]["IR_method"] == "PA-CMD"
    assert source["source_protocol"]["diffusion_total_NVE_ps"] == 8000


@pytest.mark.parametrize("beads,isotope,value,error,table", [
    (1, "H2O", .199, .002, "IV"),
    (32, "H2O", .221, .001, "III"),
    (32, "D2O", .172, .001, "III"),
    (1, "D2O", None, None, None),
])
def test_literature_reference_does_not_substitute_quantum_D2O_for_classical(beads, isotope, value, error, table):
    reference = literature_diffusion_reference(beads, isotope)
    assert reference["diffusion_angstrom2_ps"] == value
    assert reference["standard_error_angstrom2_ps"] == error
    assert reference["table"] == table
    assert not reference["comparison_eligible"]
    assert bool(reference["unavailable_reason"]) == (value is None)
