"""Independent analytic oracles and failure-sensitive tutorial comparisons."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from examples.tutorials import reference_checks as refs
from pyrpmd.input import JobSpec

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["harmonic_pimd", "periodic_site_rate", "double_well_metad", "free_diffusion", "harmonic_spectroscopy"]


@pytest.mark.parametrize("name", NAMES)
def test_tutorial_inputs_resolve_on_one_core_with_real_reference(name):
    config, system = JobSpec.from_file(ROOT / "examples/tutorials" / name / "input.inp").resolve()
    assert config["parallel"]["backend"] == "serial"
    assert config["parallel"]["threads_per_rank"] == 1
    assert system.masses[0] == pytest.approx(refs.MASS)
    assert config["sampling"]["replicas"] >= 4
    assert (ROOT / "examples/tutorials" / name / "compare.py").is_file()


def test_finite_bead_harmonic_reference_matches_independent_dense_covariance():
    beads, beta, k, mass = 7, 15., .2, 3.
    laplacian = 2*np.eye(beads) - np.roll(np.eye(beads), 1, axis=0) - np.roll(np.eye(beads), -1, axis=0)
    covariance = beads / beta * np.linalg.inv(k*np.eye(beads) + mass*(beads/beta)**2*laplacian)
    energy = 1.5*k*np.trace(covariance)/beads
    reference = refs.harmonic_reference(k, mass, beta, beads)
    assert reference["potential_energy"] == pytest.approx(energy, rel=1e-13)
    assert reference["kinetic_centroid_virial"] == pytest.approx(energy, rel=1e-13)
    assert refs.harmonic_reference(k, mass, beta, 1)["radius_of_gyration_squared_bohr2"] == pytest.approx(0., abs=1e-14)


def test_periodic_tst_reference_matches_direct_partition_integral():
    h, length, mass, beta = .006, 4., refs.MASS, refs.BETA
    partition = quad(lambda x: np.exp(-beta*h*.5*(1-np.cos(2*np.pi*x/length))), -length/2, length/2)[0]
    expected = np.exp(-beta*h)/(partition*np.sqrt(2*np.pi*beta*mass)*refs.ATOMIC_TIME_S)
    assert refs.site_rate_reference() == pytest.approx(expected, rel=1e-12)


def test_diffusion_and_frequency_references_have_correct_mass_scaling():
    assert refs.diffusion_reference(mass=2*refs.MASS) == refs.diffusion_reference()/2
    assert refs.spectroscopy_reference(mass=4*refs.MASS) == refs.spectroscopy_reference()/2
    assert refs.diffusion_reference() == pytest.approx(.00106893953, rel=1e-8)
    assert refs.spectroscopy_reference() == pytest.approx(1144.976858, rel=1e-9)


@pytest.mark.parametrize("value,error", [(1.4,.1), (None,.1), (1.,None), (1.,.5), (1.,float('nan'))])
def test_three_sem_gate_rejects_disagreement_missing_or_uninformative_errors(value, error):
    assert not refs.check("x", value, 1., error, max_relative_error=.15)["passed"]


def test_resolution_bound_is_separate_and_does_not_hide_arbitrary_discrepancy():
    result = refs.replica_check("peak", [1145.,1145.,1145.,1145.], 1144., resolution=2.)
    assert result["standard_error"] == 0. and result["resolution_bound"] == 2.
    assert result["deviation_in_standard_errors"] is None and result["passed"]
    assert not refs.replica_check("peak", [1150.]*4, 1144., resolution=2.)["passed"]


def test_failed_site_rate_is_never_replaced_by_positive_candidate(tmp_path):
    (tmp_path / "results").mkdir()
    summary = {"rate_s": None, "rate_estimate_s": refs.site_rate_reference(),
               "rate_standard_error_s": 1e8, "plateau_detected": False,
               "rate_uncertainty": {"available": True}}
    source = tmp_path / "results/summary.json"
    source.write_text(json.dumps(summary))
    before = source.read_bytes()
    assert not refs.compare("periodic_site_rate", tmp_path)["passed"]
    assert source.read_bytes() == before


@pytest.mark.parametrize("relative,method", [("metadynamics/standard.inp","metadynamics"),
    ("metadynamics/well_tempered_2d.inp","metadynamics"), ("metadynamics/grid_fixed_slab.inp","metadynamics"),
    ("opes/harmonic_2d.inp","opes"), ("opes/adaptive_explore.inp","opes")])
def test_five_smoke_inputs_use_current_fes_spelling(relative, method):
    spec = JobSpec.from_file(ROOT / "examples" / relative)
    assert spec.data["task"] == "fes" and spec.data["fes"]["method"] == method
    spec.resolve()


@pytest.mark.parametrize("name", NAMES)
def test_comparison_reads_actual_short_workflow_output_schema(tmp_path, name):
    from pyrpmd.runner import run

    spec = JobSpec.from_file(ROOT / "examples/tutorials" / name / "input.inp")
    sampling = spec.data["sampling"]
    for key in ("equilibration", "production", "interval"):
        sampling.pop(key, None)
    sampling.update(equilibration_steps=16, production_steps=64, sample_interval=2, replicas=2)
    spec.data["output"] = {"checkpoint_steps": 16}
    if name == "double_well_metad":
        spec.data["metadynamics"].update(pace=2, freeze_after_steps=8, freeze_equilibration_steps=4)
    if name == "periodic_site_rate":
        spec.data["analysis"].update(bootstrap=0, bins=12, min_count=1)
        spec.data["recrossing"] = {"parent_replicas": 2, "parent_samples": 2,
            "equilibration_steps": 2, "parent_interval_steps": 2, "child_pairs": 1,
            "child_steps": 4, "bootstrap": 0}
    if name == "free_diffusion":
        spec.data["diffusion"].update(max_lag=8, fit_range=["2 fs", "6 fs"], bootstrap=0)
    run(spec, output=tmp_path / name)
    if name == "double_well_metad":
        with pytest.raises(ValueError, match="lacks support"):
            refs.compare(name, tmp_path / name)
    else:
        result = refs.compare(name, tmp_path / name)
        assert isinstance(result["passed"], bool)
        json.dumps(result, allow_nan=False)
