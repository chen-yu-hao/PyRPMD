"""The dependency-free closed-loop pilot has independent analytic references."""
from pathlib import Path
import runpy

import numpy as np
import pytest

from pyrpmd.potentials.base import check_potential
from pyrpmd.potentials.lifecycle import CommitteePotential


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ml_lifecycle" / "closed_loop.py"


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(str(EXAMPLE))


def test_polynomial_and_reference_forces_match_independent_finite_differences(example):
    q = np.random.default_rng(19).normal(scale=0.8, size=(5, 1, 3))
    potentials = [example["DoubleWellReference"](), *example["initial_models"]()]
    for potential in potentials:
        assert check_potential(potential, q, epsilon=2e-6)["max_absolute_force_error"] < 3e-9
    exact_polynomial = example["PolynomialPotential"]([0.5, 0.0, -1.0, 0.0, 0.5]).evaluate(q)
    reference = potentials[0].evaluate(q)
    np.testing.assert_allclose(exact_polynomial.energy, reference.energy, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(exact_polynomial.forces, reference.forces, rtol=1e-14, atol=1e-14)


def test_sampler_consumes_only_explicit_rng_and_state(example):
    sampler = example["MetropolisSampler"](walkers=16, steps=8)
    committee = CommitteePotential(example["initial_models"]())
    first_rng, second_rng = np.random.default_rng(8), np.random.default_rng(8)
    first = sampler(committee, first_rng, None)
    second = sampler(committee, second_rng, None)
    np.testing.assert_array_equal(first.positions, second.positions)
    assert first_rng.bit_generator.state == second_rng.bit_generator.state
    np.testing.assert_array_equal(first.state["walkers"], second.state["walkers"])
    first.positions[:] = 100.0
    assert np.max(np.abs(first.state["walkers"])) <= sampler.boundary


def test_multiple_rounds_reduce_error_and_disagreement_on_sampled_distribution(example, tmp_path):
    report = example["run_demo"](tmp_path / "pilot")
    assert report["uncertainty_is_calibrated_error"] is False
    assert report["diagnostic_labels_enter_training"] is False
    assert report["diagnostic_reference_frames"] == 2*(1024+128)
    for comparison in report["true_error_vs_uncertainty"].values():
        before, after = comparison["before"], comparison["after"]
        assert after["force_vector_rmse_hartree_per_bohr"] < 0.1*before["force_vector_rmse_hartree_per_bohr"]
        assert after["force_disagreement_rms_hartree_per_bohr"] < 0.1*before["force_disagreement_rms_hartree_per_bohr"]
        assert after["energy_rmse_hartree"] < before["energy_rmse_hartree"]
        # Keep error and disagreement visibly different quantities.  This pilot
        # alone establishes no calibration guarantee for either distribution.
        assert not np.isclose(after["force_rmse_over_disagreement"], 1.0)
    resumed = example["make_loop"](tmp_path / "pilot", resume=True)
    assert len(resumed.dataset) >= 8  # Four labels per training round.
    assert len(resumed.dataset) <= 32
    assert (tmp_path / "pilot" / "diagnostics.json").is_file()


def test_example_resume_mid_round_matches_uninterrupted_models_and_walkers(example, tmp_path):
    full = example["make_loop"](tmp_path / "full")
    full.run()
    partial = example["make_loop"](tmp_path / "resume")
    partial.run(max_steps=2)
    resumed = example["make_loop"](tmp_path / "resume", resume=True)
    resumed.run()
    assert resumed.dataset.sha256 == full.dataset.sha256
    assert resumed.result == full.result
    with resumed.committee as resumed_committee, full.committee as full_committee:
        for left, right in zip(resumed_committee.members, full_committee.members):
            np.testing.assert_array_equal(left.coefficients, right.coefficients)
    np.testing.assert_array_equal(resumed.sampler_state["walkers"], full.sampler_state["walkers"])
    assert resumed.sampler_state["accepted"] == full.sampler_state["accepted"]
    assert resumed.sampler_state["attempted"] == full.sampler_state["attempted"]
