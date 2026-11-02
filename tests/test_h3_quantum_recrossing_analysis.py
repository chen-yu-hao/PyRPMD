"""Uncertainty checks that detect false independence and loose equivalence."""
import copy
import numpy as np
import pytest

from benchmarks.h3.quantum_recrossing_analyze import (
    equivalent_paired_difference, interval, paired_rate_statistics, validate_reactant_reference,
)


def test_same_chain_covariance_is_retained_in_full_rate():
    variation = np.array([-.03, -.02, -.01, -.005, .005, .01, .02, .03])
    # Exactly cancelling partition and transmission variation must cancel
    # in each chain's full rate, rather than being added as independent noise.
    result = paired_rate_statistics(-2. + variation, .77*np.exp(-variation), -7.27, 1.55e-18)
    expected = 1.55e-18*np.exp(5.27)*.77
    assert result["rate_geometric_mean_cm3_molecule_s"] == pytest.approx(expected, rel=2e-14)
    assert result["log_rate_statistics"]["halfwidth95"] < 1e-13
    assert result["incorrect_independence_log_rate_variance_for_comparison_only"] > 1e-5
    covariance = np.asarray(result["same_chain_log_partition_log_kappa_covariance_of_mean"])
    assert covariance[0, 1] < 0


def test_joint_seed_permutation_keeps_interval_but_mispairing_does_not():
    z = np.array([-.03, -.02, -.01, -.005, .005, .01, .02, .03])
    k = .77*np.exp(-z/2)
    original = paired_rate_statistics(-2+z, k, -7.27, 1.55e-18)
    perm = [3, 7, 2, 5, 0, 1, 6, 4]
    reordered = paired_rate_statistics((-2+z)[perm], k[perm], -7.27, 1.55e-18)
    wrong_pairs = paired_rate_statistics(-2+z, k[::-1], -7.27, 1.55e-18)
    np.testing.assert_allclose(original["rate_CI95_cm3_molecule_s"], reordered["rate_CI95_cm3_molecule_s"], rtol=2e-14, atol=0)
    assert wrong_pairs["log_rate_statistics"]["halfwidth95"] > 2*original["log_rate_statistics"]["halfwidth95"]


def test_rate_unit_change_only_rescales_the_interval():
    z = np.linspace(-.03, .03, 8)
    first = paired_rate_statistics(-2+z, .77+z/3, -7.27, 1.55e-18)
    scaled = paired_rate_statistics(-2+z, .77+z/3, -7.27, 1.55)
    np.testing.assert_allclose(np.asarray(first["rate_CI95_cm3_molecule_s"])*1e18,
                               scaled["rate_CI95_cm3_molecule_s"], rtol=3e-14)


def test_large_error_bar_cannot_establish_timestep_equivalence():
    first = np.array([.6, .8, .6, .8, .6, .8, .6, .8])
    second = np.full(8, .7)
    result = equivalent_paired_difference(first, second, .01)
    assert abs(result["mean"]) < 1e-15
    assert not result["CI95_wholly_within_margin"]
    exact_pair = equivalent_paired_difference(first, first, .01)
    assert exact_pair["CI95_wholly_within_margin"]


def test_single_chain_has_no_independent_chain_interval():
    with pytest.raises(ValueError, match="two"):
        interval([.77])


def test_nonpositive_transmission_is_not_silently_dropped():
    with pytest.raises(ValueError, match="positive"):
        paired_rate_statistics(np.full(8, -2.), [.7, .8, .7, 0., .7, .8, .7, .8], -7.27, 1.55e-18)


@pytest.mark.parametrize("field,new_value", [("temperature_K", 301.), ("reduced_mass_au", 2.),
                                             ("primary_beads", 64), ("primary_log_quantum_classical_ratio", -7.)])
def test_reference_preflight_rejects_other_hamiltonian(field, new_value):
    protocol = {"beads": 128, "temperature_K": 300., "mass_au": 2., "beta": 1024.,
                "static_domain_minimum_pair_bohr": .5, "model": {"library_sha256": "registered"}}
    parent_protocol = {"maximum_centroid_bond_bohr": 8.}
    reference = {"numerical_converged": True, "primary_beads": 128, "temperature_K": 300.,
                 "reduced_mass_au": 1., "primary_log_quantum_classical_ratio": -7.2,
                 "provenance": {"pes": protocol["model"]},
                 "bead_refinement": [{"beads": 128, "free_kernel_width_bohr": np.sqrt(8.),
                                       "domain_bohr": [.5, 8.], "log_quantum_classical_ratio": -7.2}]}
    validate_reactant_reference(reference, protocol, parent_protocol)
    changed = copy.deepcopy(reference)
    changed[field] = new_value
    with pytest.raises(ValueError, match="match"):
        validate_reactant_reference(changed, protocol, parent_protocol)
