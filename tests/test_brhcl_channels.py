"""Element bookkeeping and conservative BrHCl multichannel diagnostics."""

import json

import numpy as np
import pytest

from benchmarks.brhcl.channels import (
    BondThresholds, centroid_distances, classify_centroid, classify_centroids,
    minimax_path, select_channel_basins,
)
from benchmarks.hho2.channels import minimax_path as shared_minimax_path
from pyrpmd.constants import BOHR_ANGSTROM


# All geometries are in angstrom and fixed Br,H,Cl order.
GEOMETRIES = {
    "H+BrCl": np.array([[0., 0., 0.], [0., 5., 0.], [2.136, 0., 0.]]),
    "HCl+Br": np.array([[5., 0., 0.], [0., 0., 0.], [1.275, 0., 0.]]),
    "HBr+Cl": np.array([[0., 0., 0.], [1.414, 0., 0.], [5., 0., 0.]]),
}


@pytest.mark.parametrize("channel", GEOMETRIES)
def test_three_distinct_chemical_fragment_graphs(channel):
    result = classify_centroid(GEOMETRIES[channel], units="angstrom")
    assert result["channel"] == channel
    assert result["atom_order"] == ["Br", "H", "Cl"]
    assert len(result["distances_angstrom"]) == 3
    assert sum(state == "formed" for state in result["bond_states"].values()) == 1
    assert not result["ambiguous_bonds"]
    assert not result["overlapping_pairs"]
    assert sorted(map(len, result["components"])) == [1, 2]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("channel", GEOMETRIES)
def test_channel_invariant_to_rigid_motion_and_unit_conversion(channel):
    q = GEOMETRIES[channel]
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    transformed = (q @ rotation + [2.5, -3.1, .8]) / BOHR_ANGSTROM
    result = classify_centroid(transformed)
    assert result["channel"] == channel
    np.testing.assert_allclose(
        list(result["distances_angstrom"].values()),
        list(centroid_distances(q, units="angstrom").values()),
    )


def test_element_identity_is_not_replaced_by_sorting_pair_distances():
    # Moving the unlike halogen atoms changes which HX molecule exists.
    original = GEOMETRIES["HCl+Br"]
    assert classify_centroid(original, units="angstrom")["channel"] == "HCl+Br"
    swapped = original[[2, 1, 0]]
    assert classify_centroid(swapped, units="angstrom")["channel"] == "HBr+Cl"
    assert centroid_distances(original, units="angstrom")["H-Cl"] == pytest.approx(1.275)
    assert centroid_distances(swapped, units="angstrom")["Br-H"] == pytest.approx(1.275)


def test_distance_of_centroids_is_not_mean_bead_distance():
    q = GEOMETRIES["HBr+Cl"]
    beads = np.repeat(q[None], 2, axis=0)
    beads[:, 0, 1] = [2., -2.]
    beads[:, 1, 1] = [-2., 2.]
    assert np.linalg.norm(beads[:, 0] - beads[:, 1], axis=1).mean() > 4
    assert classify_centroid(beads[0], units="angstrom")["channel"] == "intermediate/ambiguous"
    result = classify_centroid(beads, units="angstrom")
    assert result["channel"] == "HBr+Cl"
    assert result["distances_angstrom"]["Br-H"] == pytest.approx(1.414)


@pytest.mark.parametrize("kind,channel,pair", [
    ("hbr", "HBr+Cl", (0, 1)),
    ("hcl", "HCl+Br", (1, 2)),
    ("brcl", "H+BrCl", (0, 2)),
])
def test_gray_zone_and_threshold_sensitivity(kind, channel, pair):
    thresholds = BondThresholds()
    formed, broken = getattr(thresholds, kind + "_formed"), getattr(thresholds, kind + "_broken")
    q = np.zeros((3, 3))
    q[list(set(range(3)) - set(pair))[0]] = [0, 8., 0]
    q[pair[1], 0] = (formed + broken) / 2
    result = classify_centroid(q, units="angstrom")
    assert result["channel"] == "intermediate/ambiguous"
    assert result["reason"] == "bond_threshold_gray_zone"
    assert len(result["ambiguous_bonds"]) == 1
    adjusted = {kind + "_formed": q[pair[1], 0] + .01}
    assert classify_centroid(q, units="angstrom", thresholds=adjusted)["channel"] == channel
    q[pair[1], 0] = formed
    assert classify_centroid(q, units="angstrom")["channel"] == channel
    q[pair[1], 0] = broken
    result = classify_centroid(q, units="angstrom")
    assert result["channel"] == "intermediate/ambiguous"
    assert not result["ambiguous_bonds"]


@pytest.mark.parametrize("q,reason", [
    (np.array([[0., 0, 0], [.1, 0, 0], [5., 0, 0]]), "overlapping_atoms"),
    (np.array([[0., 0, 0], [5., 0, 0], [0., 5., 0]]), "unrecognized_connectivity"),
    # Two formed HX bonds are a complex, not either separated product.
    (np.array([[-1.75, 0, 0], [0., 0, 0], [1.65, 0, 0]]), "unrecognized_connectivity"),
])
def test_overlap_dissociation_and_overcoordinated_geometries_are_not_products(q, reason):
    result = classify_centroid(q, units="angstrom")
    assert result["channel"] == "intermediate/ambiguous"
    assert result["reason"] == reason


@pytest.mark.parametrize("q", [
    np.zeros((4, 3)), np.zeros((0, 3, 3)), np.zeros((3,)),
    np.full((3, 3), np.nan), np.full((3, 3), np.inf),
    np.array([[1e308, 0, 0], [-1e308, 0, 0], [0, 0, 0]]),
])
def test_invalid_geometry_rejected(q):
    with pytest.raises(ValueError):
        classify_centroid(q)


@pytest.mark.parametrize("options", [
    {"hbr_formed": 0}, {"hcl_broken": np.inf}, {"hbr_formed": 2.5},
    {"minimum_distance": .0}, {"minimum_distance": 1.7}, {"brcl_broken": True},
])
def test_invalid_thresholds_rejected(options):
    with pytest.raises(ValueError):
        BondThresholds(**options)


def test_invalid_units_and_threshold_container_rejected():
    with pytest.raises(ValueError, match="units"):
        centroid_distances(GEOMETRIES["H+BrCl"], units="nm")
    with pytest.raises(TypeError, match="thresholds"):
        classify_centroid(GEOMETRIES["H+BrCl"], thresholds=[1, 2])


@pytest.mark.parametrize("units", ["angstrom", "bohr"])
@pytest.mark.parametrize("thresholds", [None, {"hbr_formed": 1.9, "brcl_broken": 3.0}])
def test_vectorized_labels_match_scalar_geometry_diagnostics(units, thresholds):
    rng = np.random.default_rng(8817)
    frames = np.concatenate((np.stack(list(GEOMETRIES.values())), rng.normal(size=(240, 3, 3)) * 3))
    # Include overlap, a threshold gray zone, and exact formed/broken bounds.
    boundaries = np.array([
        [[0, 0, 0], [.1, 0, 0], [8, 0, 0]],
        [[0, 0, 0], [2., 0, 0], [8, 0, 0]],
        [[0, 0, 0], [1.8, 0, 0], [8, 0, 0]],
        [[0, 0, 0], [2.35, 0, 0], [8, 0, 0]],
    ])
    frames = np.concatenate((frames, boundaries))
    if units == "bohr":
        frames = frames / BOHR_ANGSTROM
    original = frames.copy()
    expected = [classify_centroid(q, units=units, thresholds=thresholds)["channel"] for q in frames]
    actual = classify_centroids(frames, units=units, thresholds=thresholds)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(frames, original)


def test_batch_axis_cannot_be_mistaken_for_beads():
    frames = np.stack(list(GEOMETRIES.values()))
    np.testing.assert_array_equal(classify_centroids(frames, units="angstrom"), list(GEOMETRIES))
    assert classify_centroids(np.empty((0, 3, 3))).shape == (0,)
    for invalid in (np.zeros((3, 3)), np.zeros((2, 8, 3, 3)), np.full((2, 3, 3), np.nan)):
        with pytest.raises(ValueError, match="q_batch"):
            classify_centroids(invalid)


def test_distinct_basin_selection_retains_mask_and_does_not_merge_products():
    names = ["H+BrCl", "HCl+Br", "HBr+Cl", "intermediate/ambiguous"]
    fractions = np.array([
        [[.90, .05], [.99, .20]],
        [[.03, .90], [.00, .10]],
        [[.02, .02], [.00, .10]],
        [[.05, .03], [.01, .60]],
    ])
    free = np.array([[3., 2.], [-50., 0.]])
    supported = np.array([[True, True], [False, True]])
    original = fractions.copy()
    basins = select_channel_basins(names, fractions, free, supported)
    assert basins == {"H+BrCl": (0, 0), "HCl+Br": (0, 1)}
    assert select_channel_basins(names, fractions, free + 21, supported) == basins
    np.testing.assert_array_equal(fractions, original)
    # Unlike the H0/H1 entrance identities of H+HO2, different HX products
    # are never combined to turn a mixed projected bin into a pure basin.
    mixed = np.array([[[.1]], [[.45]], [[.45]]])
    assert select_channel_basins(names[:3], mixed, [[0.]], [[True]]) == {}


def test_basin_selection_excludes_nonfinite_grid_and_preserves_unknown_support():
    fraction = np.array([[[.9, np.nan], [.9, np.nan]]])
    free = np.array([[1., 2.], [np.nan, -3.]])
    support = np.array([[True, False], [True, False]])
    assert select_channel_basins(["H+BrCl"], fraction, free, support) == {"H+BrCl": (0, 0)}
    assert select_channel_basins(["H+BrCl"], fraction, free, np.zeros((2, 2), bool)) == {}


@pytest.mark.parametrize("names,fractions,minimum", [
    (["H+BrCl", "H+BrCl"], np.ones((2, 1, 1)) * .5, .8),
    (["unknown"], np.ones((1, 1, 1)), .8),
    (["H+BrCl"], np.array([[[np.nan]]]), .8),
    (["H+BrCl"], np.array([[[1.1]]]), .8),
    (["H+BrCl", "HCl+Br"], np.ones((2, 1, 1)) * .8, .8),
    (["H+BrCl"], np.ones((1, 1, 1)), 0),
])
def test_invalid_basin_probability_inputs_rejected(names, fractions, minimum):
    with pytest.raises(ValueError):
        select_channel_basins(names, fractions, [[0.]], [[True]], minimum_fraction=minimum)


def test_path_reuses_the_masked_four_neighbor_implementation():
    assert minimax_path is shared_minimax_path
    result = minimax_path(np.array([[0., np.nan], [np.nan, 0.]]), (0, 0), (1, 1))
    assert not result["connected"]
    assert result["path"] is None
