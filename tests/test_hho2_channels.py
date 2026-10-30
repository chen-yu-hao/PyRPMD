"""Independent geometry and supported-grid checks for the H + HO2 example."""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.constants import BOHR_ANGSTROM
from pyrpmd.potentials.python import import_object


MODULE = str(Path(__file__).resolve().parents[1] / "benchmarks/hho2/channels.py")
BondThresholds = import_object(MODULE + ":BondThresholds")
centroid_distances = import_object(MODULE + ":centroid_distances")
classify_centroid = import_object(MODULE + ":classify_centroid")
minimax_path = import_object(MODULE + ":minimax_path")

# Explicitly separated fragments; atom order H0,H1,O0,O1, coordinates in A.
GEOMETRIES = {
    "reactant_H0+H1O2": np.array([[6, 0, 0], [-0.31, 0.9, 0], [0, 0, 0], [1.32, 0, 0]]),
    "H2+O2": np.array([[0, 0, 0], [0.74, 0, 0], [5, 0, 0], [6.21, 0, 0]]),
    "OH+OH": np.array([[-0.96, 0, 0], [4.96, 0, 0], [0, 0, 0], [4, 0, 0]]),
    "H2O+O": np.array([[0.587, 0.757, 0], [0.587, -0.757, 0], [0, 0, 0], [5, 0, 0]]),
    "intermediate/ambiguous": np.array([[-0.96, 0, 0], [2.41, 0, 0], [0, 0, 0], [1.45, 0, 0]]),
}
GEOMETRIES["exchanged_H1+H0O2"] = GEOMETRIES["reactant_H0+H1O2"][[1, 0, 2, 3]]


@pytest.mark.parametrize("channel", list(GEOMETRIES))
def test_separated_fragments_and_peroxide_intermediate(channel):
    result = classify_centroid(GEOMETRIES[channel], units="angstrom")
    assert result["channel"] == channel
    assert len(result["distances_angstrom"]) == 6
    assert not result["ambiguous_bonds"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("channel", list(GEOMETRIES))
def test_rigid_motion_units_and_oxygen_permutation(channel):
    q = GEOMETRIES[channel]
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    moved = (q @ rotation + [1.7, -4.2, 2.3]) / BOHR_ANGSTROM
    assert classify_centroid(moved)["channel"] == channel
    assert classify_centroid(q[[0, 1, 3, 2]], units="angstrom")["channel"] == channel


@pytest.mark.parametrize("channel", list(GEOMETRIES))
def test_hydrogen_permutation_retains_exchange_identity(channel):
    expected = {
        "reactant_H0+H1O2": "exchanged_H1+H0O2",
        "exchanged_H1+H0O2": "reactant_H0+H1O2",
    }.get(channel, channel)
    q = GEOMETRIES[channel][[1, 0, 2, 3]]
    assert classify_centroid(q, units="angstrom")["channel"] == expected


def test_uses_centroid_distance_not_average_bead_distance():
    q = GEOMETRIES["OH+OH"]
    beads = np.repeat(q[None, :, :], 2, axis=0)
    beads[:, 0, 1] = [1.2, -1.2]
    # Each bead's first OH bond is in the gray zone, while its centroid is OH.
    assert classify_centroid(beads[0], units="angstrom")["channel"] == "intermediate/ambiguous"
    result = classify_centroid(beads, units="angstrom")
    assert result["channel"] == "OH+OH"
    assert result["distances_angstrom"]["H0-O0"] == pytest.approx(0.96)
    np.testing.assert_allclose(
        list(centroid_distances(beads / BOHR_ANGSTROM).values()),
        list(centroid_distances(q, units="angstrom").values()),
    )


def test_gray_zone_prevents_product_label_and_thresholds_are_configurable():
    q = GEOMETRIES["H2+O2"].copy()
    q[1, 0] = 1.2
    result = classify_centroid(q, units="angstrom")
    assert result["channel"] == "intermediate/ambiguous"
    assert result["ambiguous_bonds"] == ["H0-H1"]
    assert result["reason"] == "bond_threshold_gray_zone"
    assert classify_centroid(q, units="angstrom", thresholds={"hh_formed": 1.21})["channel"] == "H2+O2"
    # Exact boundaries are deliberately deterministic: formed <=, broken >=.
    q[1, 0] = 1.0
    assert classify_centroid(q, units="angstrom")["channel"] == "H2+O2"
    q[1, 0] = 1.4
    result = classify_centroid(q, units="angstrom")
    assert result["bond_states"]["H0-H1"] == "broken"
    assert result["channel"] == "intermediate/ambiguous"


def test_close_contact_does_not_masquerade_as_two_oh_products():
    # Two OH bonds plus a definite OO bond is a connected peroxide complex.
    result = classify_centroid(GEOMETRIES["intermediate/ambiguous"], units="angstrom")
    assert result["channel"] == "intermediate/ambiguous"
    assert result["components"] == [[0, 1, 2, 3]]


@pytest.mark.parametrize("positions", [np.zeros((3, 3)), np.zeros((0, 4, 3)), np.full((4, 3), np.nan)])
def test_bad_geometry_is_rejected(positions):
    with pytest.raises(ValueError):
        classify_centroid(positions)


@pytest.mark.parametrize("kwargs", [{"hh_formed": 1.4}, {"oo_broken": np.inf}, {"oh_formed": -1.0}, {"hh_formed": True}])
def test_invalid_bond_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError, match="thresholds"):
        BondThresholds(**kwargs)


def test_unknown_position_units_are_rejected():
    with pytest.raises(ValueError, match="units"):
        centroid_distances(GEOMETRIES["H2+O2"], units="nm")


def test_minimax_chooses_lower_detour_instead_of_short_high_barrier():
    free = np.array([[0.0, 9.0, 1.0], [2.0, 3.0, 2.0]])
    result = minimax_path(free, (0, 0), (0, 2))
    assert result["connected"]
    assert result["path"] == [[0, 0], [1, 0], [1, 1], [1, 2], [0, 2]]
    assert result["bottleneck"] == 3.0
    assert result["barrier_from_start"] == 3.0
    assert result["barrier_from_goal"] == 2.0
    shifted = minimax_path(free - 50, (0, 0), (0, 2))
    assert shifted["path"] == result["path"]
    assert shifted["bottleneck"] == -47
    assert shifted["barrier_from_start"] == result["barrier_from_start"]


@pytest.mark.parametrize("blocked", [np.nan, np.inf, -np.inf])
def test_nonfinite_strip_disconnects_grid(blocked):
    free = np.array([[0, blocked, 0], [0, blocked, 0]])
    result = minimax_path(free, (0, 0), (0, 2))
    assert not result["connected"]
    assert result["path"] is None
    assert result["bottleneck"] is None
    json.dumps(result, allow_nan=False)


def test_mask_and_masked_array_prevent_diagonal_crossing_of_holes():
    mask = np.array([[False, True], [True, False]])
    free = np.zeros((2, 2))
    assert not minimax_path(free, (0, 0), (1, 1), mask=mask)["connected"]
    assert not minimax_path(np.ma.array(free, mask=mask), (0, 0), (1, 1))["connected"]
    # A masked low-cost detour must not lower the visible barrier.
    free = np.array([[0.0, 7.0, 0.0], [0.0, 0.0, 0.0]])
    mask = np.array([[False, False, False], [False, True, False]])
    assert minimax_path(free, (0, 0), (0, 2), mask=mask)["bottleneck"] == 7.0


def test_endpoint_support_and_identity_path():
    free = np.array([[-4.0, 0.0], [1.0, np.nan]])
    assert minimax_path(free, (0, 0), (1, 1))["reason"] == "unsupported_endpoint"
    result = minimax_path(free, (0, 0), (0, 0))
    assert result["path"] == [[0, 0]]
    assert result["bottleneck"] == -4.0
    assert result["barrier_from_start"] == 0.0


@pytest.mark.parametrize("start,goal", [((0.1, 0), (1, 1)), ((-1, 0), (1, 1)), ((0, 0), (2, 1)), ((True, 0), (1, 1))])
def test_bad_grid_endpoints_are_rejected(start, goal):
    with pytest.raises(ValueError):
        minimax_path(np.zeros((2, 2)), start, goal)


def test_bad_grid_and_mask_shapes_are_rejected():
    with pytest.raises(ValueError, match="two-dimensional"):
        minimax_path(np.zeros(4), (0, 0), (0, 1))
    with pytest.raises(ValueError, match="nonempty"):
        minimax_path(np.zeros((0, 2)), (0, 0), (0, 1))
    with pytest.raises(ValueError, match="same shape"):
        minimax_path(np.zeros((2, 2)), (0, 0), (0, 1), mask=[False, True])
