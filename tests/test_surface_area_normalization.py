"""Surface concentration normalization follows the periodic lattice plane."""

from itertools import permutations

import numpy as np
import pytest

from pyrpmd.analysis.surface import normalize_surface_rates, surface_normalization
from pyrpmd.constants import BOHR_M
from pyrpmd.system import System


def _system(cell, pbc):
    return System(("H",), np.array([1837.0]), np.zeros((1, 3)),
                  cell=cell, pbc=pbc)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_area_follows_periodic_vectors_under_cell_row_permutation(order):
    # The third row is vacuum and may have an in-plane component. Neither
    # its length nor its index defines the area of the physical surface.
    cell = np.array([[4., 0., 0.], [1., 3., 0.], [2., -1., 25.]])
    pbc = np.array([True, True, False])
    reordered = cell[list(order)]
    flags = pbc[list(order)]
    options = surface_normalization(
        {"reaction_order": 2, "reference_adsorbates": 2,
         "coverage": .2, "site_count": 10}, _system(reordered, flags))
    assert options["area_cm2"] == pytest.approx(12. * (BOHR_M * 100.)**2, rel=1e-12, abs=0.)
    assert options["area_source"] == "periodic_cell_vectors"
    assert options["area_lattice_vectors"] == np.flatnonzero(flags).tolist()
    normalized = normalize_surface_rates(
        {"rate_s_inv": None, "rate_estimate_s_inv": 50.,
         "rate_standard_error_s_inv": 2.,
         "rate_percentile_95_ci_s_inv": [46., 54.]}, options)
    assert normalized["at_coverage"]["rate_estimate"] == pytest.approx(50.)
    assert normalized["at_coverage"]["rate_standard_error"] == pytest.approx(2.)
    assert normalized["at_coverage"]["rate"] is None
    assert normalized["settings"] == options


def test_area_is_rotation_and_vacuum_invariant():
    cell = np.array([[4., 0., 0.], [1., 3., 0.], [0., 0., 20.]])
    rotation, _ = np.linalg.qr(np.array([[1., 2., 3.], [2., 5., 1.], [4., 1., 2.]]))
    settings = {"reaction_order": 2, "reference_adsorbates": 1}
    original = surface_normalization(settings, _system(cell, [True, True, False]))
    cell[2] *= 3
    rotated = surface_normalization(settings, _system(cell @ rotation, [True, True, False]))
    assert rotated["area_cm2"] == pytest.approx(original["area_cm2"], rel=1e-12, abs=0.)


@pytest.mark.parametrize("pbc", [[True, True, True], [True, False, False],
                                [False, False, False]])
def test_ambiguous_surface_plane_requires_explicit_area(pbc):
    with pytest.raises(ValueError, match="explicit area.*exactly two periodic"):
        surface_normalization({"reaction_order": 2, "reference_adsorbates": 1},
                              _system(np.diag([4., 3., 20.]), pbc))


@pytest.mark.parametrize("cell,pbc", [(None, None),
                                    (np.diag([4., 3., 20.]), [True, True, True])])
def test_explicit_area_is_available_for_clusters_and_3d_periodic_slabs(cell, pbc):
    options = surface_normalization(
        {"reaction_order": 2, "reference_adsorbates": 1, "area": "7 angstrom^2"},
        _system(cell, pbc))
    assert options["area_cm2"] == pytest.approx(7e-16, rel=1e-12, abs=0.)
    assert options["area_source"] == "explicit"
    assert options["area_lattice_vectors"] is None


def test_first_order_rate_does_not_require_a_surface_plane():
    assert surface_normalization({}, _system(None, None)) == {
        "reaction_order": 1, "path_multiplicity": 1}
