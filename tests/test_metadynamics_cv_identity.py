"""Do not average replica biases defined on different built-in coordinates."""

import numpy as np
import pytest

from pyrpmd.analysis.metadynamics import bias_fes
from pyrpmd.bias import MetadynamicsBias
from pyrpmd.cv import CVStack, PeriodicDistance, SurfaceHeight, SurfaceLateral
from pyrpmd.cv_periodic import (
    FractionalCoordinate, LinearProjection, PeriodicAngle, PeriodicBondDifference,
)


def _analyze(cv1, cv2):
    biases = []
    for cv in (cv1, cv2):
        bias = MetadynamicsBias(cv, [0.3] * cv.dimension, 0.4, 1.0)
        bias.deposit([0.5] * cv.dimension, step=1, time=0.1)
        biases.append(bias)
    values = np.linspace(0.1, 0.9, 5)
    samples = np.stack(np.meshgrid(*([values] * cv1.dimension), indexing="ij"), axis=-1)
    samples = samples.reshape(-1, cv1.dimension)
    return bias_fes(biases, [samples, samples], bins=5,
                    ranges=[[0.0, 1.0]] * cv1.dimension)


def _periodic(kind, change=None):
    cell = np.diag([6.0, 7.0, 8.0])
    pbc = [True, True, True]
    if change == "cell":
        cell[0, 0] = 9.0
    elif change == "pbc":
        pbc[0] = False
    elif change == "no_cell":
        cell, pbc = None, [False, False, False]
    if kind == "distance":
        atoms = (0, 2) if change == "atoms" else (0, 1)
        return PeriodicDistance(atoms, cell, pbc)
    if kind == "angle":
        atoms = (0, 2, 1) if change == "atoms" else (0, 1, 2)
        return PeriodicAngle(atoms, cell, pbc)
    bonds = [(0, 2), (1, 2)] if change == "atoms" else [(0, 1), (1, 2)]
    return PeriodicBondDifference(cell, pbc, bonds=bonds)


@pytest.mark.parametrize("kind", ["distance", "angle", "bond_difference"])
@pytest.mark.parametrize("change", ["atoms", "cell", "pbc"])
def test_periodic_replica_cvs_require_identical_geometry_mapping(kind, change):
    with pytest.raises(ValueError, match="temperature, method and CVs"):
        _analyze(CVStack([_periodic(kind)]), CVStack([_periodic(kind, change)]))


@pytest.mark.parametrize("kind", ["distance", "angle", "bond_difference"])
def test_nonperiodic_cvs_without_a_cell_remain_valid(kind):
    result = _analyze(CVStack([_periodic(kind, "no_cell")]),
                     CVStack([_periodic(kind, "no_cell")]))
    assert result["common_support"]
    assert np.isfinite(result["free_energy"]).all()


def _projection(kind, change=None):
    masses = [1.0, 2.0, 3.0]
    origin = 0.25 if change == "origin" else 0.0
    refs = [1, 2]
    if change == "masses":
        masses[1] = 4.0
    direction = [0, 1, 0] if change == "direction" else [1, 0, 0]
    if kind == "linear":
        return LinearProjection(0, direction, masses, reference_atoms=refs, origin=origin)
    if kind == "height":
        return SurfaceHeight([0], masses, normal=direction,
                             reference_atoms=refs, origin=origin)
    if kind == "lateral":
        return SurfaceLateral([0], masses, direction=direction,
                              reference_atoms=refs, origin=origin)
    cell = np.diag([6.0, 7.0, 8.0])
    if change == "cell":
        cell[0, 0] = 9.0
    return FractionalCoordinate(0, 1 if change == "axis" else 0, cell,
                                [change != "pbc", True, True], masses,
                                reference_atoms=refs, origin=origin)


@pytest.mark.parametrize("kind", ["linear", "height", "lateral"])
@pytest.mark.parametrize("change", ["direction", "origin", "masses"])
def test_projection_replica_cvs_require_matching_coefficients_and_origin(kind, change):
    with pytest.raises(ValueError, match="temperature, method and CVs"):
        _analyze(CVStack([_projection(kind)]), CVStack([_projection(kind, change)]))


@pytest.mark.parametrize("change", ["axis", "cell", "pbc", "origin", "masses"])
def test_fractional_replica_cvs_require_matching_cell_and_projection(change):
    with pytest.raises(ValueError, match="temperature, method and CVs"):
        _analyze(CVStack([_projection("fractional")]),
                 CVStack([_projection("fractional", change)]))


@pytest.mark.parametrize("factory,kind", [
    (_periodic, "distance"), (_periodic, "angle"), (_periodic, "bond_difference"),
    (_projection, "linear"), (_projection, "height"), (_projection, "lateral"),
    (_projection, "fractional"),
])
def test_independently_constructed_identical_builtin_cvs_can_be_averaged(factory, kind):
    result = _analyze(CVStack([factory(kind)]), CVStack([factory(kind)]))
    assert result["common_support"]
    assert not result["mask"].any()
    assert np.isfinite(result["free_energy"]).all()


def test_projection_identity_checks_follow_nested_stacks():
    cv1 = LinearProjection(0, [1, 0, 0], [1.0])
    cv2 = LinearProjection(0, [0, 1, 0], [1.0])
    q = np.array([[[1.0, 2.0, 3.0]]])
    assert cv1.value(q)[0] == 1.0
    assert cv2.value(q)[0] == 2.0
    left = CVStack([CVStack([cv1]), _periodic("distance")])
    right = CVStack([CVStack([cv2]), _periodic("distance")])
    with pytest.raises(ValueError, match="temperature, method and CVs"):
        _analyze(left, right)


def test_unused_masses_do_not_change_affine_coordinate_identity():
    # Without a reference group, only the atom and direction define this CV.
    left = LinearProjection(0, [1, 0, 0], [1.0, 2.0])
    right = LinearProjection(0, [1, 0, 0], [3.0, 4.0])
    result = _analyze(CVStack([left]), CVStack([right]))
    assert result["common_support"]


def test_custom_subclass_parameters_remain_caller_owned():
    class CustomProjection(LinearProjection):
        def _affine_gradient(self):
            raise AssertionError("Do not introspect a custom coordinate as a built-in")

    left = CustomProjection(0, [1, 0, 0], [1.0])
    right = CustomProjection(0, [0, 1, 0], [1.0])
    # Custom CV equivalence remains the caller's responsibility; analysis uses
    # its supplied CV-space samples and must not call arbitrary plugin methods.
    result = _analyze(CVStack([left]), CVStack([right]))
    assert result["common_support"]
