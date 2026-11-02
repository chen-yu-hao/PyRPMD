"""Matrix-free flux metrics: dense references, finite differences and native loops.

The dense contractions here deliberately retain the pre-HVP formula.  Finite
differences provide a separate reference, including periodic image branches.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import MethodType

import numpy as np
import pytest

from pyrpmd.bias import FluxMetricBias
from pyrpmd.cv import (
    Angle, BondDifference, Cartesian, COMDistance, Distance, PeriodicDistance,
    RPMDrateCoordinate, SurfaceHeight, SurfaceLateral,
)
from pyrpmd.cv_periodic import (
    FractionalCoordinate, LinearProjection, PeriodicAngle, PeriodicBondDifference,
)
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import CABIPotential


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = np.array([
    [-1.7, .3, -.1], [.1, -.2, .2], [1.8, .5, -.3],
    [2.9, 1.3, .7], [-2.4, 2., -.6], [.5, 3., 1.1],
])
MASSES = np.array([2.4, 1.7, 1.7, 3.2, 4.1, 2.3])
CELL = np.array([[8., 0., 0.], [1.2, 7., 0.], [-.4, .7, 9.]])
KINDS = (
    "distance", "periodic_distance", "difference", "periodic_difference",
    "com", "cartesian", "height", "lateral", "projection", "fractional",
    "rate", "surface", "angle", "periodic_angle",
)
DENSE_KINDS = KINDS[:-2]


def coordinate(reference=REFERENCE, masses=MASSES):
    return RPMDrateCoordinate(
        [[0], list(range(1, len(masses)))], masses, reference,
        [[0, 1], [0, 2]], [[1, 2], [2, 3]], 12.,
        equivalent_paths=[[0, 2, 1, *range(3, len(masses))]],
    )


def make_cv(kind):
    if kind == "distance":
        return Distance([0, 1])
    if kind == "periodic_distance":
        return PeriodicDistance([0, 1], CELL)
    if kind == "difference":
        return BondDifference(atoms=[0, 1, 2])
    if kind == "periodic_difference":
        return PeriodicBondDifference(CELL, (True, True, True), atoms=[0, 1, 2])
    if kind == "com":
        return COMDistance([[0, 4], [1, 2, 3, 5]], MASSES)
    if kind == "cartesian":
        return Cartesian(2, "y")
    if kind == "height":
        return SurfaceHeight([0, 4], MASSES, normal=[.2, .4, 1.], reference_atoms=[1, 2])
    if kind == "lateral":
        return SurfaceLateral([0, 4], MASSES, direction=[1., .4, -.2], reference_atoms=[1, 2])
    if kind == "projection":
        return LinearProjection(0, [1., .3, -.4], MASSES, reference_atoms=[1, 2])
    if kind == "fractional":
        return FractionalCoordinate(0, 1, CELL, (True, True, True), MASSES, reference_atoms=[1, 2])
    if kind == "rate":
        return coordinate()
    if kind == "surface":
        return coordinate().dividing_surface(.73)
    if kind == "angle":
        return Angle([0, 1, 2])
    if kind == "periodic_angle":
        return PeriodicAngle([0, 1, 2], CELL, (True, True, True))
    raise AssertionError(kind)


def geometry(seed, kind):
    q = REFERENCE + np.random.default_rng(seed).normal(0., .17, REFERENCE.shape)
    if kind.startswith("periodic_"):
        q[0] += CELL[0]
        q[2] -= CELL[1]
    return q


def dense_reference(cv, q):
    # BondDifference originally exposed only its gradient.  Assemble its
    # Hessian from the independently retained distance Hessians.
    if isinstance(cv, BondDifference):
        distances = [
            PeriodicDistance(pair, cv.cell, cv.pbc)
            if isinstance(cv, PeriodicBondDifference) else Distance(pair)
            for pair in cv.bonds
        ]
        a = distances[0].centroid_value_gradient_hessian(q)
        b = distances[1].centroid_value_gradient_hessian(q)
        return tuple(x - y for x, y in zip(a, b))
    return cv.centroid_value_gradient_hessian(q)


class DenseMetric:
    """Pre-change physical force, kept independent of FluxMetricBias."""

    def __init__(self, cv, masses, beta):
        self.cv, self.masses, self.beta = cv, np.asarray(masses), beta

    def energy_force(self, q):
        _, gradient, hessian = dense_reference(self.cv, np.mean(q, axis=0))
        z = np.sum(gradient * gradient / self.masses[:, None])
        energy = -len(q) * np.log(np.sqrt(z / (2 * np.pi * self.beta))) / self.beta
        force = np.einsum("acbd,bd->ac", hessian, gradient / self.masses[:, None]) / (self.beta * z)
        return energy, np.broadcast_to(force, q.shape).copy()


@pytest.mark.parametrize("kind", DENSE_KINDS)
@pytest.mark.parametrize("seed", [14, 37, 81])
def test_hvp_matches_retained_dense_hessian(kind, seed):
    cv, q = make_cv(kind), geometry(seed, kind)
    direction = np.random.default_rng(seed + 91).normal(size=q.shape)
    value, gradient, hessian = dense_reference(cv, q)
    actual, grad, hvp = cv.centroid_value_gradient_hvp(q, direction)
    np.testing.assert_allclose(actual, value, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(grad, gradient, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(hvp, np.einsum("acbd,bd->ac", hessian, direction), rtol=1e-12, atol=5e-14)
    assert hvp.shape == q.shape and hvp.dtype == np.float64


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("seed", [23, 65])
def test_hvp_matches_independent_gradient_finite_difference(kind, seed):
    cv, q = make_cv(kind), geometry(seed, kind)
    direction = np.random.default_rng(seed + 3).normal(size=q.shape)
    delta = 2e-6
    expected = (cv.centroid_value_gradient(q + delta * direction)[1]
                - cv.centroid_value_gradient(q - delta * direction)[1]) / (2 * delta)
    _, _, hvp = cv.centroid_value_gradient_hvp(q, direction)
    np.testing.assert_allclose(hvp, expected, rtol=3e-7, atol=3e-9)


@pytest.mark.parametrize("seed,path,bond", [(0, 0, 0), (1, 1, 0), (2, 0, 1), (12, 1, 1)])
@pytest.mark.parametrize("surface", [False, True])
def test_all_rate_minmax_branches_match_dense_and_finite_difference(seed, path, bond, surface):
    rate = coordinate()
    q = np.random.default_rng(seed).normal(0., 2., REFERENCE.shape)
    values = np.array([
        [r - np.linalg.norm(q[f[0]] - q[f[1]]) + np.linalg.norm(q[b[0]] - q[b[1]])
         for r, f, b in zip(rate.reference_difference, forming, breaking)]
        for forming, breaking in rate.paths
    ])
    assert int(np.argmin(values.max(axis=0))) == bond
    assert int(np.argmax(values[:, bond])) == path
    cv = rate.dividing_surface(.73) if surface else rate
    v = np.random.default_rng(913).normal(size=q.shape)
    expected = dense_reference(cv, q)
    actual = cv.centroid_value_gradient_hvp(q, v)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual[2], np.einsum("acbd,bd->ac", expected[2], v), rtol=1e-12, atol=5e-14)
    delta = 2e-6
    numeric = (cv.centroid_value_gradient(q + delta * v)[1]
               - cv.centroid_value_gradient(q - delta * v)[1]) / (2 * delta)
    np.testing.assert_allclose(actual[2], numeric, rtol=3e-7, atol=3e-9)


def test_rate_ties_singularities_and_unselected_degenerate_bonds():
    q = np.array([[0., 2., 0.], [-1., 0., 0.], [1., 0., 0.]])
    rate = RPMDrateCoordinate([[0], [1, 2]], np.ones(3), q,
                             [[0, 1], [0, 2]], [[1, 2], [1, 2]], 10.,
                             equivalent_paths=[[0, 2, 1]])
    v = np.random.default_rng(19).normal(size=q.shape)
    for cv in (rate, rate.dividing_surface(.7)):
        expected = dense_reference(cv, q)
        actual = cv.centroid_value_gradient_hvp(q, v)
        np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(actual[2], np.einsum("acbd,bd->ac", expected[2], v), rtol=1e-12, atol=1e-14)
    rate.rinf = rate.com.centroid_value_gradient(q)[0]
    with pytest.raises(ValueError, match="singular"):
        rate.centroid_value_gradient_hvp(q, v)
    assert np.isfinite(rate.dividing_surface(.5).centroid_value_gradient_hvp(q, v)[2]).all()
    cv = coordinate()
    invalid = REFERENCE.copy()
    invalid[3] = invalid[2]
    with pytest.raises(ValueError, match="coincident"):
        cv.centroid_value_gradient_hvp(invalid, np.ones_like(invalid))


@pytest.mark.parametrize("kind", DENSE_KINDS)
@pytest.mark.parametrize("beads", [1, 5])
def test_metric_energy_and_every_force_component_match_dense(kind, beads):
    cv, q = make_cv(kind), geometry(55, kind)
    ring = q[None] + np.random.default_rng(44).normal(0., .02, (beads,) + q.shape)
    actual = FluxMetricBias(cv, MASSES, 2.7).energy_force(ring)
    expected = DenseMetric(cv, MASSES, 2.7).energy_force(ring)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=5e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=5e-14)


@pytest.mark.parametrize("kind", KINDS)
def test_metric_force_is_negative_full_ring_energy_derivative(kind):
    cv, q = make_cv(kind), geometry(62, kind)
    ring = np.repeat(q[None], 3, axis=0)
    direction = np.random.default_rng(912).normal(size=ring.shape)
    bias = FluxMetricBias(cv, MASSES, 2.7)
    delta = 2e-6
    numeric = -(bias.energy_force(ring + delta * direction)[0]
                - bias.energy_force(ring - delta * direction)[0]) / (2 * delta)
    energy, force = bias.energy_force(ring)
    np.testing.assert_allclose(np.sum(force * direction), numeric, rtol=3e-7, atol=3e-9)
    one_energy, one_force = bias.energy_force(ring[:1])
    np.testing.assert_allclose(energy, 3 * one_energy, rtol=1e-14)
    np.testing.assert_allclose(force, np.broadcast_to(one_force, ring.shape), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("defect", ["shape", "rank", "nan", "infinity", "complex"])
def test_hvp_rejects_invalid_directions(kind, defect):
    q = geometry(11, kind)
    direction = np.ones_like(q)
    if defect == "shape":
        direction = direction[:-1]
    elif defect == "rank":
        direction = direction[None]
    elif defect == "complex":
        direction = direction.astype(complex) + 1j
    else:
        direction[0, 0] = np.nan if defect == "nan" else np.inf
    with pytest.raises(ValueError):
        make_cv(kind).centroid_value_gradient_hvp(q, direction)


@pytest.mark.parametrize("kind", ["distance", "com", "rate", "surface"])
@pytest.mark.parametrize("mode", ["instance", "class", "subclass"])
def test_legacy_hessian_override_remains_authoritative(monkeypatch, kind, mode):
    cv = make_cv(kind)
    original = type(cv).centroid_value_gradient_hessian

    def changed(self, q):
        value, g, h = original(self, q)
        return 2 * value + .3, 2 * g, 2 * h

    if mode == "instance":
        monkeypatch.setattr(cv, "centroid_value_gradient_hessian", MethodType(changed, cv))
    elif mode == "class":
        monkeypatch.setattr(type(cv), "centroid_value_gradient_hessian", changed)
    else:
        cv.__class__ = type("LegacyCV", (type(cv),), {"centroid_value_gradient_hessian": changed})
    q, v = geometry(19, kind), np.random.default_rng(1).normal(size=REFERENCE.shape)
    expected = dense_reference(cv, q)
    actual = cv.centroid_value_gradient_hvp(q, v)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual[2], np.einsum("acbd,bd->ac", expected[2], v), rtol=1e-12, atol=5e-14)
    e, f = FluxMetricBias(cv, MASSES, 2.7).energy_force(q[None])
    de, df = DenseMetric(cv, MASSES, 2.7).energy_force(q[None])
    np.testing.assert_allclose(e, de, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(f, df, rtol=1e-12, atol=5e-14)


def test_hessian_only_plugin_keeps_explicit_legacy_path():
    class Legacy:
        estimator = "centroid"

        def centroid_value_gradient_hessian(self, q):
            return np.sum(q * q) / 2, q.copy(), np.eye(q.size).reshape(q.shape + q.shape)

    q = geometry(45, "rate")[None]
    actual = FluxMetricBias(Legacy(), MASSES, 2.7).energy_force(q)
    expected = DenseMetric(Legacy(), MASSES, 2.7).energy_force(q)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_hvp_only_plugin_and_no_fallback_on_evaluation_failure():
    class MatrixFree:
        estimator = "centroid"

        def centroid_value_gradient_hvp(self, q, v):
            return np.sum(q * q) / 2, q.copy(), v.copy()

    q = geometry(45, "rate")[None]
    bias = FluxMetricBias(MatrixFree(), MASSES, 2.7)
    energy, force = bias.energy_force(q)
    z = np.sum(q[0] ** 2 / MASSES[:, None])
    np.testing.assert_allclose(energy, -np.log(np.sqrt(z / (2 * np.pi * 2.7))) / 2.7)
    np.testing.assert_allclose(force[0], q[0] / MASSES[:, None] / (2.7 * z))

    class Broken(MatrixFree):
        def centroid_value_gradient_hvp(self, q, v):
            raise RuntimeError("deliberate HVP failure")

        def centroid_value_gradient_hessian(self, q):
            pytest.fail("a failed HVP must not silently select a different path")

    with pytest.raises(RuntimeError, match="deliberate HVP failure"):
        FluxMetricBias(Broken(), MASSES, 2.7).energy_force(q)


@pytest.mark.parametrize("component", ["gradient", "hvp"])
@pytest.mark.parametrize("defect", ["broadcast", "nan", "complex"])
def test_custom_hvp_derivatives_cannot_broadcast_or_return_invalid_forces(component, defect):
    class Invalid:
        estimator = "centroid"

        def centroid_value_gradient_hvp(self, q, v):
            bad = np.ones(3) if defect == "broadcast" else np.full_like(q, np.nan)
            if defect == "complex":
                bad = q.astype(complex) + 1j
            return 0., bad if component == "gradient" else q.copy(), bad if component == "hvp" else v.copy()

    with pytest.raises(ValueError):
        FluxMetricBias(Invalid(), MASSES, 2.7).energy_force(REFERENCE[None])


@pytest.mark.parametrize("kind", ["angle", "periodic_angle", "difference", "periodic_difference"])
def test_new_hvp_does_not_silently_ignore_custom_gradient(kind):
    cv = make_cv(kind)
    original = type(cv).centroid_value_gradient

    def changed(self, q):
        value, gradient = original(self, q)
        return 2 * value, 2 * gradient

    cv.__class__ = type("CustomFirstDerivative", (type(cv),), {"centroid_value_gradient": changed})
    with pytest.raises((ValueError, NotImplementedError), match="HVP|Hessian|override|overrid"):
        FluxMetricBias(cv, MASSES, 2.7).energy_force(geometry(22, kind)[None])


@pytest.mark.parametrize("kind", ["distance", "com", "rate", "surface"])
def test_native_descriptors_never_bypass_hvp_override(monkeypatch, kind):
    from pyrpmd.block_propagation import cv_descriptor as block_descriptor
    from pyrpmd.propagation import NativeUnavailableError, cv_descriptor

    cv = make_cv(kind)
    original = cv.centroid_value_gradient_hvp
    monkeypatch.setattr(cv, "centroid_value_gradient_hvp", lambda q, v: original(q, v))
    for descriptor in (cv_descriptor, block_descriptor):
        with pytest.raises(NativeUnavailableError, match="overridden CV"):
            descriptor(cv, len(MASSES))


def test_large_python_rate_metric_never_materializes_a_hessian(monkeypatch):
    n = 3000
    q = np.zeros((n, 3))
    q[:6] = REFERENCE
    masses = np.full(n, 2.)
    cv = coordinate(q, masses)
    original_zeros, original_einsum = np.zeros, np.einsum

    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 4:
            pytest.fail("matrix-free metric allocated a dense Hessian")
        return original_zeros(shape, *args, **kwargs)

    def einsum(expression, *args, **kwargs):
        if isinstance(expression, str) and "->acbd" in expression:
            pytest.fail("matrix-free metric constructed a dense Hessian")
        return original_einsum(expression, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", zeros)
    monkeypatch.setattr(np, "einsum", einsum)
    e, f = FluxMetricBias(cv, masses, 2.7).energy_force(q[None])
    assert np.isfinite(e) and np.isfinite(f).all() and f.shape == (1, n, 3)


@pytest.fixture(scope="module")
def harmonic_library(tmp_path_factory):
    if not native_capabilities()["available"]:
        pytest.skip("native extension required")
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None or sys.platform == "win32":
        pytest.skip("POSIX C compiler required for C ABI fixture")
    directory = tmp_path_factory.mktemp("metric-hvp-pes")
    libraries = {}

    def build(n):
        if n in libraries:
            return libraries[n]
        metadata = {
            "schema": "pyrpmd.pes/1", "abi_version": 1, "provider_name": "hvp-harmonic",
            "provider_version": "1", "artifact_sha256": {}, "natoms": n,
            "atom_ids": list(map(str, range(n))), "atomic_numbers": [1] * n,
            "state": {"label": "test", "charge": 0, "multiplicity": 1},
            "available_states": [{"label": "test", "charge": 0, "multiplicity": 1}],
            "layout": "B,N,3:C", "dtype": "float64", "position_unit": "bohr",
            "energy_unit": "hartree", "force_unit": "hartree/bohr", "boundary_conditions": "nonperiodic",
            "capabilities": {"native_batch": True, "force_kind": "analytic", "thread_safety": "reentrant",
                             "stateful": False, "requires_bead_affinity": False, "execution_device": "cpu"},
        }
        source = directory / f"harmonic_{n}.c"
        source.write_text('''
#include "pyrpmd/pes_api_v1.h"
#include <stdlib.h>
#include <string.h>
struct pyrpmd_pes_context_v1 { int alive; };
static const char metadata[] = METADATA;
int32_t pyrpmd_pes_abi_version_v1(void) { return 1; }
pyrpmd_rc_v1 pyrpmd_pes_create_v1(const pyrpmd_create_options_v1* o, pyrpmd_pes_context_v1** c, pyrpmd_error_v1* e) {
  *c = malloc(sizeof(**c)); return *c ? 0 : PYRPMD_INTERNAL_ERROR_V1;
}
pyrpmd_rc_v1 pyrpmd_pes_metadata_v1(const pyrpmd_pes_context_v1* c, char* out, int64_t size, int64_t* need, pyrpmd_error_v1* e) {
  *need = sizeof(metadata); if (size < *need) return PYRPMD_BUFFER_TOO_SMALL_V1;
  memcpy(out, metadata, sizeof(metadata)); return 0;
}
pyrpmd_rc_v1 pyrpmd_pes_evaluate_v1(pyrpmd_pes_context_v1* c, const pyrpmd_pes_input_v1* x, pyrpmd_pes_output_v1* y, pyrpmd_error_v1* e) {
  if (x->natoms != NATOMS) return PYRPMD_INVALID_ARGUMENT_V1;
  for (int64_t b = 0; b < x->nstructures; ++b) {
    y->energies[b] = 0; y->structure_status[b] = 0;
    for (int64_t j = 0; j < 3*NATOMS; ++j) {
      int64_t i = b*3*NATOMS+j; double q = x->positions[i];
      y->energies[b] += .0015*q*q; y->forces[i] = -.003*q;
    }
  }
  return 0;
}
void pyrpmd_pes_destroy_v1(pyrpmd_pes_context_v1* c) { free(c); }
'''.replace("METADATA", json.dumps(json.dumps(metadata))).replace("NATOMS", str(n)))
        library = directory / f"harmonic_{n}.so"
        subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-I", str(ROOT / "native/include"),
                        str(source), "-o", str(library)], check=True, capture_output=True, text=True)
        libraries[n] = library
        return library

    return build


@pytest.mark.parametrize("kind,seed", [(kind, 22) for kind in ("distance", "difference", "com", "cartesian")]
                         + [(kind, seed) for kind in ("rate", "surface") for seed in (0, 1, 2, 12)])
@pytest.mark.parametrize("beads", [1, 4])
def test_native_block_metric_matches_dense_force_trajectory(harmonic_library, kind, seed, beads):
    pes = CABIPotential(harmonic_library(len(MASSES)))
    try:
        dyn = Dynamics(pes, MASSES, 2.7, beads, .002)
        center = np.random.default_rng(seed).normal(0., 2., REFERENCE.shape) if kind in ("rate", "surface") else geometry(seed, kind)
        q = center[None] + np.random.default_rng(913).normal(0., .01, (beads, 6, 3))
        a = State(q, np.random.default_rng(31).normal(0., .1, q.shape))
        b = a.copy()
        cv = make_cv(kind)
        actual = dyn.propagate_block(a, 11, bias=FluxMetricBias(cv, MASSES, 2.7), backend="native", record_states=True)
        expected = dyn.propagate_block(b, 11, bias=DenseMetric(cv, MASSES, 2.7), backend="python", record_states=True)
        for x, y in ((actual.q, expected.q), (actual.p, expected.p), (actual.energies, expected.energies)):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=2e-13)
    finally:
        pes.close()


@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 12])
def test_native_fused_rate_metric_matches_dense_force_trajectory(harmonic_library, beads, seed):
    pes = CABIPotential(harmonic_library(len(MASSES)))
    try:
        dyn = Dynamics(pes, MASSES, 2.7, beads, .002)
        center = np.random.default_rng(seed).normal(0., 2., REFERENCE.shape)
        q = center[None] + np.random.default_rng(913).normal(0., .01, (beads, 6, 3))
        a = State(q, np.random.default_rng(31).normal(0., .1, q.shape))
        b = a.copy()
        cv = coordinate()
        actual = dyn.propagate(a, 11, bias=FluxMetricBias(cv, MASSES, 2.7), backend="native")
        expected = dyn.propagate(b, 11, bias=DenseMetric(cv, MASSES, 2.7), backend="python")
        for x, y in ((a.q, b.q), (a.p, b.p), (actual.energies, expected.energies)):
            np.testing.assert_allclose(x, y, rtol=1e-12, atol=2e-13)
    finally:
        pes.close()


@pytest.mark.parametrize("api", ["block", "fused"])
@pytest.mark.parametrize("seed", [0, 1, 2, 12])
def test_native_rate_force_components_match_dense_without_trajectory_attenuation(harmonic_library, api, seed):
    pes = CABIPotential(harmonic_library(len(MASSES)))
    try:
        q = np.random.default_rng(seed).normal(0., 2., REFERENCE.shape)[None]
        dt = 1e-10
        dyn = Dynamics(pes, MASSES, 2.7, 1, dt)
        state = State(q, np.zeros_like(q))
        cv = coordinate()
        bias = FluxMetricBias(cv, MASSES, 2.7)
        if api == "block":
            dyn.propagate_block(state, 1, bias=bias, backend="native")
        else:
            dyn.propagate(state, 1, bias=bias, backend="native")
        # The O(dt^2) position update is below one ULP for these coordinates.
        # Verify that condition, then the two kicks measure the force itself:
        # p/dt = F_PES(q) + F_metric(q), without a small-dt comparison tolerance.
        np.testing.assert_array_equal(state.q, q)
        metric_force = state.p / dt + .003 * q
        expected = DenseMetric(cv, MASSES, 2.7).energy_force(q)[1]
        np.testing.assert_allclose(metric_force, expected, rtol=1e-12, atol=5e-14)
    finally:
        pes.close()


@pytest.mark.parametrize("api", ["block", "fused"])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux virtual-memory limit regression")
def test_native_metric_n3000_under_one_gib_address_space(harmonic_library, api):
    pytest.importorskip("resource")
    library = harmonic_library(3000)
    # A single old 9000x9000 Hessian was 618 MiB; both native paths kept
    # multiple such arrays.  This cap verifies allocations in a fresh process.
    program = r'''
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
import numpy as np
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.bias import FluxMetricBias
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import CABIPotential
n = 3000
q = np.zeros((n, 3)); q[:3] = [[-1.7, .2, 0.], [0., 0., 0.], [1.8, .3, .1]]
masses = np.full(n, 1800.)
cv = RPMDrateCoordinate([[0], [1, 2]], masses, q, [[0, 1]], [[1, 2]], 10.)
pes = CABIPotential(sys.argv[1])
dyn = Dynamics(pes, masses, 220., 1, .001)
s = State(q[None], np.zeros((1, n, 3)))
bias = FluxMetricBias(cv, masses, 220.)
if sys.argv[2] == "block":
    out = dyn.propagate_block(s, 1, bias=bias, backend="native", record_energy=False)
else:
    out = dyn.propagate(s, 1, bias=bias, backend="native", record_centroids=False)
assert out.backend == "native" and np.isfinite(s.q).all() and np.isfinite(s.p).all()
pes.close()
print("N=3000 native metric completed within 1 GiB")
'''
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", program, str(library), api], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "completed within 1 GiB" in result.stdout
