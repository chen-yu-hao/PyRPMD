"""Independent measure, force, invariance, model identity and IBI checks."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.potentials.base import check_potential
from pyrpmd.qcmd import (FQCMDModel, FQCMDPotential, TriatomicQuasiCentroid,
                         coordinate_histograms, fqcmd_base_identity,
                         update_fqcmd_model)
from pyrpmd.system import System


def system_and_model(*, nonzero=True, provenance=None):
    system = System(("O", "H", "H"), np.array([16., 1., 1.]),
                    np.array([[0., 0., 0.], [1.8, 0., 0.], [-.5, 1.7, 0.]]))
    geometry = TriatomicQuasiCentroid([[0, 1, 2]], system.masses)
    edges = [np.linspace(.4, 3.6, 33), np.linspace(.2, 2.9, 28)]
    centers = [.5 * (e[1:] + e[:-1]) for e in edges]
    values = ([.3 * (centers[0] - 1.6) ** 2, .2 * (centers[1] - 1.8) ** 2]
              if nonzero else [np.zeros_like(x) for x in centers])
    spec = {"provider": "free"}
    model = FQCMDModel(geometry, [[0, 0, 1]], edges, values, beta=2.,
                       base_identity=fqcmd_base_identity(spec, system), provenance=provenance)
    return system, model


def test_bead_mean_internal_coordinates_are_not_cartesian_centroid_coordinates():
    system, model = system_and_model()
    system.geometry[1, 0] = 1.83  # Avoid a histogram edge at machine precision.
    angle = .5
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    q = np.stack([system.geometry @ rotation, system.geometry @ rotation.T])[None]
    target = coordinate_histograms(q, model.geometry, model.groups, model.edges, bead_average=True)
    centroid = coordinate_histograms(q.mean(1), model.geometry, model.groups, model.edges, bead_average=False)
    assert target[0].sum() == centroid[0].sum() == 2
    assert not np.array_equal(target[0], centroid[0])
    expected = np.histogram(np.linalg.norm(system.geometry[1:], axis=1), bins=model.edges[0])[0]
    np.testing.assert_array_equal(target[0], expected)


def test_exact_p1_distribution_and_count_pooling():
    system, model = system_and_model()
    q = np.stack([system.geometry, system.geometry + 3])
    quantum = coordinate_histograms(q[:, None], model.geometry, model.groups, model.edges, bead_average=True)
    classical = coordinate_histograms(q, model.geometry, model.groups, model.edges, bead_average=False)
    for actual, expected in zip(quantum, classical):
        np.testing.assert_array_equal(actual, expected)
    assert [int(c.sum()) for c in quantum] == [4, 2]


def test_correction_is_conservative_and_matches_independent_quadratic_energy():
    system, model = system_and_model()
    potential = FQCMDPotential(FreePotential(), model)
    q = np.stack([system.geometry, system.geometry * .95])
    a, b = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
    r1, r2 = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    angle = np.arccos(np.sum(a * b, axis=1) / (r1 * r2))
    energy = .3 * ((r1 - 1.6) ** 2 + (r2 - 1.6) ** 2) + .2 * (angle - 1.8) ** 2
    np.testing.assert_allclose(potential.evaluate(q).energy, energy, atol=4e-17)
    assert check_potential(potential, q)["max_absolute_force_error"] < 2e-10


def test_force_equivariance_and_equivalent_terminal_permutation():
    system, model = system_and_model()
    potential = FQCMDPotential(FreePotential(), model)
    q = system.geometry[None]
    rotation, _ = np.linalg.qr(np.random.default_rng(81).normal(size=(3, 3)))
    original = potential.evaluate(q)
    transformed = potential.evaluate(q @ rotation + np.array([3., -8., 1.]))
    np.testing.assert_allclose(transformed.energy, original.energy, atol=3e-16)
    np.testing.assert_allclose(transformed.forces, original.forces @ rotation, atol=2e-15)
    swapped = potential.evaluate(q[:, [0, 2, 1]])
    np.testing.assert_allclose(swapped.energy, original.energy, atol=3e-16)
    np.testing.assert_allclose(swapped.forces, original.forces[:, [0, 2, 1]], atol=2e-15)
    np.testing.assert_allclose(original.forces.sum(1), 0., atol=1e-16)
    np.testing.assert_allclose(np.cross(q, original.forces).sum(1), 0., atol=1e-16)


@pytest.mark.parametrize("degree", [None, 2, 4])
def test_ibi_independent_boltzmann_reference_includes_coordinate_measure(degree):
    _, model = system_and_model(nonzero=False)
    target, sampled, expected = [], [], []
    for i, (x, edge) in enumerate(zip(model.centers, model.edges)):
        jacobian = x ** 2 if i == 0 else np.sin(x)
        width = np.diff(edge)
        base = .07 * (x - 1.6) ** 2
        delta = .12 * (x - 1.5) ** 2 - .03 * x
        sampled.append(jacobian * np.exp(-model.beta * base) * width)
        target.append(jacobian * np.exp(-model.beta * (base + delta)) * width)
        expected.append(delta - delta[len(delta) // 2])
    fitted, diagnostics = update_fqcmd_model(model, target, sampled, polynomial_degree=degree)
    for actual, want in zip(fitted.corrections, expected):
        np.testing.assert_allclose(actual, want, atol=8e-16)
    assert not diagnostics["converged"]
    assert diagnostics["comparison"] == "pre-update classical vs target marginals"
    assert diagnostics["sampled_model_sha256"] == model.fingerprint


def test_ibi_fixed_point_and_explicit_regularization():
    _, model = system_and_model()
    counts = [np.arange(1, len(e), dtype=float) for e in model.edges]
    same, diagnostics = update_fqcmd_model(model, counts, counts)
    for original, updated in zip(model.corrections, same.corrections):
        np.testing.assert_allclose(np.diff(updated), np.diff(original), atol=2e-17)
    assert all(g["l1_probability_error"] == 0 for g in diagnostics["groups"])
    counts[0][0] = 0
    with pytest.raises(ValueError, match="zero-density"):
        update_fqcmd_model(model, counts, counts)
    fitted, _ = update_fqcmd_model(model, counts, counts, regularization=.01)
    assert np.isfinite(fitted.corrections[0]).all()
    sparse = [np.r_[1., np.zeros(len(e) - 2)] for e in model.edges]
    with pytest.raises(ValueError, match="common occupied"):
        update_fqcmd_model(model, sparse, sparse, regularization=.01, polynomial_degree=2)


def test_model_roundtrip_and_content_identity(tmp_path):
    _, model = system_and_model(provenance={"reference_beads": 32, "reference_seed": 781})
    path = tmp_path / "model.npz"
    model.save(path)
    restored = FQCMDModel.load(path)
    assert restored.to_dict() == model.to_dict()
    assert restored.fingerprint == model.fingerprint
    changed, _ = update_fqcmd_model(model, [np.ones(len(e) - 1) for e in model.edges],
                                   [np.ones(len(e) - 1) for e in model.edges])
    assert changed.fingerprint != model.fingerprint  # training history, even at a fixed point
    with load_checkpoint(path, require_integrity=True) as saved:
        payload = {key: saved[key].copy() for key in saved.files}
    payload["correction_0"][4] += .001
    atomic_savez(path, kind="PyRPMD.fqcmd-model.v1", **payload)
    with pytest.raises(ValueError, match="scientific fingerprint"):
        FQCMDModel.load(path)


def test_model_owns_inputs_and_freezes_parameters():
    system, original = system_and_model()
    edges = [e.copy() for e in original.edges]
    corrections = [v.copy() for v in original.corrections]
    provenance = {"beads": 16, "seeds": [2, 3]}
    model = FQCMDModel(original.geometry, original.groups, edges, corrections, beta=2.,
                       base_identity=original.base_identity, provenance=provenance)
    before = model.to_dict()
    edges[0] *= 2
    corrections[0][:] = 0
    provenance["seeds"].append(10)
    model.provenance["seeds"].append(18)
    assert model.to_dict() == before
    with pytest.raises(AttributeError, match="immutable"):
        model.beta = 3.
    with pytest.raises(ValueError):
        model.corrections[0][0] = 0
    wrapper = FQCMDPotential(FreePotential(), model)
    with pytest.raises(AttributeError):
        wrapper.model = original


def test_source_fingerprint_ignores_starting_coordinates_but_covers_mass_and_files(tmp_path):
    system, _ = system_and_model()
    source = tmp_path / "physical.py"
    source.write_text("# version 1\n")
    spec = {"provider": "python", "factory": str(source) + ":build", "params": {"k": .1}}
    before = fqcmd_base_identity(spec, system)
    other = copy.deepcopy(system)
    other.geometry += 1.
    assert fqcmd_base_identity(spec, other) == before
    other.masses[1] *= 2
    assert fqcmd_base_identity(spec, other) != before
    source.write_text("# version 2\n")
    assert fqcmd_base_identity(spec, system) != before


def test_system_validation_does_not_allow_temperature_mass_or_potential_changes():
    system, model = system_and_model()
    model.validate_system(system, 2., base_identity=model.base_identity)
    with pytest.raises(ValueError, match="temperature"):
        model.validate_system(system, 3., base_identity=model.base_identity)
    altered = copy.deepcopy(system)
    altered.masses[1] *= 2
    with pytest.raises(ValueError, match="masses"):
        model.validate_system(altered, 2., base_identity=model.base_identity)
    with pytest.raises(ValueError, match="potential"):
        model.validate_system(system, 2., base_identity="a" * 64)


def test_missing_support_and_unsupported_base_are_explicit():
    system, model = system_and_model()
    q = system.geometry[None].copy()
    q[:, 1] *= 5
    with pytest.raises(ValueError, match="fitted domain"):
        FQCMDPotential(FreePotential(), model).evaluate(q)
    with pytest.raises(ValueError, match="histogram domain"):
        coordinate_histograms(q, model.geometry, model.groups, model.edges, bead_average=False)
    for metadata in ({"centroid_pmf_beta_au": 2.}, {"quasicentroid_pmf_beta_au": 2.},
                     {"boundary_conditions": "periodic"}, {"execution_device": "cuda:0"}):
        base = FreePotential()
        base.metadata = metadata
        with pytest.raises(ValueError):
            FQCMDPotential(base, model)
    base = FreePotential()
    base.pbc = [True, False, False]
    with pytest.raises(ValueError, match="nonperiodic"):
        FQCMDPotential(base, model)


@pytest.mark.parametrize("kwargs", [
    {"groups": [[0, 1, 1]]}, {"groups": [[0, 0, 2]]}, {"groups": [[False, 0, 1]]},
    {"beta": True}, {"beta": -2}, {"base_identity": "not-a-content-hash"},
    {"edges": [np.array([.4, 1., 1., 2., 3.]), np.linspace(.2, 2.9, 28)]},
    {"edges": [np.linspace(.4, 3.6, 33), np.linspace(.0, np.pi, 28)]},
    {"provenance": {"bad": float("nan")}},
])
def test_model_rejects_invalid_specification(kwargs):
    _, model = system_and_model()
    args = dict(geometry=model.geometry, groups=model.groups, edges=model.edges,
                corrections=model.corrections, beta=model.beta, base_identity=model.base_identity)
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        FQCMDModel(**args)


def test_wrapper_borrows_base_and_preserves_its_force():
    system, model = system_and_model(nonzero=False)
    base = HarmonicPotential(.3)
    wrapped = FQCMDPotential(base, model)
    q = system.geometry[None]
    np.testing.assert_array_equal(wrapped.evaluate(q).forces, base.evaluate(q).forces)
    wrapped.close()
    np.testing.assert_array_equal(wrapped.evaluate(q).energy, base.evaluate(q).energy)
