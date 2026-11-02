"""Independent force, affine-virial and model-provenance checks for f-QCMD."""

import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.potentials import FreePotential
from pyrpmd.potentials.base import PESResult, Potential, check_potential
from pyrpmd.qcmd import QCMDModel, QuasiCentroidPotential, qcmd_base_identity
from pyrpmd.system import System


def model_args():
    edges = (np.linspace(1., 3., 31), np.linspace(.6, 2.8, 35))
    r, theta = ((e[1:] + e[:-1]) / 2 for e in edges)
    return dict(molecules=[(0, 1, 2)], masses=[16., 1., 1.], beta=3., grids=edges,
                corrections=(.03 * (r - 1.8)**3, -.04 * (theta - 1.7)**2),
                base_identity="b" * 64, training_beads=8,
                training={"regularization": .01, "nested": {"seed": 13}},
                diagnostics=[{"raw_total_variation": [.1, .2]}])


def geometry(r1=1.8, r2=1.9, theta=1.7):
    return np.array([[0., 0., 0.], [r1, 0., 0.],
                     [r2 * np.cos(theta), r2 * np.sin(theta), 0.]])


class CentralField(Potential):
    def evaluate(self, q):
        f = -.04 * q
        return PESResult(.02 * np.sum(q*q, axis=(1, 2)), f,
                         virial=np.einsum("bai,baj->bij", q, f))


def test_corrected_forces_match_energy_difference_and_affine_virial():
    potential = QuasiCentroidPotential(CentralField(), QCMDModel(**model_args()))
    q = geometry()[None] + np.random.default_rng(31).normal(scale=.05, size=(4, 3, 3))
    assert check_potential(potential, q)["max_absolute_force_error"] < 2e-10
    actual = potential.evaluate(q)
    eps = 2e-6
    numerical = np.empty((4, 3, 3))
    for i in range(3):
        for j in range(3):
            plus, minus = np.eye(3), np.eye(3)
            plus[i, j] += eps
            minus[i, j] -= eps
            numerical[:, i, j] = -(potential.evaluate(q @ plus).energy
                                    - potential.evaluate(q @ minus).energy) / (2 * eps)
    np.testing.assert_allclose(actual.virial, numerical, atol=5e-11)
    assert potential.evaluate(q).forces.dtype == np.float64


def test_correction_batch_is_not_bead_scaled_and_preserves_rigid_symmetries():
    model = QCMDModel(**model_args())
    potential = QuasiCentroidPotential(FreePotential(), model)
    q = np.repeat(geometry()[None], 9, axis=0)
    result = potential.evaluate(q)
    single = potential.evaluate(q[:1])
    np.testing.assert_allclose(result.energy, np.repeat(single.energy, 9))
    np.testing.assert_allclose(result.forces, np.repeat(single.forces, 9, axis=0))
    rotation, _ = np.linalg.qr(np.random.default_rng(77).normal(size=(3, 3)))
    transformed = potential.evaluate(q @ rotation + [5., -7., 3.])
    np.testing.assert_allclose(transformed.energy, result.energy, atol=5e-16)
    np.testing.assert_allclose(transformed.forces, result.forces @ rotation, atol=2e-15)
    np.testing.assert_allclose(result.forces.sum(axis=1), 0., atol=1e-17)
    np.testing.assert_allclose(np.cross(q, result.forces).sum(axis=1), 0., atol=1e-17)


def test_spline_halfbins_and_endpoints_have_explicit_finite_values_and_slopes():
    model = QCMDModel(**model_args())
    r, a = model.grids
    for radius, angle in [(r[0], a[0]), (r[-1], a[-1]),
                          (r[0] + .1 * (r[1] - r[0]), a[-1] - .1 * (a[1] - a[0]))]:
        energy, force = model.correction(np.array([[[radius, radius, angle]]]))
        assert np.isfinite(energy).all() and np.isfinite(force).all()
    for radius, angle in [(r[0], a[0]), (r[-1], a[-1])]:
        _, derivative = model.correction(np.array([[[radius, radius, angle]]]))
        np.testing.assert_allclose(derivative, 0., atol=3e-15)
    for radius, angle in [(r[0] - 1e-12, 1.7), (r[-1] + 1e-12, 1.7),
                          (1.8, a[0] - 1e-12), (1.8, a[-1] + 1e-12)]:
        with pytest.raises(ValueError, match="outside fitted grid"):
            model.correction(np.array([[[radius, radius, angle]]]))


def test_models_own_arrays_and_metadata_and_roundtrip_atomically(tmp_path):
    args = model_args()
    model = QCMDModel(**args)
    before = model.to_dict()
    args["corrections"][0][:] = 0
    args["training"]["nested"]["seed"] = -9
    model.training["nested"]["seed"] = -11
    assert model.to_dict() == before
    with pytest.raises(ValueError):
        model.corrections[0][:] = 0
    with pytest.raises(AttributeError, match="immutable"):
        model.beta = 4.
    with pytest.raises(AttributeError, match="immutable"):
        model.coordinates.molecules = np.array([[1, 0, 2]])
    path = tmp_path / "model.npz"
    model.save(path)
    reread = QCMDModel.load(path)
    assert reread.to_dict() == model.to_dict()
    assert reread.model_identity == model.model_identity
    assert not list(tmp_path.glob("*.tmp"))
    with load_checkpoint(path) as saved:
        raw = json.loads(str(saved["metadata"]))
    raw["interpolation"] = "unrecorded replacement"
    atomic_savez(path, kind="PyRPMD.fqcmd-model.v1", metadata=json.dumps(raw),
                 model_identity=model.model_identity)
    with pytest.raises(ValueError, match="identity|convention"):
        QCMDModel.load(path)


@pytest.mark.parametrize("key,value", [
    ("beta", True), ("beta", np.nan), ("training_beads", True),
    ("base_identity", "unknown"), ("masses", [16., 1., 2.]),
    ("masses", [16., 1., 1., 2.]), ("grids", [1., np.linspace(.5, 2.9, 20)]),
    ("grids", [np.array([1., 1.1, 1.2, 1.5, 2.]), np.linspace(.5, 2.9, 20)]),
    ("grids", [np.linspace(0., 3., 31), np.linspace(.6, 2.8, 35)]),
    ("grids", [np.linspace(1., 3., 31), np.linspace(.6, np.pi, 35)]),
    ("corrections", [np.zeros(2), np.zeros(34)]),
    ("target_counts", [np.full(30, -1.), np.ones(34)]),
    ("training", []), ("diagnostics", {}),
])
def test_invalid_or_unsupported_model_specs_fail(key, value):
    args = model_args()
    args[key] = value
    with pytest.raises((ValueError, TypeError)):
        QCMDModel(**args)


def test_base_identity_hashes_content_and_composition_but_not_initial_geometry(tmp_path):
    module = tmp_path / "pes.py"
    module.write_text("# original potential\n")
    spec = {"provider": "python", "factory": str(module) + ":build"}
    system = System(("O", "H", "H"), np.array([16., 1., 1.]), geometry())
    original = qcmd_base_identity(spec, system)
    displaced = System(system.symbols, system.masses, geometry() + 3)
    assert qcmd_base_identity(spec, displaced) == original
    module.write_text("# different potential\n")
    assert qcmd_base_identity(spec, system) != original
    model = QCMDModel(**{**model_args(), "base_identity": original})
    model.validate_system(system, 3., base_identity=original)
    with pytest.raises(ValueError, match="temperature"):
        model.validate_system(system, 4., base_identity=original)
    with pytest.raises(ValueError, match="identity"):
        model.validate_system(system, 3., base_identity="c" * 64)
    isotope = System(system.symbols, system.masses * 1.01, system.geometry)
    with pytest.raises(ValueError, match="masses"):
        model.validate_system(isotope, 3.)
    periodic = System(system.symbols, system.masses, system.geometry, cell=np.eye(3) * 20, pbc=(True,)*3)
    with pytest.raises(ValueError, match="nonperiodic"):
        model.validate_system(periodic, 3.)


def test_double_correction_and_nonphysical_pmf_base_are_rejected():
    model = QCMDModel(**model_args())
    fitted = QuasiCentroidPotential(FreePotential(), model)
    with pytest.raises(ValueError, match="original physical PES"):
        QuasiCentroidPotential(fitted, model)


@pytest.mark.parametrize("attribute,value", [
    ("required_bond_constraints", ((0, 1, 1.8),)),
    ("metadata", {"required_bond_constraints": ((0, 1, 1.8),)}),
    ("pbc", np.array([True, True, True])), ("cell", np.eye(3)),
    ("metadata", {"boundary_conditions": "periodic"}),
])
def test_wrapper_does_not_hide_unsupported_base_requirements(attribute, value):
    base = FreePotential()
    setattr(base, attribute, value)
    with pytest.raises(ValueError, match="constraints|nonperiodic"):
        QuasiCentroidPotential(base, QCMDModel(**model_args()))
