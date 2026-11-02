"""Independent Jacobians, symmetries and runtime contracts of extended CVs."""

import numpy as np
import pytest

from pyrpmd.advanced_cv import CoordinationNumber, Dihedral, PathCV, RMSD, TorchScriptCV
from pyrpmd.bias import UmbrellaBias
from pyrpmd.cv import make_cv
from pyrpmd.system import System


def check_jacobian(cv, q):
    result = cv.value_and_jacobian(q)
    numerical = np.empty_like(result.jacobian)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        delta = cv.value(plus) - cv.value(minus)
        for j, period in enumerate(cv.periods):
            if period is not None:
                delta[j] = (delta[j] + period / 2) % period - period / 2
        numerical[(slice(None),) + index] = delta / 2e-6
    np.testing.assert_allclose(result.jacobian, numerical, rtol=2e-5, atol=2e-8)


@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
@pytest.mark.parametrize("kind", ["dihedral", "coordination", "rmsd", "path"])
def test_full_ring_jacobians(kind, estimator):
    rng = np.random.default_rng(154)
    q = rng.normal(size=(3, 5, 3))
    reference = rng.normal(size=(5, 3))
    cvs = {
        "dihedral": Dihedral([0, 1, 2, 3], estimator=estimator),
        "coordination": CoordinationNumber([0, 1], [2, 3, 4], r0=0.8, estimator=estimator),
        "rmsd": RMSD(reference, weights=[1, 2, 3, 4, 5], estimator=estimator),
        "path": PathCV([reference, reference * 1.5, reference * 2], path_lambda=2,
                       scale=0.7, estimator=estimator),
    }
    check_jacobian(cvs[kind], q)
    np.testing.assert_allclose(cvs[kind].value(q + 7), cvs[kind].value(q), atol=2e-14)


def torsion_geometry(angle):
    return np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., 1.], [np.cos(angle), np.sin(angle), 1.]])


def test_dihedral_circular_mean_does_not_cross_wrong_seam():
    q = np.array([torsion_geometry(np.pi - 0.1), torsion_geometry(-np.pi + 0.1)])
    cv = Dihedral([0, 1, 2, 3], estimator="bead_average")
    assert abs(cv.value(q)[0]) == pytest.approx(np.pi)
    check_jacobian(cv, q)
    with pytest.raises(ValueError, match="circular mean"):
        cv.value(np.array([torsion_geometry(0), torsion_geometry(np.pi)]))


def test_periodic_dihedral_and_coordination_use_bond_images():
    q = torsion_geometry(0.7)[None]
    shifted = q.copy()
    shifted[0, 0] += [20, -10, 0]
    shifted[0, 3] += [0, 30, -10]
    cell = np.eye(3) * 10
    for cv in [Dihedral([0, 1, 2, 3], cell=cell), CoordinationNumber([0, 1], [2, 3], r0=1.4, cell=cell)]:
        np.testing.assert_allclose(cv.value(q), cv.value(shifted), atol=1e-14)
        check_jacobian(cv, shifted)


def test_coordination_switch_is_smooth_at_r0_and_coincidence():
    cv = CoordinationNumber([0], [1], r0=2)
    q = np.zeros((1, 2, 3))
    q[0, 1, 0] = 2
    result = cv.value_and_jacobian(q)
    assert result.value[0] == pytest.approx(0.5)
    assert result.jacobian[0, 0, 1, 0] == pytest.approx(-6 / 8)
    q[:] = 0
    assert cv.value(q)[0] == 1
    np.testing.assert_array_equal(cv.value_and_jacobian(q).jacobian, 0)
    q[0, 1, 0] = 1e100
    assert np.isfinite(cv.value_and_jacobian(q).jacobian).all()
    with pytest.raises(ValueError, match="disjoint"):
        CoordinationNumber([0, 1], [1, 2], r0=2)


def test_rmsd_proper_rotation_and_reference_singularity():
    rng = np.random.default_rng(31)
    reference = rng.normal(size=(6, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    cv = RMSD(reference)
    transformed = reference @ rotation + 2
    value, gradient = cv.squared_value_gradient(transformed)
    assert value < 1e-28
    assert np.max(np.abs(gradient)) < 1e-14
    with pytest.raises(ValueError, match="exact reference"):
        cv.value(transformed[None])
    reflected = reference.copy()
    reflected[:, 0] *= -1
    assert cv.value(reflected[None])[0] > 0.1


def test_path_matches_closed_formula_and_negative_z_convention():
    references = np.zeros((3, 1, 3))
    references[:, 0, 0] = [-1, 0, 1]
    cv = PathCV(references, path_lambda=2, scale=2, align="none")
    q = np.array([[[0.2, 0., 0.]]])
    d = (np.array([-1, 0, 1]) - 0.2)**2 / 4
    weights = np.exp(-2 * d)
    expected = [(weights @ [1, 2, 3]) / weights.sum(), -np.log(weights.sum()) / 2]
    np.testing.assert_allclose(cv.value(q), expected)
    assert cv.value(q)[1] < 0
    check_jacobian(cv, q)
    for component, index in [("s", 0), ("z", 1)]:
        scalar = PathCV(references, path_lambda=2, scale=2, align="none", component=component)
        np.testing.assert_allclose(scalar.value(q), cv.value(q)[index:index + 1])


def test_path_umbrella_force_includes_full_ring_bead_factor():
    rng = np.random.default_rng(7)
    refs = rng.normal(size=(3, 4, 3))
    q = rng.normal(size=(4, 4, 3))
    cv = PathCV(refs, path_lambda=3)
    bias = UmbrellaBias(cv, [1.5, 0.1], [2.0, 3.0])
    energy, force = bias.energy_force(q)
    for index in [(0, 0, 0), (2, 3, 1)]:
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        expected = -(bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / 2e-6
        assert force[index] == pytest.approx(expected, rel=1e-5, abs=1e-8)


def test_new_cvs_construct_from_input_specs_and_safe_reference_file(tmp_path):
    q = torsion_geometry(0.4)
    system = System(("H",) * 4, np.ones(4), q)
    reference = tmp_path / "reference.npy"
    np.save(reference, q * 1.1)
    specs = [{"kind": "dihedral", "atoms": [0, 1, 2, 3]},
             {"kind": "coordination", "group_a": [0], "group_b": [1, 2, 3], "r0": "2 bohr"},
             {"kind": "rmsd", "reference": str(reference)},
             {"kind": "path", "references": [q.tolist(), (q * 1.3).tolist()], "path_lambda": 4}]
    for spec in specs:
        assert np.isfinite(make_cv(spec, system).value(q[None])).all()
    with pytest.raises(ValueError, match=".npy"):
        RMSD(tmp_path / "reference.pickle")


@pytest.mark.parametrize("estimator", ["centroid", "bead_average", "full_ring"])
def test_torchscript_cpu_real_autograd_jacobian(tmp_path, estimator):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def forward(self, positions):
            return torch.stack([torch.sum(positions**2), torch.sum(positions[..., 0]**3)])

    q = np.random.default_rng(8).normal(size=(2, 3, 3))
    example = torch.tensor(q if estimator == "full_ring" else q[0], dtype=torch.float64)
    path = tmp_path / "cv.pt"
    torch.jit.trace(Model(), example).save(str(path))
    cv = TorchScriptCV(path, dimension=2, estimator=estimator)
    with torch.no_grad():
        check_jacobian(cv, q)
    np.testing.assert_allclose(cv.value_and_jacobian(q).value, cv.value(q))


def test_torchscript_rejects_bad_output_shape(tmp_path):
    torch = pytest.importorskip("torch")

    class Bad(torch.nn.Module):
        def forward(self, positions):
            return torch.sum(positions)

    path = tmp_path / "bad.pt"
    torch.jit.trace(Bad(), torch.zeros((2, 3), dtype=torch.float64)).save(str(path))
    with pytest.raises(ValueError, match=r"\[D\]"):
        TorchScriptCV(path).value(np.zeros((1, 2, 3)))


def test_torchscript_rejects_detached_derivatives(tmp_path):
    torch = pytest.importorskip("torch")

    class Detached(torch.nn.Module):
        def forward(self, positions):
            return torch.sum(positions).reshape(1).detach()

    path = tmp_path / "detached.pt"
    torch.jit.trace(Detached(), torch.zeros((2, 3), dtype=torch.float64)).save(str(path))
    with pytest.raises(ValueError, match="detached"):
        TorchScriptCV(path).value(np.zeros((1, 2, 3)))


def test_bias_history_identity_distinguishes_advanced_cv_parameters():
    from pyrpmd.analysis.metadynamics import _cv_signature

    refs = np.random.default_rng(12).normal(size=(2, 4, 3))
    a = PathCV(refs, path_lambda=2)
    b = PathCV(refs, path_lambda=3)
    c = PathCV(refs * 1.01, path_lambda=2)
    assert _cv_signature(a) != _cv_signature(b)
    assert _cv_signature(a) != _cv_signature(c)
    assert _cv_signature(a) == _cv_signature(PathCV(refs.copy(), path_lambda=2))
    assert _cv_signature(CoordinationNumber([0], [1], r0=1)) != _cv_signature(CoordinationNumber([0], [1], r0=2))
