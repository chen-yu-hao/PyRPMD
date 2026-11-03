"""Real MACE second derivatives: independent ASE/force-difference references."""

from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mace.modules")
pytest.importorskip("vesin_torch")
pytest.importorskip("ase")

from mace.calculators import MACECalculator

from examples.potentials._random_mace import make_random_mace
from pyrpmd.potentials.adapters import make_named_potential
from pyrpmd.potentials.batched_ml import MACEBatchedPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE
from pyrpmd.instanton import RingPolymerAction


def _structures():
    # A locally untrained model, with no download of model weights.
    coords_angstrom = np.array([
        [0.1, 0.2, 0.1], [1.0, 0.1, -0.2],
        [0.2, 1.2, 0.4], [1.4, 1.0, 0.3],
    ])
    cell = np.array([[5.1, .3, .2], [0., 4.8, .1], [0., 0., 5.0]])
    q = np.stack((coords_angstrom, coords_angstrom + 0.03)) / BOHR_ANGSTROM
    return q, cell / BOHR_ANGSTROM


def _reference(model, cell, pbc):
    # Independent upstream MACE ASE path. The test compares the batched
    # graph's derivatives with a different neighbor-list/forward construction.
    calc = MACECalculator(
        models=deepcopy(model), device="cpu",
        default_dtype="float64", compute_stress=False,
    )
    from pyrpmd.potentials.mace_derivatives import MACEASEPotential
    return MACEASEPotential(
        ["D", "T", "H", "O"], calc, provider="mace",
        cell=cell, pbc=pbc,
    )


@pytest.mark.parametrize("pbc", [[False, False, False], [True, False, True]])
def test_mace_hessian_and_hvp_against_independent_ase_and_force_differences(pbc):
    q, cell = _structures()
    model = make_random_mace(dtype="float64")
    rng = np.random.default_rng(713)
    v = rng.normal(size=q.shape)
    symbols = ["D", "T", "H", "O"]
    with MACEBatchedPotential(symbols, deepcopy(model), dtype="float64",
                              cell=cell, pbc=pbc) as batched, \
         _reference(model, cell, pbc) as independent:
        upstream = independent.hessian(q)
        native = batched.hessian(q)
        hvp = batched.hessian_vector_product(q, v)
        ase_hvp = independent.hessian_vector_product(q, v)
        assert native.shape == (2, 12, 12)
        assert hvp.shape == q.shape
        assert upstream.dtype == hvp.dtype == np.float64
        np.testing.assert_allclose(native, native.transpose(0, 2, 1), rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(native, upstream, rtol=1e-7, atol=3e-9)
        expected = np.einsum("bij,bj->bi", native, v.reshape(2, -1)).reshape(q.shape)
        np.testing.assert_allclose(hvp, expected, rtol=1e-7, atol=3e-9)
        np.testing.assert_allclose(ase_hvp, expected, rtol=1e-7, atol=3e-9)
        epsilon = 2e-5
        plus, minus = q + epsilon * v, q - epsilon * v
        difference = -(batched.evaluate(plus).forces - batched.evaluate(minus).forces) / (2 * epsilon)
        np.testing.assert_allclose(hvp, difference, rtol=1e-3, atol=3e-7)
        # HVP never calls the dense Hessian method.
        batched.hessian = lambda *_: pytest.fail("HVP must stay matrix-free")
        np.testing.assert_allclose(batched.hessian_vector_product(q, v), expected,
                                   rtol=1e-7, atol=3e-9)


def test_mace_second_derivative_strict_failures():
    q, cell = _structures()
    model = make_random_mace(dtype="float64")
    with MACEBatchedPotential(["D", "T", "H", "O"], model, dtype="float64",
                              cell=cell) as provider:
        with pytest.raises(PotentialError, match="float64"):
            provider.hessian_vector_product(q, np.ones(q.shape, dtype=np.float32))
        with pytest.raises(PotentialError, match="matching coordinates"):
            provider.hessian_vector_product(q, np.ones((1, 4, 3)))
        with pytest.raises(PotentialError, match="float64"):
            provider.hessian_vector_product(q, np.full(q.shape, np.nan))
        with torch.inference_mode(), pytest.raises(PotentialError, match="inference_mode"):
            provider.hessian(q)
        with torch.no_grad():
            assert np.isfinite(provider.hessian_vector_product(q[:1], np.ones_like(q[:1]))).all()
    with pytest.raises(PotentialError, match="closed"):
        provider.hessian(q)
    with MACEBatchedPotential(["D", "T", "H", "O"],
                              make_random_mace(dtype="float32"), dtype="float32",
                              cell=cell) as provider:
        with pytest.raises(PotentialError, match="float64"):
            provider.hessian(q)


def test_ase_hessian_scales_and_rejects_unverified_precision():
    q, cell = _structures()
    model = make_random_mace(dtype="float64")
    with _reference(model, cell, [False] * 3) as provider:
        h = provider.hessian(q[:1])
        # The public hessian is AU; MACE get_hessian reports its model's
        # Cartesian energy/length units (usually eV/angstrom**2).
        provider.atoms.set_positions(q[0] * BOHR_ANGSTROM)
        reference = provider.atoms.calc.get_hessian(atoms=provider.atoms)
        np.testing.assert_allclose(
            h[0], np.asarray(reference).reshape(12, 12) *
            EV_HARTREE * BOHR_ANGSTROM**2, rtol=1e-10, atol=1e-12,
        )
        original = provider.atoms.calc.default_dtype
        try:
            provider.atoms.calc.default_dtype = "float32"
            with pytest.raises(PotentialError, match="float64"):
                provider.hessian(q[:1])
            with pytest.raises(PotentialError, match="float64"):
                provider.hessian_vector_product(q[:1], np.ones_like(q[:1]))
        finally:
            provider.atoms.calc.default_dtype = original


def test_instanton_action_hvp_matches_dense_action_and_rejects_missing_method():
    q, cell = _structures()
    path = np.repeat(q[:1], 8, axis=0)
    path[:, 0, 0] += 0.03 * np.cos(2 * np.pi * np.arange(8) / 8)
    v = np.random.default_rng(91).normal(size=path.shape)
    masses = np.array([1837., 3670., 5500., 29165.])
    with MACEBatchedPotential(["D", "T", "H", "O"], make_random_mace(),
                              cell=cell) as provider:
        action = RingPolymerAction(provider, masses, 200.0)
        dense = action.hessian(path)
        hvp = action.hessian_vector_product(path, v)
        np.testing.assert_allclose(hvp.ravel(), dense @ v.ravel(), rtol=1e-7, atol=1e-7)
        action.potential = object()
        with pytest.raises(ValueError, match="hessian_vector_product"):
            action.hessian_vector_product(path, v)
