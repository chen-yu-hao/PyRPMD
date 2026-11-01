"""Numerical CI-NEB checks against an analytic curved reaction surface."""

import unittest

import numpy as np

from pyrpmd.potentials.base import PESResult
from pyrpmd.workflows.neb import climbing_image_neb


class CurvedDoubleWell:
    """Minima (+/-1, 0, 0), saddle (0, 0.6, 0), barrier 1 hartree."""

    def evaluate(self, q):
        x, y, z = q[:, 0].T
        deviation = y - 0.6 * (1.0 - x * x)
        energy = (x * x - 1.0) ** 2 + 3.0 * deviation**2 + z * z
        force = np.zeros_like(q)
        force[:, 0, 0] = -4.0 * x * (x * x - 1.0) - 7.2 * x * deviation
        force[:, 0, 1] = -6.0 * deviation
        force[:, 0, 2] = -2.0 * z
        return PESResult(energy, force)


class ReusedBuffers(CurvedDoubleWell):
    def __init__(self):
        self.energy = np.empty(1)
        self.force = np.empty((1, 1, 3))

    def evaluate(self, q):
        result = super().evaluate(q)
        self.energy[:] = result.energy
        self.force[:] = result.forces
        return PESResult(self.energy, self.force)


class CINEBTests(unittest.TestCase):
    def setUp(self):
        self.reactant = np.array([[-1.0, 0.0, 0.0]])
        self.product = np.array([[1.0, 0.0, 0.0]])

    def test_curved_path_finds_saddle_off_initial_interpolation(self):
        original_reactant = self.reactant.copy()
        original_product = self.product.copy()
        # An even count avoids putting an initial image at the saddle's x.
        result = climbing_image_neb(
            CurvedDoubleWell(), self.reactant, self.product, images=8
        )
        self.assertTrue(result["converged"], result["diagnostics"])
        self.assertLess(result["iterations"], 1000)
        self.assertLessEqual(result["max_force"], 1e-4)
        self.assertLessEqual(result["max_climbing_force"], 1e-4)
        self.assertLess(result["directional_curvature"], -3.9)
        np.testing.assert_allclose(result["geometry"], [[0.0, 0.6, 0.0]], atol=2e-5)
        self.assertAlmostEqual(result["energy"], 1.0, places=8)
        np.testing.assert_array_equal(result["images"][0], original_reactant)
        np.testing.assert_array_equal(result["images"][-1], original_product)
        np.testing.assert_array_equal(self.reactant, original_reactant)
        np.testing.assert_array_equal(self.product, original_product)
        self.assertAlmostEqual(np.linalg.norm(result["tangent"]), 1.0)
        self.assertEqual(result["climbing_image"], 1 + np.argmax(result["energies"][1:-1]))

    def test_reused_provider_buffers_do_not_change_the_path(self):
        reference = climbing_image_neb(
            CurvedDoubleWell(), self.reactant, self.product, images=8
        )
        result = climbing_image_neb(
            ReusedBuffers(), self.reactant, self.product, images=8
        )
        self.assertTrue(result["converged"])
        np.testing.assert_array_equal(result["images"], reference["images"])
        np.testing.assert_array_equal(result["energies"], reference["energies"])

    def test_initial_path_is_copied_and_endpoint_match_is_required(self):
        path = np.linspace(self.reactant, self.product, 7)
        path[1:-1, 0, 1] = 0.3
        original = path.copy()
        result = climbing_image_neb(
            CurvedDoubleWell(), self.reactant, self.product, initial_path=path
        )
        self.assertTrue(result["converged"], result["diagnostics"])
        np.testing.assert_array_equal(path, original)
        path[0, 0, 1] = 0.01
        with self.assertRaisesRegex(ValueError, "endpoints must match"):
            climbing_image_neb(
                CurvedDoubleWell(), self.reactant, self.product, initial_path=path
            )

    def test_warmup_and_exhausted_runs_do_not_claim_convergence(self):
        warmup = climbing_image_neb(
            CurvedDoubleWell(), self.reactant, self.product, max_steps=1
        )
        self.assertFalse(warmup["converged"])
        self.assertIsNone(warmup["climbing_image"])
        self.assertEqual(warmup["iterations"], 1)
        climbing = climbing_image_neb(
            CurvedDoubleWell(), self.reactant, self.product,
            max_steps=1, climb_after=0,
        )
        self.assertFalse(climbing["converged"])
        self.assertIsNotNone(climbing["climbing_image"])
        self.assertGreater(climbing["max_climbing_force"], 1e-4)
        np.testing.assert_allclose(
            climbing["energies"],
            CurvedDoubleWell().evaluate(climbing["images"]).energy,
        )

    def test_stationary_minimum_and_flat_path_are_not_saddles(self):
        class Harmonic:
            def evaluate(self, q):
                return PESResult(np.sum(q * q, axis=(1, 2)), -2.0 * q)

        class Flat:
            def evaluate(self, q):
                return PESResult(np.zeros(len(q)), np.zeros_like(q))

        for potential in (Harmonic(), Flat()):
            with self.subTest(potential=type(potential).__name__):
                result = climbing_image_neb(
                    potential, self.reactant, self.product, images=3, climb_after=0
                )
                self.assertFalse(result["converged"])
                self.assertIn("negative curvature", result["diagnostics"][0])

    def test_shallow_barrier_passes_directional_sign_check(self):
        class ShallowDoubleWell:
            def evaluate(self, q):
                x, y, z = q[:, 0].T
                energy = 1e-10 * (x * x - 1.0) ** 2 + y * y + z * z
                force = np.zeros_like(q)
                force[:, 0, 0] = -4e-10 * x * (x * x - 1.0)
                force[:, 0, 1] = -2.0 * y
                force[:, 0, 2] = -2.0 * z
                return PESResult(energy, force)

        result = climbing_image_neb(
            ShallowDoubleWell(), self.reactant, self.product,
            images=3, climb_after=0,
        )
        self.assertTrue(result["converged"], result["diagnostics"])
        np.testing.assert_array_equal(result["geometry"], np.zeros((1, 3)))
        np.testing.assert_allclose(result["directional_curvature"], -4e-10, rtol=2e-8)
        self.assertEqual(result["max_climbing_force"], 0.0)

    def test_degenerate_and_malformed_paths_fail_before_pes_evaluation(self):
        class Unused:
            def evaluate(self, q):
                raise AssertionError("Malformed path reached the PES")

        invalid_endpoints = [
            (self.reactant, self.reactant),
            (self.reactant, np.ones((2, 3))),
            (self.reactant, np.array([[np.nan, 0, 0]])),
            (self.reactant, np.array([[1j, 0, 0]])),
            (np.empty((0, 3)), self.product),
            (np.array([0, 0, 0]), self.product),
        ]
        for start, end in invalid_endpoints:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                climbing_image_neb(Unused(), start, end)
        path = np.linspace(self.reactant, self.product, 7)
        path[2] = path[1]
        with self.assertRaisesRegex(ValueError, "coincident"):
            climbing_image_neb(Unused(), self.reactant, self.product, initial_path=path)
        with self.assertRaisesRegex(ValueError, "initial_path"):
            climbing_image_neb(
                Unused(), self.reactant, self.product, initial_path=path[:-1]
            )

    def test_invalid_options_are_not_silently_truncated(self):
        invalid = {
            "images": [2, True, 7.5],
            "max_steps": [0, True, 3.5],
            "climb_after": [-1, False, 2.5],
            "spring_constant": [0, -1, np.nan, np.inf, True],
            "gradient_tolerance": [0, -1, np.nan, np.inf, True],
            "max_step": [0, -1, np.nan, np.inf, True],
        }
        for option, values in invalid.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(ValueError, option):
                        climbing_image_neb(
                            CurvedDoubleWell(), self.reactant, self.product,
                            **{option: value},
                        )

    def test_pes_failure_is_not_treated_as_a_converged_band(self):
        class Failed:
            def evaluate(self, q):
                return PESResult(np.zeros(len(q)), np.zeros_like(q), np.ones(len(q), int))

        class Nonfinite:
            def evaluate(self, q):
                return PESResult(np.full(len(q), np.nan), np.zeros_like(q))

        for potential in (Failed(), Nonfinite()):
            with self.subTest(potential=type(potential).__name__):
                with self.assertRaises((ValueError, RuntimeError)):
                    climbing_image_neb(potential, self.reactant, self.product)


if __name__ == "__main__":
    unittest.main()
