"""Analytical saddle tests independent of an electronic-structure backend."""

import unittest
from pathlib import Path

import numpy as np

from pyrpmd.potentials import PESResult
from pyrpmd.workflows.saddle import refine_transition_state


class Quadratic:
    def __init__(self, hessian, center, *, reuse=False):
        self.hessian = np.asarray(hessian, dtype=float)
        self.center = np.asarray(center, dtype=float)
        self.reuse = reuse
        self.result = PESResult(np.empty(1), np.empty((1, *self.center.shape)))

    def evaluate(self, q):
        delta = (q - self.center).reshape(len(q), -1)
        gradient = delta @ self.hessian
        energy = 0.5 * np.sum(delta * gradient, axis=1)
        forces = -gradient.reshape(q.shape)
        if not self.reuse:
            return PESResult(energy, forces)
        self.result.energy[:] = energy
        self.result.forces[:] = forces
        return self.result


class CurvedDoubleWell:
    """V=(x*x-1)**2 + (y-0.3*x*x)**2 + z*z/2."""

    def evaluate(self, q):
        x, y, z = q[:, 0, :].T
        transverse = y - 0.3 * x * x
        energy = (x * x - 1) ** 2 + transverse**2 + 0.5 * z * z
        gradient = np.column_stack(
            [4 * x * (x * x - 1) - 1.2 * x * transverse, 2 * transverse, z]
        )
        return PESResult(energy, -gradient[:, None, :])


class TransitionStateRefinementTests(unittest.TestCase):
    def test_rotated_quadratic_refines_an_off_saddle_guess_and_owns_buffers(self):
        rng = np.random.default_rng(34)
        rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        hessian = rotation @ np.diag([-0.4, 0.2, 0.3, 0.7, 0.9, 1.2]) @ rotation.T
        center = np.array([[-1.2, 0.1, 0.5], [1.1, -0.2, 0.4]])
        guess = center + rng.normal(scale=0.15, size=center.shape)
        original = guess.copy()
        results = []
        for reuse in (False, True):
            provider = Quadratic(hessian, center, reuse=reuse)
            result = refine_transition_state(provider, guess, gradient_tolerance=1e-9)
            self.assertTrue(result["converged"], result["diagnostics"])
            self.assertGreater(result["iterations"], 0)
            self.assertLessEqual(result["max_force"], 1e-9)
            np.testing.assert_allclose(result["geometry"], center, atol=1e-8)
            np.testing.assert_allclose(
                result["hessian_eigenvalues"],
                [-0.4, 0.2, 0.3, 0.7, 0.9, 1.2],
                atol=1e-10,
            )
            results.append(result)
        np.testing.assert_array_equal(guess, original)
        np.testing.assert_allclose(
            results[0]["geometry"], results[1]["geometry"], atol=1e-13
        )

    def test_curved_reaction_refines_from_both_sides(self):
        for x in (-0.4, 0.4):
            result = refine_transition_state(
                CurvedDoubleWell(), [[x, 0.25, -0.3]], gradient_tolerance=1e-8
            )
            self.assertTrue(result["converged"], result["diagnostics"])
            np.testing.assert_allclose(result["geometry"], 0, atol=1e-8)
            self.assertAlmostEqual(result["energy"], 1.0, places=10)

    def test_molecular_teaching_pes_with_rigid_body_zero_modes(self):
        from pyrpmd.potentials.python import import_object

        source = (
            Path(__file__).resolve().parents[1] / "examples" / "rate" / "toy_exchange.py"
        )
        provider = import_object(str(source) + ":build")()
        distance = 1.2 + np.log(3) / 1.2
        saddle = np.array([[-distance, 0, 0], [0, 0, 0], [distance, 0, 0]])
        guess = saddle + np.random.default_rng(14).normal(scale=0.02, size=saddle.shape)
        result = refine_transition_state(
            provider, guess, masses=[1836.0] * 3, gradient_tolerance=1e-8
        )
        self.assertTrue(result["converged"], result["diagnostics"])
        final = result["geometry"]
        for endpoint in (0, 2):
            self.assertAlmostEqual(
                np.linalg.norm(final[endpoint] - final[1]), distance, places=6
            )
        self.assertAlmostEqual(result["energy"], -4 * 0.08**2 / 0.48, places=9)

    def test_initial_direction_tracks_reaction_from_positive_curvature(self):
        result = refine_transition_state(
            CurvedDoubleWell(),
            [[0.9, 0.243, 0.05]],
            initial_mode=[[1.0, 0.0, 0.0]],
            gradient_tolerance=1e-8,
        )
        self.assertTrue(result["converged"], result["diagnostics"])
        np.testing.assert_allclose(result["geometry"], 0, atol=1e-8)

    def test_stationary_minimum_and_higher_index_saddle_are_not_success(self):
        for eigenvalues, count in (([1, 2, 3], 0), ([-2, -1, 3], 2)):
            result = refine_transition_state(
                Quadratic(np.diag(eigenvalues), [[0, 0, 0]]), [[0, 0, 0]]
            )
            self.assertFalse(result["converged"])
            self.assertEqual(result["iterations"], 0)
            self.assertIn(f"{count} resolved unstable modes", result["diagnostics"][0])

    def test_step_exhaustion_does_not_report_a_rate_ready_saddle(self):
        result = refine_transition_state(
            Quadratic(np.diag([-1.0, 2.0, 3.0]), [[0, 0, 0]]),
            [[1.0, 1.0, 1.0]],
            max_steps=1,
            max_step=0.01,
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 1)
        self.assertIn("max_steps", result["diagnostics"][0])
        self.assertLessEqual(
            np.linalg.norm(result["geometry"][0] - np.ones(3)), 0.010000000001
        )
        force = -result["geometry"] * np.array([-1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["max_force"], float(np.max(np.abs(force))))

    def test_zero_curvature_modes_are_preserved(self):
        provider = Quadratic(np.diag([-0.4, 0.2, 0.0]), [[0, 0, 0]])
        result = refine_transition_state(provider, [[0.2, 0.3, 12.0]])
        self.assertTrue(result["converged"], result["diagnostics"])
        np.testing.assert_allclose(result["geometry"], [[0, 0, 12]], atol=1e-5)

    def test_mass_weighted_negative_mode_threshold_is_honored(self):
        provider = Quadratic(np.diag([-1e-4, 0.2, 0.3]), [[0, 0, 0]])
        result = refine_transition_state(provider, [[0, 0, 0]], masses=[1e6])
        self.assertFalse(result["converged"])
        self.assertAlmostEqual(result["hessian_eigenvalues"][0], -1e-10)
        result = refine_transition_state(
            provider,
            [[0, 0, 0]],
            masses=[1e6],
            negative_eigenvalue_tolerance=1e-12,
        )
        self.assertTrue(result["converged"])

    def test_invalid_options_and_inputs_fail_before_evaluation(self):
        class Unevaluated:
            def evaluate(self, q):
                raise AssertionError("invalid input must not reach PES")

        for key in (
            "gradient_tolerance",
            "hessian_step",
            "max_step",
            "negative_eigenvalue_tolerance",
        ):
            for value in (0, -0.2, np.nan, np.inf, True, "bad"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    refine_transition_state(Unevaluated(), [[0, 0, 0]], **{key: value})
        for value in (True, 0, -1, 1.5):
            with self.subTest(max_steps=value), self.assertRaises(ValueError):
                refine_transition_state(Unevaluated(), [[0, 0, 0]], max_steps=value)
        for geometry in ([], [0, 0, 0], [[0, np.nan, 0]], [[1j, 0, 0]], [[True] * 3]):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                refine_transition_state(Unevaluated(), geometry)
        for masses in ([], [0], [-1], [np.inf], [1j], [True], [1, 2]):
            with self.subTest(masses=masses), self.assertRaises(ValueError):
                refine_transition_state(Unevaluated(), [[0, 0, 0]], masses=masses)
        for mode in ([[0, 0, 0]], [[np.nan, 0, 0]], [[1, 0, 0], [0, 0, 0]]):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                refine_transition_state(Unevaluated(), [[0, 0, 0]], initial_mode=mode)

    def test_pes_failures_are_not_misreported_as_convergence(self):
        class Failed:
            def evaluate(self, q):
                return PESResult(
                    np.zeros(1), np.zeros_like(q), status=np.ones(1, dtype=int)
                )

        with self.assertRaises(ValueError):
            refine_transition_state(Failed(), [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
