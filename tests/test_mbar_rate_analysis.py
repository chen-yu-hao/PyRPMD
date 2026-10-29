"""Numerical rate-PMF checks against known distributions and covariance.

These use exact Boltzmann sampling by inverse CDF, not MD or mock chemical
potentials. The multinomial test checks paired-endpoint uncertainty against
its independent analytic large-sample prediction.
"""

import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid

from pyrpmd.analysis.fes import (
    mbar_rate_pmf,
    solve_mbar,
    umbrella_reduced_potentials,
)


class MBARSolverTests(unittest.TestCase):
    def test_duplicate_hamiltonians_gauge_and_legacy_equivalence(self):
        rng = np.random.default_rng(542)
        centers = np.array([-1.0, 0.0, -1.0, 1.0])
        counts = np.array([1100, 900, 800, 1200])
        beta, stiffness = 1.4, 3.0
        x = np.concatenate(
            [
                rng.normal(
                    stiffness * c / (1 + stiffness),
                    1 / np.sqrt(beta * (1 + stiffness)),
                    n,
                )
                for c, n in zip(centers, counts)
            ]
        )
        u = umbrella_reduced_potentials(
            x[:, None], centers[:, None], [[stiffness]], beta
        )
        fit = solve_mbar(u, counts, tolerance=1e-11)
        legacy = solve_mbar(
            u, counts, tolerance=1e-11, method="legacy", combine_identical_states=False
        )
        self.assertTrue(fit.converged and legacy.converged)
        self.assertLess(fit.residual, 1e-11)
        self.assertEqual(fit.diagnostics["unique_states"], 3)
        for field in ("weights", "free_energies", "state_weights", "overlap"):
            np.testing.assert_allclose(
                getattr(fit, field), getattr(legacy, field), rtol=2e-8, atol=2e-9
            )
        np.testing.assert_allclose(fit.overlap.sum(1), 1.0, atol=1e-9)
        # State constants change free-energy gauges, not physical target weights.
        shifts = np.array([20.0, -5.0, 7.0, 1.0])
        shifted = solve_mbar(u + shifts[:, None], counts, tolerance=1e-11)
        np.testing.assert_allclose(shifted.weights, fit.weights, rtol=2e-8, atol=1e-11)
        warm = solve_mbar(
            u, counts, initial_free_energies=fit.free_energies, tolerance=1e-11
        )
        np.testing.assert_allclose(warm.weights, fit.weights, rtol=1e-9, atol=1e-12)
        self.assertTrue(warm.diagnostics["warm_start"])

    def test_disconnected_and_invalid_counts_are_not_hidden(self):
        x = np.r_[np.full(50, -10.0), np.full(50, 10.0)]
        u = 100 * (x[None] - np.array([-10.0, 10.0])[:, None]) ** 2
        fit = solve_mbar(u, [50, 50])
        self.assertTrue(fit.converged)
        self.assertEqual(len(np.unique(fit.components)), 2)
        for counts in ([49.5, 50.5], [True, True], [0, 100]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                solve_mbar(u, counts)
        with self.assertRaises(ValueError):
            solve_mbar(np.empty((0, 0)), [])


class MBARRatePMFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # V(x)=1.2*x^2*(x-2)^2 has minima at 0 and 2, and exactly
        # 1.2 hartree barrier at x=1. Its biased windows are non-Gaussian.
        beta, k = 1.3, 8.0
        centers = np.repeat(np.linspace(-0.2, 2.2, 9), 2)
        grid = np.linspace(-1.0, 3.0, 60001)
        physical = 1.2 * grid**2 * (grid - 2) ** 2
        rng = np.random.default_rng(7189)
        samples = []
        for center in centers:
            density = np.exp(-beta * (physical + 0.5 * k * (grid - center) ** 2))
            cdf = cumulative_trapezoid(density, grid, initial=0)
            cdf /= cdf[-1]
            samples.append(np.interp(rng.random(1700), cdf, grid))
        cls.samples, cls.centers, cls.beta, cls.k = samples, centers, beta, k

    def test_known_nongaussian_barrier_and_same_draw_ui_comparison(self):
        result = mbar_rate_pmf(
            self.samples,
            self.centers,
            self.k,
            self.beta,
            xi_star=1.0,
            reference=0.0,
            bins=64,
            ranges=(-0.6, 2.6),
            bootstrap=24,
            seed=941,
            compare_ui=True,
            min_count=10,
        )
        self.assertTrue(result["ready"])
        self.assertLess(abs(result["free_energy_barrier_hartree"] - 1.2), 0.10)
        self.assertEqual(result["solver"]["unique_umbrella_hamiltonians"], 9)
        self.assertEqual(len(result["window_moments"]), 18)
        self.assertEqual(result["measure"], "rate_flux")
        self.assertGreaterEqual(result["bootstrap"]["valid"], 22)
        self.assertEqual(result["bootstrap"]["attempted"], 24)
        endpoints = result["bootstrap"]["endpoint_values_hartree"]
        barriers = np.asarray(result["pmf_bootstrap_barriers"])
        np.testing.assert_allclose(endpoints[:, 1] - endpoints[:, 0], barriers)
        covariance = result["bootstrap"]["endpoint_covariance_hartree_squared"]
        expected_variance = np.array([-1.0, 1.0]) @ covariance @ np.array([-1.0, 1.0])
        self.assertAlmostEqual(
            result["barrier_standard_error_hartree"] ** 2, expected_variance, places=12
        )
        ui = result["ui_comparison"]
        self.assertTrue(ui["ready"])
        np.testing.assert_allclose(
            ui["paired_difference_bootstrap_hartree"],
            np.asarray(ui["paired_mbar_bootstrap_barriers"])
            - ui["paired_ui_bootstrap_barriers"],
        )
        # Each comparison is tied to the same original draw ID and endpoint.
        valid_records = [r for r in result["bootstrap"]["draws"] if r["valid"]]
        np.testing.assert_allclose(
            [r["ui_barrier_hartree"] for r in valid_records],
            ui["paired_ui_bootstrap_barriers"],
        )

    def test_auto_selects_supported_unsmoothed_peak_and_holds_it_fixed(self):
        result = mbar_rate_pmf(
            self.samples,
            self.centers,
            self.k,
            self.beta,
            search_range=(0.0, 1.8),
            bins=64,
            ranges=(-0.6, 2.6),
            bootstrap=3,
            seed=222,
        )
        self.assertTrue(result["ready"])
        self.assertLess(abs(result["xi_star"] - 1.0), 0.16)
        self.assertIn(result["xi_star"], result["xi"])
        self.assertEqual(result["support"]["smoothing"], "none")
        self.assertTrue(result["bootstrap"]["surface_held_fixed"])
        self.assertNotIn("xi_star", result["bootstrap"]["draws"][0])

    def test_multinomial_endpoint_covariance_and_nonuniform_density(self):
        n, beta = 12000, 2.0
        x = (np.arange(n) + 0.5) / n
        np.random.default_rng(81).shuffle(x)
        result = mbar_rate_pmf(
            [x],
            [0.5],
            1e-12,
            beta,
            bins=[0.0, 0.1, 0.4, 0.7, 1.0],
            reference=0.05,
            xi_star=0.85,
            bootstrap=400,
            seed=75,
            min_count=5,
        )
        self.assertTrue(result["ready"])
        self.assertAlmostEqual(result["free_energy_barrier_hartree"], 0.0, places=10)
        np.testing.assert_allclose(result["density"], 1.0, atol=1e-10)
        covariance = result["bootstrap"]["endpoint_covariance_hartree_squared"]
        # Multinomial counts: Cov(log(p_i),log(p_j)) ~= -1/N for i!=j.
        expected_cross = -1 / (n * beta**2)
        self.assertLess(abs(covariance[0, 1] / expected_cross - 1), 0.50)
        expected_barrier_variance = (1 / 0.1 + 1 / 0.3) / (n * beta**2)
        measured_variance = result["barrier_standard_error_hartree"] ** 2
        self.assertLess(abs(measured_variance / expected_barrier_variance - 1), 0.25)
        self.assertGreater(measured_variance, np.trace(covariance))

    def test_mask_gap_and_extrapolated_reference_are_rejected(self):
        x = np.r_[np.full(100, -1.0), np.full(100, 1.0)]
        options = dict(bins=np.linspace(-1.2, 1.2, 7), reference=-1.0, bootstrap=0)
        fixed = mbar_rate_pmf([x], [0.0], 1e-12, 1.0, xi_star=1.0, **options)
        self.assertFalse(fixed["ready"])
        self.assertEqual(fixed["status"], "masked_interval_between_endpoints")
        self.assertIsNone(fixed["barrier_standard_error_hartree"])
        auto = mbar_rate_pmf(
            [x], [0.0], 1e-12, 1.0, search_range=(-1.0, 1.0), **options
        )
        self.assertEqual(auto["status"], "masked_search_interval")
        uniform = np.linspace(0, 1, 1000)
        outside = mbar_rate_pmf(
            [uniform],
            [0.5],
            1e-12,
            1.0,
            bins=np.linspace(0, 1, 11),
            reference=0.0,
            xi_star=0.5,
            bootstrap=2,
        )
        self.assertFalse(outside["ready"])
        self.assertEqual(outside["status"], "reference_outside_histogram_centers")
        self.assertEqual(outside["bootstrap"]["attempted"], 0)
        self.assertIsNone(outside["barrier_standard_error_hartree"])

    def test_each_failed_resample_has_a_reason_and_missing_error_is_none(self):
        x = np.r_[np.full(120, 0.5), 1.5, np.full(120, 2.5)]
        np.random.default_rng(7).shuffle(x)
        result = mbar_rate_pmf(
            [x],
            [1.5],
            1e-12,
            1.0,
            bins=[0.0, 1.0, 2.0, 3.0],
            reference=0.5,
            xi_star=2.5,
            bootstrap=30,
            seed=8,
            min_count=1,
        )
        self.assertTrue(result["ready"])
        boot = result["bootstrap"]
        self.assertEqual(boot["attempted"], 30)
        self.assertEqual(boot["valid"] + len(boot["failures"]), 30)
        self.assertGreater(len(boot["failures"]), 0)
        self.assertTrue(
            all(
                f["reason"] == "masked_interval_between_endpoints"
                for f in boot["failures"]
            )
        )
        self.assertEqual(sum(boot["failure_counts"].values()), len(boot["failures"]))
        no_error = mbar_rate_pmf(
            [x],
            [1.5],
            1e-12,
            1.0,
            bins=[0.0, 1.0, 2.0, 3.0],
            reference=0.5,
            xi_star=2.5,
            bootstrap=0,
            min_count=1,
        )
        self.assertIsNone(no_error["barrier_standard_error_hartree"])
        self.assertIsNone(no_error["barrier_percentile_95_ci_hartree"])
        self.assertTrue(np.isnan(no_error["standard_error"]).all())
        one_draw = mbar_rate_pmf(
            [x],
            [1.5],
            1e-12,
            1.0,
            bins=[0.0, 1.0, 2.0, 3.0],
            reference=0.5,
            xi_star=2.5,
            bootstrap=2,
            seed=0,
            min_count=1,
        )
        self.assertEqual(one_draw["bootstrap"]["valid"], 1)
        self.assertIsNone(one_draw["barrier_standard_error_hartree"])
        self.assertIsNone(one_draw["barrier_percentile_95_ci_hartree"])

    def test_disconnected_rate_and_wrong_measure_do_not_produce_a_barrier(self):
        rng = np.random.default_rng(23)
        streams = [rng.normal(-10.0, 0.03, 100), rng.normal(10.0, 0.03, 100)]
        result = mbar_rate_pmf(
            streams,
            [-10.0, 10.0],
            100.0,
            1.0,
            reference=-10.0,
            xi_star=10.0,
            bins=20,
            bootstrap=0,
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["status"], "disconnected_overlap")
        self.assertIsNone(result["free_energy_barrier_hartree"])
        self.assertTrue(result["mask"].all())
        with self.assertRaisesRegex(ValueError, "rate_flux"):
            mbar_rate_pmf(streams, [-10.0, 10.0], 100.0, 1.0, measure="cv_density")


if __name__ == "__main__":
    unittest.main()
