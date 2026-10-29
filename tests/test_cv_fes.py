import unittest
import numpy as np
from pyrpmd.cv import (
    Distance,
    BondDifference,
    Angle,
    Cartesian,
    COMDistance,
    RPMDrateCoordinate,
    CVStack,
)
from pyrpmd.bias import UmbrellaBias, FluxMetricBias
from pyrpmd.analysis.fes import reconstruct_fes, solve_mbar, statistical_inefficiency


class CVTests(unittest.TestCase):
    def setUp(self):
        self.q = np.array(
            [
                [[-1.5, 0.1, 0.2], [0.1, 0.2, -0.1], [1.7, -0.3, 0.3]],
                [[-1.4, 0.2, 0.1], [0.0, 0.3, -0.2], [1.9, -0.1, 0.2]],
            ]
        )
        self.m = np.array([2.0, 3.0, 4.0])

    def check_jacobian(self, cv):
        analytic = cv.value_and_jacobian(self.q).jacobian
        numerical = np.empty_like(analytic)
        h = 1e-6
        for idx in np.ndindex(self.q.shape):
            plus = self.q.copy()
            minus = self.q.copy()
            plus[idx] += h
            minus[idx] -= h
            numerical[(slice(None),) + idx] = (cv.value(plus) - cv.value(minus)) / (
                2 * h
            )
        np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=1e-8)

    def test_centroid_and_bead_average_full_jacobians(self):
        for estimator in ("centroid", "bead_average"):
            for cv in (
                Distance([0, 1], estimator=estimator),
                BondDifference(atoms=[0, 1, 2], estimator=estimator),
                Angle([0, 1, 2], estimator=estimator),
                Cartesian(1, "z", estimator=estimator),
                COMDistance([[0], [1, 2]], self.m, estimator=estimator),
            ):
                with self.subTest(cv=type(cv).__name__, estimator=estimator):
                    self.check_jacobian(cv)

    def test_rpmdrate_gradient_hessian_and_flux_metric(self):
        cv = RPMDrateCoordinate(
            [[0], [1, 2]], self.m, self.q.mean(0), [[0, 1]], [[1, 2]], 10.0
        )
        self.check_jacobian(cv)
        centroid = self.q.mean(0)
        _, g, h = cv.centroid_value_gradient_hessian(centroid)
        hn = np.empty_like(h)
        delta = 1e-5
        for idx in np.ndindex(centroid.shape):
            plus = centroid.copy()
            minus = centroid.copy()
            plus[idx] += delta
            minus[idx] -= delta
            hn[:, :, idx[0], idx[1]] = (
                cv.centroid_value_gradient(plus)[1]
                - cv.centroid_value_gradient(minus)[1]
            ) / (2 * delta)
        np.testing.assert_allclose(h, hn, atol=1e-8, rtol=2e-6)
        bias = FluxMetricBias(cv, self.m, 2.0)
        force = bias.energy_force(self.q)[1]
        fn = np.empty_like(force)
        for idx in np.ndindex(self.q.shape):
            plus = self.q.copy()
            minus = self.q.copy()
            plus[idx] += delta
            minus[idx] -= delta
            fn[idx] = -(bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / (
                2 * delta
            )
        np.testing.assert_allclose(force, fn, atol=1e-8, rtol=2e-6)

    def test_umbrella_bead_scaling(self):
        cv = CVStack([Cartesian(0, 0)])
        for beads in (1, 3, 8):
            q = np.ones((beads, 1, 3)) * 2
            energy, force = UmbrellaBias(cv, [1.0], [4.0]).energy_force(q)
            self.assertEqual(energy, beads * 2.0)
            np.testing.assert_allclose(force[:, 0, 0], -4.0)
            np.testing.assert_array_equal(force[:, :, 1:], 0.0)


class FESAnalysisTests(unittest.TestCase):
    def test_mbar_2d_harmonic_density(self):
        rng = np.random.default_rng(441)
        centers = np.stack(
            np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], indexing="ij"), -1
        ).reshape(-1, 2)
        physical = np.diag([1.0, 0.7])
        bias = np.eye(2) * 2.0
        beta = 1.4
        covariance = np.linalg.inv(beta * (physical + bias))
        samples = [
            rng.multivariate_normal(
                np.linalg.solve(physical + bias, bias @ c), covariance, 6000
            )
            for c in centers
        ]
        result = reconstruct_fes(
            samples,
            centers,
            bias,
            beta,
            bins=[18, 18],
            ranges=[[-1.8, 1.8], [-1.8, 1.8]],
            bootstrap=3,
            min_count=30,
        )
        self.assertTrue(result["connected"])
        self.assertEqual(result["free_energy"].shape, (18, 18))
        x, y = np.meshgrid(
            *[0.5 * (e[1:] + e[:-1]) for e in result["edges"]], indexing="ij"
        )
        reference = 0.5 * (x * x + 0.7 * y * y)
        valid = ~result["mask"]
        difference = result["free_energy"][valid] - reference[valid]
        self.assertLess(float(np.std(difference)), 0.12)
        self.assertTrue(np.isfinite(result["standard_error"][valid]).any())
        np.testing.assert_allclose(result["overlap"].sum(1), 1.0, atol=1e-7)

    def test_disconnected_overlap_is_masked(self):
        rng = np.random.default_rng(123)
        samples = [rng.normal(-10, 0.03, (200, 1)), rng.normal(10, 0.03, (200, 1))]
        result = reconstruct_fes(
            samples, [[-10], [10]], [[100]], 1.0, bins=10, bootstrap=0
        )
        self.assertFalse(result["connected"])
        self.assertTrue(result["mask"].all())
        self.assertTrue(np.isnan(result["free_energy"]).all())

    def test_density_accounts_for_nonuniform_bin_volume(self):
        x = (np.arange(10000) + 0.5) / 10000
        # Bias is weak but nonzero, and exactly unweighted samples are used;
        # its negligible correction lets this test resolve density vs mass.
        r = reconstruct_fes(
            [x[:, None]], [[0.5]], [[1e-12]], 1.0, bins=[[0, 0.1, 0.4, 1]], bootstrap=0
        )
        np.testing.assert_allclose(r["density"], 1.0, atol=1e-10)
        np.testing.assert_allclose(r["probability_mass"], [0.1, 0.3, 0.6], atol=1e-10)

    def test_correlated_stream_reports_inefficiency(self):
        rng = np.random.default_rng(12)
        x = np.empty(10000)
        x[0] = 0
        for i in range(1, len(x)):
            x[i] = 0.9 * x[i - 1] + rng.normal()
        self.assertGreater(statistical_inefficiency(x), 10)


if __name__ == "__main__":
    unittest.main()
