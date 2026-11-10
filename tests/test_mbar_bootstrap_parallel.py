"""Parallel MBAR bootstrap draws retain seeded serial semantics."""

import unittest

import numpy as np

from pyrpmd.analysis.fes import mbar_rate_pmf


class MBARBootstrapParallelTests(unittest.TestCase):
    def setUp(self):
        sample = np.tile(np.arange(0.1, 1.0, 0.2), 40)
        np.random.default_rng(51).shuffle(sample)
        self.samples = [sample]
        self.options = dict(
            centers=[0.5],
            stiffness=1e-12,
            beta=1.0,
            bins=np.linspace(0.0, 1.0, 6),
            reference=0.1,
            xi_star=0.7,
            bootstrap=4,
            seed=72,
            min_count=1,
            min_weight_ess=None,
            min_endpoint_ess=None,
            max_center_displacement_sigma=None,
        )

    def test_parallel_draws_match_serial_draws(self):
        serial = mbar_rate_pmf(self.samples, bootstrap_workers=1, **self.options)
        parallel = mbar_rate_pmf(self.samples, bootstrap_workers=2, **self.options)

        self.assertEqual(serial["bootstrap"]["backend"], "serial")
        self.assertEqual(parallel["bootstrap"]["backend"], "thread")
        self.assertEqual(parallel["bootstrap"]["workers"], 2)
        np.testing.assert_array_equal(
            serial["pmf_bootstrap_barriers"], parallel["pmf_bootstrap_barriers"]
        )
        np.testing.assert_allclose(
            serial["standard_error"], parallel["standard_error"], equal_nan=True
        )
        self.assertEqual(serial["bootstrap"]["draws"], parallel["bootstrap"]["draws"])

    def test_worker_count_is_validated(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                mbar_rate_pmf(self.samples, bootstrap_workers=workers, **self.options)


if __name__ == "__main__":
    unittest.main()
