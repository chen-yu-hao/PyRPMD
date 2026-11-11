"""Parallel MBAR bootstrap draws retain seeded serial semantics."""

import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.analysis.fes import mbar_rate_pmf
from pyrpmd.analysis import rate_pmf as rate_pmf_module
from pyrpmd.input import rate_pmf_options


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
        self.assertEqual(serial["bootstrap"]["workers_requested"], 1)
        self.assertEqual(parallel["bootstrap"]["workers_requested"], 2)
        self.assertEqual(parallel["bootstrap"]["workers"], 2)
        np.testing.assert_array_equal(
            serial["pmf_bootstrap_barriers"], parallel["pmf_bootstrap_barriers"]
        )
        np.testing.assert_allclose(
            serial["standard_error"], parallel["standard_error"], equal_nan=True
        )
        self.assertEqual(serial["bootstrap"]["draws"], parallel["bootstrap"]["draws"])

    def test_default_workers_remain_serial(self):
        result = mbar_rate_pmf(self.samples, **self.options)
        self.assertEqual(result["bootstrap"]["workers_requested"], 1)
        self.assertEqual(result["bootstrap"]["workers"], 1)
        self.assertEqual(result["bootstrap"]["backend"], "serial")

    def test_auto_mode_defers_unusable_draws_without_changing_the_pmf(self):
        short = [np.tile(np.arange(.1, 1., .2), 4)]
        baseline = mbar_rate_pmf(short, **{**self.options, "bootstrap": 0})
        with patch.object(rate_pmf_module, "_mbar_bootstrap_draw", side_effect=AssertionError("unusable draw")):
            result = mbar_rate_pmf(short, require_block_support=True, **self.options)
        assert result["ready"] == baseline["ready"]
        np.testing.assert_array_equal(result["free_energy"], baseline["free_energy"])
        assert result["bootstrap"]["attempted"] == 0
        assert result["bootstrap"]["skipped_reason"] == "insufficient_correlated_blocks"
        assert not result["barrier_uncertainty"]["available"]
        assert result["barrier_standard_error_hartree"] is None

    def test_auto_mode_uses_all_requested_draws_when_blocks_are_supported(self):
        regular = mbar_rate_pmf(self.samples, **self.options)
        automatic = mbar_rate_pmf(self.samples, require_block_support=True, **self.options)
        assert automatic["bootstrap"]["block_plan"]["status"] == "finite_block_estimate"
        assert automatic["bootstrap"]["attempted"] == self.options["bootstrap"]
        np.testing.assert_array_equal(automatic["pmf_bootstrap_barriers"], regular["pmf_bootstrap_barriers"])

    def test_newton_multiwindow_and_compare_ui_match_serial(self):
        rng = np.random.default_rng(42)
        centers = np.array([0.2, 0.5, 0.8])
        samples = [rng.normal(center, 0.12, 80) for center in centers]
        options = dict(
            centers=centers,
            stiffness=np.full(3, 8.0),
            beta=1.0,
            bins=np.linspace(-0.2, 1.2, 20),
            reference=0.2,
            xi_star=0.8,
            bootstrap=4,
            seed=7,
            min_count=1,
            min_weight_ess=None,
            min_endpoint_ess=None,
            max_center_displacement_sigma=None,
            compare_ui=True,
            ui_points=20,
        )
        serial = mbar_rate_pmf(samples, bootstrap_workers=1, **options)
        parallel = mbar_rate_pmf(samples, bootstrap_workers=2, **options)

        self.assertEqual(serial["solver"]["details"]["method"], "damped_newton")
        self.assertEqual(parallel["solver"]["details"]["method"], "damped_newton")
        self.assertEqual(serial["solver"]["unique_umbrella_hamiltonians"], 3)
        self.assertEqual(parallel["solver"]["unique_umbrella_hamiltonians"], 3)
        np.testing.assert_array_equal(
            serial["pmf_bootstrap_barriers"], parallel["pmf_bootstrap_barriers"]
        )
        np.testing.assert_array_equal(
            serial["standard_error"], parallel["standard_error"]
        )
        self.assertEqual(serial["bootstrap"]["draws"], parallel["bootstrap"]["draws"])
        ui_serial = serial["ui_comparison"]
        ui_parallel = parallel["ui_comparison"]
        np.testing.assert_array_equal(
            ui_serial["paired_ui_bootstrap_barriers"],
            ui_parallel["paired_ui_bootstrap_barriers"],
        )
        np.testing.assert_array_equal(
            ui_serial["paired_difference_bootstrap_hartree"],
            ui_parallel["paired_difference_bootstrap_hartree"],
        )

    def test_failure_draw_order_is_stable(self):
        original = rate_pmf_module._mbar_bootstrap_draw

        def draw_with_failure(task):
            draw, indices, context = task
            if draw == 2:
                broken_context = dict(context)

                def fail_solver(*args, **kwargs):
                    raise RuntimeError("injected bootstrap failure")

                broken_context["solve_mbar"] = fail_solver
                task = draw, indices, broken_context
            return original(task)

        with patch.object(rate_pmf_module, "_mbar_bootstrap_draw", draw_with_failure):
            serial = mbar_rate_pmf(self.samples, bootstrap_workers=1, **self.options)
            parallel = mbar_rate_pmf(self.samples, bootstrap_workers=2, **self.options)

        self.assertEqual(
            [record["draw"] for record in serial["bootstrap"]["draws"]], [0, 1, 2, 3]
        )
        self.assertEqual(
            [record["draw"] for record in parallel["bootstrap"]["draws"]], [0, 1, 2, 3]
        )
        self.assertEqual(
            [record["draw"] for record in serial["bootstrap"]["draws"] if record["valid"]], [0, 1, 3]
        )
        self.assertEqual(
            [record["draw"] for record in parallel["bootstrap"]["draws"] if record["valid"]], [0, 1, 3]
        )
        self.assertEqual(serial["bootstrap"]["draws"][2]["reason"], "mbar_solver_error")
        self.assertEqual(parallel["bootstrap"]["draws"][2]["reason"], "mbar_solver_error")
        self.assertEqual(
            [failure["draw"] for failure in serial["bootstrap"]["failures"]], [2]
        )
        self.assertEqual(
            [failure["draw"] for failure in parallel["bootstrap"]["failures"]], [2]
        )
        self.assertEqual(
            serial["bootstrap"]["failure_counts"], parallel["bootstrap"]["failure_counts"]
        )

    def test_memory_cap_records_requested_and_effective_workers(self):
        u_nbytes = sum(len(sample) for sample in self.samples) * 8
        available_memory = 2 * 4 * u_nbytes
        with patch.object(
            rate_pmf_module, "_available_memory_bytes", return_value=available_memory
        ):
            with self.assertWarns(RuntimeWarning):
                result = mbar_rate_pmf(
                    self.samples, bootstrap_workers=4, **self.options
                )

        self.assertEqual(result["bootstrap"]["workers_requested"], 4)
        self.assertEqual(result["bootstrap"]["workers"], 1)
        self.assertEqual(result["bootstrap"]["backend"], "serial")
        self.assertTrue(any("memory budget" in warning for warning in result["warnings"]))

    def test_worker_count_is_validated(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                mbar_rate_pmf(self.samples, bootstrap_workers=workers, **self.options)

    def test_config_worker_source_precedence(self):
        parallel = rate_pmf_options(
            {
                "rate": {"pmf_method": "mbar"},
                "analysis": {"bootstrap": 4},
                "parallel": {"trajectory_workers": 3},
            }
        )
        self.assertEqual(parallel["bootstrap_workers"], 3)
        self.assertEqual(parallel["bootstrap_workers_source"], "parallel")
        self.assertEqual(parallel["bootstrap_total_cores"], 3)

        explicit = rate_pmf_options(
            {
                "rate": {"pmf_method": "mbar"},
                "analysis": {"bootstrap": 4, "bootstrap_workers": 1},
                "parallel": {"trajectory_workers": 3},
            }
        )
        self.assertEqual(explicit["bootstrap_workers"], 1)
        self.assertEqual(explicit["bootstrap_workers_source"], "analysis")

        mpi = rate_pmf_options(
            {
                "rate": {"pmf_method": "mbar"},
                "analysis": {"bootstrap": 4},
                "parallel": {"backend": "mpi", "trajectory_workers": 3},
            }
        )
        self.assertEqual(mpi["bootstrap_workers"], 1)
        self.assertEqual(mpi["bootstrap_workers_source"], "default")

    def test_worker_count_is_truncated_by_bootstrap_draws(self):
        with patch.object(
            rate_pmf_module, "_available_memory_bytes", return_value=10**12
        ):
            result = mbar_rate_pmf(
                self.samples,
                bootstrap=2,
                bootstrap_workers=4,
                bootstrap_workers_source="parallel",
                **{key: value for key, value in self.options.items() if key != "bootstrap"},
            )
        self.assertEqual(result["bootstrap"]["workers_requested"], 4)
        self.assertEqual(result["bootstrap"]["workers"], 2)
        self.assertEqual(result["bootstrap"]["workers_source"], "parallel")
        self.assertFalse(result["bootstrap"]["memory_capped"])


if __name__ == "__main__":
    unittest.main()
