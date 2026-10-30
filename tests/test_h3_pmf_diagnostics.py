"""Numerical safeguards for the independent saved-data H3 PMF diagnostic."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from pyrpmd.analysis.fes import umbrella_integration
from pyrpmd.potentials.python import import_object

MODULE = str(Path(__file__).resolve().parents[1] / "benchmarks/h3/pmf_diagnostics.py")
analyze_run = import_object(MODULE + ":analyze_run")
bracket = import_object(MODULE + ":_bracket")
statistics = import_object(MODULE + ":_statistics")


def _saved_run(directory):
    """A supported overlapping ensemble, with one earlier checkpoint prefix."""
    rng = np.random.default_rng(991)
    centers = np.repeat([-0.1, 0.45, 1.0], 2)
    stiffness = np.full(len(centers), 12.0)
    arrays = [rng.normal(center, 0.23, 160) for center in centers]
    grid = np.linspace(0, 1.1, 101)
    xi = 0.85
    ui, _ = umbrella_integration(arrays, centers, stiffness, 1.0, grid)
    barrier = float(np.interp(xi, grid, ui) - ui[0])
    qtst = float(np.exp(-barrier))
    run = Path(directory)
    (run / "umbrella/checkpoints").mkdir(parents=True)
    (run / "input").mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "pmf": {
                    "xi_star": xi,
                    "free_energy_barrier_hartree": barrier,
                    "k_qtst_cm3_s": qtst,
                },
                "kappa": 0.5,
                "rate_cm3_s": 0.5 * qtst,
                "rate_estimate_cm3_s": 0.5 * qtst,
            }
        )
    )
    (run / "input/resolved.json").write_text(
        json.dumps(
            {"beta": 1.0, "temperature_K": 1.0, "analysis": {"min_count": 1}}
        )
    )
    np.savez(run / "rate_pmf.npz", xi=grid)
    with h5py.File(run / "umbrella/fes.h5", "w") as h:
        h.attrs["measure"] = "rate_flux"
        h["edges_0"] = np.linspace(-0.9, 1.8, 31)
        groups = h.create_group("streams")
        for i, values in enumerate(arrays):
            window, replica = divmod(i, 2)
            group = groups.create_group(str(i))
            group.attrs["window"] = window
            group.attrs["replica"] = replica
            group["samples"] = values[:, None]
            group["center"] = [centers[i]]
            group["stiffness"] = [[stiffness[i]]]
            path = (
                run
                / "umbrella/checkpoints"
                / (f"window_{window:05d}_replica_{replica:04d}.npz")
            )
            count = len(values) - 10 if i == 1 else len(values)
            np.savez(path, values=values[:count, None], fingerprint=str(i))
    return run, xi


class PMFDiagnosticTests(unittest.TestCase):
    def test_endpoint_requires_supported_bracketing_centers(self):
        grid = np.array([-0.1, 0.1, 0.3])
        mask = np.array([False, False, True])
        self.assertEqual(bracket(grid, mask, 0.0), (0, 1, 0.5))
        self.assertIsNone(bracket(grid, mask, -0.11))
        self.assertIsNone(bracket(grid, mask, 0.2))
        self.assertIsNone(bracket(grid, mask, 0.3))

    def test_saved_stream_pairing_conditional_rate_and_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            run, xi = _saved_run(directory)
            report = analyze_run(run, bins=(12, 20), bootstrap=3, seed=22)
            self.assertFalse(report["converged"])
            self.assertEqual(report["fixed_xi_star"], xi)
            self.assertEqual(report["mbar"]["unique_umbrella_hamiltonians"], 3)
            self.assertLess(
                report["mbar"]["aggregation_target_weight_maximum_absolute_difference"],
                1e-9,
            )
            self.assertEqual(report["window_moments"][1]["checkpoint_samples"], 150)
            plan = report["bootstrap"]["block_plan"]
            for index, moment in enumerate(report["window_moments"]):
                requested = int(np.ceil(5 * moment["statistical_inefficiency"]))
                self.assertEqual(moment["requested_block_length"], requested)
                self.assertEqual(moment["block_length"], min(requested, 80))
                self.assertEqual(plan["block_lengths"][index], moment["block_length"])
            self.assertEqual(report["warnings"], plan["warnings"])
            self.assertFalse(plan["mixing_established"])
            for entry in report["per_bins"]:
                self.assertTrue(entry["endpoints_supported"])
                self.assertEqual(entry["barrier_bootstrap_hartree"]["replicates"], 3)
                expected = 0.5 * np.exp(-entry["barrier_hartree"])
                self.assertAlmostEqual(entry["same_kappa_rate_cm3_s"], expected)
                # Retaining same-draw differences includes cross-estimator and
                # within-barrier covariance, rather than adding marginal SEs.
                paired = np.asarray(entry["paired_mbar_minus_ui_draws_hartree"])
                raw = np.asarray(entry["bootstrap_barriers_hartree"])
                np.testing.assert_allclose(
                    paired, raw - report["ui"]["bootstrap_barriers_hartree"]
                )
                self.assertAlmostEqual(
                    entry["paired_mbar_minus_ui_bootstrap_hartree"]["standard_error"],
                    paired.std(ddof=1),
                )
            json.dumps(report, allow_nan=False)
            corrupt = run / "umbrella/checkpoints/window_00000_replica_0000.npz"
            np.savez(corrupt, values=np.zeros((160, 1)), fingerprint="0")
            with self.assertRaisesRegex(ValueError, "HDF5/checkpoint samples differ"):
                analyze_run(run, bins=(12,), bootstrap=0)

    def test_block_support_controls_all_uncertainty_outputs(self):
        # Control only the correlation estimate; keep identical samples and
        # point estimates so unavailable uncertainty cannot masquerade as zero.
        cases = (
            (1.0, "finite_block_estimate", 5, False),
            (40.0, "limited_blocks", 80, True),
            (160.0, "unavailable", 80, True),
        )
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _saved_run(directory)
            reference = analyze_run(run, bins=(12,), bootstrap=0)
            resample = analyze_run.__globals__["_block_resample"]
            for g, status, length, capped in cases:
                with self.subTest(status=status):
                    used_blocks = []

                    def record_block(values, block, rng):
                        used_blocks.append(int(block))
                        return resample(values, block, rng)

                    with patch.dict(
                        analyze_run.__globals__,
                        {
                            "statistical_inefficiency": lambda values: g,
                            "_block_resample": record_block,
                        },
                    ):
                        report = analyze_run(run, bins=(12,), bootstrap=3, seed=22)
                    plan = report["bootstrap"]["block_plan"]
                    self.assertEqual(plan["status"], status)
                    self.assertEqual(plan["requested_block_lengths"], [int(5 * g)] * 6)
                    self.assertEqual(plan["block_lengths"], [length] * 6)
                    self.assertEqual(plan["block_length_capped"], [capped] * 6)
                    self.assertEqual(plan["blocks_per_stream"], [160 / length] * 6)
                    self.assertEqual(used_blocks, [length] * 18)
                    self.assertEqual(bool(report["warnings"]), status != "finite_block_estimate")
                    self.assertEqual(report["warnings"], plan["warnings"])
                    self.assertEqual(report["ui"]["barrier_hartree"], reference["ui"]["barrier_hartree"])
                    entry, baseline = report["per_bins"][0], reference["per_bins"][0]
                    self.assertEqual(entry["barrier_hartree"], baseline["barrier_hartree"])
                    self.assertEqual(entry["same_kappa_rate_cm3_s"], baseline["same_kappa_rate_cm3_s"])
                    self.assertEqual(len(entry["bootstrap_barriers_hartree"]), 3)
                    self.assertEqual(len(report["ui"]["bootstrap_barriers_hartree"]), 3)
                    self.assertEqual(len(entry["paired_mbar_minus_ui_draws_hartree"]), 3)
                    for stats in (
                        report["ui"]["paired_bootstrap_hartree"],
                        entry["barrier_bootstrap_hartree"],
                        entry["paired_mbar_minus_ui_bootstrap_hartree"],
                    ):
                        self.assertEqual(stats["replicates"], 3)
                        if status == "unavailable":
                            self.assertIsNone(stats["standard_error"])
                            self.assertIsNone(stats["percentile_95_ci"])
                        else:
                            self.assertIsNotNone(stats["standard_error"])
                            self.assertIsNotNone(stats["percentile_95_ci"])
                    if status == "unavailable":
                        self.assertIsNone(entry["conditional_rate_percentile_95_ci_cm3_s"])
                    else:
                        self.assertIsNotNone(entry["conditional_rate_percentile_95_ci_cm3_s"])
                    json.dumps(report, allow_nan=False)

    def test_single_bootstrap_draw_is_not_an_interval(self):
        for values in ([], [2.0]):
            with self.subTest(values=values):
                stats = statistics(values)
                self.assertEqual(stats["replicates"], len(values))
                self.assertIsNone(stats["standard_error"])
                self.assertIsNone(stats["percentile_95_ci"])

    def test_numerical_ui_bootstrap_failure_preserves_mbar_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            run, _ = _saved_run(directory)
            original = analyze_run.__globals__["umbrella_integration"]
            calls = 0

            def fail_bootstrap(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return original(*args, **kwargs)
                raise FloatingPointError("nonfinite Gaussian UI curve")

            with patch.dict(analyze_run.__globals__, {"umbrella_integration": fail_bootstrap}):
                report = analyze_run(run, bins=(12,), bootstrap=3, seed=22)
            self.assertEqual(report["bootstrap"]["failures"], {"mbar": 0, "ui": 3})
            self.assertEqual(report["ui"]["bootstrap_barriers_hartree"], [])
            self.assertEqual(report["ui"]["paired_bootstrap_hartree"]["replicates"], 0)
            self.assertIsNone(report["ui"]["paired_bootstrap_hartree"]["standard_error"])
            self.assertEqual(report["per_bins"][0]["barrier_bootstrap_hartree"]["replicates"], 3)
            self.assertEqual(report["per_bins"][0]["paired_mbar_minus_ui_draws_hartree"], [])


if __name__ == "__main__":
    unittest.main()
