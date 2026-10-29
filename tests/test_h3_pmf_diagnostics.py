"""Numerical safeguards for the independent saved-data H3 PMF diagnostic."""

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from pyrpmd.analysis.fes import umbrella_integration
from pyrpmd.potentials.python import import_object

MODULE = str(Path(__file__).resolve().parents[1] / "benchmarks/h3/pmf_diagnostics.py")
analyze_run = import_object(MODULE + ":analyze_run")
bracket = import_object(MODULE + ":_bracket")


class PMFDiagnosticTests(unittest.TestCase):
    def test_endpoint_requires_supported_bracketing_centers(self):
        grid = np.array([-0.1, 0.1, 0.3])
        mask = np.array([False, False, True])
        self.assertEqual(bracket(grid, mask, 0.0), (0, 1, 0.5))
        self.assertIsNone(bracket(grid, mask, -0.11))
        self.assertIsNone(bracket(grid, mask, 0.2))
        self.assertIsNone(bracket(grid, mask, 0.3))

    def test_saved_stream_pairing_conditional_rate_and_checkpoint_validation(self):
        rng = np.random.default_rng(991)
        centers = np.repeat([-0.1, 0.45, 1.0], 2)
        stiffness = np.full(len(centers), 12.0)
        arrays = [rng.normal(center, 0.23, 160) for center in centers]
        grid = np.linspace(0, 1.1, 101)
        xi = 0.85
        ui, _ = umbrella_integration(arrays, centers, stiffness, 1.0, grid)
        barrier = float(np.interp(xi, grid, ui) - ui[0])
        qtst = float(np.exp(-barrier))
        with tempfile.TemporaryDirectory() as directory:
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
                    {
                        "beta": 1.0,
                        "temperature_K": 1.0,
                        "analysis": {"min_count": 1},
                    }
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
                    # An earlier checkpoint must be an exact prefix of HDF5.
                    count = len(values) - 10 if i == 1 else len(values)
                    np.savez(path, values=values[:count, None], fingerprint=str(i))
            report = analyze_run(run, bins=(12, 20), bootstrap=3, seed=22)
            self.assertFalse(report["converged"])
            self.assertEqual(report["fixed_xi_star"], xi)
            self.assertEqual(report["mbar"]["unique_umbrella_hamiltonians"], 3)
            self.assertLess(
                report["mbar"]["aggregation_target_weight_maximum_absolute_difference"],
                1e-9,
            )
            self.assertEqual(report["window_moments"][1]["checkpoint_samples"], 150)
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


if __name__ == "__main__":
    unittest.main()
