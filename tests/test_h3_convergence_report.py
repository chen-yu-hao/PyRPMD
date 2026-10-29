"""Aggregate-report schema fixtures only: no dynamics or physical validation."""

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.potentials.python import import_object

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "benchmarks/h3/convergence_report.py"
aggregate = import_object(str(REPORT) + ":aggregate")
write_report = import_object(str(REPORT) + ":write_report")
main = import_object(str(REPORT) + ":main")
settings_for = import_object(
    str(ROOT / "benchmarks/h3/convergence.py") + ":case_settings"
)
budget_for = import_object(
    str(ROOT / "benchmarks/h3/convergence.py") + ":compute_budget"
)
gates_for = import_object(str(ROOT / "benchmarks/h3/convergence.py") + ":quality_gates")
CASES = ["k400-a", "k400-b", "k800-a", "k800-b"]


def make_fixture(directory, case):
    """Synthetic numbers for parser/contract tests, never research evidence."""
    settings = settings_for(case)
    budget = budget_for(settings)
    shift = 0.0002 * CASES.index(case)
    mbar, ui = 0.030 + shift, 0.027 + shift / 2
    angles = np.arange(40) * 2 * np.pi / 40
    mb = mbar + 0.001 * np.sin(angles)
    ub = ui + 0.0012 * np.cos(angles)
    differences = mb - ub
    pmf = {
        "method": "mbar",
        "xi_star": settings["xi_star"],
        "reference": 0.0,
        "sampling_measure": "rate_flux",
        "overlap_connected": True,
        "free_energy_barrier_hartree": mbar,
        "barrier_standard_error_hartree": float(mb.std(ddof=1)),
        "bootstrap": {
            "requested": 40,
            "attempted": 40,
            "valid": 40,
            "draws": [
                {"draw": i, "valid": True, "barrier_hartree": float(value)}
                for i, value in enumerate(mb)
            ],
        },
        "ui_comparison": {
            "ready": True,
            "xi_star": settings["xi_star"],
            "reference": 0.0,
            "free_energy_barrier_hartree": ui,
            "mbar_minus_ui_hartree": mbar - ui,
            "paired_difference_bootstrap_hartree": differences.tolist(),
            "paired_mbar_bootstrap_barriers": mb.tolist(),
            "paired_ui_bootstrap_barriers": ub.tolist(),
            "paired_difference_statistics": {
                "valid": 40,
                "standard_error_hartree": float(differences.std(ddof=1)),
                "percentile_95_ci_hartree": np.quantile(
                    differences, [0.025, 0.975]
                ).tolist(),
            },
            "ui_barrier_statistics": {"standard_error_hartree": float(ub.std(ddof=1))},
            "failures": [],
        },
    }
    diagnostics = {
        "samples": budget["saved_samples"],
        "streams": budget["streams"],
        "effective_samples_minimum": 8.0,
        "effective_samples_median": 20.0,
        "effective_samples_sum": 2000.0,
        "reactant_reference_sample_count": 200,
        "dividing_surface_sample_count": 200,
        "maximum_parent_constraint_residual_bohr": 1e-11,
        "parent_surface_branch_counts": [32, 32],
        "positive_initial_flux": True,
        "kappa_range": 0.15,
        "maximum_endpoint_energy_error_kbt": 0.02,
        "checkpoint_integrity": {"passed": True},
        "native_kernel_used": True,
    }
    recrossing = {
        "kappa": 0.85 + shift,
        "kappa_standard_error": 0.02,
        "plateau_detected": True,
        "child_trajectories": 256,
        "parent_configurations": 64,
        "child_pairs": 128,
        "child_time_fs": 80.0,
    }
    production = gates_for(settings, {"pmf": pmf, **recrossing}, diagnostics)
    controls = {
        k: v
        for k, v in settings.items()
        if k not in ("case", "seed", "workers", "cpu_count")
    }
    digest = hashlib.sha256(
        json.dumps(
            controls, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    hdf5 = hashlib.sha256(case.encode()).hexdigest()
    data = {
        "schema": "pyrpmd.h3.pmf-convergence.v1",
        "case": case,
        "seed": settings["seed"],
        "smoke": False,
        "passed": True,
        "converged": False,
        "settings": settings,
        "budget": budget,
        "pmf": pmf,
        "diagnostics": diagnostics,
        "recrossing": recrossing,
        "production_gates": production,
        "gates": {**production, "sources_unchanged_during_run": True},
        "fingerprints": {"comparison_controls": digest, "final_hdf5_sha256": hdf5},
        "rate_estimate_cm3_molecule_s": 1e-11 + shift * 1e-9,
        "rate_standard_error_cm3_molecule_s": 2e-12,
        "rate_percentile_95_ci_cm3_molecule_s": [5e-12, 2e-11],
        "conditional_rate_standard_error_cm3_molecule_s": 1e-12,
        "rate_uncertainty": {
            "available": True,
            "pmf_bootstrap_valid": 40,
            "recrossing_bootstrap_valid": 100,
        },
        "bin_sensitivity": [
            {
                "bins": bins,
                "ready": True,
                "free_energy_barrier_hartree": mbar + delta,
                "bootstrap": {"requested": 0},
                "uncertainty_available": False,
            }
            for bins, delta in [(85, -0.0001), (340, 0.0001)]
        ],
        "fixture_only": True,
    }
    gates = {
        key: True
        for key in [
            "auto_residual_below_1e_9",
            "legacy_residual_below_1e_9",
            "target_weights_match_legacy",
            "gauge_free_energies_match_legacy",
            "overlap_components_match_legacy",
            "target_weights_match_independent_pymbar",
            "gauge_free_energies_match_independent_pymbar",
        ]
    }
    solver = {
        "schema": "pyrpmd.h3.mbar-solver-comparison.v1",
        "passed": True,
        "gates": gates,
        "source_hdf5_sha256": hdf5,
        "streams": budget["streams"],
        "samples": budget["saved_samples"] // 10,
        "stride": 10,
        "independent_pymbar": {
            "version": "4.0.3",
            "asymptotic_uncertainty_used": False,
            "target_weight_l1_difference": 1e-12,
            "maximum_gauge_free_energy_difference": 1e-12,
        },
        "fixture_only": True,
    }
    parent = Path(directory) / f"h3-pmf-{case}" / case / "study"
    parent.mkdir(parents=True)
    (parent / "benchmark.json").write_text(json.dumps(data))
    (parent / "solver-check.json").write_text(json.dumps(solver))
    return parent


class ConvergenceReportTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "optional matplotlib is required for figure artifact checks",
    )
    def test_exact_four_cases_values_pair_difference_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in CASES:
                make_fixture(directory, case)
            output = Path(directory) / "PMF-CONVERGENCE.md"
            report = write_report(directory, output)
            self.assertTrue(report["passed"], report["errors"])
            self.assertFalse(report["converged"])
            metrics = report["cases"]["k400-a"]["metrics"]
            self.assertAlmostEqual(
                metrics["mbar_barrier_kcal_mol"], 0.030 * 627.5094740631
            )
            comparison = report["seed_comparisons"]["k400"]["quantities"][
                "rate_cm3_molecule_s"
            ]
            self.assertAlmostEqual(comparison["difference"], 2e-13, delta=1e-25)
            self.assertAlmostEqual(
                comparison["combined_standard_error"], np.sqrt(2) * 2e-12, delta=1e-25
            )
            self.assertTrue(report["seed_comparisons"]["k400"]["descriptive_only"])
            self.assertTrue(output.is_file() and output.with_suffix(".json").is_file())
            self.assertGreater(
                (Path(directory) / "figures/pmf-comparison.png").stat().st_size, 1000
            )
            self.assertGreater(
                (Path(directory) / "figures/pmf-comparison.pdf").stat().st_size, 1000
            )
            self.assertIn("共同改变", output.read_text())

    def test_missing_and_smoke_write_partial_report_and_exit_one(self):
        with tempfile.TemporaryDirectory() as directory:
            for case in CASES[:-1]:
                make_fixture(directory, case)
            path = Path(directory) / "h3-pmf-k400-a/k400-a/study/benchmark.json"
            data = json.loads(path.read_text())
            data["smoke"] = True
            path.write_text(json.dumps(data))
            captured = io.StringIO()
            # This test checks failure reporting/exit status, not plotting;
            # normal .[test] installations need no optional graphics stack.
            with redirect_stdout(captured), patch.dict(
                main.__globals__, {"plot_report": lambda report, directory: []}
            ):
                code = main(
                    [directory, "--output", str(Path(directory) / "PMF-CONVERGENCE.md")]
                )
            self.assertEqual(code, 1)
            report = json.loads((Path(directory) / "PMF-CONVERGENCE.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(
                {"missing_case", "smoke_ineligible"}.issubset(
                    {e["code"] for e in report["errors"]}
                )
            )
            self.assertFalse(report["cases"]["k800-b"]["present"])
            self.assertIn("PYRPMD_H3_CONVERGENCE_REPORT ", captured.getvalue())

    def test_controls_solver_and_missing_uncertainty_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            parents = {case: make_fixture(directory, case) for case in CASES}
            path = parents["k400-a"] / "benchmark.json"
            data = json.loads(path.read_text())
            data["fingerprints"]["comparison_controls"] = "wrong"
            data["rate_standard_error_cm3_molecule_s"] = None
            data["production_gates"].pop("overlap_connected")
            path.write_text(json.dumps(data))
            (parents["k800-b"] / "solver-check.json").unlink()
            report = aggregate(directory)
            self.assertFalse(report["passed"])
            codes = {e["code"] for e in report["errors"]}
            self.assertTrue(
                {
                    "comparison_controls_mismatch",
                    "missing_estimate_or_uncertainty",
                    "production_gate_schema",
                    "solver_check_unavailable",
                }.issubset(codes)
            )
            self.assertIsNone(
                report["cases"]["k400-a"]["metrics"]["rate_full_se_cm3_molecule_s"]
            )
            self.assertFalse(report["seed_comparisons"]["k400"]["available"])


if __name__ == "__main__":
    unittest.main()
