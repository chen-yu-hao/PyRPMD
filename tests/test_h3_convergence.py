"""Prespecified campaign inputs and scientific acceptance cannot drift silently."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from pyrpmd.potentials.python import import_object

MODULE = str(Path(__file__).resolve().parents[1] / "benchmarks/h3/convergence.py")
case_settings = import_object(MODULE + ":case_settings")
compute_budget = import_object(MODULE + ":compute_budget")
configuration = import_object(MODULE + ":configuration")
quality_gates = import_object(MODULE + ":quality_gates")
load_samples = import_object(MODULE + ":load_samples")


class ConvergenceProtocolTests(unittest.TestCase):
    def test_four_prespecified_cases_have_distinct_global_seeds_and_exact_work_counts(
        self,
    ):
        seeds = []
        for name, windows, umbrella_steps, samples in [
            ("k400-a", 47, 1222000, 188000),
            ("k400-b", 47, 1222000, 188000),
            ("k800-a", 59, 1534000, 236000),
            ("k800-b", 59, 1534000, 236000),
        ]:
            settings = case_settings(name)
            budget = compute_budget(settings)
            seeds.append(settings["seed"])
            self.assertEqual(budget["windows"], windows)
            self.assertEqual(budget["umbrella_steps"], umbrella_steps)
            self.assertEqual(budget["saved_samples"], samples)
            self.assertEqual(budget["child_trajectories"], 256)
            self.assertEqual(budget["primary_bootstrap_draws"], 40)
            self.assertEqual(settings["xi_star"], 1.00925)
            self.assertEqual(settings["histogram_range"], [-0.3, 1.4])
            self.assertEqual(settings["bins"], 170)
            self.assertEqual(settings["sensitivity_bins"], [85, 340])
            self.assertFalse(settings["smoke"])
        self.assertEqual(seeds, [821071, 821072, 831071, 831072])
        self.assertEqual(len(set(seeds)), 4)
        with self.assertRaises(ValueError):
            case_settings("k400-a", smoke="false")

    def test_configuration_uses_real_mbar_rate_backend_and_physical_beta_k(self):
        template = Path(MODULE).parent / "convergence_inputs/base.inp"
        original = template.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            for case in ("k400-a", "k800-b"):
                config, system, settings = configuration(case, directory, workers=2)
                self.assertAlmostEqual(
                    config["beta"] * config["umbrella"]["kappa"], settings["beta_k"]
                )
                self.assertEqual(config["rate"]["pmf_method"], "mbar")
                self.assertEqual(config["rate"]["dividing_surface"], 1.00925)
                self.assertEqual(config["sampling"]["production_steps"], 10000)
                self.assertEqual(config["sampling"]["equilibration_steps"], 3000)
                self.assertEqual(config["sampling"]["replicas"], 2)
                self.assertEqual(config["analysis"]["bootstrap"], 40)
                self.assertEqual(config["analysis"]["ranges"], [[-0.3, 1.4]])
                self.assertEqual(config["parallel"]["trajectory_workers"], 2)
                self.assertEqual(tuple(system.symbols), ("H", "H", "H"))
        self.assertEqual(template.read_bytes(), original)

    def good_gate_fixture(self):
        settings = case_settings("k400-a")
        result = {
            "pmf": {
                "method": "mbar",
                "xi_star": 1.00925,
                "free_energy_barrier_hartree": 0.03,
                "overlap_connected": True,
                "bootstrap": {
                    "requested": 40,
                    "attempted": 40,
                    "valid": 40,
                    "draws": [{"draw": i, "valid": True} for i in range(40)],
                },
                "ui_comparison": {
                    "ready": True,
                    "paired_difference_bootstrap_hartree": [0.0] * 40,
                    "failures": [],
                },
            },
            "kappa": 0.9,
            "child_trajectories": 256,
            "plateau_detected": True,
        }
        diagnostics = {
            "effective_samples_minimum": 8.0,
            "reactant_reference_sample_count": 100,
            "dividing_surface_sample_count": 100,
            "maximum_parent_constraint_residual_bohr": 1e-11,
            "parent_surface_branch_counts": [32, 32],
            "positive_initial_flux": True,
            "kappa_range": 0.1,
            "maximum_endpoint_energy_error_kbt": 0.01,
            "checkpoint_integrity": {"passed": True},
            "native_kernel_used": True,
        }
        return settings, result, diagnostics

    def test_existing_energy_integrity_and_child_gates_are_not_relaxed_for_new_cases(
        self,
    ):
        settings, result, diagnostics = self.good_gate_fixture()
        self.assertTrue(all(quality_gates(settings, result, diagnostics).values()))
        diagnostics["maximum_endpoint_energy_error_kbt"] = 0.1
        gates = quality_gates(settings, result, diagnostics)
        self.assertFalse(
            gates["offset_independent_endpoint_energy_error_below_0p1_kbt"]
        )
        diagnostics["maximum_endpoint_energy_error_kbt"] = 0.01
        diagnostics["checkpoint_integrity"]["passed"] = False
        self.assertFalse(
            quality_gates(settings, result, diagnostics)[
                "final_samples_match_checkpoints"
            ]
        )
        result["child_trajectories"] = 127
        self.assertFalse(
            quality_gates(settings, result, diagnostics)["actual_child_ensemble"]
        )
        # Smoke retains these production gate definitions rather than changing
        # their numerical thresholds to make a short run appear converged.
        settings["smoke"] = True
        diagnostics["maximum_endpoint_energy_error_kbt"] = 0.101
        self.assertFalse(
            quality_gates(settings, result, diagnostics)[
                "offset_independent_endpoint_energy_error_below_0p1_kbt"
            ]
        )

    def test_missing_bootstrap_attempts_and_unpaired_ui_draws_are_visible_failures(
        self,
    ):
        settings, result, diagnostics = self.good_gate_fixture()
        result["pmf"]["bootstrap"]["draws"].pop()
        self.assertFalse(
            quality_gates(settings, result, diagnostics)[
                "bootstrap_attempts_all_recorded"
            ]
        )
        result["pmf"]["bootstrap"]["valid"] = 39
        self.assertFalse(
            quality_gates(settings, result, diagnostics)[
                "all_requested_bootstrap_draws_valid"
            ]
        )
        result["pmf"]["ui_comparison"]["paired_difference_bootstrap_hartree"].pop()
        self.assertFalse(
            quality_gates(settings, result, diagnostics)[
                "all_requested_paired_ui_draws_valid"
            ]
        )
        result["pmf"]["xi_star"] = 1.0
        self.assertFalse(
            quality_gates(settings, result, diagnostics)["prespecified_surface_used"]
        )

    def test_final_checkpoint_energy_clock_and_stream_identity_are_checked(self):
        settings = {
            "windows": 1,
            "replicas": 1,
            "production_fs": 1.0,
            "sample_interval_fs": 0.5,
            "equilibration_fs": 0.5,
            "timestep_fs": 0.1,
        }
        values, energy = np.array([[0.1], [0.2]]), np.array([1.0, 2.0])
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoints = run / "umbrella/checkpoints"
            checkpoints.mkdir(parents=True)
            with h5py.File(run / "umbrella/fes.h5", "w") as h:
                h.attrs["measure"] = "rate_flux"
                group = h.create_group("streams/0")
                group.attrs["window"], group.attrs["replica"] = 0, 0
                group["samples"], group["energies"] = values, energy
                group["center"], group["stiffness"] = [0.1], [[1.0]]
            checkpoint = checkpoints / "window_00000_replica_0000.npz"

            def save(energies=energy, elapsed=15):
                np.savez(
                    checkpoint,
                    values=values,
                    energies=energies,
                    elapsed=elapsed,
                    fingerprint="test-stream",
                    q=np.ones((2, 3, 3)),
                )

            save()
            self.assertTrue(
                load_samples(run, settings)[3]["checkpoint_integrity"]["passed"]
            )
            save(energies=energy + 1.0)
            self.assertFalse(
                load_samples(run, settings)[3]["checkpoint_integrity"]["passed"]
            )
            save(elapsed=14)
            self.assertFalse(
                load_samples(run, settings)[3]["checkpoint_integrity"]["passed"]
            )
            save()
            with h5py.File(run / "umbrella/fes.h5", "a") as h:
                h["streams/0"].attrs["window"] = 1
            checkpoint.rename(checkpoints / "window_00001_replica_0000.npz")
            integrity = load_samples(run, settings)[3]["checkpoint_integrity"]
            self.assertFalse(integrity["stream_identities_complete"])
            self.assertFalse(integrity["passed"])

    def test_smoke_is_explicit_and_cannot_mutate_formal_case(self):
        before = case_settings("k400-a")
        short = case_settings("k400-a", smoke=True)
        self.assertTrue(short["smoke"])
        self.assertLess(
            compute_budget(short)["umbrella_steps"],
            compute_budget(before)["umbrella_steps"],
        )
        self.assertEqual(short["beta_k"], before["beta_k"])
        self.assertEqual(short["xi_star"], before["xi_star"])
        short["histogram_range"][0] = -999
        self.assertEqual(case_settings("k400-a"), before)


if __name__ == "__main__":
    unittest.main()
