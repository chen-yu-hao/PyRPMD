"""End-to-end PCET input, local normalization, scan and restart regressions."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from pyrpmd.cli import main
from pyrpmd.constants import EV_TO_AU
from pyrpmd.input import JobSpec, InputError
from pyrpmd.io import referenced_files
from pyrpmd.pcet_input import PCETModelIdentity
from pyrpmd.runner import analyze_run, prepare, resume_run, run
from pyrpmd.workflows import pcet as workflow


ROOT = Path(__file__).resolve().parents[1]


def pcet_spec():
    return {
        "task": "pcet", "temperature": "300 K",
        "pcet": {
            "method": "golden_rule", "normalization": "local_first_order",
            "isotopes": ["H", "D"], "driving_force": "0 eV",
            "reorganization_energy": "0.7 eV", "electronic_coupling": "0.0001 eV",
            "electrode": {
                "model": "wide_band", "density_of_states": "1 eV^-1",
                "potential_reference": "SHE", "reference_potential_V": 0.0,
                "potentials_V": [-0.1, 0.1],
            },
            "proton": {
                "grid_min": "-1.5 angstrom", "grid_max": "1.5 angstrom",
                "grid_points": 101, "states": 6,
                "oxidized": {"kind": "harmonic", "center": "-0.2 angstrom", "force_constant": "4 eV/angstrom^2"},
                "reduced": {"kind": "harmonic", "center": "0.2 angstrom", "force_constant": "5 eV/angstrom^2"},
            },
            "numerics": {"convergence_check": False},
        },
    }


class PCETInputTests(unittest.TestCase):
    def test_system_free_model_roundtrip_and_explicit_units(self):
        config, model = JobSpec.from_mapping(pcet_spec()).resolve()
        self.assertIsInstance(model, PCETModelIdentity)
        self.assertFalse(hasattr(model, "natoms"))
        self.assertNotIn("system", config)
        self.assertNotIn("potential", config)
        self.assertEqual(config["temperature_K"], [300.0])
        self.assertAlmostEqual(config["pcet"]["electrode"]["density_per_hartree"], 1 / EV_TO_AU)
        copied = PCETModelIdentity.from_dict(json.loads(json.dumps(model.to_dict())))
        np.testing.assert_array_equal(copied.coordinate_bohr, model.coordinate_bohr)
        np.testing.assert_array_equal(copied.oxidized_hartree, model.oxidized_hartree)
        self.assertEqual(copied.oxidized_energy_origin_au, 0.0)
        self.assertFalse(copied.coordinate_bohr.flags.writeable)
        prepared = prepare(pcet_spec(), numerical=True)
        self.assertTrue(prepared["ready"])
        self.assertIn("proton_spectra", prepared["pcet_preparation"])
        self.assertIn("model", prepared)
        self.assertNotIn("system", prepared)

    def test_unsupported_science_and_ambiguous_units_rejected_at_validation(self):
        mutations = [
            lambda d: d.update(beads=32),
            lambda d: d.update(system={"cell": [1, 1, 1]}),
            lambda d: d.update(parallel={"backend": "process"}),
            lambda d: d["pcet"].update(pH=7),
            lambda d: d["pcet"].update(method="rpmd"),
            lambda d: d["pcet"].update(normalization="current_density"),
            lambda d: d["pcet"].update(reorganization_energy="0 eV"),
            lambda d: d["pcet"].update(electronic_coupling="-0.01 eV"),
            lambda d: d["pcet"]["electrode"].update(model="constant_potential"),
            lambda d: d["pcet"]["electrode"].update(potential_reference=""),
            lambda d: d["pcet"]["electrode"].update(potentials_V=["0 V"]),
            lambda d: d["pcet"]["electrode"].update(density_of_states="1 eV"),
            lambda d: d["pcet"]["electrode"].update(potentials_V=[0, 0]),
            lambda d: d.update(temperature=[300, 300]),
            lambda d: d.update(temperature=True),
            lambda d: d["pcet"]["proton"].update(states=True),
            lambda d: d["pcet"]["proton"].update(states=100),
            lambda d: d["pcet"]["numerics"].update(quadrature_rtol=1.0),
            lambda d: d["pcet"]["numerics"].update(quadrature_atol=1.0),
            lambda d: d["pcet"]["numerics"].update(relative_tolerance=1.0),
            lambda d: d["pcet"]["numerics"].update(convergence_check=1),
        ]
        for number, mutate in enumerate(mutations):
            raw = pcet_spec()
            mutate(raw)
            with self.subTest(case=number), self.assertRaises(InputError):
                JobSpec.from_mapping(raw).resolve()

    def test_only_meaningful_cli_overrides_accepted(self):
        spec = JobSpec.from_mapping(pcet_spec())
        config, _ = spec.with_overrides(temperature="315 K").resolve()
        self.assertEqual(config["temperature_K"], [315.0])
        for options in ({"workers": 1}, {"beads": 1}, {"pes": "model.so"}):
            with self.subTest(options=options), self.assertRaisesRegex(InputError, "PCET supports only"):
                spec.with_overrides(**options)

    def test_cli_validate_writes_model_identity_and_prepares_quantum_proton(self):
        source = ROOT / "examples" / "pcet" / "harmonic_scan.inp"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "resolved.json"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = main(["validate", str(source), "--temperature", "310 K", "--write-resolved", str(target)])
            self.assertEqual(status, 0)
            report = json.loads(buffer.getvalue())
            self.assertEqual(report["temperature_K"], [310.0])
            saved = json.loads(target.read_text())
            self.assertEqual(saved["model"]["schema"], "pyrpmd.pcet-model")
            self.assertNotIn("system", saved)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                status = main(["validate", str(source), "--check-pes"])
            self.assertEqual(status, 0)
            self.assertIn("proton_spectra", json.loads(buffer.getvalue())["pcet_preparation"])

    def test_tabulated_units_origins_relative_paths_and_dependency_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            x = np.linspace(-2, 2, 161)
            np.savetxt(base / "oxidized.csv", np.column_stack((x, 2 * (x + 0.2) ** 2 + 17)), delimiter=",")
            np.savetxt(base / "reduced.csv", np.column_stack((x, 2.5 * (x - 0.2) ** 2 - 4)), delimiter=",")
            raw = pcet_spec()
            for name in ("oxidized", "reduced"):
                raw["pcet"]["proton"][name] = {
                    "kind": "tabulated", "file": name + ".csv", "coordinate_unit": "angstrom", "energy_unit": "eV",
                }
            spec = JobSpec.from_mapping(raw, base_directory=base)
            config, model = spec.resolve()
            self.assertAlmostEqual(model.oxidized_energy_origin_au, 17 * EV_TO_AU)
            self.assertAlmostEqual(model.reduced_energy_origin_au, -4 * EV_TO_AU)
            hashes = referenced_files(config)
            self.assertEqual(set(hashes), {str(base / "oxidized.csv"), str(base / "reduced.csv")})
            run(spec, output=base / "result")
            source = base / "oxidized.csv"
            source.write_text(source.read_text() + "# same curve, changed source file\n")
            with self.assertRaisesRegex(RuntimeError, "changed or is missing"):
                resume_run(base / "result")

    def test_tabulated_interpolation_never_extrapolates_or_ignores_column_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "potential.dat"
            raw = pcet_spec()
            raw["pcet"]["proton"]["oxidized"] = {
                "kind": "tabulated", "file": str(source), "coordinate_unit": "angstrom", "energy_unit": "eV",
            }
            x = np.linspace(-1, 1, 51)
            np.savetxt(source, np.column_stack((x, x * x)))
            with self.assertRaisesRegex(InputError, "cover the entire"):
                JobSpec.from_mapping(raw).resolve()
            np.savetxt(source, np.column_stack((x, x * x, x)))
            with self.assertRaisesRegex(InputError, "exactly two"):
                JobSpec.from_mapping(raw).resolve()


class PCETWorkflowTests(unittest.TestCase):
    def test_actual_voltage_temperature_isotope_scan_and_local_units(self):
        raw = pcet_spec()
        raw["temperature"] = ["280 K", "340 K"]
        with tempfile.TemporaryDirectory() as directory:
            result = run(raw, output=Path(directory) / "run")
            self.assertEqual(result["completed_scan_points"], 8)
            self.assertEqual(result["rate_unit"], "s^-1")
            self.assertTrue(result["physical_applicability_unverified"])
            self.assertFalse(result["numerical_convergence"])
            self.assertFalse((Path(result["output"]) / "input" / "system.json").exists())
            self.assertTrue((Path(result["output"]) / "input" / "model.json").is_file())
            self.assertEqual(analyze_run(result["output"]), result)
            for temperature in (280.0, 340.0):
                for isotope in ("H", "D"):
                    negative, positive = [row for row in result["rates"] if row["temperature_K"] == temperature and row["isotope"] == isotope]
                    self.assertGreater(negative["reduction_rate_s"], positive["reduction_rate_s"])
                    self.assertLess(negative["oxidation_rate_s"], positive["oxidation_rate_s"])
                    self.assertAlmostEqual(positive["driving_force_au"] - negative["driving_force_au"], 0.2 * EV_TO_AU)
            self.assertEqual(len(result["kinetic_isotope_effects"]), 4)
            for entry in result["kinetic_isotope_effects"]:
                self.assertGreater(entry["reduction_KIE"], 0)
            for name in ("rates.csv", "pcet_details.json", "proton_spectra.json", "proton_wavefunctions.npz"):
                self.assertTrue((Path(result["output"]) / name).is_file())

    def test_interrupted_and_completed_resume_reuse_exactly_completed_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            original = workflow._rates
            count = 0

            def interrupt_second(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise RuntimeError("planned interruption before second PCET point")
                return original(*args, **kwargs)

            with mock.patch.object(workflow, "_rates", side_effect=interrupt_second):
                with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                    run(pcet_spec(), output=output)
            self.assertEqual(len(list((output / "checkpoints").glob("pcet-*.json"))), 1)
            with mock.patch.object(workflow, "_rates", wraps=original) as rates:
                result = resume_run(output)
                self.assertEqual(rates.call_count, 3)
            self.assertEqual(result["completed_scan_points"], 4)
            with mock.patch.object(workflow, "_rates", side_effect=AssertionError("completed point recomputed")):
                again = resume_run(output)
            self.assertEqual(again, result)
            with self.assertRaisesRegex(InputError, "does not accept"):
                resume_run(output, parallel={"trajectory_workers": 1})
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                status = main(["resume", str(output), "--workers", "0"])
            self.assertEqual(status, 2)
            self.assertIn("does not accept", errors.getvalue())

    def test_checkpoint_tampering_and_scientific_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run(pcet_spec(), output=output)
            changed = pcet_spec()
            changed["pcet"]["driving_force"] = "0.1 eV"
            with self.assertRaisesRegex(RuntimeError, "changed scientific"):
                run(changed, output=output, resume=True)
            checkpoint = next((output / "checkpoints").glob("pcet-*.json"))
            saved = json.loads(checkpoint.read_text())
            saved["result"]["rates"]["reduction_rate_s"] *= 2
            checkpoint.write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError, "identity or integrity failed"):
                resume_run(output)

    def test_zero_coupling_retains_undefined_kie_and_json_safe_logs(self):
        raw = pcet_spec()
        raw["pcet"]["electronic_coupling"] = "0 eV"
        with tempfile.TemporaryDirectory() as directory:
            result = run(raw, output=Path(directory) / "run")
            for row in result["rates"]:
                self.assertEqual(row["reduction_rate_s"], 0.0)
                self.assertEqual(row["oxidation_rate_s"], 0.0)
                self.assertIsNone(row["log_reduction_rate_s"])
            for entry in result["kinetic_isotope_effects"]:
                self.assertIsNone(entry["reduction_KIE"])
                self.assertIsNone(entry["log_reduction_KIE"])
            json.dumps(result, allow_nan=False)

    def test_proton_box_boundary_diagnostic_prevents_false_numerical_convergence(self):
        raw = pcet_spec()
        raw["pcet"]["isotopes"] = ["H"]
        raw["pcet"]["electrode"]["potentials_V"] = [0.0]
        raw["pcet"]["numerics"]["convergence_check"] = True
        proton = raw["pcet"]["proton"]
        proton.update(grid_min="-0.1 angstrom", grid_max="0.1 angstrom", states=8)
        proton["oxidized"]["center"] = "-0.02 angstrom"
        proton["reduced"]["center"] = "0.02 angstrom"
        with tempfile.TemporaryDirectory() as directory:
            result = run(raw, output=Path(directory) / "run")
            details = json.loads(Path(result["files"]["details"]).read_text())
            checks = details["cases"][0]["numerical_checks"]
            self.assertFalse(result["numerical_convergence"])
            self.assertFalse(checks["reactive_proton_checks_passed"])
            self.assertIn("independent_proton_grid_extent_scan", checks["unchecked"])

    def test_small_coupling_cannot_hide_a_coarse_unconverged_proton_grid(self):
        raw = pcet_spec()
        raw["pcet"]["isotopes"] = ["H"]
        raw["pcet"]["electrode"]["potentials_V"] = [0.0]
        raw["pcet"]["proton"].update(grid_points=21, states=4)
        raw["pcet"]["numerics"].update(convergence_check=True, relative_tolerance=1e-4)
        relative_errors = []
        with tempfile.TemporaryDirectory() as directory:
            for index, coupling in enumerate(("0.0001 eV", "1e-100 eV")):
                raw["pcet"]["electronic_coupling"] = coupling
                result = run(raw, output=Path(directory) / str(index))
                details = json.loads(Path(result["files"]["details"]).read_text())
                grid = details["cases"][0]["numerical_checks"]["grid_spacing"]
                self.assertFalse(result["numerical_convergence"])
                self.assertFalse(grid["passed"])
                relative_errors.append([grid["rates"][field]["relative_difference"] for field in workflow._RATE_FIELDS])
                if index:
                    self.assertLess(result["rates"][0]["reduction_rate_s"], 1e-12)
            np.testing.assert_allclose(relative_errors[0], relative_errors[1], atol=1e-12, rtol=1e-10)

    def test_underflowed_rates_with_distinct_finite_logs_are_not_converged(self):
        raw = pcet_spec()
        raw["pcet"]["isotopes"] = ["H"]
        raw["pcet"]["electrode"]["potentials_V"] = [0.0]
        raw["pcet"]["electronic_coupling"] = "1e-200 eV"
        raw["pcet"]["proton"].update(grid_points=21, states=4)
        raw["pcet"]["numerics"].update(convergence_check=True, relative_tolerance=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            result = run(raw, output=Path(directory) / "run")
            self.assertFalse(result["numerical_convergence"])
            row = result["rates"][0]
            self.assertEqual(row["reduction_rate_s"], 0.0)
            self.assertEqual(row["oxidation_rate_s"], 0.0)
            self.assertTrue(np.isfinite(row["log_reduction_rate_s"]))
            details = json.loads(Path(result["files"]["details"]).read_text())
            grid = details["cases"][0]["numerical_checks"]["grid_spacing"]
            self.assertFalse(grid["passed"])
            for check in grid["rates"].values():
                self.assertEqual(check["reference"], 0.0)
                self.assertEqual(check["refined"], 0.0)
                self.assertEqual(check["absolute_difference_s"], 0.0)
                self.assertGreater(check["relative_difference"], 1e-4)
                self.assertNotEqual(check["log_rate_difference"], 0.0)
                self.assertFalse(check["passed"])


if __name__ == "__main__":
    unittest.main()
