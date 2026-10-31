"""State-selected nuclear inputs resolve before creating a PES or trajectories."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.workflows.drpmd import _physics_hash


MODE = np.array([0.0, 0.0, -2**-0.5, 0.0, 0.0, 2**-0.5])


def selected_job(method="ne_rpmd"):
    section = {
        "method": method,
        "fragments": [{}, {
            "frequencies": [0.02], "modes": [MODE.tolist()], "vibrational_quanta": [1],
        }],
    }
    if method == "ne_rpmd":
        section["beta"] = "400 hartree^-1"
    return {
        "task": "drpmd", "beads": 32 if method == "ne_rpmd" else 1,
        "initial_ensemble": section,
        "system": {
            "symbols": ["H", "H", "H"], "masses": [2.0, 2.0, 2.0],
            "geometry": [[-8.0, 0.0, 0.0], [0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
            "fragments": [[0], [1, 2]],
        },
        "potential": {"provider": "harmonic", "params": {"k": 0.01}},
        "dynamics": {"timestep": "0.1 fs"},
        "collision": {"energy": "0.05 hartree", "bmax": "0.5 bohr", "trajectories": 2, "max_steps": 10},
        "outcomes": {"bond_cutoffs": {"H-H": "1.6 bohr"}, "persistence_steps": 2},
        "output": {"checkpoint_steps": 5},
    }


class StateSelectedInputTests(unittest.TestCase):
    def test_ne_rpmd_resolves_reference_beta_without_thermal_sampling(self):
        data = selected_job()
        data["sampling"] = {"parents": 2}
        original = deepcopy(data)
        with patch("pyrpmd.potentials.make_potential", side_effect=AssertionError("PES creation during input validation")):
            config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(data, original)
        self.assertEqual(config["beta"], 400.0)
        self.assertEqual(config["initial_ensemble"]["beta"], 400.0)
        self.assertAlmostEqual(config["temperature_K"], 1 / KB_AU / 400.0)
        self.assertIn("not_thermal_temperature", config["temperature_interpretation"])
        self.assertEqual(config["sampling"], {"parents": 2})
        self.assertEqual(config["collision"]["sampling"], "fixed_energy")
        self.assertEqual(config["collision"]["energy_au"], 0.05)
        self.assertEqual(config["output"]["diagnostics"], {"energy_parts": True, "ring_radii": True})
        self.assertFalse(any("Pilot default temperature" in note for note in config["_warnings"]))

    def test_qct_defaults_to_one_bead_and_never_defaults_to_a_thermal_ensemble(self):
        data = selected_job("qct")
        data.pop("beads")
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["beads"], 1)
        self.assertEqual(config["beta"], 1.0)
        self.assertEqual(config["sampling"], {})
        self.assertEqual(config["initial_ensemble"]["method"], "qct")
        self.assertIn("unused_classical", config["temperature_interpretation"])
        data["initial_ensemble"]["beta"] = 900.0
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["beta"], 900.0)

    def test_production_selected_requires_no_thermal_temperature_or_sampling(self):
        for method in ("qct", "ne_rpmd"):
            with self.subTest(method=method):
                data = selected_job(method)
                data["preset"] = "production"
                config, _ = JobSpec.from_mapping(data).resolve()
                self.assertEqual(config["sampling"], {})
                self.assertNotIn("temperature", config)
                for missing in ("beads", "dynamics"):
                    incomplete = deepcopy(data)
                    incomplete.pop(missing)
                    with self.assertRaisesRegex(InputError, "production requires explicit " + missing):
                        JobSpec.from_mapping(incomplete).resolve()

    def test_beta_bead_and_temperature_conflicts_are_rejected(self):
        cases = []
        absent_beta = selected_job()
        absent_beta["initial_ensemble"].pop("beta")
        cases.append((absent_beta, "explicit initial_ensemble.beta"))
        one_bead_ne = selected_job()
        one_bead_ne["beads"] = 1
        cases.append((one_bead_ne, "beads>=2"))
        many_bead_qct = selected_job("qct")
        many_bead_qct["beads"] = 32
        cases.append((many_bead_qct, "beads=1"))
        for method in ("qct", "ne_rpmd"):
            ambiguous = selected_job(method)
            ambiguous["temperature"] = "300 K"
            cases.append((ambiguous, "does not accept temperature"))
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(InputError, message):
                JobSpec.from_mapping(data).resolve()

    def test_initial_ensemble_is_drpmd_only_including_special_frontends(self):
        for task in ("rate", "fes", "pimd", "diffusion", "network", "pcet"):
            data = selected_job()
            data["task"] = task
            with self.subTest(task=task), self.assertRaisesRegex(InputError, "requires task='drpmd'"):
                JobSpec.from_mapping(data).resolve()

    def test_only_parent_allocation_can_be_set_under_selected_sampling(self):
        for key, value in (("equilibration_steps", 0), ("thermostat", "andersen"),
                           ("interval_steps", 1), ("integrator", "cayley-bcb-andersen"),
                           ("replicas", 1), ("timestep", "0.1 fs")):
            data = selected_job()
            data["sampling"] = {key: value}
            with self.subTest(key=key), self.assertRaisesRegex(InputError, "only sampling.parents"):
                JobSpec.from_mapping(data).resolve()

    def test_selected_collisions_require_explicit_fixed_energy(self):
        missing = selected_job()
        missing["collision"].pop("energy")
        with self.assertRaisesRegex(InputError, "requires collision.energy"):
            JobSpec.from_mapping(missing).resolve()
        for distribution in ("thermal_flux", "maxwell"):
            data = selected_job()
            data["collision"]["sampling"] = distribution
            with self.subTest(distribution=distribution), self.assertRaisesRegex(InputError, "requires collision.sampling='fixed_energy'"):
                JobSpec.from_mapping(data).resolve()

    def test_reference_content_and_masses_are_validated_during_resolve(self):
        for bad in ("imaginary", "wrong_mode_count", "nonorthogonal", "wrong_quanta", "rotation"):
            data = selected_job()
            fragment = data["initial_ensemble"]["fragments"][1]
            if bad == "imaginary":
                fragment["frequencies"] = [-0.02]
            elif bad == "wrong_mode_count":
                fragment["modes"] = [MODE.tolist(), MODE.tolist()]
            elif bad == "nonorthogonal":
                fragment["modes"] = [(2 * MODE).tolist()]
            elif bad == "wrong_quanta":
                fragment["vibrational_quanta"] = [0, 1]
            else:
                fragment["j"] = 1
            with self.subTest(bad=bad), self.assertRaises(InputError):
                JobSpec.from_mapping(data).resolve()
        isotope = selected_job()
        isotope["system"]["masses"] = [2.0, 2.0, 4.0]
        with self.assertRaisesRegex(InputError, "isotope masses"):
            JobSpec.from_mapping(isotope).resolve()

    def test_relative_hessian_files_become_data_and_change_the_physics_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            folder = base / "references"
            folder.mkdir()
            hessian = 2 * 0.02**2 * np.outer(MODE, MODE)
            np.save(folder / "hessian.npy", hessian)
            data = selected_job()
            data["initial_ensemble"]["fragments"][1] = {
                "hessian_file": "references/hessian.npy", "vibrational_quanta": [1],
            }
            config, system = JobSpec.from_mapping(data, base_directory=base).resolve()
            fragment = config["initial_ensemble"]["fragments"][1]
            self.assertNotIn("hessian_file", fragment)
            np.testing.assert_allclose(fragment["hessian"], hessian)
            original_hash = _physics_hash(config, system, 1)
            np.save(folder / "hessian.npy", 2 * hessian)
            changed, changed_system = JobSpec.from_mapping(data, base_directory=base).resolve()
            self.assertNotEqual(original_hash, _physics_hash(changed, changed_system, 1))
            # The already-resolved reference remains an immutable content snapshot.
            self.assertEqual(original_hash, _physics_hash(config, system, 1))

    def test_python_input_resolves_relative_modes_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.savetxt(base / "modes.txt", MODE[None])
            data = selected_job()
            fragment = data["initial_ensemble"]["fragments"][1]
            fragment.pop("modes")
            fragment.update(modes_file="modes.txt", frequencies=[1000.0], frequency_unit="cm^-1")
            path = base / "job.py"
            path.write_text("def build():\n    return " + repr(data) + "\n")
            config, _ = JobSpec.from_file(path).resolve()
            canonical = config["initial_ensemble"]["fragments"][1]
            self.assertNotIn("modes_file", canonical)
            self.assertEqual(canonical["frequency_unit"], "au")
            self.assertGreater(canonical["frequencies"][0], 0)
            self.assertLess(canonical["frequencies"][0], 0.01)
            np.testing.assert_allclose(canonical["modes"], [MODE])
            json.dumps(config["initial_ensemble"], allow_nan=False)

    def test_diagnostics_inline_toml_flags_and_strict_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.toml"
            path.write_text('''task = "drpmd"
beads = 1
[initial_ensemble]
method = "qct"
fragments = [{}, {}]
[system]
symbols = ["H", "H"]
masses = [2.0, 2.0]
geometry = [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
fragments = [[0], [1]]
[potential]
provider = "free"
[collision]
energy = "0.05 hartree"
bmax = "0.5 bohr"
max_steps = 10
[outcomes]
bond_cutoffs = { H-H = "0.3 bohr" }
[output]
diagnostics = { energy_parts = false, ring_radii = true }
''')
            config, _ = JobSpec.from_file(path).resolve()
            self.assertEqual(config["output"]["diagnostics"], {"energy_parts": False, "ring_radii": True})
        for descriptor in (True, None, {"energy_part": True}, {"energy_parts": 1}, {"ring_radii": "yes"}):
            data = selected_job()
            data["output"]["diagnostics"] = descriptor
            with self.subTest(descriptor=descriptor), self.assertRaisesRegex(InputError, "output.diagnostics"):
                JobSpec.from_mapping(data).resolve()
        data = selected_job()
        data["output"]["diagnostics"] = {"energy_parts": False}
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["output"]["diagnostics"], {"energy_parts": False, "ring_radii": True})

    def test_legacy_thermal_input_keeps_temperature_and_sampling_defaults(self):
        data = selected_job()
        data.pop("initial_ensemble")
        data["temperature"] = "300 K"
        data["collision"].pop("energy")
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["temperature_K"], 300.0)
        self.assertEqual(config["beta"], 1 / (KB_AU * 300.0))
        self.assertEqual(config["sampling"]["thermostat"], "andersen")
        self.assertGreater(config["sampling"]["equilibration_steps"], 0)
        self.assertEqual(config["collision"]["sampling"], "thermal_flux")
        self.assertNotIn("temperature_interpretation", config)
        self.assertNotIn("diagnostics", config["output"])
        data["output"]["diagnostics"] = {"ring_radii": True}
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["output"]["diagnostics"], {"energy_parts": False, "ring_radii": True})

    def test_nonadiabatic_combination_is_explicitly_unimplemented(self):
        data = selected_job()
        data["nonadiabatic"] = {"method": "nrpmd", "nstates": 2}
        with self.assertRaisesRegex(InputError, "combining it with NRPMD is not implemented"):
            JobSpec.from_mapping(data).resolve()


if __name__ == "__main__":
    unittest.main()
