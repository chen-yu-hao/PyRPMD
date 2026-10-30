"""User-facing regression checks across input, plugin, run and resume layers."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.input import JobSpec, InputError
from pyrpmd.io import referenced_files
from pyrpmd.potentials import make_potential
from pyrpmd.runner import run, resume_run


def small_collision():
    return {
        "task": "drpmd",
        "temperature": 1 / KB_AU,
        "beads": 1,
        "seed": 11,
        "native": False,
        "system": {
            "symbols": ["H", "H"],
            "masses": [1.0, 1.0],
            "geometry": [[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]],
            "fragments": [[0], [1]],
        },
        "potential": {"provider": "free"},
        "dynamics": {"timestep": 0.1},
        "sampling": {
            "equilibration_steps": 4,
            "production_steps": 4,
            "interval_steps": 2,
            "andersen_interval": 2,
        },
        "collision": {"bmax": 0.5, "trajectories": 2, "max_steps": 200},
        "outcomes": {"bond_cutoffs": {"H-H": 0.3}, "persistence_steps": 2},
        "output": {"checkpoint_steps": 10},
    }


class InputTests(unittest.TestCase):
    def test_adaptive_coverage_fields_reach_public_input_validation(self):
        data = {
            "task": "fes", "temperature": 300, "beads": 1, "native": False,
            "system": {"symbols": ["H"], "geometry": [[0., 0., 0.]]},
            "potential": {"provider": "builtin", "name": "harmonic"},
            "dynamics": {"timestep": "0.1 fs"},
            "sampling": {"equilibration_steps": 10, "production_steps": 100,
                         "sample_interval": 2},
            "cvs": [{"kind": "cartesian", "atom": 0, "axis": axis} for axis in ("x", "y")],
            "umbrella": {"centers": [[0., 0.]], "stiffness": 1.},
            "adaptive_umbrella": {"pilot_steps": 100, "coverage_bins": [4, 8],
                                  "coverage_ranges": [["-1 bohr", "1 bohr"], ["-2 bohr", "2 bohr"]],
                                  "min_bin_effective_samples": 5., "target_coverage": .95},
            "analysis": {"bins": [8, 16], "bootstrap": 0},
        }
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["adaptive_umbrella"]["coverage_bins"], [4, 8])
        data["adaptive_umbrella"]["target_coverage"] = 1.1
        with self.assertRaisesRegex(ValueError, "target_coverage"):
            JobSpec.from_mapping(data).resolve()

    def test_explicit_steps_do_not_parse_unused_duration_defaults(self):
        data = small_collision()
        data["dynamics"]["timestep"] = "0.3 fs"
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["sampling"]["equilibration_steps"], 4)
        self.assertEqual(config["collision"]["max_steps"], 200)
        self.assertEqual(config["output"]["checkpoint_steps"], 10)

    def test_typo_and_inconsistent_resources_rejected_before_dynamics(self):
        for change in (
            {"temprature": 300},
            {"parallel": {"trajectory_workers": 2, "trajectory_groups": 3}},
            {"beads": True},
            {"parallel": {"pes_ranks": 2}},
        ):
            data = small_collision()
            data.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                JobSpec.from_mapping(data).resolve()

    def test_gpu_slots_default_to_all_declared_devices(self):
        data = small_collision()
        data["parallel"] = {"mode": "gpu", "devices": [0, 1]}
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["parallel"]["trajectory_workers"], 2)
        data["parallel"]["trajectories_per_gpu"] = 3
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["parallel"]["trajectory_workers"], 6)

    def test_python_build_relative_plugin_and_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "my_pes.py").write_text(
                "import numpy as np\ndef make():\n return lambda x: (np.sum(x*x,axis=(1,2))/2, -x)\n"
            )
            data = small_collision()
            data["potential"] = {"provider": "python", "factory": "my_pes.py:make"}
            (path / "job.py").write_text("def build():\n return " + repr(data) + "\n")
            config, system = JobSpec.from_file(path / "job.py").resolve()
            self.assertEqual(
                config["potential"]["factory"], str(path / "my_pes.py") + ":make"
            )
            provider = make_potential(config["potential"], system)
            np.testing.assert_allclose(
                provider.evaluate(system.geometry[None]).forces, -system.geometry[None]
            )
            self.assertIn(str(path / "my_pes.py"), referenced_files(config))
            old = referenced_files(config)
            (path / "my_pes.py").write_text("# a changed potential\n")
            self.assertNotEqual(old, referenced_files(config))

    def test_cli_pes_override_replaces_builtin(self):
        data = small_collision()
        spec = JobSpec.from_mapping(data).with_overrides(pes="test.so")
        self.assertEqual(spec.data["potential"]["provider"], "cabi")


class CLIRunTests(unittest.TestCase):
    def test_public_runner_restart_and_input_change_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = small_collision()
            first = run(data, output=path / "run")
            resumed = resume_run(path / "run")
            self.assertEqual(first["records"], resumed["records"])
            self.assertEqual(first["trajectories"], 2)
            self.assertTrue((path / "run" / "manifest.json").is_file())
            self.assertTrue((path / "run" / "results" / "summary.json").is_file())
            with self.assertRaisesRegex(Exception, "not empty"):
                run(data, output=path / "run")
            data["beads"] = 2
            with self.assertRaisesRegex(Exception, "changed scientific settings"):
                run(data, output=path / "run", resume=True)

    def test_cli_toml_from_foreign_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            inputs = base / "inputs"
            inputs.mkdir()
            (inputs / "A.xyz").write_text("1\nH atom, Angstrom\nH 0 0 0\n")
            (inputs / "B.xyz").write_text("1\nH atom, Angstrom\nH 0 0 0\n")
            (inputs / "run.inp").write_text("""task = "drpmd"
temperature = "300 K"
beads = 1
[system]
reactants = ["A.xyz", "B.xyz"]
separation = "4 bohr"
[potential]
provider = "free"
[dynamics]
timestep = "0.1 fs"
[sampling]
equilibration_steps = 2
production_steps = 2
interval_steps = 1
andersen_interval = 1
[collision]
bmax = "0.5 bohr"
sampling = "fixed_energy"
energy = "0.02 hartree"
trajectories = 2
max_steps = 400
[outcomes]
persistence_steps = 2
bond_cutoffs = { H-H = "0.3 bohr" }
[output]
checkpoint_steps = 20
""")
            environment = dict(os.environ)
            src = str(Path(__file__).resolve().parents[1] / "src")
            environment["PYTHONPATH"] = (
                src + os.pathsep + environment.get("PYTHONPATH", "")
            )

            def command(*args):
                return subprocess.run(
                    [sys.executable, "-m", "pyrpmd", *args],
                    cwd=base,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=60,
                )

            checked = command("validate", str(inputs / "run.inp"))
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertTrue(json.loads(checked.stdout)["valid"])
            executed = command(
                "run", str(inputs / "run.inp"), "--output", str(base / "result")
            )
            self.assertEqual(executed.returncode, 0, executed.stderr)
            result = json.loads(executed.stdout)
            self.assertEqual(result["counts"]["nonreactive"], 2)
            self.assertIsNone(result["total_reaction"]["rate_cm3_molecule_s"])
            resumed = command("resume", str(base / "result"))
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertEqual(json.loads(resumed.stdout)["records"], result["records"])
            analyzed = command("analyze", str(base / "result"))
            self.assertEqual(analyzed.returncode, 0, analyzed.stderr)
            self.assertEqual(json.loads(analyzed.stdout)["software"], "PyRPMD")


if __name__ == "__main__":
    unittest.main()
