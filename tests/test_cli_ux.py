"""First-use behavior from a foreign working directory and actionable failures."""

from contextlib import redirect_stderr, redirect_stdout
from importlib.resources import files
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import tomllib
import unittest

from pyrpmd.cli import main, parser
from pyrpmd.input import InputError, JobSpec
from pyrpmd.scaffold import _command_text, init_project


class StarterTests(unittest.TestCase):
    def test_tutorial_runs_from_unrelated_directory_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

            def command(arguments):
                result = subprocess.run(arguments, cwd=base, env=environment,
                                        text=True, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout)

            starter = command([sys.executable, "-m", "pyrpmd", "init", "project with spaces"])
            input_path = base / "project with spaces" / "quickstart.inp"
            output = input_path.parent / "run"
            self.assertEqual(starter["input"], str(input_path))
            self.assertFalse(output.exists())
            if os.name != "nt":
                self.assertEqual(starter["next_steps_shell"], "posix")
                self.assertEqual([shlex.split(step) for step in starter["next_steps"]],
                                 starter["next_step_arguments"])
            validated, run, analyzed, resumed = [command(step) for step in starter["next_step_arguments"]]
            self.assertTrue(validated["valid"])
            self.assertTrue(validated["ready"])
            self.assertEqual(validated["output"], str(output))
            self.assertEqual(run["task"], "fes")
            self.assertEqual((run["windows"], run["streams"], run["samples"]), (5, 10, 20000))
            self.assertFalse(run["converged"])
            self.assertEqual(run, analyzed)
            self.assertEqual(run, resumed)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "fes.npz").is_file())

    def test_init_preserves_existing_files_and_rejects_second_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            result = init_project(target)
            template = Path(result["input"]).read_bytes()
            with self.assertRaisesRegex(InputError, "not empty"):
                init_project(target)
            self.assertEqual(Path(result["input"]).read_bytes(), template)
            self.assertEqual(sorted(p.name for p in target.iterdir()), ["quickstart.inp"])

    def test_init_refuses_file_and_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            sentinel = target / "important.txt"
            sentinel.write_text("preserve me")
            with self.assertRaisesRegex(InputError, "not empty"):
                init_project(target)
            with self.assertRaisesRegex(InputError, "not a directory"):
                init_project(sentinel)
            self.assertEqual(sentinel.read_text(), "preserve me")
            self.assertFalse((target / "quickstart.inp").exists())

    def test_windows_next_command_quotes_literal_paths(self):
        command = _command_text([r"C:\Program Files\Python\python.exe", "-m", "pyrpmd",
                                 "run", r"C:\O'Brien\$project\quickstart.inp"], windows=True)
        self.assertEqual(command, r"& 'C:\Program Files\Python\python.exe' '-m' 'pyrpmd' 'run' 'C:\O''Brien\$project\quickstart.inp'")

    @unittest.skipIf(os.name == "nt", "Creating symlinks may need Windows administrator privileges")
    def test_init_refuses_directory_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / "real").mkdir()
            (base / "alias").symlink_to(base / "real", target_is_directory=True)
            with self.assertRaisesRegex(InputError, "symbolic link"):
                init_project(base / "alias")
            self.assertEqual(list((base / "real").iterdir()), [])


class DiagnosticTests(unittest.TestCase):
    @staticmethod
    def invoke(arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_validation_explains_paths_without_creating_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            starter = init_project(base / "starter")
            for extra in ([], ["--check-pes"]):
                with self.subTest(extra=extra):
                    output = base / "elsewhere" / "run"
                    status, stdout, stderr = self.invoke([
                        "validate", starter["input"], "--output", str(output), *extra,
                    ])
                    self.assertEqual(status, 0, stderr)
                    result = json.loads(stdout)
                    self.assertTrue(result["valid"])
                    self.assertEqual(result["input"], starter["input"])
                    self.assertEqual(result["output"], str(output))
                    self.assertEqual(result["preset"], "pilot")
                    self.assertFalse(output.parent.exists())

    def test_syntax_error_contains_input_file_and_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, contents in (("bad.inp", 'task = "fes"\n[system\n'),
                                   ("bad.py", "def build(:\n    return {}\n")):
                with self.subTest(name=name):
                    path = (Path(temporary) / name).resolve()
                    path.write_text(contents)
                    status, stdout, stderr = self.invoke(["validate", str(path)])
                    self.assertEqual(status, 2)
                    self.assertEqual(stdout, "")
                    self.assertIn(str(path), stderr)
                    self.assertNotIn("Traceback", stderr)
                    self.assertIn("--debug", stderr)

    def test_debug_works_before_or_after_subcommand(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = str(Path(temporary) / "missing.inp")
            status, stdout, stderr = self.invoke(["validate", missing])
            self.assertEqual(status, 2)
            self.assertIn("Input file not found", stderr)
            self.assertNotIn("Traceback", stderr)
            for arguments in (["--debug", "validate", missing], ["validate", missing, "--debug"]):
                with self.subTest(arguments=arguments), self.assertRaisesRegex(InputError, "Input file not found"):
                    main(arguments)
        for arguments in (["--debug", "pes", "inspect", "missing.so"],
                          ["pes", "--debug", "inspect", "missing.so"],
                          ["pes", "inspect", "missing.so", "--debug"]):
            with self.subTest(arguments=arguments):
                self.assertTrue(parser().parse_args(arguments).debug)

    def test_typo_suggestions_still_reject_input(self):
        template = files("pyrpmd").joinpath("templates", "quickstart.inp").read_text(encoding="utf-8")
        for section, typo, correct in ((None, "temprature", "temperature"),
                                       ("sampling", "production_step", "production_steps")):
            with self.subTest(section=section):
                data = tomllib.loads(template)
                target = data if section is None else data[section]
                target[typo] = target.pop(correct)
                with self.assertRaises(InputError) as caught:
                    JobSpec.from_mapping(data).resolve()
                self.assertIn(typo, str(caught.exception))
                self.assertIn(f"'{correct}'", str(caught.exception))
                self.assertIn("Did you mean", str(caught.exception))
                self.assertIn(typo, target)
                self.assertNotIn(correct, target)

    def test_cli_overrides_do_not_crash_on_malformed_tables(self):
        for section, override in (("output", {"output": "run"}),
                                  ("potential", {"pes": "pes.so"}),
                                  ("potential", {"pes_manifest": "manifest.json"}),
                                  ("parallel", {"workers": 2})):
            with self.subTest(section=section, override=override):
                spec = JobSpec.from_mapping({"task": "fes", section: "invalid"})
                with self.assertRaisesRegex(InputError, section + " must be a table"):
                    spec.with_overrides(**override)

    def test_info_includes_diagnostic_environment(self):
        status, stdout, stderr = self.invoke(["info"])
        self.assertEqual(status, 0, stderr)
        result = json.loads(stdout)
        environment = result["environment"]
        self.assertEqual(environment["executable"], sys.executable)
        self.assertTrue(environment["python"])
        self.assertTrue(environment["platform"])
        self.assertIn("h5py", environment["dependencies"])
        self.assertTrue(environment["dependencies"]["numpy"])

    def test_info_lists_every_runnable_task_and_schema_version(self):
        status, stdout, stderr = self.invoke(["info"])
        self.assertEqual(status, 0, stderr)
        result = json.loads(stdout)
        for task in ("rate", "fes", "path_sampling", "spectroscopy", "nonadiabatic", "plumed"):
            self.assertIn(task, result["workflows"])
        self.assertEqual(result["supported_input_schema_versions"], [1, 2])
        # Every input task is advertised and reaches its own resolver, not the
        # unknown-task error. The registry also lists workflows with their own command.
        from pyrpmd.input_schema import TASKS

        self.assertLessEqual(set(TASKS), set(result["workflows"]))
        for task in TASKS:
            with self.subTest(task=task):
                try:
                    JobSpec.from_mapping({"task": task}).resolve()
                except Exception as error:  # incomplete inputs fail later, for other reasons
                    self.assertNotIn("Unknown task", str(error))

    def test_misspelled_or_missing_task_names_the_choices(self):
        for task, suggestion in (("rates", "'rate'"), ("path-sampling", "'path_sampling'"),
                                 ("PIMD", "'pimd'"), ("metadynamic", "'metadynamics'")):
            with self.subTest(task=task), self.assertRaises(InputError) as caught:
                JobSpec.from_mapping({"task": task, "fes": {}}).resolve()
            self.assertIn(f"Unknown task {task!r}", str(caught.exception))
            self.assertIn(f"Did you mean {suggestion}?", str(caught.exception))
        for data in ({}, {"task": 3}, {"task": "zzz"}):
            with self.subTest(data=data), self.assertRaises(InputError) as caught:
                JobSpec.from_mapping(data).resolve()
            self.assertIn("choose one of: rate, drpmd, fes", str(caught.exception))
            self.assertNotIn("Did you mean", str(caught.exception))

    def test_missing_optional_dependency_names_the_extra_to_install(self):
        from pyrpmd.cli import _missing_dependency_hint

        hint = _missing_dependency_hint(ModuleNotFoundError("No module named 'torch'"))
        self.assertIn("pyrpmd[gpu]", hint)
        wrapped = RuntimeError("Root stage failed: ModuleNotFoundError: No module named 'mpi4py'")
        self.assertIn("pyrpmd[mpi]", _missing_dependency_hint(wrapped))
        self.assertIn("pyrpmd[ase]", _missing_dependency_hint(ImportError("No module named 'ase.io'")))
        self.assertIsNone(_missing_dependency_hint(ImportError("No module named 'my_private_pes'")))
        self.assertIsNone(_missing_dependency_hint(ValueError("invalid temperature")))

    def test_analyze_explains_a_missing_or_unfinished_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            status, stdout, stderr = self.invoke(["analyze", str(missing)])
            self.assertEqual((status, stdout), (2, ""))
            self.assertIn("No run directory at", stderr)
            self.assertNotIn("Errno", stderr)
            self.assertFalse(missing.exists())
            empty = Path(temporary) / "empty"
            empty.mkdir()
            status, stdout, stderr = self.invoke(["analyze", str(empty)])
            self.assertEqual((status, stdout), (2, ""))
            self.assertIn("No saved summary", stderr)
            self.assertIn("pyrpmd resume", stderr)
            status, stdout, stderr = self.invoke(["analyze", str(empty), "--recompute"])
            self.assertEqual((status, stdout), (2, ""))
            self.assertIn("No saved input", stderr)
            self.assertEqual(list(empty.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
