"""Fail fast on user input and keep failed preflight requests side-effect free."""

import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pyrpmd.cli import main
from pyrpmd.constants import KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import prepare, resume_run, run
from pyrpmd.workflows.fes import _analysis_ranges


def small_fes():
    return {
        "task": "fes", "temperature": 1 / KB_AU, "beads": 1,
        "native": False, "seed": 71,
        "system": {"symbols": ["H"], "masses": [2.], "geometry": [[0., 0., 0.]]},
        "potential": {"provider": "harmonic", "params": {"k": .4}},
        "dynamics": {"timestep": .01},
        "sampling": {"equilibration_steps": 2, "production_steps": 20,
                     "sample_interval": 2, "andersen_interval": 5},
        "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
        "umbrella": {"cvs": ["x"], "centers": [0.], "stiffness": 1.},
        "analysis": {"bins": 4, "ranges": [[-2., 2.]], "bootstrap": 0},
        "output": {"checkpoint_steps": 10},
    }


class FESPreflightTests(unittest.TestCase):
    def assert_rejected_without_output(self, section, changes, message):
        data = small_fes()
        data[section].update(changes)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-run"
            with mock.patch("pyrpmd.potentials.make_potential") as provider:
                for operation in (
                    lambda: JobSpec.from_mapping(data).resolve(),
                    lambda: prepare(data),
                    lambda: run(data, output=output),
                ):
                    with self.assertRaisesRegex(InputError, message):
                        operation()
                provider.assert_not_called()
            self.assertFalse(output.exists())

    def test_invalid_windows_rejected_before_output_or_provider(self):
        for change, message in (
            ({"stiffness": -1.}, "positive definite"),
            ({"centers": [[0., 1.]]}, "dimension"),
            ({"centers": []}, "centers"),
            ({"cvs": ["missing"]}, "missing"),
            ({"stiffness": "1 hartree/angstrom"}, "unit|dimension"),
        ):
            with self.subTest(change=change):
                self.assert_rejected_without_output("umbrella", change, message)

    def test_invalid_analysis_rejected_before_sampling(self):
        for change, message in (
            ({"bins": 1}, "bins"),
            ({"bins": 2.5}, "bins"),
            ({"bins": True}, "bins"),
            ({"bins": [4, 4]}, "bins"),
            ({"bins": [[0., 1., 1.]]}, "edges"),
            ({"bins": [[0., float("nan"), 1.]]}, "edges"),
            ({"ranges": [[1., -1.]]}, "increasing"),
            ({"ranges": [[-1., 1.], [-2., 2.]]}, "ranges"),
            ({"ranges": [[0.]]}, "ranges"),
            ({"bootstrap": 1}, "bootstrap"),
            ({"overlap_threshold": -1.}, "overlap_threshold"),
            ({"overlap_threshold": float("nan")}, "overlap_threshold"),
        ):
            with self.subTest(change=change):
                self.assert_rejected_without_output("analysis", change, message)

    def test_insufficient_recording_budget_rejected_before_sampling(self):
        for production in (1, 2, 3):
            with self.subTest(production=production):
                self.assert_rejected_without_output(
                    "sampling", {"production_steps": production}, "two saved samples"
                )

    def test_extra_ranges_cannot_be_silently_truncated_during_reanalysis(self):
        from pyrpmd.cv import Cartesian, CVStack

        cv = CVStack([Cartesian(0, "x")])
        with self.assertRaisesRegex(ValueError, "one.*pair per CV"):
            _analysis_ranges({"ranges": [[-1., 1.], [-2., 2.]]}, cv)
        self.assertEqual(_analysis_ranges({"ranges": [["-1 bohr", "1 bohr"]]}, cv), [[-1., 1.]])

    def test_invalid_reanalysis_preserves_saved_outputs_and_checkpoints(self):
        from pyrpmd.workflows.fes import reanalyze_fes

        data = small_fes()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run(data, output=output)
            before = {path.relative_to(output): path.read_bytes()
                      for path in output.rglob("*") if path.is_file()}
            config, system = JobSpec.from_mapping(data).resolve()
            config["analysis"]["ranges"] = [[-2., 2.], [-3., 3.]]
            with self.assertRaisesRegex(RuntimeError, "analysis.ranges"):
                reanalyze_fes(config, system, output)
            after = {path.relative_to(output): path.read_bytes()
                     for path in output.rglob("*") if path.is_file()}
            self.assertEqual(after, before)

    def test_explicit_edges_and_units_remain_supported(self):
        data = small_fes()
        data["analysis"].update(bins=[[-2., -.2, .1, 2.]], ranges=[["-2 bohr", "2 bohr"]])
        before = copy.deepcopy(data)
        config, _ = JobSpec.from_mapping(data).resolve()
        self.assertEqual(config["analysis"]["bins"], data["analysis"]["bins"])
        self.assertEqual(data, before)
        self.assertTrue(prepare(data)["ready"])

    def test_valid_run_resume_and_saved_samples_still_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            first = run(small_fes(), output=output)
            checkpoint = next((output / "checkpoints").glob("*.npz"))
            before = checkpoint.read_bytes()
            again = resume_run(output / "checkpoints")
            self.assertEqual(first["samples"], 10)
            self.assertEqual(first["samples"], again["samples"])
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertTrue((output / "results" / "summary.json").is_file())

    def test_saved_surface_source_does_not_require_new_umbrella_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            run(small_fes(), output=source)
            data = small_fes()
            data["task"] = "surface_rate"
            del data["cvs"], data["umbrella"]
            data["analysis"] = {"method": "source", "bootstrap": 0}
            data["surface"] = {
                "reactant_basin": [-1., -.2], "dividing_surface": 0.,
                "free_energy": {"directory": str(source), "axis": 0},
            }
            # This workflow reuses the saved grid and samples recrossing
            # parents; it does not schedule new umbrella production.
            data["sampling"]["production_steps"] = 1
            data["recrossing"] = {
                "equilibration_steps": 2, "parent_interval_steps": 2, "child_steps": 4,
            }
            prepared = prepare(data)
            self.assertEqual(prepared["source_preparation"]["directory"], str(source))
            self.assertNotIn("umbrella_preparation", prepared)

    def test_cli_reports_the_field_and_leaves_no_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "bad.py"
            data = small_fes()
            data["analysis"]["bins"] = 1
            source.write_text("def build():\n    return " + repr(data) + "\n")
            output = Path(temporary) / "run"
            for command in ("validate", "run"):
                stderr, stdout = io.StringIO(), io.StringIO()
                arguments = [command, str(source), "--output", str(output)]
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                    status = main(arguments)
                self.assertEqual(status, 2)
                self.assertIn("analysis.bins", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(output.exists())


class ResumePathTests(unittest.TestCase):
    def test_missing_run_explains_the_expected_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with self.assertRaisesRegex(InputError, "run directory containing manifest.json"):
                resume_run(missing)
            self.assertFalse(missing.exists())

    def test_partial_run_reports_the_missing_saved_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "manifest.json").write_text(json.dumps({"schema": "pyrpmd.run"}))
            with self.assertRaisesRegex(InputError, "input/resolved.json is missing"):
                resume_run(directory)

    def test_output_file_is_preserved_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "existing.txt"
            destination.write_text("keep this content")
            with self.assertRaisesRegex(RuntimeError, "Output path is not a directory"):
                run(small_fes(), output=destination)
            self.assertEqual(destination.read_text(), "keep this content")


class ResumeParallelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "run"
        job = small_fes()
        # Historical resolved inputs can retain any of these accepted aliases.
        job["parallel"] = {"workers": 1, "trajectory_workers": 1, "trajectory_groups": 1}
        run(job, output=self.output)
        self.saved = {path.relative_to(self.output): path.read_bytes()
                      for path in self.output.rglob("*") if path.is_file()}

    def assert_saved_files_unchanged(self):
        after = {path.relative_to(self.output): path.read_bytes()
                 for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(after, self.saved)

    def test_worker_aliases_override_old_saved_aliases_consistently(self):
        for overrides in (
            {"workers": 2}, {"trajectory_workers": 2}, {"trajectory_groups": 2},
            {"workers": 2, "trajectory_workers": 2, "trajectory_groups": 2},
        ):
            with self.subTest(overrides=overrides), mock.patch("pyrpmd.runner.run_resolved") as dispatch:
                resume_run(self.output, parallel=overrides)
                config = dispatch.call_args.args[0]
                self.assertEqual(config["parallel"]["trajectory_workers"], 2)
                self.assertEqual(config["parallel"]["trajectory_groups"], 2)
                self.assertNotIn("workers", config["parallel"])
                self.assertTrue(dispatch.call_args.kwargs["resume"])
                original = json.loads(self.saved[Path("input/resolved.json")])
                self.assertEqual(
                    {key: value for key, value in config.items() if key != "parallel"},
                    {key: value for key, value in original.items() if key != "parallel"},
                )
                self.assert_saved_files_unchanged()

    def test_invalid_overrides_reject_before_dispatch_and_preserve_all_files(self):
        for overrides, message in (
            ({"trajectory_worker": 2}, "Did you mean.*trajectory_workers"),
            ({"workers": 0}, "integer"),
            ({"workers": True}, "integer"),
            ({"workers": 1.5}, "integer"),
            ({"workers": 2, "trajectory_groups": 3}, "disagree"),
            ({"trajectory_workers": 2, "trajectory_groups": 3}, "disagree"),
            ({"backend": "typo"}, "backend"),
            ({"scheduling": "dynamic"}, "Dynamic scheduling"),
            ({"threads_per_rank": 0}, "threads_per_rank"),
            ({1: 2}, "string field names"),
            ([2], "mapping"),
        ):
            with self.subTest(overrides=overrides), mock.patch("pyrpmd.runner.run_resolved") as dispatch:
                with self.assertRaisesRegex(InputError, message):
                    resume_run(self.output, parallel=overrides)
                dispatch.assert_not_called()
                self.assert_saved_files_unchanged()

    def test_mpi_worker_groups_and_scheduling_are_normalized_together(self):
        with mock.patch("pyrpmd.runner.run_resolved") as dispatch:
            resume_run(self.output, parallel={"backend": "mpi", "workers": 2, "scheduling": "dynamic"})
        parallel = dispatch.call_args.args[0]["parallel"]
        self.assertEqual(parallel["trajectory_workers"], 2)
        self.assertEqual(parallel["trajectory_groups"], 2)
        self.assertEqual(parallel["coordinator_ranks"], 1)
        self.assert_saved_files_unchanged()

    def test_real_worker_override_resumes_without_changing_input_or_checkpoints(self):
        result = resume_run(self.output, parallel={"workers": 2})
        self.assertEqual(result["samples"], 10)
        for path, data in self.saved.items():
            if path.parts[0] in ("input", "checkpoints") or path.name == "manifest.json":
                self.assertEqual((self.output / path).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
