"""Rate-backend integration, including real toy-PES propagation and recovery.

The analytic exchange model exercises dynamics and bookkeeping, not a physical
H + H2 rate. Analytic distribution accuracy tests live in test_mbar_rate_analysis.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from pyrpmd.input import InputError, JobSpec, rate_pmf_options
from pyrpmd.io import fingerprint
from pyrpmd.runner import run_resolved
from pyrpmd.workflows.rate import (
    _aggregate,
    _child_task,
    _child_tasks,
    _parent_task,
    _pmf_and_parent_tasks,
    _prepare,
    _recrossing_options,
    rate_coordinate,
)


class RatePMFInputTests(unittest.TestCase):
    def test_defaults_aliases_and_invalid_science_options(self):
        config = {}
        self.assertEqual(rate_pmf_options(config)["method"], "ui")
        self.assertNotIn("rate", config)
        options = rate_pmf_options(
            {
                "rate": {"pmf_method": "mbar", "dividing_surface": 1.00925},
                "recrossing": {"dividing_surface": 1.00925},
                "analysis": {"bins": [170], "ranges": [[-0.3, 1.4]], "bootstrap": 40},
            }
        )
        self.assertEqual(options["xi_star"], 1.00925)
        self.assertTrue(options["compare_ui"])
        self.assertEqual(options["ranges"], [-0.3, 1.4])
        for invalid in (
            {"rate": {"pmf_method": "guess"}},
            {"rate": {"dividing_surface": True}},
            {"rate": {"dividing_surface": float("nan")}},
            {
                "rate": {"dividing_surface": 1.0},
                "recrossing": {"dividing_surface": 1.1},
            },
            {"rate": {"pmf_search_range": [0.9, 0.8]}},
            {"rate": {"pmf_method": "mbar", "compare_ui": 1}},
            {"rate": {"pmf_method": "mbar"}, "analysis": {"bootstrap": 1}},
            {"rate": {"pmf_method": "mbar"}, "analysis": {"bins": 2}},
            {"rate": {"pmf_method": "mbar"}, "analysis": {"bins": 2.9}},
            {"rate": {"pmf_method": "mbar"}, "analysis": {"ranges": [[1.0, 0.0]]}},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(InputError):
                rate_pmf_options(invalid)


class RateMBARIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "examples/rate/pilot.inp"
        spec = JobSpec.from_file(source)
        spec.data["native"] = False
        spec.data["rate"] = {"pmf_method": "mbar", "dividing_surface": 1.0}
        spec.data["analysis"] = {
            "bins": 24,
            "ranges": [[-0.3, 1.3]],
            "bootstrap": 3,
            "min_count": 2,
        }
        spec.data["recrossing"].update(
            parent_replicas=1,
            parent_samples=2,
            child_pairs=1,
            equilibration="1 fs",
            parent_interval="0.2 fs",
            child_time="2 fs",
            bootstrap=4,
        )
        cls.config, cls.system = spec.resolve()
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "run"
        cls.result = run_resolved(cls.config, cls.system, output=cls.output)
        cls.prepared = _prepare(cls.config, cls.system, cls.output)
        cls.options = _recrossing_options(cls.prepared)
        cls.streams = []
        with h5py.File(cls.output / "umbrella/fes.h5", "r") as stored:
            for key in sorted(stored["streams"], key=int):
                group = stored["streams"][key]
                window, replica = int(group.attrs["window"]), int(
                    group.attrs["replica"]
                )
                path = (
                    cls.output
                    / "umbrella/checkpoints"
                    / f"window_{window:05d}_replica_{replica:04d}.npz"
                )
                with np.load(path, allow_pickle=False) as checkpoint:
                    cls.streams.append(
                        {
                            "window": window,
                            "replica": replica,
                            "samples": np.asarray(group["samples"]),
                            "center": np.asarray(group["center"]),
                            "stiffness": np.asarray(group["stiffness"]),
                            "last_q": np.asarray(checkpoint["q"]),
                        }
                    )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def sampled(self):
        return {"streams": self.streams, "data": {}, "summary": {"warnings": []}}

    def test_actual_pipeline_writes_supported_pmf_and_records_all_draws(self):
        result = self.result
        self.assertTrue(result["pmf"]["ready"])
        self.assertEqual(result["pmf"]["method"], "mbar")
        self.assertEqual(result["pmf"]["sampling_measure"], "rate_flux")
        self.assertEqual(result["pmf"]["xi_star"], 1.0)
        self.assertEqual(result["child_trajectories"], 4)
        self.assertFalse(result["converged"])
        bootstrap = result["pmf"]["bootstrap"]
        self.assertEqual(bootstrap["requested"], 3)
        self.assertEqual(bootstrap["attempted"], 3)
        self.assertEqual(bootstrap["valid"] + len(bootstrap["failures"]), 3)
        self.assertTrue(bootstrap["surface_held_fixed"])
        self.assertTrue(result["pmf"]["ui_comparison"]["ready"])
        summary = json.loads((self.output / "umbrella/summary.json").read_text())
        self.assertEqual(summary["bootstrap_replicates"], 0)
        self.assertEqual(result["pmf"]["intermediate_fes_bootstrap"], 0)
        with np.load(self.output / "rate_pmf.npz", allow_pickle=False) as saved:
            for key in (
                "xi",
                "free_energy",
                "mask",
                "standard_error",
                "density",
                "raw_count",
                "edges",
            ):
                self.assertIn(key, saved.files)
            self.assertEqual(str(saved["method"]), "mbar")
            self.assertEqual(float(saved["xi_star"]), 1.0)
            self.assertTrue(np.isfinite(saved["free_energy"][~saved["mask"]]).all())
            self.assertTrue(np.isnan(saved["free_energy"][saved["mask"]]).all())
        with np.load(
            self.output / "checkpoints/parent_0000.npz", allow_pickle=False
        ) as saved:
            surface = rate_coordinate(self.prepared, self.system).dividing_surface(1.0)
            self.assertEqual(str(saved["pmf_method"]), "mbar")
            for q in saved["snapshots"]:
                self.assertLess(
                    abs(surface.centroid_value_gradient(q.mean(0))[0]), 1e-8
                )
        json.dumps(result, allow_nan=False)

    def test_normal_resume_is_exact_and_rejects_method_or_surface_change(self):
        files = [self.output / "rate_pmf.npz", self.output / "recrossing.npz"]
        before = {}
        for path in files:
            with np.load(path, allow_pickle=False) as saved:
                before[path] = {
                    key: (saved[key].dtype.str, saved[key].shape, saved[key].tobytes())
                    for key in saved.files
                }
        resumed = run_resolved(
            self.config, self.system, output=self.output, resume=True
        )
        self.assertEqual(resumed["kappa"], self.result["kappa"])
        self.assertEqual(resumed["pmf"], self.result["pmf"])
        for path, expected in before.items():
            with np.load(path, allow_pickle=False) as saved:
                actual = {
                    key: (saved[key].dtype.str, saved[key].shape, saved[key].tobytes())
                    for key in saved.files
                }
            self.assertEqual(actual, expected)
        for key, value in (("pmf_method", "ui"), ("dividing_surface", 1.01)):
            changed = copy.deepcopy(self.config)
            changed["rate"][key] = value
            self.assertNotEqual(
                fingerprint(self.config, self.system), fingerprint(changed, self.system)
            )
            with self.assertRaisesRegex(RuntimeError, "changed scientific settings"):
                run_resolved(changed, self.system, output=self.output, resume=True)

    def test_masked_or_missing_endpoint_withholds_rate_and_children(self):
        config = copy.deepcopy(self.prepared)
        config["analysis"].update(ranges=[[-0.3, 0.8]], bootstrap=0)
        with tempfile.TemporaryDirectory() as output:
            pmf, tasks = _pmf_and_parent_tasks(
                self.sampled(), config, self.system, output, self.options
            )
            self.assertEqual(pmf["status"], "unsupported_pmf")
            self.assertTrue(pmf["overlap_connected"])
            self.assertFalse(pmf["ready"])
            self.assertIsNone(pmf["k_qtst_cm3_s"])
            self.assertIsNone(pmf["barrier_standard_error_hartree"])
            self.assertEqual(tasks, [])
            result = _aggregate([], pmf, config, output, self.options)
            self.assertEqual(result["status"], "unsupported_pmf")
            self.assertIsNone(result["rate_estimate_cm3_s"])
            self.assertFalse((Path(output) / "recrossing.npz").exists())

    def test_auto_surface_is_actual_supported_center_and_has_its_own_dynamics(self):
        config = copy.deepcopy(self.prepared)
        config["rate"].pop("dividing_surface")
        config["recrossing"].pop("dividing_surface")
        config["rate"]["pmf_search_range"] = [0.6, 0.9]
        config["analysis"]["bootstrap"] = 0
        options = dict(
            self.options,
            parent_samples=1,
            equilibration_steps=2,
            parent_interval_steps=1,
        )
        with tempfile.TemporaryDirectory() as output:
            pmf, tasks = _pmf_and_parent_tasks(
                self.sampled(), config, self.system, output, options
            )
            self.assertTrue(pmf["ready"])
            xi = pmf["xi_star"]
            self.assertTrue(0.6 <= xi <= 0.9)
            self.assertNotEqual(xi, 1.0)
            with np.load(Path(output) / "rate_pmf.npz", allow_pickle=False) as saved:
                selected = np.flatnonzero(saved["xi"] == xi)
                self.assertEqual(len(selected), 1)
                self.assertFalse(saved["mask"][selected[0]])
            self.assertEqual(tasks[0]["xi_star"], xi)
            parent = _parent_task(tasks[0])
            surface = rate_coordinate(config, self.system).dividing_surface(xi)
            self.assertLess(
                abs(surface.centroid_value_gradient(parent["snapshots"][0].mean(0))[0]),
                1e-8,
            )
            children = _child_tasks([parent], config, self.system, output, options, pmf)
            child = _child_task(children[0])
            self.assertEqual(child["xi_star"], xi)
            self.assertEqual(child["pmf_method"], "mbar")
            self.assertEqual(len(child["numerator"]), options["child_steps"])
            wrong = dict(tasks[0], xi_star=1.0)
            with self.assertRaisesRegex(ValueError, "checkpoint physics differs"):
                _parent_task(wrong)
            wrong_child = dict(children[0], xi_star=1.0)
            with self.assertRaisesRegex(ValueError, "checkpoint physics differs"):
                _child_task(wrong_child)

    def test_disconnected_overlap_is_not_reported_as_missing_bins(self):
        rng = np.random.default_rng(12)
        streams = []
        for center in (0.0, 1.0, 1.1):
            streams.append(
                {
                    "samples": rng.normal(center, 1e-5, 40)[:, None],
                    "center": np.array([center]),
                    "stiffness": np.array([[1e8]]),
                    "last_q": self.streams[0]["last_q"],
                }
            )
        config = copy.deepcopy(self.prepared)
        config["analysis"]["bootstrap"] = 0
        with tempfile.TemporaryDirectory() as output:
            pmf, tasks = _pmf_and_parent_tasks(
                {"streams": streams, "data": {}, "summary": {"warnings": []}},
                config,
                self.system,
                output,
                self.options,
            )
            self.assertEqual(pmf["status"], "insufficient_overlap")
            self.assertEqual(pmf["failure_reason"], "disconnected_overlap")
            self.assertFalse(pmf["overlap_connected"])
            self.assertEqual(tasks, [])


class RateUncertaintyTests(unittest.TestCase):
    def test_missing_pmf_bootstrap_is_not_a_complete_rate_error(self):
        children = [
            {
                "chain": 0,
                "parent": i,
                "pair": 0,
                "numerator": np.full(40, flux),
                "denominator": 1.0,
                "energy_drift": 0.0,
                "endpoint_energy_error_hartree_per_bead": 0.0,
            }
            for i, flux in enumerate((0.7, 0.9))
        ]
        pmf = {"k_qtst_cm3_s": 2.0, "free_energy_barrier_hartree": 0.01, "warnings": []}
        config = {"seed": 1, "beta": 5.0, "dt_au": 0.1}
        options = {
            "child_steps": 40,
            "plateau_fraction": 0.5,
            "plateau_tolerance": 0.1,
            "bootstrap": 20,
        }
        with tempfile.TemporaryDirectory() as output:
            incomplete = _aggregate(children, pmf, config, output, options)
            self.assertIsNone(incomplete["rate_standard_error_cm3_s"])
            self.assertIsNotNone(incomplete["conditional_rate_standard_error_cm3_s"])
            self.assertIn(
                "fewer_than_two_valid_pmf_bootstrap_draws",
                incomplete["rate_uncertainty"]["reasons"],
            )
            pmf["pmf_bootstrap_barriers"] = [0.005, 0.015]
            complete = _aggregate(children, pmf, config, output, options)
            self.assertTrue(complete["rate_uncertainty"]["available"])
            self.assertGreater(complete["rate_standard_error_cm3_s"], 0.0)
            self.assertEqual(complete["rate_uncertainty"]["reasons"], [])


if __name__ == "__main__":
    unittest.main()
