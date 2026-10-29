import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.workflows import drpmd


def system():
    return {
        "symbols": ["H", "H"],
        "masses": np.array([1.0, 1.0]),
        "geometry": np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]),
        "fragments": [[0], [1]],
    }


def config():
    return {
        "beta": 1.0,
        "temperature_K": 1 / drpmd.KB,
        "beads": 1,
        "dt_au": 0.1,
        "seed": 17,
        "native": False,
        "system": {"separation_au": 4.0},
        "potential": {"provider": "free"},
        "sampling": {
            "equilibration_steps": 4,
            "interval_steps": 2,
            "andersen_interval": 2,
        },
        "collision": {"trajectories": 5, "bmax_au": 0.5, "max_steps": 200},
        "outcomes": {
            "bond_cutoffs_au": {"H-H": 0.3},
            "channels": {},
            "exit_distance_au": 4.0,
            "persistence_steps": 2,
            "check_interval": 1,
        },
        "output": {"trajectory_stride": 5, "checkpoint_steps": 10},
        "parallel": {"backend": "serial"},
    }


class CollisionTests(unittest.TestCase):
    def test_flux_and_uniform_area_distributions(self):
        rng = np.random.default_rng(234)
        events = [drpmd.sample_collision(2.0, 3.0, 4.0, 5.0, rng) for _ in range(20000)]
        e = np.array([v["collision_energy_au"] for v in events])
        b = np.array([v["impact_parameter_au"] for v in events])
        self.assertAlmostEqual(float(e.mean()), 1.0, delta=0.025)
        self.assertAlmostEqual(float(e.var()), 0.5, delta=0.025)
        self.assertAlmostEqual(float(np.mean(b * b)), 8.0, delta=0.15)

    def test_launch_preserves_internal_modes_and_radial_distance(self):
        rng = np.random.default_rng(5)
        masses = np.array([1.0, 2.0, 3.0])
        fragments = [[0], [1, 2]]
        parent = State(rng.normal(size=(8, 3, 3)), rng.normal(size=(8, 3, 3)))
        child, event = drpmd.launch_collision(
            parent, masses, fragments, 2.0, 1.0, 6.0, rng
        )
        c = [drpmd.fragment_com(child.q, masses, f) for f in fragments]
        self.assertAlmostEqual(float(np.linalg.norm(c[1] - c[0])), 6.0)
        np.testing.assert_allclose(child.p.mean(axis=0).sum(axis=0), 0.0, atol=1e-14)
        for f in fragments:
            old = parent.q[:, f] - drpmd.fragment_com(parent.q, masses, f)
            new = child.q[:, f] - drpmd.fragment_com(child.q, masses, f)
            self.assertAlmostEqual(
                float(np.sum(old * old)), float(np.sum(new * new)), places=11
            )
        relative = drpmd.fragment_velocity(
            child.p, masses, fragments[1]
        ) - drpmd.fragment_velocity(child.p, masses, fragments[0])
        self.assertAlmostEqual(float(np.linalg.norm(relative)), event["speed_au"])

    def test_two_parent_chains_split_total_and_keep_sampling_clock(self):
        c = config()
        c["parallel"] = {"backend": "process", "trajectory_workers": 2}
        with tempfile.TemporaryDirectory() as directory:
            result = drpmd.run_drpmd(c, system(), directory)
            self.assertEqual(result["trajectories"], 5)
            self.assertEqual(result["counts"]["nonreactive"], 5)
            self.assertEqual(result["parent_thermal_steps"], {"0": 10, "1": 8})
            self.assertEqual(
                [r["trajectory_id"] for r in result["records"]], list(range(5))
            )
            self.assertEqual(
                [r["parent_id"] for r in result["records"]], [0, 0, 0, 1, 1]
            )
            for r in result["records"]:
                path = (
                    Path(directory) / "states" / f"initial-{r['trajectory_id']:08d}.npz"
                )
                with np.load(path, allow_pickle=False) as state:
                    self.assertIn("thermal_p", state)
                    self.assertEqual(state["q"].shape, (1, 2, 3))

    def test_bead_and_trajectory_parallelism_together(self):
        c = config()
        c["beads"] = 3
        c["collision"].update(trajectories=2, max_steps=2)
        c["parallel"] = {
            "backend": "process",
            "mode": "bead",
            "trajectory_workers": 2,
            "bead_workers": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = drpmd.run_drpmd(c, system(), directory)
            self.assertEqual(result["trajectories"], 2)
            self.assertEqual(result["counts"]["unresolved"], 2)
            self.assertIsNone(result["total_reaction"]["rate_cm3_molecule_s"])

    def test_complete_restart_no_duplicate_trajectories(self):
        c = config()
        c["collision"]["trajectories"] = 2
        with tempfile.TemporaryDirectory() as directory:
            first = drpmd.run_drpmd(c, system(), directory)
            c["resume"] = True
            second = drpmd.run_drpmd(c, system(), directory)
            self.assertEqual(first["records"], second["records"])
            c["beads"] = 2
            with self.assertRaises(ValueError):
                drpmd.run_drpmd(c, system(), directory)

    def test_restart_fingerprint_tracks_local_pes_contents(self):
        c = config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"original model")
            c["potential"]["params"] = {"model": str(path)}
            original = drpmd._physics_hash(c, system(), 1)
            path.write_bytes(b"changed model")
            self.assertNotEqual(original, drpmd._physics_hash(c, system(), 1))

    def test_terminal_checkpoint_does_not_advance_after_restart(self):
        c = config()
        c["collision"]["trajectories"] = 1
        save = drpmd._atomic_npz
        interrupted = False

        def save_then_interrupt(path, metadata, **arrays):
            nonlocal interrupted
            save(path, metadata, **arrays)
            if (
                not interrupted
                and metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1"
                and metadata.get("status") == "nonreactive"
            ):
                interrupted = True
                raise KeyboardInterrupt("interruption after terminal classification")

        with (
            tempfile.TemporaryDirectory() as full,
            tempfile.TemporaryDirectory() as resumed,
        ):
            reference = drpmd.run_drpmd(c, system(), full)
            with patch.object(drpmd, "_atomic_npz", side_effect=save_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    drpmd.run_drpmd(c, system(), resumed)
            c["resume"] = True
            actual = drpmd.run_drpmd(c, system(), resumed)
            self.assertEqual(
                actual["records"][0]["steps"], reference["records"][0]["steps"]
            )
            with (
                np.load(Path(full) / "trajectories/trajectory-00000000.npz") as a,
                np.load(Path(resumed) / "trajectories/trajectory-00000000.npz") as b,
            ):
                for key in (
                    "final_q",
                    "final_p",
                    "steps",
                    "centroid",
                    "energy_per_bead",
                ):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_interrupted_child_resumes_identically(self):
        c = config()
        c["collision"]["trajectories"] = 1
        c["output"]["checkpoint_steps"] = 3
        save = drpmd._atomic_npz
        interrupted = False

        def save_then_interrupt(path, metadata, **arrays):
            nonlocal interrupted
            save(path, metadata, **arrays)
            if (
                not interrupted
                and metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1"
                and metadata["done"] == 3
            ):
                interrupted = True
                raise KeyboardInterrupt("simulated scheduler interruption")

        with (
            tempfile.TemporaryDirectory() as full,
            tempfile.TemporaryDirectory() as resumed,
        ):
            drpmd.run_drpmd(c, system(), full)
            with patch.object(drpmd, "_atomic_npz", side_effect=save_then_interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    drpmd.run_drpmd(c, system(), resumed)
            c["resume"] = True
            drpmd.run_drpmd(c, system(), resumed)
            with (
                np.load(Path(full) / "trajectories/trajectory-00000000.npz") as a,
                np.load(Path(resumed) / "trajectories/trajectory-00000000.npz") as b,
            ):
                for key in (
                    "final_q",
                    "final_p",
                    "steps",
                    "centroid",
                    "energy_per_bead",
                ):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_flux_rate_and_unknown_bounds(self):
        c = config()
        records = [
            {
                "trajectory_id": i,
                "parent_id": i % 2,
                "status": "product",
                "channel": "test",
                "launch": {"speed_au": 2.0},
            }
            for i in range(10)
        ]
        summary = drpmd.summarize(records, c, system())
        expected = np.pi * 0.5**2 * np.sqrt(8 / (np.pi * 0.5)) * drpmd.RATE_AU_TO_CM3_S
        self.assertAlmostEqual(
            summary["total_reaction"]["rate_cm3_molecule_s"], expected
        )
        records[-1]["status"] = "failed"
        summary = drpmd.summarize(records, c, system())
        self.assertIsNone(summary["total_reaction"]["rate_cm3_molecule_s"])
        self.assertAlmostEqual(
            summary["total_reaction"]["observed_lower_cm3_molecule_s"], 0.9 * expected
        )
        self.assertAlmostEqual(
            summary["total_reaction"]["possible_upper_cm3_molecule_s"], expected
        )

    def test_initial_pes_failure_counts_and_preserves_launch(self):
        c = config()
        c["collision"]["trajectories"] = 2
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                drpmd.Dynamics, "energy", side_effect=ValueError("outside PES domain")
            ):
                result = drpmd.run_drpmd(c, system(), directory)
            self.assertEqual(result["counts"]["failed"], 2)
            self.assertIsNone(result["total_reaction"]["rate_cm3_molecule_s"])
            self.assertIsNone(result["records"][0]["initial_energy_per_bead"])
            with np.load(
                Path(directory) / "trajectories/trajectory-00000000.npz"
            ) as saved:
                self.assertEqual(saved["centroid"].shape, (0, 2, 3))
                self.assertEqual(saved["energy_per_bead"].size, 0)
                self.assertTrue(np.isfinite(saved["final_q"]).all())

    def test_dynamics_section_native_option_reaches_both_integrators(self):
        c = config()
        del c["native"]
        c["dynamics"] = {"native": False}
        c["collision"].update(trajectories=1, max_steps=1)
        original, selected = drpmd.Dynamics, []

        def factory(*args, **kwargs):
            selected.append(kwargs["native"])
            return original(*args, **kwargs)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(drpmd, "Dynamics", side_effect=factory),
        ):
            drpmd.run_drpmd(c, system(), directory)
        self.assertEqual(selected, [False, False])

    def test_two_structure_pilot_infers_and_withholds_chemical_estimates(self):
        c = config()
        c["collision"]["trajectories"] = 2
        c["outcomes"] = {"check_interval": 1, "persistence_steps": 2}
        resolved = drpmd.resolve_pilot_outcomes(
            system(), c["outcomes"], separation_au=4.0
        )
        self.assertEqual(
            resolved,
            drpmd.resolve_pilot_outcomes(system(), resolved, separation_au=4.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = drpmd.run_drpmd(c, system(), directory)
            self.assertEqual(result["counts"]["nonreactive"], 2)
            self.assertTrue(result["classification_unvalidated"])
            self.assertGreater(
                result["classification"]["bond_cutoffs_bohr"]["H-H"], 1.0
            )
            self.assertTrue(result["classification"]["assumptions"])
            for key in (
                "rate_cm3_molecule_s",
                "cross_section_bohr2",
                "observed_lower_cm3_molecule_s",
                "possible_upper_cm3_molecule_s",
            ):
                self.assertIsNone(result["total_reaction"][key])
            c["resume"] = True
            self.assertEqual(
                result["records"], drpmd.run_drpmd(c, system(), directory)["records"]
            )

    def test_pilot_discovers_composition_but_leaves_exchange_unresolved(self):
        molecule = {
            "symbols": ["H", "H", "O"],
            "masses": np.array([1.0, 1.0, 16.0]),
            "fragments": [[0], [1, 2]],
            "geometry": np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.5, 0.0, 0.0]]),
        }
        q = np.array([[[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [10.0, 0.0, 0.0]]])
        p = np.array([[[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [16.0, 0.0, 0.0]]])
        c = config()
        c["outcomes"] = drpmd.resolve_pilot_outcomes(molecule, separation_au=4.0)
        classifier = drpmd._make_classifier(c, molecule)
        self.assertEqual(
            classifier(State(q, p)), {"status": "product", "channel": "candidate:H2+O"}
        )
        molecule["symbols"] = ["H", "H", "H"]
        molecule["masses"] = np.ones(3)
        molecule["geometry"][2, 0] = 6.2
        c["outcomes"] = drpmd.resolve_pilot_outcomes(molecule, separation_au=4.0)
        classifier = drpmd._make_classifier(c, molecule)
        self.assertIsNone(classifier(State(q, p)))
        same_q = np.array([[[-10.0, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]])
        same_p = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        self.assertEqual(
            classifier(State(same_q, same_p)),
            {"status": "nonreactive", "channel": None},
        )

    def test_pilot_inference_rejects_unknown_radii_or_production(self):
        missing = system()
        missing["symbols"] = ["Xe", "H"]
        with self.assertRaisesRegex(ValueError, "no covalent radii"):
            drpmd.resolve_pilot_outcomes(missing, separation_au=4.0)
        with self.assertRaisesRegex(ValueError, "Production dRPMD requires explicit"):
            drpmd.resolve_pilot_outcomes(
                system(), separation_au=4.0, preset="production"
            )
        inconsistent = system()
        inconsistent["geometry"][1] = inconsistent["geometry"][0] + np.array(
            [0.5, 0.0, 0.0]
        )
        with self.assertRaisesRegex(ValueError, "initial connectivity does not match"):
            drpmd.resolve_pilot_outcomes(inconsistent, separation_au=4.0)


if __name__ == "__main__":
    unittest.main()
