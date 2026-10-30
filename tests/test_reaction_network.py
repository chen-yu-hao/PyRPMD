"""Scientific identity, normalization and actual multi-path workflow checks."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from pyrpmd.constants import KB_AU
from pyrpmd.input import JobSpec
from pyrpmd.potentials import make_potential
from pyrpmd.reactions import resolve_network_job, network_jobs, validate_network
from pyrpmd.workflows.network import prepare_network, run_network
from pyrpmd.workflows.prepare import numerical_hessian


ROOT = Path(__file__).resolve().parents[1]


def network_spec():
    """Two different analytic teaching saddles/PESs and entrance partitions."""
    factory = str(ROOT / "examples" / "rate" / "toy_exchange.py") + ":build"
    r1 = 1.2 + np.log(3.0) / 1.2
    r2 = 1.2 + np.log(4.0) / 1.2
    return {
        "task": "network", "temperature": 1 / (5 * KB_AU), "beads": 2,
        "seed": 17, "native": False,
        "system": {"symbols": ["H", "H", "H"], "masses": [2.0, 3.0, 5.0]},
        "dynamics": {"timestep": 0.003},
        "sampling": {"equilibration_steps": 50, "production_steps": 200,
                     "sample_interval": 2, "andersen_interval": 20},
        "umbrella": {"centers": np.linspace(0, 1, 11).tolist(), "kappa": 10.0},
        "analysis": {"bins": 30, "bootstrap": 2},
        "parallel": {"backend": "serial"},
        "output": {"checkpoint_steps": 200},
        "recrossing": {"dividing_surface": 1.0, "parent_replicas": 1,
                       "parent_samples": 2, "equilibration_steps": 20,
                       "parent_interval_steps": 5, "child_pairs": 1,
                       "child_steps": 20, "bootstrap": 10},
        "network": {
            "aggregation": "separate",
            "entrances": [
                {"id": "a_bc", "fragments": [[0], [1, 2]], "rinf": 4.0,
                 "electronic_weight": 0.25,
                 "potential": {"provider": "python", "factory": factory,
                               "params": {"coupling": 0.32}}},
                {"id": "ab_c", "fragments": [[0, 1], [2]], "rinf": 4.5,
                 "electronic_weight": 0.75,
                 "potential": {"provider": "python", "factory": factory,
                               "params": {"coupling": 0.48}}},
            ],
            "channels": [
                {"id": "exchange_forward", "fragments": [[0, 1], [2]]},
                {"id": "exchange_reverse", "fragments": [[0], [1, 2]]},
            ],
            "transition_states": [
                {"id": "ts_low", "entrance": "a_bc", "channel": "exchange_forward",
                 "geometry": [[-r1, 0, 0], [0, 0, 0], [r1, 0, 0]],
                 "forming_bonds": [[0, 1]], "breaking_bonds": [[1, 2]]},
                {"id": "ts_high", "entrance": "ab_c", "channel": "exchange_reverse",
                 "geometry": [[-r2, 0, 0], [0, 0, 0], [r2, 0, 0]],
                 "forming_bonds": [[1, 2]], "breaking_bonds": [[0, 1]]},
            ],
        },
    }


class NetworkIdentityTests(unittest.TestCase):
    def test_entrance_normalizations_distinct_real_ts_and_preparation(self):
        config, system = resolve_network_job(network_spec())
        self.assertEqual(validate_network(config, system)["transition_states"], 2)
        report = prepare_network(config, system, numerical=True)
        self.assertTrue(report["ready"])
        self.assertEqual(len(report["paths"]), 2)
        for path in report["paths"]:
            self.assertLess(path["pes_check"]["maximum_reference_force_hartree_per_bohr"], 1e-13)
        entrances = {e["id"]: e for e in config["network"]["entrances"]}
        self.assertAlmostEqual(entrances["a_bc"]["reduced_mass_au"], 1.6)
        self.assertAlmostEqual(entrances["ab_c"]["reduced_mass_au"], 2.5)
        self.assertEqual(entrances["a_bc"]["rinf_bohr"], 4.0)
        self.assertEqual(entrances["ab_c"]["rinf_bohr"], 4.5)
        geometries = []
        for _, local, path_system in network_jobs(config, system):
            geometries.append(path_system.geometry)
            provider = make_potential(local["potential"], path_system)
            try:
                hessian = numerical_hessian(provider, path_system.geometry)
                self.assertEqual(np.sum(np.linalg.eigvalsh(hessian) < -1e-6), 1)
            finally:
                provider.close()
        self.assertFalse(np.array_equal(*geometries))

    def test_logical_ids_and_fingerprints_invariant_to_input_order_and_workers(self):
        first, system = resolve_network_job(network_spec())
        raw = network_spec()
        raw["network"]["transition_states"].reverse()
        raw["network"]["entrances"].reverse()
        raw["network"]["channels"].reverse()
        raw["parallel"] = {"backend": "process", "trajectory_workers": 2}
        second, other = resolve_network_job(raw)
        self.assertEqual(first["network"], second["network"])
        np.testing.assert_array_equal(system.geometry, other.geometry)
        for a, b in zip(network_jobs(first, system), network_jobs(second, other)):
            self.assertEqual(a[0]["fingerprint"], b[0]["fingerprint"])
            self.assertEqual(a[1]["seed"], b[1]["seed"])

    def test_no_cross_entry_or_cross_surface_sum(self):
        raw = network_spec()
        raw["network"]["aggregation"] = "sum"
        with self.assertRaisesRegex(ValueError, "physical partition"):
            resolve_network_job(raw)

    def test_duplicate_ids_unknown_references_and_empty_declarations(self):
        for collection in ("entrances", "channels", "transition_states"):
            raw = network_spec()
            raw["network"][collection].append(copy.deepcopy(raw["network"][collection][0]))
            with self.subTest(collection=collection), self.assertRaisesRegex(ValueError, "Duplicate"):
                resolve_network_job(raw)
        for field in ("entrance", "channel"):
            raw = network_spec()
            raw["network"]["transition_states"][0][field] = "missing"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unknown"):
                resolve_network_job(raw)
        raw = network_spec()
        raw["network"]["channels"].append({"id": "unused"})
        with self.assertRaisesRegex(ValueError, "without transition states"):
            resolve_network_job(raw)

    def test_duplicate_surface_catches_a_rigidly_translated_ts_copy(self):
        raw = network_spec()
        duplicate = copy.deepcopy(raw["network"]["transition_states"][0])
        duplicate["id"] = "redundant"
        duplicate["geometry"] = (np.asarray(duplicate["geometry"]) + [10, -3, 2]).tolist()
        raw["network"]["transition_states"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate transition-state dividing surface"):
            resolve_network_job(raw)

    def test_multiple_nonequivalent_references_for_one_entrance_are_retained(self):
        raw = network_spec()
        extra = copy.deepcopy(raw["network"]["transition_states"][0])
        extra["id"] = "ts_other_reference"
        extra["geometry"][0][0] -= 0.2
        raw["network"]["transition_states"].append(extra)
        config, _ = resolve_network_job(raw)
        self.assertEqual(len(config["network"]["jobs"]), 3)
        self.assertEqual(sum(j["entrance"] == "a_bc" for j in config["network"]["jobs"]), 2)

    def test_invalid_isotope_identity_partition_and_atom_indices(self):
        changes = [
            ("masses", [2.0, 3.0, 6.0], "masses differ"),
            ("symbols", ["D", "H", "H"], "symbols/order"),
            ("fragments", [[0], [1, 1]], "partition"),
            ("fragments", [[True], [1, 2]], "integers"),
        ]
        for key, value, message in changes:
            raw = network_spec()
            raw["network"]["entrances"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                resolve_network_job(raw)
        raw = network_spec()
        raw["network"]["transition_states"][0]["forming_bonds"] = [[0, 1.2]]
        with self.assertRaises(ValueError):
            resolve_network_job(raw)

    def test_every_path_is_validated_by_regular_jobspec(self):
        raw = network_spec()
        raw["network"]["transition_states"][0]["sampling"] = {"unknown_knob": 3}
        with self.assertRaisesRegex(ValueError, "Unknown sampling"):
            resolve_network_job(raw)
        raw = network_spec()
        raw["network"]["transition_states"][0]["potential"] = {"params": {"coupling": 0.4}}
        with self.assertRaisesRegex(ValueError, "complete.*provider"):
            resolve_network_job(raw)

    def test_equivalent_paths_cannot_receive_a_second_symmetry_multiplier(self):
        raw = network_spec()
        raw["network"]["transition_states"][0].update(
            equivalent_paths=[[0, 2, 1]], symmetry_factor=2,
        )
        with self.assertRaisesRegex(ValueError, "duplicate path weights"):
            resolve_network_job(raw)

    def test_relative_xyz_pes_paths_and_dependency_guard(self):
        from pyrpmd.workflows.network import _initialize_path

        raw = network_spec()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "toy.py"
            shutil.copy2(ROOT / "examples" / "rate" / "toy_exchange.py", source)
            transition = raw["network"]["transition_states"][0]
            xyz = "3\nreference in bohr\n" + "\n".join(
                "H " + " ".join(str(x) for x in row) for row in transition.pop("geometry")
            ) + "\n"
            (base / "left.xyz").write_text(xyz)
            transition.update(transition_state="left.xyz", geometry_unit="bohr")
            for entrance in raw["network"]["entrances"]:
                entrance["potential"]["factory"] = "toy.py:build"
            config, system = resolve_network_job(raw, base_directory=base)
            job, local, path_system = next(network_jobs(config, system, base / "run"))
            self.assertEqual(job["reference_source"], str(base / "left.xyz"))
            self.assertEqual(local["potential"]["factory"], str(source) + ":build")
            output = Path(local["output"]["directory"])
            _initialize_path(job, local, path_system, output, False)
            _initialize_path(job, local, path_system, output, True)
            source.write_text(source.read_text() + "\n# dependency changed\n")
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                _initialize_path(job, local, path_system, output, True)

    def test_pes_override_replaces_whole_table_and_units_paths_resolve(self):
        raw = network_spec()
        raw["potential"] = {"provider": "harmonic", "params": {"k": 123}}
        raw["network"]["transition_states"][0]["potential"] = {"provider": "free"}
        config, _ = resolve_network_job(raw)
        paths = {j["id"]: j for j in config["network"]["jobs"]}
        self.assertEqual(paths["ts_low"]["config"]["potential"], {"provider": "free"})
        self.assertEqual(paths["ts_low"]["potential_scope"], "transition_state")
        self.assertEqual(paths["ts_high"]["potential_scope"], "entrance")
        self.assertNotIn("k", paths["ts_high"]["config"]["potential"]["params"])

    def test_changed_resolved_science_rejected_and_legacy_rate_unaffected(self):
        config, system = resolve_network_job(network_spec())
        bad = copy.deepcopy(config)
        bad["network"]["jobs"][0]["config"]["reaction"]["rinf"] *= 1.1
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            validate_network(bad, system)
        legacy = network_spec()
        transition = legacy.pop("network")["transition_states"][0]
        legacy["task"] = "rate"
        legacy["system"]["geometry"] = transition["geometry"]
        legacy["potential"] = {"provider": "free"}
        legacy["reaction"] = {"fragments": [[0], [1, 2]], "rinf": 4,
                              "forming_bonds": [[0, 1]], "breaking_bonds": [[1, 2]]}
        resolved, _ = JobSpec.from_mapping(legacy).resolve()
        self.assertEqual(resolved["task"], "rate")
        self.assertNotIn("network", resolved)


class NetworkWorkflowTests(unittest.TestCase):
    def test_two_actual_toy_pipelines_interruption_resume_and_matrix(self):
        raw = network_spec()
        # Two entrances with two distinct analytic saddle/PES references each.
        # These are explicit separate teaching problems, never a summed rate.
        for index, coupling in ((0, 0.40), (1, 0.56)):
            extra = copy.deepcopy(raw["network"]["transition_states"][index])
            extra["id"] += "_alternative"
            radius = 1.2 + np.log((0.16 + coupling) / 0.16) / 1.2
            extra["geometry"] = [[-radius, 0, 0], [0, 0, 0], [radius, 0, 0]]
            extra["potential"] = {
                "provider": "python",
                "factory": str(ROOT / "examples" / "rate" / "toy_exchange.py") + ":build",
                "params": {"coupling": coupling},
            }
            raw["network"]["transition_states"].append(extra)
        config, system = resolve_network_job(raw)
        from pyrpmd.workflows import network as workflow

        original = workflow.run_rate
        calls = []

        def interrupt_second(local, path_system, output, executor=None):
            calls.append(str(output))
            if len(calls) == 2:
                raise RuntimeError("planned interruption before second path")
            return original(local, path_system, output, executor=executor)

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(workflow, "run_rate", side_effect=interrupt_second):
                with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                    run_network(config, system, directory)
            progress = json.loads((Path(directory) / "network-progress.json").read_text())
            self.assertEqual(progress["completed_paths"], 1)
            config["resume"] = True
            with mock.patch.object(workflow, "run_rate", wraps=original) as call:
                result = run_network(config, system, directory)
                self.assertEqual(call.call_count, 3)
            self.assertEqual(result["completed_paths"], 4)
            self.assertEqual(result["pending_paths"], [])
            self.assertIsNone(result["total_rate_cm3_s"])
            self.assertFalse(result["converged"])
            estimates = []
            for row in result["matrix"].values():
                for cell in row.values():
                    self.assertIsNone(cell["summed_rate_cm3_s"])
                    for path in cell["transition_states"].values():
                        estimates.append(path)
                        self.assertEqual(path["result"]["child_trajectories"], 4)
                        self.assertTrue(np.isfinite(path["kappa"]))
                        self.assertTrue((Path(path["output"]) / "recrossing.npz").is_file())
                        self.assertTrue((Path(path["output"]) / "rate_pmf.csv").is_file())
            self.assertEqual(len(estimates), 4)
            self.assertEqual(len({e["normalization"]["reduced_mass_au"] for e in estimates}), 2)
            with mock.patch.object(workflow, "run_rate", side_effect=AssertionError("completed paths must not rerun")):
                again = run_network(config, system, directory)
            self.assertEqual(result, again)


if __name__ == "__main__":
    unittest.main()
