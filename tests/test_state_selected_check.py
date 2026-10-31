"""Precollision survival checks use sparse NVE blocks and a common PES zero."""

from contextlib import chdir, redirect_stdout, redirect_stderr
import copy
import ctypes
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.cli import main
from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.initial_states import PreparedInitialEnsemble
from pyrpmd.input import JobSpec
from pyrpmd.potentials import PESResult, Potential, make_potential
from pyrpmd.potentials.native import NativePESBinding
from pyrpmd.workflows import state_selected


FACTORY_SOURCE = '''
import numpy as np
from pyrpmd.potentials import PESResult

class Bond:
    def __init__(self, offset=0.0, attacker_force=0.0):
        self.offset, self.attacker_force = offset, attacker_force
        self.closed = False
    def evaluate(self, q):
        delta = q[:, 2] - q[:, 1]
        distance = np.linalg.norm(delta, axis=1)
        displacement = distance - 1.4
        energy = self.offset + 0.2 * displacement**2 - self.attacker_force * q[:, 0, 2]
        force = np.zeros_like(q)
        pair_force = -0.4 * displacement[:, None] * delta / distance[:, None]
        force[:, 2], force[:, 1] = pair_force, -pair_force
        force[:, 0, 2] = self.attacker_force
        return PESResult(energy, force)
    def close(self):
        self.closed = True

def make(**params):
    return Bond(**params)
'''


def configuration(factory, method="ne_rpmd"):
    hessian = np.zeros((6, 6))
    hessian[np.ix_([2, 5], [2, 5])] = [[0.4, -0.4], [-0.4, 0.4]]
    ensemble = {"method": method, "fragments": [{}, {"hessian": hessian.tolist(), "vibrational_quanta": [1]}]}
    if method == "ne_rpmd":
        ensemble["beta"] = 400.0
    return {
        "task": "drpmd", "beads": 8 if method == "ne_rpmd" else 1, "seed": 234,
        "native": False,
        "system": {
            "symbols": ["H", "H", "H"], "masses": [2000.0] * 3,
            "geometry": [[0.0, 0.0, -12.0], [0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
            "fragments": [[0], [1, 2]], "separation": "12 bohr",
        },
        "potential": {"provider": "python", "factory": str(factory) + ":make"},
        "initial_ensemble": ensemble,
        "dynamics": {"timestep": "0.02 fs", "block_steps": 3, "loop_backend": "python"},
        "collision": {"sampling": "fixed_energy", "energy": "0.1 eV", "bmax": "1 bohr",
                      "max_time": "0.2 fs", "trajectories": 2},
        "outcomes": {"bond_cutoffs": {"H-H": "2 bohr"}, "channels": {},
                     "exit_distance": "12 bohr", "check_interval": 1, "persistence_steps": 2},
        "output": {"directory": "production-must-not-exist"},
    }


class StateSelectedCheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.factory = self.root / "bond.py"
        self.factory.write_text(FACTORY_SOURCE)

    def test_sparse_blocks_no_thermostat_no_collision_launch_and_no_production_run(self):
        config = configuration(self.factory)
        original_projection = PreparedInitialEnsemble.diagnostics
        original_block = Dynamics.propagate_block
        calls = []

        def block(dynamics, state, steps, **kwargs):
            calls.append((steps, np.asarray(kwargs["observe"]).copy(), kwargs))
            return original_block(dynamics, state, steps, **kwargs)

        with patch.object(Dynamics, "thermalize", side_effect=AssertionError("thermostat")), \
             patch.object(Dynamics, "sample_momenta", side_effect=AssertionError("momentum resampling")), \
             patch("pyrpmd.workflows.drpmd.launch_collision", side_effect=AssertionError("collision kick")), \
             patch("pyrpmd.runner.run", side_effect=AssertionError("production run")), \
             patch.object(Dynamics, "propagate_block", autospec=True, side_effect=block), \
             patch.object(PreparedInitialEnsemble, "diagnostics", autospec=True, side_effect=original_projection) as projected:
            report = state_selected.check_state_selected(config, duration="0.14 fs", samples=2, stride=3)
        self.assertEqual([row["step"] for row in report["time_series"]], [0, 3, 6, 7])
        self.assertEqual(projected.call_count, 2 * 4)
        self.assertEqual([item[0] for item in calls], [3, 3, 1] * 2)
        self.assertTrue(all(call[2]["record_states"] and call[2]["check_energy"] for call in calls))
        self.assertTrue(all(call[2]["diagnostics"] == {"energy_parts": True, "ring_radii": True} for call in calls))
        self.assertTrue(report["controls"]["propagation_rng_unchanged"])
        self.assertFalse(report["controls"]["collision_kick"])
        self.assertEqual(report["controls"]["thermostat"], "none")
        self.assertEqual(report["propagation"]["python_steps"], 14)
        for sample in report["samples"]:
            for fragment in sample["frames"][0]["fragments"]:
                np.testing.assert_allclose(fragment["centroid_com_momentum_au"], 0.0, atol=2e-14)
        self.assertLess(report["maximum_absolute_observed_hamiltonian_drift_au"], 1e-5)
        json.dumps(report, allow_nan=False)

    def test_switch_error_cancels_arbitrary_pes_offset(self):
        for method in ("qct", "ne_rpmd"):
            first = configuration(self.factory, method)
            second = copy.deepcopy(first)
            second["potential"]["params"] = {"offset": 17.3}
            a = state_selected.check_state_selected(first, duration="0.06 fs", samples=3, stride=2)
            b = state_selected.check_state_selected(second, duration="0.06 fs", samples=3, stride=2)
            self.assertAlmostEqual(b["full_pes_reference"]["potential_energy_au"], 17.3, places=13)
            self.assertEqual(a["samples"][0]["initialization_seed_sequence"], [234, 5, 0])
            for left, right in zip(a["samples"], b["samples"]):
                self.assertAlmostEqual(left["switch_error"]["real_minus_harmonic_potential_au"], 0.0, places=13)
                self.assertAlmostEqual(right["switch_error"]["real_minus_harmonic_potential_au"], 0.0, places=12)
                for old, new in zip(left["frames"], right["frames"]):
                    np.testing.assert_allclose(new["fragments"][1]["harmonic_mode_actions"],
                                               old["fragments"][1]["harmonic_mode_actions"], atol=1e-13)
                    self.assertAlmostEqual(new["full_pes_primitive_relative_reference_au"],
                                           old["full_pes_primitive_relative_reference_au"], places=12)

    def test_inconsistent_reference_is_reported_without_energy_rescaling(self):
        config = configuration(self.factory, "qct")
        config["potential"]["params"] = {"offset": -31.0}
        config["initial_ensemble"]["fragments"][1]["hessian"] = (
            2 * np.asarray(config["initial_ensemble"]["fragments"][1]["hessian"])
        ).tolist()
        report = state_selected.check_state_selected(config, duration="0.04 fs", samples=4, stride=10)
        # The reference stiffness is twice the real one, so Vreal=Vharm/2.
        for sample in report["samples"]:
            switch = sample["switch_error"]
            self.assertAlmostEqual(switch["real_minus_harmonic_potential_au"], -0.5 * switch["harmonic_potential_au"], places=12)
        self.assertLess(report["initial_switch_error"]["real_minus_harmonic_potential_au"]["mean"], 0)

    def test_time_statistics_are_paired_and_include_final_short_stride(self):
        report = state_selected.check_state_selected(configuration(self.factory), duration="0.1 fs", samples=3, stride=2)
        self.assertEqual([row["step"] for row in report["time_series"]], [0, 2, 4, 5])
        actions = np.array([sample["frames"][-1]["fragments"][1]["harmonic_mode_actions"] for sample in report["samples"]])
        initial = np.array([sample["frames"][0]["fragments"][1]["harmonic_mode_actions"] for sample in report["samples"]])
        got = report["time_series"][-1]["fragments"][1]
        np.testing.assert_allclose(got["harmonic_mode_actions"]["mean"], actions.mean(axis=0))
        np.testing.assert_allclose(got["harmonic_mode_actions"]["std"], actions.std(axis=0, ddof=1))
        np.testing.assert_allclose(got["action_change_from_initial"]["mean"], (actions - initial).mean(axis=0))
        self.assertEqual(report["time_series"][0]["fragments"][0]["harmonic_mode_actions"]["mean"], [])

    def test_single_sample_uncertainty_is_unavailable_in_json(self):
        report = state_selected.check_state_selected(configuration(self.factory), duration="0.04 fs", samples=1, stride=2)
        decoded = json.loads(json.dumps(report, allow_nan=False))
        checked = []

        def visit(value):
            if isinstance(value, dict):
                if {"mean", "std", "sem", "minimum", "maximum"} <= set(value):
                    self.assertIsNone(value["std"])
                    self.assertIsNone(value["sem"])
                    self.assertEqual(value["mean"], value["minimum"])
                    self.assertEqual(value["mean"], value["maximum"])
                    checked.append(value)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(decoded)
        self.assertGreater(len(checked), 10)
        self.assertEqual(decoded["time_series"][-1]["hamiltonian_per_bead_au"]["mean"],
                         decoded["samples"][0]["frames"][-1]["hamiltonian_per_bead_au"])

    def test_full_pes_force_remains_active_and_geometry_is_not_constrained(self):
        config = configuration(self.factory)
        force = 0.001
        config["potential"]["params"] = {"attacker_force": force}
        report = state_selected.check_state_selected(config, duration="0.1 fs", samples=1, stride=2)
        np.testing.assert_allclose(report["full_pes_reference"]["net_force_per_fragment_au"][0], [0, 0, force])
        self.assertIn("not an isolated-fragment", report["scope"])
        frames = report["samples"][0]["frames"]
        elapsed = frames[-1]["time_au"]
        atom = frames[-1]["fragments"][0]
        self.assertAlmostEqual(atom["centroid_com_momentum_au"][2], force * elapsed, places=12)
        self.assertAlmostEqual(atom["centroid_com_position_au"][2], -12 + force * elapsed**2 / (2 * 2000), places=12)
        self.assertLess(frames[-1]["fragment_com_separation_au"], frames[0]["fragment_com_separation_au"])

    def test_cli_output_is_explicit_file_relative_to_caller_and_does_not_create_run(self):
        inputs = self.root / "inputs"
        outputs = self.root / "caller"
        inputs.mkdir()
        outputs.mkdir()
        config = configuration(self.factory, "qct")
        input_path = inputs / "case.py"
        input_path.write_text("def build():\n    return " + repr(config) + "\n")
        stream = io.StringIO()
        with chdir(outputs), redirect_stdout(stream):
            status = main(["state-check", str(input_path), "--time", "0.06 fs", "--samples", "1",
                           "--stride", "2", "--output", "reports/check.json"])
        self.assertEqual(status, 0)
        path = outputs / "reports" / "check.json"
        result = json.loads(path.read_text())
        self.assertEqual(result["controls"]["steps"], 3)
        self.assertEqual(result, json.loads(stream.getvalue()))
        self.assertFalse((inputs / "production-must-not-exist").exists())
        self.assertFalse((outputs / "production-must-not-exist").exists())

    def test_bad_requests_fail_and_provider_is_closed(self):
        for kwargs in ({"samples": 0}, {"stride": True}, {"duration": "0.035 fs"},
                       {"output": self.root / "report.txt"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                state_selected.check_state_selected(configuration(self.factory), **kwargs)
        thermal = configuration(self.factory)
        thermal.pop("initial_ensemble")
        with self.assertRaisesRegex(ValueError, "initial_ensemble"):
            state_selected.check_state_selected(thermal, duration="0.04 fs", samples=1)
        created = []

        def capture(spec, system):
            provider = make_potential(spec, system)
            created.append(provider)
            return provider

        with patch.object(state_selected, "make_potential", side_effect=capture), \
             patch.object(Dynamics, "propagate_block", side_effect=RuntimeError("PES failed")):
            with self.assertRaisesRegex(RuntimeError, "PES failed"):
                state_selected.check_state_selected(configuration(self.factory), duration="0.04 fs", samples=1)
        self.assertTrue(created[0].obj.closed)

    def test_rng_change_during_propagation_is_detected(self):
        original = Dynamics.propagate_block

        def unexpected_randomness(dynamics, *args, **kwargs):
            dynamics.rng.random()
            return original(dynamics, *args, **kwargs)

        with patch.object(Dynamics, "propagate_block", autospec=True, side_effect=unexpected_randomness):
            with self.assertRaisesRegex(RuntimeError, "changed.*RNG"):
                state_selected.check_state_selected(configuration(self.factory), duration="0.04 fs", samples=1)

    def test_nested_native_option_and_top_level_precedence_match_production(self):
        for top, nested, expected in ((None, False, False), (False, True, False), (True, False, True)):
            config = configuration(self.factory)
            config.pop("native")
            if top is not None:
                config["native"] = top
            config["dynamics"]["native"] = nested
            with self.subTest(top=top, nested=nested), \
                 patch.object(state_selected, "Dynamics", wraps=Dynamics) as constructor:
                state_selected.check_state_selected(config, duration="0.02 fs", samples=1)
            self.assertIs(constructor.call_args.kwargs["native"], expected)


class CompiledBondPotential(Potential):
    def __init__(self, library):
        self.lib = ctypes.CDLL(library)
        self.fn = self.lib.bond
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.fn.argtypes = [ctypes.c_int32, pointer, pointer, pointer,
                           np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
        self.fn.restype = None
        self.lock = threading.RLock()
        self.python_calls = 0
        self.closed = False

    def evaluate(self, q):
        self.python_calls += 1
        energy, forces, status = np.empty(len(q)), np.empty_like(q), np.empty(len(q), dtype=np.int32)
        with self.lock:
            self.fn(len(q), q, energy, forces, status)
        return PESResult(energy, forces, status)

    def native_binding(self):
        return NativePESBinding("fortran-batch-v1", self.fn, 3, self, self.lock)

    def close(self):
        self.closed = True


def make_compiled_bond(**kwargs):
    return CompiledBondPotential(**kwargs)


@unittest.skipUnless(native_capabilities()["available"], "native extension build required")
class NativeStateSelectedCheckTests(unittest.TestCase):
    def test_c_pes_runs_all_nve_steps_natively_with_python_only_at_observations(self):
        from pyrpmd import _native
        if getattr(_native, "BLOCK_DIAGNOSTICS_API", 0) < 1:
            self.skipTest("native diagnostics build required")
        compiler = shutil.which("cc") or shutil.which("gcc")
        if compiler is None:
            self.skipTest("C compiler required")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "bond.c"
            source.write_text(r'''
#include <stdint.h>
#include <math.h>
void bond(int32_t B, const double *q, double *e, double *f, int32_t *status) {
    for (int32_t b=0;b<B;b++) {
        int o=9*b; double d[3],r2=0;
        for(int j=0;j<9;j++) f[o+j]=0;
        for(int c=0;c<3;c++) {d[c]=q[o+6+c]-q[o+3+c]; r2+=d[c]*d[c];}
        double r=sqrt(r2),dr=r-1.4;
        e[b]=.2*dr*dr+17.3; status[b]=0;
        for(int c=0;c<3;c++) {double force=-.4*dr*d[c]/r; f[o+6+c]=force;f[o+3+c]=-force;}
    }
}
''')
            library = path / "bond.so"
            subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(library), "-lm"],
                           check=True, capture_output=True)
            config = configuration(path / "unused.py")
            config.update(native=True, beads=32)
            config["dynamics"]["loop_backend"] = "native"
            config["potential"] = {"provider": "python", "factory": str(Path(__file__).resolve()) + ":make_compiled_bond",
                                   "params": {"library": str(library)}}
            created = []

            def capture(spec, system):
                provider = make_potential(spec, system)
                created.append(provider)
                return provider

            with patch.object(state_selected, "make_potential", side_effect=capture):
                native = state_selected.check_state_selected(config, duration="0.14 fs", samples=2, stride=3)
            self.assertEqual(native["propagation"]["native_steps"], 14)
            self.assertEqual(native["propagation"]["python_steps"], 0)
            # One full-reference PES call plus one initial observation per draw.
            # All NVE endpoint evaluations are made directly by the native loop.
            self.assertEqual(created[0].obj.python_calls, 3)
            self.assertTrue(created[0].obj.closed)
            config["dynamics"]["loop_backend"] = "python"
            python = state_selected.check_state_selected(config, duration="0.14 fs", samples=2, stride=3)
            for a, b in zip(native["samples"], python["samples"]):
                for x, y in zip(a["frames"], b["frames"]):
                    self.assertAlmostEqual(x["hamiltonian_per_bead_au"], y["hamiltonian_per_bead_au"], places=11)
                    np.testing.assert_allclose(x["fragments"][1]["harmonic_mode_actions"],
                                               y["fragments"][1]["harmonic_mode_actions"], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
