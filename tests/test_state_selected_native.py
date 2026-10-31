"""Nuclear observations stay inside native collision blocks.

The compiled harmonic fixture counts PES calls independently of Python and is
only an analytic implementation test, not a molecular scattering benchmark.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from pyrpmd.block_propagation import ENERGY_PARTS, state_diagnostics
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.native import NativePESBinding


DIAGNOSTICS = {"energy_parts": True, "ring_radii": True}
MASSES = np.array([2.0, 3.0])
BETA = 3.5
DT = 0.002


def initial_state(beads):
    rng = np.random.default_rng(981)
    geometry = np.array([[-0.8, 0.1, 0.0], [0.7, 0.0, 0.2]])
    q = geometry[None] + rng.normal(scale=0.04, size=(beads, 2, 3))
    return State(q, rng.normal(scale=0.2, size=q.shape))


def reference_diagnostics(q, p):
    beads = len(q)
    center = q.mean(axis=0)
    momentum = p.mean(axis=0)
    potential = 0.005 * np.sum(q * q) / beads
    centroid = np.sum(momentum * momentum / (2 * MASSES[:, None]))
    internal = np.sum((p - momentum) ** 2 / (2 * MASSES[None, :, None])) / beads
    spring = 0.5 * (beads / BETA) ** 2 * np.sum(
        MASSES[None, :, None] * (q - np.roll(q, -1, axis=0)) ** 2
    ) / beads
    radii = np.sqrt(np.mean(np.sum((q - center) ** 2, axis=2), axis=0))
    return np.array([potential, centroid, internal, spring]), radii


class PythonDiagnosticsTests(unittest.TestCase):
    def test_initial_state_helper_reuses_known_energy_without_a_pes_call(self):
        for beads in (1, 3, 32):
            with self.subTest(beads=beads):
                pes = HarmonicPotential(0.01)
                dyn = Dynamics(pes, MASSES, BETA, beads, DT, native=False)
                state = initial_state(beads)
                before = state.copy()
                energy_per_bead = dyn.energy(state) / beads
                with patch.object(pes, "evaluate", side_effect=AssertionError("extra PES evaluation")):
                    got = state_diagnostics(dyn, state, DIAGNOSTICS, energy_per_bead=energy_per_bead)
                    radii_only = state_diagnostics(dyn, state, {"ring_radii": True})
                reference = state_diagnostics(dyn, state, DIAGNOSTICS)
                for name in DIAGNOSTICS:
                    np.testing.assert_allclose(got[name], reference[name], atol=2e-13)
                np.testing.assert_array_equal(got["ring_radii"], radii_only["ring_radii"])
                np.testing.assert_array_equal(state.q, before.q)
                np.testing.assert_array_equal(state.p, before.p)
                self.assertEqual(state.step, 0)

    def test_sparse_energy_parts_radii_and_hamiltonian_identity(self):
        self.assertEqual(ENERGY_PARTS, ("potential", "centroid_kinetic", "internal_kinetic", "spring"))
        mask = np.array([False, True, False, True, False, False, True])
        for beads in (1, 3, 32):
            with self.subTest(beads=beads):
                dyn = Dynamics(HarmonicPotential(0.01), MASSES, BETA, beads, DT, native=False)
                state = initial_state(beads)
                got = dyn.propagate_block(state, len(mask), backend="python", observe=mask,
                                          record_states=True, diagnostics=DIAGNOSTICS)
                np.testing.assert_array_equal(got.steps, [2, 4, 7])
                self.assertEqual(got.diagnostics["energy_parts"].shape, (3, 4))
                self.assertEqual(got.diagnostics["ring_radii"].shape, (3, 2))
                for row, (q, p) in enumerate(zip(got.q, got.p)):
                    parts, radii = reference_diagnostics(q, p)
                    np.testing.assert_allclose(got.diagnostics["energy_parts"][row], parts, atol=1e-14)
                    np.testing.assert_allclose(got.diagnostics["ring_radii"][row], radii, atol=1e-14)
                np.testing.assert_allclose(got.diagnostics["energy_parts"].sum(axis=1),
                                           got.energies / beads, atol=2e-13)
                if beads == 1:
                    np.testing.assert_array_equal(got.diagnostics["energy_parts"][:, 2:], 0.0)
                    np.testing.assert_array_equal(got.diagnostics["ring_radii"], 0.0)

    def test_diagnostics_do_not_add_python_pes_calls_to_recorded_energy(self):
        class CountingHarmonic(HarmonicPotential):
            calls = 0

            def evaluate(self, q):
                self.calls += 1
                return super().evaluate(q)

        pes = CountingHarmonic(0.01)
        dyn = Dynamics(pes, MASSES, BETA, 3, DT, native=False)
        baseline = dyn.propagate_block(initial_state(3), 5, backend="python")
        calls = pes.calls
        pes.calls = 0
        observed = dyn.propagate_block(initial_state(3), 5, backend="python", diagnostics=DIAGNOSTICS)
        self.assertEqual(pes.calls, calls)
        np.testing.assert_array_equal(observed.energies, baseline.energies)

    def test_empty_observation_mask_and_disabled_total_energy(self):
        dyn = Dynamics(HarmonicPotential(0.01), MASSES, BETA, 3, DT, native=False)
        state = initial_state(3)
        got = dyn.propagate_block(state, 4, observe=[0, 0, 0, 0], record_energy=False,
                                  record_states=True, diagnostics=DIAGNOSTICS)
        self.assertEqual(state.step, 4)
        self.assertEqual(got.diagnostics["energy_parts"].shape, (0, 4))
        self.assertEqual(got.diagnostics["ring_radii"].shape, (0, 2))
        self.assertEqual(got.q.shape, (0, 3, 2, 3))
        got = dyn.propagate_block(state, 2, record_energy=False, diagnostics=DIAGNOSTICS)
        self.assertTrue(np.isnan(got.energies).all())
        self.assertTrue(np.isfinite(got.diagnostics["energy_parts"]).all())

    def test_invalid_descriptors_fail_before_state_mutation(self):
        dyn = Dynamics(HarmonicPotential(0.01), MASSES, BETA, 3, DT, native=False)
        for descriptor in (True, {"normal_modes": True}, {"energy_parts": 1}, {"ring_radii": lambda: 1}):
            state = initial_state(3)
            previous = state.copy()
            with self.subTest(descriptor=descriptor), self.assertRaises(ValueError):
                dyn.propagate_block(state, 3, diagnostics=descriptor)
            np.testing.assert_array_equal(state.q, previous.q)
            np.testing.assert_array_equal(state.p, previous.p)
            self.assertEqual(state.step, 0)

    def test_custom_energy_is_not_silently_replaced(self):
        class CustomEnergy(Dynamics):
            def energy(self, state, bias=None):
                return super().energy(state, bias) + 7.0

        dyn = CustomEnergy(HarmonicPotential(0.01), MASSES, BETA, 3, DT, native=False)
        state = initial_state(3)
        with self.assertRaisesRegex(ValueError, "original Dynamics.energy"):
            dyn.propagate_block(state, 2, diagnostics={"energy_parts": True})
        self.assertEqual(state.step, 0)
        got = dyn.propagate_block(state, 2, diagnostics={"ring_radii": True})
        self.assertEqual(set(got.diagnostics), {"ring_radii"})
        self.assertGreater(got.energies.min(), 7.0)

    def test_bias_is_excluded_from_physical_energy_parts(self):
        from pyrpmd.bias import UmbrellaBias
        from pyrpmd.cv import Cartesian

        dyn = Dynamics(HarmonicPotential(0.01), MASSES, BETA, 3, DT, native=False)
        bias = UmbrellaBias(Cartesian(0, 0), [0.2], 0.1)
        got = dyn.propagate_block(initial_state(3), 4, bias=bias, record_states=True,
                                  diagnostics=DIAGNOSTICS)
        energies = np.array([bias.energy_force(q)[0] for q in got.q])
        np.testing.assert_allclose(got.diagnostics["energy_parts"].sum(axis=1),
                                   (got.energies - energies) / 3, atol=2e-13)


class CompiledPotential:
    def __init__(self, path, natoms=2):
        self.lib = ctypes.CDLL(str(path))
        self.natoms = natoms
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.fn = self.lib.batch
        self.fn.argtypes = [ctypes.c_int32, pointer, pointer, pointer,
                           np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
        self.fn.restype = None
        self.lib.reset.argtypes = [ctypes.c_int]
        self.lib.reset.restype = None
        self.lib.get_calls.argtypes = []
        self.lib.get_calls.restype = ctypes.c_int
        self.lib.set_atoms.argtypes = [ctypes.c_int]
        self.lib.set_atoms.restype = None
        self.lib.set_atoms(natoms)
        self.lock = threading.RLock()
        self.lib.reset(-1)

    def evaluate(self, q):
        energies = np.empty(len(q))
        force = np.empty_like(q)
        status = np.empty(len(q), dtype=np.int32)
        with self.lock:
            self.fn(len(q), q, energies, force, status)
        return energies, force, status

    def native_binding(self):
        return NativePESBinding("fortran-batch-v1", self.fn, self.natoms, self, self.lock)


@unittest.skipUnless(native_capabilities()["available"], "native extension build required")
class NativeDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyrpmd import _native

        if getattr(_native, "BLOCK_DIAGNOSTICS_API", 0) < 1:
            raise unittest.SkipTest("rebuild native extension for nuclear diagnostics")
        compiler = shutil.which("cc") or shutil.which("gcc")
        if compiler is None:
            raise unittest.SkipTest("C compiler required for native PES call counter")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        source = Path(cls.directory.name) / "diagnostics_pes.c"
        source.write_text(r'''
#include <stdint.h>
static int calls=0, fail_at=-1, atoms=2;
void reset(int fail) { calls=0; fail_at=fail; }
int get_calls(void) { return calls; }
void set_atoms(int count) { atoms=count; }
void batch(int32_t B, const double* q, double* e, double* f, int32_t* status) {
    ++calls;
    for(int32_t b=0;b<B;++b) {
        e[b]=0; status[b]=(calls==fail_at)?7:0;
        for(int j=0;j<3*atoms;++j) { int i=3*atoms*b+j; e[b]+=.005*q[i]*q[i]; f[i]=-.01*q[i]; }
    }
}
''')
        cls.library = Path(cls.directory.name) / "diagnostics_pes.so"
        subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(source), "-o", str(cls.library)],
                       check=True, capture_output=True)

    def make_dynamics(self, beads, integrator="cayley-bcb"):
        pes = CompiledPotential(self.library)
        dyn = Dynamics(pes, MASSES, BETA, beads, DT, native="required", integrator=integrator)
        return pes, dyn

    def test_native_python_equivalence_and_sparse_full_states(self):
        mask = [True, False, False, True, False, True, False, True]
        for beads in (1, 3, 32):
            for integrator in ("cayley-bcb", "normal-mode-bab"):
                with self.subTest(beads=beads, integrator=integrator):
                    pes, dyn = self.make_dynamics(beads, integrator)
                    state = initial_state(beads)
                    got = dyn.propagate_block(state.copy(), len(mask), backend="native", observe=mask,
                                              record_states=True, diagnostics=DIAGNOSTICS, reuse_forces=True)
                    expected = dyn.propagate_block(state.copy(), len(mask), backend="python", observe=mask,
                                                   record_states=True, diagnostics=DIAGNOSTICS)
                    self.assertEqual(got.backend, "native")
                    np.testing.assert_array_equal(got.steps, [1, 4, 6, 8])
                    np.testing.assert_allclose(got.q, expected.q, atol=1e-13)
                    np.testing.assert_allclose(got.p, expected.p, atol=1e-13)
                    for name in DIAGNOSTICS:
                        np.testing.assert_allclose(got.diagnostics[name], expected.diagnostics[name],
                                                   atol=2e-13, rtol=2e-12)
                    np.testing.assert_allclose(got.diagnostics["energy_parts"].sum(axis=1),
                                               got.energies / beads, atol=2e-13)

    def test_vibration_selected_initial_states_enter_unthermostatted_native_blocks(self):
        from pyrpmd.initial_states import prepare_initial_ensemble

        system = {
            "symbols": ["H", "H", "H"], "masses": [2.0, 2.0, 2.0],
            "geometry": [[-5.0, 0.0, 0.0], [0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
            "fragments": [[0], [1, 2]],
        }
        mode = [0.0, 0.0, -2**-0.5, 0.0, 0.0, 2**-0.5]
        fragment = {"frequencies": [0.5], "modes": [mode], "vibrational_quanta": [1]}
        for method, beads in (("qct", 1), ("ne_rpmd", 32)):
            with self.subTest(method=method):
                section = {"method": method, "fragments": [{}, fragment]}
                if method == "ne_rpmd":
                    section["beta"] = BETA
                prepared = prepare_initial_ensemble(section, system, beads, BETA)
                state = prepared.sample(np.random.default_rng(404))
                pes = CompiledPotential(self.library, natoms=3)
                dyn = Dynamics(pes, system["masses"], BETA, beads, DT, seed=1, native="required")
                rng_before = repr(dyn.rng.bit_generator.state)
                with patch.object(dyn, "thermalize", side_effect=AssertionError("unexpected thermostat")):
                    got = dyn.propagate_block(state, 5, backend="native", observe=[1, 0, 1, 0, 1],
                                              record_states=True, diagnostics=DIAGNOSTICS, reuse_forces=True)
                self.assertEqual(got.completed, 5)
                self.assertEqual(got.backend, "native")
                self.assertEqual(state.q.shape, (beads, 3, 3))
                self.assertEqual(got.q.shape, (3, beads, 3, 3))
                self.assertEqual(repr(dyn.rng.bit_generator.state), rng_before)
                np.testing.assert_allclose(got.diagnostics["energy_parts"].sum(axis=1),
                                           got.energies / beads, atol=3e-12)

    def test_native_observers_make_no_extra_pes_calls_or_python_callbacks(self):
        pes, dyn = self.make_dynamics(32)
        original = initial_state(32)
        baseline_state, observed_state = original.copy(), original.copy()
        baseline = dyn.propagate_block(baseline_state, 9, backend="native", record_energy=False,
                                       reuse_forces=True)
        baseline_calls = pes.lib.get_calls()
        self.assertEqual(baseline_calls, 10)
        pes.lib.reset(-1)
        with patch.object(pes, "evaluate", side_effect=AssertionError("unexpected Python PES call")):
            got = dyn.propagate_block(observed_state, 9, backend="native", record_energy=False,
                                      reuse_forces=True, diagnostics=DIAGNOSTICS)
        self.assertEqual(pes.lib.get_calls(), baseline_calls)
        np.testing.assert_array_equal(baseline_state.q, observed_state.q)
        np.testing.assert_array_equal(baseline_state.p, observed_state.p)
        self.assertTrue(np.isnan(baseline.energies).all() and np.isnan(got.energies).all())
        self.assertTrue(np.isfinite(got.diagnostics["energy_parts"]).all())

    def test_native_failure_truncates_diagnostics_to_observed_accepted_prefix(self):
        pes, dyn = self.make_dynamics(3)
        state = initial_state(3)
        pes.lib.reset(4)  # With endpoint reuse, fail the third physical step.
        got = dyn.propagate_block(state, 6, backend="native", observe=[1, 0, 1, 1, 0, 1],
                                  reuse_forces=True, diagnostics=DIAGNOSTICS, raise_on_error=False)
        self.assertEqual(got.completed, 2)
        self.assertEqual(state.step, 2)
        self.assertIn("status 7", got.error)
        np.testing.assert_array_equal(got.steps, [1])
        self.assertEqual(got.diagnostics["energy_parts"].shape, (1, 4))
        self.assertEqual(got.diagnostics["ring_radii"].shape, (1, 2))
        pes.lib.reset(-1)
        reference = initial_state(3)
        expected = dyn.propagate_block(reference, 2, backend="native", observe=[1, 0],
                                       reuse_forces=True, diagnostics=DIAGNOSTICS)
        np.testing.assert_array_equal(state.q, reference.q)
        np.testing.assert_array_equal(state.p, reference.p)
        for name in DIAGNOSTICS:
            np.testing.assert_array_equal(got.diagnostics[name], expected.diagnostics[name])

    def test_old_extension_falls_back_only_when_new_observer_is_requested(self):
        pes, dyn = self.make_dynamics(3)
        with patch.object(dyn.integrator._native, "BLOCK_DIAGNOSTICS_API", 0):
            legacy = dyn.propagate_block(initial_state(3), 2, backend="native")
            self.assertEqual(legacy.backend, "native")
            self.assertIsNone(legacy.diagnostics)
            fallback = dyn.propagate_block(initial_state(3), 2, diagnostics=DIAGNOSTICS)
            self.assertEqual(fallback.backend, "python")
            self.assertIn("nuclear diagnostics", fallback.fallback_reason)
            state = initial_state(3)
            before = state.copy()
            with self.assertRaisesRegex(RuntimeError, "nuclear diagnostics"):
                dyn.propagate_block(state, 2, backend="native", diagnostics=DIAGNOSTICS)
            np.testing.assert_array_equal(state.q, before.q)
            np.testing.assert_array_equal(state.p, before.p)
            self.assertEqual(state.step, 0)


if __name__ == "__main__":
    unittest.main()
