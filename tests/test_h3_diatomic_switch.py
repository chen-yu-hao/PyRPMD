"""Independent physics oracles for the opt-in native diatom preparation loop."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

from benchmarks.h3.diatomic_switch import (
    DiatomicSwitchError, DiatomicSwitcher, _Options, _Output,
)
from pyrpmd.constants import FS_TO_AU

MU = 918.5763237198094
OMEGA = .02008
REQ = 1.40147
ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "benchmarks/h3/build/libbkmp2.so"
COMPILER = os.name == "posix" and shutil.which(shlex.split(os.environ.get("CXX", "c++"))[0])


def harmonic(**kwargs):
    return DiatomicSwitcher(None, mass=2*MU, omega=OMEGA, equilibrium=REQ,
                            target_omega=kwargs.pop("target_omega", OMEGA), **kwargs)


@unittest.skipUnless(COMPILER, "The experimental standalone switch requires a C++17 compiler")
class AnalyticDiatomicSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.same = harmonic()
        cls.changed = harmonic(target_omega=1.6*OMEGA, energy_zero_hartree=-.1745)

    def test_static_ring_matches_independent_exact_normal_modes_with_second_order_error(self):
        # Exact complete harmonic Hamiltonian, independent of the BAB splitting.
        # Odd P exercises the direct DFT; P32/64 exercises radix-2 transforms.
        for beads in (1, 5, 32, 64):
            with self.subTest(beads=beads):
                index = np.arange(beads)
                r = REQ+.05*np.cos(2*np.pi*index/beads)+.01*np.cos(4*np.pi*index/beads)
                p = .3*np.sin(2*np.pi*index/beads)+.2
                frequencies = np.sqrt(OMEGA**2+(2*beads/600*np.sin(np.pi*index/beads))**2)
                qr, pr = np.fft.fft(r-REQ), np.fft.fft(p)
                time = 2*FS_TO_AU
                expected_r = REQ+np.fft.ifft(np.cos(frequencies*time)*qr
                    + np.sin(frequencies*time)/(MU*frequencies)*pr).real
                expected_p = np.fft.ifft(np.cos(frequencies*time)*pr
                    - MU*frequencies*np.sin(frequencies*time)*qr).real
                errors = []
                for dt in (.02, .01):
                    result = self.same.run(r, p, beta=600., dt_fs=dt,
                        switch_time_fs=0., hold_time_fs=2., trace_stride_fs=.2)
                    errors.append(max(np.max(abs(result.r-expected_r)),
                                      np.max(abs(result.p-expected_p))/(MU*OMEGA)))
                    self.assertEqual(result.summary["status"], "completed")
                    self.assertEqual(result.summary["switching_work_per_bead_hartree"], 0.)
                self.assertLess(errors[0], 4e-6)
                self.assertGreater(errors[1]/errors[0], .24)
                self.assertLess(errors[1]/errors[0], .26)

    def test_driven_harmonic_matches_independent_ode_and_work_balance_converges(self):
        r0, p0, duration = REQ+.15, .3, 10*FS_TO_AU
        def derivative(t, y):
            s = t/duration
            ramp = s**3*(10+s*(-15+6*s))
            ramp_rate = 30*s*s*(1-s)**2/duration
            x = y[0]-REQ
            potential_difference = .5*MU*((1.6*OMEGA)**2-OMEGA**2)*x*x
            return [y[1]/MU, -MU*((1-ramp)*OMEGA**2+ramp*(1.6*OMEGA)**2)*x,
                    ramp_rate*potential_difference]
        oracle = solve_ivp(derivative, [0, duration], [r0, p0, 0.],
                           method="DOP853", rtol=2e-13, atol=2e-13)
        self.assertTrue(oracle.success)
        residuals = []
        for dt in (.02, .01):
            result = self.changed.run([r0], [p0], beta=600., dt_fs=dt,
                                      switch_time_fs=10., trace_stride_fs=.2)
            np.testing.assert_allclose(result.r, oracle.y[0, -1], atol=3e-5, rtol=0)
            np.testing.assert_allclose(result.p, oracle.y[1, -1], atol=7e-5, rtol=0)
            self.assertAlmostEqual(result.summary["switching_work_per_bead_hartree"],
                                   oracle.y[2, -1], delta=8e-8)
            residuals.append(result.summary["maximum_absolute_energy_balance_residual_per_bead_hartree"])
            delta_h = (result.summary["final_ring_hamiltonian_per_bead_hartree"]
                       - result.summary["initial_harmonic_energy_per_bead_hartree"])
            self.assertGreater(abs(delta_h), 1000*residuals[-1])
        self.assertGreater(residuals[1]/residuals[0], .24)
        self.assertLess(residuals[1]/residuals[0], .26)

    def test_slow_p1_harmonic_switch_approaches_adiabatic_action(self):
        r0, p0 = REQ+.15, .3
        initial_action = (p0*p0/(2*MU)+.5*MU*OMEGA**2*(r0-REQ)**2)/OMEGA
        errors = []
        for duration in (0., 10., 100.):
            result = self.changed.run([r0], [p0], beta=600., dt_fs=.01,
                                      switch_time_fs=duration, trace_stride_fs=1.)
            final_action = (result.p[0]**2/(2*MU)
                + .5*MU*(1.6*OMEGA)**2*(result.r[0]-REQ)**2)/(1.6*OMEGA)
            errors.append(abs(final_action/initial_action-1))
        self.assertGreater(errors[0], .5)
        self.assertLess(errors[1], .02)
        self.assertLess(errors[2], 3e-5)
        # This is a classical P=1 action test, not a quantum-state guarantee.

    def test_sudden_jump_keeps_phase_and_counts_offset_work(self):
        model = harmonic(target_offset=.012, energy_zero_hartree=-.1745)
        r = np.array([REQ-.02, REQ+.01, REQ+.04])
        p = np.array([.1, -.2, .4])
        original_r, original_p = r.copy(), p.copy()
        result = model.run(r, p, beta=600., switch_time_fs=0., hold_time_fs=0.)
        np.testing.assert_array_equal(result.r, r)
        np.testing.assert_array_equal(result.p, p)
        np.testing.assert_array_equal(r, original_r)
        np.testing.assert_array_equal(p, original_p)
        self.assertIsNot(result.r, r)
        row = result.trace[0]
        self.assertEqual(row["lambda"], 1.)
        self.assertAlmostEqual(row["switching_work_per_bead_hartree"], .012, places=14)
        self.assertAlmostEqual(row["energy_balance_residual_per_bead_hartree"], 0., places=14)
        self.assertEqual(result.summary["energy_zero_hartree"], -.1745)
        self.assertAlmostEqual(row["primitive_internal_per_bead_hartree"],
            row["kinetic_per_bead_hartree"]+row["full_potential_per_bead_hartree"]
            +.1745-row["spring_per_bead_hartree"], places=14)
        json.dumps(result.summary, allow_nan=False)
        json.dumps(result.trace, allow_nan=False)

    def test_hold_average_uses_every_step_and_trace_always_contains_switch_endpoint(self):
        r = REQ+.03*np.cos(2*np.pi*np.arange(32)/32)
        p = .2*np.sin(2*np.pi*np.arange(32)/32)
        kwargs = dict(beta=600., dt_fs=.02, switch_time_fs=.7, hold_time_fs=1.3)
        dense = self.changed.run(r, p, trace_stride_fs=.02, **kwargs)
        sparse = self.changed.run(r, p, trace_stride_fs=.2, **kwargs)
        np.testing.assert_array_equal(dense.r, sparse.r)
        np.testing.assert_array_equal(dense.p, sparse.p)
        self.assertEqual(dense.summary["hold_time_average"], sparse.summary["hold_time_average"])
        times = np.array([row["time_fs"] for row in dense.trace])
        holding = times >= .7-1e-12
        for name, expected in dense.summary["hold_time_average"].items():
            values = np.array([row[name] for row in dense.trace])[holding]
            actual = trapezoid(values, times[holding])/1.3
            self.assertAlmostEqual(actual, expected, delta=abs(expected)*1e-12+1e-14)
        np.testing.assert_allclose([sparse.trace[0]["time_fs"], sparse.trace[-1]["time_fs"]], [0., 2.])
        self.assertTrue(any(abs(row["time_fs"]-.7)<1e-12 for row in sparse.trace))
        self.assertIsNone(sparse.trace[0]["hold_energy_drift_per_bead_hartree"])
        work = [row["switching_work_per_bead_hartree"] for row in dense.trace if row["time_fs"] >= .7-1e-12]
        self.assertTrue(all(value == work[0] for value in work))

    def test_rejected_endpoint_keeps_last_accepted_state_and_is_never_silently_removed(self):
        model = DiatomicSwitcher(None, mass=2., omega=.01, equilibrium=2., target_omega=.01)
        result = model.run([.3], [-1.], beta=600., dt_fs=.002,
                           switch_time_fs=0., hold_time_fs=1., raise_on_error=False)
        self.assertEqual(result.summary["status"], "failed")
        self.assertGreater(result.summary["completed_steps"], 0)
        self.assertGreater(result.r[0], 0.)
        self.assertLess(result.r[0], .3)
        self.assertAlmostEqual(result.trace[-1]["time_fs"], result.summary["completed_time_fs"])
        self.assertEqual(result.trace[-1]["centroid_bond_bohr"], result.r[0])
        with self.assertRaises(DiatomicSwitchError) as caught:
            model.run([.3], [-1.], beta=600., dt_fs=.002, switch_time_fs=0., hold_time_fs=1.)
        np.testing.assert_array_equal(caught.exception.result.r, result.r)
        np.testing.assert_array_equal(caught.exception.result.p, result.p)

    def test_units_grid_and_input_shape_guards(self):
        for kwargs in ({"beta": 0}, {"beta": float("nan")}, {"dt_fs": 0},
                       {"switch_time_fs": -.1}, {"switch_time_fs": .031},
                       {"hold_time_fs": .031}, {"trace_stride_fs": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.same.run([REQ], [0.], **{"beta": 600., **kwargs})
        for r, p in (([], []), ([[REQ]], [[0.]]), ([REQ], [0., 1.]), ([np.nan], [0.])):
            with self.subTest(r=r, p=p), self.assertRaises(ValueError):
                self.same.run(r, p, beta=600.)

    def test_real_inputs_are_validated_before_float_conversion(self):
        for r, p in ((np.array([REQ+1j]), [0.]), ([REQ], np.array([2j])),
                     ([True], [0.]), ([REQ], np.array([0.], dtype=object)),
                     ([str(REQ)], [0.])):
            with self.subTest(r=r, p=p), self.assertRaises(ValueError):
                self.same.run(r, p, beta=600., switch_time_fs=0., hold_time_fs=0.)
        for value in (np.complex128(600+1j), True, "600"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.same.run([REQ], [0.], beta=value, switch_time_fs=0., hold_time_fs=0.)
        for mass in (np.complex128(2*MU+1j), np.array([2*MU+1j, 2*MU]), True, str(2*MU)):
            with self.subTest(mass=mass), self.assertRaises(ValueError):
                DiatomicSwitcher(None, mass=mass, omega=OMEGA, equilibrium=REQ, target_omega=OMEGA)
        with self.assertRaises(ValueError):
            harmonic(target_omega=np.complex128(OMEGA+1j))

    def test_overflowing_initial_reference_fails_before_accepting_a_trace_point(self):
        # The target energy is finite, but K + Vharm overflows. The second
        # observation must not be ignored after constructing this reference.
        model = DiatomicSwitcher(None, mass=.2, omega=.01, equilibrium=2., target_omega=.01,
                                  energy_zero_hartree=1e308, target_offset=-1e308)
        r, p = np.array([2.]), np.array([np.sqrt(2e307)])
        kwargs = dict(beta=600., switch_time_fs=0., hold_time_fs=0.)
        result = model.run(r, p, raise_on_error=False, **kwargs)
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(result.summary["native_status"], 2)
        self.assertEqual(result.summary["completed_steps"], 0)
        self.assertEqual(result.trace, [])
        np.testing.assert_array_equal(result.r, r)
        np.testing.assert_array_equal(result.p, p)
        json.dumps(result.summary, allow_nan=False)
        with self.assertRaisesRegex(DiatomicSwitchError, "initial diatomic energy reference"):
            model.run(r, p, **kwargs)

    def test_hold_trapezoid_avoids_intermediate_overflow_and_rejects_accumulator_overflow(self):
        model = DiatomicSwitcher(None, mass=2., omega=.01, equilibrium=2., target_omega=.01,
                                  energy_zero_hartree=1e308)
        r, p = np.array([2.01]), np.array([.2])
        kwargs = dict(beta=600., dt_fs=.02, switch_time_fs=0., trace_stride_fs=.2)
        # The endpoint sum 1e308 + 1e308 overflows, although these first two
        # integrals and their time averages are representable.
        accepted = model.run(r, p, hold_time_fs=.04, **kwargs)
        self.assertEqual(accepted.summary["status"], "completed")
        self.assertEqual(accepted.summary["completed_steps"], 2)
        self.assertAlmostEqual(accepted.summary["hold_time_average"]["full_potential_per_bead_hartree"]/1e308, 1.)
        self.assertFalse(np.array_equal(accepted.r, r))
        # The third accumulated integral truly exceeds float64. Reject its
        # phase point and every partial aggregate, retaining the second step.
        failed = model.run(r, p, hold_time_fs=.06, raise_on_error=False, **kwargs)
        self.assertEqual(failed.summary["status"], "failed")
        self.assertEqual(failed.summary["native_status"], 2)
        self.assertIn("hold time integral", failed.summary["error"])
        self.assertEqual(failed.summary["completed_steps"], 2)
        self.assertEqual(failed.summary["accepted_hold_duration_fs"], .04)
        self.assertEqual(failed.summary["hold_time_average"], accepted.summary["hold_time_average"])
        self.assertEqual(failed.trace, accepted.trace)
        np.testing.assert_array_equal(failed.r, accepted.r)
        np.testing.assert_array_equal(failed.p, accepted.p)
        for result in (accepted, failed):
            json.dumps(result.summary, allow_nan=False)
            json.dumps(result.trace, allow_nan=False)
        with self.assertRaisesRegex(DiatomicSwitchError, "hold time integral") as caught:
            model.run(r, p, hold_time_fs=.06, **kwargs)
        np.testing.assert_array_equal(caught.exception.result.r, accepted.r)

    def test_binary_provenance_is_exposed_before_any_trajectory(self):
        self.assertTrue(self.same.library_path.is_file())
        self.assertEqual(hashlib.sha256(self.same.library_path.read_bytes()).hexdigest(),
                         self.same.metadata["library_sha256"])
        self.assertIn("-std=c++17", self.same.metadata["compiler_flags"])
        self.assertEqual(len(self.same.metadata["cpp_sha256"]), 64)
        json.dumps(self.same.metadata, allow_nan=False)

    def test_direct_abi_trace_capacity_cannot_wrap_at_int64_limit(self):
        maximum = np.iinfo(np.int64).max
        for switching, holding in ((maximum, 0), (0, maximum)):
            with self.subTest(switching=switching, holding=holding):
                options = _Options(1, switching, holding, 1, .1, 600., MU, OMEGA,
                                   REQ, 0., 99., OMEGA, REQ, 0., 1)
                r, p = np.array([REQ]), np.zeros(1)
                r_out, p_out = np.full(1, -7.), np.full(1, -8.)
                trace = np.full((1, 18), -9.)
                output, error = _Output(), ctypes.create_string_buffer(256)
                status = self.same._run(ctypes.byref(options), None,
                    r, p, r_out, p_out, trace, 1, ctypes.byref(output), error, len(error))
                self.assertEqual(status, 3)
                self.assertIn(b"Insufficient trace capacity", error.value)
                self.assertEqual(output.completed_steps, 0)
                self.assertEqual(output.trace_count, 0)
                np.testing.assert_array_equal(r_out, -7.)
                np.testing.assert_array_equal(p_out, -8.)
                np.testing.assert_array_equal(trace, -9.)


@unittest.skipUnless(COMPILER and LIBRARY.is_file(), "Build verified BKMP2 for the real-PES switch checks")
class BKMP2DiatomicSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmarks.h3.pes import BKMP2Potential
        cls.model = DiatomicSwitcher(BKMP2Potential(), mass=2*MU, omega=OMEGA,
                                    equilibrium=1.4014719755049045)

    def test_asymptote_energy_and_bond_gradient_spotcheck(self):
        checks = self.model.metadata["asymptote_checks"]
        self.assertEqual(len(checks), 5)
        self.assertLess(max(abs(row["energy_difference_hartree"]) for row in checks), 4e-12)
        self.assertLess(max(abs(row["bond_gradient_difference_hartree_per_bohr"]) for row in checks), 1e-14)
        self.assertAlmostEqual(self.model.energy_zero_hartree, -.17449577089730772, places=13)

    def test_full_pes_domain_checked_even_at_lambda_zero(self):
        r = np.array([REQ, .19]+[REQ]*30)
        p = np.zeros(32)
        with self.assertRaisesRegex(DiatomicSwitchError, "including at lambda=0") as caught:
            self.model.run(r, p, beta=600., switch_time_fs=1., hold_time_fs=0.)
        result = caught.exception.result
        self.assertEqual(result.summary["completed_steps"], 0)
        self.assertEqual(result.summary["status"], "failed")
        self.assertEqual(result.trace, [])
        np.testing.assert_array_equal(result.r, r)
        np.testing.assert_array_equal(result.p, p)
        json.dumps(result.summary, allow_nan=False)

    def test_overridden_python_radial_semantics_are_not_bypassed(self):
        from benchmarks.h3.pes import BKMP2Potential
        potential = BKMP2Potential()
        potential.radial = lambda r: (0., np.zeros(3))
        with self.assertRaisesRegex(TypeError, "overridden radial"):
            DiatomicSwitcher(potential, mass=2*MU, omega=OMEGA, equilibrium=REQ)

    def test_class_radial_override_is_also_rejected(self):
        from benchmarks.h3.pes import BKMP2Potential
        with patch.object(BKMP2Potential, "radial", lambda self, r: (0., np.zeros(3))):
            with self.assertRaisesRegex(TypeError, "overridden radial"):
                DiatomicSwitcher(BKMP2Potential(), mass=2*MU, omega=OMEGA, equilibrium=REQ)

    def test_real_noncentroid_ring_short_switch_has_second_order_work_balance(self):
        index = np.arange(32)
        r = self.model.equilibrium+.07+.015*np.cos(2*np.pi*index/32)
        p = .1+.2*np.sin(2*np.pi*index/32)
        residuals = []
        for dt in (.02, .01):
            result = self.model.run(r, p, beta=600., dt_fs=dt,
                                    switch_time_fs=1., hold_time_fs=1., trace_stride_fs=.2)
            self.assertEqual(result.summary["status"], "completed")
            residuals.append(result.summary["maximum_absolute_energy_balance_residual_per_bead_hartree"])
            self.assertLess(result.summary["maximum_absolute_hold_energy_drift_per_bead_hartree"], 2e-7)
            self.assertGreater(result.trace[0]["spring_per_bead_hartree"], 0.)
            json.dumps(result.trace, allow_nan=False)
        self.assertGreater(residuals[1]/residuals[0], .22)
        self.assertLess(residuals[1]/residuals[0], .28)


if __name__ == "__main__":
    unittest.main()
