"""Independent physical limits for the fixed-geometry electrochemical PCET model.

The references below use analytic harmonic oscillators and ordinary energy
quadrature.  They do not use the production log-quadrature or rate estimator
to construct expected values.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

from pyrpmd.constants import ATOMIC_TIME_S, EV_TO_AU, KB_AU
from pyrpmd.pcet import (
    ISOTOPE_MASSES_AU,
    electrode_log_kernels,
    golden_rule_rates,
    solve_proton,
)


_KB_EV = 1.380649e-23 / 1.602176634e-19


def _ordinary_electrode_integrals(gap_ev, reorganization_ev, temperature):
    """Integrate the forward and reverse physical expressions independently."""
    thermal = _KB_EV * temperature
    lam = reorganization_ev
    normalizer = math.sqrt(4.0 * math.pi * lam * thermal)
    bound = abs(gap_ev) + lam + 14.0 * math.sqrt(2.0 * lam * thermal)

    def reduction(energy):
        gaussian = math.exp(-((lam + gap_ev - energy) ** 2) / (4.0 * lam * thermal))
        return expit(-energy / thermal) * gaussian / normalizer

    def oxidation(energy):
        gaussian = math.exp(-((lam - gap_ev + energy) ** 2) / (4.0 * lam * thermal))
        return expit(energy / thermal) * gaussian / normalizer

    points = sorted({0.0, gap_ev - lam, gap_ev + lam})
    return tuple(
        quad(function, -bound, bound, points=points, epsabs=1e-14, epsrel=2e-12)[0]
        for function in (reduction, oxidation)
    )


def _harmonic_spectrum(mass, stiffness, center=0.0, nstates=8, offset=0.0):
    grid = np.linspace(-3.0, 3.0, 1201)
    potential = 0.5 * stiffness * (grid - center) ** 2 + offset
    return solve_proton(
        grid, potential, mass, nstates, energy_origin_au=offset
    )


def _harmonic_log_partition(mass, stiffness, temperature):
    quantum = math.sqrt(stiffness / mass)
    scaled = quantum / (KB_AU * temperature)
    return -0.5 * scaled - math.log(-math.expm1(-scaled))


class ProtonGridScienceTests(unittest.TestCase):
    def test_harmonic_energies_normalization_and_isotope_scaling(self):
        spectra = {}
        stiffness = 0.03
        for isotope in ("H", "D"):
            mass = ISOTOPE_MASSES_AU[isotope]
            spectrum = _harmonic_spectrum(mass, stiffness, nstates=6)
            spectra[isotope] = spectrum
            exact = math.sqrt(stiffness / mass) * (np.arange(6) + 0.5)
            # The declared method is second-order finite differences, not DVR.
            np.testing.assert_allclose(spectrum.energies_au, exact, rtol=1.5e-4)
            vectors = spectrum.wavefunctions
            self.assertEqual(vectors.shape, (1201, 6))
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(6), atol=2e-13)
            np.testing.assert_array_equal(vectors[[0, -1]], 0.0)
            self.assertLess(float(np.max(spectrum.boundary_probabilities)), 1e-10)
        expected_ratio = math.sqrt(ISOTOPE_MASSES_AU["H"] / ISOTOPE_MASSES_AU["D"])
        self.assertAlmostEqual(
            spectra["D"].energies_au[0] / spectra["H"].energies_au[0],
            expected_ratio,
            delta=5e-5,
        )

    def test_displaced_well_franck_condon_overlap_for_hydrogen_and_deuterium(self):
        stiffness, distance = 0.03, 0.6
        overlaps = {}
        for isotope in ("H", "D"):
            mass = ISOTOPE_MASSES_AU[isotope]
            left = _harmonic_spectrum(mass, stiffness, -distance / 2.0, nstates=3)
            right = _harmonic_spectrum(mass, stiffness, distance / 2.0, nstates=3)
            overlap_squared = float(left.wavefunctions[:, 0] @ right.wavefunctions[:, 0]) ** 2
            exact = math.exp(-math.sqrt(mass * stiffness) * distance**2 / 2.0)
            np.testing.assert_allclose(overlap_squared, exact, rtol=2e-4)
            overlaps[isotope] = overlap_squared
        self.assertGreater(overlaps["H"], overlaps["D"])

    def test_energy_origin_removes_constant_without_removing_zero_point_energy(self):
        mass = ISOTOPE_MASSES_AU["H"]
        original = _harmonic_spectrum(mass, 0.03, nstates=4)
        shifted = _harmonic_spectrum(mass, 0.03, nstates=4, offset=7.25)
        np.testing.assert_allclose(shifted.energies_au, original.energies_au, atol=2e-12)
        np.testing.assert_allclose(shifted.potential_au, original.potential_au, atol=1e-15)
        self.assertAlmostEqual(shifted.energy_origin_au, 7.25)
        self.assertGreater(original.energies_au[0], 0.0)

    def test_grid_refinement_reduces_harmonic_discretization_error(self):
        mass, stiffness = ISOTOPE_MASSES_AU["H"], 0.03
        exact = 0.5 * math.sqrt(stiffness / mass)
        errors = []
        for size in (301, 601):
            grid = np.linspace(-3.0, 3.0, size)
            spectrum = solve_proton(grid, 0.5 * stiffness * grid**2, mass, 2)
            errors.append(abs(spectrum.energies_au[0] - exact))
        # Halving h should reduce the leading O(h^2) eigenvalue error fourfold.
        self.assertGreater(errors[0] / errors[1], 3.8)
        self.assertLess(errors[0] / errors[1], 4.2)


class ElectrodeIntegralScienceTests(unittest.TestCase):
    def test_fermi_kernels_match_independent_forward_and_reverse_integrals(self):
        gaps_ev = np.asarray([-0.2, 0.0, 0.2])
        log_red, log_ox, _ = electrode_log_kernels(
            gaps_ev * EV_TO_AU,
            temperature_K=300.0,
            reorganization_au=0.7 * EV_TO_AU,
            rtol=2e-11,
            atol=1e-12,
        )
        independent = np.asarray(
            [_ordinary_electrode_integrals(gap, 0.7, 300.0) for gap in gaps_ev]
        )
        np.testing.assert_allclose(np.exp(log_red), independent[:, 0], rtol=2e-9)
        np.testing.assert_allclose(np.exp(log_ox), independent[:, 1], rtol=2e-9)
        # Independent numerical reference in eV, not assembled from production constants.
        np.testing.assert_allclose(
            np.exp(log_red),
            [0.005379896007443975, 0.00018081623574768633, 2.3492095482594676e-6],
            rtol=2e-9,
        )
        np.testing.assert_allclose(
            np.log(independent[:, 0] / independent[:, 1]),
            -gaps_ev / (_KB_EV * 300.0),
            atol=2e-10,
        )

    def test_large_reorganization_kernel_retains_exponentially_small_rates(self):
        # At zero driving force the integral has a separate symmetric sech form.
        # This checks a very small value without integrating a tiny raw integrand.
        lam_ev, temperature = 5.0, 100.0
        scaled = lam_ev / (_KB_EV * temperature)

        def symmetric_integrand(value):
            decay = math.exp(-abs(value))
            sech = 2.0 * decay / (1.0 + decay * decay)
            return math.exp(-(value * value) / scaled) * sech

        integral = quad(symmetric_integrand, -50.0, 50.0, epsabs=1e-13, epsrel=1e-12)[0]
        expected = -scaled / 4.0 - math.log(2.0 * math.sqrt(math.pi * scaled)) + math.log(integral)
        log_red, log_ox, _ = electrode_log_kernels(
            np.asarray([0.0]),
            temperature_K=temperature,
            reorganization_au=lam_ev * EV_TO_AU,
        )
        self.assertAlmostEqual(float(log_red[0]), expected, delta=2e-8)
        self.assertAlmostEqual(float(log_ox[0]), expected, delta=2e-8)


class PCETRateScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spectra = {
            isotope: (
                _harmonic_spectrum(ISOTOPE_MASSES_AU[isotope], 0.03, -0.15),
                _harmonic_spectrum(ISOTOPE_MASSES_AU[isotope], 0.055, 0.15),
            )
            for isotope in ("H", "D")
        }
        cls.options = dict(
            temperature_K=300.0,
            driving_force_au=-0.002,
            reorganization_au=0.025,
            coupling_au=4e-5,
            density_per_hartree=1.0 / EV_TO_AU,
        )

    def test_unequal_well_rates_obey_analytic_thermal_detailed_balance_for_both_isotopes(self):
        free_energies = {}
        temperature = self.options["temperature_K"]
        thermal = KB_AU * temperature
        for isotope, spectra in self.spectra.items():
            rates = golden_rule_rates(*spectra, **self.options)
            mass = ISOTOPE_MASSES_AU[isotope]
            log_z_o = _harmonic_log_partition(mass, 0.03, temperature)
            log_z_r = _harmonic_log_partition(mass, 0.055, temperature)
            expected_proton_free_energy = -thermal * (log_z_r - log_z_o)
            expected_log_ratio = -self.options["driving_force_au"] / thermal + log_z_r - log_z_o
            actual_log_ratio = rates["log_reduction_rate_s"] - rates["log_oxidation_rate_s"]
            self.assertAlmostEqual(actual_log_ratio, expected_log_ratio, delta=1e-4)
            self.assertAlmostEqual(rates["proton_free_energy_difference_au"], expected_proton_free_energy, delta=1e-7)
            self.assertAlmostEqual(
                rates["reaction_free_energy_au"],
                self.options["driving_force_au"] + expected_proton_free_energy,
                delta=1e-7,
            )
            self.assertAlmostEqual(rates["expected_log_rate_ratio"], actual_log_ratio, delta=1e-9)
            self.assertLess(abs(rates["detailed_balance_log_residual"]), 1e-9)
            for direction in ("reduction", "oxidation"):
                contributions = np.asarray(rates[f"{direction}_contributions_s"])
                self.assertTrue(np.all(contributions >= 0.0))
                np.testing.assert_allclose(contributions.sum(), rates[f"{direction}_rate_s"], rtol=2e-12)
            free_energies[isotope] = rates["proton_free_energy_difference_au"]
        # Same mass-independent potentials must produce different H/D ZPE corrections.
        self.assertGreater(free_energies["H"], free_energies["D"])

    def test_identical_proton_wells_give_unit_isotope_effect_and_correct_rate_units(self):
        gap_ev = -0.1
        options = dict(self.options, driving_force_au=gap_ev * EV_TO_AU)
        expected_kernel = _ordinary_electrode_integrals(
            gap_ev, options["reorganization_au"] / EV_TO_AU, options["temperature_K"]
        )
        prefactor_s = (
            2.0 * math.pi * options["density_per_hartree"]
            * options["coupling_au"] ** 2 / ATOMIC_TIME_S
        )
        isotope_rates = {}
        for isotope in ("H", "D"):
            spectrum = self.spectra[isotope][0]
            rates = golden_rule_rates(spectrum, spectrum, **options)
            isotope_rates[isotope] = rates
            np.testing.assert_allclose(
                [rates["reduction_rate_s"], rates["oxidation_rate_s"]],
                prefactor_s * np.asarray(expected_kernel),
                rtol=2e-9,
            )
        for direction in ("reduction", "oxidation"):
            self.assertAlmostEqual(
                isotope_rates["H"][f"{direction}_rate_s"]
                / isotope_rates["D"][f"{direction}_rate_s"],
                1.0,
                delta=2e-10,
            )

    def test_rate_scales_quadratically_with_electronic_coupling_and_vanishes_at_zero(self):
        spectra = self.spectra["H"]
        reference = golden_rule_rates(*spectra, **self.options)
        doubled = golden_rule_rates(
            *spectra, **dict(self.options, coupling_au=2.0 * self.options["coupling_au"])
        )
        zero = golden_rule_rates(*spectra, **dict(self.options, coupling_au=0.0))
        for direction in ("reduction", "oxidation"):
            self.assertAlmostEqual(doubled[f"{direction}_rate_s"] / reference[f"{direction}_rate_s"], 4.0, delta=1e-10)
            self.assertEqual(zero[f"{direction}_rate_s"], 0.0)
            self.assertIsNone(zero[f"log_{direction}_rate_s"])

    def test_interchanging_states_and_reversing_bias_interchanges_rates(self):
        oxidized, reduced = self.spectra["H"]
        forward = golden_rule_rates(oxidized, reduced, **self.options)
        reverse = golden_rule_rates(
            reduced, oxidized,
            **dict(self.options, driving_force_au=-self.options["driving_force_au"]),
        )
        np.testing.assert_allclose(
            [forward["reduction_rate_s"], forward["oxidation_rate_s"]],
            [reverse["oxidation_rate_s"], reverse["reduction_rate_s"]],
            rtol=2e-10,
        )

    def test_invalid_physical_inputs_are_rejected(self):
        invalid = (
            ("temperature_K", 0.0),
            ("temperature_K", float("nan")),
            ("driving_force_au", float("inf")),
            ("reorganization_au", -0.1),
            ("coupling_au", float("nan")),
            ("density_per_hartree", -1.0),
            ("density_per_hartree", float("inf")),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    golden_rule_rates(*self.spectra["H"], **dict(self.options, **{field: value}))
        # Electron transfer does not change the isotope of the transferred nucleus.
        with self.assertRaises(ValueError):
            golden_rule_rates(self.spectra["H"][0], self.spectra["D"][1], **self.options)


class ProtonGridInputTests(unittest.TestCase):
    def test_invalid_grids_and_spectra_are_rejected(self):
        grid = np.linspace(-3.0, 3.0, 101)
        potential = 0.015 * grid**2
        mass = ISOTOPE_MASSES_AU["H"]
        for bad_grid in (grid[::-1], np.zeros_like(grid), grid**3):
            with self.subTest(grid=bad_grid[:3]):
                with self.assertRaises(ValueError):
                    solve_proton(bad_grid, potential, mass, 4)
        bad_potential = potential.copy()
        bad_potential[20] = float("nan")
        for bad_mass, bad_states, curve in (
            (0.0, 4, potential),
            (-1.0, 4, potential),
            (mass, 0, potential),
            (mass, 200, potential),
            (mass, 4, bad_potential),
        ):
            with self.subTest(mass=bad_mass, states=bad_states):
                with self.assertRaises(ValueError):
                    solve_proton(grid, curve, bad_mass, bad_states)


if __name__ == "__main__":
    unittest.main()
