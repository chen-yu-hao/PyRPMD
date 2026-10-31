"""Analytic checks of the physical initialization, not merely array shapes."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pyrpmd.constants import ANGSTROM_TO_AU, WAVENUMBER_TO_AU
from pyrpmd.dynamics import State
from pyrpmd.initial_states import normalize_initial_ensemble, prepare_initial_ensemble
from pyrpmd.system import System


def atom_diatom(masses=(1836.0, 2000.0, 2000.0), omega=0.02, quantum=0, method="ne_rpmd", beta=400.0):
    masses = np.asarray(masses)
    geometry = np.array([[0.0, 0.0, -12.0], [0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    system = System(("H", "H", "H"), masses, geometry, ((0,), (1, 2)))
    # Analytic heteronuclear stretch eigenvector, orthogonal to sqrt(m) COM.
    mode = np.zeros((1, 2, 3))
    mode[0, 0, 2] = -np.sqrt(masses[2] / (masses[1] + masses[2]))
    mode[0, 1, 2] = np.sqrt(masses[1] / (masses[1] + masses[2]))
    section = {
        "method": method,
        "fragments": [{}, {"frequencies": [omega], "modes": mode.tolist(), "vibrational_quanta": [quantum]}],
    }
    if method == "ne_rpmd":
        section["beta"] = beta
    return system, section, mode.reshape(6)


def mode_coordinates(state, system, mode):
    q = (state.q[:, 1:] - system.geometry[None, 1:]) * np.sqrt(system.masses)[None, 1:, None]
    p = state.p[:, 1:] / np.sqrt(system.masses)[None, 1:, None]
    return q.reshape(len(q), 6) @ mode, p.reshape(len(p), 6) @ mode


def central_hessian(geometry, pairs):
    """Independent Cartesian Hessian of pairwise harmonic bond lengths."""
    hessian = np.zeros((3 * len(geometry), 3 * len(geometry)))
    for first, second, stiffness in pairs:
        direction = geometry[first] - geometry[second]
        direction = direction / np.linalg.norm(direction)
        block = stiffness * np.outer(direction, direction)
        a, b = slice(3 * first, 3 * first + 3), slice(3 * second, 3 * second + 3)
        hessian[a, a] += block
        hessian[b, b] += block
        hessian[a, b] -= block
        hessian[b, a] -= block
    return hessian


def atom_triatom():
    geometry = np.array([[0.0, 0.0, -15.0], [0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.4, 1.6, 0.0]])
    masses = np.array([1836.0, 3000.0, 4000.0, 5000.0])
    system = System(("H", "H", "H", "H"), masses, geometry, ((0,), (1, 2, 3)))
    hessian = central_hessian(geometry[1:], [(0, 1, 1.2), (0, 2, 0.8), (1, 2, 0.5)])
    section = {"method": "ne_rpmd", "beta": 500.0,
               "fragments": [{}, {"hessian": hessian.tolist(), "vibrational_quanta": [0, 0, 1]}]}
    return system, section, hessian


class InitialEnsembleAnalyticTests(unittest.TestCase):
    def test_harmonic_canonical_variances_and_free_translation_spreads(self):
        system, section, mode = atom_diatom()
        prepared = prepare_initial_ensemble(section, system, 8, 400.0)
        rng = np.random.default_rng(422)
        qs, ps, atom_spreads, com_spreads = [], [], [], []
        for _ in range(4000):
            state = prepared.sample(rng)
            q, p = mode_coordinates(state, system, mode)
            qs.append(prepared.ring_transform @ q)
            ps.append(prepared.ring_transform @ p)
            delta = state.q[:, 0] - state.q[:, 0].mean(axis=0)
            atom_spreads.append(np.mean(np.sum(delta * delta, axis=1)))
            com = np.average(state.q[:, 1:], axis=1, weights=system.masses[1:])
            com_spreads.append(np.mean(np.sum((com - com.mean(axis=0)) ** 2, axis=1)))
        # Exact finite-P canonical covariances, independently from the sampled
        # arrays; do not compare against a sampler-produced energy statistic.
        expected_p = 8.0 / 400.0
        expected_q = expected_p / (0.02 ** 2 + prepared.ring_frequencies ** 2)
        np.testing.assert_allclose(np.var(qs, axis=0), expected_q, rtol=0.08)
        np.testing.assert_allclose(np.var(ps, axis=0), expected_p, rtol=0.08)
        # Sum csc^2(pi*k/P)=(P^2-1)/3, including all three free directions.
        expected_atom = 400.0 / (4 * system.masses[0]) * (1 - 1 / 8 ** 2)
        expected_com = 400.0 / (4 * system.masses[1:].sum()) * (1 - 1 / 8 ** 2)
        self.assertAlmostEqual(np.mean(atom_spreads) / expected_atom, 1.0, delta=0.04)
        self.assertAlmostEqual(np.mean(com_spreads) / expected_com, 1.0, delta=0.04)

    def test_excitation_adds_only_nu_quanta_to_centroid_for_any_bead_count(self):
        # Identical random draws isolate the kick exactly, including its sqrt(P)
        # normalization; no noisy subtraction of unrelated ensemble averages.
        for beads in (2, 5, 32):
            system, ground, mode = atom_diatom()
            excited = copy.deepcopy(ground)
            excited["fragments"][1]["vibrational_quanta"] = [2]
            first = prepare_initial_ensemble(ground, system, beads, 400.0)
            second = prepare_initial_ensemble(excited, system, beads, 400.0)
            for seed in range(8):
                a = first.sample(np.random.default_rng(seed))
                b = second.sample(np.random.default_rng(seed))
                aq, ap = mode_coordinates(a, system, mode)
                bq, bp = mode_coordinates(b, system, mode)
                dq, dp = first.ring_transform @ (bq - aq), first.ring_transform @ (bp - ap)
                np.testing.assert_allclose(dq[1:], 0, atol=2e-13)
                np.testing.assert_allclose(dp[1:], 0, atol=2e-14)
                self.assertAlmostEqual((dp[0] ** 2 + 0.02 ** 2 * dq[0] ** 2) / (2 * beads), 2 * 0.02, places=13)

    def test_qct_contains_half_quantum_and_uniform_phase(self):
        system, section, mode = atom_diatom(method="qct", quantum=2)
        prepared = prepare_initial_ensemble(section, system, 1, 400.0)
        rng = np.random.default_rng(724)
        coordinates = []
        for _ in range(1200):
            state = prepared.sample(rng)
            q, p = mode_coordinates(state, system, mode)
            self.assertAlmostEqual(float(0.5 * (p[0] ** 2 + 0.02 ** 2 * q[0] ** 2)), 2.5 * 0.02, places=13)
            coordinates.append([q[0], p[0]])
        coordinates = np.asarray(coordinates)
        np.testing.assert_allclose(coordinates.mean(axis=0) / np.sqrt(coordinates.var(axis=0)), 0, atol=0.07)
        self.assertAlmostEqual(np.mean(coordinates[:, 0] ** 2) / (2.5 / 0.02), 1.0, delta=0.06)
        self.assertAlmostEqual(np.mean(coordinates[:, 1] ** 2) / (2.5 * 0.02), 1.0, delta=0.06)

    def test_finite_bead_ground_energy_converges_to_quantum_harmonic_value(self):
        system, section, _ = atom_diatom()
        exact = 0.5 * 0.02 / np.tanh(0.5 * 400.0 * 0.02)
        errors = []
        for beads in (2, 8, 32, 128):
            metadata = prepare_initial_ensemble(section, system, beads, 400.0).metadata
            energy = metadata["fragments"][1]["expected_mode_energies_before_j0_projection_au"][0]
            errors.append(abs(energy - exact))
        self.assertTrue(all(later < earlier for earlier, later in zip(errors, errors[1:])))
        self.assertLess(errors[-1] / exact, 0.0006)

    def test_primitive_estimator_uses_negative_spring_term(self):
        system, section, mode = atom_diatom()
        prepared = prepare_initial_ensemble(section, system, 8, 400.0)
        mode_q, mode_p = np.zeros(8), np.zeros(8)
        mode_q[-1], mode_p[-1] = 0.4, 0.03
        bead_q, bead_p = prepared.ring_transform.T @ mode_q, prepared.ring_transform.T @ mode_p
        q = np.tile(system.geometry, (8, 1, 1))
        p = np.zeros_like(q)
        q[:, 1:] += (bead_q[:, None] * mode).reshape(8, 2, 3) / np.sqrt(system.masses)[None, 1:, None]
        p[:, 1:] = (bead_p[:, None] * mode).reshape(8, 2, 3) * np.sqrt(system.masses)[None, 1:, None]
        value = prepared.diagnostics(State(q, p))["fragments"][1]
        expected = (0.03 ** 2 + (0.02 ** 2 - (2 * 8 / 400.0) ** 2) * 0.4 ** 2) / (2 * 8)
        self.assertAlmostEqual(value["primitive_harmonic_energy_au"], expected, places=16)
        self.assertAlmostEqual(value["harmonic_mode_actions"][0], expected / 0.02 - 0.5, places=14)

    def test_mass_weighted_hessian_frequency_tracks_isotope_reduced_mass(self):
        stiffness = 0.9
        for mass_b, mass_c in ((1836.0, 1836.0), (1836.0, 3672.0)):
            system, section, _ = atom_diatom(masses=(1836.0, mass_b, mass_c))
            hessian = central_hessian(system.geometry[1:], [(0, 1, stiffness)])
            section["fragments"][1] = {"hessian": hessian.tolist()}
            prepared = prepare_initial_ensemble(section, system, 16, 400.0)
            expected = np.sqrt(stiffness * (1 / mass_b + 1 / mass_c))
            self.assertAlmostEqual(prepared.references[1].frequencies[0], expected, places=15)
            state = prepared.sample(np.random.default_rng(22))
            centroid = np.average(state.q[:, 1:].mean(axis=0), axis=0, weights=system.masses[1:])
            np.testing.assert_allclose(centroid, np.average(system.geometry[1:], axis=0, weights=system.masses[1:]), atol=1e-14)

    def test_nonlinear_reference_projects_six_rigid_modes_and_j0(self):
        system, section, hessian = atom_triatom()
        prepared = prepare_initial_ensemble(section, system, 16, 500.0)
        reference = prepared.references[1]
        self.assertEqual(reference.nmodes, 3)
        self.assertEqual(reference.rotational_rank, 3)
        root_mass = np.repeat(np.sqrt(system.masses[1:]), 3)
        eigenvalues = np.linalg.eigvalsh(hessian / np.outer(root_mass, root_mass))
        np.testing.assert_allclose(reference.frequencies ** 2, eigenvalues[-3:], rtol=1e-12)
        for seed in range(20):
            state = prepared.sample(np.random.default_rng(seed))
            diagnostic = prepared.diagnostics(state)
            for fragment in diagnostic["fragments"]:
                np.testing.assert_allclose(fragment["centroid_com_momentum_au"], 0, atol=2e-14)
                np.testing.assert_allclose(fragment["centroid_angular_momentum_au"], 0, atol=2e-13)

    def test_general_linear_molecule_has_two_rotations_and_four_vibrations(self):
        geometry = np.array([[0, 0, -12], [0, 0, -2], [0, 0, 0], [0, 0, 2]], dtype=float)
        system = System(("H",) * 4, np.full(4, 2000.0), geometry, ((0,), (1, 2, 3)))
        modes = np.zeros((4, 3, 3))
        modes[0, :, 2] = np.array([1, 0, -1]) / np.sqrt(2)
        modes[1, :, 2] = np.array([1, -2, 1]) / np.sqrt(6)
        modes[2, :, 0] = np.array([1, -2, 1]) / np.sqrt(6)
        modes[3, :, 1] = np.array([1, -2, 1]) / np.sqrt(6)
        section = {"method": "qct", "fragments": [{}, {"modes": modes.tolist(), "frequencies": [0.01, 0.02, 0.03, 0.03]}]}
        prepared = prepare_initial_ensemble(section, system, 1, 500.0)
        self.assertEqual(prepared.references[1].nmodes, 4)
        self.assertEqual(prepared.references[1].rotational_rank, 2)
        diagnostic = prepared.diagnostics(prepared.sample(np.random.default_rng(8)))
        np.testing.assert_allclose(diagnostic["fragments"][1]["centroid_angular_momentum_au"], 0, atol=1e-13)

    def test_diagnostics_invariant_under_rigid_rotations_and_translation_kicks(self):
        system, section, _ = atom_triatom()
        prepared = prepare_initial_ensemble(section, system, 16, 500.0)
        state = prepared.sample(np.random.default_rng(5))
        original = prepared.diagnostics(state)
        axis = np.array([1.0, 2.0, -1.0]) / np.sqrt(6)
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        rotation = np.eye(3) + np.sin(0.83) * cross + (1 - np.cos(0.83)) * (cross @ cross)
        transformed = State(state.q @ rotation.T + [3, -5, 2], state.p @ rotation.T + system.masses[None, :, None] * np.array([0.03, 0.01, -0.02]))
        changed = prepared.diagnostics(transformed)
        for before, after in zip(original["fragments"], changed["fragments"]):
            for key in ("primitive_harmonic_mode_energies_au", "harmonic_mode_actions", "atom_ring_radii_of_gyration_au"):
                np.testing.assert_allclose(after[key], before[key], rtol=1e-10, atol=1e-12)

    def test_local_rng_determinism_and_owned_reference_buffers(self):
        system, section, _ = atom_diatom()
        prepared = prepare_initial_ensemble(section, system, 8, 400.0)
        a = prepared.sample(np.random.default_rng([123, 5, 47]))
        for seed in range(4):
            prepared.sample(np.random.default_rng(seed))
        b = prepared.sample(np.random.default_rng([123, 5, 47]))
        np.testing.assert_array_equal(a.q, b.q)
        np.testing.assert_array_equal(a.p, b.p)
        metadata = prepared.metadata
        original_hash = metadata["fragments"][1]["reference_sha256"]
        section["fragments"][1]["frequencies"][0] = 7.0
        system.geometry[:] = 1234.0
        metadata["fragments"][1]["frequencies_au"][0] = 8.0
        self.assertEqual(prepared.metadata["fragments"][1]["reference_sha256"], original_hash)
        np.testing.assert_array_equal(a.q, prepared.sample(np.random.default_rng([123, 5, 47])).q)
        with self.assertRaises(ValueError):
            prepared.references[1].frequencies[0] = 5.0
        with self.assertRaises(TypeError):
            prepared.sample(None)
        json.dumps(prepared.metadata, allow_nan=False)
        json.dumps(prepared.diagnostics(a), allow_nan=False)

    def test_units_file_contents_and_canonical_idempotence(self):
        system, section, _ = atom_diatom()
        section["fragments"][1]["frequency_unit"] = "cm^-1"
        section["fragments"][1]["frequencies"] = [0.02 / WAVENUMBER_TO_AU]
        section["beta"] = "400 hartree^-1"
        normalized = normalize_initial_ensemble(section)
        self.assertEqual(normalized, normalize_initial_ensemble(normalized))
        self.assertAlmostEqual(normalized["fragments"][1]["frequencies"][0], 0.02)
        hessian = central_hessian(system.geometry[1:], [(0, 1, 0.4)])
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".npy", ".txt"):
                path = Path(directory) / ("hessian" + suffix)
                supplied = hessian * ANGSTROM_TO_AU ** 2
                if suffix == ".npy":
                    np.save(path, supplied)
                else:
                    np.savetxt(path, supplied)
                section["fragments"][1] = {"hessian_file": str(path), "hessian_unit": "hartree/angstrom^2"}
                normalized = normalize_initial_ensemble(section)
                path.unlink()
                np.testing.assert_allclose(normalized["fragments"][1]["hessian"], hessian)
                prepared = prepare_initial_ensemble(normalized, system, 4, 400.0)
                self.assertAlmostEqual(prepared.references[1].frequencies[0], np.sqrt(0.4 / 1000))

    def test_atomic_free_rings_and_single_bead_qct_have_zero_centroid_momentum(self):
        system = System(("H", "H"), np.array([2000.0, 4000.0]), np.array([[0., 0., -2.], [0., 0., 2.]]), ((0,), (1,)))
        for method, beads in (("qct", 1), ("ne_rpmd", 32)):
            section = {"method": method, "fragments": [{}, {}]}
            if method == "ne_rpmd":
                section["beta"] = 300.0
            prepared = prepare_initial_ensemble(section, system, beads, 300.0)
            state = prepared.sample(np.random.default_rng(3))
            np.testing.assert_allclose(state.q.mean(axis=0), system.geometry, atol=1e-14)
            np.testing.assert_allclose(state.p.mean(axis=0), 0, atol=1e-14)
            if method == "qct":
                np.testing.assert_array_equal(state.p, 0)
            else:
                self.assertGreater(np.linalg.norm(state.q - system.geometry), 0)
            json.dumps(prepared.diagnostics(state), allow_nan=False)


class InitialEnsembleValidationTests(unittest.TestCase):
    def test_invalid_method_quantum_units_and_sources(self):
        system, section, _ = atom_diatom()
        invalid = []
        for key, value in (("method", "thermal"), ("beta", 0), ("beta", True), ("mystery", 1)):
            item = copy.deepcopy(section)
            item[key] = value
            invalid.append(item)
        item = copy.deepcopy(section)
        item.pop("beta")
        invalid.append(item)
        for key, value in (("vibrational_quanta", [True]), ("vibrational_quanta", [-1]),
                           ("vibrational_quanta", [1.2]), ("vibrational_quanta", []),
                           ("j", 1), ("frequencies", [-0.02]), ("frequencies", [float("nan")]),
                           ("frequency_unit", "bohr"), ("hessian", np.eye(6).tolist())):
            item = copy.deepcopy(section)
            item["fragments"][1][key] = value
            invalid.append(item)
        for item in invalid:
            with self.subTest(item=item):
                with self.assertRaises(ValueError):
                    prepare_initial_ensemble(item, system, 8, 400.0)

    def test_wrong_modes_hessians_and_isotope_reference_fail(self):
        system, section, _ = atom_diatom()
        wrong_modes = []
        for factor in (2.0, 0.0):
            item = copy.deepcopy(section)
            item["fragments"][1]["modes"] = (np.asarray(item["fragments"][1]["modes"]) * factor).tolist()
            wrong_modes.append(item)
        item = copy.deepcopy(section)
        item["fragments"][1]["modes"] = [[[0, 0, 1 / np.sqrt(2)], [0, 0, 1 / np.sqrt(2)]]]
        wrong_modes.append(item)
        for item in wrong_modes:
            with self.assertRaises(ValueError):
                prepare_initial_ensemble(item, system, 8, 400.0)
        isotope = System(system.symbols, np.array([1836, 2000, 4000]), system.geometry, system.fragments)
        with self.assertRaisesRegex(ValueError, "isotope"):
            prepare_initial_ensemble(section, isotope, 8, 400.0)
        hessian = central_hessian(system.geometry[1:], [(0, 1, 0.4)])
        asymmetric = hessian.copy()
        asymmetric[0, 1] = 0.1
        for value in (np.eye(5), np.zeros((6, 6)), -hessian, asymmetric):
            item = {"method": "ne_rpmd", "beta": 400, "fragments": [{}, {"hessian": value.tolist()}]}
            with self.assertRaises(ValueError):
                prepare_initial_ensemble(item, system, 8, 400.0)

    def test_incompatible_beads_beta_and_system_fail(self):
        system, section, _ = atom_diatom(method="qct")
        with self.assertRaisesRegex(ValueError, "beads=1"):
            prepare_initial_ensemble(section, system, 32, 400.0)
        system, section, _ = atom_diatom()
        for beads, beta in ((True, 400), (0, 400), (1, 400), (8, -1), (8, 500)):
            with self.assertRaises(ValueError):
                prepare_initial_ensemble(section, system, beads, beta)
        for changes in ({"fragments": [[0, 1], [1, 2]]}, {"pbc": [True, False, False]}, {"fixed_atoms": [0]}):
            invalid = system.to_dict()
            invalid.update(changes)
            with self.assertRaises(ValueError):
                prepare_initial_ensemble(section, invalid, 8, 400.0)


if __name__ == "__main__":
    unittest.main()
