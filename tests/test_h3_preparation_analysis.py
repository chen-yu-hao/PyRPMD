"""Independent counting/unknown-bound checks; no PES or trajectory simulation."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import unittest

import numpy as np

from benchmarks.h3.analyze_preparation_study import (
    Case, POSSIBLE, binomial_estimate, clopper_pearson, pair_cases, paired_estimate, paired_audit_means,
    unresolved_diagnostics,
)


class PreparationAnalysisTests(unittest.TestCase):
    def test_binomial_boundaries_and_unknown_denominator(self):
        low, high = clopper_pearson(0, 20)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1-.025**(1/20))
        reverse = clopper_pearson(20, 20)
        self.assertAlmostEqual(reverse[0], 1-high)
        self.assertEqual(reverse[1], 1)
        result = binomial_estimate(["product", "nonreactive", "unresolved", "failed"], 8.)
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["unknown"], 2)
        self.assertEqual(result["sigma_identification_lower_angstrom2"], 2.)
        self.assertEqual(result["sigma_identification_upper_angstrom2"], 6.)

    def test_known_discordances_and_sign(self):
        a = ["nonreactive"]*4+["product"]*3
        b = ["product"]*2+["nonreactive"]*2+["nonreactive"]+["product"]*2
        result = paired_estimate(a, b, 7.)
        self.assertEqual(result["gained_reactivity_definite"], 2)
        self.assertEqual(result["lost_reactivity_definite"], 1)
        self.assertEqual(result["delta_sigma_angstrom2"], 1.)
        reverse = paired_estimate(b, a, 7.)
        self.assertAlmostEqual(reverse["delta_sigma_conservative_ci95_lower_angstrom2"],
                               -result["delta_sigma_conservative_ci95_upper_angstrom2"])
        self.assertAlmostEqual(reverse["delta_sigma_conservative_ci95_upper_angstrom2"],
                               -result["delta_sigma_conservative_ci95_lower_angstrom2"])

    def test_unknown_identification_bounds_exhaust_all_assignments(self):
        a = ["product", "nonreactive", "unresolved", "failed", "unresolved"]
        b = ["unresolved", "failed", "product", "nonreactive", "failed"]
        result = paired_estimate(a, b, 5.)
        gains, losses, differences = [], [], []
        assignments = [list(itertools.product(POSSIBLE[x], POSSIBLE[y])) for x, y in zip(a, b)]
        for completed in itertools.product(*assignments):
            plus = sum(x == 0 and y == 1 for x, y in completed)
            minus = sum(x == 1 and y == 0 for x, y in completed)
            gains.append(plus)
            losses.append(minus)
            differences.append(plus-minus)
            aa = ["product" if x else "nonreactive" for x, _ in completed]
            bb = ["product" if y else "nonreactive" for _, y in completed]
            resolved = paired_estimate(aa, bb, 5.)
            self.assertLessEqual(result["delta_sigma_conservative_ci95_lower_angstrom2"],
                                 resolved["delta_sigma_conservative_ci95_lower_angstrom2"])
            self.assertGreaterEqual(result["delta_sigma_conservative_ci95_upper_angstrom2"],
                                    resolved["delta_sigma_conservative_ci95_upper_angstrom2"])
        self.assertEqual(result["gained_reactivity_definite"], min(gains))
        self.assertEqual(result["gained_reactivity_possible"], max(gains))
        self.assertEqual(result["lost_reactivity_definite"], min(losses))
        self.assertEqual(result["lost_reactivity_possible"], max(losses))
        self.assertEqual(result["delta_sigma_identification_lower_angstrom2"], min(differences))
        self.assertEqual(result["delta_sigma_identification_upper_angstrom2"], max(differences))
        self.assertIsNone(result["delta_sigma_angstrom2"])

    @staticmethod
    def case(variant, ids=(10, 11)):
        config = {"method": "ne_rpmd", "v": 0, "total_energy_ev": .55, "beads": 32,
                  "samples": len(ids), "seed": 9, "bmin": 0., "bmax": 3.5,
                  "bond_cutoff_bohr": 2.5, "beta_override": None}
        summary = {"omega_au": .02, "collision_energy_ev": .28, "beta_au": 280.}
        rows = {i: {"id": i, "status": "nonreactive", "b_bohr": i/10,
                    "initial_axial_bohr": 10., "initial_bond_bohr": 1.4,
                    "initial_rg_bohr": [.1, .2, .2]} for i in ids}
        return Case(Path("unused"), variant, config, summary, rows, [])

    def test_pairing_checks_ids_and_serialized_fields(self):
        a, b = self.case("ne_paper_subset"), self.case("ne_uncut")
        result = pair_cases(a, b, "check", same_conditioning=False)
        self.assertEqual(result["paired_samples"], 2)
        b.rows[10]["initial_axial_bohr"] += .1
        self.assertEqual(pair_cases(a, b, "check", same_conditioning=False)
                         ["serialized_launch_checks"]["initial_axial_bohr"]["different"], 1)
        with self.assertRaisesRegex(ValueError, "initial_axial"):
            pair_cases(a, b, "check", same_conditioning=True)
        b.rows[10]["b_bohr"] += .1
        with self.assertRaisesRegex(ValueError, "b_bohr"):
            pair_cases(a, b, "check", same_conditioning=False)
        with self.assertRaisesRegex(ValueError, "equal ID sets"):
            pair_cases(a, self.case("ne_uncut", (11, 12)), "check", same_conditioning=False)

    def test_endpoint_derivative_uses_centroid_momentum(self):
        case = self.case("ne_uncut", (10,))
        q = np.array([[[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]]])
        # Expansion r_dot = r / tau with tau=2 atomic time units.
        mass = 1837.1526474396187
        case.rows[10].update(status="unresolved", endpoint_q_bohr=q.tolist(),
                             endpoint_p_au=(mass*q/2).tolist())
        result = unresolved_diagnostics(case)[0]
        self.assertTrue(result["all_centroid_pair_distances_increasing"])
        self.assertTrue(result["all_centroid_pairs_beyond_bond_cutoff"])
        for row in result["centroid_pairs"]:
            self.assertAlmostEqual(row["separation_derivative_bohr_per_atomic_time"],
                                   row["centroid_distance_bohr"]/2)
        self.assertEqual(case.rows[10]["status"], "unresolved")
        self.assertFalse(result["status_changed"])

    def test_audit_sem_is_of_paired_means_and_failures_are_counted(self):
        a, b = self.case("direct-none"), self.case("switch200-none")
        for case, values in ((a, {10: 100., 11: 200.}), (b, {11: 203., 10: 105.})):
            case.rows = {i: {"id": i, "status": "prepared", "preparation_study": {
                "time_averages": {"hold": {"values": {"bond_spread_bohr": value}}}}}
                for i, value in values.items()}
        row = paired_audit_means(a, b, "check")[0]
        self.assertEqual(row["successful_pairs"], 2)
        self.assertEqual(row["mean_difference"], 4.)
        self.assertAlmostEqual(row["sem_difference"], 1.)
        b.rows[10]["status"] = "failed"
        row = paired_audit_means(a, b, "check")[0]
        self.assertEqual(row["pairs_with_a_failure"], 1)
        self.assertEqual(row["successful_pairs"], 1)
        self.assertEqual(row["mean_difference"], 3.)
        self.assertIsNone(row["sem_difference"])


if __name__ == "__main__":
    unittest.main()
