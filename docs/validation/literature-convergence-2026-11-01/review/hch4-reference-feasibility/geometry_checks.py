#!/usr/bin/env python3
"""Geometry-only checks for an UNIMPLEMENTED H+CH4 reference estimator.

No potential is loaded, no partition function is estimated and no dynamics run.
The frozen production CV is read/imported only as an independent comparison.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, AU_RATE_TO_CM3_S, KB_AU
from pyrpmd.cv import RPMDrateCoordinate

INPUTS = (
    "benchmarks/hch4/reproduction_protocol.json",
    "benchmarks/hch4/geometries.json",
    "benchmarks/hch4/source.json",
    "benchmarks/hch4/reproduction_input.py",
    "benchmarks/hch4/pes.py",
    "src/pyrpmd/cv.py",
    "src/pyrpmd/constants.py",
    "src/pyrpmd/workflows/rate.py",
    "src/pyrpmd/bias.py",
)


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}


def rotation(phi, theta, psi):
    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    return rz(phi) @ ry @ rz(psi)


def body_map(z):
    r, s, v = z[:3]
    body = np.array([[0, 0, r], [s * np.sqrt(1-v*v), 0, s*v], z[3:6], z[6:9]])
    return (body @ rotation(*z[9:]).T).ravel()


def main():
    before = hashes()
    protocol = json.loads((ROOT / INPUTS[0]).read_text())
    geometry = json.loads((ROOT / INPUTS[1]).read_text())
    physics = protocol["planned_physics"]
    qref = np.asarray(geometry["official_input"]["coordinates"]) * ANGSTROM_TO_AU
    masses = np.asarray(physics["masses_amu"]) * AMU_TO_AU
    m, carbon = masses[0], masses[1]
    methane_mass = masses[:5].sum()
    mu = m * methane_mass / (m + methane_mass)
    beta, L = 1 / (KB_AU * 300), 30.0
    c = np.linalg.norm(qref[5]-qref[3]) - np.linalg.norm(qref[3]-qref[1])
    cv = RPMDrateCoordinate(
        physics["fragments_zero_based"], masses, qref,
        physics["forming_bonds_zero_based"], physics["breaking_bonds_zero_based"], L,
        equivalent_paths=physics["equivalent_path_permutations"], estimator="centroid",
    )
    hydrogens = [0, 2, 3, 4]

    def entry(q):
        com = np.average(q[:5], axis=0, weights=masses[:5])
        vec = q[5]-com
        return L-np.linalg.norm(vec), vec/np.linalg.norm(vec)

    def channels(q):
        return np.array([c+np.linalg.norm(q[h]-q[1])-np.linalg.norm(q[5]-q[h])
                         for h in hydrogens])

    def branch(q, h, alpha):
        return (1-alpha)*entry(q)[0] + alpha*(c+np.linalg.norm(q[h]-q[1])
                                             - np.linalg.norm(q[5]-q[h]))

    maxima = dict(root_residual_bohr=0., metric_relative_error=0.,
                  cartesian_gradient_absolute_error=0., xi_surface_absolute_error=0.,
                  coarea_gradient_absolute_error=0., rotation_absolute_error=0.,
                  permutation_absolute_error=0., gauge_jacobian_relative_error=0.)
    rng = np.random.default_rng(202611011851)
    total, active, inactive, positive_branch = 0, 0, 0, 0
    minimum_denominator = float("inf")
    minimum_radial_derivative_magnitude = float("inf")
    for alpha in (.999, 1.007):
        for h in hydrogens:
            for _ in range(16):
                q = qref-qref[1]
                q[hydrogens] += rng.normal(scale=.12, size=(4, 3))
                u = rng.normal(size=3); u /= np.linalg.norm(u)
                com = np.average(q[:5], axis=0, weights=masses[:5])
                v = q[h]-com
                r = np.linalg.norm(q[h]-q[1])
                def radial(rho):
                    return (1-alpha)*(L-np.linalg.norm(v+rho*u)) + alpha*(c+r-rho)
                f0 = radial(0)
                assert f0 > 0
                low, high = 0., f0/(alpha-abs(1-alpha))+1e-12
                assert radial(high) <= 1e-12
                for _ in range(80):
                    mid = (low+high)/2
                    if radial(mid) > 0:
                        low = mid
                    else:
                        high = mid
                rho = (low+high)/2
                q[5] = q[h]+rho*u
                s0, uR = entry(q)
                values = channels(q)
                D = s0-values.max()
                dS = -alpha-(1-alpha)*np.dot(u, uR)
                minimum_radial_derivative_magnitude = min(minimum_radial_derivative_magnitude, -dS)
                assert dS <= -(alpha-abs(1-alpha))+1e-14
                maxima["root_residual_bohr"] = max(maxima["root_residual_bohr"], abs(branch(q, h, alpha)))
                ub = (q[h]-q[1])/r
                gradient0 = masses[:, None]/methane_mass * uR
                gradient0[5] = -uR
                gradient1 = np.zeros((6, 3))
                gradient1[5] = -u
                gradient1[h] = ub+u
                gradient1[1] = -ub
                analytic = (1-alpha)*gradient0+alpha*gradient1
                finite = np.empty((6, 3))
                step = 1e-5
                for atom in range(6):
                    for axis in range(3):
                        plus, minus = q.copy(), q.copy()
                        plus[atom, axis] += step; minus[atom, axis] -= step
                        finite[atom, axis] = (branch(plus, h, alpha)-branch(minus, h, alpha))/(2*step)
                maxima["cartesian_gradient_absolute_error"] = max(maxima["cartesian_gradient_absolute_error"], float(np.max(np.abs(finite-analytic))))
                metric = ((1-alpha)**2/mu + alpha**2*((3+2*np.dot(ub, u))/m+1/carbon)
                          +2*alpha*(1-alpha)*np.dot(uR, u)/mu)
                from_finite = np.sum(finite*finite/masses[:, None])
                maxima["metric_relative_error"] = max(maxima["metric_relative_error"], abs(from_finite/metric-1))
                total += 1
                if D > 0:
                    positive_branch += 1
                    minimum_denominator = min(minimum_denominator, D)
                if np.argmax(values) != hydrogens.index(h):
                    inactive += 1
                    continue
                active += 1
                xi, grad_xi = cv.centroid_value_gradient(q)
                sval, grad_s = cv.dividing_surface(alpha).centroid_value_gradient(q)
                maxima["xi_surface_absolute_error"] = max(maxima["xi_surface_absolute_error"], abs(xi-alpha), abs(sval))
                maxima["coarea_gradient_absolute_error"] = max(maxima["coarea_gradient_absolute_error"], float(np.max(np.abs(D*grad_xi-grad_s))))
                assert np.allclose(grad_s, analytic, rtol=0, atol=1e-12)
                rot = rotation(.37, .73, -.42)
                transformed = q@rot.T+np.array([.3, -.7, .2])
                maxima["rotation_absolute_error"] = max(maxima["rotation_absolute_error"], abs(cv.centroid_value_gradient(transformed)[0]-xi))
                for permutation in physics["equivalent_path_permutations"]:
                    maxima["permutation_absolute_error"] = max(maxima["permutation_absolute_error"], abs(cv.centroid_value_gradient(q[permutation])[0]-xi))

    for _ in range(12):
        z = np.r_[rng.uniform(1.6, 3, 2), rng.uniform(-.8, .8), rng.normal(size=6),
                  rng.uniform(-2, 2), rng.uniform(.3, 2.8), rng.uniform(-2, 2)]
        jac = np.empty((12, 12))
        step = 1e-5
        for j in range(12):
            plus, minus = z.copy(), z.copy()
            plus[j] += step; minus[j] -= step
            jac[:, j] = (body_map(plus)-body_map(minus))/(2*step)
        expected = z[0]**2*z[1]**2*np.sin(z[10])
        maxima["gauge_jacobian_relative_error"] = max(maxima["gauge_jacobian_relative_error"], abs(abs(np.linalg.det(jac))/expected-1))

    # The near-saddle perturbations may all have a unique active root. A
    # deliberately stretched, non-equilibrium shape exercises exclusion; its
    # existence does not estimate its Boltzmann weight or domain eligibility.
    counterexample = qref-qref[1]
    counterexample[0] = [6., 0., 0.]
    counterexample[2] = [6., 2., 0.]
    low, high = 0., 10.
    for _ in range(80):
        rho = (low+high)/2
        counterexample[5] = counterexample[0]+np.array([0., rho, 0.])
        if branch(counterexample, 0, .999) > 0:
            low = rho
        else:
            high = rho
    own_surface = branch(counterexample, 0, .999)
    other_surface = cv.dividing_surface(.999).centroid_value_gradient(counterexample)[0]
    assert abs(own_surface) < 1e-10 and other_surface > 0.1
    assert total == 128 and active > 0
    assert maxima["root_residual_bohr"] < 1e-10
    assert maxima["metric_relative_error"] < 1e-8
    assert maxima["cartesian_gradient_absolute_error"] < 2e-8
    assert maxima["xi_surface_absolute_error"] < 1e-10
    assert maxima["coarea_gradient_absolute_error"] < 1e-10
    assert maxima["rotation_absolute_error"] < 1e-12
    assert maxima["permutation_absolute_error"] < 1e-12
    assert maxima["gauge_jacobian_relative_error"] < 1e-8
    after = hashes()
    assert after == before
    prefactor = 4*np.pi*L*L/np.sqrt(2*np.pi*beta*mu)
    report = {
        "schema": "pyrpmd.hch4-reference-feasibility.geometry-only.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "geometry_checks_passed_estimator_unimplemented_unvalidated",
        "PES_calls": 0, "dynamics_steps": 0, "rate_estimates": 0,
        "scope": "128 synthetic perturbed-geometry roots and twelve rotation-gauge Jacobians; not equilibrium samples",
        "geometries": {"total": total, "active_channel": active, "inactive_channel": inactive,
                       "positive_xi_denominator": positive_branch},
        "separate_channel_exclusion_counterexample": {
            "classification": "deliberately non-equilibrium, no potential or physical probability evaluated",
            "own_channel_surface_bohr": own_surface,
            "max_channel_surface_bohr": other_surface,
            "coordinates_bohr": counterexample.tolist(),
        },
        "constants": {"beta_hartree_inverse": beta, "masses_electron_mass": masses.tolist(),
                      "relative_reduced_mass_electron_mass": mu, "reference_difference_bohr": c,
                      "reference_breaking_bohr": float(np.linalg.norm(qref[3]-qref[1])),
                      "reference_forming_bohr": float(np.linalg.norm(qref[5]-qref[3])),
                      "reactant_flux_prefactor_au": prefactor,
                      "reactant_flux_prefactor_cm3_molecule_s": prefactor*AU_RATE_TO_CM3_S},
        "minimum_xi_denominator_bohr": minimum_denominator,
        "minimum_radial_derivative_magnitude": minimum_radial_derivative_magnitude,
        "maximum_errors": maxima,
        "source_hashes_before": before, "source_hashes_after": after,
        "source_unchanged": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
