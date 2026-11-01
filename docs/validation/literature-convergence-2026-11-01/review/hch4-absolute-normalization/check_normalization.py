"""Deterministic normalization algebra only: no pilot data, PES or RNG.

This program never imports a PyRPMD or benchmark module. The only scientific
library operations are reading CODATA constants and one-dimensional Gaussian
quadrature. The result includes a known thermal prefactor, never a pilot rate.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import scipy
from scipy.constants import physical_constants, k as BOLTZMANN_SI
from scipy.integrate import quad

ROOT = Path('/workspace/scratch/a5e4820daedd/PyRPMD')
OUT = Path(__file__).resolve().parent
PROPOSAL = ROOT / 'docs/validation/literature-convergence-2026-11-01/review/hch4-reference-feasibility/pilot-design/proposal.json'
INPUTS = [
    PROPOSAL,
    ROOT / 'docs/validation/literature-convergence-2026-11-01/review/hch4-reference-efficiency-design/proposal.json',
    ROOT / 'benchmarks/hch4/reference_pilot.py',
    ROOT / 'benchmarks/hch4/efficiency_pilot.py',
    ROOT / 'benchmarks/hch4/reproduction_protocol.json',
    ROOT / 'benchmarks/hch4/reproduction_input.py',
    ROOT / 'benchmarks/hch4/geometries.json',
    ROOT / 'src/pyrpmd/cv.py',
    ROOT / 'src/pyrpmd/bias.py',
    ROOT / 'src/pyrpmd/constants.py',
    ROOT / 'src/pyrpmd/workflows/rate.py',
]


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mul(scale, vector):
    return [scale * x for x in vector]


def unit(vector):
    return mul(1 / math.sqrt(dot(vector, vector)), vector)


def main():
    wall, cpu = time.monotonic(), time.process_time()
    before = [identity(path) for path in INPUTS]
    proposal = json.loads(PROPOSAL.read_text())
    physics = proposal['physics']
    declared = physics['coordinate_constants']
    temperature = physics['temperature_K']
    length = physics['Rinf_bohr']
    amu = physical_constants['atomic mass constant'][0]
    electron_mass = physical_constants['electron mass'][0]
    hartree = physical_constants['Hartree energy'][0]
    bohr = physical_constants['Bohr radius'][0]
    atomic_time = physical_constants['atomic unit of time'][0]
    hydrogen = physics['masses_amu'][5] * amu / electron_mass
    carbon = physics['masses_amu'][1] * amu / electron_mass
    methane = carbon + 4 * hydrogen
    reduced = hydrogen * methane / (hydrogen + methane)
    beta = hartree / (BOLTZMANN_SI * temperature)
    positive_speed = 1 / math.sqrt(2 * math.pi * beta * reduced)
    prefactor_au = 4 * math.pi * length**2 * positive_speed
    conversion = bohr**3 / atomic_time * 1e6
    prefactor_cm3 = prefactor_au * conversion
    reduced_kg = ((physics['masses_amu'][5] * amu)
                  * (sum(physics['masses_amu'][:5]) * amu)
                  / (sum(physics['masses_amu']) * amu))
    si_prefactor_cm3 = (4 * math.pi * (length * bohr)**2
                       * math.sqrt(BOLTZMANN_SI * temperature / (2 * math.pi * reduced_kg)) * 1e6)
    assert math.isclose(reduced, declared['relative_reduced_mass_electron_mass'], rel_tol=1e-14)
    assert math.isclose(beta, declared['beta_hartree_inverse'], rel_tol=1e-14)
    assert math.isclose(prefactor_au, declared['reactant_flux_prefactor_au'], rel_tol=1e-14)
    assert math.isclose(prefactor_cm3, declared['reactant_flux_prefactor_cm3_molecule_s'], rel_tol=1e-14)
    # Independently tabulated CODATA atomic constants can have rounded last digits.
    assert math.isclose(si_prefactor_cm3, prefactor_cm3, rel_tol=1e-9)
    one_sided_flux, one_error = quad(lambda v: v * math.exp(-v * v / 2) / math.sqrt(2 * math.pi),
                                     0, math.inf, epsabs=1e-13, epsrel=1e-13)
    radial_mean, radial_error = quad(lambda v: math.sqrt(2 / math.pi) * v**3 * math.exp(-v * v / 2),
                                     0, math.inf, epsabs=1e-13, epsrel=1e-13)
    capture_ratio = 4 * one_sided_flux / radial_mean
    assert math.isclose(one_sided_flux, 1 / math.sqrt(2 * math.pi), rel_tol=1e-13)
    assert math.isclose(radial_mean, math.sqrt(8 / math.pi), rel_tol=1e-13)
    assert math.isclose(capture_ratio, 1., rel_tol=1e-13)
    # Unit-vector algebra, not molecular geometries or potential evaluations.
    vector_triples = [([0, 0, 1], [0, 0, 1], [0, 0, 1]),
                      ([0, 0, 1], [0, 0, -1], [1, 0, 0]),
                      ([1, 2, 3], [-2, 3, 1], [3, -1, 2])]
    metrics = []
    masses = [hydrogen, carbon, hydrogen, hydrogen, hydrogen, hydrogen]
    for triple in vector_triples:
        ub, u, uR = map(unit, triple)
        # Carbon index1; a tagged bound H index0; free H index5.
        g0 = [mul(mass / methane, uR) for mass in masses[:5]] + [mul(-1, uR)]
        g1 = [[0., 0., 0.] for _ in masses]
        g1[0] = [x + y for x, y in zip(ub, u)]
        g1[1], g1[5] = mul(-1, ub), mul(-1, u)
        for alpha in (0., .999, 1., 1.007):
            gradients = [[(1 - alpha) * a + alpha * b for a, b in zip(first, second)]
                         for first, second in zip(g0, g1)]
            cartesian = sum(dot(gradient, gradient) / mass for gradient, mass in zip(gradients, masses))
            analytic = ((1 - alpha)**2 / reduced
                        + alpha**2 * ((3 + 2 * dot(ub, u)) / hydrogen + 1 / carbon)
                        + 2 * alpha * (1 - alpha) * dot(uR, u) / reduced)
            summed = [sum(row[j] for row in gradients) for j in range(3)]
            com_conditioned = cartesian - dot(summed, summed) / sum(masses)
            assert math.isclose(cartesian, analytic, rel_tol=2e-14)
            assert math.isclose(com_conditioned, cartesian, rel_tol=2e-14)
            metrics.append({'alpha': alpha, 'unit_vectors_ub_u_uR': [ub, u, uR],
                            'full_Cartesian_metric': cartesian, 'closed_form_metric': analytic,
                            'COM_conditioned_metric': com_conditioned,
                            'translation_gradient_sum': summed})
    after = [identity(path) for path in INPUTS]
    assert before == after
    result = {
        'schema': 'pyrpmd.hch4.absolute-normalization-algebra-check.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'scope': 'Known thermal prefactor and algebra only; no physical rate estimate or pilot-data conversion.',
        'python': sys.version, 'scipy': scipy.__version__, 'inputs_unchanged': True, 'inputs': before,
        'constants': {'atomic_mass_kg': amu, 'electron_mass_kg': electron_mass,
                      'hartree_J': hartree, 'bohr_m': bohr, 'atomic_time_s': atomic_time,
                      'Boltzmann_J_K': BOLTZMANN_SI},
        'fixed_parameters': {'temperature_K': temperature, 'Rinf_bohr': length,
                             'hydrogen_electron_masses': hydrogen, 'carbon_electron_masses': carbon,
                             'methane_electron_masses': methane, 'reduced_mass_electron_masses': reduced,
                             'beta_hartree_inverse': beta},
        'entry_prefactor': {'positive_inward_speed_atomic_units': positive_speed,
                            'bohr3_per_atomic_time': prefactor_au,
                            'atomic_rate_unit_cm3_per_s': conversion,
                            'cm3_per_molecule_s': prefactor_cm3,
                            'independent_SI_cm3_per_molecule_s': si_prefactor_cm3,
                            'SI_vs_atomic_relative_difference': si_prefactor_cm3 / prefactor_cm3 - 1},
        'Gaussian_oracles': {'dimensionless_positive_flux': one_sided_flux,
                             'dimensionless_mean_3D_speed': radial_mean,
                             'quadrature_error_estimates': [one_error, radial_error],
                             'surface_area_flux_to_cross_section_speed_ratio': capture_ratio},
        'metric_checks': metrics, 'new_PES_calls': 0, 'molecular_geometries_evaluated': 0,
        'RNG_draws': 0, 'pilot_data_files_read': 0, 'converted_pilot_rates': 0,
        'elapsed_wall_seconds_after_imports': time.monotonic() - wall,
        'CPU_seconds_after_imports': time.process_time() - cpu,
    }
    (OUT / 'algebra-checks.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passed': True, 'entry_prefactor_cm3_per_molecule_s': prefactor_cm3,
                      'metric_checks': len(metrics), 'PES_calls': 0, 'RNG_draws': 0,
                      'pilot_data_files_read': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
