"""Independent classical harmonic-TST scale and endpoint handoff checks.

This deterministic calculation is not an RPMD rate calculation. It uses the
BKMP2 electronic barrier, a rigid linear rotor, classical harmonic vibrations,
and two labeled saddle neighborhoods. It includes neither zero-point energy,
tunneling, anharmonicity nor dynamical recrossing. The comparison with the
classical entry in CPC 184, 833 (2013), Table 2 checks the physical scale and
channel convention; it is not an exact reproduction of that sampled result.

For one labeled saddle, with hbar=1, h=2*pi, and beta=1/(k_B*T),

  k = (1/(2*pi*beta)) * (2*pi*beta/mu)**(3/2)
      * (I_TS/I_H2) * omega_H2/(beta**2 * product(omega_TS_stable))
      * exp(-beta*Delta_V).

This is (k_B*T/h) times the translational partition-function ratio
Q_TS/(Q_H*Q_H2), the linear-rotor ratio, and the vibrational ratio. For labeled
nuclei both rotational symmetry numbers are one: q_rot=2*I/beta. The stable
TS modes are the symmetric stretch and BOTH components of the degenerate bend;
q_vib is the product of 1/(beta*omega). The unstable stretch supplies the flux
factor rather than an oscillator partition function. Five rigid zero modes
(three translations and two rotations) are excluded. The reactant has one
H2 stretch. Counting A-B-C and A-C-B saddle neighborhoods multiplies this
one-channel result by two, once. In the PyRPMD dynamics the two neighborhoods
are already represented by equivalent_paths, so symmetry_factor stays one.

The prefactor is independently calculated in SI units and compared with the
atomic-unit expression in the rate workflow. Endpoint and legacy explicit-TS
jobs pass through the actual run preparation function; their reaction
coordinates, derivatives, mass metrics, channels and entrance prefactors are
then compared on the same geometries. No dynamics or statistical convergence
claim is made by these checks.

Build the pinned PES with ``python benchmarks/h3/build_pes.py`` and run::

    PYTHONPATH=src python benchmarks/transition_state_search/h3_rate_sanity.py \
        --output h3-rate-sanity.json
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import scipy
from scipy.constants import k as BOLTZMANN_SI, physical_constants

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.pes import BKMP2Potential, DEFAULT_LIBRARY
from benchmarks.transition_state_search.h3 import (
    MASS_AMU, MASS_AU, endpoints, independent_reference, job,
)
from pyrpmd.constants import AU_RATE_TO_CM3_S, KB_AU, WAVENUMBER_TO_AU
from pyrpmd.input import JobSpec
from pyrpmd.io import jsonable
from pyrpmd.workflows.prepare import numerical_hessian
from pyrpmd.workflows.rate import _prepare, rate_coordinate


PAPER = {
    "citation": (
        "Yu. V. Suleimanov, J. W. Allen, W. H. Green, RPMDrate: Bimolecular "
        "chemical reaction rates from ring polymer molecular dynamics, "
        "Computer Physics Communications 184, 833-840 (2013)"
    ),
    "doi": "10.1016/j.cpc.2012.10.017",
    "repository_url": "https://dspace.mit.edu/entities/publication/a4555c62-26a0-4500-8953-5dd1905583cc",
    "author_full_text_url": "https://www.researchgate.net/publication/279840192_RPMDrate_Bimolecular_chemical_reaction_rates_from_ring_polymer_molecular_dynamics",
    "locations": "Eq. (12) entrance prefactor; Table 2 classical H+H2 row; Figure 1 two channels",
    "temperature_K": 300.0,
    "classical_tst_cm3_molecule_s": 1.56e-18,
    "classical_kappa": 0.994,
    "classical_rate_cm3_molecule_s": 1.55e-18,
    "comparison_target": "Classical TST entry, not the quantum RPMD entry or recrossing-corrected rate",
}


def entrance_prefactor(beta, reduced_mass, rinf):
    """The analytic prefactor used by the rate workflow, in atomic units."""
    return 4 * np.pi * rinf**2 / np.sqrt(2 * np.pi * beta * reduced_mass)


def harmonic_tst(potential, reference):
    geometry = np.asarray(reference["saddle_geometry_bohr"])
    step = 3e-5
    eigenvalues = np.linalg.eigvalsh(numerical_hessian(potential, geometry, step) / MASS_AU)
    threshold = 1e-8
    positive = eigenvalues[eigenvalues > threshold]
    zero = eigenvalues[np.abs(eigenvalues) <= threshold]
    negative = eigenvalues[eigenvalues < -threshold]
    assert len(positive) == 3 and len(zero) == 5 and len(negative) == 1
    stable = np.sqrt(positive)
    # At this collinear equal-mass saddle the two smallest stable eigenvalues
    # are the degenerate bend. Both must contribute to the classical partition.
    assert np.isclose(stable[0], stable[1], rtol=1e-9)
    h2 = reference["h2_equilibrium_distance_bohr"]
    gradient_plus = potential.radial([h2 + step, 99.0, 99.0])[1][0]
    gradient_minus = potential.radial([h2 - step, 99.0, 99.0])[1][0]
    curvature_h2 = (gradient_plus - gradient_minus) / (2 * step)
    omega_h2 = np.sqrt(curvature_h2 / (MASS_AU / 2))
    inertia_ts = MASS_AU * np.sum(geometry[:, 0]**2)
    inertia_h2 = MASS_AU * h2**2 / 2
    reduced_mass = 2 * MASS_AU / 3
    rates = []
    for temperature in (300.0, 1000.0, 1500.0):
        beta = 1 / (KB_AU * temperature)
        translation = (2 * np.pi * beta / reduced_mass)**1.5
        rotation = inertia_ts / inertia_h2
        vibration = omega_h2 / (beta**2 * np.prod(stable))
        flux = 1 / (2 * np.pi * beta)
        boltzmann = np.exp(-beta * reference["electronic_barrier_hartree"])
        one_saddle = flux * translation * rotation * vibration * boltzmann
        total = 2 * one_saddle * AU_RATE_TO_CM3_S
        prefactor = entrance_prefactor(beta, reduced_mass, 30.0) * AU_RATE_TO_CM3_S
        # Independent dimensional expression using meters, kilograms, seconds.
        bohr_m = physical_constants["Bohr radius"][0]
        mass_kg = MASS_AMU * physical_constants["atomic mass constant"][0]
        prefactor_si = (4 * np.pi * (30 * bohr_m)**2
                        * np.sqrt(BOLTZMANN_SI * temperature / (2 * np.pi * (2 * mass_kg / 3)))
                        * 1e6)
        assert np.isclose(prefactor, prefactor_si, rtol=1e-10, atol=0)
        rates.append({
            "temperature_K": temperature,
            "beta_hartree_inverse": beta,
            "translational_partition_ratio_bohr3": translation,
            "rotational_partition_ratio": rotation,
            "vibrational_partition_ratio": vibration,
            "flux_factor_inverse_atomic_time": flux,
            "electronic_barrier_boltzmann_factor": boltzmann,
            "one_labeled_saddle_rate_cm3_molecule_s": one_saddle * AU_RATE_TO_CM3_S,
            "two_labeled_saddles_rate_cm3_molecule_s": total,
            "entrance_prefactor_r30_cm3_molecule_s": prefactor,
            "entrance_prefactor_independent_si_cm3_molecule_s": prefactor_si,
            "prefactor_atomic_vs_si_relative_error": abs(prefactor / prefactor_si - 1),
            "implied_flux_pmf_barrier_r30_hartree": -np.log(total / prefactor) / beta,
        })
    comparison = {
        "harmonic_to_published_classical_tst_ratio": (
            rates[0]["two_labeled_saddles_rate_cm3_molecule_s"] / PAPER["classical_tst_cm3_molecule_s"]
        ),
        "exact_reproduction_claimed": False,
        "interpretation": (
            "A separate classical rigid-rotor/harmonic calculation checks the rate scale. "
            "Its differences from the sampled anharmonic TST are not a numerical error estimate. "
            "It cannot validate tunneling, bead convergence, recrossing or the quantum rate."
        ),
    }
    return {
        "formula_atomic_units": "2/(2*pi*beta) * (2*pi*beta/mu)**1.5 * (I_TS/I_H2) * omega_H2/(beta**2*product(omega_TS_stable)) * exp(-beta*Delta_V)",
        "partition_function_conventions": {
            "nuclei": "labeled, distinguishable, all with the hydrogen-1 atomic mass",
            "linear_rotor_symmetry_number_reactant": 1,
            "linear_rotor_symmetry_number_per_labeled_saddle": 1,
            "linear_rotor_partition_function": "2*I/beta in atomic units",
            "classical_oscillator_partition_function": "1/(beta*omega), with angular frequency omega in atomic units",
            "saddle_neighborhoods": 2,
            "saddle_neighborhood_labels": ["A-B-C", "A-C-B"],
            "electronic_statistical_weight_ratio": 1.0,
            "zero_point_energy_included": False,
            "tunneling_included": False,
            "anharmonicity_included": False,
            "recrossing_included": False,
            "nuclear_spin_weights_included": False,
            "implementation_symmetry_factor": 1.0,
            "implementation_note": "equivalent_paths already includes both saddle neighborhoods; do not multiply the sampled rate by two again",
        },
        "isotope_mass_amu": MASS_AMU,
        "isotope_mass_electron_masses": MASS_AU,
        "entrance_reduced_mass_electron_masses": reduced_mass,
        "hessian_step_bohr": step,
        "resolved_eigenvalue_threshold_au": threshold,
        "all_mass_weighted_eigenvalues_au": eigenvalues.tolist(),
        "excluded_rigid_mode_count": len(zero),
        "excluded_unstable_mode_count": len(negative),
        "stable_mode_count": len(stable),
        "stable_mode_labels": ["bend component 1", "bend component 2", "symmetric stretch"],
        "stable_mode_wavenumbers_cm_minus1": (stable / WAVENUMBER_TO_AU).tolist(),
        "reactant_h2_stretch_wavenumber_cm_minus1": omega_h2 / WAVENUMBER_TO_AU,
        "saddle_moment_of_inertia_electron_mass_bohr2": inertia_ts,
        "reactant_h2_moment_of_inertia_electron_mass_bohr2": inertia_h2,
        "electronic_barrier_hartree": reference["electronic_barrier_hartree"],
        "rate_atomic_unit_to_cm3_molecule_s": AU_RATE_TO_CM3_S,
        "rates": rates,
        "published_classical_tst_comparison": comparison,
    }


def compare_handoff(reference, library):
    reactant, product = endpoints(reference["h2_equilibrium_distance_bohr"])
    data = job(reactant, product, library, {
        "gradient_tolerance": 1e-7, "neb": {"gradient_tolerance": 1e-5, "images": 7},
    })
    data["temperature"] = "300 K"
    data["reaction"].update(rinf="30 bohr", equivalent_paths=[[0, 2, 1]],
                            electronic_weight=1.0, symmetry_factor=1.0)
    data["umbrella"] = {"start": -0.05, "stop": 1.05, "count": 111, "kappa": "30 hartree"}
    config, initial_system = JobSpec.from_mapping(data).resolve()
    legacy = copy.deepcopy(data)
    legacy["system"].pop("reactant")
    legacy["system"].pop("product")
    # The pre-existing published-rate input, independently of the endpoint run.
    legacy["system"]["geometry"] = [[-1.7570000251, 0, 0], [0, 0, 0], [1.7570000251, 0, 0]]
    legacy["reaction"].pop("preparation")
    legacy["reaction"].update(fragments=[[0], [1, 2]], forming_bonds=[[0, 1]], breaking_bonds=[[1, 2]])
    legacy_config, legacy_system = JobSpec.from_mapping(legacy).resolve()
    with tempfile.TemporaryDirectory(prefix="pyrpmd-rate-sanity-", dir=Path(library).parent) as temporary:
        endpoint_path = Path(temporary) / "endpoint"
        explicit_path = Path(temporary) / "explicit"
        endpoint_path.mkdir()
        explicit_path.mkdir()
        endpoint_config, endpoint_system = _prepare(config, initial_system, endpoint_path, True)
        explicit_config, explicit_system = _prepare(legacy_config, legacy_system, explicit_path, True)
        saved = json.loads((endpoint_path / "preparation.json").read_text())
    assert saved["ready"]
    assert np.array_equal(endpoint_system.geometry, np.asarray(saved["transition_state"]))
    endpoint_cv = rate_coordinate(endpoint_config, endpoint_system)
    explicit_cv = rate_coordinate(explicit_config, explicit_system)
    independent = np.asarray(reference["saddle_geometry_bohr"])
    asymptote = np.array([[-30.0, 0, 0], [-reference["h2_equilibrium_distance_bohr"] / 2, 0, 0],
                         [reference["h2_equilibrium_distance_bohr"] / 2, 0, 0]])
    rng = np.random.default_rng(274611)
    geometries = [independent, independent[[0, 2, 1]], reactant, product, asymptote]
    for base in (independent, reactant, product, asymptote):
        geometries.extend(base + rng.normal(scale=0.15, size=(32, 3, 3)))
    errors = {"value": 0.0, "gradient": 0.0, "hessian": 0.0, "mass_metric": 0.0,
              "equivalent_channel_value": 0.0, "equivalent_channel_gradient": 0.0}
    permutation = [0, 2, 1]
    for geometry in geometries:
        value, gradient, hessian = endpoint_cv.centroid_value_gradient_hessian(geometry)
        other_value, other_gradient, other_hessian = explicit_cv.centroid_value_gradient_hessian(geometry)
        metric = np.sum(gradient**2 / endpoint_system.masses[:, None])
        other_metric = np.sum(other_gradient**2 / explicit_system.masses[:, None])
        errors["value"] = max(errors["value"], abs(value - other_value))
        errors["gradient"] = max(errors["gradient"], np.max(np.abs(gradient - other_gradient)))
        errors["hessian"] = max(errors["hessian"], np.max(np.abs(hessian - other_hessian)))
        errors["mass_metric"] = max(errors["mass_metric"], abs(metric - other_metric))
        permuted_value, permuted_gradient = endpoint_cv.centroid_value_gradient(geometry[permutation])
        errors["equivalent_channel_value"] = max(errors["equivalent_channel_value"], abs(value - permuted_value))
        errors["equivalent_channel_gradient"] = max(errors["equivalent_channel_gradient"],
                                                    np.max(np.abs(gradient[permutation] - permuted_gradient)))
    fields = ("fragments", "forming_bonds", "breaking_bonds", "equivalent_paths", "rinf",
              "electronic_weight", "symmetry_factor")
    equal_reaction = all(endpoint_config["reaction"][key] == explicit_config["reaction"][key] for key in fields)
    assert equal_reaction and np.array_equal(endpoint_system.masses, explicit_system.masses)
    assert all(value < 1e-10 for value in errors.values())
    q_endpoint = endpoint_system.geometry - endpoint_system.geometry.mean(0)
    geometry_error = np.max(np.abs(q_endpoint - independent))
    assert geometry_error < 1e-8
    def prefactor(prepared, system):
        ma, mb = [system.masses[fragment].sum() for fragment in prepared["reaction"]["fragments"]]
        return entrance_prefactor(prepared["beta"], ma * mb / (ma + mb), prepared["reaction"]["rinf"])
    endpoint_prefactor = prefactor(endpoint_config, endpoint_system)
    explicit_prefactor = prefactor(explicit_config, explicit_system)
    assert endpoint_prefactor == explicit_prefactor
    saddle_values = [endpoint_cv.centroid_value_gradient(independent[p])[0]
                     for p in ([0, 1, 2], permutation)]
    assert np.allclose(saddle_values, 1.0, rtol=0, atol=1e-10)
    return {
        "scope": "Actual run preparation and deterministic rate-coordinate/prefactor checks; no dynamics",
        "passed": True,
        "temperature_K": 300.0,
        "beads": 1,
        "random_geometry_seed": 274611,
        "test_geometry_count": len(geometries),
        "endpoint_preparation_ready": saved["ready"],
        "endpoint_geometry_bohr": endpoint_system.geometry.tolist(),
        "explicit_legacy_geometry_bohr": explicit_system.geometry.tolist(),
        "endpoint_geometry_max_abs_difference_from_independent_reference_after_centering_bohr": geometry_error,
        "saved_transition_state_exactly_matches_sampling_system": True,
        "reaction_and_channel_settings_equal": equal_reaction,
        "reaction": {key: endpoint_config["reaction"][key] for key in fields},
        "reference_bond_differences_endpoint_bohr": endpoint_cv.reference_difference,
        "reference_bond_differences_explicit_bohr": explicit_cv.reference_difference,
        "channel_paths": endpoint_cv.paths,
        "coordinate_values_at_two_equivalent_saddles": saddle_values,
        "coordinate_value_at_r30_reference": endpoint_cv.centroid_value_gradient(asymptote)[0],
        "max_absolute_errors": errors,
        "masses_exactly_equal": True,
        "prefactor_endpoint_atomic_units": endpoint_prefactor,
        "prefactor_explicit_atomic_units": explicit_prefactor,
        "prefactors_exactly_equal": True,
        "interpretation": (
            "The endpoint run selects the same labeled entrance channel and both equivalent product "
            "arrangements as the historical explicit-TS input. For symmetric H3 the reference bond "
            "difference is zero, so a common change in the two saddle bond lengths does not change xi. "
            "These checks exclude a new endpoint handoff, coordinate, channel or entrance-prefactor "
            "difference at the tested geometries; they do not validate statistical sampling."
        ),
    }


def validate(library):
    potential = BKMP2Potential(library)
    try:
        reference = independent_reference(potential)
        harmonic = harmonic_tst(potential, reference)
        metadata = dict(potential.metadata)
    finally:
        potential.close()
    handoff = compare_handoff(reference, library)
    source_files = (
        "benchmarks/transition_state_search/h3_rate_sanity.py", "benchmarks/transition_state_search/h3.py",
        "benchmarks/h3/pes.py", "src/pyrpmd/cv.py", "src/pyrpmd/constants.py",
        "src/pyrpmd/workflows/rate.py", "src/pyrpmd/workflows/prepare.py",
        "src/pyrpmd/workflows/neb.py", "src/pyrpmd/workflows/saddle.py",
    )
    return {
        "schema": "pyrpmd.h3-rate-sanity.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
        "pes_provenance": metadata,
        "published_comparator": PAPER,
        "reference_geometry_method": reference["method"],
        "harmonic_tst": harmonic,
        "endpoint_vs_explicit_handoff": handoff,
        "physical_rpmd_validation": False,
        "limitations": [
            "The harmonic partition functions approximate the classical rate; they do not reproduce an exact anharmonic PMF.",
            "The two-channel factor is used once in this independent formula; the production coordinate already includes those channels.",
            "No quantum vibrational populations, zero-point energies, tunneling or nuclear-spin restrictions are included.",
            "No recrossing trajectories, statistical uncertainty, bead convergence or time-step convergence are tested here.",
            "Agreement in scale argues against a simple units/channel/prefactor explanation for the old quantum pilot discrepancy; it does not prove every discrepancy is sampling error.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.library.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    first = result["harmonic_tst"]["rates"][0]
    print(json.dumps({
        "output": str(args.output),
        "classical_harmonic_tst_300K_cm3_molecule_s": first["two_labeled_saddles_rate_cm3_molecule_s"],
        "published_classical_tst_300K_cm3_molecule_s": PAPER["classical_tst_cm3_molecule_s"],
        "endpoint_handoff_checks_passed": result["endpoint_vs_explicit_handoff"]["passed"],
        "physical_rpmd_validation": False,
    }))


if __name__ == "__main__":
    main()
