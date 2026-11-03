"""Primitive WKB ZCT/SCT transmission and assembled VTST results."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from .constants import AMU_TO_AU, KB_AU
from .vtst_core import VTSTError, _finite_array, _jsonable, _positive, _positive_integer
from .vtst_path import MEPPath
from .vtst_thermo import (
    PathModeAnalysis, PartitionSpecies, _rate_record, analyze_path_modes,
    canonical_variational_profile,
)


def _turning_points(
    s: np.ndarray,
    potential: PchipInterpolator,
    values: np.ndarray,
    energy: float,
    maximum_index: int,
) -> tuple[float, float] | None:
    difference = values - energy
    left = None
    for index in range(maximum_index - 1, -1, -1):
        if difference[index] <= 0.0 <= difference[index + 1]:
            left = float(s[index]) if difference[index] == 0.0 else brentq(
                lambda coordinate: float(potential(coordinate) - energy),
                float(s[index]),
                float(s[index + 1]),
            )
            break
    right = None
    for index in range(maximum_index, len(s) - 1):
        if difference[index] >= 0.0 >= difference[index + 1]:
            right = float(s[index + 1]) if difference[index + 1] == 0.0 else brentq(
                lambda coordinate: float(potential(coordinate) - energy),
                float(s[index]),
                float(s[index + 1]),
            )
            break
    if left is None or right is None or right <= left:
        return None
    return left, right


def wkb_transmission_probability(
    energy_hartree: float,
    s_bohr: Sequence[float],
    adiabatic_potential_hartree: Sequence[float],
    *,
    effective_mass_ratio: Sequence[float] | None = None,
    scaling_mass: float = AMU_TO_AU,
) -> dict[str, float]:
    """Primitive WKB transmission through the barrier containing its maximum."""
    energy = float(energy_hartree)
    if not math.isfinite(energy):
        raise VTSTError("energy_hartree must be finite")
    s = _finite_array(s_bohr, None, "s_bohr")
    potential_values = _finite_array(
        adiabatic_potential_hartree, (len(s),), "adiabatic_potential_hartree"
    )
    if s.ndim != 1 or len(s) < 3 or not np.all(np.diff(s) > 0):
        raise VTSTError("s_bohr must be a strictly increasing array with at least three points")
    ratio = (
        np.ones(len(s), dtype=float)
        if effective_mass_ratio is None
        else _finite_array(effective_mass_ratio, (len(s),), "effective_mass_ratio")
    )
    if np.any(ratio <= 0.0) or np.any(ratio > 1.0 + 1.0e-12):
        raise VTSTError("effective_mass_ratio must lie in (0,1]")
    mu = _positive(scaling_mass, "scaling_mass")
    maximum_index = int(np.argmax(potential_values))
    maximum = float(potential_values[maximum_index])
    threshold = max(float(potential_values[0]), float(potential_values[-1]))
    if energy < threshold:
        return {"probability": 0.0, "action": math.inf}
    if energy >= maximum:
        return {"probability": 1.0, "action": 0.0}
    potential = PchipInterpolator(s, potential_values, extrapolate=False)
    mass_ratio = PchipInterpolator(s, ratio, extrapolate=False)
    points = _turning_points(s, potential, potential_values, energy, maximum_index)
    if points is None:
        return {"probability": 0.0, "action": math.inf}
    left, right = points

    def integrand(coordinate: float) -> float:
        barrier = max(float(potential(coordinate)) - energy, 0.0)
        local_ratio = min(max(float(mass_ratio(coordinate)), 1.0e-15), 1.0)
        return math.sqrt(2.0 * mu * local_ratio * barrier)

    # Integrate each monotone PCHIP interval separately.  Gauss-Legendre
    # nodes exclude the square-root turning points, and splitting at every
    # interpolation knot gives a stable deterministic error estimate even
    # when the barrier is sampled on a nonuniform MEP grid.
    internal = s[(s > left) & (s < right)]
    breaks = np.concatenate(([left], internal, [right]))

    def integrate(order: int) -> float:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        total = 0.0
        for lower, upper in zip(breaks[:-1], breaks[1:]):
            midpoint = 0.5 * (lower + upper)
            half_width = 0.5 * (upper - lower)
            coordinates = midpoint + half_width * nodes
            values = np.array([integrand(float(value)) for value in coordinates])
            total += half_width * float(np.dot(weights, values))
        return total

    coarse = integrate(16)
    action = integrate(32)
    error = abs(action - coarse)
    if error > max(1.0e-9, 5.0e-7 * max(action, 1.0e-12)):
        refined = integrate(64)
        error = abs(refined - action)
        action = refined
    if not math.isfinite(action) or error > max(1.0e-8, 2.0e-6 * max(action, 1.0e-12)):
        raise VTSTError("WKB action quadrature did not converge")
    probability = math.exp(-2.0 * action) if action < 400.0 else 0.0
    return {
        "probability": min(max(probability, 0.0), 1.0),
        "action": float(action),
        "turning_point_left_bohr": left,
        "turning_point_right_bohr": right,
        "quadrature_error": float(error),
    }


@dataclass
class TunnellingCorrection:
    method: str
    kappa: float
    log_kappa: float
    threshold_energy_hartree: float
    barrier_energy_hartree: float
    reference_energy_hartree: float
    quadrature_points: int
    effective_mass_ratio_minimum: float

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


def thermal_tunnelling_correction(
    temperature_kelvin: float,
    s_bohr: Sequence[float],
    adiabatic_potential_hartree: Sequence[float],
    reference_index: int,
    *,
    effective_mass_ratio: Sequence[float] | None = None,
    scaling_mass: float = AMU_TO_AU,
    quadrature_points: int = 64,
    method: str = "zct",
) -> TunnellingCorrection:
    """Boltzmann-average primitive WKB transmission for a CVT surface."""
    temperature = _positive(temperature_kelvin, "temperature_kelvin")
    beta = 1.0 / (KB_AU * temperature)
    s = _finite_array(s_bohr, None, "s_bohr")
    values = _finite_array(
        adiabatic_potential_hartree, (len(s),), "adiabatic_potential_hartree"
    )
    if not isinstance(reference_index, (int, np.integer)) or not 0 <= int(reference_index) < len(s):
        raise VTSTError("reference_index is out of range")
    reference_index = int(reference_index)
    ratio = (
        np.ones(len(s), dtype=float)
        if effective_mass_ratio is None
        else _finite_array(effective_mass_ratio, (len(s),), "effective_mass_ratio")
    )
    nodes = _positive_integer(quadrature_points, "quadrature_points", minimum=16)
    threshold = max(float(values[0]), float(values[-1]))
    barrier = float(np.max(values))
    reference = float(values[reference_index])
    if barrier <= threshold:
        raise VTSTError(
            "vibrationally adiabatic path has no barrier above both channel thresholds"
        )
    x, weight = np.polynomial.legendre.leggauss(nodes)
    energies = threshold + 0.5 * (x + 1.0) * (barrier - threshold)
    probabilities = np.array(
        [
            wkb_transmission_probability(
                energy,
                s,
                values,
                effective_mass_ratio=ratio,
                scaling_mass=scaling_mass,
            )["probability"]
            for energy in energies
        ]
    )
    shifted = np.exp(-beta * (energies - threshold))
    integral_below = 0.5 * (barrier - threshold) * float(
        np.dot(weight, probabilities * shifted)
    )
    # Primitive WKB takes P(E)=1 above the barrier; this excludes
    # nonclassical reflection by construction.
    integral_above = math.exp(-beta * (barrier - threshold)) / beta
    scaled_integral = integral_below + integral_above
    log_kappa = (
        math.log(beta)
        + beta * (reference - threshold)
        + math.log(scaled_integral)
    )
    kappa = math.exp(log_kappa) if log_kappa <= math.log(sys.float_info.max) else math.inf
    return TunnellingCorrection(
        method=str(method),
        kappa=float(kappa),
        log_kappa=float(log_kappa),
        threshold_energy_hartree=threshold,
        barrier_energy_hartree=barrier,
        reference_energy_hartree=reference,
        quadrature_points=nodes,
        effective_mass_ratio_minimum=float(np.min(ratio)),
    )


@dataclass
class VTSTResult:
    path: MEPPath
    modes: PathModeAnalysis
    reactants: list[PartitionSpecies]
    temperatures: list[dict[str, Any]]
    conventions: dict[str, Any]
    endpoint_validation: dict[str, Any]
    warnings: list[str]
    physical_converged: bool = False

    def to_dict(self, *, include_hessians: bool = True) -> dict[str, Any]:
        return {
            "schema": "pyrpmd.vtst-result",
            "schema_version": 1,
            "physical_converged": bool(self.physical_converged),
            "conventions": _jsonable(self.conventions),
            "endpoint_validation": _jsonable(self.endpoint_validation),
            "warnings": list(self.warnings),
            "reactants": [item.to_dict() for item in self.reactants],
            "path": self.path.to_dict(include_hessians=include_hessians),
            "mode_analysis": self.modes.to_dict(),
            "temperatures": _jsonable(self.temperatures),
        }


def compute_vtst(
    path: MEPPath,
    masses: Sequence[float],
    reactants: Sequence[PartitionSpecies],
    temperatures_kelvin: Iterable[float],
    *,
    transition_state_symmetry_number: int = 1,
    transition_state_electronic_degeneracy: int = 1,
    reaction_degeneracy: int = 1,
    endpoint_validation: dict[str, Any] | None = None,
    tunnelling_quadrature_points: int = 64,
) -> VTSTResult:
    """Compute TST, CVT, CVT/ZCT, and CVT/SCT on a completed MEP."""
    temperatures = [float(value) for value in temperatures_kelvin]
    if not temperatures:
        raise VTSTError("at least one temperature is required")
    if any(not math.isfinite(value) or value <= 0 for value in temperatures):
        raise VTSTError("temperatures must be positive and finite")
    mass = _finite_array(masses, (path.natoms,), "masses")
    modes = analyze_path_modes(path, mass)
    adiabatic = path.energies_hartree + modes.zero_point_energy_hartree
    records = []
    for temperature in temperatures:
        cvt = canonical_variational_profile(
            path,
            mass,
            modes,
            reactants,
            temperature,
            transition_state_symmetry_number=transition_state_symmetry_number,
            transition_state_electronic_degeneracy=transition_state_electronic_degeneracy,
            reaction_degeneracy=reaction_degeneracy,
        )
        zct = thermal_tunnelling_correction(
            temperature,
            path.s_bohr,
            adiabatic,
            cvt.cvt_index,
            effective_mass_ratio=np.ones(len(path.s_bohr)),
            scaling_mass=path.scaling_mass,
            quadrature_points=tunnelling_quadrature_points,
            method="zct",
        )
        sct = thermal_tunnelling_correction(
            temperature,
            path.s_bohr,
            adiabatic,
            cvt.cvt_index,
            effective_mass_ratio=modes.effective_mass_ratio_sct,
            scaling_mass=path.scaling_mass,
            quadrature_points=tunnelling_quadrature_points,
            method="sct",
        )
        base = float(cvt.cvt["log_rate_au"])
        molecularity = len(reactants)
        records.append(
            {
                **cvt.to_dict(),
                "zct": zct.to_dict(),
                "sct": sct.to_dict(),
                "cvt_zct": {
                    **_rate_record(base + zct.log_kappa, molecularity),
                    "kappa": zct.kappa,
                    "log_kappa": zct.log_kappa,
                },
                "cvt_sct": {
                    **_rate_record(base + sct.log_kappa, molecularity),
                    "kappa": sct.kappa,
                    "log_kappa": sct.log_kappa,
                },
            }
        )
    warnings = [
        "physical_converged is false until MEP step, path extent, Hessian displacement, and tunnelling quadrature are independently converged",
        "harmonic transverse vibrations and classical rigid rotors are used",
        "ZCT/SCT use primitive ground-state WKB; nonclassical reflection, LCT, and microcanonical OMT are not included",
        "Cartesian projection is used for generalized normal modes",
    ]
    return VTSTResult(
        path=path,
        modes=modes,
        reactants=list(reactants),
        temperatures=records,
        conventions={
            "internal_units": {
                "length": "bohr",
                "energy": "hartree",
                "mass": "electron_mass",
                "time": "atomic_time",
            },
            "mass_scaled_coordinate": "x_i=sqrt(m_i/mu)*q_i",
            "scaling_mass_electron_masses": path.scaling_mass,
            "scaling_mass_amu": path.scaling_mass / AMU_TO_AU,
            "normal_mode_projection": "cartesian_mass_scaled_translation_rotation_and_path_tangent",
            "vibrational_partition": "quantum_harmonic_including_zero_point_energy",
            "rotation_partition": "classical_rigid_rotor",
            "translation_partition": "ideal_gas_number_density_per_bohr3",
            "electronic_partition": "explicit_integer_degeneracy",
            "reaction_degeneracy": int(reaction_degeneracy),
            "transition_state_symmetry_number": int(transition_state_symmetry_number),
            "transition_state_electronic_degeneracy": int(
                transition_state_electronic_degeneracy
            ),
            "tunnelling": "primitive_ground_state_wkb_zct_and_centrifugal_dominant_sct",
        },
        endpoint_validation=endpoint_validation or {"checked": False},
        warnings=warnings,
        physical_converged=False,
    )
