"""Generalized modes and canonical variational partition functions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Sequence

import numpy as np

from .constants import AMU_TO_AU, ATOMIC_TIME_S, BOHR_M, KB_AU
from .vtst_core import (
    VTSTError, _finite_array, _jsonable, _positive, _positive_integer,
    generalized_normal_modes,
)
from .vtst_path import MEPPath

@dataclass
class PathModeAnalysis:
    frequencies_au: list[np.ndarray]
    vectors_scaled: list[np.ndarray]
    curvature_bohr_inverse: np.ndarray
    curvature_components: list[np.ndarray]
    effective_frequency_au: np.ndarray
    transverse_turning_point_bohr: np.ndarray
    effective_mass_ratio_sct: np.ndarray
    zero_point_energy_hartree: np.ndarray
    removed_rigid_modes: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


def analyze_path_modes(
    path: MEPPath,
    masses: Sequence[float],
    *,
    negative_tolerance: float = 1.0e-9,
    zero_tolerance: float = 1.0e-10,
) -> PathModeAnalysis:
    """Calculate Cartesian-projected generalized modes and SCT effective mass."""
    mass = _finite_array(masses, (path.natoms,), "masses")
    frequencies: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    rigid_counts = []
    for geometry, hessian, tangent in zip(
        path.geometries_bohr,
        path.hessians_hartree_per_bohr2,
        path.tangent_scaled,
    ):
        modes = generalized_normal_modes(
            geometry,
            mass,
            hessian,
            tangent_scaled=tangent,
            scaling_mass=path.scaling_mass,
            negative_tolerance=negative_tolerance,
            zero_tolerance=zero_tolerance,
            require_positive=True,
        )
        frequencies.append(modes.frequencies_au)
        vectors.append(modes.vectors_scaled)
        rigid_counts.append(modes.removed_rigid_modes)
    counts = {len(item) for item in frequencies}
    if len(counts) != 1:
        raise VTSTError(
            "the number of transverse modes changes along the path; a continuous SCT model is undefined"
        )
    edge_order = 2 if len(path.s_bohr) >= 3 else 1
    derivative = np.gradient(
        path.tangent_scaled, path.s_bohr, axis=0, edge_order=edge_order
    )
    # Numerically remove the tangent component from dv/ds.
    derivative -= (
        np.sum(derivative * path.tangent_scaled, axis=1)[:, None]
        * path.tangent_scaled
    )
    components: list[np.ndarray] = []
    curvature = np.empty(len(path.s_bohr), dtype=float)
    effective_frequency = np.empty(len(path.s_bohr), dtype=float)
    turning = np.empty(len(path.s_bohr), dtype=float)
    for index, (change, mode_vectors, omega) in enumerate(
        zip(derivative, vectors, frequencies)
    ):
        component = mode_vectors.T @ change
        components.append(component)
        kappa = float(np.linalg.norm(component))
        curvature[index] = kappa
        if omega.size == 0:
            # Removing the rigid modes and reaction tangent can exhaust the
            # vibrational subspace, for example for a diatomic stretch.  This
            # is the exact zero-curvature limit: keep finite diagnostics and
            # leave the SCT effective-mass ratio at one.
            effective_frequency[index] = 0.0
            turning[index] = 0.0
            continue
        if kappa <= 1.0e-14:
            # The limiting value is irrelevant because a = kappa*tbar is zero;
            # retain a finite value for stable differentiation.
            effective_frequency[index] = float(np.exp(np.mean(np.log(omega))))
        else:
            weights = component / kappa
            effective_frequency[index] = math.sqrt(float(np.sum((weights * omega) ** 2)))
        turning[index] = math.sqrt(
            1.0 / (path.scaling_mass * effective_frequency[index])
        )
    turning_derivative = np.gradient(
        turning, path.s_bohr, edge_order=edge_order
    )
    mass_ratio = np.ones(len(path.s_bohr), dtype=float)
    nonzero = curvature > 1.0e-14
    a_bar = np.abs(curvature[nonzero] * turning[nonzero])
    exponent = -2.0 * a_bar - a_bar**2 + turning_derivative[nonzero] ** 2
    mass_ratio[nonzero] = np.exp(np.minimum(exponent, 0.0))
    mass_ratio = np.minimum(mass_ratio, 1.0)
    # At the saddle the normalized-gradient reaction-path direction is
    # undefined because the gradient vanishes.  The centrifugal-dominant SCT
    # prescription therefore takes the linear interpolation from the nearest
    # nonstationary points (Bao & Truhlar 2017, Eq. 85).
    saddle = path.saddle_index
    if 0 < saddle < len(mass_ratio) - 1:
        mass_ratio[saddle] = float(
            np.interp(
                path.s_bohr[saddle],
                path.s_bohr[[saddle - 1, saddle + 1]],
                mass_ratio[[saddle - 1, saddle + 1]],
            )
        )
    zpe = np.array([0.5 * float(np.sum(omega)) for omega in frequencies])
    return PathModeAnalysis(
        frequencies_au=frequencies,
        vectors_scaled=vectors,
        curvature_bohr_inverse=curvature,
        curvature_components=components,
        effective_frequency_au=effective_frequency,
        transverse_turning_point_bohr=turning,
        effective_mass_ratio_sct=mass_ratio,
        zero_point_energy_hartree=zpe,
        removed_rigid_modes=np.asarray(rigid_counts, dtype=int),
    )


def moment_of_inertia(
    geometry_bohr: np.ndarray, masses: Sequence[float]
) -> np.ndarray:
    geometry = _finite_array(geometry_bohr, None, "geometry_bohr")
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise VTSTError("geometry_bohr must have shape (N,3)")
    mass = _finite_array(masses, (len(geometry),), "masses")
    if np.any(mass <= 0):
        raise VTSTError("masses must be positive")
    center = np.sum(mass[:, None] * geometry, axis=0) / float(mass.sum())
    x = geometry - center
    inertia = np.eye(3) * np.sum(mass[:, None] * x * x) - (x.T * mass) @ x
    return 0.5 * (inertia + inertia.T)


def _log_rotation_partition(
    geometry: np.ndarray,
    masses: np.ndarray,
    beta: float,
    symmetry_number: int,
) -> tuple[float, int]:
    symmetry = _positive_integer(symmetry_number, "symmetry_number")
    moments = np.linalg.eigvalsh(moment_of_inertia(geometry, masses))
    cutoff = max(float(np.max(moments, initial=0.0)) * 1.0e-8, 1.0e-14)
    positive = moments[moments > cutoff]
    if len(positive) == 0:
        if symmetry != 1:
            raise VTSTError("an atom requires symmetry_number=1")
        return 0.0, 0
    if len(positive) == 2:
        value = math.log(2.0 / beta) + 0.5 * float(np.log(positive).sum())
    elif len(positive) == 3:
        value = (
            0.5 * math.log(math.pi)
            + 1.5 * math.log(2.0 / beta)
            + 0.5 * float(np.log(positive).sum())
        )
    else:
        raise VTSTError("invalid inertia tensor")
    return value - math.log(symmetry), len(positive)


def harmonic_log_partition(frequencies_au: Sequence[float], beta: float) -> float:
    """Quantum harmonic partition including zero-point energy."""
    beta = _positive(beta, "beta")
    frequencies = _finite_array(frequencies_au, None, "frequencies_au")
    if frequencies.ndim != 1 or np.any(frequencies <= 0):
        raise VTSTError("frequencies_au must be a one-dimensional positive array")
    x = beta * frequencies
    return -float((0.5 * x + np.log(-np.expm1(-x))).sum())


@dataclass
class PartitionSpecies:
    """A gas-phase harmonic species used in the reactant partition function."""

    geometry_bohr: np.ndarray
    masses: np.ndarray
    frequencies_au: np.ndarray
    symmetry_number: int = 1
    electronic_degeneracy: int = 1
    label: str = "reactant"

    def __post_init__(self) -> None:
        self.geometry_bohr = _finite_array(self.geometry_bohr, None, "geometry_bohr")
        if self.geometry_bohr.ndim != 2 or self.geometry_bohr.shape[1] != 3:
            raise VTSTError("geometry_bohr must have shape (N,3)")
        self.masses = _finite_array(self.masses, (len(self.geometry_bohr),), "masses")
        if np.any(self.masses <= 0):
            raise VTSTError("masses must be positive")
        self.frequencies_au = _finite_array(
            self.frequencies_au, None, "frequencies_au"
        )
        if self.frequencies_au.ndim != 1 or np.any(self.frequencies_au <= 0):
            raise VTSTError("frequencies_au must be a one-dimensional positive array")
        self.symmetry_number = _positive_integer(
            self.symmetry_number, "symmetry_number"
        )
        self.electronic_degeneracy = _positive_integer(
            self.electronic_degeneracy, "electronic_degeneracy"
        )
        self.label = str(self.label)

    def partition_diagnostics(self, beta: float) -> dict[str, Any]:
        beta = _positive(beta, "beta")
        log_translation = 1.5 * math.log(float(self.masses.sum()) / (2.0 * math.pi * beta))
        log_rotation, rotational_dimensions = _log_rotation_partition(
            self.geometry_bohr, self.masses, beta, self.symmetry_number
        )
        log_vibration = harmonic_log_partition(self.frequencies_au, beta)
        log_electronic = math.log(self.electronic_degeneracy)
        return {
            "label": self.label,
            "log_partition": log_translation
            + log_rotation
            + log_vibration
            + log_electronic,
            "log_translation_partition_per_bohr3": log_translation,
            "log_rotation_partition": log_rotation,
            "log_vibrational_partition": log_vibration,
            "log_electronic_degeneracy": log_electronic,
            "frequencies_au": self.frequencies_au.tolist(),
            "zero_point_energy_hartree": 0.5 * float(self.frequencies_au.sum()),
            "symmetry_number": self.symmetry_number,
            "electronic_degeneracy": self.electronic_degeneracy,
            "rotational_dimensions": rotational_dimensions,
            "normalization": "per_bohr^3",
        }

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


def species_from_hessian(
    geometry_bohr: np.ndarray,
    masses: Sequence[float],
    hessian: np.ndarray,
    *,
    symmetry_number: int = 1,
    electronic_degeneracy: int = 1,
    label: str = "reactant",
    scaling_mass: float = AMU_TO_AU,
    negative_tolerance: float = 1.0e-9,
    zero_tolerance: float = 1.0e-10,
) -> PartitionSpecies:
    modes = generalized_normal_modes(
        geometry_bohr,
        masses,
        hessian,
        tangent_scaled=None,
        scaling_mass=scaling_mass,
        negative_tolerance=negative_tolerance,
        zero_tolerance=zero_tolerance,
        require_positive=True,
    )
    return PartitionSpecies(
        geometry_bohr=np.asarray(geometry_bohr),
        masses=np.asarray(masses),
        frequencies_au=modes.frequencies_au,
        symmetry_number=symmetry_number,
        electronic_degeneracy=electronic_degeneracy,
        label=label,
    )


def fragment_species_from_path_endpoint(
    path: MEPPath,
    masses: Sequence[float],
    fragments: Sequence[Sequence[int]],
    *,
    endpoint: int = 0,
    symmetry_numbers: Sequence[int] | None = None,
    electronic_degeneracies: Sequence[int] | None = None,
) -> list[PartitionSpecies]:
    """Construct fragment partitions from endpoint Hessian diagonal blocks."""
    mass = _finite_array(masses, (path.natoms,), "masses")
    if endpoint not in (0, -1, len(path.s_bohr) - 1):
        raise VTSTError("endpoint must select the first or last MEP point")
    index = 0 if endpoint == 0 else len(path.s_bohr) - 1
    groups = [tuple(int(atom) for atom in fragment) for fragment in fragments]
    flat = [atom for group in groups for atom in group]
    if any(not group for group in groups) or sorted(flat) != list(range(path.natoms)):
        raise VTSTError("fragments must form a complete disjoint atom partition")
    symmetry = list(symmetry_numbers or [1] * len(groups))
    degeneracy = list(electronic_degeneracies or [1] * len(groups))
    if len(symmetry) != len(groups) or len(degeneracy) != len(groups):
        raise VTSTError("one symmetry number and electronic degeneracy are required per fragment")
    result = []
    full_hessian = path.hessians_hartree_per_bohr2[index]
    for fragment_index, atoms in enumerate(groups):
        coordinates = [3 * atom + axis for atom in atoms for axis in range(3)]
        block = full_hessian[np.ix_(coordinates, coordinates)]
        result.append(
            species_from_hessian(
                path.geometries_bohr[index, list(atoms)],
                mass[list(atoms)],
                block,
                symmetry_number=symmetry[fragment_index],
                electronic_degeneracy=degeneracy[fragment_index],
                label=f"reactant_{fragment_index}",
                scaling_mass=path.scaling_mass,
            )
        )
    return result


def _transition_state_partition(
    geometry: np.ndarray,
    masses: np.ndarray,
    frequencies: np.ndarray,
    beta: float,
    symmetry_number: int,
    electronic_degeneracy: int,
) -> dict[str, Any]:
    log_translation = 1.5 * math.log(float(masses.sum()) / (2.0 * math.pi * beta))
    log_rotation, rotational_dimensions = _log_rotation_partition(
        geometry, masses, beta, symmetry_number
    )
    log_vibration = harmonic_log_partition(frequencies, beta)
    log_electronic = math.log(_positive_integer(electronic_degeneracy, "electronic_degeneracy"))
    return {
        "log_partition": log_translation
        + log_rotation
        + log_vibration
        + log_electronic,
        "log_translation_partition_per_bohr3": log_translation,
        "log_rotation_partition": log_rotation,
        "log_vibrational_partition": log_vibration,
        "log_electronic_degeneracy": log_electronic,
        "rotational_dimensions": rotational_dimensions,
    }


def _rate_record(log_rate_au: float, molecularity: int) -> dict[str, Any]:
    molecularity = _positive_integer(molecularity, "molecularity")
    power = 3 * (molecularity - 1)
    log_rate_si = float(
        log_rate_au + power * math.log(BOHR_M) - math.log(ATOMIC_TIME_S)
    )
    value = math.exp(log_rate_si) if log_rate_si <= math.log(sys.float_info.max) else None
    return {
        "rate": value,
        "rate_unit": "s^-1" if power == 0 else f"m^{power}/s",
        "log_rate_au": float(log_rate_au),
        "log_rate_si": log_rate_si,
        "range_status": (
            "overflow" if value is None else "underflow" if value == 0.0 else "finite"
        ),
    }


@dataclass
class CVTTemperatureResult:
    temperature_kelvin: float
    beta_hartree_inverse: float
    delta_g_hartree: np.ndarray
    log_rate_profile_au: np.ndarray
    cvt_index: int
    tst_index: int
    tst: dict[str, Any]
    cvt: dict[str, Any]
    reactant_partition: list[dict[str, Any]]
    transition_state_partition_at_cvt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


def canonical_variational_profile(
    path: MEPPath,
    masses: Sequence[float],
    modes: PathModeAnalysis,
    reactants: Sequence[PartitionSpecies],
    temperature_kelvin: float,
    *,
    transition_state_symmetry_number: int = 1,
    transition_state_electronic_degeneracy: int = 1,
    reaction_degeneracy: int = 1,
    reactant_energy_hartree: float | None = None,
) -> CVTTemperatureResult:
    """Return TST and CVT profiles using gas-phase number-density conventions."""
    temperature = _positive(temperature_kelvin, "temperature_kelvin")
    beta = 1.0 / (KB_AU * temperature)
    mass = _finite_array(masses, (path.natoms,), "masses")
    reactant_list = list(reactants)
    if not reactant_list or not all(isinstance(item, PartitionSpecies) for item in reactant_list):
        raise VTSTError("reactants must contain PartitionSpecies objects")
    combined = np.concatenate([item.masses for item in reactant_list])
    if combined.shape != mass.shape or not np.allclose(
        np.sort(combined), np.sort(mass), rtol=1.0e-12, atol=0.0
    ):
        raise VTSTError("reactant masses must match the reacting system")
    symmetry = _positive_integer(
        transition_state_symmetry_number, "transition_state_symmetry_number"
    )
    electronic = _positive_integer(
        transition_state_electronic_degeneracy,
        "transition_state_electronic_degeneracy",
    )
    degeneracy = _positive_integer(reaction_degeneracy, "reaction_degeneracy")
    reactant_diagnostics = [item.partition_diagnostics(beta) for item in reactant_list]
    log_reactants = float(sum(item["log_partition"] for item in reactant_diagnostics))
    reference_energy = (
        float(path.energies_hartree[0])
        if reactant_energy_hartree is None
        else float(reactant_energy_hartree)
    )
    if not math.isfinite(reference_energy):
        raise VTSTError("reactant_energy_hartree must be finite")
    log_rate = np.empty(len(path.s_bohr), dtype=float)
    delta_g = np.empty(len(path.s_bohr), dtype=float)
    partitions = []
    for index, (geometry, energy, frequencies) in enumerate(
        zip(path.geometries_bohr, path.energies_hartree, modes.frequencies_au)
    ):
        partition = _transition_state_partition(
            geometry,
            mass,
            frequencies,
            beta,
            symmetry,
            electronic,
        )
        partitions.append(partition)
        log_ratio = partition["log_partition"] - log_reactants + math.log(degeneracy)
        activation = float(energy - reference_energy)
        delta_g[index] = activation - log_ratio / beta
        log_rate[index] = -math.log(2.0 * math.pi * beta) - beta * delta_g[index]
    cvt_index = int(np.argmin(log_rate))
    tst_index = path.saddle_index
    return CVTTemperatureResult(
        temperature_kelvin=temperature,
        beta_hartree_inverse=beta,
        delta_g_hartree=delta_g,
        log_rate_profile_au=log_rate,
        cvt_index=cvt_index,
        tst_index=tst_index,
        tst={
            **_rate_record(float(log_rate[tst_index]), len(reactant_list)),
            "s_bohr": float(path.s_bohr[tst_index]),
            "delta_g_hartree": float(delta_g[tst_index]),
        },
        cvt={
            **_rate_record(float(log_rate[cvt_index]), len(reactant_list)),
            "s_bohr": float(path.s_bohr[cvt_index]),
            "delta_g_hartree": float(delta_g[cvt_index]),
            "variational_transmission_coefficient": float(
                math.exp(min(0.0, log_rate[cvt_index] - log_rate[tst_index]))
            ),
        },
        reactant_partition=reactant_diagnostics,
        transition_state_partition_at_cvt=partitions[cvt_index],
    )
