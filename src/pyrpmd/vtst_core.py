"""Shared numerical primitives for reaction-path VTST.

Internal coordinates, masses, energies, frequencies and time use atomic units.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .constants import AMU_TO_AU


class VTSTError(ValueError):
    """Raised when a VTST calculation is undefined or numerically unresolved."""


def _finite_array(value: Any, shape: tuple[int, ...] | None, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or (shape is not None and array.shape != shape):
        expected = "" if shape is None else f" with shape {shape}"
        raise VTSTError(f"{name} must be a finite real array{expected}")
    array = np.array(array, dtype=float, order="C", copy=True)
    if not np.isfinite(array).all():
        raise VTSTError(f"{name} must be a finite real array")
    return array


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise VTSTError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise VTSTError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return result


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise VTSTError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise VTSTError(f"{name} must be an integer >= {minimum}")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _mass_vector(masses: Sequence[float], natoms: int) -> np.ndarray:
    mass = _finite_array(masses, (natoms,), "masses")
    if np.any(mass <= 0):
        raise VTSTError("masses must be positive")
    return np.repeat(mass, 3)


def mass_scaled_coordinates(
    geometry_bohr: np.ndarray,
    masses: Sequence[float],
    *,
    scaling_mass: float = AMU_TO_AU,
) -> np.ndarray:
    """Return flattened isoinertial coordinates in bohr."""
    geometry = _finite_array(geometry_bohr, None, "geometry_bohr")
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise VTSTError("geometry_bohr must have shape (N,3)")
    mu = _positive(scaling_mass, "scaling_mass")
    repeated = _mass_vector(masses, len(geometry))
    return geometry.ravel() * np.sqrt(repeated / mu)


def cartesian_coordinates(
    scaled: np.ndarray,
    masses: Sequence[float],
    *,
    scaling_mass: float = AMU_TO_AU,
) -> np.ndarray:
    """Convert flattened isoinertial coordinates back to Cartesian bohr."""
    x = _finite_array(scaled, None, "scaled coordinates")
    if x.ndim != 1 or x.size % 3:
        raise VTSTError("scaled coordinates must be a flattened 3N vector")
    natoms = x.size // 3
    mu = _positive(scaling_mass, "scaling_mass")
    repeated = _mass_vector(masses, natoms)
    return (x * np.sqrt(mu / repeated)).reshape(natoms, 3)


def _scaled_gradient(
    cartesian_gradient: np.ndarray,
    masses: Sequence[float],
    scaling_mass: float,
) -> np.ndarray:
    gradient = _finite_array(cartesian_gradient, None, "gradient")
    if gradient.ndim != 2 or gradient.shape[1] != 3:
        raise VTSTError("gradient must have shape (N,3)")
    repeated = _mass_vector(masses, len(gradient))
    return gradient.ravel() * np.sqrt(scaling_mass / repeated)


def _scaled_hessian(
    cartesian_hessian: np.ndarray,
    masses: Sequence[float],
    scaling_mass: float,
) -> np.ndarray:
    natoms = len(masses)
    hessian = _finite_array(cartesian_hessian, (3 * natoms, 3 * natoms), "hessian")
    if not np.allclose(hessian, hessian.T, rtol=1e-8, atol=1e-11):
        raise VTSTError("hessian must be symmetric")
    repeated = _mass_vector(masses, natoms)
    result = hessian * scaling_mass / np.sqrt(repeated[:, None] * repeated[None, :])
    return 0.5 * (result + result.T)


def _provider_energy_gradient(potential: Any, geometry: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate a PyRPMD PES and copy provider-owned buffers immediately."""
    from .potentials.base import coordinates, validate_result

    q = coordinates(np.asarray(geometry, dtype=float)[None])
    result = validate_result(potential.evaluate(q), q.shape)
    return float(result.energy[0]), -np.array(result.forces[0], dtype=float, copy=True)


def _analytic_hessian(potential: Any, geometry: np.ndarray) -> np.ndarray | None:
    method = getattr(potential, "hessian", None)
    if not callable(method):
        return None
    q = np.asarray(geometry, dtype=float)[None]
    raw = np.asarray(method(q))
    natoms = q.shape[1]
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise VTSTError("potential.hessian returned non-finite or non-real values")
    if raw.shape == (1, 3 * natoms, 3 * natoms):
        matrix = raw[0]
    elif raw.shape == (3 * natoms, 3 * natoms):
        matrix = raw
    elif raw.shape == (1, natoms, 3, natoms, 3):
        matrix = raw[0].reshape(3 * natoms, 3 * natoms)
    else:
        raise VTSTError(
            "potential.hessian must return (B,3N,3N), (3N,3N), or (B,N,3,N,3)"
        )
    matrix = np.array(matrix, dtype=float, copy=True)
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-11):
        raise VTSTError("potential.hessian returned a nonsymmetric matrix")
    return 0.5 * (matrix + matrix.T)


def evaluate_energy_gradient_hessian(
    potential: Any,
    geometry_bohr: np.ndarray,
    *,
    finite_difference_step: float = 1.0e-4,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate energy, Cartesian gradient, and unweighted Cartesian Hessian.

    An advertised analytic Hessian is authoritative: malformed or failed
    analytic calls are never replaced silently by finite differences.
    """
    geometry = _finite_array(geometry_bohr, None, "geometry_bohr")
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise VTSTError("geometry_bohr must have shape (N,3)")
    step = _positive(finite_difference_step, "finite_difference_step")
    energy, gradient = _provider_energy_gradient(potential, geometry)
    hessian = _analytic_hessian(potential, geometry)
    if hessian is None:
        flat = geometry.ravel()
        hessian = np.empty((flat.size, flat.size), dtype=float)
        for column in range(flat.size):
            plus = flat.copy()
            minus = flat.copy()
            plus[column] += step
            minus[column] -= step
            _, gradient_plus = _provider_energy_gradient(potential, plus.reshape(geometry.shape))
            _, gradient_minus = _provider_energy_gradient(potential, minus.reshape(geometry.shape))
            hessian[:, column] = (
                gradient_plus.ravel() - gradient_minus.ravel()
            ) / (2.0 * step)
        hessian = 0.5 * (hessian + hessian.T)
    return energy, gradient, hessian


def _orthonormal_columns(vectors: np.ndarray, *, tolerance: float = 1.0e-10) -> np.ndarray:
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2:
        raise VTSTError("basis vectors must be a matrix")
    if array.shape[1] == 0:
        return np.empty((array.shape[0], 0), dtype=float)
    u, singular, _ = np.linalg.svd(array, full_matrices=False)
    cutoff = max(float(singular[0]) * tolerance, tolerance)
    rank = int(np.count_nonzero(singular > cutoff))
    return u[:, :rank]


def rigid_body_basis(
    geometry_bohr: np.ndarray,
    masses: Sequence[float],
    *,
    scaling_mass: float = AMU_TO_AU,
) -> np.ndarray:
    """Mass-scaled translations and non-null rotations as orthonormal columns."""
    geometry = _finite_array(geometry_bohr, None, "geometry_bohr")
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise VTSTError("geometry_bohr must have shape (N,3)")
    mass = _finite_array(masses, (len(geometry),), "masses")
    if np.any(mass <= 0):
        raise VTSTError("masses must be positive")
    mu = _positive(scaling_mass, "scaling_mass")
    center = np.sum(mass[:, None] * geometry, axis=0) / float(mass.sum())
    centered = geometry - center
    weight = np.sqrt(mass / mu)[:, None]
    translations = np.column_stack(
        [(weight * axis).ravel() for axis in np.eye(3)]
    )
    rotations = np.column_stack(
        [(weight * np.cross(axis, centered)).ravel() for axis in np.eye(3)]
    )
    return _orthonormal_columns(np.column_stack((translations, rotations)))


def _complement(basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.eye(basis.shape[0])
    return np.linalg.qr(basis, mode="complete")[0][:, basis.shape[1] :]


@dataclass
class GeneralizedModes:
    """Modes perpendicular to a reaction-path dividing surface."""

    eigenvalues_omega2: np.ndarray
    frequencies_au: np.ndarray
    vectors_scaled: np.ndarray
    tangent_scaled: np.ndarray | None
    removed_rigid_modes: int
    removed_total_modes: int

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self.__dict__)


def generalized_normal_modes(
    geometry_bohr: np.ndarray,
    masses: Sequence[float],
    hessian: np.ndarray,
    *,
    tangent_scaled: np.ndarray | None = None,
    scaling_mass: float = AMU_TO_AU,
    negative_tolerance: float = 1.0e-9,
    zero_tolerance: float = 1.0e-10,
    require_positive: bool = True,
) -> GeneralizedModes:
    """Project translations, rotations, and optionally the path tangent.

    The default is a Cartesian projection.  The returned vectors are expressed
    in the isoinertial coordinate basis and are orthonormal in that basis.
    """
    geometry = _finite_array(geometry_bohr, None, "geometry_bohr")
    if geometry.ndim != 2 or geometry.shape[1] != 3:
        raise VTSTError("geometry_bohr must have shape (N,3)")
    mass = _finite_array(masses, (len(geometry),), "masses")
    if np.any(mass <= 0):
        raise VTSTError("masses must be positive")
    negative = _positive(negative_tolerance, "negative_tolerance", allow_zero=True)
    zero = _positive(zero_tolerance, "zero_tolerance", allow_zero=True)
    rigid = rigid_body_basis(geometry, mass, scaling_mass=scaling_mass)
    columns = [rigid]
    tangent = None
    if tangent_scaled is not None:
        tangent = _finite_array(tangent_scaled, (3 * len(geometry),), "tangent_scaled")
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-14:
            raise VTSTError("tangent_scaled must have nonzero norm")
        tangent = tangent / norm
        columns.insert(0, tangent[:, None])
    constraints = _orthonormal_columns(np.column_stack(columns))
    complement = _complement(constraints)
    scaled_hessian = _scaled_hessian(hessian, mass, scaling_mass)
    # The kinetic mass in x coordinates is scaling_mass, so omega^2 = H_x/mu.
    reduced = complement.T @ scaled_hessian @ complement / scaling_mass
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, vectors = np.linalg.eigh(reduced)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    negative_cutoff = max(negative, scale * 1.0e-12)
    zero_cutoff = max(zero, scale * 1.0e-13)
    if require_positive and np.any(eigenvalues < -negative_cutoff):
        count = int(np.count_nonzero(eigenvalues < -negative_cutoff))
        raise VTSTError(
            f"generalized transition state has {count} transverse imaginary mode(s)"
        )
    if require_positive and np.any(np.abs(eigenvalues) <= zero_cutoff):
        count = int(np.count_nonzero(np.abs(eigenvalues) <= zero_cutoff))
        raise VTSTError(
            f"generalized transition state has {count} unresolved transverse zero mode(s)"
        )
    if require_positive:
        kept = eigenvalues > zero_cutoff
    else:
        kept = np.abs(eigenvalues) > zero_cutoff
    eigenvalues = eigenvalues[kept]
    vectors_scaled = complement @ vectors[:, kept]
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    return GeneralizedModes(
        eigenvalues_omega2=eigenvalues,
        frequencies_au=frequencies,
        vectors_scaled=vectors_scaled,
        tangent_scaled=tangent,
        removed_rigid_modes=rigid.shape[1],
        removed_total_modes=constraints.shape[1],
    )


def _saddle_mode(
    geometry: np.ndarray,
    masses: np.ndarray,
    hessian: np.ndarray,
    *,
    scaling_mass: float,
    negative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    rigid = rigid_body_basis(geometry, masses, scaling_mass=scaling_mass)
    complement = _complement(rigid)
    scaled_hessian = _scaled_hessian(hessian, masses, scaling_mass)
    reduced = complement.T @ scaled_hessian @ complement / scaling_mass
    eigenvalues, vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    cutoff = _positive(negative_tolerance, "negative_tolerance", allow_zero=True)
    negative = np.flatnonzero(eigenvalues < -cutoff)
    if len(negative) != 1:
        raise VTSTError(
            f"MEP start requires exactly one resolved unstable mode; found {len(negative)}"
        )
    mode = complement @ vectors[:, negative[0]]
    mode /= np.linalg.norm(mode)
    return mode, eigenvalues
