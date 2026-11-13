"""Thermodynamic Suzuki--Chin estimators in atomic units.

The sampling potential contains an explicit beta-squared force correction.
Differentiating it therefore differs from averaging the sampling energy.
These estimators apply to canonical SC configurations (and SC NPT with the
enthalpy fluctuation); they do not apply to a PIGLET nonequilibrium measure.

References: Kapil, Behler and Ceriotti, JCP 145, 234103 (2016),
doi:10.1063/1.4971438; Kapil et al., JCTC 15, 3237 (2019),
doi:10.1021/acs.jctc.8b01297, supporting information S1 and S3.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from .constants import KB_AU
from .path_integral import _positive
from .potentials.base import PotentialError, evaluate_checked
from .quantum_statistics import _block_layout, _jackknife_error, _series


def _paths(potential, positions):
    from .pimd_methods import SuzukiChinPotential
    if not isinstance(potential, SuzukiChinPotential):
        raise TypeError("SC estimators require SuzukiChinPotential")
    paths = np.asarray(positions)
    expected = (potential.nbeads, len(potential.masses), 3)
    if (paths.dtype.kind not in "fiu" or paths.ndim != 4
            or paths.shape[1:] != expected or not len(paths)
            or not np.isfinite(paths).all()):
        raise ValueError("SC paths must be finite [sample,P,N,3] arrays matching the potential")
    return np.asarray(paths, dtype=float)


def suzuki_chin_estimators(potential, positions, *, volume=None,
                          finite_difference_step=1e-3, require_pressure=True):
    """Return one ring's physical energy/pressure and heat-capacity ingredients.

    The pressure uses the expected centroid kinetic contribution N/(beta*V),
    which reduces noise compared with fictitious centroid momenta.  The latter
    are used by the dynamical barostat, with the same mean.
    """
    q = potential._q(positions)
    values = suzuki_chin_estimators_batch(potential, q[None], volumes=volume,
                                        finite_difference_step=finite_difference_step,
                                        require_pressure=require_pressure)
    return {key: float(value[0]) for key, value in values.items()}


def suzuki_chin_estimators_batch(potential, positions, *, volumes=None,
                                finite_difference_step=1e-3, require_pressure=True):
    """Fixed-cell SC estimator columns for canonical paths [sample,P,N,3].

    Periodic rings must be continuous/unwrapped across beads.  A provider's
    cell is kept fixed throughout every beta derivative.  For a trajectory
    with changing cells, call the scalar estimator after restoring each cell;
    ``volumes`` alone does not change a provider's cell.
    With ``require_pressure=False`` a provider without virials still supplies
    energy and heat-capacity ingredients, and omits ``pressure_au``.  No virial
    is inferred from Cartesian coordinates when the cell derivative is absent.

    Write Usc = sum_j(w_j*V_j + c_j*|F_j|^2/m), c_j proportional to beta^2.
    For delta=q-q_centroid, the total thermodynamic energy is

        E = D/(2*beta) + [sum(w*V + 3*c*F^2/m) - sum(delta*Fsc)/2]/P.

    The separate physical potential and kinetic energies are respectively
    sum(w*V+2*c*F^2/m)/P and E minus that quantity.  The even-bead operator
    estimators are also supplied at alpha=0, with distinct names.

    Heat-capacity curvature is a''(beta), where
    a(x)=x*Usc(q_centroid+sqrt(x/beta)*delta, x)/P.  Central differences at h
    and h/2 provide an explicit sensitivity check; h/2 supplies the result.
    The force-square coefficient is recomputed at x, not held at beta.
    """
    paths = _paths(potential, positions)
    if not isinstance(require_pressure, (bool, np.bool_)):
        raise ValueError("require_pressure must be a boolean")
    relative_step = _positive(finite_difference_step, "finite_difference_step")
    if relative_step >= 1 or relative_step < 8 * np.finfo(float).eps:
        raise ValueError("finite_difference_step must be resolvable and strictly below 1")
    samples, beads, atoms, _ = paths.shape
    beta = potential.beta
    flat = paths.reshape(-1, atoms, 3)
    effective, physical, correction_flat = potential._evaluate_sc(flat)
    forces = effective.forces.reshape(paths.shape)
    physical_forces = physical.forces.reshape(paths.shape)
    physical_energies = physical.energy.reshape(samples, beads)
    correction = correction_flat.reshape(samples, beads).mean(axis=1)
    base_potential = (physical_energies * potential.weights).mean(axis=1)
    centroid = paths.mean(axis=1, keepdims=True)
    offsets = paths - centroid
    centroid_force = np.sum(offsets * forces, axis=(1, 2, 3)) / beads
    dof = 3 * atoms
    energy = dof / (2 * beta) + base_potential + 3 * correction - 0.5 * centroid_force
    physical_potential = base_potential + 2 * correction

    def residual(shift):
        scale = 1 + shift
        scaled = centroid + np.sqrt(scale) * offsets
        q = scaled.reshape(flat.shape)
        result = evaluate_checked(potential.potential, q)
        v = (result.energy.reshape(samples, beads) * potential.weights).mean(axis=1)
        f = result.forces.reshape(paths.shape)
        g = np.sum(f * f / potential.masses[None, None, :, None], axis=(2, 3))
        c = (g * potential.coefficients).mean(axis=1)
        # Subtract the path-specific potential baseline before differencing.
        # Its known linear beta term is added back to a' analytically, keeping
        # arbitrary constant PES offsets out of the second derivative.
        cube_minus_one = shift * (3 + shift * (3 + shift))
        return beta * (scale * (v - base_potential)
                       + scale**3 * (c - correction) + cube_minus_one * correction)

    def derivatives(step):
        plus, minus = residual(step), residual(-step)
        h = beta * step
        return base_potential + (plus - minus) / (2 * h), (plus + minus) / h**2

    coarse_first, coarse_second = derivatives(relative_step)
    fine_first, fine_second = derivatives(relative_step / 2)
    values = {
        "normalization_dof": np.full(samples, dof, dtype=float),
        "internal_energy_hartree": energy,
        "potential_energy_hartree": physical_potential,
        "kinetic_energy_hartree": energy - physical_potential,
        "scaled_action_first_beta_derivative": energy - dof / (2 * beta),
        "scaled_action_second_beta_derivative": fine_second,
        "finite_difference_relative_beta_step": np.full(samples, relative_step / 2),
        "finite_difference_energy_coarse_minus_fine_hartree": coarse_first - fine_first,
        "finite_difference_curvature_coarse_minus_fine": coarse_second - fine_second,
        "finite_difference_energy_disagreement_hartree": energy - dof / (2 * beta) - fine_first,
    }
    if potential.alpha == 0:
        v_op = physical_energies[:, ::2].mean(axis=1)
        k_op = dof / (2 * beta) - np.sum(
            offsets[:, ::2] * physical_forces[:, ::2], axis=(1, 2, 3)) / beads
        values.update({"operator_potential_energy_hartree": v_op,
                       "operator_kinetic_energy_hartree": k_op,
                       "operator_internal_energy_hartree": v_op + k_op})
    if volumes is not None:
        volumes = np.asarray(volumes)
        if (volumes.dtype.kind not in "fiu" or volumes.shape not in ((), (samples,))
                or not np.isfinite(volumes).all() or np.any(volumes <= 0)):
            raise ValueError("volumes must be positive finite bohr^3 values, scalar or [sample]")
        if effective.virial is None and require_pressure:
            raise PotentialError("SC pressure requires physical configurational virials, including cell derivatives")
        if effective.virial is not None:
            w = effective.virial.reshape(samples, beads, 3, 3)
            trace = np.trace(w, axis1=2, axis2=3).mean(axis=1)
            values["pressure_au"] = (dof / beta + trace - centroid_force) / (3 * volumes)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("SC estimator overflow; inspect beta, coordinates and the finite-difference step")
    return values


def suzuki_chin_thermodynamics(estimator_series, beta, *, volumes=None,
                              pressure_au=None, block_size=None):
    """Combine canonical SC estimator columns into E and Cv, or H and Cp.

    ``estimator_series`` is the column mapping produced by ``estimators_batch``
    or a sequence of scalar dictionaries from ``estimators``.  For NPT supply
    both the sampled volumes and the constant external pressure; the result
    is Cp.  Do not interpret it as Cv.  Neither heat capacity is the variance
    of the fictitious ring-polymer energy or momenta.

        C/kB = D/2 + beta^2 * (Var(E [+ p_external*V]) - mean(a''))

    Block jackknife errors are conditional on blocks exceeding correlation
    times; bead, timestep and finite-difference convergence are separate.
    This function does not assert convergence from a short finite series.
    """
    beta = _positive(beta, "beta")
    if not isinstance(estimator_series, Mapping):
        rows = list(estimator_series)
        if not rows or not all(isinstance(row, Mapping) for row in rows):
            raise ValueError("SC estimator series must be a column mapping or scalar estimator dictionaries")
        if any(set(row) != set(rows[0]) for row in rows):
            raise ValueError("SC estimator dictionaries must contain the same fields")
        estimator_series = {key: [row[key] for row in rows] for key in rows[0]}
    required = ("internal_energy_hartree", "scaled_action_second_beta_derivative", "normalization_dof")
    if any(key not in estimator_series for key in required):
        raise ValueError("SC thermodynamics requires energy, scaled action curvature and normalization_dof")
    energy, second, dofs = [_series(estimator_series[key], key, minimum=2) for key in required]
    if not (energy.shape == second.shape == dofs.shape):
        raise ValueError("SC estimator columns must have equal lengths")
    if dofs[0] <= 0 or dofs[0] != int(dofs[0]) or np.any(dofs != dofs[0]):
        raise ValueError("normalization_dof must be a constant positive integer")
    dof = float(dofs[0])
    fluctuating = energy.copy()
    npt = pressure_au is not None
    if volumes is not None and not npt:
        raise ValueError("sampled volumes require a constant external pressure_au for NPT thermodynamics")
    if npt:
        pressure_value = np.asarray(pressure_au)
        if (pressure_value.dtype.kind not in "fiu" or pressure_value.shape != ()
                or not np.isfinite(pressure_value)):
            raise ValueError("pressure_au must be a finite real constant external pressure")
        if volumes is None:
            raise ValueError("NPT heat capacity requires the sampled volumes")
        volumes = _series(volumes, "volumes", minimum=2)
        if volumes.shape != energy.shape or np.any(volumes <= 0):
            raise ValueError("sampled volumes must be positive and match estimator lengths")
        fluctuating += float(pressure_au) * volumes
    scaled = beta * fluctuating
    curvature = beta**2 * second
    centered = scaled - scaled.mean()
    heat = 0.5 * dof + np.mean(centered**2) - curvature.mean()
    diagnostics = _block_layout((scaled, centered**2, curvature, beta * energy), block_size)
    count, length = diagnostics["complete_blocks"], diagnostics["block_length_frames"]
    error_energy = error_heat = error_enthalpy = None
    if count >= 4:
        n = count * length
        blocks = centered[:n].reshape(count, length)
        mean_c = blocks.mean(axis=1)
        mean_cc = (blocks**2).mean(axis=1)
        mean_s = curvature[:n].reshape(count, length).mean(axis=1)
        leave_c = (mean_c.sum() - mean_c) / (count - 1)
        leave_cc = (mean_cc.sum() - mean_cc) / (count - 1)
        leave_s = (mean_s.sum() - mean_s) / (count - 1)
        error_heat = _jackknife_error(0.5 * dof + leave_cc - leave_c**2 - leave_s)
        means_e = energy[:n].reshape(count, length).mean(axis=1)
        error_energy = float(means_e.std(ddof=1) / math.sqrt(count))
        error_enthalpy = float(mean_c.std(ddof=1) / (math.sqrt(count) * beta))
    diagnostics["reason"] = None if count >= 4 else "fewer_than_four_complete_blocks"
    kind = "Cp" if npt else "Cv"
    report = {
        "method": "suzuki_chin", "ensemble": "NPT" if npt else "NVT",
        "estimator": "thermodynamic_centroid_virial_and_scaled_coordinates",
        "temperature_K": 1 / (KB_AU * beta), "samples": len(energy),
        "internal_energy_hartree": float(energy.mean()),
        "internal_energy_standard_error_hartree": error_energy,
        "heat_capacity_kind": kind, "heat_capacity_over_kB": float(heat),
        "heat_capacity_standard_error_over_kB": error_heat,
        "heat_capacity_hartree_per_K": float(heat * KB_AU),
        "diagnostics": diagnostics,
        "scope": "canonical_SC_temperature_independent_PES; unconstrained_3D_particles",
    }
    if npt:
        report.update({"enthalpy_hartree": float(fluctuating.mean()),
                       "enthalpy_standard_error_hartree": error_enthalpy,
                       "external_pressure_au": float(pressure_au),
                       "mean_volume_bohr3": float(volumes.mean())})
    for key in ("potential_energy_hartree", "kinetic_energy_hartree", "pressure_au",
                "operator_potential_energy_hartree", "operator_kinetic_energy_hartree",
                "operator_internal_energy_hartree"):
        if key in estimator_series:
            column = _series(estimator_series[key], key, minimum=2)
            if column.shape != energy.shape:
                raise ValueError("SC estimator columns must have equal lengths")
            report[key] = float(column.mean())
    if "finite_difference_curvature_coarse_minus_fine" in estimator_series:
        delta = _series(estimator_series["finite_difference_curvature_coarse_minus_fine"],
                        "finite difference curvature", minimum=2)
        if delta.shape != energy.shape:
            raise ValueError("SC estimator columns must have equal lengths")
        report["finite_difference"] = {
            "heat_capacity_curvature_coarse_minus_fine_over_kB": float(-beta**2 * delta.mean()),
            "maximum_dimensionless_curvature_change": float(beta**2 * np.max(np.abs(delta))),
            "scope": "same_sample_h_and_h_over_2_sensitivity; not a convergence guarantee",
        }
    return report


__all__ = ["suzuki_chin_estimators", "suzuki_chin_estimators_batch", "suzuki_chin_thermodynamics"]
