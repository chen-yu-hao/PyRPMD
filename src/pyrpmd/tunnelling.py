"""Ground-state instanton splittings for equivalent, bound Cartesian wells.

This is an open-path calculation, not a thermal rate or a closed-ring orbit.
Reference: Richardson & Althorpe, JCP 134, 054109 (2011),
https://doi.org/10.1063/1.3530589. See docs/low-temperature.md for scope.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.linalg.blas import dsyr2
from scipy.optimize import minimize, root

from .constants import WAVENUMBER_TO_AU
from .instanton import InstantonError, _mask_and_masses, _physical_hessians, _positive
from .low_temperature import HarmonicState, _integer
from .potentials.base import coordinates, validate_result


def _phase_projected_spectrum(operator, tangent):
    """Restrict a symmetric fluctuation operator to the phase complement.

    A Householder reflection sends the geometric phase tangent to the first
    coordinate. Removing that coordinate gives the same spectrum as Q.T @ H @ Q
    for any orthonormal complement Q. The reflection costs O(D**2) work without
    constructing Q or multiplying dense D-by-D matrices. The final symmetric
    eigensolve is still dense and costs O(D**3).

    Write R = I - 2*u*u.T and w = H*u - (u.T*H*u)*u. Then
    R*H*R = H - 2*(u*w.T + w*u.T). A symmetric rank-two BLAS update computes
    only the lower triangle consumed by eigvalsh, using one owned matrix and
    O(D) vector storage. Neither input is modified.
    """
    vector = tangent / np.linalg.norm(tangent)
    vector[0] += 1.0 if vector[0] >= 0.0 else -1.0
    vector /= np.linalg.norm(vector)
    product = operator @ vector
    product -= float(vector @ product) * vector
    reduced = np.array(operator[1:, 1:], order="F", copy=True)
    reduced = dsyr2(-2.0, vector[1:], product[1:], a=reduced, lower=1, overwrite_a=1)
    return np.linalg.eigvalsh(reduced, UPLO="L")


def tunnelling_splitting(potential, masses, initial_path, imaginary_time_au, *,
                         equivalent_wells=False, active_mask=None,
                         endpoint_hessians=None, hessian_step_bohr=1e-4,
                         gradient_tolerance=1e-8, max_iterations=2000,
                         phase_tolerance=0.05):
    """Optimize one open kink and return its Gaussian ground-doublet splitting.

    Endpoints are fixed stationary minima; all interior beads are optimized.
    imaginary_time_au is the TOTAL open-path duration, not a temperature.
    equivalent_wells=True is an explicit assertion of symmetry equivalence;
    equal energies/frequencies alone do not establish this equivalence.

    This API supports bound or constrained multidimensional Cartesian models
    with positive-definite endpoint Hessians. It deliberately rejects free
    translations/rotations: rovibrational and multiwell permutation-symmetry
    splitting patterns need a separate treatment. It never derives a splitting
    by reinterpreting a thermal rate. Increase both duration and bead count
    independently to check the infinite-time and continuum limits.
    """
    if equivalent_wells is not True:
        raise ValueError("equivalent_wells=True must explicitly assert symmetry-equivalent wells")
    q = coordinates(initial_path).copy()
    if len(q) < 9:
        raise ValueError("an open splitting path requires at least 9 nodes including endpoints")
    mask, mass = _mask_and_masses(q.shape[1:], masses, active_mask)
    if not np.allclose(q[:, ~mask], q[0, ~mask], rtol=0, atol=1e-12):
        raise ValueError("inactive coordinates must be identical on the open path")
    if np.linalg.norm(q[-1, mask]-q[0, mask]) < 1e-8:
        raise ValueError("distinct well endpoints are required")
    duration = _positive(imaginary_time_au, "imaginary_time_au")
    step = _positive(hessian_step_bohr, "hessian_step_bohr")
    tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    phase_limit = _positive(phase_tolerance, "phase_tolerance")
    max_iterations = _integer(max_iterations, "max_iterations")
    intervals, d = len(q)-1, len(mass)
    n = intervals-1
    if n*d > 4096:
        raise ValueError("dense splitting Hessian is limited to 4096 interior coordinates")
    dtau = duration / intervals
    if endpoint_hessians is None:
        endpoint_hessians = (None, None)
    if len(endpoint_hessians) != 2:
        raise ValueError("endpoint_hessians must contain two Cartesian Hessians")
    wells = [HarmonicState.from_potential(potential, endpoint, masses, active_mask=mask,
                                         hessian=hessian, hessian_step_bohr=step,
                                         force_tolerance=tolerance)
             for endpoint, hessian in zip((q[0], q[-1]), endpoint_hessians)]
    left, right = [well.spectrum()[0] for well in wells]
    if not np.isclose(wells[0].energy_hartree, wells[1].energy_hartree, rtol=0, atol=1e-10):
        raise InstantonError("equivalent wells must have equal electronic energies")
    if not np.allclose(left, right, rtol=1e-5, atol=1e-12):
        raise InstantonError("equivalent wells must have matching harmonic spectra")
    if duration * math.sqrt(float(left[0])) < 10:
        raise InstantonError("imaginary-time duration too short to isolate the ground doublet")
    reference = wells[0].energy_hartree
    template = q.copy()

    def unpack(x):
        path = template.copy()
        path[1:-1, mask] = x.reshape(n, d)
        return path

    def evaluate(x):
        path = unpack(x)
        value = validate_result(potential.evaluate(path), path.shape)
        delta = np.diff(path[:, mask], axis=0)
        spring = float(np.sum(mass*delta**2)/(2*dtau))
        v = value.energy-reference
        action = spring + dtau*float(v[1:-1].sum()+0.5*(v[0]+v[-1]))
        gradient = (mass/dtau*(2*path[1:-1, mask]-path[:-2, mask]-path[2:, mask])
                    - dtau*value.forces[1:-1, mask])
        return action, gradient.ravel(), spring

    def hessian(x):
        physical = _physical_hessians(potential, unpack(x)[1:-1], mask, step)
        h = np.zeros((n, d, n, d))
        spring = np.diag(mass/dtau)
        for j in range(n):
            h[j, :, j, :] = dtau*physical[j] + 2*spring
            if j:
                h[j, :, j-1, :] = -spring
                h[j-1, :, j, :] = -spring
        return h.reshape(n*d, n*d)

    solution = minimize(lambda x: evaluate(x)[:2], q[1:-1, mask].ravel(), jac=True,
                        method="L-BFGS-B", options={"gtol": tolerance, "ftol": 1e-15,
                                                   "maxiter": max_iterations})
    x = solution.x
    if np.max(np.abs(evaluate(x)[1])) > tolerance:
        # Refine the nearly-flat phase direction without artificially pinning a
        # centroid or a particular barrier-crossing bead.
        refinement = root(lambda z: evaluate(z)[1], x, jac=hessian,
                          options={"xtol": 1e-10, "maxfev": max_iterations})
        x = refinement.x
    action, gradient, spring = evaluate(x)
    residual = float(np.max(np.abs(gradient)))
    if residual > tolerance or action <= 0 or spring <= 0:
        raise InstantonError("open instanton is not a stationary nontrivial kink")
    path = unpack(x)
    repeated_mass = np.tile(mass, n)
    operator = hessian(x) / (dtau*np.sqrt(repeated_mass[:, None]*repeated_mass[None, :]))
    tangent = ((path[2:, mask]-path[:-2, mask])*np.sqrt(mass)/(2*dtau)).ravel()
    tangent /= np.linalg.norm(tangent)
    scale = float(left[0])
    phase_residual = float(np.linalg.norm(operator@tangent)/scale)
    if phase_residual > phase_limit:
        raise InstantonError("open-path phase mode unresolved; increase duration and/or beads")
    spectrum = _phase_projected_spectrum(operator, tangent)
    if np.any(spectrum <= scale*1e-8):
        raise InstantonError("splitting kink must have no negative or extra zero fluctuation modes")
    # Dirichlet harmonic reference: use the SAME grid and number of modes.
    springs = (2*np.sin(np.pi*np.arange(1, intervals)/(2*intervals))/dtau)**2
    logdet_reference = float(np.log(springs[:, None]+left).sum())
    logdet_kink = float(np.log(spectrum).sum())
    log_ratio = 0.5*(logdet_reference-logdet_kink)
    # Norm of the translation collective coordinate: int m*qdot^2=2*S_spring.
    # It tends to S0 for the infinite-time zero-energy trajectory.
    log_prefactor = math.log(2) + 0.5*math.log(spring/math.pi) + log_ratio
    log_split = log_prefactor-action
    if log_split > math.log(sys.float_info.max):
        raise InstantonError("splitting exceeds floating-point range")
    splitting = math.exp(log_split)
    return {"method": "open_path_ground_doublet_instanton", "splitting_hartree": splitting,
            "splitting_wavenumber_cm^-1": splitting/WAVENUMBER_TO_AU,
            "log_splitting_hartree": log_split, "action_hbar": action,
            "log_prefactor_au": log_prefactor, "log_determinant_ratio": log_ratio,
            "gradient_max": residual, "phase_mode_relative_residual": phase_residual,
            "imaginary_time_au": duration, "nintervals": intervals,
            "path_bohr": path.tolist(), "numerical_converged": True,
            "physical_converged": False,
            "limitations": ["symmetry-equivalent bound/constrained wells; one tunnelling path",
                            "ground doublet, dilute-instanton semiclassical approximation",
                            "no free-rotor, asymmetric-well or multiwell symmetry pattern",
                            "independent duration, bead, Hessian and PES convergence required"]}
