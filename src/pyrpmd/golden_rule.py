"""Normal-regime golden-rule instantons for bound/constrained Cartesian models.

Atomic units (hbar=1) throughout, except temperature and reported SI rates.
Two open imaginary-time paths meet at two independent hopping coordinates;
there is no adiabatic imaginary-time phase mode to remove.  The coordinate
fluctuation Hessian must be positive definite and the optimized action must
have negative curvature in the time allocated to electronic state 1.

Richardson, J. Chem. Phys. 143, 134116 (2015), doi:10.1063/1.4932362,
Eqs. (4), (7), and (27)-(29).  Inverted-regime negative imaginary times and
free-molecule translation/rotation prefactors are deliberately not supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import brentq, minimize

from .constants import ATOMIC_TIME_S, KB_AU
from .instanton import (InstantonError, _CountingPotential, _mask_and_masses,
                       _physical_hessians, _positive)
from .potentials.base import coordinates, evaluate_checked
from .validation import finite_real, positive_integer


def _join_paths(path0, path1, masses, active_mask):
    paths = [coordinates(path0), coordinates(path1)]
    if paths[0].shape[1:] != paths[1].shape[1:]:
        raise ValueError("the two golden-rule paths must have the same atoms and Cartesian shape")
    if min(len(path) for path in paths) < 3:
        raise ValueError("each golden-rule segment requires at least two intervals (three nodes)")
    if (not np.allclose(paths[0][0], paths[1][-1], rtol=0, atol=1e-12)
            or not np.allclose(paths[0][-1], paths[1][0], rtol=0, atol=1e-12)):
        raise ValueError("golden-rule paths must share reverse endpoints: path0 a->b, path1 b->a")
    mask, mass = _mask_and_masses(paths[0].shape[1:], masses, active_mask)
    for path in paths:
        if not np.allclose(path[:, ~mask], paths[0][0, ~mask], rtol=0, atol=1e-12):
            raise ValueError("inactive coordinates must be identical on both golden-rule segments")
    n0, n1 = len(paths[0])-1, len(paths[1])-1
    if (n0+n1)*len(mass) > 4096:
        raise ValueError("dense golden-rule Hessian is limited to 4096 active path coordinates")
    return np.concatenate((paths[0][:-1], paths[1][:-1])), mask, mass, n0, n1


def _segments(q, n0, n1):
    return (q[:n0+1], np.concatenate((q[n0:], q[:1])))


class GoldenRuleAction:
    """Two trapezoidal open-path actions with a variable positive time split.

    ``tau1_au`` belongs to potential1; potential0 occupies ``beta-tau1_au``.
    A common ``energy_reference_hartree`` is subtracted from BOTH surfaces,
    so the total action shifts by exactly ``-beta*reference``.  A reactant
    partition supplied to the rate must use this same energy origin.
    """

    def __init__(self, potential0, potential1, masses, temperature_kelvin, *,
                 active_mask=None, energy_reference_hartree=0.0,
                 hessian_step_bohr=1e-4):
        for potential in (potential0, potential1):
            if not callable(getattr(potential, "evaluate", None)):
                raise ValueError("both golden-rule potentials must implement evaluate(x)")
        self.potentials = (potential0, potential1)
        self.masses = np.array(masses, copy=True)
        self.temperature_kelvin = _positive(temperature_kelvin, "temperature_kelvin")
        self.beta = 1/(KB_AU*self.temperature_kelvin)
        self.active_mask = active_mask
        self.energy_reference = finite_real(energy_reference_hartree, "energy_reference_hartree")
        self.hessian_step = _positive(hessian_step_bohr, "hessian_step_bohr")

    def _times(self, n0, n1, tau1):
        tau1 = finite_real(tau1, "tau1_au")
        if not 0 < tau1 < self.beta:
            raise ValueError("normal-regime golden-rule instantons require 0 < tau1 < beta")
        return (self.beta-tau1, tau1), ((self.beta-tau1)/n0, tau1/n1)

    def evaluate(self, path0, path1, tau1_au):
        q, mask, mass, n0, n1 = _join_paths(path0, path1, self.masses, self.active_mask)
        return self._evaluate(q, mask, mass, n0, n1, tau1_au)

    def _evaluate(self, q, mask, mass, n0, n1, tau1):
        times, steps = self._times(n0, n1, tau1)
        p = n0+n1
        gradient = np.zeros_like(q)
        mixed = np.zeros_like(q)
        parts = []
        mappings = (np.arange(n0+1), np.r_[np.arange(n0, p), 0])
        for potential, path, mapping, duration, dtau, sign in zip(
                self.potentials, _segments(q, n0, n1), mappings, times, steps, (-1, 1)):
            value = evaluate_checked(potential, path)
            weights = np.ones(len(path))
            weights[[0, -1]] = 0.5
            delta = np.diff(path[:, mask], axis=0)
            spring = float(np.sum(mass*delta**2)/(2*dtau))
            potential_action = float(dtau*np.dot(weights, value.energy-self.energy_reference))
            spring_gradient = np.zeros((len(path), len(mass)))
            spring_gradient[:-1] -= mass*delta/dtau
            spring_gradient[1:] += mass*delta/dtau
            potential_gradient = -dtau*weights[:, None]*value.forces[:, mask]
            g = np.zeros_like(path)
            g[:, mask] = spring_gradient+potential_gradient
            np.add.at(gradient, mapping, g)
            g[:, mask] = sign*(-spring_gradient+potential_gradient)/duration
            np.add.at(mixed, mapping, g)
            parts.append({"action_hbar": spring+potential_action,
                          "spring_action_hbar": spring,
                          "potential_action_hbar": potential_action,
                          "energy_hartree": (potential_action-spring)/duration,
                          "endpoint_energies_hartree": [float(value.energy[0]), float(value.energy[-1])]})
        return {"action_hbar": sum(part["action_hbar"] for part in parts),
                "gradient": gradient, "mixed_time_gradient": mixed,
                "energy_gap_hartree": parts[1]["energy_hartree"]-parts[0]["energy_hartree"],
                "time_hessian_partial_hartree2": sum(
                    2*part["spring_action_hbar"]/duration**2 for part, duration in zip(parts, times)),
                "hopping_energy_gaps_hartree": [parts[0]["endpoint_energies_hartree"][i]
                                                -parts[1]["endpoint_energies_hartree"][1-i]
                                                for i in (0, 1)],
                "hopping_distance_bohr": float(np.linalg.norm(q[0]-q[n0])),
                "segments": parts}

    def hessian(self, path0, path1, tau1_au):
        """Unweighted Cartesian coordinate Hessian; no phase is removed."""
        q, mask, mass, n0, n1 = _join_paths(path0, path1, self.masses, self.active_mask)
        return self._hessian(q, mask, mass, n0, n1, tau1_au)

    def _hessian(self, q, mask, mass, n0, n1, tau1):
        _, steps = self._times(n0, n1, tau1)
        p, d = n0+n1, len(mass)
        h = np.zeros((p, d, p, d))
        mappings = (np.arange(n0+1), np.r_[np.arange(n0, p), 0])
        for potential, path, mapping, dtau in zip(
                self.potentials, _segments(q, n0, n1), mappings, steps):
            physical = _physical_hessians(potential, path, mask, self.hessian_step)
            weights = np.ones(len(path))
            weights[[0, -1]] = 0.5
            spring = np.diag(mass/dtau)
            for i, bead in enumerate(mapping):
                h[bead, :, bead, :] += dtau*weights[i]*physical[i]
            for a, b in zip(mapping[:-1], mapping[1:]):
                h[a, :, a, :] += spring
                h[b, :, b, :] += spring
                h[a, :, b, :] -= spring
                h[b, :, a, :] -= spring
        return h.reshape(p*d, p*d)


def _spring_factor(n0, n1, tau0, tau1, beta):
    """P-by-P preconditioner, shared by every mass-weighted coordinate.

    The diagonal regularizer only makes this numerical change of variables
    invertible. It is never added to the physical action or its derivatives.
    """
    p = n0+n1
    matrix = np.eye(p)/beta
    edges = np.r_[np.full(n0, n0/tau0), np.full(n1, n1/tau1)]
    for a, weight in enumerate(edges):
        b = (a+1) % p
        matrix[a, a] += weight
        matrix[b, b] += weight
        matrix[a, b] -= weight
        matrix[b, a] -= weight
    return cholesky(matrix, lower=True, check_finite=False)


@dataclass
class GoldenRuleInstantonResult:
    path0_bohr: np.ndarray
    path1_bohr: np.ndarray
    temperature_kelvin: float
    tau0_au: float
    tau1_au: float
    action_hbar: float
    action0_hbar: float
    action1_hbar: float
    energy0_hartree: float
    energy1_hartree: float
    energy_gap_hartree: float
    gradient_max: float
    time_schur_curvature_hartree2: float
    time_hessian_partial_hartree2: float
    log_coordinate_prefactor: float
    coordinate_hessian_eigenvalues: np.ndarray
    energy_reference_hartree: float
    optimizer_diagnostics: dict
    verified_coupling_hartree: float | None = None
    hopping_energy_gaps_hartree: list | None = None
    hopping_distance_bohr: float = 0.0
    converged: bool = True

    @property
    def numerical_converged(self):
        return self.converged

    def to_dict(self, *, include_path=True):
        report = {key: value for key, value in vars(self).items()
                  if key not in ("path0_bohr", "path1_bohr", "coordinate_hessian_eigenvalues")}
        report.update(method="normal_regime_golden_rule_instanton",
                      nintervals0=len(self.path0_bohr)-1, nintervals1=len(self.path1_bohr)-1,
                      coordinate_hessian_eigenvalues=self.coordinate_hessian_eigenvalues.tolist(),
                      coordinate_negative_modes=0, removed_phase_modes=0, removed_rigid_modes=0,
                      unstable_time_modes=1, numerical_converged=self.converged,
                      scope="normal_regime_bound_or_constrained_condon_golden_rule")
        if include_path:
            report.update(path0_bohr=self.path0_bohr.tolist(), path1_bohr=self.path1_bohr.tolist())
        return report


def optimize_golden_rule_instanton(potential0, potential1, masses, temperature_kelvin,
                                   initial_path0, initial_path1, *,
                                   initial_tau_fraction=0.5,
                                   tau_fraction_bounds=(1e-4, 1-1e-4),
                                   tau_fraction_tolerance=1e-10, active_mask=None,
                                   energy_reference_hartree=0.0, hessian_step_bohr=1e-4,
                                   gradient_tolerance=1e-8, energy_tolerance_hartree=1e-10,
                                   max_iterations=2000, max_tau_iterations=100,
                                   regime="normal", rigid_body=False):
    """Optimize two paths and their time split, refreshing Hessians only as needed.

    At fixed positive time split the nuclear coordinates are minimized with
    spring-preconditioned L-BFGS. If energy changes reach roundoff before the
    requested gradient tolerance, up to twenty additional steps use its saved
    inverse Hessian and a gradient-norm line search. On actual stagnation a
    physical Hessian enables Newton refinement. These refinements share the
    same iteration budget and are counted explicitly. A strictly positive to
    negative E1-E0 bracket must establish a normal time saddle independently
    of the absolute residual tolerance. Brent then solves E1-E0=0, the saddle
    equation for tau1. The final true Hessian must be positive definite in
    coordinates and its time Schur complement negative. All conditions are
    checked independently of solver status. If time-root noise leaves the
    energy matching condition unresolved, the final Hessian and Schur
    complement provide a coupled coordinate/time Newton refinement, followed
    by a fresh final Hessian. No adiabatic phase projection or automatic
    inverted-regime continuation is applied.
    """
    if regime != "normal":
        raise ValueError("only the normal regime with two positive imaginary times is supported")
    if not isinstance(rigid_body, (bool, np.bool_)) or rigid_body:
        raise ValueError("golden-rule rigid_body must be False; free molecular rotations/translations are unsupported")
    tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    energy_tolerance = _positive(energy_tolerance_hartree, "energy_tolerance_hartree")
    time_tolerance = _positive(tau_fraction_tolerance, "tau_fraction_tolerance")
    iterations = positive_integer(max_iterations, "max_iterations")
    time_iterations = positive_integer(max_tau_iterations, "max_tau_iterations")
    fraction = finite_real(initial_tau_fraction, "initial_tau_fraction")
    bounds = np.asarray(tau_fraction_bounds)
    if (bounds.dtype.kind not in "fiu" or bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] < fraction < bounds[1] < 1):
        raise ValueError("tau_fraction_bounds must bracket initial_tau_fraction strictly inside (0,1)")
    if time_tolerance >= bounds[1]-bounds[0]:
        raise ValueError("tau_fraction_tolerance must be smaller than the time-fraction bracket")
    verified_coupling = None
    absent = object()
    for potential in (potential0, potential1):
        declared = getattr(potential, "coupling_hartree", absent)
        if declared is not absent:
            declared = abs(finite_real(declared, "surface coupling_hartree"))
            if declared == 0:
                raise ValueError("surface coupling_hartree must be nonzero")
            if verified_coupling is not None and declared != verified_coupling:
                raise ValueError("golden-rule surfaces declare inconsistent coupling magnitudes")
            verified_coupling = declared
    counted = [_CountingPotential(potential0), _CountingPotential(potential1)]
    action = GoldenRuleAction(*counted, masses, temperature_kelvin, active_mask=active_mask,
                              energy_reference_hartree=energy_reference_hartree,
                              hessian_step_bohr=hessian_step_bohr)
    q, mask, mass, n0, n1 = _join_paths(initial_path0, initial_path1, masses, active_mask)
    p, d = n0+n1, len(mass)
    template = q.copy()
    sqrt_mass = np.sqrt(mass)
    cache, history = {}, []

    def fixed_time(f):
        f = float(f)
        if f in cache:
            return cache[f]
        tau1 = action.beta*f
        factor = _spring_factor(n0, n1, action.beta-tau1, tau1, action.beta)
        seed = q if not cache else cache[min(cache, key=lambda t: abs(t-f))][0]
        z0 = (factor.T @ (seed[:, mask]*sqrt_mass)).ravel()
        last = {}

        def objective(z):
            if "z" in last and np.array_equal(z, last["z"]):
                return last["value"]
            path = template.copy()
            path[:, mask] = solve_triangular(factor.T, z.reshape(p, d), lower=False,
                                             check_finite=False)/sqrt_mass
            value = action._evaluate(path, mask, mass, n0, n1, tau1)
            gradient = solve_triangular(factor, value["gradient"][:, mask]/sqrt_mass,
                                        lower=True, check_finite=False).ravel()
            pair = value["action_hbar"], gradient
            last.update(z=np.array(z, copy=True), path=path, evaluation=value, value=pair)
            return pair

        # Bound ||g_cart||_inf by max(sqrt(m))*||L||_inf*||g_z||_inf.
        scaled_tolerance = tolerance/(float(sqrt_mass.max())*float(np.abs(factor).sum(axis=1).max()))
        solution = minimize(objective, z0, method="L-BFGS-B", jac=True,
                            options={"gtol": scaled_tolerance, "ftol": 0.0,
                                     "maxiter": iterations, "maxls": 50, "maxcor": 20})
        objective(solution.x)
        z = np.array(solution.x, copy=True)
        polish_steps = polish_evaluations = 0
        for _ in range(min(20, max(0, iterations-int(solution.nit)))):
            residual = float(np.max(np.abs(last["evaluation"]["gradient"][:, mask])))
            if residual <= tolerance:
                break
            gradient = last["value"][1].copy()
            direction = -solution.hess_inv @ gradient
            old_norm = float(gradient @ gradient)
            accepted = False
            for backtrack in range(16):
                candidate = z+(0.5**backtrack)*direction
                _, new_gradient = objective(candidate)
                polish_evaluations += 1
                new_residual = float(np.max(np.abs(last["evaluation"]["gradient"][:, mask])))
                if float(new_gradient @ new_gradient) < old_norm or new_residual <= tolerance:
                    z = candidate
                    polish_steps += 1
                    accepted = True
                    break
            if not accepted:
                objective(z)
                polish_evaluations += 1
                break
        newton_steps = newton_evaluations = 0
        remaining = iterations-int(solution.nit)-polish_steps
        residual = float(np.max(np.abs(last["evaluation"]["gradient"][:, mask])))
        if residual > tolerance and remaining > 0:
            physical_hessian = action._hessian(last["path"], mask, mass, n0, n1, tau1)
            try:
                newton_factor = cho_factor(physical_hessian, lower=True, check_finite=False)
            except np.linalg.LinAlgError as exc:
                raise InstantonError("normal-regime coordinate refinement requires a positive Hessian; "
                                     "coordinate saddles and free rigid modes are unsupported") from exc
            for _ in range(min(5, remaining)):
                current = last["path"].copy()
                gradient = last["evaluation"]["gradient"][:, mask].ravel().copy()
                if float(np.max(np.abs(gradient))) <= tolerance:
                    break
                delta = cho_solve(newton_factor, -gradient, check_finite=False).reshape(p, d)
                old_norm = float(gradient @ gradient)
                accepted = False
                for backtrack in range(16):
                    candidate = (factor.T @ ((current[:, mask]+(0.5**backtrack)*delta)*sqrt_mass)).ravel()
                    objective(candidate)
                    newton_evaluations += 1
                    new_gradient = last["evaluation"]["gradient"][:, mask].ravel()
                    if float(new_gradient @ new_gradient) < old_norm or np.max(np.abs(new_gradient)) <= tolerance:
                        z = candidate
                        newton_steps += 1
                        accepted = True
                        break
                if not accepted:
                    objective(z)
                    newton_evaluations += 1
                    break
        path, value = last["path"], last["evaluation"]
        residual = float(np.max(np.abs(value["gradient"][:, mask])))
        if residual > tolerance:
            raise InstantonError(f"golden-rule coordinate minimization failed at tau1/beta={f:.8g}: "
                                 f"gradient {residual:.3g} exceeds {tolerance:.3g} ({solution.message})")
        row = {"tau1_fraction": f, "action_hbar": value["action_hbar"],
               "energy_gap_hartree": value["energy_gap_hartree"], "gradient_max": residual,
               "iterations": int(solution.nit)+polish_steps+newton_steps,
               "lbfgs_iterations": int(solution.nit),
               "gradient_polishing_steps": polish_steps,
               "gradient_polishing_evaluations": polish_evaluations,
               "physical_hessian_refinement_steps": newton_steps,
               "physical_hessian_refinement_evaluations": newton_evaluations,
               "gradient_evaluations": int(solution.nfev)+polish_evaluations+newton_evaluations}
        history.append(row)
        cache[f] = path, value
        return path, value

    def gap(f):
        return fixed_time(f)[1]["energy_gap_hartree"]

    initial_gap = gap(fraction)
    root_iterations = 0
    width = min(0.1, float(bounds[1]-bounds[0])/4)
    while True:
        lower, upper = max(float(bounds[0]), fraction-width), min(float(bounds[1]), fraction+width)
        g_lower, g_upper = gap(lower), gap(upper)
        # Existence of a normal-regime maximum is independent of an absolute
        # stopping tolerance. A weak inverted model can have |gap| below that
        # tolerance everywhere; neither a small gap nor clipping establishes
        # a root. Direct sign comparisons also avoid product underflow.
        if g_lower > 0 and g_upper < 0:
            if initial_gap != 0.0:
                fraction, solved = brentq(gap, lower, upper, xtol=time_tolerance,
                                          rtol=4*np.finfo(float).eps, maxiter=time_iterations,
                                          full_output=True, disp=False)
                root_iterations = int(solved.iterations)
                if not solved.converged:
                    raise InstantonError("golden-rule time allocation did not converge")
            break
        if lower == bounds[0] and upper == bounds[1]:
            raise InstantonError("no stationary time split with resolved normal-regime signs in the "
                                 "requested positive-time bracket; tighten coordinate/energy tolerances "
                                 "or widen the positive bracket. This method does not continue into "
                                 "the Marcus inverted regime or accept an unresolved time direction")
        width *= 2
    path, evaluation = fixed_time(fraction)
    tau1, tau0 = action.beta*fraction, action.beta*(1-fraction)
    repeated_mass = np.tile(mass, p)
    coupled_steps = 0
    while True:
        hessian = action._hessian(path, mask, mass, n0, n1, tau1)
        weighted = hessian/np.sqrt(repeated_mass[:, None]*repeated_mass[None, :])
        eigenvalues, vectors = np.linalg.eigh(weighted)
        cutoff = max(float(np.max(np.abs(eigenvalues)))*1e-10, np.finfo(float).tiny)
        if eigenvalues[0] <= cutoff:
            raise InstantonError("golden-rule coordinate Hessian must be positive definite; "
                                 "free rigid/extra zero modes and coordinate saddles are unsupported")
        mixed = evaluation["mixed_time_gradient"][:, mask].ravel()/np.sqrt(repeated_mass)
        projection = vectors.T @ mixed
        relaxation = float(np.sum(projection**2/eigenvalues))
        partial = evaluation["time_hessian_partial_hartree2"]
        schur = partial-relaxation
        if not math.isfinite(schur) or schur >= -max(abs(partial), relaxation, np.finfo(float).tiny)*1e-10:
            raise InstantonError("golden-rule time Schur curvature must be resolved and negative; "
                                 "degenerate or non-normal stationary solutions are unsupported")
        cartesian_gradient = evaluation["gradient"][:, mask].ravel()
        energy_gap = evaluation["energy_gap_hartree"]
        if np.max(np.abs(cartesian_gradient)) <= tolerance and abs(energy_gap) <= energy_tolerance:
            break
        if coupled_steps >= min(5, time_iterations-root_iterations):
            raise InstantonError("golden-rule coupled time/coordinate refinement failed; "
                                 "segment energies and gradients must satisfy both requested tolerances")
        gradient_projection = vectors.T @ (cartesian_gradient/np.sqrt(repeated_mass))
        delta_time = (-energy_gap+float(np.sum(projection*gradient_projection/eigenvalues)))/schur
        delta_coordinates = (-vectors @ ((gradient_projection+projection*delta_time)/eigenvalues)
                             /np.sqrt(repeated_mass)).reshape(p, d)
        old_merit = float(np.sum((cartesian_gradient/tolerance)**2)+(energy_gap/energy_tolerance)**2)
        accepted = False
        for backtrack in range(16):
            factor = 0.5**backtrack
            proposed_time = tau1+factor*delta_time
            if not bounds[0] < proposed_time/action.beta < bounds[1]:
                continue
            proposed_path = path.copy()
            proposed_path[:, mask] += factor*delta_coordinates
            proposed = action._evaluate(proposed_path, mask, mass, n0, n1, proposed_time)
            merit = float(np.sum((proposed["gradient"][:, mask]/tolerance)**2)
                          +(proposed["energy_gap_hartree"]/energy_tolerance)**2)
            if merit < old_merit:
                path, evaluation, tau1 = proposed_path, proposed, proposed_time
                tau0 = action.beta-tau1
                coupled_steps += 1
                accepted = True
                break
        if not accepted:
            raise InstantonError("golden-rule coupled coordinate/time Newton step did not reduce the residual")
    log_prefactor = -0.5*(d*(n0*math.log(tau0/n0)+n1*math.log(tau1/n1))
                          +float(np.log(eigenvalues).sum()))
    total = {key: sum(provider.statistics[key] for provider in counted) for key in counted[0].statistics}
    diagnostics = dict(total, method="fixed_time_L-BFGS-B_with_Brent_time_saddle",
                       surfaces=[dict(provider.statistics) for provider in counted],
                       time_evaluations=len(history), time_root_iterations=root_iterations,
                       coordinate_iterations=sum(row["iterations"] for row in history),
                       coordinate_hessian_refinement_steps=sum(row["physical_hessian_refinement_steps"] for row in history),
                       coupled_refinement_steps=coupled_steps,
                       history=history, spring_preconditioner_dimension=p,
                       coordinate_hessian_dimension=p*d, coordinate_negative_modes=0,
                       removed_phase_modes=0, removed_rigid_modes=0, unstable_time_modes=1,
                       time_fraction_bounds=bounds.astype(float).tolist(),
                       resolved_time_bracket={"lower_fraction": lower, "upper_fraction": upper,
                                              "lower_energy_gap_hartree": g_lower,
                                              "upper_energy_gap_hartree": g_upper})
    path0, path1 = _segments(path, n0, n1)
    parts = evaluation["segments"]
    return GoldenRuleInstantonResult(
        path0.copy(), path1.copy(), action.temperature_kelvin, tau0, tau1,
        evaluation["action_hbar"], parts[0]["action_hbar"], parts[1]["action_hbar"],
        parts[0]["energy_hartree"], parts[1]["energy_hartree"],
        evaluation["energy_gap_hartree"], float(np.max(np.abs(evaluation["gradient"][:, mask]))),
        schur, partial, log_prefactor, eigenvalues, action.energy_reference, diagnostics,
        verified_coupling_hartree=verified_coupling,
        hopping_energy_gaps_hartree=evaluation["hopping_energy_gaps_hartree"],
        hopping_distance_bohr=evaluation["hopping_distance_bohr"])


def golden_rule_rate(result, *, coupling_hartree, log_reactant_partition,
                     reaction_degeneracy=1):
    """Golden-rule SCI rate in s^-1 for an explicit dimensionless reactant Q.

    ``log_reactant_partition`` includes the reactant electronic-energy
    Boltzmann factor with the same energy reference as the optimized action.
    The constant Condon coupling is a nonzero real number; its sign does not
    affect the rate. Weak-coupling validity is an independent model assumption.
    """
    if not isinstance(result, GoldenRuleInstantonResult) or not result.converged:
        raise InstantonError("a converged normal-regime golden-rule instanton is required")
    coupling = finite_real(coupling_hartree, "coupling_hartree")
    if coupling == 0:
        raise ValueError("coupling_hartree must be a nonzero real constant")
    if (result.verified_coupling_hartree is not None
            and abs(coupling) != result.verified_coupling_hartree):
        raise ValueError("coupling_hartree magnitude differs from the coupling verified by the surfaces")
    log_qr = finite_real(log_reactant_partition, "log_reactant_partition")
    degeneracy = positive_integer(reaction_degeneracy, "reaction_degeneracy")
    if result.time_schur_curvature_hartree2 >= 0:
        raise InstantonError("negative golden-rule time Schur curvature required")
    log_rate = (2*math.log(abs(coupling))+0.5*math.log(2*math.pi)
                -0.5*math.log(-result.time_schur_curvature_hartree2)
                +result.log_coordinate_prefactor-result.action_hbar-log_qr+math.log(degeneracy))
    log_si = log_rate-math.log(ATOMIC_TIME_S)
    if not math.isfinite(log_si):
        raise InstantonError("nonfinite logarithmic golden-rule rate")
    value = math.exp(log_si) if log_si <= math.log(sys.float_info.max) else None
    return {"method": "normal_regime_golden_rule_instanton", "rate": value,
            "rate_unit": "s^-1", "log_rate_au": log_rate, "log_rate_si": log_si,
            "range_status": "overflow" if value is None else "underflow" if value == 0 else "finite",
            "coupling_hartree": coupling, "log_reactant_partition": log_qr,
            "verified_coupling_hartree": result.verified_coupling_hartree,
            "reactant_partition_normalization": "dimensionless",
            "reaction_degeneracy": degeneracy, "temperature_kelvin": result.temperature_kelvin,
            "energy_reference_hartree": result.energy_reference_hartree,
            "tau0_au": result.tau0_au, "tau1_au": result.tau1_au,
            "action_hbar": result.action_hbar,
            "time_schur_curvature_hartree2": result.time_schur_curvature_hartree2,
            "log_coordinate_prefactor": result.log_coordinate_prefactor,
            "numerical_converged": True,
            "limitations": ["leading golden-rule semiclassical approximation; weak real constant Condon coupling",
                            "bound/constrained dimensionless reactant partition; no free rigid modes",
                            "positive imaginary times only; Marcus inverted regime unsupported",
                            "independent convergence of both segment grids, PES and reactant partition required"]}
