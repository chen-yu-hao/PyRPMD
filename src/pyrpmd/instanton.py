"""Thermal ring-polymer instanton paths and a one-dimensional rate cross-check.

The Cartesian path solver is a stationary-action calculation, not molecular
dynamics.  Its folded half-ring representation fixes imaginary-time phase; the
full-ring Hessian is retained for mode diagnostics.  Quantitative rates in this
module are restricted to one active Cartesian degree of freedom, with an
explicit reactant partition function.  Multidimensional action optimization
does not by itself supply the transverse fluctuation/rotation prefactor.

References: Richardson & Althorpe, JCP 131, 214106 (2009),
doi:10.1063/1.3267318; Richardson, JCP 144, 114106 (2016),
doi:10.1063/1.4943866.  All inputs use bohr, electron masses, hartree,
and kelvin; hbar=1 in atomic units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import root

from .constants import ATOMIC_TIME_S, BOHR_M, KB_AU
from .potentials.base import coordinates, validate_result
from .validation import finite_positive


class InstantonError(RuntimeError):
    """The requested stationary orbit or its rate prefactor is not valid."""


_positive = partial(
    finite_positive,
    shape="zero_dim",
    message="{name} must be a positive finite scalar",
)


def _mask_and_masses(shape, masses, active_mask):
    raw = np.asarray(masses)
    if raw.dtype.kind not in "fiu" or raw.shape != (shape[0],):
        raise ValueError("masses must be a real electron-mass array [atoms]")
    masses = raw.astype(float)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("masses must be positive and finite")
    if active_mask is None:
        mask = np.ones(shape, dtype=bool)
    else:
        mask = np.asarray(active_mask)
        if mask.dtype.kind != "b" or mask.shape != shape or not mask.any():
            raise ValueError("active_mask must be a nonempty boolean [atoms,3] array")
        mask = mask.copy()
    return mask, np.broadcast_to(masses[:, None], shape)[mask]


def instanton_seed(transition_state, direction, nbeads, amplitude_bohr):
    """Build a folded cosine seed around a TS, along a Cartesian direction.

    ``amplitude_bohr`` is the Euclidean norm of the endpoint displacement.
    A nonzero starting spread is essential: the collapsed TS is also a root.
    ``direction`` is usually the unstable mass-weighted TS mode transformed
    back to Cartesian coordinates.  It is normalized internally.
    """
    ts = coordinates(np.asarray(transition_state)[None])[0]
    d = np.asarray(direction)
    if d.dtype.kind not in "fiu" or d.shape != ts.shape or not np.isfinite(d).all():
        raise ValueError("direction must be a finite real [atoms,3] array")
    norm = float(np.linalg.norm(d))
    if norm == 0:
        raise ValueError("direction must be nonzero")
    if isinstance(nbeads, (bool, np.bool_)) or not isinstance(nbeads, (int, np.integer)) or nbeads < 8 or nbeads % 2:
        raise ValueError("nbeads must be an even integer >= 8")
    a = _positive(amplitude_bohr, "amplitude_bohr")
    # Reflection through two spring links pairs both turning-point beads.
    # This is the conventional P/2-bead folded thermal-instanton representation.
    return ts + a * np.cos(2 * np.pi * (np.arange(nbeads) + 0.5) / nbeads)[:, None, None] * d / norm


def _physical_hessians(potential, q, mask, step):
    """Finite differences of physical forces; all beads evaluated together."""
    ids = np.argwhere(mask)
    h = np.empty((len(q), len(ids), len(ids)))
    for j, (atom, component) in enumerate(ids):
        plus, minus = q.copy(), q.copy()
        plus[:, atom, component] += step
        minus[:, atom, component] -= step
        # Plugins may reuse their force buffer; preserve it before the next call.
        fp = validate_result(potential.evaluate(plus), plus.shape).forces[:, mask].copy()
        fm = validate_result(potential.evaluate(minus), minus.shape).forces[:, mask].copy()
        h[:, :, j] = (fm - fp) / (2 * step)
    return 0.5 * (h + h.transpose(0, 2, 1))


def crossover_temperature(potential, transition_state, masses, *, active_mask=None,
                          hessian_step_bohr=1e-4, force_tolerance=1e-6):
    """Return Tc in kelvin for a stationary first-order barrier.

    The active-space mass-weighted PES Hessian must have one negative mode.
    Rigid translations/rotations can appear as zero modes. ``active_mask``
    freezes Cartesian components and defines a constrained model; it is not
    a general projector onto rigid modes. This test concerns the supplied
    geometry, not TS optimization. Tc = |omega_b| / (2*pi*kB) in atomic units.
    """
    q = coordinates(np.asarray(transition_state)[None])
    mask, mass = _mask_and_masses(q.shape[1:], masses, active_mask)
    step = _positive(hessian_step_bohr, "hessian_step_bohr")
    tolerance = _positive(force_tolerance, "force_tolerance")
    result = validate_result(potential.evaluate(q), q.shape)
    if np.max(np.abs(result.forces[:, mask])) > tolerance:
        raise InstantonError("transition_state is not stationary in the active coordinates")
    h = _physical_hessians(potential, q, mask, step)[0]
    eig = np.linalg.eigvalsh(h / np.sqrt(mass[:, None] * mass[None, :]))
    cutoff = max(float(np.max(np.abs(eig))) * 1e-7, np.finfo(float).tiny)
    negative = eig[eig < -cutoff]
    if len(negative) != 1:
        raise InstantonError("transition_state must have exactly one negative active Hessian mode")
    return float(np.sqrt(-negative[0]) / (2 * np.pi * KB_AU))


class RingPolymerAction:
    """Primitive Euclidean action for an even, closed Cartesian ring.

    S/hbar = sum_j [m (q_j-q_{j-1})^2 / (2*beta_P) + beta_P*V(q_j)],
    beta_P=1/(P*kB*T).  This differs from an RPMD real-time Hamiltonian.
    ``energy_reference_hartree`` shifts the energy zero consistently in S and
    E; any reactant partition function must use that same shifted zero.
    """

    def __init__(self, potential, masses, temperature_kelvin, *, active_mask=None,
                 energy_reference_hartree=0.0, hessian_step_bohr=1e-4):
        self.potential = potential
        self.masses = np.asarray(masses).copy()
        self.temperature_kelvin = _positive(temperature_kelvin, "temperature_kelvin")
        self.beta = 1.0 / (KB_AU * self.temperature_kelvin)
        self.active_mask = active_mask
        self.energy_reference = float(energy_reference_hartree)
        if not np.isfinite(self.energy_reference):
            raise ValueError("energy_reference_hartree must be finite")
        self.hessian_step = _positive(hessian_step_bohr, "hessian_step_bohr")

    def _validate(self, path):
        q = coordinates(path)
        if len(q) < 8 or len(q) % 2:
            raise ValueError("an instanton path requires an even number of beads >= 8")
        mask, mass = _mask_and_masses(q.shape[1:], self.masses, self.active_mask)
        if not np.allclose(q[:, ~mask], q[0, ~mask], rtol=0, atol=1e-12):
            raise ValueError("inactive coordinates must be identical on every bead")
        return q, mask, mass

    def evaluate(self, path):
        """Return action, full Cartesian gradient, E=dS/dbeta and its parts."""
        q, mask, mass = self._validate(path)
        p = len(q)
        dtau = self.beta / p
        value = validate_result(self.potential.evaluate(q), q.shape)
        v = value.energy - self.energy_reference
        active = q[:, mask]
        difference = active - np.roll(active, 1, axis=0)
        spring = float(np.sum(mass * difference**2) / (2 * dtau))
        potential = float(dtau * np.sum(v))
        gradient = np.zeros_like(q)
        gradient[:, mask] = (mass / dtau * (2 * active - np.roll(active, 1, axis=0)
                                          - np.roll(active, -1, axis=0))
                             - dtau * value.forces[:, mask])
        energy = float(np.mean(v) - spring / self.beta)
        return {"action_hbar": spring + potential, "gradient": gradient,
                "energy_hartree": energy, "spring_action_hbar": spring,
                "potential_action_hbar": potential}

    def hessian(self, path):
        """Dense action Hessian in bead-major active Cartesian coordinates."""
        q, mask, mass = self._validate(path)
        p, d = len(q), len(mass)
        if p * d > 4096:
            raise ValueError("dense instanton Hessian is limited to 4096 active bead coordinates")
        dtau = self.beta / p
        h = np.zeros((p, d, p, d))
        physical = _physical_hessians(self.potential, q, mask, self.hessian_step)
        spring = np.diag(mass / dtau)
        for j in range(p):
            h[j, :, j, :] += dtau * physical[j] + 2 * spring
            h[j, :, (j - 1) % p, :] -= spring
            h[j, :, (j + 1) % p, :] -= spring
        return h.reshape(p * d, p * d)


@dataclass
class InstantonResult:
    path_bohr: np.ndarray
    temperature_kelvin: float
    action_hbar: float
    energy_hartree: float
    spring_action_hbar: float
    potential_action_hbar: float
    gradient_max: float
    spread_bohr: float
    negative_modes: int
    near_zero_modes: int
    phase_mode_relative_residual: float
    hessian_eigenvalues: np.ndarray
    converged: bool
    function_evaluations: int
    message: str

    def to_dict(self, *, include_path=True):
        output = {key: value for key, value in vars(self).items()
                  if key not in ("path_bohr", "hessian_eigenvalues")}
        output.update(nbeads=len(self.path_bohr),
                      hessian_eigenvalues=self.hessian_eigenvalues.tolist(),
                      scope="stationary_action_path_not_a_multidimensional_rate",
                      numerical_converged=self.converged,
                      physical_converged=False)
        if include_path:
            output["path_bohr"] = self.path_bohr.tolist()
        return output


def optimize_instanton(potential, masses, temperature_kelvin, initial_path, *,
                      active_mask=None, transition_state=None,
                      energy_reference_hartree=0.0, hessian_step_bohr=1e-4,
                      gradient_tolerance=1e-8, max_iterations=200,
                      minimum_spread_bohr=1e-5, raise_on_failure=True):
    """Find a folded periodic stationary orbit from a noncollapsed path seed.

    ``initial_path[P,N,3]`` must obey link reflection q[j]=q[P-1-j],
    as produced by ``instanton_seed``. Bead reflection q[j]=q[P-j] is also
    accepted, but may have an extra unstable discretized phase at small P.
    The half-ring stationarity equations are solved with their analytic spring
    and finite-difference physical Hessian.  This removes the phase freedom
    without adding a centroid constraint that would distort asymmetric barriers.
    A root must be noncollapsed, stationary, and have exactly one unstable
    full-ring mode.  Optional ``transition_state`` also rejects T >= Tc.

    This is a local solver: different paths can exist, and an appropriate seed,
    bead study and independent PES validation remain necessary.  The default
    raises on failure.  ``raise_on_failure=False`` returns diagnostics only.
    ``max_iterations`` bounds the solver's residual-evaluation budget.
    """
    action = RingPolymerAction(potential, masses, temperature_kelvin,
                              active_mask=active_mask,
                              energy_reference_hartree=energy_reference_hartree,
                              hessian_step_bohr=hessian_step_bohr)
    q, mask, mass = action._validate(initial_path)
    tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    minimum_spread = _positive(minimum_spread_bohr, "minimum_spread_bohr")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    p, d = len(q), len(mass)
    half = p // 2
    mapping = np.concatenate((np.arange(half), np.arange(half - 1, -1, -1)))
    if not np.allclose(q, q[:half][mapping], atol=1e-9, rtol=1e-9):
        half = p // 2 + 1
        mapping = np.concatenate((np.arange(half), np.arange(half - 2, 0, -1)))
        if not np.allclose(q, q[:half][mapping], atol=1e-9, rtol=1e-9):
            raise ValueError("initial_path must be folded under link or bead reflection")
    if float(np.max(np.ptp(q[:, mask], axis=0))) <= minimum_spread:
        raise InstantonError("initial path is collapsed; use a nonzero instanton_seed")
    if transition_state is not None:
        ts_geometry = coordinates(np.asarray(transition_state)[None])[0]
        if ts_geometry.shape != q.shape[1:] or not np.allclose(ts_geometry[~mask], q[0, ~mask], rtol=0, atol=1e-12):
            raise ValueError("transition_state inactive coordinates must match the constrained path slice")
        tc = crossover_temperature(potential, ts_geometry, masses,
                                   active_mask=mask, hessian_step_bohr=hessian_step_bohr)
        if action.temperature_kelvin >= tc:
            raise InstantonError(f"thermal instanton requires temperature below crossover ({tc:.8g} K)")
    template = q.copy()
    multiplicity = np.bincount(mapping, minlength=half)
    # Normalize each row by its multiplicity.  The equations still find exactly
    # the same stationary point; endpoint and interior residuals are comparable.
    def unpack(x):
        path = template.copy()
        path[:, mask] = x.reshape(half, d)[mapping]
        return path

    def residual(x):
        g = action.evaluate(unpack(x))["gradient"][:, mask]
        reduced = np.zeros((half, d))
        np.add.at(reduced, mapping, g)
        return (reduced / multiplicity[:, None]).ravel()

    def jacobian(x):
        h = action.hessian(unpack(x)).reshape(p, d, p, d)
        reduced = np.zeros((half, d, half, d))
        for j in range(p):
            for k in range(p):
                reduced[mapping[j], :, mapping[k], :] += h[j, :, k, :]
        reduced /= multiplicity[:, None, None, None]
        return reduced.reshape(half * d, half * d)

    solution = root(residual, q[:half, mask].ravel(), jac=jacobian, method="hybr",
                    options={"xtol": min(tolerance, 1e-9), "maxfev": int(max_iterations)})
    path = unpack(solution.x)
    evaluation = action.evaluate(path)
    h = action.hessian(path)
    repeated_mass = np.tile(mass, p)
    hmw = h / np.sqrt(repeated_mass[:, None] * repeated_mass[None, :])
    eigenvalues = np.linalg.eigvalsh(hmw)
    # Scaling by the smallest curvature, rather than the largest spring mode,
    # prevents large P from hiding the true unstable mode.
    curvature_scale = max(abs(float(eigenvalues[0])), np.finfo(float).tiny)
    mode_cutoff = curvature_scale * 1e-5
    negative = int(np.count_nonzero(eigenvalues < -mode_cutoff))
    near_zero = int(np.count_nonzero(np.abs(eigenvalues) <= mode_cutoff))
    frequencies = np.fft.fftfreq(p)
    tangent = np.fft.ifft(2j * np.pi * frequencies[:, None]
                         * np.fft.fft(path[:, mask], axis=0), axis=0).real
    tangent = (tangent * np.sqrt(mass)).ravel()
    tangent_norm = float(np.linalg.norm(tangent))
    phase_residual = (float(np.linalg.norm(hmw @ tangent)) /
                      max(curvature_scale * tangent_norm, np.finfo(float).tiny))
    spread = float(np.max(np.ptp(path[:, mask], axis=0)))
    gradient_max = float(np.max(np.abs(evaluation["gradient"][:, mask])))
    # A phase eigenvalue is only approximately zero on a finite primitive ring.
    # Report its residual for P convergence; do not call it an exact zero mode.
    converged = bool(gradient_max <= tolerance and spread > minimum_spread and negative == 1)
    reasons = []
    if gradient_max > tolerance:
        reasons.append(f"stationary residual {gradient_max:.3g} exceeds tolerance")
    if spread <= minimum_spread:
        reasons.append("orbit collapsed")
    if negative != 1:
        reasons.append(f"expected one negative mode, found {negative}")
    result = InstantonResult(
        path, action.temperature_kelvin, evaluation["action_hbar"],
        evaluation["energy_hartree"], evaluation["spring_action_hbar"],
        evaluation["potential_action_hbar"], gradient_max, spread, negative,
        near_zero, phase_residual, eigenvalues, converged, int(solution.nfev),
        "; ".join(reasons) if reasons else "stationary noncollapsed first-order orbit",
    )
    if not converged and raise_on_failure:
        raise InstantonError(result.message)
    return result


def instanton_rate_1d(potential, masses, temperature_kelvin, initial_path, *,
                     active_mask, reactant_partition, normalization="dimensionless",
                     beta_relative_step=0.01, **optimizer_options):
    """Compute a 1D canonical steepest-descent instanton rate cross-check.

    k*Qr = sqrt[-(dE/dbeta)/(2*pi)] * exp(-S), in atomic units.
    ``reactant_partition`` must be positive, dimensionless for a bound reactant
    (rate in s^-1), or per bohr for a 1D scattering entrance (velocity in m/s).
    It must use the same energy zero as ``energy_reference_hartree``.  No 3D
    bimolecular standard-state conversion or transverse prefactor is inferred.

    E=dS/dbeta is differentiated using optimized orbits at beta*(1+-h).
    h and h/2 are both evaluated to report derivative-discretization error;
    bead and semiclassical error are independent.  This approximation is not
    uniform close to crossover and is not an exact quantum rate.
    """
    q = coordinates(initial_path)
    mask, _ = _mask_and_masses(q.shape[1:], masses, active_mask)
    if int(mask.sum()) != 1:
        raise ValueError("instanton_rate_1d requires exactly one active Cartesian coordinate")
    partition = _positive(reactant_partition, "reactant_partition")
    h = _positive(beta_relative_step, "beta_relative_step")
    if h >= 0.1:
        raise ValueError("beta_relative_step must be smaller than 0.1")
    if normalization not in ("dimensionless", "per_bohr"):
        raise ValueError("normalization must be dimensionless or per_bohr")
    if "raise_on_failure" in optimizer_options:
        raise ValueError("rate calculation requires successful orbit optimization")
    central = optimize_instanton(potential, masses, temperature_kelvin, q,
                                active_mask=mask, **optimizer_options)
    beta = 1.0 / (KB_AU * central.temperature_kelvin)
    energies = []
    derivatives = []
    for step in (h, h / 2):
        pair = []
        for sign in (-1, 1):
            nearby = optimize_instanton(potential, masses, central.temperature_kelvin / (1 + sign * step),
                                       central.path_bohr, active_mask=mask, **optimizer_options)
            pair.append(nearby.energy_hartree)
        energies.append(pair)
        derivatives.append((pair[1] - pair[0]) / (2 * step * beta))
    derivative = derivatives[-1]
    if not np.isfinite(derivative) or derivative >= 0:
        raise InstantonError("one-dimensional instanton requires dE/dbeta < 0")
    relative_error = abs(derivatives[1] - derivatives[0]) / abs(derivative)
    if relative_error > 0.05:
        raise InstantonError("period-energy derivative is not resolved; refine beta step and orbit tolerance")
    log_prefactor = 0.5 * math.log(-derivative / (2 * np.pi))
    log_rate = log_prefactor - central.action_hbar - math.log(partition)
    conversion = 1 / ATOMIC_TIME_S if normalization == "dimensionless" else BOHR_M / ATOMIC_TIME_S
    log_rate_si = log_rate + math.log(conversion)
    if log_rate_si > math.log(np.finfo(float).max):
        raise InstantonError("rate exceeds floating-point range; inspect partition normalization")
    return {
        "method": "one_dimensional_canonical_steepest_descent_instanton",
        "temperature_kelvin": central.temperature_kelvin,
        "rate": float(math.exp(log_rate_si)),
        "rate_unit": "s^-1" if normalization == "dimensionless" else "m/s",
        "log_rate_au": log_rate,
        "log_rate_si": log_rate_si,
        "reactant_partition": partition,
        "reactant_partition_normalization": normalization,
        "d_energy_d_beta_hartree2": derivative,
        "beta_derivative_relative_difference": relative_error,
        "beta_relative_step": h / 2,
        "numerical_converged": central.converged,
        "physical_converged": False,
        "instanton": central.to_dict(),
        "limitations": ["one active Cartesian degree of freedom",
                        "semiclassical, nonuniform near crossover",
                        "bead/PES/partition convergence must be checked separately"],
    }
