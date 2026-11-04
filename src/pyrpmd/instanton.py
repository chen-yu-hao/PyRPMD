"""Thermal ring-polymer instanton paths and a one-dimensional rate cross-check.

The Cartesian path solver is a stationary-action calculation, not molecular
dynamics.  Its folded half-ring representation fixes imaginary-time phase; the
full-ring Hessian is retained for mode diagnostics.  Quantitative rates in this
module retain the original one-dimensional cross-check. Multidimensional
Gaussian rates and cached-Hessian TST/Wigner/Eckart estimates are available
in pyrpmd.low_temperature; open-path splittings are in pyrpmd.tunnelling.

References: Richardson & Althorpe, JCP 131, 214106 (2009),
doi:10.1063/1.3267318; Richardson, JCP 144, 114106 (2016),
doi:10.1063/1.4943866.  All inputs use bohr, electron masses, hartree,
and kelvin; hbar=1 in atomic units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from scipy.optimize import root

from .constants import ATOMIC_TIME_S, BOHR_M, KB_AU
from .potentials.base import coordinates, evaluate_checked
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


class _CountingPotential:
    """Count actual provider work without changing the provider's interface."""

    def __init__(self, potential):
        self.potential = potential
        self.statistics = {
            "force_calls": 0, "force_configurations": 0,
            "physical_hessian_evaluations": 0,
            "finite_difference_hessian_evaluations": 0,
            "hessian_provider_calls": 0, "hessian_provider_configurations": 0,
        }

    def evaluate(self, q):
        self.statistics["force_calls"] += 1
        self.statistics["force_configurations"] += len(q)
        return self.potential.evaluate(q)

    def __getattr__(self, name):
        value = getattr(self.potential, name)
        if name != "hessian" or not callable(value):
            return value

        def counted(q):
            self.statistics["hessian_provider_calls"] += 1
            self.statistics["hessian_provider_configurations"] += len(q)
            return value(q)
        return counted


def _physical_hessians(potential, q, mask, step):
    """Physical, unweighted Hessians, in hartree/bohr**2.

    An optional ``potential.hessian(x)`` receives float64 ``[B,N,3]`` and
    returns finite symmetric float64 ``[B,3*N,3*N]`` (or its equivalent
    ``[B,N,3,N,3]`` view). The complete Cartesian tensor is validated before
    selecting active coordinates. A missing method enables central force
    differences; an invalid method, invalid result or provider error fails
    explicitly, without falling back to a different derivative calculation.
    """
    counts = potential.statistics if isinstance(potential, _CountingPotential) else None
    if counts is not None:
        counts["physical_hessian_evaluations"] += 1
    missing = object()
    provider = getattr(potential, "hessian", missing)
    if provider is not missing:
        if not callable(provider):
            raise ValueError("potential.hessian must be callable when present")
        full = np.asarray(provider(q))
        n = q.shape[1]
        if full.shape == (len(q), n, 3, n, 3):
            full = full.reshape(len(q), 3*n, 3*n)
        if full.dtype != np.dtype(np.float64):
            raise ValueError("potential.hessian must return float64 without changing precision")
        if full.shape != (len(q), 3*n, 3*n) or not np.isfinite(full).all():
            raise ValueError("potential.hessian must return finite [B,3N,3N] Cartesian Hessians")
        if not np.allclose(full, full.transpose(0, 2, 1), rtol=1e-8, atol=1e-12):
            raise ValueError("potential.hessian must return symmetric Cartesian Hessians")
        active = mask.ravel()
        h = full[:, active][:, :, active].copy()
        return 0.5 * (h + h.transpose(0, 2, 1))
    if counts is not None:
        counts["finite_difference_hessian_evaluations"] += 1
    ids = np.argwhere(mask)
    h = np.empty((len(q), len(ids), len(ids)))
    for j, (atom, component) in enumerate(ids):
        plus, minus = q.copy(), q.copy()
        plus[:, atom, component] += step
        minus[:, atom, component] -= step
        # Plugins may reuse their force buffer; preserve it before the next call.
        fp = evaluate_checked(potential, plus).forces[:, mask].copy()
        fm = evaluate_checked(potential, minus).forces[:, mask].copy()
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
    result = evaluate_checked(potential, q)
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
        value = evaluate_checked(self.potential, q)
        return self._from_values(q, mask, mass, value.energy, value.forces)

    def _from_values(self, q, mask, mass, energies, forces):
        """Assemble the action, also used with a folded provider evaluation."""
        p = len(q)
        dtau = self.beta / p
        v = energies - self.energy_reference
        active = q[:, mask]
        difference = active - np.roll(active, 1, axis=0)
        spring = float(np.sum(mass * difference**2) / (2 * dtau))
        potential = float(dtau * np.sum(v))
        gradient = np.zeros_like(q)
        gradient[:, mask] = (mass / dtau * (2 * active - np.roll(active, 1, axis=0)
                                          - np.roll(active, -1, axis=0))
                             - dtau * forces[:, mask])
        energy = float(np.mean(v) - spring / self.beta)
        return {"action_hbar": spring + potential, "gradient": gradient,
                "energy_hartree": energy, "spring_action_hbar": spring,
                "potential_action_hbar": potential}

    def hessian_vector_product(self, path, direction):
        """Matrix-free action-Hessian product in full Cartesian shape.

        ``direction`` has the same float64 ``[P,N,3]`` shape as ``path``.
        The returned vector has that same shape; inactive components are zero.
        The calculation does not allocate the dense action Hessian and requires
        the physical PES's exact HVP interface. Missing or invalid provider
        implementations fail rather than being approximated. This method is
        suitable for external iterative eigensolvers, while the default
        instanton optimizer and Gaussian prefactor still use the dense Hessian.
        """
        q, mask, mass = self._validate(path)
        v = np.asarray(direction)
        if (v.dtype != np.dtype("float64") or v.shape != q.shape
                or not np.isfinite(v).all()):
            raise ValueError("direction must be finite float64 [P,N,3]")
        if np.any(v[:, ~mask] != 0):
            raise ValueError("direction must vanish in inactive coordinates")
        provider = getattr(self.potential, "hessian_vector_product", None)
        if provider is None or not callable(provider):
            raise ValueError("matrix-free action Hessian requires potential.hessian_vector_product(x, v)")
        physical = np.asarray(provider(q, np.ascontiguousarray(v)))
        if (physical.dtype != np.dtype("float64") or physical.shape != q.shape
                or not np.isfinite(physical).all()):
            raise ValueError("potential.hessian_vector_product must return finite float64 [P,N,3]")
        dtau = self.beta / len(q)
        active = v[:, mask]
        spring = mass / dtau * (
            2 * active - np.roll(active, 1, axis=0)
            - np.roll(active, -1, axis=0)
        )
        product = np.zeros_like(v)
        product[:, mask] = spring + dtau * physical[:, mask]
        return np.ascontiguousarray(product)

    def hessian(self, path):
        """Dense action Hessian in bead-major active Cartesian coordinates."""
        q, mask, mass = self._validate(path)
        p, d = len(q), len(mass)
        if p * d > 4096:
            raise ValueError("dense instanton Hessian is limited to 4096 active bead coordinates")
        dtau = self.beta / p
        physical = _physical_hessians(self.potential, q, mask, self.hessian_step)
        return self._from_hessians(physical, mass, dtau)

    @staticmethod
    def _from_hessians(physical, mass, dtau):
        p, d = len(physical), len(mass)
        h = np.zeros((p, d, p, d))
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
    action_hessian: np.ndarray | None = None
    optimizer_diagnostics: dict = field(default_factory=dict)

    def to_dict(self, *, include_path=True):
        output = {key: value for key, value in vars(self).items()
                  if key not in ("path_bohr", "hessian_eigenvalues", "action_hessian")}
        output.update(nbeads=len(self.path_bohr),
                      hessian_eigenvalues=self.hessian_eigenvalues.tolist(),
                      scope="stationary_action_path_not_a_multidimensional_rate",
                      numerical_converged=self.converged,
                      physical_converged=False)
        if include_path:
            output["path_bohr"] = self.path_bohr.tolist()
        return output


def _bofill_update(hessian, displacement, gradient_change):
    """Symmetric PSB/SR1 interpolation satisfying the secant condition.

    Bofill, J. Comput. Chem. 15, 1 (1994), doi:10.1002/jcc.540150102.
    The weighted SR1 term is evaluated without dividing by r.dot(s), so an
    orthogonal secant error continuously reduces to the Powell update.
    """
    s = displacement
    r = gradient_change - hessian @ s
    ss = float(s @ s)
    rr = float(r @ r)
    if ss <= np.finfo(float).tiny or rr <= np.finfo(float).eps**2:
        return hessian, False
    sr = float(s @ r)
    weight = min(1.0, max(0.0, sr*sr/(ss*rr)))
    psb = (np.outer(r, s) + np.outer(s, r))/ss - sr/(ss*ss)*np.outer(s, s)
    update = (1-weight)*psb + sr/(ss*rr)*np.outer(r, r)
    return hessian + update, True


def _phase_projected_spectrum(hessian, tangent):
    """Spectrum on the geometric phase tangent's orthogonal complement.

    A Householder reflection takes the normalized tangent onto coordinate zero.
    Dropping that coordinate then gives the same restriction as Q.T @ H @ Q
    for any orthonormal complement Q, without allocating a dense Q or doing two
    cubic-cost matrix products. This removes a geometrical direction, never an
    eigenvalue chosen merely because it is small or negative.
    """
    vector = tangent / np.linalg.norm(tangent)
    vector = vector.copy()
    vector[0] += 1.0 if vector[0] >= 0.0 else -1.0
    vector /= np.linalg.norm(vector)
    product = hessian @ vector
    reduced = hessian[1:, 1:].copy()
    reduced -= 2*np.outer(vector[1:], product[1:])
    reduced -= 2*np.outer(product[1:], vector[1:])
    reduced += 4*float(vector @ product)*np.outer(vector[1:], vector[1:])
    return np.linalg.eigvalsh(reduced)


def _bofill_stationary(evaluate, hessian, x, scale, multiplicity, dimension,
                       tolerance, max_iterations, trust_radius, refresh_interval):
    """Damped Newton solution of stationarity equations with Bofill updates.

    The merit function is the squared gradient, never the action: a thermal
    instanton is an action saddle. A scaled symmetric Hessian retains negative
    curvature; the pseudoinverse omits only numerically unresolved directions.
    Trust acceptance tests the actual gradient reduction against the model.
    """
    statistics = {"hessian_updates": 0, "hessian_refreshes": 0,
                  "accepted_steps": 0, "rejected_steps": 0,
                  "iterations": 0, "history": []}
    z = x * scale
    g = evaluate(x) * scale**-1
    nfev = 1
    row_scale = scale / np.repeat(multiplicity, dimension)
    residual = float(np.max(np.abs(g * row_scale)))
    statistics["history"].append({"gradient_max": residual, "trust_radius_bohr": trust_radius})
    if residual <= tolerance or max_iterations == 1:
        statistics["termination"] = "gradient tolerance" if residual <= tolerance else "evaluation budget"
        return x, nfev, statistics
    h = hessian(x) / (scale[:, None]*scale[None, :])
    radius = trust_radius
    consecutive_rejections = 0
    refresh_due = False
    termination = "evaluation budget"
    while nfev < max_iterations:
        if refresh_due:
            h = hessian(z / scale) / (scale[:, None]*scale[None, :])
            statistics["hessian_refreshes"] += 1
            refresh_due = False
        eigenvalues, vectors = np.linalg.eigh(h)
        cutoff = max(float(np.max(np.abs(eigenvalues))) * 1e-10, np.finfo(float).tiny)
        usable = np.abs(eigenvalues) > cutoff
        projection = vectors.T @ g
        inverse = np.zeros_like(eigenvalues)
        inverse[usable] = 1/eigenvalues[usable]
        delta = -vectors @ (projection*inverse)
        size = float(np.max(np.linalg.norm((delta/scale).reshape(-1, dimension), axis=1)))
        if not np.isfinite(size):
            raise InstantonError("nonfinite Bofill Newton step")
        if size <= 10*np.finfo(float).eps*max(1.0, float(np.max(np.abs(z/scale)))):
            termination = "unresolved Hessian null-space or floating-point step"
            break
        limited = size > radius
        if limited:
            delta *= radius/size
        model_gradient = g + h @ delta
        norm2 = float(g @ g)
        predicted = norm2 - float(model_gradient @ model_gradient)
        trial = z + delta
        new_g = evaluate(trial / scale) / scale
        nfev += 1
        statistics["iterations"] += 1
        actual = norm2 - float(new_g @ new_g)
        ratio = actual/predicted if predicted > 0 else -math.inf
        new_residual = float(np.max(np.abs(new_g * row_scale)))
        accepted = bool(actual > 0 and ratio > 1e-4)
        # A sub-roundoff norm change still counts if the requested physical
        # force tolerance is reached. No solver status alone implies success.
        accepted |= new_residual <= tolerance
        if accepted:
            h, updated = _bofill_update(h, delta, new_g-g)
            statistics["hessian_updates"] += int(updated)
            z, g, residual = trial, new_g, new_residual
            statistics["accepted_steps"] += 1
            consecutive_rejections = 0
            if ratio < 0.25:
                radius *= 0.5
            elif ratio > 0.75 and limited:
                radius = min(trust_radius, 2*radius)
            statistics["history"].append({"gradient_max": residual,
                                          "trust_radius_bohr": radius})
            if residual <= tolerance:
                termination = "gradient tolerance"
                break
            if refresh_interval and statistics["accepted_steps"] % refresh_interval == 0:
                refresh_due = True
        else:
            statistics["rejected_steps"] += 1
            consecutive_rejections += 1
            radius *= 0.5
            if consecutive_rejections >= 3:
                refresh_due = True
                consecutive_rejections = 0
        if radius < 1e-12:
            termination = "trust radius exhausted"
            break
    statistics["termination"] = termination
    return z / scale, nfev, statistics


def optimize_instanton(potential, masses, temperature_kelvin, initial_path, *,
                      active_mask=None, transition_state=None,
                      energy_reference_hartree=0.0, hessian_step_bohr=1e-4,
                      gradient_tolerance=1e-8, max_iterations=200,
                      minimum_spread_bohr=1e-5, raise_on_failure=True,
                      retain_hessian=False, method="root",
                      hessian_update_interval=0, trust_radius_bohr=0.3,
                      phase_tolerance=0.05):
    """Find a folded periodic stationary orbit from a noncollapsed path seed.

    ``initial_path[P,N,3]`` must obey link reflection q[j]=q[P-1-j],
    as produced by ``instanton_seed``. Bead reflection q[j]=q[P-j] is also
    accepted, but may have an extra unstable discretized phase at small P.
    The half-ring stationarity equations are solved with their analytic spring
    and provider or finite-difference physical Hessian. This removes the phase freedom
    without adding a centroid constraint that would distort asymmetric barriers.
    A root must be noncollapsed, stationary, and have exactly one physical
    unstable mode, with the resolved discrete phase treated as described below.
    Optional ``transition_state`` also rejects T >= Tc.

    This is a local solver: different paths can exist, and an appropriate seed,
    bead study and independent PES validation remain necessary.  The default
    raises on failure.  ``raise_on_failure=False`` returns diagnostics only.
    ``method='root'`` retains the SciPy hybrid solver. ``method='bofill'`` uses
    a mass-scaled Newton trust method with symmetric Bofill Hessian updates;
    it solves for an action saddle, rather than minimizing the action.
    ``trust_radius_bohr`` bounds the Euclidean displacement of each bead.
    A physical Hessian is computed initially, on trust-model stagnation, every
    ``hessian_update_interval`` accepted steps when nonzero, and at the final
    path for mode diagnostics. Zero disables scheduled Hessian refreshes.
    At finite P the imaginary-time phase curvature may be slightly negative.
    If the raw spectrum has extra negative modes, the geometric Fourier phase
    direction is projected out only when its relative Hessian residual is at
    most ``phase_tolerance``. The projected spectrum must still have exactly
    one negative mode. Raw counts and whether projection was used are retained
    in diagnostics; ``hessian_eigenvalues`` always contains the full spectrum.
    A single-negative path retains the legacy path-only convergence criterion;
    the rate API independently requires a resolved phase even for that case.
    ``max_iterations`` bounds the solver's residual-evaluation budget.
    Both methods evaluate only the unique folded beads; the final spectrum
    and cached Hessian still describe the entire ring. Provider call counts
    and evaluated configurations are in ``result.optimizer_diagnostics``.
    ``retain_hessian=True`` retains the final unweighted Cartesian action
    Hessian for pyrpmd.low_temperature.instanton_rate, avoiding another set
    of PES finite differences. The cache is excluded from JSON diagnostics.
    """
    if not isinstance(retain_hessian, (bool, np.bool_)):
        raise ValueError("retain_hessian must be boolean")
    if not isinstance(raise_on_failure, (bool, np.bool_)):
        raise ValueError("raise_on_failure must be boolean")
    if method not in ("root", "bofill"):
        raise ValueError("method must be 'root' or 'bofill'")
    if (isinstance(hessian_update_interval, (bool, np.bool_))
            or not isinstance(hessian_update_interval, (int, np.integer))
            or hessian_update_interval < 0):
        raise ValueError("hessian_update_interval must be a nonnegative integer")
    trust_radius = _positive(trust_radius_bohr, "trust_radius_bohr")
    phase_limit = _positive(phase_tolerance, "phase_tolerance")
    if method == "root" and (hessian_update_interval or trust_radius != 0.3):
        raise ValueError("Hessian update and trust radius options require method='bofill'")
    counted = _CountingPotential(potential)
    action = RingPolymerAction(counted, masses, temperature_kelvin,
                              active_mask=active_mask,
                              energy_reference_hartree=energy_reference_hartree,
                              hessian_step_bohr=hessian_step_bohr)
    q, mask, mass = action._validate(initial_path)
    tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    minimum_spread = _positive(minimum_spread_bohr, "minimum_spread_bohr")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    p, d = len(q), len(mass)
    if p*d > 4096:
        raise ValueError("dense instanton Hessian is limited to 4096 active bead coordinates")
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
        tc = crossover_temperature(counted, ts_geometry, masses,
                                      active_mask=mask, hessian_step_bohr=hessian_step_bohr)
        if action.temperature_kelvin >= tc:
            raise InstantonError(f"thermal instanton requires temperature below crossover ({tc:.8g} K)")
    template = q.copy()
    multiplicity = np.bincount(mapping, minlength=half)
    dtau = action.beta / p
    # Normalize each row by its multiplicity.  The equations still find exactly
    # the same stationary point; endpoint and interior residuals are comparable.
    def unpack(x):
        path = template.copy()
        path[:, mask] = x.reshape(half, d)[mapping]
        return path

    def evaluate_folded(x):
        path = unpack(x)
        unique = path[:half]
        value = evaluate_checked(counted, unique)
        return action._from_values(path, mask, mass, value.energy[mapping], value.forces[mapping])

    def folded_gradient(x):
        g = evaluate_folded(x)["gradient"][:, mask]
        reduced = np.zeros((half, d))
        np.add.at(reduced, mapping, g)
        return reduced.ravel()

    def folded_hessian(x):
        physical = _physical_hessians(counted, unpack(x)[:half], mask, action.hessian_step)
        reduced = np.zeros((half, d, half, d))
        for j in range(half):
            reduced[j, :, j, :] += multiplicity[j]*dtau*physical[j]
        spring = np.diag(mass/dtau)
        for j in range(p):
            a, b = mapping[j], mapping[(j-1) % p]
            reduced[a, :, a, :] += spring
            reduced[b, :, b, :] += spring
            reduced[a, :, b, :] -= spring
            reduced[b, :, a, :] -= spring
        return reduced.reshape(half * d, half * d)

    x0 = q[:half, mask].ravel()
    if method == "root":
        rows = np.repeat(multiplicity, d)
        solution = root(lambda x: folded_gradient(x)/rows, x0,
                        jac=lambda x: folded_hessian(x)/rows[:, None], method="hybr",
                        options={"xtol": min(tolerance, 1e-9), "maxfev": int(max_iterations)})
        x, nfev = solution.x, int(solution.nfev)
        diagnostics = {"hessian_updates": 0, "hessian_refreshes": max(0, int(solution.njev)-1),
                       "termination": str(solution.message)}
    else:
        # Scaling includes the folded multiplicity, preserving a symmetric
        # Hessian even for bead-reflected endpoints. Dividing mass by its mean
        # keeps the scaled coordinates near the magnitude of Cartesian bohr.
        scale = np.sqrt(multiplicity[:, None]*mass[None, :]/float(np.mean(mass))).ravel()
        x, nfev, diagnostics = _bofill_stationary(
            folded_gradient, folded_hessian, x0, scale, multiplicity, d,
            tolerance, int(max_iterations), trust_radius, int(hessian_update_interval))
    path = unpack(x)
    evaluation = evaluate_folded(x)
    physical = _physical_hessians(counted, path[:half], mask, action.hessian_step)
    h = action._from_hessians(physical[mapping], mass, dtau)
    if method == "root":
        diagnostics["hessian_refreshes"] = max(
            0, counted.statistics["physical_hessian_evaluations"]
            - int(transition_state is not None) - 2)
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
    diagnostics.update(raw_negative_modes=negative, raw_near_zero_modes=near_zero,
                       phase_projection_applied=False,
                       projected_negative_modes=None, projected_near_zero_modes=None,
                       phase_tolerance=phase_limit,
                       hessian_spectrum_scope="full_ring_including_discrete_phase")
    if negative > 1 and tangent_norm > 1e-12 and phase_residual <= phase_limit:
        projected = _phase_projected_spectrum(hmw, tangent)
        negative = int(np.count_nonzero(projected < -mode_cutoff))
        diagnostics.update(phase_projection_applied=True,
                           projected_negative_modes=negative,
                           projected_near_zero_modes=int(np.count_nonzero(np.abs(projected) <= mode_cutoff)))
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
    if diagnostics["raw_negative_modes"] > 1 and phase_residual > phase_limit:
        reasons.append(f"imaginary-time phase residual {phase_residual:.3g} exceeds "
                       f"phase_tolerance {phase_limit:.3g}; phase unresolved, increase beads "
                       "and refine the orbit before classifying the additional negative modes")
    result = InstantonResult(
        path, action.temperature_kelvin, evaluation["action_hbar"],
        evaluation["energy_hartree"], evaluation["spring_action_hbar"],
        evaluation["potential_action_hbar"], gradient_max, spread, negative,
        near_zero, phase_residual, eigenvalues, converged, nfev,
        "; ".join(reasons) if reasons else "stationary noncollapsed first-order orbit",
        h if retain_hessian else None,
        dict(counted.statistics, method=method, unique_beads=half,
             hessian_update_interval=int(hessian_update_interval),
             trust_radius_bohr=trust_radius if method == "bofill" else None,
             **diagnostics),
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
