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
        value = validate_result(self.potential.evaluate(q), q.shape)
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
    return np.linalg.eigvalsh(0.5*(reduced+reduced.T))


def _check_stationary_result(result, action, q, mask, mass, gradient_tolerance,
                             minimum_spread, hessian, *, require_convergence,
                             function_evaluations, message, optimizer_diagnostics=None):
    result = action._from_values(q, mask, mass, result.energy, result.forces)
    active = q[:, mask]
    gradient = result["gradient"][:, mask].ravel()
    # Translation in imaginary time is the instanton phase/zero mode.
    tangent = (np.roll(active, -1, axis=0)-np.roll(active, 1, axis=0)).ravel()
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise InstantonError("collapsed instanton has no phase tangent")
    residual = float(np.linalg.norm(hessian @ tangent) /
                     max(np.linalg.norm(hessian) * tangent_norm, np.finfo(float).tiny))
    spectrum = _phase_projected_spectrum(hessian, tangent)
    scale = max(float(np.max(np.abs(spectrum))), 1.0)
    cutoff = scale*1e-8
    negative = int(np.count_nonzero(spectrum < -cutoff))
    zero = int(np.count_nonzero(np.abs(spectrum) <= cutoff))
    spread = float(np.max(np.linalg.norm(q-q.mean(axis=0), axis=(1, 2))))
    if np.max(np.abs(gradient)) > gradient_tolerance:
        raise InstantonError("instanton gradient exceeds the requested tolerance")
    if spread < minimum_spread:
        raise InstantonError("instanton collapsed to the transition state")
    if negative != 1:
        raise InstantonError("instanton Hessian must have one negative mode after phase projection")
    if not np.isfinite(residual) or residual > 0.1:
        raise InstantonError("imaginary-time phase direction is not sufficiently stationary")
    converged = bool(result.get("converged", True))
    if require_convergence and not converged:
        raise InstantonError("instanton solver did not report convergence")
    return InstantonResult(q.copy(), action.temperature_kelvin,
                           result["action_hbar"], result["energy_hartree"],
                           result["spring_action_hbar"], result["potential_action_hbar"],
                           float(np.max(np.abs(gradient))), spread, negative, zero,
                           residual, spectrum, converged, function_evaluations,
                           str(message), hessian.copy(),
                           dict(optimizer_diagnostics or {}))


def solve_instanton(potential, initial_path, masses, temperature_kelvin, *,
                    active_mask=None, energy_reference_hartree=0.0,
                    hessian_step_bohr=1e-4, gradient_tolerance=1e-7,
                    minimum_spread_bohr=1e-3, max_evaluations=400,
                    require_convergence=True, method="bofill"):
    """Solve a folded thermal instanton and validate the full-ring stationary path.

    ``method='root'`` uses SciPy's hybr root solver and is retained for small
    exact calculations. ``method='bofill'`` is the production path: it performs
    one exact finite-difference or provider Hessian evaluation, then uses Bofill
    updates with a trust radius and periodic exact refreshes. The returned
    full-ring path and Hessian are validated against the same potential. This
    is a constrained stationary action, not a proof of rate convergence.
    """
    q0 = coordinates(initial_path)
    if len(q0) < 8 or len(q0) % 2:
        raise ValueError("an instanton path requires an even number of beads >= 8")
    if method not in ("root", "bofill"):
        raise ValueError("method must be 'root' or 'bofill'")
    if isinstance(max_evaluations, (bool, np.bool_)) or not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 2:
        raise ValueError("max_evaluations must be an integer >= 2")
    gradient_tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    minimum_spread = _positive(minimum_spread_bohr, "minimum_spread_bohr")
    counted = _CountingPotential(potential)
    action = RingPolymerAction(counted, masses, temperature_kelvin,
                               active_mask=active_mask,
                               energy_reference_hartree=energy_reference_hartree,
                               hessian_step_bohr=hessian_step_bohr)
    q0, mask, mass = action._validate(q0)
    half = len(q0)//2
    if not np.allclose(q0[half:], q0[:half][::-1], rtol=0, atol=1e-10):
        raise ValueError("initial_path must have folded reflection symmetry")
    shape = q0[:half, mask].shape
    inactive = q0[0, ~mask].copy()

    def expand(x):
        first = np.asarray(x).reshape(shape)
        q = np.empty_like(q0)
        q[:half, mask] = first
        q[half:, mask] = first[::-1]
        q[:, ~mask] = inactive
        return q

    def folded_values(x):
        q = expand(x)
        value = validate_result(counted.evaluate(q[:half]), q[:half].shape)
        energies = np.concatenate((value.energy, value.energy[::-1]))
        forces = np.concatenate((value.forces, value.forces[::-1]))
        return q, value, action._from_values(q, mask, mass, energies, forces)

    if method == "root":
        def residual(x):
            return folded_values(x)[2]["gradient"][:half, mask].ravel()
        solution = root(residual, q0[:half, mask].ravel(), method="hybr",
                        options={"maxfev": int(max_evaluations), "xtol": 1e-10})
        q = expand(solution.x)
        final = validate_result(counted.evaluate(q), q.shape)
        hessian = RingPolymerAction._from_hessians(
            _physical_hessians(counted, q, mask, action.hessian_step),
            mass, action.beta/len(q))
        diagnostics = {"method": "root", "exact_hessian_refreshes": 1,
                       **counted.statistics}
        return _check_stationary_result(final, action, q, mask, mass,
                                        gradient_tolerance, minimum_spread, hessian,
                                        require_convergence=require_convergence,
                                        function_evaluations=int(solution.nfev),
                                        message=str(solution.message),
                                        optimizer_diagnostics=diagnostics)

    q, value, assembled = folded_values(q0[:half, mask].ravel())
    x = q0[:half, mask].ravel().copy()
    gradient = assembled["gradient"][:half, mask].ravel()
    full_hessian = RingPolymerAction._from_hessians(
        _physical_hessians(counted, q, mask, action.hessian_step),
        mass, action.beta/len(q))
    hessian = full_hessian[:len(x), :len(x)].copy()
    trust = min(max(float(np.linalg.norm(q0-q0.mean(axis=0)))/4, 0.02), 0.5)
    refreshes = 1
    accepted = 0
    rejected = 0
    converged = False
    message = "maximum evaluations reached"
    evaluations = 1
    tiny = np.finfo(float).tiny
    while evaluations < int(max_evaluations):
        max_gradient = float(np.max(np.abs(gradient)))
        if max_gradient <= gradient_tolerance:
            converged = True
            message = "gradient tolerance reached"
            break
        spectrum, vectors = np.linalg.eigh(0.5*(hessian+hessian.T))
        floor = max(float(np.max(np.abs(spectrum)))*1e-10, 1e-10)
        inverse = np.sign(spectrum)/np.maximum(np.abs(spectrum), floor)
        step = -(vectors*inverse) @ (vectors.T @ gradient)
        norm = float(np.linalg.norm(step))
        if norm > trust:
            step *= trust/norm
        candidate = x + step
        cq, cvalue, cassem = folded_values(candidate)
        evaluations += 1
        cgradient = cassem["gradient"][:half, mask].ravel()
        predicted = -float(gradient @ step + 0.5*step @ hessian @ step)
        actual = 0.5*float(gradient @ gradient-cgradient @ cgradient)
        ratio = actual/max(abs(predicted), tiny)
        if actual > 0 and ratio > 0.05:
            hessian, updated = _bofill_update(hessian, step, cgradient-gradient)
            x, q, value, assembled, gradient = candidate, cq, cvalue, cassem, cgradient
            accepted += 1
            trust = min(0.5, trust*(1.5 if ratio > 0.75 else 1.0))
            if not updated or accepted % 8 == 0:
                full_hessian = RingPolymerAction._from_hessians(
                    _physical_hessians(counted, q, mask, action.hessian_step),
                    mass, action.beta/len(q))
                hessian = full_hessian[:len(x), :len(x)].copy()
                refreshes += 1
        else:
            rejected += 1
            trust = max(trust*0.25, 1e-6)
            if rejected % 3 == 0:
                full_hessian = RingPolymerAction._from_hessians(
                    _physical_hessians(counted, q, mask, action.hessian_step),
                    mass, action.beta/len(q))
                hessian = full_hessian[:len(x), :len(x)].copy()
                refreshes += 1
    final = validate_result(counted.evaluate(q), q.shape)
    hessian = RingPolymerAction._from_hessians(
        _physical_hessians(counted, q, mask, action.hessian_step),
        mass, action.beta/len(q))
    refreshes += 1
    diagnostics = {"method": "bofill", "exact_hessian_refreshes": refreshes,
                   "accepted_steps": accepted, "rejected_steps": rejected,
                   "final_trust_radius_bohr": trust, **counted.statistics}
    result = _check_stationary_result(final, action, q, mask, mass,
                                      gradient_tolerance, minimum_spread, hessian,
                                      require_convergence=require_convergence,
                                      function_evaluations=evaluations,
                                      message=message,
                                      optimizer_diagnostics=diagnostics)
    result.converged = converged
    return result


def gaussian_rate(instanton, *, reactant_frequencies_au, reactant_energy_hartree=0.0,
                  symmetry_factor=1.0):
    """Multidimensional Gaussian instanton rate, with explicit assumptions.

    The reactant frequencies are positive angular frequencies in atomic units;
    zero and rotational modes must already be removed. The instanton Hessian
    is the full dimensionless S/hbar Hessian in bead-major Cartesian coordinates.
    Exactly one negative mode is required after geometric phase projection.
    The logarithmic formula uses the conventional thermal-instanton determinant
    ratio and a user-supplied symmetry factor.  This is not a substitute for
    path/bead/temperature convergence or independent physical validation.
    """
    if instanton.action_hessian is None:
        raise ValueError("instanton result does not retain its action Hessian")
    frequencies = np.asarray(reactant_frequencies_au, dtype=float)
    if frequencies.ndim != 1 or len(frequencies) == 0 or not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("reactant_frequencies_au must be a nonempty positive vector")
    symmetry = _positive(symmetry_factor, "symmetry_factor")
    reactant_energy = float(reactant_energy_hartree)
    if not np.isfinite(reactant_energy):
        raise ValueError("reactant_energy_hartree must be finite")
    beta = 1.0/(KB_AU*instanton.temperature_kelvin)
    # Reactant vibrational partition includes zero-point energy explicitly.
    log_q = -np.sum(0.5*beta*frequencies + np.log(-np.expm1(-beta*frequencies)))
    tangent = (np.roll(instanton.path_bohr, -1, axis=0)
               - np.roll(instanton.path_bohr, 1, axis=0)).ravel()
    spectrum = _phase_projected_spectrum(instanton.action_hessian, tangent)
    scale = max(float(np.max(np.abs(spectrum))), 1.0)
    cutoff = scale*1e-8
    negative = spectrum[spectrum < -cutoff]
    positive = spectrum[spectrum > cutoff]
    if len(negative) != 1 or len(positive) != len(spectrum)-1:
        raise InstantonError("Gaussian rate requires one negative and no unresolved fluctuation modes")
    log_prefactor = (math.log(symmetry) - math.log(beta)
                     + 0.5*(np.sum(np.log(beta*frequencies**2))
                            - np.sum(np.log(positive))))
    log_rate_au = (log_prefactor - instanton.action_hbar + beta*reactant_energy - log_q)
    return float(np.exp(log_rate_au)/ATOMIC_TIME_S)


def instanton_rate_1d(potential, path, mass, temperature_kelvin, *,
                      barrier_angular_frequency, reactant_angular_frequency,
                      symmetry_factor=1.0):
    """Return a semiclassical one-dimensional rate in s^-1.

    This is an independently testable special case. The supplied angular
    frequencies are positive atomic units; the barrier value enters the
    fluctuation prefactor and the reactant value the harmonic partition factor.
    The instanton must be validated by solve_instanton before calling this.
    """
    if not isinstance(path, InstantonResult):
        raise TypeError("path must be an InstantonResult")
    barrier = _positive(barrier_angular_frequency, "barrier_angular_frequency")
    reactant = _positive(reactant_angular_frequency, "reactant_angular_frequency")
    symmetry = _positive(symmetry_factor, "symmetry_factor")
    beta = 1.0/(KB_AU*temperature_kelvin)
    q_reactant = np.exp(-0.5*beta*reactant)/(1-np.exp(-beta*reactant))
    prefactor = symmetry/(beta*math.sqrt(2*math.pi))*math.sqrt(barrier/reactant)
    return float(prefactor*np.exp(-path.action_hbar)/(q_reactant*ATOMIC_TIME_S))
