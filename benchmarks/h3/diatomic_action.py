"""Full-potential j=0 diatomic action quantization and QCT preparation.

This is an opt-in H3 benchmark, not a change to PyRPMD's core initial sampler.
All quantities use atomic units.  Energies are relative to the optimized well
minimum.  ``J(E) = integral(p_r dr)/pi = v+1/2`` is leading-order EBK/WKB
quantization; its energies are not exact radial Schrodinger eigenvalues.

Preparation on a classical periodic orbit follows the usual diatomic QCT
construction: choose a uniform *time* phase, not a uniform bond length.  See
Marjollet and Welsch, IJQC 121, e26447 (2021), Section 2.1,
https://doi.org/10.1002/qua.26447, and the foundational QCT work of Karplus,
Porter and Sharma, JCP 43, 3259 (1965), https://doi.org/10.1063/1.1697301.
The two-turning-point semiclassical action convention is explicitly given in
J. Mol. Spectrosc. 330, 20-27 (2016), Eq. (1),
https://doi.org/10.1016/j.jms.2016.09.008.

Numerically, r = midpoint - halfwidth*cos(theta) regularizes both turning
points.  Gauss-Legendre quadrature is refined for the action and period. A
separately refined monotone half-period CDF supplies the inverse time map.
Every sampled momentum is evaluated from the actual potential, so sampling
does not interpolate or rescale the energy. The supported radial domain must
contain one well and two simple turning points; failures are explicit.

Example::

    model = DiatomicAction.from_bkmp2(potential, reduced_mass=918.5763237198094)
    orbit = model.quantize(0)
    r, p_r = orbit.sample_phase(0.25)  # p_r = mu*dr/dt, positive outward
    q, p = orbit.sample_cartesian(rng, masses=(1837.1526474396187,)*2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, minimize_scalar
from scipy.special import roots_legendre


H2_REDUCED_MASS = 918.5763237198094
AU_TIME_FS = 0.024188843265857


class ActionDomainError(ValueError):
    """The requested bound orbit is not supported inside the supplied domain."""


class ActionConvergenceError(RuntimeError):
    """An explicitly bounded numerical refinement did not converge."""


@lru_cache(maxsize=16)
def _gauss(order):
    nodes, weights = roots_legendre(order)
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class DiatomicOrbit:
    """A fixed-energy, j=0 classical orbit with a uniform-time phase map.

    ``sample_phase(0)`` is the inner turning point, ``sample_phase(0.5)`` the
    outer turning point. The first half-cycle has positive radial momentum.
    No orientation, collision velocity, centroid excitation or thermostat is
    added. The optional Cartesian adapter returns only the two-atom fragment.
    """

    energy_hartree: float
    energy_absolute_hartree: float
    action: float
    period_au: float
    turning_points_bohr: tuple[float, float]
    reduced_mass: float
    v: int | None
    numerical_diagnostics: dict
    _model: object = field(repr=False, compare=False)
    _inverse_time: object = field(repr=False, compare=False)

    @property
    def period_fs(self):
        return self.period_au * AU_TIME_FS

    @property
    def metadata(self):
        return {
            "method": "full_potential_EBK_action_QCT_j0", "v": self.v, "j": 0,
            "action_au": self.action, "target_action_au": None if self.v is None else self.v + 0.5,
            "energy_above_well_minimum_hartree": self.energy_hartree,
            "absolute_energy_hartree": self.energy_absolute_hartree,
            "turning_points_bohr": list(self.turning_points_bohr),
            "period_au": self.period_au, "period_fs": self.period_fs,
            "reduced_mass_au": self.reduced_mass,
            "sampling": "one uniform time phase per draw; signed p_r=mu*dr/dt",
            "scope": "Leading-order semiclassical action energy, not an exact quantum eigenvalue or quantum phase-space distribution",
            "reference": self._model.metadata,
            "numerics": dict(self.numerical_diagnostics),
        }

    def sample_phase(self, phase):
        """Return (r,p_r) for time/T in [0,1]; supports scalars and arrays.

        Phase 1 is identified with phase 0. Outside values are rejected rather
        than silently wrapped, so a caller cannot mistake radians for cycles.
        """
        raw = np.asarray(phase)
        if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise ValueError("time phase must be a finite real number in [0,1]")
        cycle = np.asarray(raw, dtype=float) % 1.0
        outward = cycle < 0.5
        fraction = np.where(outward, 2 * cycle, 2 * (1 - cycle))
        theta = np.asarray(self._inverse_time(fraction))
        r, gap = self._model._coordinates_and_gap(theta, self.energy_hartree,
                                                 self.turning_points_bohr)
        tolerance = 5e-12 * max(1.0, self.energy_hartree)
        if np.any(gap < -tolerance):
            raise ActionConvergenceError("Sampled orbit lies outside the classical energy shell")
        p = np.sqrt(2 * self.reduced_mass * np.maximum(gap, 0)) * np.where(outward, 1, -1)
        # Roots may carry a few ulps of potential error. Turning-point momentum
        # is exactly zero; the root residual is separately reported and tested.
        p = np.where((fraction == 0) | (fraction == 1), 0.0, p)
        if raw.ndim == 0:
            return float(r), float(p)
        return r, p

    def sample(self, rng, size=None):
        """Use exactly one Generator.random draw per requested orbit point."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("sample requires a numpy.random.Generator")
        return self.sample_phase(rng.random(size=size))

    def cartesian_at_phase(self, phase, *, masses=None, axis=(0.0, 0.0, 1.0)):
        """Return COM-rest Cartesian (q,p), shape [...,2,3], with zero rotation.

        ``axis`` points from atom B to atom C. The masses must have the same
        reduced mass as this orbit; omitted masses mean equal atoms of mass
        2*mu. Whole-fragment orientation and the collision launch are external.
        """
        masses = np.asarray((2 * self.reduced_mass,) * 2 if masses is None else masses, dtype=float)
        axis = np.asarray(axis, dtype=float)
        if masses.shape != (2,) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("masses must contain two positive finite atomic masses")
        mu = float(masses[0] * masses[1] / masses.sum())
        if not np.isclose(mu, self.reduced_mass, rtol=1e-12, atol=0):
            raise ValueError("Cartesian masses do not match the orbit reduced mass")
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.linalg.norm(axis) > 0:
            raise ValueError("axis must be a finite nonzero three-vector")
        axis = axis / np.linalg.norm(axis)
        r, p = map(np.asarray, self.sample_phase(phase))
        relative = r[..., None] * axis
        momentum = p[..., None] * axis
        q = np.stack((-masses[1] / masses.sum() * relative,
                       masses[0] / masses.sum() * relative), axis=-2)
        p = np.stack((-momentum, momentum), axis=-2)
        return q, p

    def sample_cartesian(self, rng, *, masses=None, axis=(0.0, 0.0, 1.0), size=None):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("sample_cartesian requires a numpy.random.Generator")
        return self.cartesian_at_phase(rng.random(size=size), masses=masses, axis=axis)


class DiatomicAction:
    """Bounded single-well action solver for scalar V(r), dV/dr callables.

    The potential may have any absolute energy zero; ``values`` subtracts the
    fitted minimum. The domain and equilibrium bracket are in bohr. This is
    restricted to a single smooth well and simple, interior turning points;
    rotation, multiple wells, dissociation and hard-wall quantization are not
    supplied. Derivatives must correspond to the same scalar radial potential.
    """

    def __init__(self, potential, reduced_mass, *, gradient, domain=(0.2, 12.0),
                 equilibrium_bracket=None, quadrature_order=64, max_quadrature_order=1024,
                 quadrature_rtol=2e-10, quadrature_atol=1e-11,
                 time_cells=128, max_time_cells=8192, time_tolerance=1e-8,
                 provenance=None):
        if not callable(potential) or not callable(gradient):
            raise TypeError("potential and gradient must be scalar radial callables")
        self._potential, self._gradient = potential, gradient
        self.reduced_mass = _positive(reduced_mass, "reduced_mass")
        self.domain = tuple(float(x) for x in domain)
        if len(self.domain) != 2 or not np.isfinite(self.domain).all() or not 0 < self.domain[0] < self.domain[1]:
            raise ActionDomainError("domain must be a finite positive interval (r_min,r_max)")
        bracket = self.domain if equilibrium_bracket is None else tuple(float(x) for x in equilibrium_bracket)
        if len(bracket) != 2 or not np.isfinite(bracket).all() or not self.domain[0] <= bracket[0] < bracket[1] <= self.domain[1]:
            raise ActionDomainError("equilibrium bracket must lie inside the radial domain")
        self.quadrature_order = _integer(quadrature_order, "quadrature_order", 8)
        self.max_quadrature_order = _integer(max_quadrature_order, "max_quadrature_order", 2 * self.quadrature_order)
        self.quadrature_rtol = _positive(quadrature_rtol, "quadrature_rtol")
        self.quadrature_atol = _positive(quadrature_atol, "quadrature_atol")
        self.time_cells = _integer(time_cells, "time_cells", 16)
        self.max_time_cells = _integer(max_time_cells, "max_time_cells", 2 * self.time_cells)
        self.time_tolerance = _positive(time_tolerance, "time_tolerance")
        self.provenance = dict(provenance or {})
        minimum = minimize_scalar(lambda r: float(self._potential(r)), bounds=bracket,
                                  method="bounded", options={"xatol": 1e-13})
        if not minimum.success or not np.isfinite(minimum.fun) or not bracket[0] < minimum.x < bracket[1]:
            raise ActionDomainError("Could not locate an interior well minimum")
        self.equilibrium_bohr, self.minimum_hartree = float(minimum.x), float(minimum.fun)
        left_grid = np.linspace(self.domain[0], self.equilibrium_bohr, 129)[:-1]
        right_grid = np.linspace(self.equilibrium_bohr, self.domain[1], 129)[1:]
        left_gradient, right_gradient = self.gradients(left_grid), self.gradients(right_grid)
        scale = max(1.0, float(np.max(np.abs(np.r_[left_gradient, right_gradient]))))
        if np.any(left_gradient > 1e-10 * scale) or np.any(right_gradient < -1e-10 * scale):
            raise ActionDomainError("Radial domain does not contain a single monotone-sided well")
        self.energy_limit_hartree = float(np.min(self.values(np.asarray(self.domain))))
        if not self.energy_limit_hartree > 0:
            raise ActionDomainError("Both domain edges must lie above the well minimum")
        step = min(1e-4 * max(1.0, self.equilibrium_bohr),
                   0.1 * (self.equilibrium_bohr - self.domain[0]),
                   0.1 * (self.domain[1] - self.equilibrium_bohr))
        curvature = (float(self._gradient(self.equilibrium_bohr + step))
                     - float(self._gradient(self.equilibrium_bohr - step))) / (2 * step)
        if not np.isfinite(curvature) or curvature <= 0:
            raise ActionDomainError("Well minimum must have positive finite curvature")
        self.harmonic_omega = float(np.sqrt(curvature / self.reduced_mass))

    @classmethod
    def from_bkmp2(cls, potential=None, *, library=None, reduced_mass=H2_REDUCED_MASS,
                   domain=(0.2, 12.0), **kwargs):
        """Reuse or construct the verified BKMP2 provider, r01=r, r12=r02=99."""
        from benchmarks.h3.pes import BKMP2Potential, SOURCE
        if potential is not None and library is not None:
            raise ValueError("Supply a potential or a library, not both")
        if len(domain) != 2 or not 0.2 <= domain[0] < domain[1] < 198.0:
            raise ActionDomainError("BKMP2 domain requires 0.2 <= r_min < r_max < 198 bohr")
        potential = BKMP2Potential(library) if potential is None else potential
        provenance = {"model": "BKMP2 isolated H2", "spectator_pair_distances_bohr": [99.0, 99.0],
                      "source": SOURCE, "build": getattr(potential, "provenance", None)}
        kwargs.setdefault("equilibrium_bracket", (1.2, 1.6))
        return cls(lambda r: potential.radial([r, 99.0, 99.0])[0], reduced_mass,
                   gradient=lambda r: potential.radial([r, 99.0, 99.0])[1][0],
                   domain=domain, provenance=provenance, **kwargs)

    @property
    def metadata(self):
        return {"domain_bohr": list(self.domain), "equilibrium_bohr": self.equilibrium_bohr,
                "minimum_absolute_hartree": self.minimum_hartree,
                "boundary_energy_limit_hartree": self.energy_limit_hartree,
                "harmonic_omega_au": self.harmonic_omega,
                "quadrature": {"initial_order": self.quadrature_order,
                               "maximum_order": self.max_quadrature_order,
                               "relative_tolerance": self.quadrature_rtol,
                               "absolute_tolerance_au": self.quadrature_atol},
                "inverse_time_cdf": {"initial_cells": self.time_cells,
                                     "maximum_cells": self.max_time_cells,
                                     "angle_change_tolerance_rad": self.time_tolerance},
                "energy_zero": "optimized isolated-diatom potential minimum",
                "provenance": self.provenance}

    def _evaluate(self, function, r):
        r = np.asarray(r, dtype=float)
        if not np.isfinite(r).all() or np.any((r < self.domain[0]) | (r > self.domain[1])):
            raise ActionDomainError("Radial evaluation would leave the supplied potential domain")
        values = np.fromiter((float(function(float(x))) for x in r.ravel()), dtype=float, count=r.size).reshape(r.shape)
        if not np.isfinite(values).all():
            raise ActionDomainError("Radial potential or derivative returned a nonfinite value")
        return float(values) if r.ndim == 0 else values

    def values(self, r):
        """Actual radial potential relative to its well minimum."""
        return self._evaluate(self._potential, r) - self.minimum_hartree

    def gradients(self, r):
        return self._evaluate(self._gradient, r)

    def turning_points(self, energy):
        energy = _positive(energy, "energy above the minimum")
        if energy >= self.energy_limit_hartree:
            raise ActionDomainError("Energy reaches a domain edge; two interior turning points are not available")
        function = lambda r: self.values(r) - energy
        left = brentq(function, self.domain[0], self.equilibrium_bohr, xtol=5e-15, rtol=4*np.finfo(float).eps)
        right = brentq(function, self.equilibrium_bohr, self.domain[1], xtol=5e-15, rtol=4*np.finfo(float).eps)
        if not self.domain[0] < left < self.equilibrium_bohr < right < self.domain[1]:
            raise ActionDomainError("Turning points are not strictly inside the radial domain")
        if not self.gradients(left) < 0 < self.gradients(right):
            raise ActionDomainError("Turning points must be simple roots with opposite slopes")
        return float(left), float(right)

    def _coordinates_and_gap(self, theta, energy, turning):
        """Evaluate actual E-V stably, including arbitrarily close to a root.

        The force integral near an endpoint avoids magnifying a few ulps of
        root residual into a spurious nonzero turning-point momentum. Its
        energy zero differs from E only by the reported root residual.
        """
        theta = np.asarray(theta, dtype=float)
        left, right = turning
        width = right - left
        dl, dr = width*np.sin(theta/2)**2, width*np.cos(theta/2)**2
        r = np.where(theta <= np.pi/2, left + dl, right - dr)
        gap = np.asarray(energy - self.values(r))
        # Avoid subtracting nearly equal potential energies near a root.
        # Integrate the analytic slope over the tiny displacement instead.
        nodes, weights = _gauss(3)
        for mask, distance, sign, start in ((dl < width*1e-5, dl, -1.0, left),
                                            (dr < width*1e-5, dr, 1.0, right)):
            if np.any(mask):
                d = distance[mask]
                positions = (start + d[:, None]*(nodes[None, :]+1)/2 if sign < 0
                             else start - d[:, None]*(nodes[None, :]+1)/2)
                gap[mask] = sign*d/2 * (self.gradients(positions) @ weights)
        if np.any(gap < 0):
            raise ActionDomainError("Encountered a forbidden interior segment of the proposed bound orbit")
        return r, gap

    def _regularized(self, theta, energy, turning):
        """Return action and half-period integrands in the smooth theta variable."""
        theta = np.asarray(theta, dtype=float)
        left, right = turning
        width = right-left
        _, gap = self._coordinates_and_gap(theta, energy, turning)
        jacobian = 0.5*width*np.sin(theta)
        momentum = np.sqrt(2*self.reduced_mass*gap)
        action = momentum*jacobian/np.pi
        with np.errstate(divide="ignore", invalid="ignore"):
            time = self.reduced_mass*jacobian/momentum
        at_left, at_right = theta == 0, theta == np.pi
        time = np.where(at_left, np.sqrt(self.reduced_mass*width/(-2*self.gradients(left))), time)
        time = np.where(at_right, np.sqrt(self.reduced_mass*width/(2*self.gradients(right))), time)
        if not np.isfinite(action).all() or not np.isfinite(time).all() or np.any(time <= 0):
            raise ActionConvergenceError("Nonfinite or nonpositive regularized orbit integrand")
        return action, time

    def _integrals(self, energy, turning=None):
        turning = self.turning_points(energy) if turning is None else turning
        previous = None
        order = self.quadrature_order
        while order <= self.max_quadrature_order:
            nodes, weights = _gauss(order)
            action, time = self._regularized((nodes+1)*np.pi/2, energy, turning)
            current = np.array([action @ weights, time @ weights]) * np.pi/2
            if previous is not None:
                difference = np.abs(current-previous)
                if np.all(difference <= self.quadrature_atol + self.quadrature_rtol*np.abs(current)):
                    return float(current[0]), float(current[1]), {
                        "quadrature_order": order, "action_quadrature_change_au": float(difference[0]),
                        "half_period_quadrature_change_au": float(difference[1])}
            previous, order = current, order*2
        raise ActionConvergenceError("Action/period quadrature did not converge within max_quadrature_order")

    def action(self, energy):
        """Return J=integral(p_r dr)/pi, in units where hbar=1."""
        if float(energy) == 0:
            return 0.0
        return self._integrals(energy)[0]

    def _time_map(self, energy, turning, half_period):
        nodes, weights = _gauss(4)
        previous_inverse = previous_period = None
        cells = self.time_cells
        check_fractions = np.linspace(0, 1, 4097)
        while cells <= self.max_time_cells:
            theta = np.linspace(0, np.pi, cells+1)
            step = np.pi/cells
            quadrature_theta = theta[:-1, None] + step*(nodes[None, :]+1)/2
            _, integrand = self._regularized(quadrature_theta.ravel(), energy, turning)
            increments = step/2*(integrand.reshape(cells, -1) @ weights)
            accumulated = np.r_[0.0, np.cumsum(increments)]
            period = float(accumulated[-1])
            fractions = accumulated/period
            if not np.all(np.diff(fractions) > 0):
                raise ActionConvergenceError("Time CDF is not strictly monotone")
            inverse = PchipInterpolator(fractions, theta, extrapolate=False)
            if previous_inverse is not None:
                angle_change = float(np.max(np.abs(inverse(check_fractions)-previous_inverse(check_fractions))))
                period_change = abs(period-previous_period)/period
                agreement = abs(period-half_period)/half_period
                if (angle_change < self.time_tolerance and period_change < self.time_tolerance
                        and agreement < max(self.time_tolerance, 5*self.quadrature_rtol)):
                    return inverse, {"time_cdf_cells": cells,
                                     "inverse_time_max_angle_change_rad": angle_change,
                                     "time_cdf_half_period_relative_change": period_change,
                                     "time_cdf_vs_independent_period_relative_difference": agreement}
            previous_inverse, previous_period, cells = inverse, period, cells*2
        raise ActionConvergenceError("Inverse time CDF did not converge within max_time_cells")

    def orbit_at_energy(self, energy, *, v=None):
        """Prepare a time-uniform classical orbit at a specified relative energy."""
        energy = _positive(energy, "energy above the minimum")
        turning = self.turning_points(energy)
        action, half_period, diagnostics = self._integrals(energy, turning)
        inverse, time_diagnostics = self._time_map(energy, turning, half_period)
        diagnostics.update(time_diagnostics)
        diagnostics["turning_point_max_energy_residual_hartree"] = float(np.max(np.abs(self.values(np.asarray(turning))-energy)))
        return DiatomicOrbit(energy, energy+self.minimum_hartree, action, 2*half_period,
                             turning, self.reduced_mass, v, diagnostics, self, inverse)

    def quantize(self, v):
        """Solve J(E)=v+1/2 without replacing it by a harmonic or quantum energy."""
        v = _integer(v, "v", 0)
        target = v+0.5
        ceiling = self.energy_limit_hartree*(1-1e-9)
        upper = min(1.25*target*self.harmonic_omega, ceiling)
        while self.action(upper) < target:
            if upper == ceiling:
                raise ActionDomainError(f"v={v} cannot be quantized inside domain {self.domain}; expand a valid bound-state domain or choose a lower action")
            upper = min(2*upper, ceiling)
        energy = brentq(lambda e: self.action(e)-target, 0.0, upper,
                        xtol=5e-15, rtol=4*np.finfo(float).eps, maxiter=128)
        result = self.orbit_at_energy(energy, v=v)
        if abs(result.action-target) > max(5*self.quadrature_atol, 5*self.quadrature_rtol*target):
            raise ActionConvergenceError("Quantized energy does not satisfy the requested action tolerance")
        return result
