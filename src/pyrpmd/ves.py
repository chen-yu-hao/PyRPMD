"""Variational enhanced sampling of one or two centroid collective variables.

The bias is a tensor-product Legendre expansion, without its constant gauge
mode. For a fixed target density p, the coefficient gradient is
``<f>_p - <f>_V`` and its Hessian is ``beta * cov_V(f)``. Averaged stochastic
gradient descent uses the full covariance, and applies the *averaged*
coefficients to the dynamics. See Valsson & Parrinello, PRL 113, 090601 (2014),
doi:10.1103/PhysRevLett.113.090601, and JCTC 11, 1996 (2015),
doi:10.1021/acs.jctc.5b00076 for the adaptive well-tempered target.

All energies and coordinates use atomic units. Optimization is conditional on
the specified finite CV domain: exterior observations are counted and excluded,
not clipped into the statistics. A bounded, continuously differentiable tangent
extension of each polynomial and harmonic exterior walls define the MD forces.
The walls vanish inside the domain and do not change its conditional density.
Finite-basis stationarity is not a claim of physical or statistical convergence.
"""

from __future__ import annotations

from itertools import product
from collections.abc import Mapping

import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.special import logsumexp

from .cv import CVStack
from .validation import checked_integer, finite_positive


def _array(value, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numbers")
    if shape is not None and raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return np.array(raw, dtype=np.float64, copy=True)


def _counts(values, dimension, name, minimum=1, maximum=64):
    if np.isscalar(values):
        values = [values] * dimension
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != dimension:
        raise ValueError(f"{name} needs one integer per CV")
    return tuple(checked_integer(v, name, minimum, maximum) for v in values)


class LegendreBasis:
    """Gauge-free tensor products on a finite one- or two-dimensional domain.

    ``evaluate`` returns ``[..., K]`` and ``values_and_gradients`` additionally
    returns ``[..., K, D]``. Orders include all degrees 0..order on each axis;
    only the all-zero multi-index is removed. Exterior basis values use
    ``L(b) + L'(b)*d/(1+abs(d))``, b=+/-1, d=z-b, in scaled coordinates z.
    This extension is bounded and C1 at both boundaries. It is used only for
    forces; target quadrature and moment accumulation stay inside the domain.
    """

    def __init__(self, ranges, orders=4):
        self.ranges = _array(ranges, "VES ranges")
        if self.ranges.ndim != 2 or self.ranges.shape[1] != 2 or len(self.ranges) not in (1, 2):
            raise ValueError("VES ranges must contain one or two [lower, upper] pairs")
        if np.any(self.ranges[:, 1] <= self.ranges[:, 0]):
            raise ValueError("VES ranges must increase")
        self.dimension = len(self.ranges)
        self.orders = _counts(orders, self.dimension, "VES orders", maximum=32)
        if np.prod(np.asarray(self.orders) + 1) - 1 > 255:
            raise ValueError("VES expansion exceeds 255 nonconstant basis functions")
        self.indices = np.asarray(list(product(*(range(n + 1) for n in self.orders)))[1:], dtype=int)
        self.size = len(self.indices)
        self._scale = 2.0 / (self.ranges[:, 1] - self.ranges[:, 0])
        if not np.isfinite(self._scale).all() or np.any(self._scale <= 0):
            raise ValueError("VES ranges have an unrepresentable width")
        self.ranges.flags.writeable = False
        self.indices.flags.writeable = False

    def _points(self, values):
        values = _array(values, "VES CV values")
        if values.ndim < 1 or values.shape[-1] != self.dimension:
            raise ValueError("VES CV values must have shape [..., dimension]")
        return values

    def contains(self, values):
        values = self._points(values)
        return np.all((values >= self.ranges[:, 0]) & (values <= self.ranges[:, 1]), axis=-1)

    def _axis(self, z, order, gradient):
        clipped = np.clip(z, -1.0, 1.0)
        values = legvander(clipped, order).reshape(clipped.shape + (order + 1,))
        exterior = np.abs(z) > 1.0
        deriv = np.zeros_like(values)
        # The differentiated three-term recurrence is regular at +/-1.
        if order:
            deriv[..., 1] = 1.0
        for n in range(1, order):
            deriv[..., n + 1] = ((2 * n + 1) * (values[..., n] + clipped * deriv[..., n])
                                - n * deriv[..., n - 1]) / (n + 1)
        if np.any(exterior):
            d = z - clipped
            magnitude = np.abs(d)
            # d/(1+|d|) written this way stays finite for very large values.
            bounded = np.sign(d) * (1.0 - 1.0 / (1.0 + magnitude))
            values = values + deriv * bounded[..., None]
            deriv = deriv / (1.0 + magnitude[..., None])**2
        return values, deriv if gradient else None

    def _evaluate(self, values, gradient):
        points = self._points(values)
        z = (points - self.ranges[:, 0]) * self._scale - 1.0
        if not np.isfinite(z).all():
            raise ValueError("VES scaled CV values overflow")
        axes = [self._axis(z[..., d], order, gradient) for d, order in enumerate(self.orders)]
        selected = [a[0][..., self.indices[:, d]] for d, a in enumerate(axes)]
        basis = np.ones(points.shape[:-1] + (self.size,))
        for values_d in selected:
            basis *= values_d
        if not gradient:
            return basis
        derivatives = np.empty(basis.shape + (self.dimension,))
        for d, axis in enumerate(axes):
            component = axis[1][..., self.indices[:, d]] * self._scale[d]
            for other in range(self.dimension):
                if other != d:
                    component = component * selected[other]
            derivatives[..., d] = component
        return basis, derivatives

    def evaluate(self, values):
        return self._evaluate(values, False)

    def values_and_gradients(self, values):
        return self._evaluate(values, True)

    def quadrature(self, order=32):
        """Return tensor Gauss-Legendre points and positive Lebesgue weights."""
        orders = _counts(order, self.dimension, "VES quadrature_order", maximum=256)
        if any(n < p + 1 for n, p in zip(orders, self.orders)):
            raise ValueError("VES quadrature_order must exceed the basis order on each axis")
        if np.prod(orders) * self.size > 4_000_000:
            raise ValueError("VES quadrature exceeds four million basis entries")
        axes, weights = [], []
        for d, n in enumerate(orders):
            nodes, w = leggauss(n)
            axes.append(self.ranges[d, 0] + (nodes + 1.0) / self._scale[d])
            weights.append(w / self._scale[d])
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, self.dimension)
        weight = np.prod(np.stack(np.meshgrid(*weights, indexing="ij"), axis=-1), axis=-1).ravel()
        return points, weight


def validate_centroid_cv(cv):
    """Require an explicit centroid contract and nonperiodic CV domain."""
    if getattr(cv, "dimension", None) not in (1, 2):
        raise ValueError("VES supports one or two centroid CV dimensions")
    components = cv.cvs if isinstance(cv, CVStack) else (cv,)
    if any(getattr(c, "estimator", None) != "centroid" for c in components):
        raise ValueError("VES requires explicitly declared centroid CVs")
    if len(cv.periods) != cv.dimension or any(p is not None for p in cv.periods):
        raise ValueError("VES Legendre expansion requires nonperiodic CVs")
    return cv


class VESBias:
    """Linear VES bias with resumable block statistics and averaged SGD.

    ``observe`` consumes one production MD observation; coefficients update
    every ``pace`` calls. The learning rate has energy units (default 0.5/beta).
    ``target_update_interval`` is in coefficient iterations, not MD steps.
    Well-tempered target log densities are represented in the same basis and
    integrated by Gauss-Legendre quadrature in log space. Increase quadrature
    order and basis order separately in convergence studies.

    ``physical_energy`` includes the walls. ``energy_force`` returns P times
    this physical energy and its exact bead derivatives, as required by H_P.
    ``free_energy`` is the current finite-stage variational estimate, using
    the target against which the last coefficient update was made. It never
    substitutes the asymptotic well-tempered scaling at finite iteration.
    """

    method = "ves_legendre_averaged_sgd"

    def __init__(self, cv, beta, ranges, orders=4, *, target="uniform", bias_factor=10.0,
                 learning_rate=None, pace=100, target_update_interval=100,
                 quadrature_order=32, wall_stiffness=1.0, state=None):
        self.cv = validate_centroid_cv(cv)
        self.beta = finite_positive(beta, "VES beta", coerce=False)
        self.basis = LegendreBasis(ranges, orders)
        if self.basis.dimension != cv.dimension:
            raise ValueError("VES ranges disagree with CV dimension")
        if target not in ("uniform", "well_tempered"):
            raise ValueError("VES target must be uniform or well_tempered")
        self.target = target
        self.bias_factor = finite_positive(bias_factor, "VES bias_factor", coerce=False)
        if self.bias_factor <= 1:
            raise ValueError("VES bias_factor must exceed one")
        self.learning_rate = finite_positive(0.5 / self.beta if learning_rate is None else learning_rate,
                                             "VES learning_rate", coerce=False)
        self.pace = checked_integer(pace, "VES pace")
        self.target_update_interval = checked_integer(target_update_interval, "VES target_update_interval")
        self.quadrature_order = _counts(quadrature_order, cv.dimension, "VES quadrature_order", maximum=256)
        self.quadrature_points, weights = self.basis.quadrature(self.quadrature_order)
        self._quadrature_log_weights = np.log(weights)
        self._quadrature_basis = self.basis.evaluate(self.quadrature_points)
        wall = _array(wall_stiffness, "VES wall_stiffness")
        if wall.ndim == 0:
            wall = np.full(cv.dimension, float(wall))
        if wall.shape != (cv.dimension,) or np.any(wall <= 0):
            raise ValueError("VES wall_stiffness must be positive with one value per CV")
        self.wall_stiffness = wall
        size = self.basis.size
        self.coefficients = np.zeros(size)
        self.instantaneous = np.zeros(size)
        self.target_coefficients = np.zeros(size)
        self.reconstruction_target = np.zeros(size)
        self.observations = self.outside = self.pending_count = self.pending_inside = 0
        self.updates = self.target_updates = 0
        self._mean = np.zeros(size)
        self._m2 = np.zeros((size, size))
        self.last_gradient = np.zeros(size)
        self.last_covariance = np.zeros((size, size))
        self._refresh_target()
        if state is not None:
            self.load_state_dict(state)

    def _target_statistics(self, coefficients):
        logs = self._quadrature_basis @ coefficients
        log_z = float(logsumexp(logs + self._quadrature_log_weights))
        mass = np.exp(logs + self._quadrature_log_weights - log_z)
        mean = mass @ self._quadrature_basis
        if not np.isfinite(log_z) or not np.isfinite(mean).all():
            raise FloatingPointError("VES target normalization overflow")
        # All retained Legendre functions integrate to zero for a uniform target.
        if not np.any(coefficients):
            mean = np.zeros(self.basis.size)
        return log_z, mean

    def _refresh_target(self):
        self.target_log_normalization, self.target_mean = self._target_statistics(self.target_coefficients)

    def log_target(self, values):
        """Normalized target log density; exactly -inf outside its domain."""
        value = self.basis.evaluate(values) @ self.target_coefficients - self.target_log_normalization
        return np.where(self.basis.contains(values), value, -np.inf)

    def _walls(self, values):
        points = self.basis._points(values)
        displacement = points - np.clip(points, self.basis.ranges[:, 0], self.basis.ranges[:, 1])
        return 0.5 * np.sum(self.wall_stiffness * displacement**2, axis=-1), self.wall_stiffness * displacement

    def physical_energy(self, values):
        energy = self.basis.evaluate(values) @ self.coefficients + self._walls(values)[0]
        if not np.isfinite(energy).all():
            raise FloatingPointError("VES bias energy overflow")
        return energy

    def energy_force(self, q):
        q = _array(q, "VES coordinates")
        if q.ndim != 3 or q.shape[-1] != 3 or min(q.shape) < 1:
            raise ValueError("VES coordinates must have nonempty [P,N,3] shape")
        geometry = self.cv.value_and_jacobian(q)
        values = _array(geometry.value, "VES CV values", (self.cv.dimension,))
        jac = _array(geometry.jacobian, "VES CV Jacobian", (self.cv.dimension,) + q.shape)
        if not np.allclose(jac, jac.sum(axis=1)[:, None] / len(q), rtol=1e-11, atol=1e-13):
            raise ValueError("VES centroid CV Jacobian must be identical on every bead")
        basis, gradient = self.basis.values_and_gradients(values)
        wall, wall_gradient = self._walls(values)
        energy = len(q) * float(basis @ self.coefficients + wall)
        force = -len(q) * np.einsum("d,dpac->pac", self.coefficients @ gradient + wall_gradient, jac)
        if not np.isfinite(energy) or not np.isfinite(force).all():
            raise FloatingPointError("VES bias energy or force overflow")
        return energy, force

    def free_energy(self, values):
        """Return a finite-stage F estimate up to a constant, only in domain."""
        if not np.all(self.basis.contains(values)):
            raise ValueError("VES free-energy estimates are restricted to the optimization domain")
        return self.basis.evaluate(values) @ (-self.coefficients - self.reconstruction_target / self.beta)

    def _advance(self, mean, covariance):
        gradient = self.target_mean - mean
        hessian = self.beta * covariance
        instantaneous = self.instantaneous - self.learning_rate * (
            gradient + hessian @ (self.instantaneous - self.coefficients))
        averaged = self.coefficients + (instantaneous - self.coefficients) / (self.updates + 2)
        if not np.isfinite(instantaneous).all() or not np.isfinite(averaged).all():
            raise FloatingPointError("VES optimizer overflow; reduce learning_rate")
        target_coefficients = self.target_coefficients.copy()
        target_log_z, target_mean = self.target_log_normalization, self.target_mean.copy()
        update_target = self.target == "well_tempered" and (self.updates + 1) % self.target_update_interval == 0
        if update_target:
            # F_new=-V_new-log(p_old)/beta; constants cancel on normalization.
            target_coefficients = (self.beta * averaged + target_coefficients) / self.bias_factor
            target_log_z, target_mean = self._target_statistics(target_coefficients)
        # Commit only after every numerical operation has succeeded.
        self.instantaneous = instantaneous
        self.coefficients = averaged
        self.reconstruction_target = self.target_coefficients.copy()
        self.target_coefficients = target_coefficients
        self.target_log_normalization, self.target_mean = target_log_z, target_mean
        self.updates += 1
        self.target_updates += int(update_target)
        self.last_gradient = gradient.copy()
        self.last_covariance = covariance.copy()

    def observe(self, values):
        """Accumulate a production observation and atomically update at pace.

        Exterior points contribute to the MD clock and exterior count only.
        An entirely exterior block fails explicitly; widen the domain or
        improve confinement/equilibration instead of optimizing empty moments.
        """
        values = _array(values, "VES observation", (self.cv.dimension,))
        inside = bool(self.basis.contains(values))
        n = self.pending_inside + int(inside)
        mean, m2 = self._mean.copy(), self._m2.copy()
        if inside:
            feature = self.basis.evaluate(values)
            delta = feature - mean
            mean += delta / n
            m2 += np.outer(delta, feature - mean)
        if self.pending_count + 1 == self.pace:
            if n == 0:
                raise ValueError("VES block has no observations inside its domain; check ranges, walls and equilibration")
            covariance = (m2 + m2.T) / (2.0 * n)
            self._advance(mean, covariance)
            mean = np.zeros_like(mean)
            m2 = np.zeros_like(m2)
            n = 0
        self._mean, self._m2 = mean, m2
        self.observations += 1
        self.outside += int(not inside)
        self.pending_count = (self.pending_count + 1) % self.pace
        self.pending_inside = n

    def update_from_moments(self, mean, covariance):
        """Consume a complete block summarized by its mean and population covariance.

        This supports independently integrated reference moments or external
        samplers. It records one in-domain block of ``pace`` observations and
        must not be mixed with a partially accumulated block.
        """
        if self.pending_count:
            raise ValueError("VES supplied moments require an empty pending block")
        mean = _array(mean, "VES basis mean", (self.basis.size,))
        covariance = _array(covariance, "VES basis covariance", (self.basis.size, self.basis.size))
        if np.any(np.abs(mean) > 1.0 + 1e-12):
            raise ValueError("VES in-domain Legendre means must lie in [-1, 1]")
        self._validate_covariance(covariance, "VES supplied covariance")
        self._advance(mean, covariance)
        self.observations += self.pace

    @staticmethod
    def _validate_covariance(value, name):
        scale = max(1.0, float(np.max(np.abs(value))))
        if (not np.allclose(value, value.T, rtol=0, atol=1e-12 * scale)
                or np.linalg.eigvalsh(value)[0] < -1e-10 * scale):
            raise ValueError(f"{name} must be symmetric positive semidefinite")

    def _parameters(self):
        return {"beta": self.beta, "ranges": self.basis.ranges.tolist(), "orders": list(self.basis.orders),
                "target": self.target, "bias_factor": self.bias_factor, "learning_rate": self.learning_rate,
                "pace": self.pace, "target_update_interval": self.target_update_interval,
                "quadrature_order": list(self.quadrature_order), "wall_stiffness": self.wall_stiffness.tolist(),
                "basis": "legendre_tensor_without_constant", "optimizer": "averaged_sgd_full_covariance",
                "boundary": "bounded_tangent_extension_conditional_moments"}

    def state_dict(self):
        """Return complete JSON-safe optimizer, target and partial-block state."""
        return {"schema_version": 1, "parameters": self._parameters(),
                "coefficients": self.coefficients.tolist(), "instantaneous": self.instantaneous.tolist(),
                "target_coefficients": self.target_coefficients.tolist(),
                "reconstruction_target": self.reconstruction_target.tolist(),
                "observations": self.observations, "outside": self.outside,
                "pending_count": self.pending_count, "pending_inside": self.pending_inside,
                "updates": self.updates, "target_updates": self.target_updates,
                "mean": self._mean.tolist(), "m2": self._m2.tolist(),
                "last_gradient": self.last_gradient.tolist(), "last_covariance": self.last_covariance.tolist()}

    def load_state_dict(self, state):
        """Validate every state field before changing a live bias."""
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("invalid VES checkpoint fields")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("unsupported VES checkpoint schema")
        if state["parameters"] != self._parameters():
            raise ValueError("VES checkpoint parameters differ")
        counters = {k: checked_integer(state[k], "VES checkpoint " + k, 0)
                    for k in ("observations", "outside", "pending_count", "pending_inside", "updates", "target_updates")}
        count, pending, n, updates = (counters[k] for k in ("observations", "pending_count", "pending_inside", "updates"))
        if (count != updates * self.pace + pending or pending >= self.pace or n > pending
                or counters["outside"] > count or counters["outside"] < pending - n
                or count - counters["outside"] < updates + n):
            raise ValueError("VES checkpoint statistics disagree with observation clock")
        expected_target = updates // self.target_update_interval if self.target == "well_tempered" else 0
        if counters["target_updates"] != expected_target:
            raise ValueError("VES checkpoint target clock disagrees with optimizer clock")
        size = self.basis.size
        arrays = {k: _array(state[k], "VES checkpoint " + k, (size,)) for k in
                  ("coefficients", "instantaneous", "target_coefficients", "reconstruction_target", "mean", "last_gradient")}
        arrays.update({k: _array(state[k], "VES checkpoint " + k, (size, size)) for k in ("m2", "last_covariance")})
        self._validate_covariance(arrays["m2"], "VES checkpoint m2")
        self._validate_covariance(arrays["last_covariance"], "VES checkpoint covariance")
        if (np.any(np.abs(arrays["mean"]) > 1.0 + 1e-12)
                or (n == 0 and np.any(arrays["mean"]))
                or (n < 2 and np.any(arrays["m2"]))
                or (updates == 0 and any(np.any(arrays[k]) for k in
                    ("coefficients", "instantaneous", "target_coefficients", "reconstruction_target", "last_gradient", "last_covariance")))):
            raise ValueError("VES checkpoint moments disagree with their counts")
        if self.target == "uniform" or counters["target_updates"] == 0:
            if np.any(arrays["target_coefficients"]) or np.any(arrays["reconstruction_target"]):
                raise ValueError("VES checkpoint has an invalid uniform target")
        elif updates % self.target_update_interval == 0:
            expected = (self.beta * arrays["coefficients"] + arrays["reconstruction_target"]) / self.bias_factor
            if not np.array_equal(arrays["target_coefficients"], expected):
                raise ValueError("VES checkpoint target update disagrees with coefficients")
        elif not np.array_equal(arrays["target_coefficients"], arrays["reconstruction_target"]):
            raise ValueError("VES checkpoint reconstruction target disagrees with optimizer")
        log_z, target_mean = self._target_statistics(arrays["target_coefficients"])
        for key, value in counters.items():
            setattr(self, key, value)
        for key, value in arrays.items():
            setattr(self, "_" + key if key in ("mean", "m2") else key, value)
        self.target_log_normalization, self.target_mean = log_z, target_mean
