"""Biases return the energy added to H_P and its full physical force.

User umbrella energies W are physical energies: H_P receives P*W. The flux
metric is a direct mathematical port of RPMDrate/Caracal's MIT-licensed
add_bias_potential/umbrella routines; see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations
import hashlib
import itertools
import json
import numpy as np


def displacement(values, centers, periods=None):
    delta = np.asarray(values, float) - np.asarray(centers, float)
    if periods is not None:
        for i, period in enumerate(periods):
            if period is not None:
                delta[..., i] = (delta[..., i] + period / 2) % period - period / 2
    return delta


class UmbrellaBias:
    def __init__(self, cv, center, stiffness):
        self.cv = cv
        self.center = np.asarray(center, float)
        k = np.asarray(stiffness, float)
        if k.ndim == 0:
            k = np.eye(cv.dimension) * k
        if k.ndim == 1:
            k = np.diag(k)
        if self.center.shape != (cv.dimension,) or k.shape != (
            cv.dimension,
            cv.dimension,
        ):
            raise ValueError("umbrella center/stiffness has the wrong CV dimension")
        if not np.isfinite(k).all() or not np.isfinite(self.center).all():
            raise ValueError("umbrella parameters must be finite")
        if not np.allclose(k, k.T) or np.linalg.eigvalsh(k).min() <= 0:
            raise ValueError("umbrella stiffness must be symmetric positive definite")
        self.stiffness = k

    def physical_energy(self, values):
        d = displacement(values, self.center, self.cv.periods)
        return 0.5 * np.einsum("...i,ij,...j->...", d, self.stiffness, d)

    def energy_force(self, q):
        result = self.cv.value_and_jacobian(q)
        d = displacement(result.value, self.center, self.cv.periods)
        p = len(q)
        energy = p * 0.5 * float(d @ self.stiffness @ d)
        forces = -p * np.einsum("d,dpac->pac", self.stiffness @ d, result.jacobian)
        return energy, forces


class FluxMetricBias:
    """-P*log(fs)/beta with fs=sqrt(g M^-1 g/(2*pi*beta)).

    Only centroid CVs with an analytic centroid Hessian are accepted. This
    belongs to the RPMDrate rate-PMF measure, not generic FES.
    """

    def __init__(self, cv, masses, beta):
        if getattr(cv, "estimator", None) != "centroid":
            raise ValueError("rate flux metric requires a centroid coordinate")
        if not hasattr(cv, "centroid_value_gradient_hessian"):
            raise ValueError("rate flux metric requires the CV Hessian")
        self.cv = cv
        self.masses = np.asarray(masses, float)
        self.beta = float(beta)

    def energy_force(self, q):
        _, g, h = self.cv.centroid_value_gradient_hessian(np.mean(q, axis=0))
        z = float(np.sum(g * g / self.masses[:, None]))
        if not np.isfinite(z) or z <= 1e-30:
            raise ValueError("rate coordinate has a zero/invalid mass metric")
        fs = np.sqrt(z / (2 * np.pi * self.beta))
        f = np.einsum("acbd,bd->ac", h, g / self.masses[:, None]) / (self.beta * z)
        return -len(q) * np.log(fs) / self.beta, np.broadcast_to(f, q.shape).copy()


class CompositeBias:
    def __init__(self, *biases):
        self.biases = tuple(b for b in biases if b is not None)

    def energy_force(self, q):
        energy = 0.0
        force = np.zeros_like(q)
        for bias in self.biases:
            e, f = bias.energy_force(q)
            energy += e
            force += f
        return energy, force


_BUILTIN_BIAS_METHODS = {
    cls: cls.energy_force for cls in (UmbrellaBias, FluxMetricBias, CompositeBias)
}


class WallBias:
    """Scalar CV walls, W=kappa*max(lower-s,0)^2+kappa*max(s-upper,0)^2.

    The convention matches PLUMED EXP=2, EPS=1, OFFSET=0 (no factor 1/2).
    Bounds and kappa are canonical units. The full ring Jacobian gives forces
    for centroid or bead-average CVs; the Hamiltonian receives P*W.
    """

    def __init__(self, cv, kappa, *, lower=None, upper=None):
        self.cv = cv
        self.kappa = float(kappa)
        self.lower = None if lower is None else float(lower)
        self.upper = None if upper is None else float(upper)
        if cv.dimension != 1 or tuple(cv.periods) != (None,):
            raise ValueError("walls require a scalar nonperiodic CV")
        if not np.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("wall kappa must be positive and finite")
        if lower is None and upper is None:
            raise ValueError("wall requires lower or upper")
        if any(x is not None and not np.isfinite(x) for x in (self.lower, self.upper)):
            raise ValueError("wall bounds must be finite")
        if lower is not None and upper is not None and self.lower >= self.upper:
            raise ValueError("wall lower must be below upper")

    def energy_force(self, q):
        result = self.cv.value_and_jacobian(q)
        value = float(result.value[0])
        delta = (min(value - self.lower, 0.0) if self.lower is not None else 0.0)
        delta += max(value - self.upper, 0.0) if self.upper is not None else 0.0
        return (len(q) * self.kappa * delta**2,
                -2.0 * len(q) * self.kappa * delta * result.jacobian[0])


class MetadynamicsBias:
    """History-dependent standard or well-tempered metadynamics in CV space.

    Heights are physical energies: ``energy_force`` adds P*V to H_P and
    differentiates the CV's *full* ring Jacobian.  The tempering exponent is
    -beta*V/(gamma-1), never -beta*P*V/(gamma-1).  See Barducci, Bussi and
    Parrinello, PRL 100, 020603 (2008), doi:10.1103/PhysRevLett.100.020603.

    A periodic dimension uses a wrapped Gaussian (a sum over images), not a
    minimum-image Gaussian with a discontinuous force at half a period.
    Depositions are explicit events; evaluating forces never changes history.
    This class deliberately has no frozen native-bias descriptor.

    Change history through ``deposit`` or ``truncate_history``. ``hills``
    returns a serializable snapshot, so editing its records cannot invalidate
    force caches. Pass a modified snapshot to a new bias to reanalyze it.
    """

    history_dependent = True
    _signed_bias_before = False

    def __init__(self, cv, sigma, height, beta, *, mode="well_tempered",
                 bias_factor=None, hills=None):
        self.cv = cv
        self.sigma = np.atleast_1d(np.asarray(sigma, float)).copy()
        if self.sigma.shape != (cv.dimension,) or not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0):
            raise ValueError("metadynamics sigma must contain one positive finite width per CV")
        self.sigma.flags.writeable = False
        self.height, self.beta = float(height), float(beta)
        if not np.isfinite(self.height) or self.height <= 0 or not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("metadynamics height and beta must be positive and finite")
        if mode not in ("standard", "well_tempered"):
            raise ValueError("metadynamics mode must be standard or well_tempered")
        self.mode = mode
        if mode == "standard":
            if bias_factor is not None:
                raise ValueError("standard metadynamics does not use bias_factor")
            self.bias_factor = None
        else:
            self.bias_factor = float(10.0 if bias_factor is None else bias_factor)
            if not np.isfinite(self.bias_factor) or self.bias_factor <= 1:
                raise ValueError("well-tempered bias_factor must be finite and > 1")
        if len(cv.periods) != cv.dimension or any(
            p is not None and (not np.isfinite(p) or p <= 0) for p in cv.periods
        ):
            raise ValueError("CV periods must be positive finite numbers or None")
        self._hills = []
        self._centers = np.empty((16, cv.dimension))
        self._heights = np.empty(16)
        for hill in ([] if hills is None else hills):
            self._append(hill)

    @property
    def hills(self):
        """A detached, JSON-serializable snapshot of the deposited history."""
        return [{**hill, "center": hill["center"].copy()} for hill in self._hills]

    @property
    def hill_count(self):
        return len(self._hills)

    @staticmethod
    def _kernel(delta, sigma, period, *, gradient=True):
        """Unit-height Gaussian and its derivative, including smooth images."""
        if period is None:
            value = np.exp(-0.5 * (delta / sigma) ** 2)
            return value, -delta / sigma**2 * value if gradient else None
        d = (delta + 0.5 * period) % period - 0.5 * period
        ratio = sigma / period
        if ratio >= 0.2:
            # Poisson summation converges quickly for broad Gaussians. The
            # normalization makes the deposited height exactly the input w.
            count = max(1, int(np.ceil(np.sqrt(2 * 36.0) / (2 * np.pi * ratio))))
            modes = np.arange(1, count + 1, dtype=float)
            coeff = np.exp(-0.5 * (2 * np.pi * ratio * modes) ** 2)
            phase = d[..., None] * (2 * np.pi / period) * modes
            norm = 1.0 + 2 * coeff.sum()
            value = (1.0 + 2 * np.sum(coeff * np.cos(phase), axis=-1)) / norm
            derivative = (-2 * np.sum(coeff * modes * np.sin(phase), axis=-1) *
                          (2 * np.pi / period) / norm) if gradient else None
            return value, derivative
        count = int(np.ceil(8.5 * ratio)) + 1
        images = np.arange(-count, count + 1, dtype=float) * period
        shifts = d[..., None] + images
        terms = np.exp(-0.5 * (shifts / sigma) ** 2)
        norm = np.exp(-0.5 * (images / sigma) ** 2).sum()
        derivative = np.sum(-shifts / sigma**2 * terms, axis=-1) / norm if gradient else None
        return terms.sum(axis=-1) / norm, derivative

    def _history_arrays(self):
        # Numeric buffers grow at deposition; live slices exclude rolled-back
        # hills without rebuilding history dictionaries during force calls.
        return self._centers[:self.hill_count], self._heights[:self.hill_count]

    def _evaluate(self, values, *, gradient):
        values = np.asarray(values, float)
        if values.ndim < 1 or values.shape[-1] != self.cv.dimension or not np.isfinite(values).all():
            raise ValueError("metadynamics values must be finite [...,D]")
        if not self._hills:
            return np.zeros(values.shape[:-1]), np.zeros_like(values) if gradient else None
        centers, heights = self._history_arrays()
        delta = values[..., None, :] - centers
        factors, derivatives = [], []
        for j, (sigma, period) in enumerate(zip(self.sigma, self.cv.periods)):
            f, g = self._kernel(delta[..., j], sigma, period, gradient=gradient)
            factors.append(f)
            derivatives.append(g)
        energy = np.sum(heights * np.prod(factors, axis=0), axis=-1)
        if not gradient:
            return energy, None
        gradient = np.empty_like(values)
        for j in range(self.cv.dimension):
            term = heights * derivatives[j]
            for k in range(self.cv.dimension):
                if k != j:
                    term = term * factors[k]
            gradient[..., j] = np.sum(term, axis=-1)
        return energy, gradient

    def physical_energy_gradient(self, values):
        return self._evaluate(values, gradient=True)

    def physical_energy(self, values):
        return self._evaluate(values, gradient=False)[0]

    def energy_force(self, q):
        result = self.cv.value_and_jacobian(q)
        energy, gradient = self.physical_energy_gradient(result.value)
        p = len(q)
        return p * float(energy), -p * np.einsum("d,dpac->pac", gradient, result.jacobian)

    def _append(self, hill):
        center = np.asarray(hill["center"], float)
        height, time = float(hill["height"]), float(hill["time"])
        step = hill["step"]
        before = float(hill.get("bias_before", 0.0))
        if center.shape != (self.cv.dimension,) or not np.isfinite(center).all():
            raise ValueError("invalid metadynamics hill center")
        if (not np.isfinite([height, time, before]).all() or height < 0 or time < 0
                or (before < -1e-12 and not self._signed_bias_before)):
            raise ValueError("invalid metadynamics hill height/time/bias")
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError("metadynamics hill step must be a positive integer")
        walker = hill.get("walker")
        if walker is not None and (isinstance(walker, (bool, np.bool_))
                or not isinstance(walker, (int, np.integer)) or walker < 0):
            raise ValueError("metadynamics walker must be a nonnegative integer")
        if self._hills:
            previous = self._hills[-1]
            if walker is not None and "walker" in previous:
                if ((step, walker) <= (previous["step"], previous["walker"])
                        or (step == previous["step"] and time != previous["time"])
                        or (step > previous["step"] and time <= previous["time"])):
                    raise ValueError("shared metadynamics hills must increase by (step, walker) with consistent times")
            elif step <= previous["step"] or time <= previous["time"]:
                raise ValueError("metadynamics hill clocks must strictly increase")
        count = self.hill_count
        if count == len(self._heights):
            centers = np.empty((max(16, 2 * count), self.cv.dimension))
            heights = np.empty(len(centers))
            centers[:count], heights[:count] = self._centers, self._heights
            self._centers, self._heights = centers, heights
        self._centers[count], self._heights[count] = center, height
        self._hills.append({"center": center.tolist(), "height": height,
                           "step": int(step), "time": time, "bias_before": before,
                           **({"walker": int(walker)} if walker is not None else {})})

    def truncate_history(self, count):
        """Discard hills after ``count``, for transactional step rollback."""
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or not 0 <= count <= len(self._hills)):
            raise ValueError("history count must be an integer between zero and the current hill count")
        del self._hills[count:]

    def truncate_hills(self, count):
        """Compatibility name for :meth:`truncate_history`."""
        self.truncate_history(count)

    def deposit(self, values, *, step, time, walker=None):
        """Add one hill after a successful integration step, without RNG use."""
        center = np.asarray(values, float)
        if center.shape != (self.cv.dimension,):
            raise ValueError("one metadynamics deposition needs a single [D] center")
        before = float(self.physical_energy(center))
        height = self.height
        if self.mode == "well_tempered":
            height *= np.exp(-self.beta * before / (self.bias_factor - 1.0))
        hill = {"center": center, "height": height, "step": step,
                "time": time, "bias_before": before,
                **({"walker": walker} if walker is not None else {})}
        self._append(hill)
        last = self._hills[-1]
        return {**last, "center": last["center"].copy()}

    def rollback_hills(self, count):
        """Compatibility alias that preserves subclass cache restoration."""
        self.truncate_history(count)

    @property
    def fes_factor(self):
        """Asymptotic F=-factor*V+C; finite-time bias is only an estimate."""
        return 1.0 if self.mode == "standard" else self.bias_factor / (self.bias_factor - 1.0)


class GridMetadynamicsBias(MetadynamicsBias):
    """Opt-in 1D/2D Hermite interpolation of the accumulated physical bias.

    ``grid_bins`` counts intervals, with both boundary nodes retained for a
    nonperiodic CV and one nonduplicated period for a periodic CV. Values and
    first/mixed derivatives of every Gaussian are added to *all* nodes; no
    Gaussian tail cutoff is introduced. Forces differentiate the interpolated
    energy, giving a C1 bias across cell edges and periodic seams.

    Lookup costs O(4**D), independent of hill count. Depositing one hill costs
    O(number of nodes), and HILLS remains an audit history. A checkpoint may
    retain a verified grid cache; without one, replay reconstructs exactly the
    same floating-point sum. No mutable scratch is shared with callers.
    """

    grid_algorithm = "tensor_hermite_v1"
    # Hermite interpolation need not preserve positivity between nodes. The
    # actual signed interpolated V is the bias used by both forces and WT
    # tempering; clipping it would change that algorithm and its derivatives.
    _signed_bias_before = True

    def __init__(self, cv, sigma, height, beta, *, grid_ranges, grid_bins=None,
                 mode="well_tempered", bias_factor=None, hills=None, grid_state=None):
        super().__init__(cv, sigma, height, beta, mode=mode, bias_factor=bias_factor)
        if cv.dimension not in (1, 2):
            raise ValueError("grid metadynamics supports one or two CV dimensions")
        bounds = np.asarray(grid_ranges, dtype=float)
        if bounds.shape != (cv.dimension, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("metadynamics grid ranges must be finite increasing [lower,upper] per CV")
        with np.errstate(over="ignore", invalid="ignore"):
            widths = bounds[:, 1] - bounds[:, 0]
        if not np.isfinite(widths).all():
            raise ValueError("metadynamics grid widths must be finite")
        for j, period in enumerate(cv.periods):
            if period is not None and not np.isclose(widths[j], period, rtol=1e-10, atol=1e-12):
                raise ValueError("a periodic metadynamics grid must span exactly one period")
        if grid_bins is None:
            proposed = np.maximum(2.0, np.ceil(5.0 * widths / self.sigma))
            if not np.isfinite(proposed).all() or np.prod(proposed) > 1_000_000:
                raise ValueError("metadynamics grid exceeds one million nodes")
            grid_bins = proposed.astype(np.int64)
        if np.isscalar(grid_bins):
            grid_bins = [grid_bins] * cv.dimension
        raw = np.asarray(grid_bins)
        if raw.shape != (cv.dimension,) or raw.dtype.kind not in "iu" or np.any(raw < 2):
            raise ValueError("metadynamics grid bins must contain integer interval counts >=2")
        nodes = [int(n) + (period is None) for n, period in zip(raw, cv.periods)]
        if np.prod(nodes, dtype=float) > 1_000_000:
            raise ValueError("metadynamics grid exceeds one million nodes")
        self.grid_ranges = bounds.copy()
        self.grid_bins = raw.astype(np.int64)
        self.grid_spacing = widths / self.grid_bins
        self.grid_axes = tuple(np.linspace(lo, hi, count, endpoint=period is None)
                               for (lo, hi), count, period in zip(bounds, nodes, cv.periods))
        for array in (self.grid_ranges, self.grid_bins, self.grid_spacing, *self.grid_axes):
            array.flags.writeable = False
        self._grid = np.zeros((1 << cv.dimension, *nodes))
        self._grid.flags.writeable = False
        if grid_state is None:
            for hill in [] if hills is None else hills:
                self._append(hill)
        else:
            # Validate every history record without summing it again. A cache
            # is accepted only for this exact algorithm, geometry and history.
            for hill in [] if hills is None else hills:
                self._coordinates(np.asarray(hill["center"], dtype=float))
                super()._append(hill)
            values = np.asarray(grid_state.get("values"))
            metadata = grid_state.get("metadata")
            if (values.dtype.kind not in "fiu" or values.shape != self._grid.shape
                    or not np.isfinite(values).all()):
                raise ValueError("metadynamics grid cache has invalid coefficient values")
            if metadata != self._grid_metadata(values):
                raise ValueError("metadynamics grid cache disagrees with its settings or hill history")
            self._grid = np.array(values, dtype=float, copy=True)
            self._grid.flags.writeable = False

    def _grid_metadata(self, values=None):
        history = json.dumps(self.hills, sort_keys=True, separators=(",", ":"), allow_nan=False)
        coefficients = self._grid if values is None else values
        # Fixed little-endian float64/C ordering makes direct-API cache checks
        # independent of the array layout and the machine's native byte order.
        encoded = np.asarray(coefficients, dtype="<f8").tobytes(order="C")
        return {"algorithm": self.grid_algorithm, "ranges": self.grid_ranges.tolist(),
                "bins": self.grid_bins.tolist(), "sigma": self.sigma.tolist(),
                "periods": list(self.cv.periods), "hills": len(self.hills),
                "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
                "values_sha256": hashlib.sha256(encoded).hexdigest()}

    def grid_state(self):
        """Read-only values and history-bound metadata for an atomic checkpoint."""
        return {"values": self._grid, "metadata": self._grid_metadata()}

    def _coordinates(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim < 1 or values.shape[-1] != self.cv.dimension or not np.isfinite(values).all():
            raise ValueError("metadynamics values must be finite [...,D]")
        indices, fractions = [], []
        for j, period in enumerate(self.cv.periods):
            lo, hi = self.grid_ranges[j]
            x = values[..., j]
            if period is None:
                if np.any(x < lo) or np.any(x > hi):
                    raise ValueError(f"metadynamics CV {j} is outside its grid range [{lo}, {hi}]")
                x = x - lo
            else:
                x = (x - lo) % period
            scaled = x / self.grid_spacing[j]
            left = np.minimum(np.floor(scaled).astype(np.int64), self.grid_bins[j] - 1)
            indices.append(left)
            fractions.append(scaled - left)
        return values, indices, fractions

    @staticmethod
    def _basis(t, spacing):
        t2, t3 = t * t, t * t * t
        # Indices 0/1 multiply the left/right node value; 2/3 multiply
        # the corresponding physical derivative, hence the spacing factors.
        basis = (2*t3 - 3*t2 + 1, -2*t3 + 3*t2,
                 (t3 - 2*t2 + t)*spacing, (t3 - t2)*spacing)
        derivative = ((6*t2 - 6*t)/spacing, (-6*t2 + 6*t)/spacing,
                      3*t2 - 4*t + 1, 3*t2 - 2*t)
        return basis, derivative

    def physical_energy(self, values):
        """Energy of the interpolated grid used by the force calculation."""
        return self.physical_energy_gradient(values)[0]

    def physical_energy_gradient(self, values):
        values, left, fractions = self._coordinates(values)
        bases = [self._basis(t, dx) for t, dx in zip(fractions, self.grid_spacing)]
        energy = np.zeros(values.shape[:-1], dtype=float)
        gradient = np.zeros_like(values)
        for choices in itertools.product(range(4), repeat=self.cv.dimension):
            mask = sum((choice // 2) << j for j, choice in enumerate(choices))
            indices = tuple((left[j] + choice % 2) % len(self.grid_axes[j])
                            if self.cv.periods[j] is not None else left[j] + choice % 2
                            for j, choice in enumerate(choices))
            coefficient = self._grid[(mask, *indices)]
            term = coefficient
            for j, choice in enumerate(choices):
                term = term * bases[j][0][choice]
            energy += term
            for axis in range(self.cv.dimension):
                term = coefficient
                for j, choice in enumerate(choices):
                    term = term * bases[j][int(j == axis)][choice]
                gradient[..., axis] += term
        if not np.isfinite(energy).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite interpolated metadynamics bias")
        return energy, gradient

    def _hill_values(self, hill):
        factors = [self._kernel(axis - center, sigma, period)
                   for axis, center, sigma, period in
                   zip(self.grid_axes, hill["center"], self.sigma, self.cv.periods)]
        result = np.empty_like(self._grid)
        for mask in range(1 << self.cv.dimension):
            term = float(hill["height"])
            for j in range(self.cv.dimension):
                shape = [1] * self.cv.dimension
                shape[j] = len(self.grid_axes[j])
                term = term * factors[j][(mask >> j) & 1].reshape(shape)
            result[mask] = term
        return result

    def _append(self, hill):
        self._coordinates(np.asarray(hill["center"], dtype=float))
        previous_count = self.hill_count
        super()._append(hill)
        try:
            updated = self._grid + self._hill_values(self._hills[-1])
            if not np.isfinite(updated).all():
                raise FloatingPointError("nonfinite metadynamics grid deposition")
            updated.flags.writeable = False
        except BaseException:
            super().truncate_history(previous_count)
            raise
        self._grid = updated

    def truncate_history(self, count):
        previous_count = self.hill_count
        super().truncate_history(count)
        if count == previous_count:
            return
        # Only an exceptional post-deposition failure reaches this replay.
        # Subtraction would introduce roundoff and spoil exact resume.
        restored = np.zeros_like(self._grid)
        for hill in self.hills:
            restored += self._hill_values(hill)
        restored.flags.writeable = False
        self._grid = restored
