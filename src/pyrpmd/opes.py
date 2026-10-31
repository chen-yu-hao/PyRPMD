"""Fixed-bandwidth, uncompressed on-the-fly probability enhanced sampling.

This implements the OPES_METAD probability/bias equations, not a sum of
metadynamics hills. See Invernizzi & Parrinello, J. Phys. Chem. Lett. 11,
2731 (2020), doi:10.1021/acs.jpclett.0c00497. The kernel-center estimate of
Z follows the algorithm described by PLUMED's OPES implementation. Kernels
are normalized, smooth wrapped Gaussians; bandwidth adaptation, kernel
compression and shared walkers are deliberately absent.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy.special import logsumexp


def normalized_weights(log_weights):
    """Normalize finite log weights without exponentiating their offset."""
    values = np.asarray(log_weights, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("log weights must be a nonempty finite vector")
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


class OPESBias:
    """OPES_METAD with explicit, transactional probability-update events.

    ``sigma`` is fixed in canonical CV units, ``barrier`` is a physical
    energy, and ``beta`` is the physical inverse temperature. The bias
    lower bound is ``-barrier``. The bias contribution to H_P is P*V and
    the KDE weight is exp(beta*V), without an extra bead factor.

    The method retains every kernel: evaluating a point and adding a kernel
    cost O(n), and memory costs O(n). ``Z`` is the arithmetic mean of the
    normalized KDE at its kernel centers, evaluated incrementally exactly
    for this uncompressed, fixed-width representation.
    """

    history_dependent = True
    method = "opes_metad_fixed_bandwidth"

    def __init__(self, cv, sigma, beta, *, barrier, bias_factor=10.0, state=None):
        self.cv = cv
        self.sigma = np.atleast_1d(np.asarray(sigma, float)).copy()
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            representable = np.isfinite(self.sigma**2).all() and np.isfinite(1.0 / self.sigma**2).all()
        if (cv.dimension not in (1, 2) or self.sigma.shape != (cv.dimension,)
                or not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0)
                or not representable):
            raise ValueError("OPES sigma needs one positive representable width per CV (one or two dimensions)")
        self.sigma.flags.writeable = False
        self.beta, self.barrier, self.bias_factor = map(float, (beta, barrier, bias_factor))
        if (not np.isfinite([self.beta, self.barrier, self.bias_factor]).all()
                or self.beta <= 0 or self.barrier <= 0 or self.bias_factor <= 1):
            raise ValueError("OPES beta/barrier must be positive and finite; bias_factor must be finite and > 1")
        self.prefactor = (1.0 - 1.0 / self.bias_factor) / self.beta
        self.log_epsilon = -self.barrier / self.prefactor
        if not np.isfinite(self.log_epsilon):
            raise ValueError("OPES barrier/beta combination is not representable")
        self.periods = tuple(cv.periods)
        if len(self.periods) != cv.dimension or any(
                p is not None and (not np.isfinite(p) or p <= 0) for p in self.periods):
            raise ValueError("CV periods must be positive finite numbers or None")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if any(period is not None and not np.isfinite(
                    [4.0 * period, 2 * np.pi / period, sigma / period]).all()
                   for sigma, period in zip(self.sigma, self.periods)):
                raise ValueError("OPES periodic width/period combination is not representable")
        self._centers = np.empty((0, cv.dimension))
        self._log_weights = np.empty(0)
        self._log_center_sums = np.empty(0)
        self._steps = np.empty(0, dtype=np.int64)
        self._times = np.empty(0)
        self.log_z = 0.0
        if state is not None:
            self._load_state(state)

    def __len__(self):
        return len(self._log_weights)

    @property
    def effective_sample_size(self):
        """Weight-only ESS; does not account for trajectory autocorrelation."""
        if not len(self):
            return 0.0
        weights = normalized_weights(self._log_weights)
        return float(1.0 / (weights @ weights))

    def _values(self, values):
        values = np.asarray(values, float)
        if values.ndim < 1 or values.shape[-1] != self.cv.dimension or not np.isfinite(values).all():
            raise ValueError("OPES values must be finite [...,D]")
        return values

    def _kernel(self, values, centers, *, gradient=False):
        """Log normalized product kernel and its logarithmic derivative."""
        with np.errstate(over="ignore"):
            delta = values[..., None, :] - centers
        logs = np.zeros(delta.shape[:-1])
        scores = np.empty_like(delta) if gradient else None
        for j, (sigma, period) in enumerate(zip(self.sigma, self.periods)):
            d = delta[..., j]
            if period is None:
                with np.errstate(over="ignore"):
                    logs += -0.5 * (d / sigma)**2 - np.log(np.sqrt(2 * np.pi) * sigma)
                if gradient:
                    scores[..., j] = -d / sigma**2
            elif sigma / period >= 2:
                # Nonzero Fourier modes are below floating-point precision.
                logs -= np.log(period)
                if gradient:
                    scores[..., j] = 0.0
            elif sigma / period >= 0.2:
                # Poisson-summed wrapped-normal density. Unlike a minimum-
                # image Gaussian, its derivative is continuous at the seam.
                d = np.remainder(values[..., None, j], period) - np.remainder(centers[:, j], period)
                d = np.where(d > period / 2, d - period, np.where(d < -period / 2, d + period, d))
                ratio = sigma / period
                count = max(1, int(np.ceil(np.sqrt(72.0) / (2 * np.pi * ratio))))
                modes = np.arange(1, count + 1, dtype=float)
                coeff = np.exp(-0.5 * (2 * np.pi * ratio * modes)**2)
                phase = d[..., None] * (2 * np.pi / period) * modes
                numerator = 1 + 2 * np.sum(coeff * np.cos(phase), axis=-1)
                logs += np.log(numerator) - np.log(period)
                if gradient:
                    scores[..., j] = (-2 * (2 * np.pi / period)
                        * np.sum(coeff * modes * np.sin(phase), axis=-1) / numerator)
            else:
                d = np.remainder(values[..., None, j], period) - np.remainder(centers[:, j], period)
                d = np.where(d > period / 2, d - period, np.where(d < -period / 2, d + period, d))
                count = int(np.ceil(8.5 * sigma / period)) + 1
                shifts = d[..., None] + np.arange(-count, count + 1) * period
                with np.errstate(over="ignore"):
                    image_logs = -0.5 * (shifts / sigma)**2
                combined = logsumexp(image_logs, axis=-1)
                logs += combined - np.log(np.sqrt(2 * np.pi) * sigma)
                if gradient:
                    with np.errstate(invalid="ignore", over="ignore"):
                        weights = np.exp(image_logs - combined[..., None])
                        products = np.zeros_like(shifts)
                        np.multiply(weights, -shifts / sigma**2, out=products, where=weights > 0)
                    scores[..., j] = np.sum(products, axis=-1)
        return logs, scores

    def _probability(self, values, *, gradient=False):
        values = self._values(values)
        if not len(self):
            raise ValueError("OPES probability requires at least one update")
        flat = values.reshape(-1, self.cv.dimension)
        out = np.empty(len(flat))
        score = np.empty_like(flat) if gradient else None
        # Bound the temporary point x history arrays during grid evaluation.
        chunk = max(1, 65536 // len(self))
        log_weights = self._log_weights - logsumexp(self._log_weights)
        for start in range(0, len(flat), chunk):
            logs, derivatives = self._kernel(flat[start:start + chunk], self._centers, gradient=gradient)
            terms = logs + log_weights
            norm = logsumexp(terms, axis=-1)
            out[start:start + chunk] = norm
            if gradient:
                finite = np.isfinite(norm)
                local = np.zeros((len(norm), self.cv.dimension))
                weights = np.exp(terms[finite] - norm[finite, None])[..., None]
                products = np.zeros_like(derivatives[finite])
                # Far-away kernels can have -inf log density and infinite
                # score. Their exactly zero responsibility must remain zero.
                np.multiply(weights, derivatives[finite], out=products, where=weights > 0)
                local[finite] = np.sum(products, axis=-2)
                score[start:start + chunk] = local
        return out.reshape(values.shape[:-1]), None if score is None else score.reshape(values.shape)

    def log_probability(self, values):
        """Log of the normalized, finite-bandwidth unbiased density estimate."""
        return self._probability(values)[0]

    def physical_energy_gradient(self, values):
        values = self._values(values)
        if not len(self):
            return np.zeros(values.shape[:-1]), np.zeros_like(values)
        log_probability, score = self._probability(values, gradient=True)
        ratio = log_probability - self.log_z
        log_argument = np.logaddexp(ratio, self.log_epsilon)
        energy = self.prefactor * log_argument
        coefficient = np.exp(ratio - log_argument)
        gradient = self.prefactor * coefficient[..., None] * score
        if not np.isfinite(energy).all() or not np.isfinite(gradient).all():
            raise ValueError("OPES energy/gradient is not representable")
        return energy, gradient

    def physical_energy(self, values):
        values = self._values(values)
        if not len(self):
            return np.zeros(values.shape[:-1])
        ratio = self.log_probability(values) - self.log_z
        return self.prefactor * np.logaddexp(ratio, self.log_epsilon)

    def energy_force(self, q):
        result = self.cv.value_and_jacobian(q)
        energy, gradient = self.physical_energy_gradient(result.value)
        return (len(q) * float(energy),
                -len(q) * np.einsum("d,dpac->pac", gradient, result.jacobian))

    def update(self, values, *, step, time):
        """Add one weighted kernel after a completed dynamics step.

        The complete new state is prepared before committing, so validation
        or numerical errors do not leave a partially deposited kernel.
        """
        center = self._values(values).copy()
        if center.shape != (self.cv.dimension,):
            raise ValueError("one OPES update requires a single [D] center")
        if (isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer))
                or step < 1 or step > np.iinfo(np.int64).max or not np.isfinite(time) or time <= 0):
            raise ValueError("OPES update needs a positive integer step and positive finite time")
        if len(self) and (step <= self._steps[-1] or time <= self._times[-1]):
            raise ValueError("OPES update clocks must strictly increase")
        for j, period in enumerate(self.periods):
            if period is not None:
                center[j] = (center[j] + period / 2) % period - period / 2
        before = float(self.physical_energy(center))
        log_weight = self.beta * before
        centers = np.concatenate((self._centers, center[None]))
        weights = np.append(self._log_weights, log_weight)
        pair_logs, _ = self._kernel(center, centers)
        # K is symmetric at fixed bandwidth. Column sums update in O(n).
        center_sums = np.append(np.logaddexp(self._log_center_sums, pair_logs[:-1]),
                                logsumexp(pair_logs))
        log_z = float(logsumexp(weights + center_sums) - logsumexp(weights) - np.log(len(weights)))
        if not np.isfinite(log_weight) or not np.isfinite(log_z):
            raise ValueError("OPES update produced an invalid normalization")
        self._centers, self._log_weights = centers, weights
        self._log_center_sums, self.log_z = center_sums, log_z
        self._steps, self._times = np.append(self._steps, step), np.append(self._times, time)
        return {"step": int(step), "time": float(time), "center": center.tolist(),
                "log_weight": log_weight, "bias_before": before, "log_z": log_z,
                "weight_effective_sample_size": self.effective_sample_size}

    def state_dict(self):
        """Detached, JSON-compatible restart state, including exact accumulators."""
        return {"schema_version": 1, "method": self.method, "sigma": self.sigma.tolist(),
                "beta": self.beta, "barrier": self.barrier, "bias_factor": self.bias_factor,
                "periods": list(self.periods), "centers": self._centers.tolist(),
                "log_weights": self._log_weights.tolist(), "steps": self._steps.tolist(),
                "times": self._times.tolist(), "log_center_sums": self._log_center_sums.tolist(),
                "log_z": self.log_z}

    def _load_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("invalid OPES state fields")
        current = self.state_dict()
        for key in ("schema_version", "method", "sigma", "beta", "barrier", "bias_factor", "periods"):
            if state[key] != current[key]:
                raise ValueError("OPES state parameters differ: " + key)
        centers = np.asarray(state["centers"], float)
        log_weights = np.asarray(state["log_weights"], float)
        times = np.asarray(state["times"], float)
        sums = np.asarray(state["log_center_sums"], float)
        raw_steps = np.asarray(state["steps"])
        n = len(log_weights) if log_weights.ndim == 1 else -1
        if n == 0:
            centers = centers.reshape(0, self.cv.dimension)
        if (centers.shape != (n, self.cv.dimension) or times.shape != (n,)
                or sums.shape != (n,) or raw_steps.shape != (n,)
                or (n and raw_steps.dtype.kind not in "iu")
                or not all(np.isfinite(a).all() for a in (centers, log_weights, times, sums))
                or np.any(raw_steps < 1) or np.any(raw_steps > np.iinfo(np.int64).max) or np.any(times <= 0)
                or np.any(raw_steps[1:] <= raw_steps[:-1]) or np.any(times[1:] <= times[:-1])):
            raise ValueError("invalid OPES history arrays or clocks")
        log_z = float(state["log_z"])
        if not np.isfinite(log_z):
            raise ValueError("invalid OPES normalization")
        # Validate cached sums against the kernel representation in bounded
        # chunks, then retain their saved bits for deterministic continuation.
        chunk = max(1, 65536 // max(n, 1))
        for start in range(0, n, chunk):
            logs, _ = self._kernel(centers[start:start + chunk], centers)
            if not np.allclose(logsumexp(logs, axis=-1), sums[start:start + chunk], rtol=0, atol=1e-10):
                raise ValueError("OPES state kernel normalization is inconsistent")
        expected_z = float(logsumexp(log_weights + sums) - logsumexp(log_weights) - np.log(n)) if n else 0.0
        if not np.isclose(log_z, expected_z, rtol=0, atol=1e-10):
            raise ValueError("OPES state Z normalization is inconsistent")
        self._centers, self._log_weights = centers.copy(), log_weights.copy()
        self._steps, self._times = raw_steps.astype(np.int64), times.copy()
        self._log_center_sums, self.log_z = sums.copy(), log_z
