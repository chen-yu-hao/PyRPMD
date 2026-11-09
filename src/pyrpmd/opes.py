"""Uncompressed on-the-fly probability enhanced sampling.

This implements the OPES_METAD probability/bias equations, not a sum of
metadynamics hills. See Invernizzi & Parrinello, J. Phys. Chem. Lett. 11,
2731 (2020), doi:10.1021/acs.jpclett.0c00497. The kernel-center estimate of
Z follows the algorithm described by PLUMED's OPES implementation. Kernels
are normalized, smooth wrapped Gaussians. The optional explore mode estimates
the sampled target density with unweighted kernels. Optional adaptive widths
use online observed fluctuations and a Scott/ESS factor, storing each kernel's
own width. This explicit rule is not PLUMED's diffusion-window adaptation.
Shared histories retain every kernel in deterministic event order.
Kernel compression is not implemented.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import numpy as np
from scipy.special import logsumexp

from .path_integral import _integer
from .walker_exchange import allgather_centers, make_record, validate_record


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

    With no kernels, the density is zero and V=prefactor*log(epsilon)
    is ``-barrier``. This constant determines the first METAD kernel's
    weight and therefore cannot be discarded during initialization.
    Schema-1 checkpoints retain their historical zero initial bias when
    resumed; new runs and schema-2 checkpoints use the equation above.

    The method retains every kernel: evaluating a point and adding a kernel
    cost O(n), and memory costs O(n). ``Z`` is the arithmetic mean of the
    normalized KDE at its kernel centers, evaluated incrementally exactly
    for this uncompressed, fixed-width representation.
    """

    history_dependent = True
    method = "opes_metad_fixed_bandwidth"

    def __init__(self, cv, sigma, beta, *, barrier, bias_factor=10.0, state=None,
                 mode="metad", adaptive_sigma=False, sigma_min=None):
        self.cv = cv
        if mode not in ("metad", "explore") or not isinstance(adaptive_sigma, (bool, np.bool_)):
            raise ValueError("OPES mode must be metad/explore and adaptive_sigma must be boolean")
        self.mode, self.adaptive_sigma = mode, bool(adaptive_sigma)
        if mode != "metad" or adaptive_sigma:
            self.method = "opes_" + mode + ("_adaptive_bandwidth" if adaptive_sigma else "_fixed_bandwidth")
        self.sigma = np.atleast_1d(np.asarray(sigma, float)).copy()
        with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
            representable = np.isfinite(self.sigma**2).all() and np.isfinite(1.0 / self.sigma**2).all()
        if (cv.dimension not in (1, 2) or self.sigma.shape != (cv.dimension,)
                or not np.isfinite(self.sigma).all() or np.any(self.sigma <= 0)
                or not representable):
            raise ValueError("OPES sigma needs one positive representable width per CV (one or two dimensions)")
        self.sigma.flags.writeable = False
        if sigma_min is not None and not adaptive_sigma:
            raise ValueError("sigma_min requires adaptive_sigma=True")
        self.sigma_min = (None if not adaptive_sigma else
                          np.atleast_1d(np.asarray(self.sigma * 0.01 if sigma_min is None else sigma_min, float)).copy())
        if self.sigma_min is not None:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                valid = np.isfinite(self.sigma_min**2).all() and np.isfinite(1 / self.sigma_min**2).all()
            if (self.sigma_min.shape != self.sigma.shape or not valid or np.any(self.sigma_min <= 0)
                    or np.any(self.sigma_min > self.sigma)):
                raise ValueError("sigma_min needs positive representable widths no greater than initial sigma")
            self.sigma_min.flags.writeable = False
        self.beta, self.barrier, self.bias_factor = map(float, (beta, barrier, bias_factor))
        if (not np.isfinite([self.beta, self.barrier, self.bias_factor]).all()
                or self.beta <= 0 or self.barrier <= 0 or self.bias_factor <= 1):
            raise ValueError("OPES beta/barrier must be positive and finite; bias_factor must be finite and > 1")
        self.prefactor = ((self.bias_factor - 1.0) if mode == "explore"
                          else (1.0 - 1.0 / self.bias_factor)) / self.beta
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
        self._sigmas = np.empty((0, cv.dimension))
        self._mean = np.zeros(cv.dimension)
        self._m2 = np.zeros(cv.dimension)
        self._last_unwrapped = np.zeros(cv.dimension)
        self.log_z = 0.0
        self._state_schema = 2
        self._initial_bias = -self.barrier
        if state is not None:
            if (not isinstance(state, Mapping)
                    or isinstance(state.get("schema_version"), (bool, np.bool_))
                    or not isinstance(state.get("schema_version"), (int, np.integer))
                    or state["schema_version"] not in (1, 2)):
                raise ValueError("invalid OPES state schema_version")
            if state["schema_version"] == 1:
                # A legacy restart must continue the trajectory represented
                # by its saved weights, including an empty pre-update state.
                self._state_schema = 1
                self._initial_bias = 0.0
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

    def _fixed_kernel(self, values, centers, *, gradient=False, sigma=None):
        """Log normalized product kernel and its logarithmic derivative."""
        with np.errstate(over="ignore"):
            delta = values[..., None, :] - centers
        logs = np.zeros(delta.shape[:-1])
        scores = np.empty_like(delta) if gradient else None
        for j, (sigma, period) in enumerate(zip(self.sigma if sigma is None else sigma, self.periods)):
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

    def _kernel(self, values, centers, *, gradient=False, sigmas=None):
        if not self.adaptive_sigma:
            return self._fixed_kernel(values, centers, gradient=gradient)
        sigmas = self._sigmas if sigmas is None else sigmas
        if np.shape(sigmas) != np.shape(centers):
            raise ValueError("adaptive OPES kernel widths must match centers")
        logs = np.empty(values.shape[:-1] + (len(centers),))
        scores = np.empty(logs.shape + (self.cv.dimension,)) if gradient else None
        for index, (center, width) in enumerate(zip(centers, sigmas)):
            local, derivative = self._fixed_kernel(values, center[None], gradient=gradient, sigma=width)
            logs[..., index] = local[..., 0]
            if gradient:
                scores[..., index, :] = derivative[..., 0, :]
        return logs, scores

    def _adapt(self, center, weights):
        """Welford observed fluctuation, then d-dimensional Scott/ESS scaling."""
        n = len(weights)
        if n == 1:
            return self.sigma.copy(), center.copy(), np.zeros_like(center), center.copy()
        delta = center - self._centers[-1]
        for j, period in enumerate(self.periods):
            if period is not None:
                delta[j] = (delta[j] + period / 2) % period - period / 2
        unwrapped = self._last_unwrapped + delta
        increment = unwrapped - self._mean
        mean = self._mean + increment / n
        m2 = self._m2 + increment * (unwrapped - mean)
        width = self.sigma.copy()
        if n >= 3:
            neff = 1.0 / np.sum(normalized_weights(weights)**2)
            factor = (4.0 / (self.cv.dimension + 2) / max(neff, 1.0)) ** (1.0 / (self.cv.dimension + 4))
            width = np.maximum(np.sqrt(np.maximum(m2, 0.0) / (n - 1)) * factor, self.sigma_min)
        if not all(np.isfinite(x).all() for x in (width, mean, m2, unwrapped)):
            raise ValueError("adaptive OPES fluctuation is not representable")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if not np.isfinite(width**2).all() or not np.isfinite(1 / width**2).all():
                raise ValueError("adaptive OPES width is not representable")
        return width, mean, m2, unwrapped

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
            return np.full(values.shape[:-1], self._initial_bias), np.zeros_like(values)
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
            return np.full(values.shape[:-1], self._initial_bias)
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
        log_weight = 0.0 if self.mode == "explore" else self.beta * before
        centers = np.concatenate((self._centers, center[None]))
        weights = np.append(self._log_weights, log_weight)
        if self.adaptive_sigma:
            width, mean, m2, last = self._adapt(center, weights)
            sigmas = np.concatenate((self._sigmas, width[None]))
            old_at_new, _ = self._kernel(center, self._centers)
            new_at_all, _ = self._fixed_kernel(centers, center[None], sigma=width)
            center_sums = np.append(np.logaddexp(self._log_center_sums, old_at_new),
                                    logsumexp(new_at_all[..., 0]))
        else:
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
        if self.adaptive_sigma:
            self._sigmas, self._mean, self._m2, self._last_unwrapped = sigmas, mean, m2, last
        return {"step": int(step), "time": float(time), "center": center.tolist(),
                "log_weight": log_weight, "bias_before": before, "log_z": log_z,
                "weight_effective_sample_size": self.effective_sample_size}

    def state_dict(self):
        """Detached, JSON-compatible restart state, including exact accumulators."""
        state = {"schema_version": self._state_schema, "method": self.method, "sigma": self.sigma.tolist(),
                "beta": self.beta, "barrier": self.barrier, "bias_factor": self.bias_factor,
                "periods": list(self.periods), "centers": self._centers.tolist(),
                "log_weights": self._log_weights.tolist(), "steps": self._steps.tolist(),
                "times": self._times.tolist(), "log_center_sums": self._log_center_sums.tolist(),
                "log_z": self.log_z}
        if self._state_schema == 2:
            state["initial_bias"] = self._initial_bias
        if self.mode != "metad" or self.adaptive_sigma:
            state.update(mode=self.mode, adaptive_sigma=self.adaptive_sigma)
        if self.adaptive_sigma:
            state.update(sigma_min=self.sigma_min.tolist(), kernel_sigmas=self._sigmas.tolist(),
                         adaptive_mean=self._mean.tolist(), adaptive_m2=self._m2.tolist(),
                         adaptive_last_unwrapped=self._last_unwrapped.tolist())
        return state

    def _load_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("invalid OPES state fields")
        current = self.state_dict()
        identity = ["schema_version", "method", "sigma", "beta", "barrier", "bias_factor", "periods"]
        if self._state_schema == 2:
            identity += ["initial_bias"]
        if self.mode != "metad" or self.adaptive_sigma:
            identity += ["mode", "adaptive_sigma"]
        if self.adaptive_sigma:
            identity += ["sigma_min"]
        for key in identity:
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
        sigmas = None
        if self.mode == "explore" and np.any(log_weights != 0):
            raise ValueError("OPES explore requires unweighted kernels")
        if (n and self.mode == "metad" and self._state_schema == 2
                and log_weights[0] != self.beta * self._initial_bias):
            # The normalized KDE is invariant to a common weight offset,
            # so its caches alone cannot detect a legacy first weight being
            # relabeled as the new initialization convention.
            raise ValueError("OPES first kernel weight disagrees with initial_bias")
        if self.adaptive_sigma:
            sigmas = np.asarray(state["kernel_sigmas"], float).reshape(-1, self.cv.dimension)
            if sigmas.shape != centers.shape or not np.isfinite(sigmas).all() or np.any(sigmas < self.sigma_min):
                raise ValueError("invalid adaptive OPES kernel widths")
            # Recompute the recurrence rather than trusting its restart caches.
            replay = OPESBias(self.cv, self.sigma, self.beta, barrier=self.barrier, bias_factor=self.bias_factor,
                              mode=self.mode, adaptive_sigma=True, sigma_min=self.sigma_min)
            for index, center in enumerate(centers):
                width, mean, m2, last = replay._adapt(center, log_weights[:index + 1])
                if not np.allclose(width, sigmas[index], rtol=1e-12, atol=1e-12):
                    raise ValueError("adaptive OPES kernel width disagrees with history")
                replay._centers = centers[:index + 1]
                replay._mean, replay._m2, replay._last_unwrapped = mean, m2, last
            for key, expected in (("adaptive_mean", replay._mean), ("adaptive_m2", replay._m2),
                                  ("adaptive_last_unwrapped", replay._last_unwrapped)):
                value = np.asarray(state[key], float)
                if value.shape != (self.cv.dimension,) or not np.isfinite(value).all() or not np.allclose(value, expected, rtol=1e-12, atol=1e-12):
                    raise ValueError("inconsistent adaptive OPES recurrence state")
        # Validate cached sums against the kernel representation in bounded
        # chunks, then retain their saved bits for deterministic continuation.
        chunk = max(1, 65536 // max(n, 1))
        for start in range(0, n, chunk):
            logs, _ = self._kernel(centers, centers[start:start + chunk],
                                   sigmas=None if sigmas is None else sigmas[start:start + chunk])
            if not np.allclose(logsumexp(logs, axis=0), sums[start:start + chunk], rtol=0, atol=1e-10):
                raise ValueError("OPES state kernel normalization is inconsistent")
        expected_z = float(logsumexp(log_weights + sums) - logsumexp(log_weights) - np.log(n)) if n else 0.0
        if not np.isclose(log_z, expected_z, rtol=0, atol=1e-10):
            raise ValueError("OPES state Z normalization is inconsistent")
        self._centers, self._log_weights = centers.copy(), log_weights.copy()
        self._steps, self._times = raw_steps.astype(np.int64), times.copy()
        self._log_center_sums, self.log_z = sums.copy(), log_z
        if self.adaptive_sigma:
            self._sigmas = sigmas.copy()
            self._mean = np.asarray(state["adaptive_mean"], float).copy()
            self._m2 = np.asarray(state["adaptive_m2"], float).copy()
            self._last_unwrapped = np.asarray(state["adaptive_last_unwrapped"], float).copy()


class OPESExploreBias(OPESBias):
    """OPES-explore with literal kernel widths in the declared CV units.

    The kernel width is exactly the supplied width before adaptation. PLUMED
    OPES_METAD_EXPLORE instead multiplies its input SIGMA by sqrt(gamma);
    equivalent fixed physical widths therefore require SIGMA/sqrt(gamma)
    as the PLUMED input.
    """

    def __init__(self, cv, sigma, beta, **kwargs):
        super().__init__(cv, sigma, beta, mode="explore", **kwargs)


class SharedOPESBias(OPESBias):
    """Fixed-bandwidth OPES with one shared, uncompressed kernel history.

    A synchronous batch computes every kernel weight from the same pre-commit
    bias and then commits kernels in walker order.  Asynchronous callers may
    instead publish ``prepare_local_payload`` through ``AsyncWalkerStore`` and
    feed the resulting immutable records to ``merge_records``.
    """

    def __init__(self, cv, sigma, beta, *, walkers, barrier, bias_factor=10.0,
                 mode="metad", state=None):
        self.walkers = _integer(walkers, "walkers", 2)
        super().__init__(cv, sigma, beta, barrier=barrier,
                         bias_factor=bias_factor, mode=mode,
                         adaptive_sigma=False, state=None)
        self.method = "opes_" + mode + "_shared_fixed_bandwidth"
        self._records: dict[tuple[int, int], dict] = {}
        self._next_sequences = np.zeros(self.walkers, dtype=np.int64)
        if state is not None:
            self._load_shared_state(state)

    @property
    def records(self):
        return json.loads(json.dumps(self._ordered_records(), allow_nan=False))

    def prepare_local_payload(self, values) -> dict:
        center = self._wrapped_center(values)
        before = float(self.physical_energy(center))
        log_weight = 0.0 if self.mode == "explore" else self.beta * before
        if not np.isfinite(log_weight):
            raise ValueError("shared OPES kernel weight is not representable")
        return {"center": center.tolist(), "log_weight": float(log_weight),
                "bias_before": before}

    def update_walkers(self, values, *, step, time):
        centers = np.asarray(values, dtype=np.float64)
        if (centers.shape != (self.walkers, self.cv.dimension)
                or not np.isfinite(centers).all()):
            raise ValueError("shared OPES update needs finite [walkers,dimension] centers")
        wrapped = np.asarray([self._wrapped_center(center) for center in centers])
        before = np.asarray(self.physical_energy(wrapped), dtype=np.float64)
        log_weights = (np.zeros(self.walkers) if self.mode == "explore"
                       else self.beta * before)
        if not np.isfinite(log_weights).all():
            raise ValueError("shared OPES kernel weights are not representable")
        records = []
        for walker in range(self.walkers):
            records.append(make_record(
                "opes", walker, int(self._next_sequences[walker]),
                step=step, time=time,
                payload={"center": wrapped[walker].tolist(),
                         "log_weight": float(log_weights[walker]),
                         "bias_before": float(before[walker])}))
        self.merge_records(records)
        return [{**record["event"], "log_z": self.log_z,
                 "weight_effective_sample_size": self.effective_sample_size}
                for record in records]

    def update_mpi(self, values, *, step, time, comm=None):
        centers = allgather_centers(values, walkers=self.walkers, comm=comm)
        return self.update_walkers(centers, step=step, time=time)

    def merge_records(self, records: Iterable[Mapping]) -> int:
        candidate = dict(self._records)
        changed = 0
        for raw in records:
            record = validate_record(raw, method="opes")
            event = record["event"]
            walker = event["walker"]
            if walker >= self.walkers:
                raise ValueError("OPES record walker is outside this ensemble")
            payload = event["payload"]
            if not isinstance(payload, Mapping) or set(payload) != {
                    "center", "log_weight", "bias_before"}:
                raise ValueError("invalid shared OPES payload fields")
            center = np.asarray(payload["center"])
            if (center.dtype.kind not in "iuf" or center.shape != (self.cv.dimension,)
                    or not np.isfinite(center).all()):
                raise ValueError("invalid shared OPES center")
            for name in ("log_weight", "bias_before"):
                if not np.isfinite(float(payload[name])):
                    raise ValueError("invalid shared OPES payload")
            if self.mode == "explore" and float(payload["log_weight"]) != 0.0:
                raise ValueError("shared OPES explore kernels must be unweighted")
            identity = (walker, event["sequence"])
            previous = candidate.get(identity)
            if previous is not None:
                if previous != record:
                    raise ValueError("conflicting duplicate OPES event")
                continue
            candidate[identity] = record
            changed += 1
        if not changed:
            return 0
        self._validate_candidate(candidate)
        previous = self._records
        previous_sequences = self._next_sequences.copy()
        self._records = candidate
        for walker in range(self.walkers):
            self._next_sequences[walker] = sum(
                owner == walker for owner, _ in candidate)
        try:
            self._rebuild()
        except BaseException:
            self._records = previous
            self._next_sequences = previous_sequences
            self._rebuild()
            raise
        return changed

    def _validate_candidate(self, candidate):
        for walker in range(self.walkers):
            events = sorted(
                (record["event"] for (owner, _), record in candidate.items()
                 if owner == walker), key=lambda event: event["sequence"])
            if [event["sequence"] for event in events] != list(range(len(events))):
                raise ValueError("OPES walker sequence has a gap")
            for previous, current in zip(events, events[1:]):
                if (current["step"] <= previous["step"]
                        or current["time"] <= previous["time"]):
                    raise ValueError("OPES walker clocks must strictly increase")

    def state_dict(self):
        return {"schema_version": 1, "method": self.method,
                "sigma": self.sigma.tolist(), "beta": self.beta,
                "barrier": self.barrier, "bias_factor": self.bias_factor,
                "mode": self.mode, "periods": list(self.periods),
                "walkers": self.walkers, "initial_bias": self._initial_bias,
                "records": self.records, "log_z": self.log_z,
                "compression": "none"}

    def _load_shared_state(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("invalid shared OPES state fields")
        current = self.state_dict()
        for key in ("schema_version", "method", "sigma", "beta", "barrier",
                    "bias_factor", "mode", "periods", "walkers",
                    "initial_bias", "compression"):
            if state[key] != current[key]:
                raise ValueError("shared OPES state parameters differ: " + key)
        records = state["records"]
        if not isinstance(records, list):
            raise ValueError("shared OPES records must be a list")
        self.merge_records(records)
        expected = float(state["log_z"])
        if not np.isfinite(expected) or expected != self.log_z:
            raise ValueError("shared OPES state normalization is inconsistent")

    def _wrapped_center(self, values):
        center = self._values(values).copy()
        if center.shape != (self.cv.dimension,):
            raise ValueError("one shared OPES update needs one [D] center")
        for index, period in enumerate(self.periods):
            if period is not None:
                center[index] = (center[index] + period / 2) % period - period / 2
        return center

    def _ordered_records(self):
        return sorted(self._records.values(), key=lambda record: (
            record["event"]["step"], record["event"]["time"],
            record["event"]["walker"], record["event"]["sequence"]))

    def _rebuild(self):
        ordered = self._ordered_records()
        if not ordered:
            self._centers = np.empty((0, self.cv.dimension))
            self._log_weights = np.empty(0)
            self._log_center_sums = np.empty(0)
            self._steps = np.empty(0, dtype=np.int64)
            self._times = np.empty(0)
            self.log_z = 0.0
            return
        self._centers = np.asarray(
            [record["event"]["payload"]["center"] for record in ordered],
            dtype=np.float64)
        self._log_weights = np.asarray(
            [record["event"]["payload"]["log_weight"] for record in ordered],
            dtype=np.float64)
        self._steps = np.asarray(
            [record["event"]["step"] for record in ordered], dtype=np.int64)
        self._times = np.asarray(
            [record["event"]["time"] for record in ordered], dtype=np.float64)
        pair_logs, _ = self._kernel(self._centers, self._centers)
        self._log_center_sums = logsumexp(pair_logs, axis=0)
        self.log_z = float(
            logsumexp(self._log_weights + self._log_center_sums)
            - logsumexp(self._log_weights) - np.log(len(self._log_weights)))
        if not np.isfinite(self.log_z):
            raise ValueError("shared OPES normalization is not representable")


__all__ = [
    "OPESBias", "OPESExploreBias", "SharedOPESBias", "normalized_weights",
]
