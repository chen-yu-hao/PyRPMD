"""Restrained path learning and signed umbrella reweighting for thermal NRPMD.

The path is a string in explicitly scaled, nonperiodic collective variables.
It follows the perpendicular restrained mean force, with fixed endpoints and
equal-arclength redistribution. This is finite-restraint path exploration, not
a committor, a metric-correct minimum-free-energy path, or transition-path MC.

Only fresh samples from the final frozen umbrellas enter MBAR. MBAR acts on
the nonnegative absolute SEO measure; the original SEO signs are retained in
every physical average. Neither MC sweeps nor adaptation rounds are time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy

import numpy as np

from .analysis.fes import (
    cv_statistical_inefficiency, solve_mbar, statistical_inefficiency,
)
from .analysis.mbar_problem import umbrella_problem
from .bias import UmbrellaBias
from .dynamics import CayleyIntegrator
from .nonadiabatic import _positive_integer
from .thermal_nrpmd import (
    ThermalEnsemble, ThermalNRPMDSampler, _finite_positive,
    _nonnegative_integer, _static_umbrella_identity,
)


def _readonly(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _finite_array(value, name, ndim=None):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()
            or (ndim is not None and raw.ndim != ndim)):
        raise ValueError(f"{name} must be a finite real array"
                         + (f" with {ndim} dimensions" if ndim is not None else ""))
    return np.array(raw, dtype=float, copy=True)


def _sign_threshold(value):
    value = _finite_positive(value, "min_average_sign")
    if value > 1:
        raise ValueError("min_average_sign must not exceed one")
    return value


def _cv_values(cv, states):
    values = _finite_array([cv.value_and_jacobian(state.q).value for state in states],
                           "CV values", 2)
    if values.shape != (len(states), cv.dimension):
        raise ValueError("CV values have an inconsistent dimension")
    return values


def _varying_inefficiency(values):
    # A constant sign/population adds no fluctuating observable to diagnose.
    # A constant *coordinate* is still treated as unresolved by the CV helper.
    return (statistical_inefficiency(values)
            if len(values) >= 4 and np.ptp(values) > 1e-14 else 1.0)


@dataclass(frozen=True)
class AdaptivePathOptions:
    """Budgets and numerical guards, in the existing atomic-unit convention.

    ``max_step`` and ``path_tolerance`` use coordinates divided by cv_scales.
    ``max_step`` bounds normal force motion before arclength redistribution;
    redistribution can move an image farther. The effective-sample guard is
    for the positive absolute measure, with heuristic autocorrelation inflation.
    Signed ESS is reported separately. Neither certifies independent chains or
    physical convergence.
    """

    max_iterations: int = 5
    pilot_samples: int = 100
    pilot_burn_in: int = 100
    pilot_stride: int = 1
    production_samples: int = 500
    production_burn_in: int = 200
    production_stride: int = 2
    relaxation: float = 0.25
    max_step: float = 0.2
    path_tolerance: float = 0.02
    min_average_sign: float = 0.05
    min_effective_samples: float = 10.0
    overlap_threshold: float = 1e-3
    nuclear_step: float = 0.1
    centroid_step: float = 0.1

    def __post_init__(self):
        for key in ("max_iterations", "pilot_samples", "pilot_stride",
                    "production_samples", "production_stride"):
            object.__setattr__(self, key, _positive_integer(getattr(self, key), key))
        for key in ("pilot_burn_in", "production_burn_in"):
            object.__setattr__(self, key, _nonnegative_integer(getattr(self, key), key))
        for key in ("relaxation", "max_step", "path_tolerance", "min_effective_samples",
                    "overlap_threshold", "nuclear_step", "centroid_step"):
            object.__setattr__(self, key, _finite_positive(getattr(self, key), key))
        object.__setattr__(self, "min_average_sign", _sign_threshold(self.min_average_sign))
        if self.relaxation > 1 or self.overlap_threshold >= 1:
            raise ValueError("relaxation must be <=1 and overlap_threshold must be <1")


@dataclass(frozen=True)
class ReweightedThermalEnsemble:
    """Positive MBAR target weights with separate initial SEO signs.

    ``weights`` target the unbiased absolute SEO distribution. They are not
    signed physical probabilities. Use ``expectation`` for signed averages or
    ``resample_parents`` for the existing thermal NRPMD correlation routines.
    The latter samples an empirical weighted distribution, not new independent
    equilibrium configurations. Source indices are retained for later bootstrap.
    """

    ensemble: ThermalEnsemble
    weights: np.ndarray
    cv_values: np.ndarray
    centers: np.ndarray
    stiffness: np.ndarray
    window_indices: np.ndarray
    sample_indices: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    min_average_sign: float = 0.05
    min_effective_samples: float = 10.0

    def __post_init__(self):
        if not isinstance(self.ensemble, ThermalEnsemble):
            raise TypeError("ensemble must be a ThermalEnsemble")
        n = len(self.ensemble.states)
        weights = _finite_array(self.weights, "weights", 1)
        values = _finite_array(self.cv_values, "cv_values", 2)
        centers = _finite_array(self.centers, "centers", 2)
        stiffness = _finite_array(self.stiffness, "stiffness", 3)
        if (weights.shape != (n,) or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=1e-12)
                or values.shape[0] != n or values.shape[1] != centers.shape[1]
                or stiffness.shape != (len(centers), centers.shape[1], centers.shape[1])):
            raise ValueError("invalid reweighted ensemble dimensions/normalization")
        for key in ("window_indices", "sample_indices"):
            value = np.asarray(getattr(self, key))
            if value.shape != (n,) or value.dtype.kind not in "iu" or np.any(value < 0):
                raise ValueError(f"{key} must contain one nonnegative integer per sample")
            if key == "window_indices" and np.any(value >= len(centers)):
                raise ValueError("window index exceeds window count")
            object.__setattr__(self, key, _readonly(value, int))
        for key, value in (("weights", weights), ("cv_values", values),
                           ("centers", centers), ("stiffness", stiffness)):
            object.__setattr__(self, key, _readonly(value))
        object.__setattr__(self, "diagnostics", copy.deepcopy(self.diagnostics))
        object.__setattr__(self, "min_average_sign", _sign_threshold(self.min_average_sign))
        object.__setattr__(self, "min_effective_samples", _finite_positive(
            self.min_effective_samples, "min_effective_samples"))

    @property
    def average_sign(self):
        return float(self.weights @ self.ensemble.signs)

    @property
    def unavailable_reasons(self):
        reasons = []
        if not self.diagnostics.get("mbar_converged", False):
            reasons.append("mbar_not_converged")
        if not self.diagnostics.get("overlap_connected", False):
            reasons.append("disconnected_umbrella_overlap")
        if self.average_sign < self.min_average_sign:
            reasons.append("average_sign_below_threshold")
        error = self.diagnostics.get("average_sign_standard_error_diagnostic", np.inf)
        if self.average_sign <= 2 * error:
            reasons.append("average_sign_not_resolved_by_diagnostic")
        if self.diagnostics.get("correlation_adjusted_ess_diagnostic", 0.) < self.min_effective_samples:
            reasons.append("target_effective_samples_below_threshold")
        return reasons

    def _require_available(self):
        reasons = self.unavailable_reasons
        if reasons:
            raise ValueError("thermal reweighting unavailable: " + ", ".join(reasons))

    def expectation(self, values):
        """sum(w * sign * A)/sum(w * sign); never clip signs or populations."""
        self._require_available()
        values = _finite_array(values, "observable")
        if values.ndim < 1 or len(values) != len(self.weights):
            raise ValueError("observable must have one leading entry per source sample")
        answer = np.tensordot(self.weights * self.ensemble.signs, values, axes=(0, 0))
        answer = answer / self.average_sign
        return float(answer) if np.ndim(answer) == 0 else answer

    def resample_parents(self, count, *, seed=0):
        """Multinomial importance resampling; copies every nuclear/mapping DOF.

        Draw with positive target weights, then retain each parent's SEO sign
        and projector estimator. Do not redraw a low-sign realization: that
        would bias the empirical sampling distribution.
        """
        self._require_available()
        count = _positive_integer(count, "count")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.weights), count, replace=True, p=self.weights)
        source = self.ensemble
        return ThermalEnsemble(
            tuple(source.states[i] for i in indices), source.signs[indices],
            source.electronic_populations[indices], source.masses, source.beta,
            source.potential, source.sweeps[indices],
            {"distribution": "MBAR_resampled_absolute_SEO_projected_trace",
             "potential_split": "trace", "static_bias": False,
             "source_static_bias": True, "parent_resampling": "mbar_multinomial",
             "source_window_indices": self.window_indices[indices].tolist(),
             "source_sample_indices": self.sample_indices[indices].tolist(),
             "source_flat_indices": indices.tolist(),
             "source_samples": len(self.weights), "resampled_parents": count,
             "unique_source_parents": len(np.unique(indices)),
             "source_reweighting": self.summary(),
             "uncertainty_requires_source_chain_bootstrap": True,
             "physical_converged": False},
        )

    def summary(self):
        return {**copy.deepcopy(self.diagnostics),
                "samples": len(self.weights), "average_sign": self.average_sign,
                "min_average_sign": self.min_average_sign,
                "min_effective_samples": self.min_effective_samples,
                "available": not self.unavailable_reasons,
                "unavailable_reasons": self.unavailable_reasons,
                "physical_converged": False}


def reweight_thermal_umbrellas(windows, cv, centers, stiffness, *,
                              min_average_sign=0.05, min_effective_samples=10.,
                              overlap_threshold=1e-3):
    """Combine frozen thermal umbrellas on exactly the same base distribution.

    The physical reduced potential cancels in MBAR, leaving beta*W_k. It may
    cancel only for identical provider, beta, masses, bead count and electronic
    representation. The bias/CV identity recorded by each sampler is checked.
    Window offsets returned in diagnostics describe the ABSOLUTE SEO measure.
    They are not the physical signed free-energy profile.
    """
    windows = tuple(windows)
    if not windows or not all(isinstance(w, ThermalEnsemble) for w in windows):
        raise ValueError("windows must contain ThermalEnsemble objects")
    min_average_sign = _sign_threshold(min_average_sign)
    min_effective_samples = _finite_positive(min_effective_samples, "min_effective_samples")
    overlap_threshold = _finite_positive(overlap_threshold, "overlap_threshold")
    if overlap_threshold >= 1:
        raise ValueError("overlap_threshold must be <1")
    centers = _finite_array(centers, "centers", 2)
    if centers.shape != (len(windows), cv.dimension):
        raise ValueError("centers must have shape [windows,CV dimension]")
    raw_stiffness = _finite_array(stiffness, "stiffness")
    if raw_stiffness.ndim > 3:
        raise ValueError("invalid stiffness dimensions")
    if raw_stiffness.ndim == 3 and len(raw_stiffness) != len(windows):
        raise ValueError("per-window stiffness must have shape [K,D,D]")
    stiffness_by_window = [
        UmbrellaBias(cv, center, raw_stiffness[i] if raw_stiffness.ndim == 3 else raw_stiffness)
        for i, center in enumerate(centers)
    ]
    first = windows[0]
    shape, mapping_shape = first.states[0].q.shape, first.states[0].mapping_q.shape
    samples, inefficiencies = [], []
    for i, (window, bias) in enumerate(zip(windows, stiffness_by_window)):
        if (window.potential is not first.potential or window.beta != first.beta
                or not np.array_equal(window.masses, first.masses)
                or window.states[0].q.shape != shape
                or window.states[0].mapping_q.shape != mapping_shape):
            raise ValueError("thermal umbrellas must share potential, beta, masses, beads and states")
        diagnostics = window.sampling_diagnostics
        if (diagnostics.get("static_bias") is not True
                or diagnostics.get("distribution") != "absolute_SEO_projected_trace_with_static_umbrella"
                or diagnostics.get("static_bias_identity") != _static_umbrella_identity(bias)):
            raise ValueError(f"window {i} does not match its frozen umbrella/CV identity")
        values = _cv_values(cv, window.states)
        samples.append(values)
        inefficiencies.append(max(
            cv_statistical_inefficiency(values, cv.periods),
            _varying_inefficiency(window.signs),
            *(_varying_inefficiency(window.electronic_populations[:, j])
              for j in range(mapping_shape[1])),
        ))
    values = np.concatenate(samples)
    counts = np.array([len(w.states) for w in windows], dtype=int)
    kappa = np.array([b.stiffness for b in stiffness_by_window])
    u_kn = umbrella_problem(values, centers, kappa, first.beta, cv.periods)
    fit = solve_mbar(u_kn, counts, overlap_threshold=overlap_threshold, return_state_weights=False)
    source = ThermalEnsemble(
        tuple(state for w in windows for state in w.states),
        np.concatenate([w.signs for w in windows]),
        np.concatenate([w.electronic_populations for w in windows]),
        first.masses, first.beta, first.potential, np.concatenate([w.sweeps for w in windows]),
        {"distribution": "unweighted_collection_of_frozen_thermal_umbrellas",
         "static_bias": True, "physical_converged": False},
    )
    frame_g = np.repeat(inefficiencies, counts)
    sign = float(fit.weights @ source.signs)
    weighted_g = float(np.sum(frame_g * fit.weights**2))
    sign_error = float(np.sqrt(np.sum(frame_g * fit.weights**2 * (source.signs-sign)**2)))
    return ReweightedThermalEnsemble(
        source, fit.weights, values, centers, kappa,
        np.repeat(np.arange(len(windows)), counts),
        np.concatenate([np.arange(n) for n in counts]),
        {"mbar_converged": bool(fit.converged), "mbar_residual": float(fit.residual),
         "overlap_connected": len(set(fit.components)) == 1,
         "overlap": fit.overlap.tolist(), "overlap_components": fit.components.tolist(),
         "overlap_threshold": overlap_threshold,
         "absolute_measure_window_free_energies": fit.free_energies.tolist(),
         "samples_per_window": counts.tolist(),
         "window_statistical_inefficiency_diagnostics": inefficiencies,
         "importance_ess": float(1 / np.sum(fit.weights**2)),
         "correlation_adjusted_ess_diagnostic": float(1 / weighted_g),
         "average_sign_standard_error_diagnostic": sign_error,
         "sign_error_scope": "heuristic_fixed_MBAR_weights_not_source_chain_bootstrap",
         "signed_ess_diagnostic": float(sign**2 / weighted_g),
         "uncertainty_requires_source_chain_bootstrap": True,
         "sampling_time_is_physical_time": False},
        min_average_sign, min_effective_samples,
    )


def _redistribute(path):
    """Equal-arclength interpolation; duplicate candidate images are collapsed."""
    distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate(([True], distances > 1e-14))
    points = path[keep]
    arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if len(points) < 2 or arc[-1] <= 1e-14:
        raise ValueError("reaction path collapsed to a single image")
    target = np.linspace(0., arc[-1], len(path))
    result = np.column_stack([np.interp(target, arc, points[:, j]) for j in range(path.shape[1])])
    result[0], result[-1] = path[0], path[-1]
    return result


def _update_path(centers, means, stiffness, scales, relaxation, max_step):
    scaled = centers / scales
    direction = (means - centers) / scales
    # Follow the actual restrained mean force in the chosen scaled metric.
    # Normalizing its positive Hessian scale keeps relaxation dimensionless.
    scaled_stiffness = stiffness * scales[:, None] * scales[None, :]
    direction = direction @ scaled_stiffness / np.linalg.eigvalsh(scaled_stiffness)[-1]
    for i in range(1, len(scaled)-1):
        tangent = scaled[i+1]-scaled[i-1]
        length = np.linalg.norm(tangent)
        if length <= 1e-14:
            raise ValueError("reaction path has an undefined interior tangent")
        tangent /= length
        direction[i] -= np.dot(direction[i], tangent)*tangent
    direction[[0, -1]] = 0.
    direction *= relaxation
    lengths = np.linalg.norm(direction, axis=1)
    direction *= np.minimum(1., max_step / np.maximum(lengths, np.finfo(float).tiny))[:, None]
    updated = _redistribute(scaled + direction) * scales
    updated[0], updated[-1] = centers[0], centers[-1]
    return updated


@dataclass(frozen=True)
class AdaptiveNRPMDPathResult:
    centers: np.ndarray
    initial_centers: np.ndarray
    cv_scales: np.ndarray
    history: tuple[dict, ...]
    reweighted: ReweightedThermalEnsemble
    stopping_reason: str

    def __post_init__(self):
        for key in ("centers", "initial_centers", "cv_scales"):
            object.__setattr__(self, key, _readonly(getattr(self, key)))
        object.__setattr__(self, "history", tuple(copy.deepcopy(self.history)))

    def summary(self):
        return {"method": "signed_thermal_nrpmd_restrained_string",
                "centers": self.centers.tolist(), "initial_centers": self.initial_centers.tolist(),
                "cv_scales": self.cv_scales.tolist(), "adaptation_rounds": len(self.history),
                "stopping_reason": self.stopping_reason,
                "path_change_below_tolerance": self.stopping_reason == "path_change_below_tolerance",
                "production_path_frozen": True, "pilot_samples_used_in_MBAR": False,
                "sampling_time_is_physical_time": False,
                "reweighting": self.reweighted.summary(), "physical_converged": False}


def run_adaptive_nrpmd_path(potential, masses, beta, images, nstates, cv, stiffness, *,
                          options=None, cv_scales=None, seed=0):
    """Learn a restrained path, freeze it, and reweight fresh thermal windows.

    ``images[K,P,N,3]`` are initial nuclear rings. Endpoint *CV centers* remain
    fixed; endpoint nuclei still sample their harmonic windows. A conservative
    sign diagnostic stops path motion when signed means are unresolved, then
    production may use a larger sample budget on the last well-defined path.
    No force bias or thermostat is added to subsequent NRPMD propagation.

    Supported CVs are the same immutable built-ins as thermal static umbrellas;
    periodic path topology is intentionally not inferred. Different unit CVs
    require meaningful explicit cv_scales. The potential is borrowed and must
    remain fixed throughout pilot, production, reweighting and propagation.
    """
    options = AdaptivePathOptions() if options is None else options
    if not isinstance(options, AdaptivePathOptions):
        raise TypeError("options must be AdaptivePathOptions")
    images = _finite_array(images, "images", 4)
    if images.shape[0] < 3 or min(images.shape[1:]) < 1 or images.shape[-1] != 3:
        raise ValueError("images must have shape [K>=3,P,N,3]")
    checker = CayleyIntegrator(masses, beta, images.shape[1], 0., native=False)
    if images.shape[1:] != checker.shape:
        raise ValueError("images do not match the nuclear masses")
    nstates = _positive_integer(nstates, "nstates")
    if any(p is not None for p in cv.periods):
        raise ValueError("adaptive NRPMD paths currently require nonperiodic CVs")
    centers = _finite_array([cv.value_and_jacobian(q).value for q in images], "initial centers", 2)
    if centers.shape != (len(images), cv.dimension):
        raise ValueError("initial CV centers have an inconsistent dimension")
    scales = np.ones(cv.dimension) if cv_scales is None else _finite_array(cv_scales, "cv_scales", 1)
    if scales.shape != (cv.dimension,) or np.any(scales <= 0):
        raise ValueError("cv_scales must contain one positive scale per CV")
    if cv_scales is None and len(set(cv.units)) > 1:
        raise ValueError("mixed CV units require explicit cv_scales")
    if (np.any(np.linalg.norm(np.diff(centers / scales, axis=0), axis=1) <= 1e-14)
            or np.linalg.norm((centers[-1]-centers[0])/scales) <= 1e-14):
        raise ValueError("initial path needs distinct adjacent images and distinct endpoints")
    prototype = UmbrellaBias(cv, centers[0], stiffness)
    _static_umbrella_identity(prototype)  # reject mutable/custom CVs before starting work
    cv = copy.deepcopy(cv)
    stiffness = np.array(prototype.stiffness, copy=True)
    initial_centers = centers.copy()
    seeds = np.random.SeedSequence(seed).spawn((options.max_iterations+1)*len(images))
    history = []
    stopping = "adaptation_budget_exhausted"

    def draw(q, center, child_seed, *, production):
        sampler = ThermalNRPMDSampler(
            potential, checker.masses, checker.beta, q, nstates,
            seed=child_seed, nuclear_step=options.nuclear_step,
            centroid_step=options.centroid_step, bias=UmbrellaBias(cv, center, stiffness),
        )
        return sampler.sample(
            options.production_samples if production else options.pilot_samples,
            burn_in=options.production_burn_in if production else options.pilot_burn_in,
            stride=options.production_stride if production else options.pilot_stride,
        )

    for iteration in range(options.max_iterations):
        means, signs, errors, unresolved = [], [], [], []
        for i, center in enumerate(centers):
            pilot = draw(images[i], center, seeds[iteration*len(images)+i], production=False)
            values = _cv_values(cv, pilot.states)
            sign = float(pilot.signs.mean())
            error = float(np.sqrt(_varying_inefficiency(pilot.signs)*np.var(pilot.signs)/len(pilot.signs)))
            signs.append(sign)
            errors.append(error)
            if sign < options.min_average_sign or sign <= 2*error:
                unresolved.append(i)
                means.append(center.copy())
            else:
                means.append(np.einsum("n,nd->d", pilot.signs, values)/pilot.signs.sum())
            images[i] = pilot.states[-1].q
        record = {"round": iteration, "centers": centers.tolist(),
                  "average_signs": signs, "sign_standard_error_diagnostics": errors,
                  "unresolved_sign_windows": unresolved,
                  "pilot_samples_per_window": options.pilot_samples}
        if unresolved:
            record["maximum_scaled_displacement"] = None
            history.append(record)
            stopping = "unresolved_pilot_sign"
            break
        updated = _update_path(centers, np.array(means), stiffness, scales,
                               options.relaxation, options.max_step)
        change = float(np.max(np.linalg.norm((updated-centers)/scales, axis=1)))
        record["maximum_scaled_displacement"] = change
        record["updated_centers"] = updated.tolist()
        history.append(record)
        centers = updated
        if change <= options.path_tolerance:
            stopping = "path_change_below_tolerance"
            break

    # These RNG streams are disjoint from every pilot round, including unused
    # rounds after early termination. Pilot samples never enter target weights.
    windows = tuple(draw(images[i], center, seeds[options.max_iterations*len(images)+i],
                         production=True) for i, center in enumerate(centers))
    reweighted = reweight_thermal_umbrellas(
        windows, cv, centers, stiffness, min_average_sign=options.min_average_sign,
        min_effective_samples=options.min_effective_samples,
        overlap_threshold=options.overlap_threshold,
    )
    return AdaptiveNRPMDPathResult(centers, initial_centers, scales, tuple(history), reweighted, stopping)
