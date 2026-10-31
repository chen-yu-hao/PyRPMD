"""Trace thermodynamics, isotope fractionation and open-path momentum estimates.

All energies, masses, lengths and momenta use atomic units.  No function in
this module interprets fictitious PIMD/RPMD momenta as nuclear quantum momenta.
References and normalization conventions are in docs/quantum-statistics.md.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp

from .analysis.fes import statistical_inefficiency
from .constants import KB_AU
from .path_integral import (
    OpenPathSamples, PathIntegralAction, PathIntegralSamples,
    _integer, _masses, _positive,
)
from .potentials.base import validate_result


def _series(values, name, minimum=1):
    data = np.asarray(values)
    if data.dtype.kind not in "fiu" or data.ndim != 1 or len(data) < minimum:
        raise ValueError(f"{name} must be a real vector with at least {minimum} entries")
    data = np.asarray(data, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must be finite")
    return data


def _block_layout(streams, block_size=None):
    # Some derivative terms vanish identically (e.g. P=1 primitive springs)
    # and classical isotope weights can be algebraically constant.  Such a
    # stream contributes no estimator variance; it must not force one block.
    # This convention does not establish mixing of the configuration chain.
    inefficiency = max(1.0 if np.ptp(stream) == 0 else float(statistical_inefficiency(stream))
                       for stream in streams)
    block_size = max(1, int(np.ceil(inefficiency))) if block_size is None else _integer(block_size, "block_size")
    count = len(streams[0]) // block_size
    return {"statistical_inefficiency": inefficiency,
            "block_length_frames": block_size, "complete_blocks": count,
            "effective_samples": len(streams[0]) / inefficiency,
            "uncertainty_scope": "conditional_on_blocks_exceeding_the_correlation_time"}


def _jackknife_error(values):
    values = np.asarray(values, dtype=float)
    count = len(values)
    return float(np.sqrt((count - 1) / count * np.sum((values - values.mean())**2)))


def _mean_cosine(momentum, displacement):
    """Fourier mean with at most 256*4096 intermediate real elements."""
    result = np.zeros_like(momentum)
    for pi in range(0, len(momentum), 256):
        stop = min(pi + 256, len(momentum))
        for si in range(0, len(displacement), 4096):
            result[pi:stop] += np.cos(momentum[pi:stop, None] * displacement[None, si:si + 4096]).sum(axis=1)
    return result / len(displacement)


def _scaled_temperature_derivatives(action, paths, terms, relative_step):
    """Derivatives after scaling fluctuations so the free spring stays fixed."""
    relative_step = _positive(relative_step, "finite_difference_step")
    if relative_step >= 1 or relative_step < 8 * np.finfo(float).eps:
        raise ValueError("finite_difference_step must be resolvable and strictly below 1 (relative beta step)")
    center = paths.mean(axis=1, keepdims=True)
    fluctuation = paths - center
    potential_reference = np.asarray(terms["potential_action"]) / action.beta
    correction_reference = np.asarray(terms["correction_action"])

    def residual(shift):
        beta = action.beta * (1 + shift)
        scaled = center + np.sqrt(1 + shift) * fluctuation
        target = PathIntegralAction(action.potential, action.masses_au,
                                   1 / (KB_AU * beta), method=action.method)
        evaluated = target.evaluate_batch(scaled)
        # Subtract the path-specific baseline potential before differentiating.
        # Its known linear beta term is added to A' analytically.  This avoids
        # unnecessary cancellation from a large constant PES energy offset.
        physical = beta * (evaluated["potential_action"] / beta - potential_reference)
        return physical + evaluated["correction_action"] - correction_reference

    def derivatives(step):
        plus, minus = residual(step), residual(-step)
        h = action.beta * step
        first = potential_reference + (plus - minus) / (2 * h)
        second = (plus + minus) / h**2
        if not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError("scaled-coordinate finite difference overflow; change step or inspect the PES")
        return first, second

    coarse_first, coarse_second = derivatives(relative_step)
    fine_first, fine_second = derivatives(relative_step / 2)
    dof = 3 * paths.shape[2]
    coarse_energy = 0.5 * dof / action.beta + coarse_first
    fine_energy = 0.5 * dof / action.beta + fine_first
    coarse_heat = 0.5 * dof - action.beta**2 * coarse_second.mean() + (action.beta * coarse_first).var()
    fine_heat = 0.5 * dof - action.beta**2 * fine_second.mean() + (action.beta * fine_first).var()
    sensitivity = {
        "relative_beta_step_requested": relative_step,
        "relative_beta_step_used": relative_step / 2,
        "energy_hartree_coarse_minus_fine": float(coarse_energy.mean() - fine_energy.mean()),
        "heat_capacity_over_kB_coarse_minus_fine": float(coarse_heat - fine_heat),
        "maximum_dimensionless_first_derivative_change": float(action.beta * np.max(np.abs(coarse_first - fine_first))),
        "maximum_dimensionless_second_derivative_change": float(action.beta**2 * np.max(np.abs(coarse_second - fine_second))),
        "scope": "same_sample_h_and_h_over_2_sensitivity; not proof_of_finite_difference_convergence",
        "additional_PES_evaluations_per_configuration": 4,
    }
    return fine_energy, fine_second, sensitivity


def energy_heat_capacity(samples_or_action, positions=None, *, block_size=None,
                         estimator="thermodynamic", finite_difference_step=1e-3):
    """Internal energy and constant-volume heat capacity of a closed trace.

    Accept a ``PathIntegralSamples`` result, or an action plus configurations
    sampled from that same action.  The caller must not pass primitive samples
    labelled as TI samples (or conversely).  For action S and beta=1/kBT,

        E = dP/(2 beta) + <S'>,
        Cv/kB = dP/2 - beta² <S''> + beta² Var(S').

    TI contributes 3*S_TI/beta to E and 6*S_TI/beta² to S''.
    The variance is an ensemble central moment, not the ring-Hamiltonian
    fluctuation.  Finite correlated samples can yield negative estimates;
    these are returned with conditional block errors, never clipped.

    Optional ``estimator='scaled_coordinates'`` rescales fluctuations around
    each centroid by sqrt(beta'/beta), keeping the spring action invariant.
    The residual trace normalization is beta'^(-d/2), so the formulas above
    use d instead of dP and derivatives of the physical-plus-TI potential
    action at the scaled coordinates.  Central finite differences at relative
    beta steps h and h/2 are compared; the finer step supplies the reported
    estimate.  ``finite_difference_step`` is dimensionless (default 1e-3).
    Its sensitivity diagnostic must be inspected separately from sampling
    error, especially for numerically noisy forces or very small steps.
    """
    if isinstance(samples_or_action, PathIntegralSamples):
        if positions is not None:
            raise ValueError("positions must be omitted when passing PathIntegralSamples")
        samples = samples_or_action
        action, paths, terms = samples.action, samples.positions, samples.terms
    elif isinstance(samples_or_action, PathIntegralAction):
        action, paths = samples_or_action, np.asarray(positions)
        terms = action.evaluate_batch(paths)
    else:
        raise TypeError("closed-path thermodynamics requires PathIntegralSamples or PathIntegralAction")
    energy = _series(terms["energy_estimator"], "energy estimator", minimum=2)
    second = _series(terms["second_beta_derivative"], "second beta derivative", minimum=2)
    if energy.shape != second.shape or paths.ndim != 4 or len(paths) != len(energy):
        raise ValueError("sample/estimator shapes disagree")
    if estimator not in ("thermodynamic", "scaled_coordinates"):
        raise ValueError("heat-capacity estimator must be thermodynamic or scaled_coordinates")
    finite_difference = None
    normalization_dof = 3 * paths.shape[1] * paths.shape[2]
    if estimator == "scaled_coordinates":
        energy, second, finite_difference = _scaled_temperature_derivatives(
            action, paths, terms, finite_difference_step)
        normalization_dof = 3 * paths.shape[2]
    scaled_energy = action.beta * energy
    scaled_second = action.beta**2 * second
    heat = 0.5 * normalization_dof - scaled_second.mean() + scaled_energy.var()
    centered_squared = (scaled_energy - scaled_energy.mean())**2
    diagnostics = _block_layout((scaled_energy, centered_squared, scaled_second), block_size)
    count, length = diagnostics["complete_blocks"], diagnostics["block_length_frames"]
    error_energy = error_heat = None
    if count >= 4:
        n = count * length
        blocks_e = scaled_energy[:n].reshape(count, length)
        blocks_s = scaled_second[:n].reshape(count, length)
        mean_e = blocks_e.mean(axis=1)
        # Center before squaring to avoid cancellation for PES energy offsets.
        centered = blocks_e - blocks_e.mean()
        mean_c = centered.mean(axis=1)
        mean_cc = (centered**2).mean(axis=1)
        mean_s = blocks_s.mean(axis=1)
        leave_c = (mean_c.sum() - mean_c) / (count - 1)
        leave_cc = (mean_cc.sum() - mean_cc) / (count - 1)
        leave_s = (mean_s.sum() - mean_s) / (count - 1)
        leave_heat = 0.5 * normalization_dof - leave_s + leave_cc - leave_c**2
        error_heat = _jackknife_error(leave_heat)
        error_energy = float(mean_e.std(ddof=1) / math.sqrt(count) / action.beta)
    diagnostics["reason"] = None if count >= 4 else "fewer_than_four_complete_blocks"
    return {"method": action.method, "temperature_K": action.temperature_K,
            "estimator": estimator, "finite_difference": finite_difference,
            "beads": int(paths.shape[1]), "samples": len(energy),
            "internal_energy_hartree": float(energy.mean()),
            "internal_energy_standard_error_hartree": error_energy,
            "heat_capacity_over_kB": float(heat),
            "heat_capacity_standard_error_over_kB": error_heat,
            "heat_capacity_hartree_per_K": float(heat * KB_AU),
            "diagnostics": diagnostics, "physical_converged": False,
            "scope": "closed_trace_thermodynamics_of_temperature_independent_PES"}


def isotope_substitution_free_energy(samples, new_masses_au, *, estimator="scaled_coordinates",
                                     minimum_effective_samples=20.0, minimum_ess_fraction=0.01,
                                     block_size=None):
    """Mass-substitution free energy from closed primitive or TI samples.

    The default scaled-coordinate estimator rescales each atom's fluctuations
    around its centroid by sqrt(m_old/m_new), preserving its free spring
    action.  The remaining prefactor is prod(m_new/m_old)^(3/2), including
    the integration Jacobian.  The TI force-squared action is reevaluated at
    the transformed coordinates with the *new* masses.

    The optional thermodynamic estimator uses untransformed coordinates and
    the prefactor prod(m_new/m_old)^(3P/2); its overlap often deteriorates with
    P.  ESS, weight concentration and a correlation estimate are diagnostics,
    not proof of adequate overlap; unsampled tails cannot be diagnosed.
    """
    if not isinstance(samples, PathIntegralSamples):
        raise TypeError("isotope reweighting requires closed PathIntegralSamples")
    if estimator not in ("scaled_coordinates", "thermodynamic"):
        raise ValueError("estimator must be scaled_coordinates or thermodynamic")
    minimum_effective_samples = _positive(minimum_effective_samples, "minimum_effective_samples")
    minimum_ess_fraction = _positive(minimum_ess_fraction, "minimum_ess_fraction")
    if minimum_ess_fraction > 1:
        raise ValueError("minimum_ess_fraction must be <= 1")
    action = samples.action
    masses = _masses(new_masses_au)
    if masses.shape != action.masses_au.shape:
        raise ValueError("new masses must match the source atom count")
    q = samples.positions
    if q.ndim != 4 or len(q) < 2:
        raise ValueError("isotope reweighting requires at least two closed configurations")
    ratio = masses / action.masses_au
    if not np.isfinite(ratio).all():
        raise ValueError("mass ratio overflow")
    target = PathIntegralAction(action.potential, masses, action.temperature_K, method=action.method)
    if estimator == "scaled_coordinates":
        center = q.mean(axis=1, keepdims=True)
        transformed = center + (q - center) / np.sqrt(ratio)[None, None, :, None]
        prefactor = 1.5 * np.log(ratio).sum()
    else:
        transformed = q
        prefactor = 1.5 * q.shape[1] * np.log(ratio).sum()
    target_terms = target.evaluate_batch(transformed)
    source_terms = samples.terms
    # Under the scaled transform the spring action is exactly invariant;
    # omitting its numerical subtraction improves cancellation at large P.
    keys = ("potential_action", "correction_action") if estimator == "scaled_coordinates" else ("spring_action", "potential_action", "correction_action")
    log_weights = sum(np.asarray(source_terms[key]) - target_terms[key] for key in keys)
    log_weights = _series(log_weights, "log weights", minimum=2)
    if len(log_weights) != len(q):
        raise ValueError("sample/weight shapes disagree")
    log_mean = float(logsumexp(log_weights) - math.log(len(log_weights)))
    normalized = np.exp(log_weights - logsumexp(log_weights))
    ess = float(1 / np.sum(normalized**2))
    shifted = np.exp(log_weights - np.max(log_weights))
    diagnostics = _block_layout((shifted,), block_size)
    correlated_ess = ess / diagnostics["statistical_inefficiency"]
    adequate = correlated_ess >= minimum_effective_samples and ess / len(q) >= minimum_ess_fraction
    count, length = diagnostics["complete_blocks"], diagnostics["block_length_frames"]
    error = None
    if count >= 4:
        block_means = shifted[:count * length].reshape(count, length).mean(axis=1)
        leave = (block_means.sum() - block_means) / (count - 1)
        if np.all(leave > 0):
            error = _jackknife_error(np.log(leave))
    diagnostics.update({"reweighting_ess": ess, "reweighting_ess_fraction": ess / len(q),
                        "correlation_adjusted_ess": correlated_ess,
                        "maximum_normalized_weight": float(normalized.max()),
                        "overlap_adequate": bool(adequate),
                        "minimum_effective_samples": minimum_effective_samples,
                        "minimum_ess_fraction": minimum_ess_fraction,
                        "overlap_limit": "ESS cannot detect unsampled target-distribution tails"})
    log_ratio = float(prefactor + log_mean)
    return {"estimator": estimator, "action_method": action.method,
            "temperature_K": action.temperature_K, "beads": q.shape[1], "samples": len(q),
            "substitutions": [{"atom": int(i), "old_mass_au": float(action.masses_au[i]), "new_mass_au": float(masses[i])}
                              for i in np.flatnonzero(masses != action.masses_au)],
            "log_partition_ratio_new_over_old": log_ratio,
            "free_energy_new_minus_old_hartree": -log_ratio / action.beta,
            "log_partition_ratio_standard_error": error,
            "diagnostics": diagnostics, "physical_converged": False}


def isotope_fractionation(phase_a, phase_b, *, allow_poor_overlap=False):
    """Return alpha_(A/B)=(new/old isotope ratio in A)/(new/old ratio in B).

    Inputs are results from ``isotope_substitution_free_energy`` for the same
    isotope substitutions and temperature.  The free-energy errors are combined
    assuming independently sampled phases.  Failed overlap diagnostics are
    rejected unless explicitly requested for diagnostic output.
    """
    if not isinstance(allow_poor_overlap, (bool, np.bool_)):
        raise ValueError("allow_poor_overlap must be boolean")
    if not np.isclose(phase_a["temperature_K"], phase_b["temperature_K"], rtol=1e-12, atol=0):
        raise ValueError("fractionation phases must have the same temperature")
    substitutions = lambda x: sorted((v["old_mass_au"], v["new_mass_au"]) for v in x["substitutions"])
    if substitutions(phase_a) != substitutions(phase_b) or not substitutions(phase_a):
        raise ValueError("fractionation requires identical nonempty mass substitutions in both phases")
    adequate = all(x["diagnostics"]["overlap_adequate"] for x in (phase_a, phase_b))
    if not adequate and not allow_poor_overlap:
        raise ValueError("isotope fractionation has poor reweighting overlap; sample intermediate masses or improve sampling")
    log_alpha = phase_a["log_partition_ratio_new_over_old"] - phase_b["log_partition_ratio_new_over_old"]
    if not np.isfinite(log_alpha):
        raise ValueError("fractionation requires finite log partition ratios")
    errors = [x["log_partition_ratio_standard_error"] for x in (phase_a, phase_b)]
    error = None if any(x is None for x in errors) else math.hypot(*errors)
    alpha = math.exp(log_alpha) if log_alpha <= math.log(np.finfo(float).max) else None
    return {"alpha_A_over_B": alpha, "log_alpha_A_over_B": float(log_alpha),
            "1000_log_alpha": float(1000 * log_alpha), "log_alpha_standard_error": error,
            "temperature_K": phase_a["temperature_K"], "overlap_adequate": bool(adequate),
            "definition": "(new/old isotope ratio in phase A)/(new/old isotope ratio in phase B)",
            "physical_converged": False}


def fractionation_mass_integration(masses_au, kinetic_A_hartree, kinetic_B_hartree, temperature_K):
    """Integrate beta*(K_A-K_B) d(log m) over one isotope mass ladder.

    K is the physical quantum kinetic energy of the substituted atom, not
    fictitious-momentum kinetic energy.  Input kinetic estimates must be
    consistent with their path-integral factorization.  The quadrature-grid
    error and sampling uncertainty require independent refinement.
    """
    masses = _series(masses_au, "mass ladder", minimum=2)
    ka = _series(kinetic_A_hartree, "phase A kinetic energies", minimum=2)
    kb = _series(kinetic_B_hartree, "phase B kinetic energies", minimum=2)
    temperature = _positive(temperature_K, "temperature_K")
    if ka.shape != masses.shape or kb.shape != masses.shape or np.any(masses <= 0) or np.any(np.diff(masses) <= 0):
        raise ValueError("mass ladder must be positive, strictly increasing, and match both kinetic arrays")
    log_alpha = float(trapezoid(ka - kb, x=np.log(masses)) / (KB_AU * temperature))
    return {"log_alpha_A_over_B": log_alpha,
            "alpha_A_over_B": math.exp(log_alpha) if log_alpha <= math.log(np.finfo(float).max) else None,
            "1000_log_alpha": 1000 * log_alpha, "temperature_K": temperature,
            "method": "trapezoidal_mass_thermodynamic_integration",
            "mass_grid_points": len(masses), "physical_converged": False}


def quantum_kinetic_energy(samples, *, estimator="auto", block_size=None):
    """Per-atom physical kinetic energy from closed-path coordinate estimators.

    Primitive samples default to the centroid-virial estimator,
    K_a=3/(2*beta)-sum_j[(q_ja-qbar_a).F_ja]/(2*P).  TI samples use the
    mass derivative K_a=(3*P/2-S_spring,a+S_TI,a)/beta, including the explicit
    inverse-mass dependence of the TI action.  The primitive virial formula
    is rejected for TI samples, for which it is not a consistent estimator.
    """
    if not isinstance(samples, PathIntegralSamples):
        raise TypeError("quantum kinetic energy requires closed PathIntegralSamples")
    action, q = samples.action, samples.positions
    if estimator == "auto":
        estimator = "centroid_virial" if action.method == "primitive" else "thermodynamic"
    if estimator not in ("centroid_virial", "thermodynamic"):
        raise ValueError("kinetic estimator must be auto, centroid_virial or thermodynamic")
    if estimator == "centroid_virial" and action.method != "primitive":
        raise ValueError("primitive centroid-virial kinetic estimator is not valid for TI samples")
    if q.ndim != 4 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError("kinetic estimates require at least two finite closed configurations")
    frames, beads, atoms, _ = q.shape
    flat = np.ascontiguousarray(q.reshape(frames * beads, atoms, 3))
    forces = validate_result(action.potential.evaluate(flat), flat.shape).forces.reshape(q.shape)
    if estimator == "centroid_virial":
        values = 1.5 / action.beta - np.sum((q - q.mean(axis=1, keepdims=True)) * forces, axis=(1, 3)) / (2 * beads)
    else:
        delta = q - np.roll(q, 1, axis=1)
        spring = beads / (2 * action.beta) * np.sum(delta**2 * action.masses_au[None, None, :, None], axis=(1, 3))
        correction = np.zeros_like(spring)
        if action.method == "takahashi_imada":
            correction = action.beta**3 / (24 * beads**3) * np.sum(forces**2 / action.masses_au[None, None, :, None], axis=(1, 3))
        values = (1.5 * beads - spring + correction) / action.beta
    errors, diagnostics = [], []
    for atom in range(atoms):
        layout = _block_layout((values[:, atom],), block_size)
        count, length = layout["complete_blocks"], layout["block_length_frames"]
        error = None if count < 4 else float(values[:count * length, atom].reshape(count, length).mean(axis=1).std(ddof=1) / math.sqrt(count))
        errors.append(error)
        diagnostics.append(layout)
    return {"kinetic_energy_per_atom_hartree": values.mean(axis=0).tolist(),
            "standard_error_per_atom_hartree": errors, "estimator": estimator,
            "action_method": action.method, "temperature_K": action.temperature_K,
            "diagnostics_per_atom": diagnostics, "physical_converged": False}


def momentum_distribution_from_open_paths(samples, momenta_au, *, bandwidth_bohr=None, block_size=None):
    """Estimate a 1D nuclear momentum marginal from one-component-open paths.

    For normalized end-to-end density g(s), the normalized momentum density is
    ``n(p)=FT[g](p)/(2*pi*g(0))``. A symmetric Gaussian kernel of declared bandwidth
    h gives ``FT[g](p)=exp(-h²*p²/2)*mean(cos(p*s))``, and g(0) is estimated using
    the same kernel.  This preserves the infinite-grid Fourier normalization;
    it does *not* renormalize a short momentum grid or clip negative noise.
    Finite bandwidth introduces bias: check bandwidth, momentum range, beads
    and sampling independently.  The output contains the retained integral,
    negative area and conditional jackknife errors for this purpose.
    """
    if not isinstance(samples, OpenPathSamples):
        raise TypeError("momentum distribution requires OpenPathSamples, not closed-path or fictitious momenta")
    displacement = _series(samples.displacements, "open end-to-end displacements", minimum=4)
    momentum = _series(momenta_au, "momentum grid", minimum=3)
    if np.any(np.diff(momentum) <= 0) or momentum[0] >= 0 or momentum[-1] <= 0:
        raise ValueError("momentum grid must increase strictly and span negative and positive momenta")
    diagnostics = _block_layout((displacement, displacement**2), block_size)
    if bandwidth_bohr is None:
        scale = float(displacement.std(ddof=1))
        if scale <= 0:
            raise ValueError("constant open-path displacement cannot determine a momentum distribution")
        bandwidth = 1.06 * scale * max(1.0, diagnostics["effective_samples"])**(-0.2)
    else:
        bandwidth = _positive(bandwidth_bohr, "bandwidth_bohr")
    origin_kernel = np.exp(-0.5 * (displacement / bandwidth)**2) / (math.sqrt(2 * math.pi) * bandwidth)
    g0 = float(origin_kernel.mean())
    if g0 <= np.finfo(float).tiny:
        raise ValueError("open-path density at zero is unresolved; improve sampling or bandwidth")
    probe_indices = np.unique(np.linspace(0, len(momentum) - 1, min(9, len(momentum))).astype(int))
    probes = tuple(np.cos(momentum[i] * displacement) for i in probe_indices)
    diagnostics = _block_layout((displacement, displacement**2, origin_kernel) + probes, block_size)
    diagnostics["correlation_observables"] = "displacement, displacement squared, origin-density kernel and representative cosine transforms"
    diagnostics["cosine_probe_momenta_au"] = momentum[probe_indices].tolist()
    diagnostics["origin_density_kernel_effective_samples"] = float(origin_kernel.sum()**2 / np.sum(origin_kernel**2))
    diagnostics["uncertainty_scope"] = "conditional_on_blocks_exceeding_all_kernel_and_Fourier_correlation_times; only representative_cosines_probed"
    damping = np.exp(-0.5 * (bandwidth * momentum)**2)
    cosine_mean = _mean_cosine(momentum, displacement)
    density = damping * cosine_mean / (2 * math.pi * g0)
    count, length = diagnostics["complete_blocks"], diagnostics["block_length_frames"]
    error = None
    if count >= 4:
        # Bound both the stored block matrix and runtime: group adjacent blocks
        # if the short autocorrelation estimate would create thousands of blocks.
        group = max(1, int(np.ceil(count / 128)))
        length *= group
        count = len(displacement) // length
        diagnostics["jackknife_block_length_frames"] = length
        diagnostics["jackknife_complete_blocks"] = count
        block_cos = np.array([_mean_cosine(momentum, displacement[b * length:(b + 1) * length])
                              for b in range(count)])
        block_origin = origin_kernel[:count * length].reshape(count, length).mean(axis=1)
        leave_origin = (block_origin.sum() - block_origin) / (count - 1)
        if np.all(leave_origin > np.finfo(float).tiny):
            leave_cos = (block_cos.sum(axis=0) - block_cos) / (count - 1)
            leave_density = damping[None] * leave_cos / (2 * math.pi * leave_origin[:, None])
            error = np.sqrt((count - 1) / count * np.sum((leave_density - leave_density.mean(axis=0))**2, axis=0))
    return {"momentum_au": momentum, "density_per_atomic_momentum": density,
            "conditional_standard_error": error, "bandwidth_bohr": float(bandwidth),
            "end_to_end_density_at_zero_per_bohr": g0,
            "integral_on_supplied_momentum_grid": float(trapezoid(density, x=momentum)),
            "negative_area_on_supplied_grid": float(trapezoid(np.maximum(-density, 0), x=momentum)),
            "normalization": "Fourier transform divided by 2*pi*g(0); no finite-grid renormalization",
            "atom": samples.action.atom, "axis": samples.action.axis,
            "temperature_K": samples.action.temperature_K, "imaginary_time_links": samples.positions.shape[1] - 1,
            "diagnostics": diagnostics, "physical_converged": False}


def harmonic_log_partition(beta, frequencies_au, *, beads=None, method="primitive"):
    """Independent analytic harmonic trace, exact or discretized, for validation.

    Frequencies enumerate Cartesian normal modes (not atoms); all must be
    strictly positive.  ``beads=None`` returns the exact quantum partition
    function.  A finite P uses the determinant of the cyclic Gaussian action,
    log Z=-sum log[2*sinh(P*asinh(beta*omega_eff/(2P)))].
    """
    beta = _positive(beta, "beta")
    frequencies = _series(frequencies_au, "frequencies")
    if np.any(frequencies <= 0):
        raise ValueError("harmonic frequencies must be positive")
    if method not in ("primitive", "takahashi_imada", "ti"):
        raise ValueError("unknown harmonic action method")
    x = beta * frequencies
    if beads is None:
        y = 0.5 * x
    else:
        beads = _integer(beads, "beads")
        effective = x * np.sqrt(1 + x**2 / (12 * beads**2)) if method != "primitive" else x
        y = beads * np.arcsinh(effective / (2 * beads))
    if not np.isfinite(y).all():
        raise ValueError("harmonic partition parameters overflow")
    return float(-np.sum(y + np.log(-np.expm1(-2 * y))))
