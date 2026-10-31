"""Paired-flux statistics shared by gas, site and surface rate workflows.

Children sharing one parent configuration remain a single resampling unit.
The caller supplies its existing RNG, so the workflow's stream identity and
subsequent PMF draws are preserved. This module performs no propagation or I/O.
"""

from __future__ import annotations

import numpy as np

from .fes import _block_resample, statistical_inefficiency


def _plateau_diagnostics(curve, start, estimate, drift, drift_error,
                         boot_curves, uncertainty_available, tolerance):
    """Require resolved small variation, including oscillations, in the tail.

    Center each bootstrap curve on its own tail mean: uncertainty in the
    plateau's *height* must not be confused with uncertainty in its shape.
    The 95th percentile of the maximum centered bootstrap error supplies a
    simultaneous finite-sample envelope over the saved tail times. Neither
    a large error bar nor a zero half-tail drift can establish a plateau.
    This is conditional on the parent-block bootstrap describing mixing.
    """
    tail = curve[start:]
    centered = tail - estimate
    deviation = float(np.max(np.abs(centered)))
    limit = tolerance * abs(estimate)
    radius = drift_bound = deviation_bound = None
    reasons = []
    if len(tail) < 10:
        reasons.append("fewer_than_ten_tail_samples")
    if estimate <= 0:
        reasons.append("nonpositive_transmission")
    if not uncertainty_available:
        reasons.append("recrossing_uncertainty_unavailable")
    else:
        tails = boot_curves[:, start:]
        centered_boot = tails - tails.mean(axis=1, keepdims=True)
        radius = float(np.quantile(np.max(np.abs(centered_boot - centered), axis=1), .95))
        deviation_bound = deviation + radius
        drift_bound = abs(drift) + 2 * drift_error
        # Allow only floating-point roundoff at the comparison boundary.
        rounding = 32 * np.finfo(float).eps * max(abs(estimate), deviation, np.finfo(float).tiny)
        if drift_bound > limit + rounding:
            reasons.append("tail_drift_exceeds_tolerance")
        if deviation_bound > limit + rounding:
            reasons.append("tail_variation_exceeds_tolerance")
    return {
        "passed": not reasons, "reasons": reasons,
        "method": "half_tail_drift_and_simultaneous_centered_bootstrap_envelope",
        "scope": "selected_finite_time_tail", "tail_samples": len(tail),
        "relative_tolerance": tolerance, "absolute_tolerance": limit,
        "maximum_tail_deviation": deviation,
        "bootstrap_95_envelope_radius": radius,
        "maximum_tail_deviation_bound": deviation_bound,
        "absolute_drift_plus_two_standard_errors": drift_bound,
        "uncertainty_available": uncertainty_available,
    }


def summarize_recrossing(
    children, *, child_steps, plateau_fraction, plateau_tolerance, bootstrap, rng,
    extra_parent_observables=None, return_bootstrap_indices=False,
):
    """Group paired fluxes and resample chains and correlated parent blocks.

    A single sufficiently sampled chain supplies a *conditional* block error:
    its estimated correlation length is assumed to describe mixing. Multiple
    child pairs from one configuration cannot supply that missing evidence.
    Workflows may impose a stricter independent-chain policy for their primary
    uncertainty fields without changing the retained diagnostic draws.

    ``extra_parent_observables`` maps ``(chain, parent)`` to one or more finite
    observables whose correlation times must also bound the block length.
    Optional resample indices preserve covariance with quantities estimated
    from the same constrained parents (for example a surface mass metric).
    """
    if (isinstance(child_steps, (bool, np.bool_))
            or not isinstance(child_steps, (int, np.integer)) or child_steps < 2):
        raise ValueError("recrossing.child_steps must be at least 2")
    if not np.isfinite(plateau_tolerance) or plateau_tolerance <= 0:
        raise ValueError("recrossing.plateau_tolerance must be positive and finite")
    if not np.isfinite(plateau_fraction) or not 0 < plateau_fraction <= 1:
        raise ValueError("recrossing.plateau_fraction must be in (0,1]")
    grouped, identities = {}, set()
    for child in children:
        identity = (child["chain"], child["parent"], child["pair"])
        if identity in identities:
            raise ValueError("duplicate recrossing child identity")
        identities.add(identity)
        numerator = np.asarray(child["numerator"], float)
        denominator = float(child["denominator"])
        if (
            numerator.shape != (child_steps,)
            or not np.isfinite(numerator).all()
            or not np.isfinite(denominator)
            or denominator < 0
            or (denominator == 0 and np.any(numerator != 0))
        ):
            raise ValueError("invalid recrossing child flux record")
        key = identity[:2]
        entry = grouped.setdefault(key, [np.zeros(child_steps), 0.0])
        entry[0] += numerator
        entry[1] += denominator
    keys = sorted(grouped)
    extra = None
    if extra_parent_observables is not None:
        if set(extra_parent_observables) != set(keys):
            raise ValueError("extra parent observables must match the recrossing parents")
        extra = np.asarray([np.atleast_1d(extra_parent_observables[key]) for key in keys], dtype=float)
        if extra.ndim != 2 or not extra.shape[1] or not np.isfinite(extra).all():
            raise ValueError("extra parent observables must be finite equally sized vectors")
    num = np.asarray([grouped[key][0] for key in keys])
    den = np.asarray([grouped[key][1] for key in keys])
    if not len(keys) or den.sum() <= 0:
        raise ValueError("zero initial reactive flux; transmission cannot be estimated")
    curve = num.sum(axis=0) / den.sum()
    tail = min(len(curve), max(2, int(np.ceil(len(curve) * plateau_fraction))))
    start, middle = len(curve) - tail, len(curve) - tail + tail // 2
    estimate = float(curve[start:].mean())
    drift = float(curve[middle:].mean() - curve[start:middle].mean())
    chain_indices = {
        chain: np.asarray([i for i, key in enumerate(keys) if key[0] == chain])
        for chain in sorted({key[0] for key in keys})
    }
    blocks, effective_blocks = {}, {}
    for chain, ix in chain_indices.items():
        observable = np.divide(
            num[ix, start:].mean(axis=1), den[ix],
            out=np.zeros(len(ix)), where=den[ix] > 0,
        )
        blocks[chain] = max(1, int(np.ceil(statistical_inefficiency(observable))))
        if extra is not None:
            for column in extra[ix].T:
                # A constant metric adds no within-chain variance. The generic
                # correlation estimator conservatively returns the whole chain
                # for constants; using that here would reduce resampling to
                # circular rotations and erase genuine parent-flux uncertainty.
                if np.ptp(column) > 0:
                    blocks[chain] = max(
                        blocks[chain], int(np.ceil(statistical_inefficiency(column)))
                    )
        # Zero-flux configurations contribute no information to a ratio of
        # flux sums and must not make a one-configuration sample look adequate.
        effective_blocks[chain] = float(np.count_nonzero(den[ix] > 0) / blocks[chain])
    independent_chains = sum(np.any(den[ix] > 0) for ix in chain_indices.values())
    nonzero_parents = int(np.count_nonzero(den > 0))
    scope = "unavailable"
    support_reasons = []
    if independent_chains >= 2:
        scope = "independent_parent_chains"
    elif sum(effective_blocks.values()) >= 2:
        scope = "conditional_on_single_chain_mixing"
    else:
        if nonzero_parents < 2:
            support_reasons.append("fewer_than_two_nonzero_flux_parent_configurations")
        support_reasons.append("fewer_than_two_effective_parent_blocks")

    # Keep the established chain-choice / circular-block RNG draw order.
    boot_curves = []
    boot_indices = []
    chains = list(chain_indices)
    for _ in range(bootstrap):
        selected = []
        for chain in rng.choice(chains, size=len(chains), replace=True):
            ix = chain_indices[chain]
            selected.extend(_block_resample(ix, blocks[chain], rng).tolist())
        ix = np.asarray(selected, int)
        if den[ix].sum() > 0:
            boot_curves.append(num[ix].sum(axis=0) / den[ix].sum())
            if return_bootstrap_indices:
                boot_indices.append(ix.copy())
    boot_curves = np.asarray(boot_curves).reshape((-1, len(curve)))
    draws = boot_curves[:, start:].mean(axis=1)
    reasons = list(support_reasons)
    if len(draws) < 2:
        reasons.append("fewer_than_two_valid_recrossing_bootstrap_draws")
    rejected = bootstrap - len(draws)
    if rejected:
        reasons.append("recrossing_bootstrap_contains_zero_flux_draws")
    available = not reasons
    error = float(draws.std(ddof=1)) if available else None
    curve_error = boot_curves.std(axis=0, ddof=1) if available else np.full_like(curve, np.nan)
    drift_error = (
        float((boot_curves[:, middle:].mean(axis=1) - boot_curves[:, start:middle].mean(axis=1)).std(ddof=1))
        if available else None
    )
    plateau_diagnostics = _plateau_diagnostics(
        curve, start, estimate, drift, drift_error, boot_curves,
        available, plateau_tolerance,
    )
    plateau = plateau_diagnostics["passed"]
    result = {
        "parent_ids": keys, "parent_numerator": num, "parent_denominator": den,
        "curve": curve, "start": start, "kappa": estimate, "drift": drift,
        "bootstrap_curves": boot_curves, "bootstrap_kappa": draws,
        "standard_error": error, "curve_standard_error": curve_error,
        "drift_standard_error": drift_error, "plateau_detected": plateau,
        "plateau_diagnostics": plateau_diagnostics,
        "parent_chains": len(chain_indices),
        # Public diagnostics must survive JSON checkpoint/summary round trips
        # without changing mapping keys. Internal RNG indexing stays integer.
        "parent_block_lengths": {str(chain): length for chain, length in blocks.items()},
        "uncertainty": {
            "available": available, "reasons": reasons, "scope": scope,
            "condition": (
                "The estimated correlation blocks describe mixing within this single chain; independent-chain agreement has not been established."
                if scope == "conditional_on_single_chain_mixing" else None
            ),
            "independent_parent_chains": int(independent_chains),
            "nonzero_flux_parent_configurations": nonzero_parents,
            "effective_parent_blocks": {str(chain): count for chain, count in effective_blocks.items()},
            "bootstrap_requested": int(bootstrap), "bootstrap_valid": len(draws),
            "bootstrap_rejected": rejected,
        },
    }
    if return_bootstrap_indices:
        result["bootstrap_parent_indices"] = boot_indices
    return result
