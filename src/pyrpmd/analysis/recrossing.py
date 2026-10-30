"""Paired-flux statistics shared by gas, site and surface rate workflows.

Children sharing one parent configuration remain a single resampling unit.
The caller supplies its existing RNG, so the workflow's stream identity and
subsequent PMF draws are preserved. This module performs no propagation or I/O.
"""

from __future__ import annotations

import numpy as np

from .fes import _block_resample, statistical_inefficiency


def summarize_recrossing(
    children, *, child_steps, plateau_fraction, plateau_tolerance, bootstrap, rng
):
    """Group paired fluxes and resample chains and correlated parent blocks.

    A single sufficiently sampled chain supplies a *conditional* block error:
    its estimated correlation length is assumed to describe mixing. Multiple
    child pairs from one configuration cannot supply that missing evidence.
    Workflows may impose a stricter independent-chain policy for their primary
    uncertainty fields without changing the retained diagnostic draws.
    """
    if (isinstance(child_steps, (bool, np.bool_))
            or not isinstance(child_steps, (int, np.integer)) or child_steps < 2):
        raise ValueError("recrossing.child_steps must be at least 2")
    if not np.isfinite(plateau_tolerance) or plateau_tolerance <= 0:
        raise ValueError("recrossing.plateau_tolerance must be positive and finite")
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
    chains = list(chain_indices)
    for _ in range(bootstrap):
        selected = []
        for chain in rng.choice(chains, size=len(chains), replace=True):
            ix = chain_indices[chain]
            selected.extend(_block_resample(ix, blocks[chain], rng).tolist())
        ix = np.asarray(selected, int)
        if den[ix].sum() > 0:
            boot_curves.append(num[ix].sum(axis=0) / den[ix].sum())
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
    threshold = max(plateau_tolerance * abs(estimate), 2 * (drift_error or 0.0))
    plateau = bool(tail >= 10 and abs(drift) <= threshold and estimate > 0)
    return {
        "parent_ids": keys, "parent_numerator": num, "parent_denominator": den,
        "curve": curve, "start": start, "kappa": estimate, "drift": drift,
        "bootstrap_curves": boot_curves, "bootstrap_kappa": draws,
        "standard_error": error, "curve_standard_error": curve_error,
        "drift_standard_error": drift_error, "plateau_detected": plateau,
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
