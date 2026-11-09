"""Rate-owned umbrella diagnostics, separate from sampling and PMF choice.

UI retains the historical MBAR overlap/support checks, without a generic FES
bootstrap. MBAR rates reuse their own fitted weights, so exporting compatible
umbrella diagnostics never solves the same production problem twice.
"""

from __future__ import annotations

import numpy as np


def ui_umbrella_diagnostics(streams, config, options):
    """Check UI's empirical support and overlap; do not estimate uncertainty."""
    from ..analysis.fes import reconstruct_fes

    data = reconstruct_fes(
        [s["samples"] for s in streams],
        [s["center"] for s in streams],
        [s["stiffness"] for s in streams],
        config["beta"],
        bins=options["bins"],
        ranges=None if options["ranges"] is None else [options["ranges"]],
        bootstrap=0,
        min_count=options["min_count"],
        overlap_threshold=options["overlap_threshold"],
    )
    data["measure"] = "rate_flux"
    return data


def mbar_umbrella_diagnostics(estimate, streams, beta, min_count):
    """Export the selected rate fit in the historical stream-indexed schema.

    The rate solver pools identical Hamiltonians. Split each destination
    state's overlap column by its constituent stream counts to recover the
    original stream-level matrix; frame weights and stream order are unchanged.
    This is an algebraic expansion, not a second fit or bootstrap.
    """
    from ..analysis.umbrella import _histogram

    counts = np.asarray([len(s["samples"]) for s in streams])
    state_map = np.asarray(estimate["solver"]["stream_to_state"], dtype=int)
    state_counts = np.bincount(state_map, weights=counts)
    fractions = counts / state_counts[state_map]
    overlap = np.asarray(estimate["overlap"])[np.ix_(state_map, state_map)] * fractions[None, :]
    edges = [np.asarray(estimate["edges"])]
    mass, density, raw_count, free, mask = _histogram(
        np.concatenate([s["samples"] for s in streams]), estimate["weights"],
        edges, beta, min_count,
    )
    connected = bool(estimate["overlap_connected"])
    if not connected or not estimate["solver"]["converged"]:
        free[:] = np.nan
        mask[:] = True
    # These files are point-estimate diagnostics, not a second uncertainty
    # product. The authoritative paired-endpoint errors are in rate_pmf.*.
    error = np.full_like(free, np.nan)
    return {
        "free_energy": free, "probability_mass": mass, "density": density,
        "standard_error": error, "mask": mask, "raw_count": raw_count,
        "edges": edges, "overlap": overlap,
        "components": np.asarray(estimate["components"])[state_map],
        "connected": connected,
        "window_free_energies": estimate["window_free_energies"],
        "statistical_inefficiency": estimate["statistical_inefficiency"],
        "effective_samples_per_stream": estimate["effective_samples_per_stream"],
        "weight_ess": estimate["weight_ess"], "weights": estimate["weights"],
        "bootstrap_replicates": 0, "bootstrap_requested": 0, "bootstrap_attempted": 0,
        "bootstrap_valid_per_bin": np.zeros(free.shape, dtype=int),
        "bootstrap_reference_bin": None, "bootstrap_diagnostics": [],
        "bootstrap_block_plan": estimate["bootstrap"]["block_plan"],
        "standard_error_scope": "unavailable; rate_pmf owns paired-endpoint bootstrap",
        "measure": "rate_flux", "energy_unit": "hartree",
    }


def write_rate_umbrella_diagnostics(sampled, data, system, method, *, solver=None):
    """Preserve sample/FES exports and adaptive provenance after rate analysis."""
    from .fes import _write_fes, resolve_umbrellas

    config = sampled["sampling_config"]
    cv, _, _, _ = resolve_umbrellas(config, system)

    def annotate(result):
        result["summary"].update(
            analysis_owner="rate", role="rate_umbrella_diagnostics",
            rate_estimator=method, uncertainty_owner="rate_pmf",
        )
        if solver is not None and not solver["converged"]:
            result["summary"]["status"] = "mbar_not_converged"
            result["summary"]["warnings"].append(
                "MBAR did not converge; diagnostic histogram free energies are masked."
            )
        postprocess = sampled.get("postprocess")
        return result if postprocess is None else postprocess(result)

    sampled["data"] = data
    sampled["summary"] = _write_fes(
        sampled["output"], data, sampled["streams"], cv, config, annotate,
    )
