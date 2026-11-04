"""Provider-free presentation of saved path-sampling rate diagnostics."""

from copy import deepcopy


def collect_path_sampling_summary(run):
    """Retain the producer's qualification and uncertainty without promotion.

    Conditional-on-flux errors and FFS single-bank diagnostics must not be
    presented as complete rate uncertainties.  This collector only copies
    explicitly saved fields; it never imports input code or recreates a PES.
    """
    summary = run["_summary"]
    fields = (
        "method", "path_sampling_method", "exchange", "samples", "burn_in", "workers", "interfaces",
        "completed", "sampling_support_complete", "operations_completed", "burn_in_completed",
        "coordinate", "coordinate_unit", "forward_dynamics",
        "ensemble_order", "ensemble_samples", "pending_tickets",
        "conditional_probabilities", "overall_crossing_probability",
        "crossing_probability_product", "crossing_probability_standard_error_block",
        "flux_au", "flux_s_inv", "flux_standard_error_block_au",
        "rate_s_inv", "rate_estimate_au", "rate_estimate_s_inv",
        "rate_standard_error_au", "rate_standard_error_block_au",
        "rate_standard_error_s_inv", "rate_standard_error_block_s_inv",
        "rate_standard_error_block_conditional_on_flux_au",
        "block_diagnostics", "block_size", "complete_blocks", "supported_blocks",
        "flux_diagnostics", "effective_crossings", "a_history_time_au",
        "discarded_unknown_history_time_au", "local_transition_support_complete",
        "requires_independent_flux", "memory_loss_approximation",
        "swap_attempts", "swap_accepts", "swap_length_rejections",
        "move_attempts", "move_accepts", "length_rejections",
        "zero_swap_attempts", "zero_swap_accepts", "zero_swap_length_rejections",
        "qualification", "uncertainty_qualification", "physical_converged",
    )
    run["metrics"].update({key: deepcopy(summary[key]) for key in fields if key in summary})
    run["metrics"]["rate_unit"] = "s^-1"
    if summary.get("rate_s_inv") is None:
        run["findings"].append({
            "level": "info", "code": "path_sampling_rate_withheld",
            "message": "Path sampling reports candidate first-order rates. The saved "
                       "physical rate remains withheld; completed sampling does not "
                       "establish physical convergence.",
        })
    for key, code in (
        ("qualification", "path_sampling_qualification"),
        ("uncertainty_qualification", "path_sampling_uncertainty"),
    ):
        if isinstance(summary.get(key), str) and summary[key]:
            run["findings"].append({"level": "info", "code": code, "message": summary[key]})
