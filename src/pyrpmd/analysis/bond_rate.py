"""Safeguarded 2D centroid-bond QTST and signed-flux postprocessing.

Requires an explicitly calibrated physical reference flux and conditional
normal speeds. A finite two-bond umbrella PMF cannot define bimolecular
reactant volume by itself, and old RPMDrate-xi recrossing cannot be reused.
"""
from __future__ import annotations
from .rate_postprocess import channel_rates, surface_rate_2d


def bond_2d_qtst(free_energy, edges, *, beta, curve, positive_speed=None,
                 reference_curve=None, reference_speed=None, entrance_rate_au=None,
                 surface_id=None, reference_surface_id=None, mask=None):
    """2D QTST candidate with a common-ensemble physical entrance bridge.

    Both curves must be complete physical flux sections of the same unbiased
    ensemble. Input metadata equality alone does not establish this fact.
    The speed is E[v_n Theta(v_n)] at each location, including angular and
    mass-metric dependence.
    """
    if not surface_id or surface_id != reference_surface_id:
        raise ValueError("a matching canonical surface identity is required")
    if positive_speed is None or reference_speed is None:
        raise ValueError("conditional positive normal speeds are required for both curves")
    if reference_curve is None or entrance_rate_au is None:
        raise ValueError("the physical free-reactant entrance-flux bridge is required")
    result = surface_rate_2d(
        free_energy, edges, beta=beta, curve=curve, mask=mask,
        normal_speed=positive_speed, reference_curve=reference_curve,
        reference_normal_speed=reference_speed, reference_rate_au=entrance_rate_au,
        measure="cv_density",
    )
    result["scope"] = ("candidate; matched physical entrance flux, constrained "
                       "velocity distribution, and convergence require independent evidence")
    result["surface_identity"] = surface_id
    return result


def bond_2d_signed_rate(static, time_au, parent_numerator, parent_denominator,
                        parent_ids, channel_names, *, surface_identity,
                        bootstrap=200, seed=0, **kwargs):
    """Channel-resolved signed flux from the *same* constrained bond surface.

    Child pairs sharing a parent must already be aggregated. Do not replace
    this estimator with product counts or the old xi transmission coefficient.
    """
    if static.get("surface_identity") != surface_identity:
        raise ValueError("signed dynamics and static 2D surface do not match")
    result = channel_rates(
        static["k_tst_au"], time_au, parent_numerator, parent_denominator,
        parent_ids, channel_names, bootstrap=bootstrap, seed=seed,
        source_identity=surface_identity, flux_identity=surface_identity, **kwargs,
    )
    return {
        "static":static, "dynamics":result, "converged":False,
        "rate_cm3_s":None,
        "physical_status":"candidate_until_entrance_and_dividing_surface_dynamics_converge",
    }
