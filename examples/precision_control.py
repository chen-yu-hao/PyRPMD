"""Synthetic pilot-planning examples; no dynamics or physical claims.

Run from the repository with:
    PYTHONPATH=src python examples/precision_control.py
"""

from dataclasses import replace
import json

from pyrpmd.precision import (
    CheckSchedule, NumericalRequirement, PrecisionSpec, PrecisionTarget,
    RateEvidence, SamplingDesign, ScientificIdentity,
)
from pyrpmd.precision_planning import recommend_supplement


def main():
    identity = ScientificIdentity("synthetic-model-T300", 32, 1.0, "fixed-umbrella")
    spec = PrecisionSpec(identity, PrecisionTarget(.1), CheckSchedule((4, 8)),
                         max_cost=1000., numerical=(NumericalRequirement("beads", 64, .05),))
    design = SamplingDesign(4, 10000, 4, 40, 8, 1000)
    pilot = RateEvidence(
        identity.identity_id, source_ids=("synthetic-pilot",), rate=1.,
        pmf_log_standard_error=.01, recrossing_log_standard_error=.02,
        components_independent=True, degrees_of_freedom=19,
        pmf_ready=True, overlap_connected=True, minimum_umbrella_ess=100.,
        independent_parent_chains=4, effective_parent_blocks=20.,
        plateau_established=True, energy_complete=True, maximum_energy_error_kbt=.01,
    )
    cases = {
        "pmf_limited": replace(pilot, pmf_log_standard_error=.3),
        "parent_limited": replace(pilot, independent_parent_chains=1),
        "child_horizon_limited": replace(pilot, plateau_established=False,
                                          plateau_failure="horizon"),
    }
    costs = {"add_umbrella_streams": 100., "add_parent_chains": 150.,
             "extend_child_horizon": 200.}
    result = {}
    for name, evidence in cases.items():
        plan = recommend_supplement(spec, evidence, design, costs=costs, remaining_cost=1000.)
        result[name] = {"status": plan["status"], "action": plan["actions"][0]["kind"],
                        "estimated_cost": plan["estimated_cost"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
