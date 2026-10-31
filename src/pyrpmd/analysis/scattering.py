"""Equal-weight, fixed-energy scattering estimates with explicit missing outcomes.

This module estimates contributions from the sampled impact-parameter disk. It
does not convert a specified initial state and collision energy into a thermal
rate, infer quantum state-to-state probabilities, or establish convergence.
"""

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np


_STATUSES = ("product", "nonreactive", "unresolved", "failed")
_INITIALIZATION_MODES = ("independent", "parent_chain", "unknown")


def _positive_finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be positive and finite") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _integer(value, name, minimum):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _state_key(descriptor):
    if descriptor is not None and not isinstance(descriptor, (Mapping, str)):
        raise ValueError("initial_state must be a JSON-safe mapping, string, or None")
    try:
        encoded = json.dumps(descriptor, sort_keys=True, allow_nan=False,
                             separators=(",", ":"))
    except (TypeError, ValueError):
        raise ValueError("initial_state must be JSON-safe with finite values") from None
    return encoded, json.loads(encoded)


def _wilson(count, n):
    z = 1.959963984540054
    p = count / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_fixed_energy_scattering(
    records,
    *,
    area_bohr2,
    initialization_mode,
    channels=(),
    collision_energy_au=None,
    initial_state=None,
    initial_state_key="initial_state",
    classifier_validated=True,
    cutoff_validated=True,
    impact_sampling="uniform_area",
    weights=None,
    bootstrap_seed=0,
    bootstrap_replicates=1000,
    min_parent_clusters=8,
):
    """Summarize one collision energy, including channel and initial-state groups.

    Parameters
    ----------
    records : iterable of mappings
        Each record needs ``status``; a product also needs a nonempty string
        ``channel``. ``launch.collision_energy_au`` may supply the common energy.
        ``parent_id`` identifies independent parent chains when that mode is
        selected. The JSON-safe ``initial_state`` descriptor groups nuclear
        initial states; ``initial_state`` here supplies a fallback descriptor.
    area_bohr2 : float
        The sampled area, pi*bmax**2. Records must sample b=bmax*sqrt(U).
    initialization_mode : {"independent", "parent_chain", "unknown"}
        An explicit assertion about sampling, never inferred from trajectory or
        parent counts. Independent means independent draws of the complete
        initial condition, not merely independent collision orientations.
    weights : array-like, optional
        Only equal, positive weights are supported. A common factor cancels in
        the probability estimate; electronic degeneracies are not applied here.
    classifier_validated, cutoff_validated : bool
        Caller-supplied convergence/validation gates. An unvalidated classifier
        withholds chemical outcome bounds as well as final estimates. An
        unvalidated bmax withholds final estimates; disk-only bounds remain.
    min_parent_clusters : int, default 8
        A cautious software gate, not a universal statistical guarantee. Parent
        resampling assumes independent chains and preserves all children of each
        sampled parent. The caller must assess whether enough chains were run.

    Notes
    -----
    All launched records, including numerical failures, stay in the denominator.
    ``observed_lower_bohr2`` and ``possible_upper_bohr2`` bracket assignments of
    missing outcomes in this finite sample; they are not confidence intervals.
    Failures are distinct from trajectories censored at their time limit, even
    though both have unknown eventual outcomes for these conservative bounds.

    Wilson intervals require independent initialization, a single initial state,
    complete outcomes and validated classification/cutoff. A parent-cluster
    percentile bootstrap is available for parent-chain initialization under the
    corresponding gates. It is withheld when parent fractions have no variation,
    including zero-event samples: a degenerate bootstrap cannot bound unseen
    events. Pooled different initial states have an empirical-mixture estimate,
    but no pooled interval; conditional intervals are calculated per state.
    """
    records = list(records)
    if not records:
        raise ValueError("Cannot summarize zero trajectories")
    area = _positive_finite(area_bohr2, "area_bohr2")
    if initialization_mode not in _INITIALIZATION_MODES:
        raise ValueError(f"initialization_mode must be one of {_INITIALIZATION_MODES}")
    if impact_sampling != "uniform_area":
        raise ValueError("Only equal-weight uniform_area impact sampling is supported")
    if not isinstance(initial_state_key, str) or not initial_state_key:
        raise ValueError("initial_state_key must be a nonempty string")
    for name, value in (("classifier_validated", classifier_validated),
                        ("cutoff_validated", cutoff_validated)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
    seed = _integer(bootstrap_seed, "bootstrap_seed", 0)
    replicates = _integer(bootstrap_replicates, "bootstrap_replicates", 0)
    if 0 < replicates < 100:
        raise ValueError("bootstrap_replicates must be zero or at least 100")
    minimum = _integer(min_parent_clusters, "min_parent_clusters", 2)
    if weights is not None:
        weight = np.asarray(weights, dtype=float)
        if (weight.shape != (len(records),) or not np.isfinite(weight).all()
                or np.any(weight <= 0) or not np.all(weight == weight[0])):
            raise ValueError("Only equal, positive finite trajectory weights are supported")

    if isinstance(channels, str):
        raise ValueError("channels must be an iterable of nonempty channel names")
    try:
        channel_names = set(channels)
    except TypeError:
        raise ValueError("channels must be an iterable of nonempty channel names") from None
    if any(not isinstance(name, str) or not name.strip() for name in channel_names):
        raise ValueError("channels must contain nonempty string names")
    reference_energy = (None if collision_energy_au is None else
                        _positive_finite(collision_energy_au, "collision_energy_au"))
    descriptors, state_indices, parent_keys = {}, {}, []
    trajectory_ids = set()
    for i, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("status") not in _STATUSES:
            raise ValueError("Every record must have a recognized trajectory status")
        channel = record.get("channel")
        if ((channel is not None and (not isinstance(channel, str) or not channel.strip()))
                or (record["status"] == "product" and channel is None)):
            raise ValueError("Product records require a nonempty string channel")
        if record["status"] == "product":
            channel_names.add(channel)
        if record.get("classification_unvalidated", False):
            classifier_validated = False
        trajectory_id = record.get("trajectory_id")
        if trajectory_id is not None:
            try:
                if trajectory_id in trajectory_ids:
                    raise ValueError("Duplicate trajectory_id would double-count a launch")
                trajectory_ids.add(trajectory_id)
            except TypeError:
                raise ValueError("trajectory_id must be hashable") from None

        launch = record.get("launch", {})
        if not isinstance(launch, Mapping):
            raise ValueError("launch metadata must be a mapping")
        if launch.get("sampling", "fixed_energy") != "fixed_energy":
            raise ValueError("Fixed-energy analysis cannot mix thermal collision samples")
        if launch.get("impact_sampling", "uniform_area") != "uniform_area":
            raise ValueError("Only uniform_area impact sampling is supported")
        if "area_au" in launch:
            launch_area = _positive_finite(launch["area_au"], "launch.area_au")
            if not math.isclose(area, launch_area, rel_tol=1e-12):
                raise ValueError("Records must use one common impact-parameter area")
        if "collision_energy_au" in launch:
            energy = _positive_finite(launch["collision_energy_au"],
                                      "launch.collision_energy_au")
            if reference_energy is None:
                reference_energy = energy
            elif not math.isclose(reference_energy, energy, rel_tol=1e-12):
                raise ValueError("Records must use one common fixed collision energy")

        key, descriptor = _state_key(record.get(initial_state_key, initial_state))
        descriptors[key] = descriptor
        state_indices.setdefault(key, []).append(i)
        parent = record.get("parent_id")
        if parent is not None:
            if (isinstance(parent, (bool, np.bool_))
                    or not isinstance(parent, (str, int, np.integer))
                    or (isinstance(parent, str) and not parent)):
                raise ValueError("parent_id must be an integer, nonempty string, or None")
            parent = ("string", parent) if isinstance(parent, str) else ("integer", int(parent))
        parent_keys.append(parent)
    if reference_energy is None:
        raise ValueError("Specify collision_energy_au or include it in launch metadata")
    channel_names = sorted(channel_names)

    def summarize_group(indices, *, mixed_states=False, seed_index=0):
        subset = [records[i] for i in indices]
        n = len(subset)
        counts = Counter(record["status"] for record in subset)
        unknown_count = counts["unresolved"] + counts["failed"]
        products = np.array([record["status"] == "product" for record in subset])
        matrix = np.column_stack([products] + [
            [record["status"] == "product" and record["channel"] == channel
             for record in subset] for channel in channel_names
        ]).astype(float)
        parents = {}
        for local, source in enumerate(indices):
            parents.setdefault(parent_keys[source], []).append(local)
        parents_complete = None not in parents
        parent_count = len(parents) - int(not parents_complete)
        reasons = []
        if not classifier_validated:
            reasons.append("classifier_unvalidated")
        if not cutoff_validated:
            reasons.append("impact_cutoff_unvalidated")
        if unknown_count:
            reasons.append("unresolved_or_failed_outcomes")
        if mixed_states:
            reasons.append("mixed_initial_states_require_stratified_uncertainty")
        if initialization_mode == "unknown":
            reasons.append("initialization_independence_not_established")
        elif initialization_mode == "parent_chain":
            if not parents_complete:
                reasons.append("missing_parent_ids")
            if parent_count < minimum:
                reasons.append("too_few_independent_parent_clusters")
            if not replicates:
                reasons.append("bootstrap_disabled")
        draws, cluster_fractions = None, None
        if initialization_mode == "parent_chain" and not reasons:
            groups = list(parents.values())
            cluster_counts = np.array([matrix[group].sum(axis=0) for group in groups])
            sizes = np.array([len(group) for group in groups])
            cluster_fractions = cluster_counts / sizes[:, None]
            rng = np.random.default_rng(np.random.SeedSequence([seed, 19, seed_index]))
            draws = np.empty((replicates, matrix.shape[1]))
            for j in range(replicates):
                selected = rng.integers(0, parent_count, size=parent_count)
                draws[j] = cluster_counts[selected].sum(axis=0) / sizes[selected].sum()

        def estimate(column):
            count = int(matrix[:, column].sum())
            fraction = count / n
            entry_reasons = list(reasons)
            wilson, bootstrap, method = None, None, "unavailable"
            if not entry_reasons and initialization_mode == "independent":
                wilson = [area * bound for bound in _wilson(count, n)]
                method = "iid_wilson"
            elif not entry_reasons and initialization_mode == "parent_chain":
                if np.ptp(cluster_fractions[:, column]) == 0:
                    entry_reasons.append("no_between_parent_fraction_variation")
                else:
                    bootstrap = (area * np.quantile(draws[:, column], [0.025, 0.975])).tolist()
                    method = "parent_cluster_percentile_bootstrap"
            return {
                "count": count,
                "event_fraction": fraction,
                "candidate_only": not bool(classifier_validated),
                "cross_section_bohr2": (
                    area * fraction
                    if not unknown_count and classifier_validated and cutoff_validated else None
                ),
                "observed_lower_bohr2": area * fraction if classifier_validated else None,
                "possible_upper_bohr2": (
                    area * (count + unknown_count) / n if classifier_validated else None
                ),
                "iid_wilson95_bohr2": wilson,
                "parent_cluster_bootstrap95_bohr2": bootstrap,
                "uncertainty": {
                    "method": method,
                    "confidence_level": 0.95 if method != "unavailable" else None,
                    "reason": entry_reasons[0] if entry_reasons else None,
                    "unavailability_reasons": entry_reasons,
                    "assumed_independent_units": (
                        n if initialization_mode == "independent" else
                        parent_count if initialization_mode == "parent_chain" and parents_complete
                        else None
                    ),
                },
            }

        return {
            "trajectories": n,
            "counts": {status: counts[status] for status in _STATUSES},
            "parents": parent_count,
            "all_trajectories_resolved": unknown_count == 0,
            "unknown_outcomes": {
                "unresolved": counts["unresolved"],
                "failed": counts["failed"],
                "included_in_denominator": True,
                "treated_as_nonreactive": False,
            },
            "total_reaction": estimate(0),
            "channels": {channel: estimate(i + 1) for i, channel in enumerate(channel_names)},
        }

    summary = summarize_group(range(len(records)), mixed_states=len(state_indices) > 1)
    summary.update({
        "format": "PyRPMD.scattering.summary.v1",
        "sampling": "fixed_energy",
        "impact_sampling": impact_sampling,
        "weighting": "equal_weight",
        "area_bohr2": area,
        "collision_energy_au": reference_energy,
        "initialization_mode": initialization_mode,
        "classification_unvalidated": not bool(classifier_validated),
        "cutoff_unvalidated": not bool(cutoff_validated),
        "bounds_scope": "sampled_impact_parameter_disk",
        "initial_state_scope": (
            "sampled_initial_state_mixture" if len(state_indices) > 1 else
            "unspecified" if next(iter(descriptors.values())) is None else "single_initial_state"
        ),
        "initial_state": next(iter(descriptors.values())) if len(descriptors) == 1 else None,
        "bootstrap_replicates": replicates,
        "min_parent_clusters": minimum,
        "uncertainty_note": (
            "Missing-outcome bounds are finite-sample assignment bounds, not confidence limits. "
            "Wilson requires independent complete initial conditions. Parent resampling assumes "
            "independent chains; the minimum cluster count is a software gate, not proof of accuracy. "
            "Intervals are marginal per channel, not simultaneous confidence regions."
        ),
        "convergence_note": (
            "Bounds apply only within the sampled impact disk and assume correct resolved outcomes. "
            "Check bmax, launch/exit separation, outcome criteria, maximum time, timestep, bead count "
            "and initial-state stability separately. Fixed-energy data are not a thermal rate."
        ),
        "initial_state_groups": [],
    })
    for i, key in enumerate(sorted(state_indices)):
        group = summarize_group(state_indices[key],
                                seed_index=i + 1 if len(state_indices) > 1 else 0)
        group.update({
            "initial_state": descriptors[key],
            "initial_state_id": "sha256:" + hashlib.sha256(key.encode()).hexdigest(),
            "sampling_fraction": len(state_indices[key]) / len(records),
        })
        summary["initial_state_groups"].append(group)
    return summary
