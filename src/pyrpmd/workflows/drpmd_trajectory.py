"""Restartable adiabatic and mapping-state collision propagation.

Collision assembly, centroid connectivity and flux estimators are adapted from
chen-yu-hao/RPMDrate, commit 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78,
rpmdrate/drpmd.py (MIT). Original RPMDrate copyright (c) 2012 Joshua W. Allen,
William H. Green and Yury V. Suleimanov; see THIRD_PARTY_NOTICES.md and LICENSE.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import time

import numpy as np

from pyrpmd.dynamics import State
from pyrpmd.validation import checked_integer as _integer
from .drpmd_classification import _normalize_label
from .drpmd_persistence import _checkpoint_clock_matches, _load_npz, _mapping_arrays


def _nrpmd_trajectory(dynamics, state, classifier, config, out, trajectory_id,
                      parent_id, launch, physics_hash, resume, *, save_arrays, write_metadata):
    """Signed mapping-state scattering with complete electronic restart state.

    The Python step boundary is intentional: scalar-PES fused loops cannot
    propagate mapping variables. Failed physical/classifier steps roll back
    together, while checkpoint I/O failures abort the job without relabeling it.
    """
    from pyrpmd.nonadiabatic import NRPMDState

    c, outcomes = config["collision"], config.get("outcomes", {})
    na = config["nonadiabatic"]
    max_steps = _integer(c["max_steps"], "collision.max_steps")
    check = _integer(outcomes.get("check_interval", 10), "outcomes.check_interval")
    persistence = _integer(outcomes.get("persistence_steps", check * 3),
                           "outcomes.persistence_steps", 0)
    output = config.get("output", {})
    stride = _integer(output.get("trajectory_stride", max(1, max_steps // 1000)), "trajectory_stride")
    checkpoint_interval = _integer(output.get("checkpoint_steps", max(1, max_steps // 10)), "checkpoint_steps")
    tolerance = float(na["asymptotic_coupling_tolerance"])
    base = Path(out) / "trajectories" / f"trajectory-{trajectory_id:08d}"
    result_path, checkpoint = base.with_suffix(".json"), base.with_suffix(".checkpoint.npz")
    if result_path.exists():
        if not resume:
            raise FileExistsError(f"Trajectory already exists: {result_path}; use resume explicitly")
        result = json.loads(result_path.read_text())
        if result.get("physics_hash") != physics_hash or result.get("trajectory_id") != trajectory_id:
            raise ValueError("Completed trajectory does not match this restart")
        population = np.asarray(result.get("electronic_populations"), dtype=float)
        if population.shape != (na["nstates"],) or not np.isfinite(population).all():
            raise ValueError("Completed NRPMD trajectory lacks valid electronic populations")
        return result
    initial = final = drift = None
    done, candidate, candidate_since = 0, None, None
    status, channel, reason = "unresolved", None, "maximum_propagation_time"
    trace_steps, trace_q, trace_energy, trace_population = [], [], [], []
    if resume and checkpoint.exists():
        meta, arrays = _load_npz(checkpoint, expected_kind="PyRPMD.drpmd.trajectory-checkpoint.v1")
        if meta.get("physics_hash") != physics_hash or meta.get("trajectory_id") != trajectory_id:
            raise ValueError("Trajectory checkpoint has incompatible physical parameters")
        if "mapping_q" not in arrays or "mapping_p" not in arrays:
            raise ValueError("NRPMD checkpoint lacks electronic mapping state")
        state = NRPMDState(arrays["q"], arrays["p"], meta["step"], meta["time"],
                           arrays["mapping_q"], arrays["mapping_p"])
        if state.mapping_q.shape != (config["beads"], na["nstates"]):
            raise ValueError("NRPMD checkpoint mapping shape does not match configuration")
        done = meta["done"]
        initial, final, drift = meta["initial"], meta["final"], meta["drift"]
        candidate = tuple(meta["candidate"]) if meta["candidate"] is not None else None
        candidate_since = meta["candidate_since"]
        status, channel, reason = meta["status"], meta["channel"], meta["reason"]
        trace_steps = arrays["trace_steps"].tolist()
        trace_q = list(arrays["trace_centroid"])
        trace_energy = arrays["trace_energy"].tolist()
        trace_population = list(arrays["trace_populations"])
        _integer(done, "checkpoint.done", 0)
        if done > max_steps or state.step != done or not _checkpoint_clock_matches(
                state.time, done, config["dt_au"]):
            raise ValueError("NRPMD checkpoint has inconsistent trajectory clock")
        length = len(trace_steps)
        if (arrays["trace_centroid"].shape != (length, *state.q.shape[1:])
                or arrays["trace_energy"].shape != (length,)
                or arrays["trace_populations"].shape != (length, na["nstates"])
                or arrays["trace_steps"].dtype.kind not in "iu"
                or (length and (trace_steps[0] != 0 or trace_steps[-1] > done))
                or np.any(np.diff(trace_steps) <= 0)
                or any(not np.isfinite(arrays[key]).all() for key in
                       ("trace_centroid", "trace_energy", "trace_populations"))):
            raise ValueError("NRPMD checkpoint has invalid trajectory traces")
    else:
        try:
            if dynamics.asymptotic_coupling(state) > tolerance:
                raise ValueError("Incoming electronic channels are not asymptotically decoupled")
            initial = float(dynamics.energy(state)) / config["beads"]
            if not math.isfinite(initial):
                raise FloatingPointError("Nonfinite initial NRPMD energy")
            final, drift = initial, 0.0
            trace_steps, trace_q, trace_energy, trace_population = (
                [0], [state.q.mean(axis=0).copy()], [initial], [state.populations().copy()])
        except Exception as exc:
            status, reason = "failed", f"initial_energy: {type(exc).__name__}: {exc}"
    started = time.perf_counter()

    def save_checkpoint():
        save_arrays(checkpoint, {
            "format": "PyRPMD.drpmd.trajectory-checkpoint.v1", "physics_hash": physics_hash,
            "trajectory_id": trajectory_id, "parent_id": parent_id,
            "step": state.step, "time": state.time, "done": done,
            "initial": initial, "final": final, "drift": drift,
            "candidate": candidate, "candidate_since": candidate_since,
            "status": status, "channel": channel, "reason": reason, "launch": launch,
            "propagation": {"native_steps": 0, "python_steps": done},
        }, q=state.q, p=state.p, **_mapping_arrays(state),
            trace_steps=np.asarray(trace_steps, dtype=np.int64),
            trace_centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
            trace_energy=np.asarray(trace_energy),
            trace_populations=np.asarray(trace_population).reshape((-1, na["nstates"])))

    while done < max_steps and status == "unresolved":
        before = state.copy()
        previous_candidate, previous_since = candidate, candidate_since
        try:
            dynamics.step(state)
            energy = float(dynamics.energy(state)) / config["beads"]
            populations = state.populations()
            if not math.isfinite(energy) or not np.isfinite(populations).all():
                raise FloatingPointError("Nonfinite NRPMD diagnostics")
            number = done + 1
            if classifier is not None and number % check == 0:
                label = _normalize_label(classifier(state))
                if label is not None and dynamics.asymptotic_coupling(state) > tolerance:
                    label = None
                if label != candidate:
                    candidate, candidate_since = label, number if label else None
                if label is not None and number - candidate_since >= persistence:
                    status, channel = label
                    reason = "stable_separated_outgoing_decoupled_channels"
        except Exception as exc:
            state = before
            candidate, candidate_since = previous_candidate, previous_since
            status, channel, reason = "failed", None, f"{type(exc).__name__}: {exc}"
            break
        done += 1
        final, drift = energy, max(drift, abs(energy - initial))
        if done % stride == 0 or done == max_steps or status != "unresolved":
            trace_steps.append(done)
            trace_q.append(state.q.mean(axis=0).copy())
            trace_energy.append(energy)
            trace_population.append(populations.copy())
        if done % checkpoint_interval == 0 or status != "unresolved":
            save_checkpoint()
    if classifier is None and status == "unresolved":
        reason = "no_validated_product_classifier"
    save_checkpoint()
    save_arrays(base.with_suffix(".npz"), {
        "format": "PyRPMD.drpmd.trajectory.v1", "physics_hash": physics_hash,
        "trajectory_id": trajectory_id, "parent_id": parent_id, "launch": launch,
        "hamiltonian_convention": "sum_beads_beta_over_P",
        "units": {"q": "bohr", "p": "atomic", "time": "atomic", "energy": "hartree"},
        "electronic_estimator": "bead_mean_MMST_signed",
    }, final_q=state.q, final_p=state.p, **_mapping_arrays(state, "final_"),
        steps=np.asarray(trace_steps, dtype=np.int64),
        time_au=np.asarray(trace_steps) * config["dt_au"],
        centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
        energy_per_bead=np.asarray(trace_energy),
        electronic_populations=np.asarray(trace_population).reshape((-1, na["nstates"])))
    record = {
        "trajectory_id": trajectory_id, "parent_id": parent_id, "physics_hash": physics_hash,
        "status": status, "channel": channel, "reason": reason, "steps": done,
        "classification_unvalidated": bool(outcomes.get("classifier_unvalidated", False)),
        "time_au": done * config["dt_au"], "initial_energy_per_bead": initial,
        "final_energy_per_bead": final, "max_energy_drift_per_bead": drift,
        "launch": launch, "wall_seconds": time.perf_counter() - started,
        "propagation": {"native_steps": 0, "python_steps": done},
        "electronic_populations": state.populations().tolist(),
    }
    write_metadata(result_path, record)
    return record


def _trajectory(
    dynamics,
    state,
    classifier,
    config,
    out,
    trajectory_id,
    parent_id,
    launch,
    physics_hash,
    resume,
    *,
    extend_unresolved=False,
    save_arrays,
    write_metadata,
):
    c, outcomes = config["collision"], config.get("outcomes", {})
    max_steps = _integer(c["max_steps"], "collision.max_steps")
    check = _integer(outcomes.get("check_interval", 10), "outcomes.check_interval")
    persistence = _integer(
        outcomes.get("persistence_steps", check * 3), "outcomes.persistence_steps", 0
    )
    output_config = config.get("output", {})
    stride = _integer(
        output_config.get("trajectory_stride", max(1, max_steps // 1000)),
        "trajectory_stride",
    )
    checkpoint_interval = _integer(
        output_config.get("checkpoint_steps", max(1, max_steps // 10)),
        "checkpoint_steps",
    )
    base = out / "trajectories" / f"trajectory-{trajectory_id:08d}"
    result_path, checkpoint = base.with_suffix(".json"), base.with_suffix(
        ".checkpoint.npz"
    )
    previous_wall_seconds = 0.0
    if resume and result_path.exists():
        result = json.loads(result_path.read_text())
        if (
            result.get("physics_hash") != physics_hash
            or result.get("trajectory_id") != trajectory_id
        ):
            raise ValueError("Completed trajectory does not match this restart")
        if not (extend_unresolved and result.get("status") == "unresolved"
                and result.get("steps", max_steps) < max_steps):
            return result
        if not checkpoint.exists():
            raise ValueError("Extending an unresolved trajectory requires its full checkpoint")
        previous_wall_seconds = float(result.get("wall_seconds", 0.0))
    if result_path.exists() and not resume:
        raise FileExistsError(
            f"Trajectory already exists: {result_path}; use resume explicitly"
        )
    initial = final = drift = None
    candidate, candidate_since, done = None, None, 0
    status, channel, reason = "unresolved", None, "maximum_propagation_time"
    from pyrpmd.block_propagation import ENERGY_PARTS, state_diagnostics

    diagnostic_spec = output_config.get("diagnostics")
    if diagnostic_spec is None and config.get("initial_ensemble"):
        diagnostic_spec = {"energy_parts": True, "ring_radii": True}
    diagnostic_spec = {k: v for k, v in (diagnostic_spec or {}).items() if v}
    trace_steps, trace_q, trace_energy = [], [], []
    diagnostic_traces = {key: [] for key in diagnostic_spec}
    previous_propagation = {"native_steps": 0, "python_steps": 0}
    if resume and checkpoint.exists():
        meta, arrays = _load_npz(
            checkpoint, expected_kind="PyRPMD.drpmd.trajectory-checkpoint.v1"
        )
        if (
            meta.get("physics_hash") != physics_hash
            or meta.get("trajectory_id") != trajectory_id
        ):
            raise ValueError(
                "Trajectory checkpoint has incompatible physical parameters"
            )
        state = State(arrays["q"], arrays["p"], meta["step"], meta["time"])
        done = meta["done"]
        initial, final, drift = meta["initial"], meta["final"], meta["drift"]
        candidate = tuple(meta["candidate"]) if meta["candidate"] is not None else None
        candidate_since = meta["candidate_since"]
        status = meta.get("status", "unresolved")
        channel, reason = meta.get("channel"), meta.get(
            "reason", "maximum_propagation_time"
        )
        trace_steps = list(arrays["trace_steps"])
        trace_q = list(arrays["trace_centroid"])
        trace_energy = list(arrays["trace_energy"])
        _integer(done, "checkpoint.done", 0)
        length = len(trace_steps)
        if (done > max_steps or state.step != done
                or not _checkpoint_clock_matches(state.time, done, config["dt_au"])
                or state.q.shape != (config["beads"], len(dynamics.masses), 3)
                or state.p.shape != state.q.shape
                or arrays["trace_centroid"].shape != (length, *state.q.shape[1:])
                or arrays["trace_energy"].shape != (length,)
                or arrays["trace_steps"].dtype.kind not in "iu"
                or (length and (trace_steps[0] != 0 or trace_steps[-1] > done))
                or np.any(np.diff(trace_steps) <= 0)
                or any(not np.isfinite(arrays[key]).all()
                       for key in ("q", "p", "trace_centroid", "trace_energy"))):
            raise ValueError("Trajectory checkpoint has inconsistent clocks, shapes or traces")
        if meta.get("diagnostics", {}) != diagnostic_spec:
            raise ValueError("Cannot change trajectory diagnostics when resuming an existing trace")
        for key in diagnostic_spec:
            name = "trace_" + key
            expected_shape = (length, 4 if key == "energy_parts" else state.q.shape[1])
            if (name not in arrays or arrays[name].shape != expected_shape
                    or not np.isfinite(arrays[name]).all()):
                raise ValueError(f"Checkpoint has missing or invalid {key} diagnostics")
            diagnostic_traces[key] = list(arrays[name])
        if "propagation" in meta:
            previous_propagation.update(meta["propagation"])
        elif done:
            # Older checkpoints did not record which backend ran these steps.
            # Preserve them without inventing a Python/native attribution.
            previous_propagation["untracked_steps"] = done
    else:
        try:
            if launch.get("initial_validation_error"):
                raise ValueError(launch["initial_validation_error"])
            initial = float(dynamics.energy(state)) / config["beads"]
            if not math.isfinite(initial):
                raise FloatingPointError("Nonfinite initial total ring energy")
            final, drift = initial, 0.0
            trace_steps, trace_q, trace_energy = (
                [0],
                [state.q.mean(axis=0).copy()],
                [initial],
            )
            for key, value in state_diagnostics(
                    dynamics, state, diagnostic_spec, energy_per_bead=initial).items():
                diagnostic_traces[key].append(value)
        except Exception as exc:
            # A valid parent may launch outside a PES domain after rotation or
            # rearrangement. Preserve its q/p and count this collision as failed.
            initial = final = drift = None
            trace_steps, trace_q, trace_energy = [], [], []
            diagnostic_traces = {key: [] for key in diagnostic_spec}
            status, channel = "failed", None
            stage = "initial_preparation" if launch.get("initial_validation_error") else "initial_energy"
            reason = f"{stage}: {type(exc).__name__}: {exc}"
    started = time.perf_counter()
    stats_before = copy.deepcopy(dynamics.propagation_stats)

    def trajectory_propagation():
        counts = dict(previous_propagation)
        for key in ("native_steps", "python_steps"):
            counts[key] += dynamics.propagation_stats[key] - stats_before[key]
        return counts

    def diagnostic_arrays(prefix=""):
        return {
            prefix + key: np.asarray(values, dtype=float).reshape(
                (-1, 4 if key == "energy_parts" else state.q.shape[1]))
            for key, values in diagnostic_traces.items()
        }

    def save_checkpoint():
        save_arrays(
            checkpoint,
            {
                "format": "PyRPMD.drpmd.trajectory-checkpoint.v1",
                "physics_hash": physics_hash,
                "trajectory_id": trajectory_id,
                "parent_id": parent_id,
                "step": state.step,
                "time": state.time,
                "done": done,
                "initial": initial,
                "final": final,
                "drift": drift,
                "candidate": candidate,
                "candidate_since": candidate_since,
                "status": status,
                "channel": channel,
                "reason": reason,
                "launch": launch,
                "propagation": trajectory_propagation(),
                "diagnostics": diagnostic_spec,
            },
            q=state.q,
            p=state.p,
            trace_steps=np.asarray(trace_steps, dtype=np.int64),
            trace_centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
            trace_energy=np.asarray(trace_energy),
            **diagnostic_arrays("trace_"),
        )

    # Optional native outcome observers preserve the same check clock and
    # persistence state while allowing a block to span several checks. A custom
    # Python classifier still receives the live full state at every boundary.
    from pyrpmd.block_propagation import native_outcome_supported

    native_events = status == "unresolved" and native_outcome_supported(dynamics)
    descriptor_factory = getattr(classifier, "native_outcome_descriptor", None)
    modern_native = getattr(dynamics.integrator._native, "DRPMD_OUTCOMES_API", 0) >= 1
    previous_available = modern_native or dynamics.loop_backend == "python" or not dynamics.native
    checkpoint_current = False

    while done < max_steps and status == "unresolved":
        descriptor = descriptor_factory() if native_events and callable(descriptor_factory) else None
        count = min(dynamics.block_steps, 4096, max_steps - done,
                    checkpoint_interval - done % checkpoint_interval)
        if classifier is not None and descriptor is None:
            count = min(count, check - done % check)
        python_classifier = classifier is not None and descriptor is None
        # A native classifier can yield a near-threshold check to the original
        # Python predicate. Keep only its pre-final state for exception rollback.
        record_previous = classifier is not None and previous_available
        record_states = python_classifier and not record_previous
        before = state.copy() if record_states else None
        start_done = done
        stats_before_block = copy.deepcopy(dynamics.propagation_stats)
        options = {"record_states": record_states, "raise_on_error": False}
        if diagnostic_spec:
            options["diagnostics"] = diagnostic_spec
        if record_previous:
            options["record_previous"] = True
        if native_events:
            options["reuse_forces"] = True
        if descriptor is not None:
            options["outcome"] = {
                "descriptor": descriptor,
                "check_interval": check,
                "persistence_steps": persistence,
                "start_step": done,
                "candidate": candidate,
                "candidate_since": candidate_since,
            }
        block = dynamics.propagate_block(state, count, **options)
        checkpoint_current = False
        valid = block.completed
        failure = block.error
        previous_candidate, previous_since = candidate, candidate_since
        energy = block.energies / config["beads"]
        pending_check = False
        if descriptor is not None:
            event = block.outcome
            candidate, candidate_since = event["candidate"], event["candidate_since"]
            pending_check = event.get("pending_check", False)
            # Earlier checks in this block remain accepted if the final Python
            # check raises. Roll back to this event state, not the block start.
            previous_candidate, previous_since = candidate, candidate_since
            if event["resolved"] is not None:
                status, channel = event["resolved"]
                reason = ("stable_separated_outgoing_products" if status == "product"
                          else "stable_separated_outgoing_reactants")
        # A Python classifier boundary is always the final step of its block.
        if not failure and (python_classifier or pending_check) and (done + valid) % check == 0:
            try:
                label = _normalize_label(classifier(state))
                block.centroids[valid - 1] = state.q.mean(axis=0)
                number = done + valid
                if label != candidate:
                    candidate, candidate_since = label, number if label else None
                if label is not None and number - candidate_since >= persistence:
                    status, channel = label
                    reason = ("stable_separated_outgoing_products" if status == "product"
                              else "stable_separated_outgoing_reactants")
            except Exception as exc:
                # A classifier failure rolls back only its step, just like the
                # original per-step workflow, not the whole completed prefix.
                valid -= 1
                if record_previous:
                    state = State(block.previous_q, block.previous_p,
                                  block.previous_step, block.previous_time)
                elif valid:
                    state = State(block.q[valid-1], block.p[valid-1],
                                  int(block.steps[valid-1]), float(block.times[valid-1]))
                else:
                    state = before
                candidate, candidate_since = previous_candidate, previous_since
                failure = f"{type(exc).__name__}: {exc}"
        # Only accepted physical steps contribute to trajectory accounting.
        # A classifier may reject the final step after the block committed it.
        if valid != block.completed:
            dynamics.propagation_stats = stats_before_block
            dynamics.propagation_stats[block.backend + "_steps"] += valid
            if block.fallback_reason:
                reasons = dynamics.propagation_stats["fallback_reasons"]
                reasons[block.fallback_reason] = reasons.get(block.fallback_reason, 0) + valid
        if valid:
            final = float(energy[valid-1])
            drift = max(drift, float(np.max(np.abs(energy[:valid] - initial))))
            numbers = start_done + np.arange(1, valid + 1)
            keep = (numbers % stride == 0) | (numbers == max_steps)
            if status != "unresolved" and not failure:
                keep[-1] = True
            selected = np.flatnonzero(keep)
            trace_steps.extend(numbers[selected].tolist())
            trace_q.extend(block.centroids[selected].copy())
            trace_energy.extend(energy[selected].tolist())
            for key in diagnostic_spec:
                diagnostic_traces[key].extend(block.diagnostics[key][selected].copy())
            done += valid
        if failure:
            status, channel, reason = "failed", None, failure
            break
        # I/O exceptions remain job failures, never chemical trajectory failures.
        if done % checkpoint_interval == 0 or status != "unresolved":
            save_checkpoint()
            checkpoint_current = True
    if classifier is None and status == "unresolved":
        reason = "no_validated_product_classifier"
        checkpoint_current = False
    if not checkpoint_current:
        save_checkpoint()
    save_arrays(
        base.with_suffix(".npz"),
        {
            "format": "PyRPMD.drpmd.trajectory.v1",
            "physics_hash": physics_hash,
            "trajectory_id": trajectory_id,
            "parent_id": parent_id,
            "units": {
                "q": "bohr",
                "p": "atomic",
                "time": "atomic",
                "energy": "hartree",
            },
            "hamiltonian_convention": "sum_beads_beta_over_P",
            "launch": launch,
            "diagnostics": diagnostic_spec,
            "energy_parts_columns": list(ENERGY_PARTS) if "energy_parts" in diagnostic_spec else None,
            "energy_parts_convention": "unbiased_H_P_over_P" if "energy_parts" in diagnostic_spec else None,
        },
        final_q=state.q,
        final_p=state.p,
        steps=np.asarray(trace_steps, dtype=np.int64),
        time_au=np.asarray(trace_steps) * config["dt_au"],
        centroid=np.asarray(trace_q).reshape((-1, *state.q.shape[1:])),
        energy_per_bead=np.asarray(trace_energy),
        **diagnostic_arrays(),
    )
    record = {
        "trajectory_id": trajectory_id,
        "parent_id": parent_id,
        "physics_hash": physics_hash,
        "status": status,
        "channel": channel,
        "reason": reason,
        "steps": done,
        "classification_unvalidated": bool(
            outcomes.get("classifier_unvalidated", False)
        ),
        "time_au": done * config["dt_au"],
        "initial_energy_per_bead": initial,
        "final_energy_per_bead": final,
        "max_energy_drift_per_bead": drift,
        "launch": launch,
        "wall_seconds": previous_wall_seconds + time.perf_counter() - started,
        "propagation": trajectory_propagation(),
    }
    if "initial_state" in launch:
        record["initial_state"] = launch["initial_state"]
    if "initial_diagnostics" in launch:
        record["initial_diagnostics"] = launch["initial_diagnostics"]
    write_metadata(result_path, record)
    return record
