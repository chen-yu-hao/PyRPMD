"""Operation-named sampling inputs, normalized to the existing stage engines.

Legacy flat inputs are left untouched.  New stage tables have independent
thermostats and exact time/step expressions; normalization never relies on a
legacy alias winning silently over another declaration.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from .input_schema import InputError, _integer, _unknown_fields
from .units import parse_quantity, time_steps


def _table(value, path):
    if not isinstance(value, Mapping):
        raise InputError(f"{path} must be a table")
    return dict(value)


def _fields(table, allowed, path):
    unknown = set(table) - set(allowed)
    if unknown:
        raise _unknown_fields(unknown, allowed, path)


def duration_steps(value, dt_au, *, path, allow_zero=False):
    """Read one unambiguous, operation-local time or exact step count."""
    minimum = 0 if allow_zero else 1
    if isinstance(value, Mapping):
        if set(value) != {"steps"}:
            raise InputError(f"{path} must be a time string or a table containing only steps")
        return _integer(value["steps"], path + ".steps", minimum)
    if not isinstance(value, str):
        raise InputError(f"{path} must be a time string or {{steps = N}}")
    try:
        return time_steps(value, dt_au, allow_zero=allow_zero)
    except (TypeError, ValueError) as error:
        raise InputError(f"{path}; dynamics.timestep = {dt_au:g} autime: {error}") from error


def _thermostat_input(value, dt_au, path, *, parent=False):
    raw = _table(value, path)
    _fields(raw, {"type", "scheme", "schedule", "refresh_every",
                  "mean_refresh_interval", "gamma_centroid", "params"}, path)
    mode = raw.get("type", "andersen")
    if not isinstance(mode, str):
        raise InputError(f"{path}.type must be a thermostat name")
    if parent and mode != "andersen":
        raise InputError(f"{path}.type currently requires 'andersen' for constrained parents")
    result = {"thermostat": mode}
    if mode == "andersen":
        if raw.get("scheme", "global_refresh") != "global_refresh":
            raise InputError(f"{path}.scheme must be 'global_refresh'")
        schedule = raw.get("schedule", "fixed")
        if schedule not in ("fixed", "geometric"):
            raise InputError(f"{path}.schedule must be 'fixed' or 'geometric'")
        key = "refresh_every" if schedule == "fixed" else "mean_refresh_interval"
        other = "mean_refresh_interval" if schedule == "fixed" else "refresh_every"
        if other in raw:
            raise InputError(f"{path}.{other} conflicts with schedule='{schedule}'; use {key}")
        if any(name in raw for name in ("params", "gamma_centroid")):
            raise InputError(f"{path}: Andersen uses {key}, not params/gamma_centroid")
        result.update(
            andersen_interval=duration_steps(raw.get(key, {"steps": 10}), dt_au,
                                             path=path + "." + key),
            andersen_schedule=schedule,
        )
    else:
        if any(key in raw for key in ("scheme", "schedule", "refresh_every", "mean_refresh_interval")):
            raise InputError(f"{path}: momentum-refresh settings require type='andersen'")
        if "gamma_centroid" in raw:
            result["gamma_centroid"] = raw["gamma_centroid"]
        if "params" in raw:
            result["thermostat_params"] = _table(raw["params"], path + ".params")
    return result


def normalize_sampling_stage(spec, dt_au, *, path="umbrella.sampling"):
    """Return canonical *partial* sampling settings for a stage or window.

    Callers applying overrides must remove inherited thermostat-specific settings
    when an explicit thermostat table is present, then validate the merged sampler.
    The two legacy sample-interval keys are always updated together.
    """
    raw = _table(spec, path)
    _fields(raw, {"equilibration", "production", "sample_every",
                  "chains_per_window", "thermostat", "integrator"}, path)
    result = {}
    for source, target in (("equilibration", "equilibration_steps"),
                           ("production", "production_steps"),
                           ("sample_every", "interval_steps")):
        if source in raw:
            result[target] = duration_steps(raw[source], dt_au, path=path + "." + source,
                                            allow_zero=source == "equilibration")
    if "interval_steps" in result:
        result["sample_interval"] = result["interval_steps"]
    if "chains_per_window" in raw:
        result["replicas"] = _integer(raw["chains_per_window"], path + ".chains_per_window")
    if "thermostat" in raw:
        result.update(_thermostat_input(raw["thermostat"], dt_au, path + ".thermostat"))
    if "integrator" in raw:
        if not isinstance(raw["integrator"], str):
            raise InputError(f"{path}.integrator must be a name")
        result["integrator"] = raw["integrator"]
    return result


def _put(table, key, value, path, *, aliases=()):
    conflict = next((name for name in (key, *aliases) if name in table), None)
    if conflict is not None:
        raise InputError(f"{path} conflicts with the existing {conflict} setting")
    table[key] = value


def _umbrella_analysis(d, raw):
    path = "umbrella.analysis"
    raw = _table(raw, path)
    _fields(raw, {"estimator", "integration_points", "bins", "ranges", "bootstrap_replicates",
                  "bootstrap_workers", "seed", "min_count", "overlap_threshold", "max_grid_points"}, path)
    analysis = _table(d.get("analysis", {}), "analysis")
    rate = _table(d.get("rate", {}), "rate")
    if "estimator" in raw:
        if d.get("task") == "rate":
            _put(rate, "pmf_method", raw["estimator"], path + ".estimator")
        else:
            _put(analysis, "method", raw["estimator"], path + ".estimator")
    if "integration_points" in raw:
        if d.get("task") != "rate":
            raise InputError("umbrella.analysis.integration_points requires task='rate'")
        _put(rate, "pmf_points", raw["integration_points"], path + ".integration_points")
    for key, value in raw.items():
        if key in ("estimator", "integration_points"):
            continue
        if key == "bootstrap_replicates":
            _put(analysis, "bootstrap", value, path + "." + key,
                 aliases=("bootstrap_replicates", "bootstrap_samples"))
        else:
            _put(analysis, key, value, path + "." + key)
    d["analysis"] = analysis
    if rate:
        d["rate"] = rate


def _parent_input(raw, dt_au, output):
    path = "recrossing.parent"
    raw = _table(raw, path)
    _fields(raw, {"chains", "equilibration", "draw_every", "configurations_per_chain",
                  "thermostat", "constraint_tolerance"}, path)
    for source, target in (("chains", "parent_replicas"),
                           ("configurations_per_chain", "parent_samples")):
        if source in raw:
            _put(output, target, _integer(raw[source], path + "." + source), path + "." + source)
    for source, target, old_time in (("equilibration", "equilibration_steps", "equilibration"),
                                     ("draw_every", "parent_interval_steps", "parent_interval")):
        if source in raw:
            value = duration_steps(raw[source], dt_au, path=path + "." + source,
                                   allow_zero=source == "equilibration")
            _put(output, target, value, path + "." + source, aliases=(old_time,))
    if "constraint_tolerance" in raw:
        _put(output, "constraint_tolerance", raw["constraint_tolerance"], path + ".constraint_tolerance")
    if "thermostat" in raw:
        _put(output, "parent_thermostat", _thermostat_input(raw["thermostat"], dt_au,
                                                          path + ".thermostat", parent=True),
             path + ".thermostat")


def _children_input(raw, dt_au, output):
    path = "recrossing.children"
    raw = _table(raw, path)
    if "thermostat" in raw or "bias" in raw or "constraint" in raw:
        raise InputError("recrossing.children cannot inherit sampling thermostats, umbrella bias or dividing-surface constraints; select method='trpmd' explicitly for internal-mode friction")
    _fields(raw, {"method", "pairs_per_configuration", "duration", "friction_scale", "batch_pairs"}, path)
    for source, target in (("method", "dynamics"), ("friction_scale", "friction_scale")):
        if source in raw:
            _put(output, target, raw[source], path + "." + source)
    for source, target in (("pairs_per_configuration", "child_pairs"), ("batch_pairs", "child_batch_pairs")):
        if source in raw:
            aliases = ("child_pairs_per_parent",) if source == "pairs_per_configuration" else ()
            _put(output, target, _integer(raw[source], path + "." + source), path + "." + source,
                 aliases=aliases)
    if "duration" in raw:
        _put(output, "child_steps", duration_steps(raw["duration"], dt_au, path=path + ".duration"),
             path + ".duration", aliases=("child_time",))


def normalize_stage_input(d, *, independent_rate_parent=False):
    """Consume operation-named stages before generic schema validation.

This must follow FES alias normalization and precede ``resolve_dynamics`` so
production completeness checks see the normalized umbrella sampling section.
"""
    task = d.get("task")
    umbrella = d.get("umbrella", {})
    if not isinstance(umbrella, Mapping):
        return  # The generic schema reports the malformed table.
    rate = d.get("rate", {})
    if not isinstance(rate, Mapping):
        return
    nested = "recrossing" not in d and isinstance(rate.get("recrossing"), Mapping)
    recrossing = rate.get("recrossing", {}) if nested else d.get("recrossing", {})
    if not isinstance(recrossing, Mapping):
        return
    umbrella_new = any(key in umbrella for key in ("sampling", "analysis"))
    recrossing_new = any(key in recrossing for key in ("parent", "children", "analysis"))
    reaction = d.get("reaction", {})
    reaction_new = "preparation" in d or (
        isinstance(reaction, Mapping)
        and (any(key in reaction for key in ("definition", "reactant_fragments", "product_fragments"))
             or (isinstance(reaction.get("preparation"), Mapping)
                 and any(key in reaction["preparation"] for key in ("ts_action", "check_channel"))))
    )
    independent = task == "rate" and (independent_rate_parent or d.get("schema_version", 1) == 2 or umbrella_new
                                      or recrossing_new or reaction_new)
    if not umbrella_new and not recrossing_new and not independent:
        return
    if umbrella_new and task not in ("rate", "fes"):
        raise InputError("Operation-named umbrella stages currently require task='rate' or task='fes'")
    if umbrella_new and task == "fes" and d.get("fes", {}).get("method", "umbrella") != "umbrella":
        raise InputError("umbrella.sampling/analysis require fes.method='umbrella'")
    if recrossing_new and task != "rate":
        raise InputError("Separate recrossing parent/children stages currently require task='rate'")
    dynamics = _table(d.get("dynamics", {}), "dynamics")
    try:
        dt_au = float(parse_quantity(dynamics.get("timestep", "0.1 fs"), "time"))
    except (TypeError, ValueError) as error:
        raise InputError(f"dynamics.timestep: {error}") from error
    if not math.isfinite(dt_au) or dt_au <= 0:
        raise InputError("dynamics.timestep must be positive and finite")
    if umbrella_new:
        umbrella = deepcopy(dict(umbrella))
        if "sampling" in umbrella:
            if "sampling" in d:
                raise InputError("Use either umbrella.sampling or sampling, not both")
            d["sampling"] = normalize_sampling_stage(umbrella.pop("sampling"), dt_au)
        if "analysis" in umbrella:
            _umbrella_analysis(d, umbrella.pop("analysis"))
        d["umbrella"] = umbrella
    if recrossing_new or independent:
        recrossing = deepcopy(dict(recrossing))
        if "parent" in recrossing:
            _parent_input(recrossing.pop("parent"), dt_au, recrossing)
        if "children" in recrossing:
            _children_input(recrossing.pop("children"), dt_au, recrossing)
        if "analysis" in recrossing:
            analysis = _table(recrossing.pop("analysis"), "recrossing.analysis")
            _fields(analysis, {"bootstrap_replicates", "plateau_fraction", "plateau_tolerance"}, "recrossing.analysis")
            for key, value in analysis.items():
                _put(recrossing, "bootstrap" if key == "bootstrap_replicates" else key,
                     value, "recrossing.analysis." + key)
        if independent:
            # Defaults belong to the constrained ensemble, never to the PMF
            # thermostat or its number of workers/parent chains.
            recrossing.setdefault("parent_thermostat", _thermostat_input({}, dt_au,
                                  "recrossing.parent.thermostat", parent=True))
            recrossing.setdefault("parent_samples", 4)
        d["recrossing"] = recrossing
        if nested:
            d["rate"] = {key: value for key, value in d["rate"].items() if key != "recrossing"}
