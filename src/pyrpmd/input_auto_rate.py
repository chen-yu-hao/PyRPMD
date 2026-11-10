"""Starting settings for the opt-in AUS rate controller."""

from __future__ import annotations

from collections.abc import Mapping
import math

from .constants import KB_AU
from .input_schema import InputError, _integer, _unknown_fields
from .units import parse_quantity


def _default_steps(duration, dt):
    # Generated durations are starting heuristics, rounded to the stage clock.
    # Explicit user durations retain the ordinary exact-multiple validation.
    return max(1, int(round(parse_quantity(duration, "time") / dt)))


def _sampling_defaults(sampling, inherited, dt):
    for aliases, key, value in (
        (("equilibration", "equilibration_steps"), "equilibration_steps", _default_steps("100 fs", dt)),
        (("production", "production_steps"), "production_steps", _default_steps("1000 fs", dt)),
        (("interval", "save_interval", "interval_steps", "sample_interval"), "interval_steps", _default_steps("1 fs", dt)),
        (("replicas",), "replicas", 2),
    ):
        if not any(name in sampling or name in inherited for name in aliases):
            sampling[key] = value


def normalize_auto_rate_input(config, *, original=None):
    """Fill omitted starting settings after operation-named stage normalization.

    Explicit physical settings and sampling starts remain authoritative. The
    controller can subsequently increase statistical budgets and adapt the
    umbrella plan. Disabled and absent requests retain the ordinary workflow.
    """
    rate = config.get("rate", {})
    if not isinstance(rate, Mapping) or "auto" not in rate:
        return
    raw = rate["auto"]
    if isinstance(raw, bool):
        if not raw:
            return
        raw = {}
    if not isinstance(raw, Mapping):
        raise InputError("rate.auto must be boolean or a table")
    allowed = {"enabled", "target_relative_error", "max_rounds"}
    if set(raw) - allowed:
        raise _unknown_fields(set(raw) - allowed, allowed, "rate.auto")
    if not isinstance(raw.get("enabled", True), bool):
        raise InputError("rate.auto.enabled must be boolean")
    if not raw.get("enabled", True):
        rate["auto"] = False
        return
    if config.get("task") != "rate":
        raise InputError("rate.auto requires task='rate'")
    target = raw.get("target_relative_error", 0.1)
    if (isinstance(target, bool) or not isinstance(target, (int, float))
            or not math.isfinite(target) or not 0 < target < 1):
        raise InputError("rate.auto.target_relative_error must be finite and between zero and one")
    rate["auto"] = {
        "target_relative_error": float(target),
        "max_rounds": _integer(raw.get("max_rounds", 8), "rate.auto.max_rounds"),
    }
    if "temperature" not in config:
        raise InputError("rate.auto requires explicit temperature")
    temperature = parse_quantity(config["temperature"], "temperature")
    if not math.isfinite(temperature) or temperature <= 0:
        raise InputError("temperature must be positive and finite")
    if rate.get("pmf_source") not in (None, "umbrella"):
        raise InputError("rate.auto requires locally sampled AUS umbrellas, not pmf_source")
    stage = rate.get("pmf", config)
    if not isinstance(stage, Mapping):
        raise InputError("rate.pmf must be a table")
    if stage is not config and stage.get("method", "umbrella") != "umbrella":
        raise InputError("rate.auto requires rate.pmf.method='umbrella'")

    search = rate.setdefault("pmf_search_range", [0.0, 1.05])
    if (not isinstance(search, (list, tuple)) or len(search) != 2
            or any(isinstance(value, bool) for value in search)):
        raise InputError("rate.pmf_search_range must contain two dimensionless coordinates")
    try:
        lower, upper = (float(parse_quantity(value, "dimensionless")) for value in search)
    except (TypeError, ValueError) as error:
        raise InputError("rate.pmf_search_range must contain two dimensionless coordinates") from error
    if not math.isfinite(lower) or not math.isfinite(upper) or not 0 <= lower < upper:
        raise InputError("rate.pmf_search_range must satisfy 0 <= lower < upper")
    upper = max(1.0, upper)
    spacing = 0.05
    umbrella_owner = config if "umbrella" in config else stage
    umbrella = umbrella_owner.setdefault("umbrella", {})
    if not isinstance(umbrella, Mapping):
        raise InputError("umbrella must be a table")
    if not any(key in umbrella for key in ("centers", "centers_au", "grid", "regions", "windows", "start", "stop", "count")):
        umbrella.update(start=-spacing, stop=upper + spacing,
                        count=int(math.ceil(upper / spacing)) + 3)
    if not any(key in umbrella for key in ("stiffness", "stiffness_au", "kappa", "force_constant", "force_matrix")):
        umbrella["stiffness"] = KB_AU * temperature / spacing**2

    dynamics = config.setdefault("dynamics", {})
    if not isinstance(dynamics, Mapping):
        raise InputError("dynamics must be a table")
    pmf_dynamics = stage.get("dynamics", dynamics)
    if not isinstance(pmf_dynamics, Mapping):
        raise InputError("rate.pmf.dynamics must be a table")
    parent_dt = parse_quantity(dynamics.get("timestep", "0.1 fs"), "time")
    dt = parse_quantity(pmf_dynamics.get("timestep", parent_dt), "time")
    if not all(math.isfinite(value) and value > 0 for value in (parent_dt, dt)):
        raise InputError("timestep must be positive and finite")
    output = config.setdefault("output", {})
    if not isinstance(output, Mapping):
        raise InputError("output must be a table")
    if "checkpoint_steps" not in output and "checkpoint_interval" not in output:
        output["checkpoint_steps"] = _default_steps("10 fs", parent_dt)
    shared_sampling = config.setdefault("sampling", {})
    sampling = stage.setdefault("sampling", {})
    if not isinstance(shared_sampling, Mapping) or not isinstance(sampling, Mapping):
        raise InputError("sampling must be a table")
    inherited = shared_sampling if stage is not config else {}
    _sampling_defaults(sampling, inherited, dt)
    if stage is not config:
        _sampling_defaults(shared_sampling, {}, parent_dt)
    _integer(sampling.get("replicas", inherited.get("replicas", 2)), "automatic PMF sampling.replicas", 2)

    analysis = stage.setdefault("analysis", {})
    if not isinstance(analysis, Mapping) or not isinstance(config.get("analysis", {}), Mapping):
        raise InputError("analysis must be a table")
    inherited_analysis = config.get("analysis", {}) if stage is not config else {}
    aliases = ("bootstrap", "bootstrap_replicates", "bootstrap_samples")
    bootstrap = next((analysis[key] for key in aliases if key in analysis), None)
    if bootstrap is None:
        bootstrap = next((inherited_analysis[key] for key in aliases if key in inherited_analysis), None)
    if bootstrap is None:
        analysis["bootstrap"] = bootstrap = 200
    _integer(bootstrap, "automatic PMF analysis.bootstrap", 100)
    if "bins" not in analysis and "bins" not in inherited_analysis:
        analysis["bins"] = 120
    if "ranges" not in analysis and "ranges" not in inherited_analysis:
        analysis["ranges"] = [[-2 * spacing, upper + 2 * spacing]]

    adaptive_owner = config if "adaptive_umbrella" in config else stage
    adaptive = adaptive_owner.setdefault("adaptive_umbrella", {})
    if not isinstance(adaptive, Mapping):
        raise InputError("adaptive_umbrella must be a table")
    for key in ("enabled", "adapt_stiffness"):
        if key in adaptive and adaptive[key] is not True:
            raise InputError(f"rate.auto requires adaptive_umbrella.{key}=true")
        adaptive[key] = True
    adaptive.setdefault("pilot_steps", _default_steps("200 fs", dt))
    adaptive.setdefault("pilot_equilibration_steps", _default_steps("100 fs", dt))
    adaptive.setdefault("target_overlap", 0.03)
    adaptive.setdefault("min_effective_samples", 20.0)
    adaptive.setdefault("min_bin_effective_samples", 5.0)
    adaptive.setdefault("target_coverage", 1.0)
    if "coverage_bins" not in adaptive and "coverage_ranges" not in adaptive:
        adaptive.update(coverage_bins=[int(math.ceil(upper / spacing))],
                        coverage_ranges=[[0.0, upper]])

    if "recrossing" not in config and "recrossing" in rate:
        config["recrossing"] = rate.pop("recrossing")
    recrossing = config.setdefault("recrossing", {})
    if not isinstance(recrossing, Mapping):
        raise InputError("recrossing must be a table")
    if recrossing.get("dynamics", "rpmd") != "rpmd":
        raise InputError("rate.auto requires recrossing.dynamics='rpmd'")
    # Stage normalization supplies four configurations by default; distinguish
    # that default from an explicit count before installing the auto start.
    source = config if original is None else original
    source_rate = source.get("rate", {})
    source_recrossing = source.get("recrossing", source_rate.get("recrossing", {}))
    parent = source_recrossing.get("parent", {})
    if "parent_samples" not in source_recrossing and "configurations_per_chain" not in parent:
        legacy_parent_count = (source.get("sampling", {}).get("parents")
                               if source.get("schema_version", 1) == 1 else None)
        recrossing["parent_samples"] = 16 if legacy_parent_count is None else legacy_parent_count
    recrossing.setdefault("parent_replicas", 4)
    _integer(recrossing["parent_replicas"], "automatic recrossing.parent_replicas", 2)
    if not any(key in recrossing for key in ("child_pairs", "child_pairs_per_parent")):
        recrossing["child_pairs"] = 2
    for key, duration, default in (
        ("child_steps", "child_time", "50 fs"),
        ("equilibration_steps", "equilibration", "20 fs"),
        ("parent_interval_steps", "parent_interval", "5 fs"),
    ):
        if key not in recrossing and duration not in recrossing:
            recrossing[key] = _default_steps(default, parent_dt)
    recrossing.setdefault("bootstrap", 200)
    _integer(recrossing["bootstrap"], "automatic recrossing.bootstrap", 100)
    reaction = config.setdefault("reaction", {})
    if not isinstance(reaction, Mapping):
        raise InputError("reaction must be a table")
    if config.get("schema_version", 1) == 2 and not any(
        key in reaction for key in ("definition", "fragments", "reactant_fragments", "forming_bonds", "breaking_bonds")
    ):
        reaction["definition"] = "infer"
