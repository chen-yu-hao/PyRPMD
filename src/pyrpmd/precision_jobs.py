"""Materialize frozen precision jobs for the existing thermal-rate executor.

The bridge neither schedules nor launches trajectories. F can pass ``config``
and a freshly selected output directory to ``runner.run_resolved`` with the
returned ``run_kwargs``. ``umbrella_streams`` means replicas *per window*.
Only resolved, uniform, fixed local umbrellas and a fixed dividing surface are
supported; freeze an automatically selected pilot surface before production.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .input import rate_pmf_options, rate_pmf_stage
from .io import fingerprint, jsonable, referenced_files
from .precision import SamplingDesign, ScientificIdentity, evidence_from_diagnostics
from .precision_artifacts import read_rate_diagnostics
from .propagation import scientific_config
from .system import System


def _digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _steps(value, ratio, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be an integer step count")
    target = value * ratio
    rounded = round(target)
    if (rounded < (0 if zero else 1)
            or not math.isclose(target, rounded, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f"{name} cannot preserve its physical duration at the requested timestep")
    return rounded


def _resolved_layout(config, system):
    from .workflows.fes import resolve_window_plan
    from .workflows.rate_pmf import configure_rate_umbrella
    from .workflows.rate_options import _recrossing_dynamics

    if config.get("task") != "rate" or any(system.pbc) or system.fixed_atoms:
        raise ValueError("Precision jobs support thermal gas RPMD rate only")
    for name in ("initial_ensemble", "nonadiabatic", "electronic_friction", "path_integral"):
        if config.get(name):
            raise ValueError(f"Precision thermal RPMD jobs do not support {name}")
    _positive(config.get("dt_au"), "resolved dt_au")
    _positive(config.get("beta"), "resolved beta")
    if type(config.get("beads")) is not int or config["beads"] < 1:
        raise ValueError("Resolved beads must be a positive integer")
    if _recrossing_dynamics(config)[0] != "rpmd":
        raise ValueError("Precision jobs require unthermostatted RPMD children")
    options = rate_pmf_options(config)
    if options["source"] is not None or options["sampler"] != "umbrella":
        raise ValueError("Precision production requires a fresh local umbrella PMF")
    if options["xi_star"] is None:
        raise ValueError("Freeze the pilot dividing surface before materializing production")
    reaction = config.get("reaction", {})
    if not all(name in reaction for name in ("fragments", "forming_bonds", "breaking_bonds")):
        raise ValueError("Resolve the reaction and freeze its geometry before precision production")
    prepared = configure_rate_umbrella(config, system, reaction)
    stage = rate_pmf_stage(prepared, system)
    effective = stage if stage is not None else prepared
    for scope in (config, effective):
        for name in ("adaptive_umbrella", "replica_exchange", "metadynamics", "opes"):
            if scope.get(name):
                raise ValueError(f"Precision production does not support {name}")
    if "sampling" not in effective:
        raise ValueError("Precision jobs require resolved sampling settings")
    _positive(effective.get("dt_au"), "resolved PMF dt_au")
    for scope in (config, effective):
        dt = scope["dt_au"]
        sampling_dt = scope.get("sampling", {}).get("dt_au", dt)
        if not math.isclose(sampling_dt, dt, rel_tol=1e-14):
            raise ValueError("Precision jobs require each resolved sampling clock to match its stage")
    _, cv_specs, windows = resolve_window_plan(effective, system)
    if any(window.sampling for window in windows):
        raise ValueError("Precision v1 requires uniform sampling; per-window overrides are unsupported")
    return prepared, effective, options, cv_specs, windows


def _operator(config):
    dynamics = deepcopy(scientific_config(config).get("dynamics", {}))
    dynamics.pop("timestep", None)
    dynamics.setdefault("integrator", "cayley-bcb")
    return dynamics


def _sampling_policy(sampling, dt, *, include_save=True):
    result = {k: deepcopy(v) for k, v in sampling.items() if k not in {
        "replicas", "parents", "production", "equilibration", "production_steps",
        "equilibration_steps", "dt_au", "timestep", "sample_interval", "interval_steps",
        "interval", "save_interval", "andersen_interval",
    }}
    result.setdefault("thermostat", "andersen")
    if result["thermostat"] == "andersen":
        interval = _steps(sampling.get("andersen_interval", 10), 1., "andersen_interval")
        result["andersen_period_au"] = float(format(interval * dt, ".14g"))
        result.setdefault("andersen_schedule", "fixed")
    if include_save:
        interval = sampling.get("sample_interval", sampling.get("interval_steps", 1))
        if "interval_steps" in sampling and interval != sampling["interval_steps"]:
            raise ValueError("Resolved umbrella save intervals disagree")
        _steps(interval, 1., "sample_interval")
        result["sample_period_au"] = float(format(interval * dt, ".14g"))
    return result


def identity_from_config(config, system):
    """Content identity excluding seeds and sampling amounts.

    Geometry, masses, temperature, PES specification and explicit model-file
    hashes, reaction/normalization, operators and physical thermostat/save clocks
    are pinned. Bias identity includes actual centers, physical force matrices,
    estimator, histogram and fixed surface. Independent PMF-clock ratios are
    retained so a controlled timestep comparison scales both clocks together.
    """
    from .cv import geometry_convention_identity
    from .potentials import energy_convention_identity

    prepared, stage, options, cv_specs, windows = _resolved_layout(config, system)
    raw = scientific_config(prepared)
    omit = {"seed", "beads", "dt_au", "sampling", "recrossing", "rate", "analysis",
            "umbrella", "dynamics", "system", "output", "parallel", "resume", "_resume",
            "_warnings", "_input_path", "_base_directory", "_precision_provenance",
            "preset", "description"}
    model = {k: v for k, v in raw.items() if k not in omit}
    rec = prepared.get("recrossing", prepared.get("rate", {}).get("recrossing", {}))
    parent = rec.get("parent_thermostat", prepared.get("sampling", {}))
    model.update(
        system=system.to_dict(), operator=_operator(prepared),
        parent_thermostat=_sampling_policy(parent, prepared["dt_au"], include_save=False),
        pmf_operator=_operator(stage),
        pmf_sampling=_sampling_policy(stage["sampling"], stage["dt_au"]),
        pmf_clock_ratio=float(format(stage["dt_au"] / prepared["dt_au"], ".14g")),
        constraint_tolerance=rec.get("constraint_tolerance", 1e-10),
        plateau_tolerance=rec.get("plateau_tolerance", .1),
        cv_geometry_convention=geometry_convention_identity(prepared, system),
        potential_energy_convention=energy_convention_identity(prepared),
        input_files=referenced_files({k: prepared[k] for k in ("potential", "reaction", "preparation") if k in prepared}),
    )
    bias = {
        "coordinate": cv_specs,
        "windows": [{"center": w.center, "stiffness": w.stiffness} for w in windows],
        "pmf": {k: v for k, v in options.items()
                if k not in ("bootstrap", "analysis_seed", "source", "compare_ui")},
        "measure": "rate_flux",
    }
    return ScientificIdentity(_digest(model), int(prepared["beads"]),
                              float(prepared["dt_au"]), _digest(bias))


def _rescale_sampling(sampling, old_dt, new_dt, design=None):
    result = deepcopy(sampling)
    ratio = old_dt / new_dt
    interval = result.get("sample_interval", result.get("interval_steps", 1))
    if "interval_steps" in result and result["interval_steps"] != interval:
        raise ValueError("Resolved umbrella save intervals disagree")
    interval = _steps(interval, ratio, "sampling.sample_interval")
    result.update(sample_interval=interval, interval_steps=interval,
                  andersen_interval=_steps(result.get("andersen_interval", 10), ratio,
                                           "sampling.andersen_interval"))
    for name in ("production_steps", "equilibration_steps"):
        if name in result:
            result[name] = _steps(result[name], ratio, "sampling." + name,
                                  zero=name == "equilibration_steps")
    if design is not None:
        result.update(replicas=design.umbrella_streams,
                      production_steps=design.umbrella_steps,
                      equilibration_steps=design.umbrella_burnin_steps)
    for name in ("production", "equilibration", "interval", "save_interval"):
        result.pop(name, None)
    result.update(dt_au=new_dt, timestep=new_dt)
    return result


def _rescale_output(output, ratio):
    result = deepcopy(output)
    result.pop("directory", None)
    result.pop("checkpoint_interval", None)
    result["checkpoint_steps"] = _steps(result.get("checkpoint_steps", 1000), ratio,
                                        "output.checkpoint_steps")
    if "trajectory_stride" in result:
        result["trajectory_stride"] = _steps(result["trajectory_stride"], ratio,
                                              "output.trajectory_stride")
    return result


def materialize_rate_job(base_config, system, frozen, job_id):
    """Resolve one predeclared job; the caller must use a fresh run directory.

    No pilot checkpoint, adaptive bias or shared external PMF can enter this contract.
    Integer cadence conversion is exact (within floating roundoff); unsupported
    nonintegral clocks fail before a job can be submitted.
    """
    jobs = [j for j in frozen.jobs if j["job_id"] == job_id]
    if len(jobs) != 1:
        raise ValueError("Job is not part of the frozen production design")
    planned = jobs[0]
    baseline = next(j for j in frozen.jobs if j["group"] == "baseline")
    identity = identity_from_config(base_config, system)
    if identity.identity_id != baseline["scientific_identity"]:
        raise ValueError("Base configuration does not match the frozen scientific identity")
    config = deepcopy(base_config)
    if config.get("resume") or config.get("_resume"):
        raise ValueError("Materialize fresh production from an unresumed base configuration")
    target = ScientificIdentity(**{k: v for k, v in planned["identity"].items() if k != "identity_id"})
    design = SamplingDesign(**planned["sampling"])
    if planned.get("sampling_design_id", design.design_id) != design.design_id:
        raise ValueError("Frozen job sampling digest disagrees with its counts")
    _, base_stage, _, _, windows = _resolved_layout(base_config, system)
    old_dt, new_dt = identity.timestep_au, target.timestep_au
    ratio = old_dt / new_dt
    config.update(seed=planned["seed"], beads=target.beads, dt_au=new_dt, resume=False)
    config.pop("_resume", None)
    config.setdefault("dynamics", {})["timestep"] = new_dt
    scoped = "pmf" in config.get("rate", {})
    config["sampling"] = _rescale_sampling(config["sampling"], old_dt, new_dt,
                                           None if scoped else design)
    config["output"] = _rescale_output(config.get("output", {}), ratio)
    # The execution layer chooses an absolute fresh destination using this key.
    config["output"]["directory"] = job_id
    rec_parent = config if "recrossing" in config else config.setdefault("rate", {})
    rec = rec_parent.setdefault("recrossing", {})
    for name in ("equilibration", "parent_interval", "child_time", "child_pairs_per_parent"):
        rec.pop(name, None)
    rec.update(parent_replicas=design.parent_chains, parent_samples=design.parent_samples,
               equilibration_steps=design.parent_burnin_steps,
               parent_interval_steps=design.parent_interval_steps,
               child_pairs=design.child_pairs, child_steps=design.child_steps,
               plateau_fraction=design.plateau_fraction)
    if "parent_thermostat" in rec:
        bath = rec["parent_thermostat"]
        bath["andersen_interval"] = _steps(bath.get("andersen_interval", 10), ratio,
                                            "recrossing.parent_thermostat.andersen_interval")
    if scoped:
        stage = config["rate"]["pmf_stage"]["config"]
        stage_dt = base_stage["dt_au"] / ratio
        stage.update(seed=planned["seed"], beads=target.beads, dt_au=stage_dt, resume=False)
        stage.pop("_resume", None)
        stage.setdefault("dynamics", {})["timestep"] = stage_dt
        stage["sampling"] = _rescale_sampling(stage["sampling"], base_stage["dt_au"], stage_dt, design)
        stage["output"] = _rescale_output(stage.get("output", {}), ratio)
        request = config["rate"]["pmf"]
        request.setdefault("dynamics", {})["timestep"] = stage_dt
        request["sampling"] = {k: deepcopy(v) for k, v in stage["sampling"].items() if k != "dt_au"}
    actual_identity = identity_from_config(config, system)
    if actual_identity.identity_id != target.identity_id:
        raise ValueError("Materialized job changes an unplanned scientific setting")
    provenance = {
        "format": "PyRPMD.precision.production.v1", "phase": "production", "job_id": job_id,
        "epoch_id": frozen.epoch_id, "design_id": frozen.design_id,
        "sampling_design_id": design.design_id, "seed": planned["seed"],
        "scientific_identity": target.identity_id, "frozen_before_sampling": True,
        "independent_from_pilot": True, "independent_initialization": True,
        "fresh_output_required": True,
    }
    config["_precision_provenance"] = deepcopy(provenance)
    expected = fingerprint(scientific_config(config), system)
    return {"job_id": job_id, "config": config, "system": system.to_dict(),
            "scientific_identity": target.identity_id, "identity": target.to_dict(),
            "provenance": provenance, "expected_physics_fingerprint": expected,
            "expected_input_files": referenced_files(config),
            "sampling": asdict(design), "sampling_design_id": design.design_id,
            "umbrella_window_count": len(windows),
            "run_kwargs": {"resume": False}, "output_key": job_id}


def record_from_rate_job(directory, job, frozen):
    """Authenticate one completed replica for ``evidence_from_replicas``.

    A missing/failed positive rate factor raises an explicit error; callers must
    retain that failed job and stop or repair the production epoch, never omit it
    from a look. Diagnostic failures with defined factors remain in the record.
    """
    planned = next((j for j in frozen.jobs if j["job_id"] == job.get("job_id")), None)
    if planned is None or planned["scientific_identity"] != job.get("scientific_identity"):
        raise ValueError("Execution job is not part of the frozen production")
    planned_design = SamplingDesign(**planned["sampling"])
    if job.get("sampling_design_id") != planned_design.design_id:
        raise ValueError("Execution sampling design differs from the frozen job")
    snapshot = read_rate_diagnostics(directory)
    if snapshot["published_generation"] != "original":
        raise ValueError("Frozen production replicas cannot include adaptive extension generations")
    metadata = snapshot["metadata"]
    resolved, manifest = metadata["resolved_input"], metadata["manifest"]
    expected_provenance = job["provenance"]
    if (resolved.get("_precision_provenance") != expected_provenance
            or expected_provenance.get("job_id") != planned["job_id"]
            or expected_provenance.get("scientific_identity") != planned["scientific_identity"]
            or expected_provenance.get("sampling_design_id") != planned_design.design_id
            or expected_provenance.get("seed") != planned["seed"]
            or any(expected_provenance.get(name) is not True for name in (
                "frozen_before_sampling", "independent_from_pilot", "independent_initialization",
                "fresh_output_required"))
            or expected_provenance.get("epoch_id") != frozen.epoch_id
            or expected_provenance.get("design_id") != frozen.design_id
            or resolved.get("seed") != planned["seed"] or resolved.get("resume") is not False):
        raise ValueError("Saved rate does not match the frozen fresh-production provenance")
    root = Path(directory).resolve()
    if not (root / "input/system.json").is_file():
        root = (root / "run").resolve()
    system_path = root / "input/system.json"
    if not system_path.resolve().is_relative_to(root):
        raise ValueError("Saved production system must remain inside its run directory")
    stored_system = json.loads(system_path.read_text(encoding="utf-8"))
    if _digest(stored_system) != _digest(job["system"]):
        raise ValueError("Saved production system differs from the materialized job")
    system = System.from_dict(stored_system)
    digest = fingerprint(scientific_config(resolved), system)
    if (digest != manifest.get("physics_fingerprint")
            or digest != job["expected_physics_fingerprint"]
            or manifest.get("files", {}) != job["expected_input_files"]):
        raise ValueError("Saved production input/model fingerprint differs from the materialized job")
    if snapshot["source_family_id"] is None:
        raise ValueError("Complete production source identity is unavailable")
    if set(snapshot["source_ids"]).intersection(frozen.pilot_source_ids):
        raise ValueError("Pilot samples cannot enter independent production")
    pmf = metadata["pmf"]
    if pmf.get("method") == "source" or pmf.get("source_sha256") is not None:
        raise ValueError("Independent production cannot reuse an estimated PMF")
    _, stage, options, _, windows = _resolved_layout(resolved, system)
    if pmf.get("method") != options["method"] or pmf.get("xi_star") != options["xi_star"]:
        raise ValueError("Saved PMF estimator or dividing surface differs from the frozen job")
    design = planned_design
    rec = snapshot["recrossing"]
    parents = design.parent_chains * design.parent_samples
    if (rec["parent_configurations"] != parents or rec["child_pairs"] != parents * design.child_pairs
            or rec["child_steps"] != design.child_steps):
        raise ValueError("Saved recrossing output does not complete the frozen sampling design")
    umbrella = metadata["umbrella"]
    streams = len(windows) * design.umbrella_streams
    stride = stage["sampling"].get("sample_interval", stage["sampling"].get("interval_steps", 1))
    if umbrella.get("streams") != streams or umbrella.get("samples") != streams * (design.umbrella_steps // stride):
        raise ValueError("Saved umbrella output does not complete the frozen sampling design")
    qtst = _positive(pmf.get("k_qtst_cm3_s"), "saved k_qtst_cm3_s")
    kappa = _positive(snapshot["recrossing"].get("kappa"), "saved kappa")
    observed = snapshot.get("rate_estimate_cm3_s")
    if observed is None or not math.isclose(qtst * kappa, observed, rel_tol=1e-10, abs_tol=0.):
        raise ValueError("Saved rate estimate disagrees with its PMF and recrossing factors")
    identity = ScientificIdentity(**{k: v for k, v in job["identity"].items() if k != "identity_id"})
    return {"job_id": job["job_id"], "source_id": snapshot["source_ids"][0],
            "source_family_id": snapshot["source_family_id"], "epoch_id": frozen.epoch_id,
            "design_id": frozen.design_id, "seed": planned["seed"],
            "sampling_design_id": job["sampling_design_id"],
            "independent_initialization": True, "log_qtst": math.log(qtst),
            "log_kappa": math.log(kappa), "diagnostics": evidence_from_diagnostics(snapshot, identity)}
