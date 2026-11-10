"""Warm start: initialize a *new* run from a configuration saved by an old run.

A warm start is not a resume. The new run has its own input, its own
scientific fingerprint and its own output directory; no sample, estimator
or RNG state of the parent is reused. Only these are imported:

* one ring-polymer configuration q from a parent checkpoint. A changed bead
  count is handled by ring-polymer normal-mode (Fourier) interpolation, which
  preserves the centroid exactly and keeps the lowest internal modes;
* optionally, for metadynamics, the parent's deposited hills as a static
  *initial bias* (``import_bias=True``).

Momenta are always drawn again from the Maxwell distribution of the new
temperature and bead count, with the new input's seed. Every warm start must
run the new input's equilibration stage (``sampling.equilibration_steps > 0``).

The imported data are stored in ``config["warm_start"]`` and therefore enter
the new run's fingerprint and ``input/resolved.json``, so ``resume`` of the new
run restarts from exactly the same imported state. The parent run directory is
only read once, when the new run is created. The manifest records the parent's
physics fingerprint under ``initialized_from``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .input import InputError

SCHEMA = "pyrpmd.warm_start"
SCHEMA_VERSION = 1

# Workflows whose fresh-start path reads the imported ring configuration.
_FES_METHODS = ("umbrella", "metadynamics", "opes", "abf", "eabf", "steered", "ves")
_UNSUPPORTED_REASON = {
    "rate": "rate stages construct constrained transition-state and umbrella states; seed them from the rate input instead",
    "drpmd": "dRPMD parents are prepared from reactant sampling, not a single configuration",
    "surface_rate": "surface rates prepare their own constrained parents",
    "site_rate": "site rates prepare their own constrained parents",
    "network": "reaction networks run several dependent path workflows",
    "nonadiabatic": "nonadiabatic workflows sample nuclear and electronic initial conditions together",
    "pcet": "PCET has no molecular ring-polymer configuration",
    "spectroscopy": "spectroscopy prepares separate thermalization and correlation stages",
    "path_sampling": "path sampling requires a complete initial path, not one configuration",
}


def _fail(message):
    raise InputError("Warm start: " + message)


# ---------------------------------------------------------------------------
# Ring-polymer bead interpolation


def interpolate_beads(q, beads):
    """Resample ring coordinates ``q[P, N, 3]`` to ``beads`` by Fourier modes.

    The ring is cyclic in imaginary time, so its bead coordinates are expanded
    in the discrete Fourier (free ring-polymer normal-mode) basis. Increasing P
    zero-pads the high-frequency modes; decreasing P keeps the lowest modes.
    The centroid (zero mode) is preserved exactly, and resampling to the same
    P returns ``q`` unchanged. An odd-P Nyquist-free spectrum makes
    up-then-down resampling exact.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 3 or q.shape[-1] != 3:
        raise ValueError("ring configuration must have shape [beads, atoms, 3]")
    old = q.shape[0]
    if isinstance(beads, (bool, np.bool_)) or not isinstance(beads, (int, np.integer)) or beads < 1:
        raise ValueError("beads must be a positive integer")
    beads = int(beads)
    if old == beads:
        return q.copy()
    if old == 1 or beads == 1:
        centroid = q.mean(axis=0)
        return np.repeat(centroid[None], beads, axis=0)
    modes = np.fft.rfft(q, axis=0)
    target = np.zeros((beads // 2 + 1,) + q.shape[1:], dtype=complex)
    keep = min(old // 2, beads // 2)
    target[:keep + 1] = modes[:keep + 1]
    if old % 2 == 0 and keep == old // 2 and beads > old:
        # The old Nyquist cosine becomes an ordinary +/-k pair on a finer ring.
        target[keep] *= 0.5
    if beads % 2 == 0 and keep == beads // 2 and old > beads:
        # On the coarser ring only the cosine part of mode k is representable.
        target[keep] = 2.0 * modes[keep].real
    result = np.fft.irfft(target * (beads / old), n=beads, axis=0)
    return np.ascontiguousarray(result)


# ---------------------------------------------------------------------------
# Parent discovery


def _run_directory(source):
    path = Path(source).expanduser().resolve()
    if not path.exists():
        _fail(f"{path} does not exist")
    if path.is_dir():
        if path.name == "checkpoints" and (path.parent / "manifest.json").is_file():
            return path.parent, None
        if (path / "manifest.json").is_file():
            return path, None
    else:
        # A checkpoint file anywhere below <run>/checkpoints.
        for candidate in path.parents:
            if (candidate / "manifest.json").is_file():
                if (candidate / "checkpoints") in path.parents:
                    return candidate, path
                break
    _fail(f"no run manifest.json found for {path}; pass a PyRPMD run directory or a checkpoint "
          "file inside its checkpoints directory")


def _candidates(run, chosen, beads, natoms):
    """Return {relative checkpoint path: q array [..., P, N, 3]}."""
    from .checkpoint import CheckpointError, load_checkpoint

    files = [chosen] if chosen is not None else sorted((run / "checkpoints").rglob("*.npz"))
    found, rejected = {}, []
    for file in files:
        relative = file.resolve().relative_to(run).as_posix()
        try:
            with load_checkpoint(file, require_integrity=True) as data:
                if "q" not in data:
                    raise CheckpointError("no ring coordinates q")
                q = np.array(data["q"], dtype=float)
                hills = str(data["hills"]) if "hills" in data else None
        except (CheckpointError, OSError, ValueError, KeyError) as error:
            if chosen is not None:
                _fail(f"cannot read ring coordinates from {relative}: {error}")
            rejected.append(relative)
            continue
        if q.ndim < 3 or q.shape[-3:] != (beads, natoms, 3):
            if chosen is not None:
                _fail(f"{relative} has q with shape {list(q.shape)}, not [..., {beads}, {natoms}, 3]")
            continue
        found[relative] = (q, hills)
    if not found:
        _fail(f"no integrity-verified checkpoint with ring coordinates in {run / 'checkpoints'}")
    return found


def _select(found, index):
    choices = []
    for name, (q, _) in found.items():
        count = int(np.prod(q.shape[:-3], dtype=int))
        choices.append((name, count))
    if len(found) > 1:
        listing = ", ".join(f"{name} ({count} configuration{'s' if count != 1 else ''})"
                            for name, count in choices)
        _fail("the parent run has several checkpoints; pass one checkpoint file instead of the run "
              "directory. Candidates: " + listing)
    name, count = choices[0]
    q, hills = found[name]
    flat = q.reshape((-1,) + q.shape[-3:])
    if index is None:
        if count != 1:
            _fail(f"{name} holds {count} configurations; select one with init_index (0..{count - 1})")
        index = 0
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < count:
        _fail(f"init_index must be an integer in 0..{count - 1} for {name}")
    return name, int(index), flat[int(index)].copy(), hills, count


def _has_barostat(value):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "barostat" in str(key).lower() and item not in (None, False, "none", {}):
                return True
            if _has_barostat(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_barostat(item) for item in value)
    return False


# ---------------------------------------------------------------------------
# Applicability


def _method(config):
    from .workflows.common import fes_method

    task = config.get("task")
    if task in ("fes", "metadynamics", "opes"):
        return fes_method(config)
    return None


def check_supported(config):
    """Reject workflows whose initial state is not a single ring configuration."""
    task = config.get("task")
    if task in _UNSUPPORTED_REASON:
        _fail(f"task='{task}' is not supported: {_UNSUPPORTED_REASON[task]}")
    if "tempering" in config:
        _fail("tempering ensembles are not supported")
    adaptive = config.get("adaptive_umbrella")
    if adaptive and (not isinstance(adaptive, Mapping) or adaptive.get("enabled", True)):
        _fail("adaptive umbrella pilots are not supported")
    method = _method(config)
    if task in ("fes", "metadynamics", "opes"):
        if method not in _FES_METHODS:
            _fail(f"fes.method='{method}' is not supported")
    elif task == "pimd" and "path_integral" in config:
        pass
    elif task in ("pimd", "diffusion"):
        engine = config.get("sampling", {}).get("engine", "legacy")
        if engine != "legacy":
            _fail(f"sampling.engine='{engine}' prepares its own initial state; use the legacy engine")
    elif task == "plumed":
        pass
    else:
        _fail(f"task='{task}' is not supported")
    sampling = config.get("sampling", {})
    neq = sampling.get("equilibration_steps")
    if neq is None and task in ("pimd", "diffusion") and "path_integral" not in config:
        neq = 100  # the materials default
    if isinstance(neq, (bool, np.bool_)) or not isinstance(neq, (int, np.integer)) or neq < 1:
        _fail("the new input must request equilibration (sampling.equilibration_steps >= 1); "
              "a warm-start configuration is re-equilibrated at the new settings before production")


# ---------------------------------------------------------------------------
# Initial metadynamics bias


def _metadynamics_parts(config, system, label):
    from .workflows.metadynamics import resolve_metadynamics

    if _method(config) != "metadynamics":
        _fail(f"importing a bias requires metadynamics in the {label} run")
    table = config.get("metadynamics", {})
    if table.get("shared_walkers") or table.get("multiple_walkers"):
        _fail(f"importing a bias does not support shared or multiple walkers ({label} run)")
    if table.get("grid") is not None:
        _fail(f"importing a bias does not support grid metadynamics ({label} run)")
    try:
        cv, selected, options = resolve_metadynamics(copy.deepcopy(config), system)
    except ValueError as error:
        _fail(f"{label} metadynamics input is invalid: {error}")
    return cv, selected, options


def _canonical(value):
    from .io import jsonable

    return json.dumps(jsonable(value), sort_keys=True, allow_nan=False)


def _shape_factor(options):
    """Asymptotic V = -f F: f = 1 (standard) or (gamma-1)/gamma (well-tempered)."""
    if options["mode"] == "standard":
        return 1.0
    gamma = float(options["bias_factor"])
    return (gamma - 1.0) / gamma


def import_bias(parent_config, parent_system, config, system, hills_json):
    """Return the scaled initial-bias record, or raise if it is not transferable."""
    _, parent_selected, parent = _metadynamics_parts(parent_config, parent_system, "parent")
    _, selected, current = _metadynamics_parts(config, system, "new")
    if _canonical(parent_selected) != _canonical(selected):
        _fail("importing a bias requires identical metadynamics CVs (definitions, atoms, estimators and units)")
    if not np.allclose(parent["sigma"], current["sigma"], rtol=1e-12, atol=0.0):
        _fail("importing a bias requires identical Gaussian widths (metadynamics.sigma)")
    if hills_json is None:
        _fail("the selected parent checkpoint has no metadynamics hill history")
    hills = json.loads(hills_json)
    if not isinstance(hills, list) or not hills:
        _fail("the selected parent checkpoint has no deposited hills to import")
    same_temperature = np.isclose(parent_config["beta"], config["beta"], rtol=1e-12, atol=0.0)
    well_tempered = parent["mode"] == "well_tempered" and current["mode"] == "well_tempered"
    if not same_temperature and not well_tempered:
        _fail("the temperature changed; an imported bias can only be rescaled when both runs are "
              "well-tempered metadynamics")
    # Both biases estimate the same F(s): V_old = -f_old F, V_new = -f_new F.
    scale = _shape_factor(current) / _shape_factor(parent)
    records = []
    for hill in hills:
        center = [float(x) for x in hill["center"]]
        height = float(hill["height"]) * scale
        if len(center) != len(current["sigma"]) or not np.isfinite(center).all() or not np.isfinite(height) \
                or height < 0:
            _fail("the parent hill history is invalid")
        records.append({"center": center, "height": height})
    return {
        "kind": "metadynamics_hills",
        "role": "initial_bias",
        "hills": records,
        "scale": scale,
        "rule": ("well_tempered: V_new = [(gamma_new-1)/gamma_new] / [(gamma_old-1)/gamma_old] * V_old, "
                 "assuming F(s) is unchanged by the new settings; standard metadynamics uses factor 1"),
        "parent": {"mode": parent["mode"], "bias_factor": parent["bias_factor"],
                   "beta": float(parent_config["beta"]), "hills": len(records)},
        "temperature_changed": not bool(same_temperature),
    }


# ---------------------------------------------------------------------------
# Public entry points


def prepare(config, system, source, *, index=None, import_bias_history=False):
    """Return a copy of resolved ``config`` that starts from ``source``.

    ``source`` is a parent run directory or one checkpoint file inside it.
    """
    from .runner import _read_identity

    if isinstance(config.get("warm_start"), Mapping):
        _fail("the configuration already contains a warm start")
    if config.get("resume"):
        _fail("a warm start creates a new run and cannot be combined with resume")
    check_supported(config)
    run, chosen = _run_directory(source)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema", "pyrpmd.run") != "pyrpmd.run" or "physics_fingerprint" not in manifest:
        _fail(f"{run / 'manifest.json'} is not a PyRPMD run manifest")
    parent_config = json.loads((run / "input" / "resolved.json").read_text())
    if parent_config.get("task") == "pcet":
        _fail("a PCET run has no ring-polymer configuration")
    parent_system = _read_identity(parent_config, run)
    if list(parent_system.symbols) != list(system.symbols):
        _fail("the parent and new systems must contain the same atoms in the same order")
    parent_cell = getattr(parent_system, "cell", None)
    cell = getattr(system, "cell", None)
    periodic = any(getattr(system, "pbc", ())) or any(getattr(parent_system, "pbc", ()))
    if periodic:
        if (parent_cell is None or cell is None or list(parent_system.pbc) != list(system.pbc)
                or not np.allclose(parent_cell, cell, rtol=0.0, atol=1e-12)):
            _fail("periodic warm starts require the same cell and periodicity as the parent")
        if _has_barostat(parent_config):
            _fail("the parent used a barostat; its final cell is not imported, so its coordinates "
                  "cannot be combined with the new input cell")
    old_beads = int(parent_config["beads"])
    found = _candidates(run, chosen, old_beads, system.natoms)
    name, index, q, hills, _ = _select(found, index)
    if not np.isfinite(q).all():
        _fail(f"{name} contains non-finite coordinates")
    q = interpolate_beads(q, int(config["beads"]))
    fixed = list(getattr(system, "fixed_atoms", ()) or ())
    if fixed:
        expected = np.broadcast_to(system.geometry[fixed], q[:, fixed].shape)
        if not np.allclose(q[:, fixed], expected, rtol=0.0, atol=1e-10):
            _fail("fixed atoms in the parent configuration differ from the new input geometry")
        q[:, fixed] = expected
    record = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "initialized_from": {
            "physics_fingerprint": manifest["physics_fingerprint"],
            "checkpoint": name,
            "index": index,
            "beads": old_beads,
            "task": parent_config.get("task"),
            "method": _method(parent_config),
        },
        "beads": int(config["beads"]),
        "bead_interpolation": "identity" if old_beads == int(config["beads"]) else "ring_polymer_fourier",
        "momenta": "resampled_maxwell",
        "rng": "new_seed",
        "equilibration": "required",
        "configuration": q.tolist(),
    }
    if import_bias_history:
        record["initial_bias"] = import_bias(parent_config, parent_system, config, system, hills)
    result = copy.deepcopy(config)
    result["warm_start"] = record
    return result, run


def initial_ring(config, system, beads=None):
    """Return the fresh-start ring coordinates for a supported workflow."""
    beads = int(config["beads"] if beads is None else beads)
    record = config.get("warm_start")
    if not record:
        return np.repeat(system.geometry[None], beads, axis=0)
    q = np.array(record["configuration"], dtype=float)
    if q.shape != (beads, system.natoms, 3) or not np.isfinite(q).all():
        raise ValueError("warm-start configuration does not match the run's beads and atoms")
    return q


def initial_hills(config):
    """Imported initial metadynamics hills, or None."""
    record = config.get("warm_start") or {}
    bias = record.get("initial_bias")
    return None if not bias else bias["hills"]


def report(config):
    """The summary entry that marks a warm-started run."""
    record = config.get("warm_start")
    if not record:
        return None
    result = {key: record[key] for key in ("initialized_from", "beads", "bead_interpolation",
                                            "momenta", "rng", "equilibration")}
    result["warm_start"] = True
    if record.get("initial_bias"):
        bias = record["initial_bias"]
        result["initial_bias"] = {key: bias[key] for key in ("kind", "role", "scale", "rule", "parent",
                                                             "temperature_changed")}
        result["initial_bias"]["hills"] = len(bias["hills"])
    return result
