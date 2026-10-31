"""dRPMD restart identity, metadata reads and state serialization.

Collision assembly, centroid connectivity and flux estimators are adapted from
chen-yu-hao/RPMDrate, commit 9a44d6b378d8bf835d67e1b3b7f78b27bdc84a78,
rpmdrate/drpmd.py (MIT). Original RPMDrate copyright (c) 2012 Joshua W. Allen,
William H. Green and Yury V. Suleimanov; see THIRD_PARTY_NOTICES.md and LICENSE.
"""

from __future__ import annotations

from functools import partial
import importlib.util
import json
import math
from pathlib import Path
import tomllib

from pyrpmd.io import file_sha256, fingerprint_payload, jsonable
from pyrpmd.propagation import scientific_config

_jsonable = partial(jsonable, dataclasses="asdict", callables=False, to_dict="ignore")


def _checkpoint_clock_matches(elapsed, steps, dt):
    """Allow the roundoff of a binary64 clock advanced once per accepted step.

    Repeated ``time += dt`` need not equal one multiplication ``steps * dt``.
    With unit roundoff u=2**-53, gamma(n)=n*u/(1-n*u) bounds the relative
    forward error of n floating-point operations.  The extra operation covers
    the reference multiplication; retaining the former tolerance floor also
    accepts existing checkpoints with short clocks.  Step counts are checked
    separately and exactly by both readers.
    """
    try:
        expected = steps * dt
        roundoff = (steps + 1) * 2.0**-53
    except OverflowError:
        return False
    if (not math.isfinite(elapsed) or not math.isfinite(expected)
            or roundoff >= 1):
        return False
    relative_bound = roundoff / (1 - roundoff)
    return math.isclose(elapsed, expected,
                        rel_tol=max(1e-11, relative_bound), abs_tol=1e-12)


def _load_npz(path, *, expected_kind=None):
    from ..checkpoint import CheckpointError, load_checkpoint

    with load_checkpoint(path, expected_kind=expected_kind) as data:
        meta = json.loads(str(data["metadata"].item()))
        if expected_kind is not None and meta.get("format") != expected_kind:
            raise CheckpointError(
                f"Expected checkpoint payload format {expected_kind!r}, got {meta.get('format')!r}"
            )
        arrays = {key: data[key].copy() for key in data.files if key != "metadata"}
    return meta, arrays


def _dependency_fingerprints(config):
    """Hash declared local PES/model/classifier inputs, without loading models.

    Python transitive imports, system shared libraries and remote i-PI engines
    are outside this check and must be pinned by the run environment.
    """
    files = {}

    def add_file(value, base=Path(".")):
        try:
            path = Path(value)
            path = path.resolve() if path.is_absolute() else (base / path).resolve()
            exists = path.is_file()
        except (OSError, ValueError):
            return False
        if exists and str(path) not in files:
            files[str(path)] = file_sha256(path)
        return exists

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("factory", "function", "module") and isinstance(item, str):
                    module = item.split(":", 1)[0]
                    if not add_file(module):
                        try:
                            spec = importlib.util.find_spec(module)
                        except (ImportError, ValueError, AttributeError):
                            spec = None
                        if spec is not None and spec.origin not in (
                            None,
                            "built-in",
                            "frozen",
                        ):
                            add_file(spec.origin)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, (str, Path)):
            add_file(value)

    walk(config.get("potential", {}))
    walk(config.get("outcomes", {}))
    manifest = config.get("potential", {}).get("manifest")
    if manifest:
        path = Path(manifest).resolve()
        data = (
            json.loads(path.read_text())
            if path.suffix.lower() == ".json"
            else tomllib.loads(path.read_text())
        )
        for key in ("objects", "wrapper_sources"):
            for name in data.get(key, []):
                add_file(name, path.parent)
    return files


def _physics_hash(config, system, parents):
    config = scientific_config(config)
    scientific = {
        k: v
        for k, v in config.items()
        if k not in ("parallel", "output", "resume", "_resume")
        and not k.startswith("_")
    }
    if config.get("initial_ensemble"):
        # Independent draws are keyed by global trajectory ID. Work allocation
        # is not a physical parent chain and must not affect their identity.
        scientific["sampling"] = {
            k: v for k, v in scientific.get("sampling", {}).items() if k != "parents"
        }
        parents = 0
    payload = {
        "config": scientific,
        "system": _jsonable(system),
        "parents": parents,
        "local_dependencies_sha256": _dependency_fingerprints(config),
        "format": "PyRPMD.drpmd.physics.v1",
    }
    return fingerprint_payload(payload, dataclasses="asdict", callables=False, to_dict="ignore")


def _restore_rng(dynamics, state):
    if state is not None:
        dynamics.rng.bit_generator.state = state


def _rng_state(dynamics):
    return _jsonable(dynamics.rng.bit_generator.state)


def _mapping_arrays(state, prefix=""):
    if not hasattr(state, "mapping_q"):
        return {}
    return {prefix + "mapping_q": state.mapping_q, prefix + "mapping_p": state.mapping_p}
