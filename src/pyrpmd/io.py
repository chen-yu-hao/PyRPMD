"""Atomic metadata and portable array storage for reproducible runs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import tomllib
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path

import numpy as np


def jsonable(
    value,
    *,
    nonfinite="raise",
    callables=True,
    dataclasses="to_dict",
    to_dict="recursive",
):
    """Convert portable values with an explicit policy for missing numbers.

    Configuration/checkpoint metadata is strict. Numerical reports may select
    ``nonfinite='null'`` to represent unsampled bins in valid JSON. ``callables``
    retains the manifest's callable identity convention; stream identities use
    their historical ``repr`` fallback instead. ``to_dict`` controls whether an
    object's explicit representation is recursively converted, kept shallow
    (legacy FES), or ignored (legacy dRPMD). Dataclasses can explicitly select
    their fields with ``dataclasses='asdict'``.
    """
    if nonfinite not in ("raise", "null"):
        raise ValueError("nonfinite must be 'raise' or 'null'")
    if dataclasses not in ("to_dict", "asdict"):
        raise ValueError("dataclasses must be 'to_dict' or 'asdict'")
    if to_dict not in ("recursive", "shallow", "ignore"):
        raise ValueError("to_dict must be 'recursive', 'shallow' or 'ignore'")

    def convert(item):
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, Path):
            return str(item)
        if (
            dataclasses == "asdict"
            and is_dataclass(item)
            and not isinstance(item, type)
        ):
            return convert(asdict(item))
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(x) for x in item]
        if isinstance(item, float) and not np.isfinite(item):
            if nonfinite == "null":
                return None
            raise ValueError("Nonfinite values are not permitted in JSON metadata")
        if to_dict != "ignore" and hasattr(item, "to_dict"):
            result = item.to_dict()
            return convert(result) if to_dict == "recursive" else result
        if callables and callable(item):
            return {"python_callable": f"{item.__module__}:{item.__qualname__}"}
        return item

    return convert(value)


def atomic_write_text(path, text):
    """Publish UTF-8 text using a unique temporary beside the destination.

    A failed write, flush, or replacement leaves the previous file intact and
    removes the temporary. Separate writers never share a temporary filename;
    the last completed replacement wins. This is not a run-directory lock.
    All fallible durability work precedes the commit at ``os.replace``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(
    path,
    value,
    *,
    nonfinite="raise",
    sort_keys=False,
    callables=True,
    dataclasses="to_dict",
    to_dict="recursive",
    ensure_ascii=False,
    trailing_newline=True,
):
    """Atomically write JSON with explicit conversion and text-format policies."""
    raw = (
        json.dumps(
            jsonable(
                value,
                nonfinite=nonfinite,
                callables=callables,
                dataclasses=dataclasses,
                to_dict=to_dict,
            ),
            ensure_ascii=ensure_ascii,
            indent=2,
            sort_keys=sort_keys,
            allow_nan=False,
        )
        + ("\n" if trailing_newline else "")
    )
    atomic_write_text(path, raw)


# Reports deliberately represent missing/nonfinite estimates as JSON null.
report_jsonable = partial(
    jsonable,
    nonfinite="null",
    callables=False,
    to_dict="shallow",
)
write_report_json = partial(
    write_json,
    nonfinite="null",
    callables=False,
    to_dict="shallow",
    ensure_ascii=True,
)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def fingerprint_payload(
    value,
    *,
    compact=False,
    nonfinite="raise",
    default=None,
    callables=True,
    dataclasses="to_dict",
    to_dict="recursive",
    allow_nan=False,
):
    """Hash a canonical JSON payload while preserving versioned encodings."""
    options = {"sort_keys": True, "allow_nan": allow_nan}
    if compact:
        options["separators"] = (",", ":")
    if default is not None:
        options["default"] = default
    return hashlib.sha256(
        json.dumps(
            jsonable(
                value,
                nonfinite=nonfinite,
                callables=callables,
                dataclasses=dataclasses,
                to_dict=to_dict,
            ),
            **options,
        ).encode()
    ).hexdigest()


def fingerprint(config, system, extras=None):
    """Return the stable identity for a manifest or a sampling stream.

    With ``extras=None``, retain the manifest's original config/system schema.
    Supplying a stream mapping selects the existing v1 stream schema, which
    ignores analysis-only and execution settings but retains sampling measure,
    flux axis and CV geometry. Keeping both schemas byte-compatible lets old
    manifests and checkpoints resume after implementation refactors.
    Versioned geometry and potential-energy conventions remain part of both
    identities when their providers require them.
    """
    if extras is not None:
        from .propagation import scientific_config

        config = scientific_config(config)
    data = {
        k: v
        for k, v in config.items()
        if k
        not in (
            "output",
            "parallel",
            "resume",
            "_resume",
            "_warnings",
            "_input_path",
            "_base_directory",
        )
        and (extras is None or k != "analysis")
    }
    if extras is None:
        value = {"config": data, "system": system.to_dict()}
    else:
        value = {
            "schema": 1,
            "physics": data,
            "sampling_measure": config.get("analysis", {}).get("measure", "cv_density"),
            "system": system.to_dict(),
            "stream": extras,
        }
        if "flux_axis" in config.get("analysis", {}):
            value["sampling_flux_axis"] = config["analysis"]["flux_axis"]
    from .cv import geometry_convention_identity

    convention = geometry_convention_identity(config, system)
    if convention is not None:
        value["cv_geometry_convention"] = convention
    from .potentials import energy_convention_identity

    energy_convention = energy_convention_identity(config)
    if energy_convention is not None:
        value["potential_energy_convention"] = energy_convention
    if extras is None:
        return fingerprint_payload(value)
    return fingerprint_payload(
        value,
        compact=True,
        nonfinite="null",
        default=repr,
        callables=False,
        to_dict="shallow",
        allow_nan=True,
    )


def referenced_files(config):
    """Hash explicit inputs and local plugin sources, including .o link inputs.

    This is not a Python dependency lock: indirect imports and remote potential
    servers must be pinned by the user's environment as well.
    """
    files = {}
    manifests = []

    def add(path):
        path = Path(path).resolve()
        if path.is_file() and str(path) not in files:
            files[str(path)] = file_sha256(path)

    def visit(value, key=None):
        if isinstance(value, dict):
            for k, v in value.items():
                if k == "directory":
                    continue
                visit(v, k)
        elif isinstance(value, (list, tuple)):
            for v in value:
                visit(v)
        elif isinstance(value, str):
            if key in ("factory", "module", "function", "initializer"):
                module = value.partition(":")[0]
                if module.endswith((".py", ".so")):
                    add(module)
                elif ":" in value or key == "module":
                    try:
                        spec = importlib.util.find_spec(module)
                        if (
                            spec
                            and spec.origin
                            and spec.origin not in ("built-in", "frozen")
                        ):
                            add(spec.origin)
                    except (ImportError, ValueError, AttributeError):
                        pass
            elif value.startswith("/"):
                add(value)
                if key == "manifest":
                    manifests.append(Path(value))

    visit(config)
    for path in manifests:
        if not path.is_file():
            continue
        contents = path.read_text(encoding="utf-8")
        manifest = (
            json.loads(contents)
            if path.suffix.lower() == ".json"
            else tomllib.loads(contents)
        )
        for key in ("objects", "wrapper_sources"):
            for item in manifest.get(key, []):
                add(path.parent / Path(item))
    return files


def save_hdf5(path, arrays, metadata=None):
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema_version"] = 1
            f.attrs["metadata_json"] = json.dumps(
                jsonable(metadata or {}), allow_nan=False
            )
            for name, array in arrays.items():
                a = np.asarray(array)
                f.create_dataset(name, data=a, compression="gzip" if a.ndim else None)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
