"""Atomic metadata and portable array storage for reproducible runs."""

from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import tomllib
import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(x) for x in value]
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if callable(value):
        return {"python_callable": f"{value.__module__}:{value.__qualname__}"}
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def fingerprint(config, system):
    """Science identity excludes output destination and execution topology."""
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
    }
    value = {"config": data, "system": system.to_dict()}
    from .cv import geometry_convention_identity

    convention = geometry_convention_identity(config, system)
    if convention is not None:
        value["cv_geometry_convention"] = convention
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


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
