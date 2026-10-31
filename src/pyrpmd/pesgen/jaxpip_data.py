"""Checked molecular reference datasets shared by Caracal and JaxPIP.

No JAX, PySCF or native-library import is needed to read an existing dataset.
All reference arrays use bohr, hartree and positive Cartesian energy gradients.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import tomllib

import numpy as np

from .formats import read_xyz
from .reference import (PySCFReference, PySCFSettings, ReferencePoint, atomic_json,
                        canonical_hash, file_hash)

SPLITS = ("train", "validation", "test")


def checked_json(path, schema):
    value = json.loads(Path(path).read_text())
    data = value.get("data", {})
    if value.get("sha256") != canonical_hash(data) or data.get("schema") != schema:
        raise ValueError(f"invalid schema/checksum: {path}")
    return data


def write_checked(path, data):
    atomic_json(Path(path), {"sha256": canonical_hash(data), "data": data})


def member(root, name):
    """Reject absolute paths and traversal in portable bundle manifests."""
    root = Path(root).resolve()
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("manifest members must be relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("manifest member escapes its directory")
    return path


@contextmanager
def output_lock(directory):
    """Exclusive per-directory lock; importing this module remains portable."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another writer owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def geometry_key(symbols, positions):
    """Conservative rotation/translation/permutation-invariant duplicate check.

    Sorted element-pair distance multisets can reject distinct homometric
    structures too. This is intentionally conservative, not a molecular identity
    test. Near-duplicates and time-correlated samples still need grouped splits.
    """
    q = np.asarray(positions, float)
    pairs = []
    for i in range(len(symbols)):
        for j in range(i):
            distance = float(np.linalg.norm(q[i] - q[j]))
            if not np.isfinite(distance) or distance < 1e-8:
                raise ValueError("coincident/nonfinite nuclei are not supported")
            pairs.append((*sorted((symbols[i], symbols[j])), round(distance, 10)))
    return canonical_hash(sorted(pairs))


def reference_contract(point):
    """A single electronic reference convention must cover the entire dataset."""
    if len(point.symbols) < 2:
        raise ValueError("JaxPIP requires at least two nuclei")
    info = point.provenance
    if info.get("engine") != "PySCF" or info.get("scf_converged") is not True:
        raise ValueError("JaxPIP datasets require converged PySCF ReferencePoint data")
    if not isinstance(info.get("version"), str) or not isinstance(info.get("settings"), dict):
        raise ValueError("missing PySCF version/settings provenance")
    settings = PySCFSettings(**info["settings"])
    from ..system import System
    system = System(point.symbols, np.ones(len(point.symbols)), point.positions,
                    charge=settings.charge, multiplicity=settings.spin + 1)
    electrons = int(system.atomic_numbers.sum()) - settings.charge
    if electrons < settings.spin or (electrons - settings.spin) % 2:
        raise ValueError("electron count and spin are inconsistent")
    return {"symbols": list(point.symbols), "atomic_numbers": system.atomic_numbers.tolist(),
            "charge": settings.charge, "multiplicity": settings.spin + 1,
            "engine": "PySCF", "version": info["version"], "settings": asdict(settings)}


def check_points(points):
    if set(points) - set(SPLITS) or len(points.get("train", [])) < 2:
        raise ValueError("use train/validation/test splits, with at least two training points")
    contract, seen = None, {}
    for split in SPLITS:
        for point in points.get(split, []):
            current = reference_contract(point)
            if contract is None:
                contract = current
            elif current != contract:
                raise ValueError("reference atom order, electronic state or PySCF settings differ")
            key = geometry_key(point.symbols, point.positions)
            if key in seen and seen[key] != split:
                raise ValueError("duplicate geometry across dataset splits (including rigid transforms/permutations)")
            seen[key] = split
    return contract


def load_dataset(directory):
    directory = Path(directory).resolve()
    manifest = checked_json(directory / "dataset.json", "pyrpmd.jaxpip-dataset/1")
    if manifest.get("units") != {"positions": "bohr", "energy": "hartree", "gradient": "hartree/bohr"}:
        raise ValueError("dataset units must be atomic units with positive gradients")
    points = {s: [] for s in SPLITS}
    for item in manifest["records"]:
        if item.get("split") not in points:
            raise ValueError("invalid dataset split")
        path = member(directory, item["file"])
        if file_hash(path) != item["sha256"]:
            raise ValueError("dataset reference checksum mismatch")
        points[item["split"]].append(ReferencePoint.load(path))
    if check_points(points) != manifest["contract"]:
        raise ValueError("dataset contract mismatch")
    return manifest, points


def collect_dataset(input_file, output, *, resume=False):
    """Collect explicit XYZ splits or import existing checked reference JSON files.

    Fresh references request only energy and gradient, never unnecessary Hessians
    or population analyses. Interrupted QM work reuses the PES-generation cache.
    """
    if type(resume) is not bool:
        raise ValueError("resume must be Boolean")
    input_file, output = Path(input_file).resolve(), Path(output).resolve()
    spec = tomllib.loads(input_file.read_text())
    if set(spec) - {"schema", "geometries", "references", "qm"} or spec.get("schema") != "pyrpmd.jaxpip-data/1":
        raise ValueError("expected strict pyrpmd.jaxpip-data/1 input")
    mode = "geometries" if "geometries" in spec else "references"
    if ("geometries" in spec) == ("references" in spec) or (mode == "references" and "qm" in spec):
        raise ValueError("select either geometries + qm, or references with their existing provenance")
    source = spec[mode]
    if not isinstance(source, dict) or set(source) - set(SPLITS) or "train" not in source:
        raise ValueError("explicit train and optional validation/test splits are required")
    files, inputs = {}, {}
    for split, names in source.items():
        names = [names] if mode == "geometries" else names
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise ValueError("geometries are XYZ paths; references are nonempty lists of JSON paths")
        inputs[split] = [(input_file.parent / n).resolve() for n in names]
        for p in inputs[split]:
            files[str(p)] = file_hash(p)
    settings = PySCFSettings(**spec.get("qm", {})) if mode == "geometries" else None
    version = importlib.metadata.version("pyscf") if mode == "geometries" else None
    fingerprint = canonical_hash({"spec": spec, "files": files, "pyscf": version,
                                  "code": file_hash(Path(__file__)),
                                  "reference_code": file_hash(Path(__file__).with_name("reference.py"))})
    with output_lock(output):
        runfile = output / "collection.json"
        if runfile.exists():
            old = checked_json(runfile, "pyrpmd.jaxpip-collection/1")
            if not resume or old["fingerprint"] != fingerprint:
                raise ValueError("existing dataset needs resume=True and identical inputs/settings")
            if (output / "dataset.json").exists():
                return load_dataset(output)[0]
        else:
            if any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("refusing a nonempty unrelated output directory")
            write_checked(runfile, {"schema": "pyrpmd.jaxpip-collection/1", "fingerprint": fingerprint})
        points = {s: [] for s in SPLITS}
        cache = output / "references"
        cache.mkdir(exist_ok=True)
        if mode == "geometries":
            from .workflow import _reference
            engine = PySCFReference(settings)
            for split, paths in inputs.items():
                symbols, frames = read_xyz(paths[0])
                for q in frames:
                    points[split].append(_reference(engine, symbols, q, cache, hessian=False, populations=False))
        else:
            for split, paths in inputs.items():
                points[split] = [ReferencePoint.load(p) for p in paths]
        contract = check_points(points)
        records = []
        for split in SPLITS:
            for i, point in enumerate(points[split]):
                name = f"references/{split}-{i:06d}.json"
                point.save(output / name)
                records.append({"split": split, "file": name, "sha256": file_hash(output / name)})
        manifest = {"schema": "pyrpmd.jaxpip-dataset/1", "fingerprint": fingerprint,
                    "contract": contract, "records": records, "source_files": files,
                    "units": {"positions": "bohr", "energy": "hartree", "gradient": "hartree/bohr"},
                    "split_warning": "Choose independent trajectory/path groups; duplicate checks do not prove independence."}
        write_checked(output / "dataset.json", manifest)
        return manifest
