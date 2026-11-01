"""Resolve an atomically published rate generation without running dynamics.

The root result summary is the publication point. Earlier root artifacts remain
the original result; a pending generation must never be inferred from directory
contents or from the extension ledger alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from .io import file_sha256


EXTENSION_FORMAT = "PyRPMD.rate.extensions.v1"
RECEIPT_FORMAT = "PyRPMD.rate.generation-receipt.v1"


def generation_name(production_steps):
    if type(production_steps) is not int or production_steps < 1:
        raise ValueError("Invalid rate generation production budget")
    return f"extensions/rate/production-{production_steps:012d}"


def read_generation(directory, pointer):
    """Validate the receipt and all published artifacts of one generation."""
    root = Path(directory).resolve()
    if not isinstance(pointer, dict) or pointer.get("format") != EXTENSION_FORMAT:
        raise ValueError("Unknown published rate extension format")
    target = pointer.get("production_steps")
    relative = generation_name(target)
    if pointer.get("generation") != relative:
        raise ValueError("Published rate extension has a conflicting generation path")
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("Rate generation must remain inside its run directory")
    receipt_path = path / "receipt.json"
    if file_sha256(receipt_path) != pointer.get("receipt_sha256"):
        raise ValueError("Rate generation receipt checksum differs")
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if (not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT
            or receipt.get("production_steps") != target
            or receipt.get("generation") != relative
            or receipt.get("physics_fingerprint") != manifest.get("physics_fingerprint")):
        raise ValueError("Rate generation receipt does not match the original run")
    artifacts = receipt.get("artifacts")
    summary = json.loads((path / "summary.json").read_text())
    if (not isinstance(summary, dict) or summary.get("task") != "rate"
            or summary.get("effective_production_steps") != target):
        raise ValueError("Rate generation summary has a conflicting production budget")
    required = {"generation.json", "summary.json", "rate_pmf.json", "rate_pmf.npz",
                "umbrella/fes.npz", "umbrella/summary.json"}
    if summary.get("kappa") is not None:
        required.add("recrossing.npz")
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise ValueError("Rate generation receipt lacks required scientific artifacts")
    for name, digest in artifacts.items():
        if (not isinstance(name, str) or Path(name).is_absolute()
                or ".." in Path(name).parts
                or not (path / name).resolve().is_relative_to(path.resolve())):
            raise ValueError("Invalid rate generation artifact path")
        if file_sha256(path / name) != digest:
            raise ValueError(f"Rate generation artifact checksum differs: {name}")
    return path, summary, receipt


def active_rate_directory(directory, summary=None):
    """Return the committed artifact folder, retaining legacy root layouts."""
    root = Path(directory).resolve()
    if summary is None:
        summary_path = root / "results" / "summary.json"
        if not summary_path.is_file():
            return root
        summary = json.loads(summary_path.read_text())
    if "rate_extension" not in summary:
        return root
    pointer = summary["rate_extension"]
    path, saved, _ = read_generation(root, pointer)
    # A copied pointer must not relabel a different scientific summary.
    for key, value in saved.items():
        if summary.get(key) != value:
            raise ValueError(f"Published rate summary differs from its generation: {key}")
    return path
