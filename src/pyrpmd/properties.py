"""Batch interface for molecular dipole and polarizability surfaces.

Coordinates are bohr, dipoles are e*bohr, polarizabilities are atomic units
(a0**3). Values are total-system properties per frame, not per-atom values.
The laboratory axes are preserved; averaging molecular rotations would change
Raman/SFG observables. No PES-force or ring-bead scaling is applied here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .constants import BOHR_ANGSTROM
from .potentials.base import coordinates


@dataclass
class PropertyResult:
    """Total dipole [batch,3] and symmetric polarizability [batch,3,3]."""

    dipole: np.ndarray | None = None
    polarizability: np.ndarray | None = None


@runtime_checkable
class PropertyModel(Protocol):
    """Contract implemented by Python, quantum-chemistry, and ML adapters."""

    def evaluate(self, positions: np.ndarray) -> PropertyResult:
        """Evaluate finite real coordinates [batch,atom,3] in bohr."""
        ...


def validate_property_result(value, batch_size, *, require=()):
    """Validate and own property buffers before another model evaluation.

    Models may return ``PropertyResult`` or a mapping with the singular keys
    ``dipole`` and ``polarizability``. A missing requested property is an error,
    not a zero spectrum. Complex resonant polarizabilities are outside this
    nonresonant interface and rejected explicitly.
    """
    if (isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer")
    if isinstance(require, str) or any(name not in ("dipole", "polarizability") for name in require):
        raise ValueError("require must contain dipole and/or polarizability")
    if isinstance(value, Mapping):
        if set(value) - {"dipole", "polarizability"}:
            raise ValueError("property result contains unknown fields")
        value = PropertyResult(**value)
    if not isinstance(value, PropertyResult):
        raise ValueError("property model must return PropertyResult or a property mapping")
    arrays = {}
    for name, shape in (("dipole", (batch_size, 3)),
                        ("polarizability", (batch_size, 3, 3))):
        raw = getattr(value, name)
        if raw is None:
            if name in require:
                raise ValueError(f"property model did not provide required {name}")
            arrays[name] = None
            continue
        raw = np.asarray(raw)
        if raw.dtype.kind not in "iuf" or raw.shape != shape or not np.isfinite(raw).all():
            raise ValueError(f"{name} must be a finite real array with shape {shape}")
        arrays[name] = np.array(raw, dtype=float, order="C", copy=True)
    alpha = arrays["polarizability"]
    if alpha is not None and not np.allclose(alpha, alpha.swapaxes(-1, -2), rtol=1e-8, atol=1e-10):
        raise ValueError("nonresonant polarizability must be symmetric in laboratory axes")
    if all(v is None for v in arrays.values()):
        raise ValueError("a property model must return at least one property")
    return PropertyResult(**arrays)


class CallablePropertyModel:
    """Wrap an ordinary callable using the atomic-unit property contract."""

    def __init__(self, evaluator):
        self.obj = evaluator
        self.evaluator = getattr(evaluator, "evaluate", evaluator)
        if not callable(self.evaluator):
            raise ValueError("evaluator must be callable or implement evaluate")
        self.metadata = dict(getattr(evaluator, "metadata", {}) or {})

    def evaluate(self, positions):
        q = coordinates(positions)
        return validate_property_result(self.evaluator(q.copy()), len(q))

    def close(self):
        close = getattr(self.obj, "close", None)
        if callable(close):
            close()


class TorchScriptPropertyModel:
    """Load a process-local TorchScript dipole/polarizability model.

    Standard forward signature: ``model(inputs: Dict[str, Tensor])``. The
    dictionary contains ``positions[B,N,3]``, ``atomic_numbers[N]`` (int64),
    ``cell[3,3]`` and ``pbc[3]`` (bool). Cell vectors are rows. Nonperiodic
    systems receive a zero cell and false PBC. The return value is a dictionary
    with ``dipole[B,3]`` and/or ``polarizability[B,3,3]``. Model-specific packages
    can provide a small scripted wrapper implementing this common contract.

    Set model units explicitly; converted outputs always use PyRPMD atomic
    units. Only TorchScript inference needs PyTorch; importing this module and
    using ordinary Python property models has no PyTorch dependency.
    """

    def __init__(self, path, atomic_numbers, *, cell=None, pbc=None,
                 device="cpu", dtype="float64", position_unit="bohr",
                 dipole_unit="au", polarizability_unit="au"):
        z = np.asarray(atomic_numbers)
        if (z.ndim != 1 or not len(z) or z.dtype.kind not in "iu"
                or np.any(z < 1) or np.any(z > 118)):
            raise ValueError("atomic_numbers must be a vector of integers in [1,118]")
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if position_unit not in ("bohr", "angstrom"):
            raise ValueError("position_unit must be bohr or angstrom")
        if dipole_unit not in ("au", "debye"):
            raise ValueError("dipole_unit must be au or debye")
        if polarizability_unit not in ("au", "angstrom3"):
            raise ValueError("polarizability_unit must be au or angstrom3")
        from .geometry import validate_cell_pbc
        validated_cell, validated_pbc = validate_cell_pbc(cell, pbc)
        self.cell = np.zeros((3, 3)) if validated_cell is None else validated_cell.copy()
        self.pbc = np.zeros(3, dtype=bool) if validated_pbc is None else validated_pbc.copy()
        self.atomic_numbers = np.array(z, dtype=np.int64, copy=True)
        for a in (self.cell, self.pbc, self.atomic_numbers):
            a.flags.writeable = False
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("TorchScript property model path must be a file")
        self.position_scale = 1.0 if position_unit == "bohr" else BOHR_ANGSTROM
        # 1 D = 3.33564095198152e-30 C m; e*a0 = 8.4783536255e-30 C m.
        from scipy.constants import physical_constants
        self.dipole_scale = (1.0 if dipole_unit == "au" else
                             3.33564095198152e-30 / physical_constants["atomic unit of electric dipole mom."][0])
        self.polarizability_scale = 1.0 if polarizability_unit == "au" else BOHR_ANGSTROM ** -3
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchScript property models require PyTorch; install PyRPMD[gpu]") from exc
        self._torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.model = torch.jit.load(str(self.path), map_location=self.device)
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.metadata = {
            "provider": "torchscript_properties", "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "atomic_numbers": self.atomic_numbers.tolist(), "cell_bohr": self.cell.tolist(),
            "pbc": self.pbc.tolist(), "device": str(self.device), "dtype": dtype,
            "model_position_unit": position_unit, "model_dipole_unit": dipole_unit,
            "model_polarizability_unit": polarizability_unit,
            "output_units": {"dipole": "e*bohr", "polarizability": "au"},
            "signature": "Dict[str,Tensor] -> Dict[str,Tensor]",
        }

    def evaluate(self, positions):
        q = coordinates(positions)
        if q.shape[1] != len(self.atomic_numbers):
            raise ValueError("property-model atom count differs from atomic_numbers")
        torch = self._torch
        inputs = {
            "positions": torch.tensor(q * self.position_scale, dtype=self.dtype, device=self.device),
            "atomic_numbers": torch.tensor(self.atomic_numbers.copy(), dtype=torch.int64, device=self.device),
            "cell": torch.tensor(self.cell * self.position_scale, dtype=self.dtype, device=self.device),
            "pbc": torch.tensor(self.pbc.copy(), dtype=torch.bool, device=self.device),
        }
        with torch.no_grad():
            raw = self.model(inputs)
        if not isinstance(raw, dict) or not raw or set(raw) - {"dipole", "polarizability"}:
            raise ValueError("TorchScript model must return a dipole/polarizability tensor dictionary")
        out = {}
        for key, value in raw.items():
            if not isinstance(value, torch.Tensor) or value.is_complex() or not value.is_floating_point():
                raise ValueError("TorchScript properties must be real floating-point tensors")
            scale = self.dipole_scale if key == "dipole" else self.polarizability_scale
            out[key] = value.detach().cpu().to(torch.float64).numpy() * scale
        return validate_property_result(out, len(q))


def evaluate_property_trajectory(model, positions, *, chunk_frames=128, require=()):
    """Evaluate [time,bead,atom,3] coordinates without averaging nonlinear surfaces.

    Returned arrays retain time and bead axes for spectroscopy estimators.
    ``chunk_frames`` bounds model batches; no trajectory-wide tensor is moved
    to a GPU. Model property availability must remain consistent across chunks.
    """
    q = np.asarray(positions)
    if (q.dtype.kind not in "iuf" or q.ndim != 4 or q.shape[-1] != 3
            or min(q.shape) < 1 or not np.isfinite(q).all()):
        raise ValueError("positions must be finite real [time,bead,atom,3]")
    if (isinstance(chunk_frames, (bool, np.bool_))
            or not isinstance(chunk_frames, (int, np.integer)) or chunk_frames < 1):
        raise ValueError("chunk_frames must be a positive integer")
    flat = q.reshape(-1, q.shape[-2], 3)
    collected = {"dipole": [], "polarizability": []}
    present = None
    for start in range(0, len(flat), chunk_frames):
        batch = np.ascontiguousarray(flat[start:start + chunk_frames], dtype=float)
        result = validate_property_result(model.evaluate(batch), len(batch), require=require)
        names = {key for key in collected if getattr(result, key) is not None}
        if present is not None and names != present:
            raise ValueError("property availability changed between model batches")
        present = names
        for key in names:
            collected[key].append(getattr(result, key))
    result = {}
    for key in present:
        joined = np.concatenate(collected[key], axis=0)
        result[key] = joined.reshape(*q.shape[:2], *joined.shape[1:])
    return result
