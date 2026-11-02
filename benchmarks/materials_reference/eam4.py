"""Static-only adapter for the pinned, author-attributed Ni/H EAM4 routine.

This is deliberately not registered as a production PyRPMD potential provider.
It evaluates one geometry, one H and a declared mobile/fixed Ni partition.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from .build_eam4 import SOURCE_SHA256, sha256_file


class EAM4StaticReference:
    """Evaluate original eV/angstrom values with no coefficient/unit changes."""

    def __init__(self, library: str | Path):
        self.library_path = Path(library).resolve()
        receipt = json.loads(self.library_path.with_name("build.json").read_text())
        if receipt.get("schema") != "pyrpmd.materials-reference.eam4-static-build.v1":
            raise ValueError("unrecognized EAM4 build receipt")
        if receipt["original_source"]["sha256"] != SOURCE_SHA256:
            raise ValueError("EAM4 source identity mismatch")
        if receipt["library"]["sha256"] != sha256_file(self.library_path):
            raise ValueError("EAM4 library identity mismatch")
        self.build_receipt = receipt
        self._library = ctypes.CDLL(str(self.library_path))
        self._evaluate = self._library.pyrpmd_eam4_static
        pointer = ctypes.POINTER(ctypes.c_double)
        self._evaluate.argtypes = [ctypes.c_int] * 4 + [pointer] * 4 + [ctypes.POINTER(ctypes.c_int)]
        self._evaluate.restype = None

    def evaluate(self, q_angstrom, *, mobile_ni: int, lengths_angstrom=(24.0, 26.0, 30.0),
                 periodic_xy: bool = False, energy_only: bool = False):
        """Return energy and dV/dq; fixed rows are zero, or all NaN in energy-only mode.

        Atom order is H, mobile Ni, fixed Ni. Coincident minimum-image atoms and
        cells too small for the unmodified 10 angstrom cutoff are rejected.
        Arbitrary cells, multiple H, mixed bead counts and dynamics are outside
        this adapter's contract.
        """
        q = np.array(q_angstrom, dtype=np.float64, copy=True, order="C")
        lengths = np.array(lengths_angstrom, dtype=np.float64, copy=True)
        if q.ndim != 2 or q.shape[1] != 3 or q.shape[0] < 2 or not np.isfinite(q).all():
            raise ValueError("provide finite (H + at least one Ni, 3) coordinates")
        if not isinstance(mobile_ni, (int, np.integer)) or isinstance(mobile_ni, (bool, np.bool_)):
            raise ValueError("mobile_ni must be an integer")
        if not 0 <= mobile_ni <= len(q)-1:
            raise ValueError("invalid mobile/fixed Ni partition")
        if lengths.shape != (3,) or not np.isfinite(lengths).all() or (lengths <= 0).any():
            raise ValueError("three finite positive cell lengths are required")
        if periodic_xy and (lengths[:2] <= 20.0).any():
            raise ValueError("XY minimum-image lengths must exceed twice the 10 angstrom cutoff")
        separation = q[:, None, :] - q[None, :, :]
        if periodic_xy:
            separation[:, :, :2] -= lengths[:2] * np.rint(separation[:, :, :2] / lengths[:2])
        squared = np.sum(separation * separation, axis=-1)
        np.fill_diagonal(squared, np.inf)
        if np.any(squared < 1e-16):
            raise ValueError("coincident atoms are outside the source domain")
        coordinates = np.asfortranarray(q.T)
        gradient = np.empty_like(coordinates, order="F")
        energy = ctypes.c_double()
        status = ctypes.c_int()
        ptr = ctypes.POINTER(ctypes.c_double)
        self._evaluate(int(mobile_ni), len(q)-1-int(mobile_ni), int(bool(periodic_xy)), int(bool(energy_only)),
                       coordinates.ctypes.data_as(ptr), lengths.ctypes.data_as(ptr), ctypes.byref(energy),
                       gradient.ctypes.data_as(ptr), ctypes.byref(status))
        if status.value:
            raise ValueError(f"EAM4 static native evaluation rejected the input (status {status.value})")
        return energy.value, gradient.T.copy()
