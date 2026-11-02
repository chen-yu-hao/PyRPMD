"""Unmodified CBE/PES-2008 electronic energy and forces; no rate validation."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

from pyrpmd.potentials import PESResult, Potential, PotentialError
from pyrpmd.potentials.base import coordinates, validate_result
from pyrpmd.potentials.cabi import _lock_for

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text())
DEFAULT_LIBRARY = HERE / "build" / "libcbe2008.so"
_DOUBLE_ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_INT_ARRAY = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


class CBE2008Potential(Potential):
    """[H,C,H,H,H,H], last H incoming; retain all upstream conversion constants.

    The reported gradient has a known 2.06 ppm scale difference from the
    derivative of the reported energy, due to upstream rounded constants.
    See source.json and verify.py. Do not treat this adapter as a validated rate.
    """

    def __init__(self, library=None):
        self.path = Path(library or DEFAULT_LIBRARY).resolve()
        receipt = self.path.with_suffix(".build.json")
        if not self.path.is_file() or not receipt.is_file():
            raise FileNotFoundError("Build the pinned CBE/PES-2008 library with benchmarks/hch4/build_pes.py first")
        self.provenance = json.loads(receipt.read_text())
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        wrapper_digest = hashlib.sha256((HERE / "wrapper.f90").read_bytes()).hexdigest()
        if (self.provenance.get("source") != SOURCE
                or self.provenance.get("library_sha256") != digest
                or self.provenance.get("wrapper_sha256") != wrapper_digest):
            raise PotentialError("CBE/PES-2008 library/source/wrapper provenance mismatch")
        self._lock = _lock_for(self.path)
        self.lib = ctypes.CDLL(str(self.path))
        self.batch = self.lib.pyrpmd_cbe2008_batch
        self.batch.argtypes = [ctypes.c_int32, _DOUBLE_ARRAY, _DOUBLE_ARRAY, _DOUBLE_ARRAY, _INT_ARRAY]
        self.batch.restype = None
        self.metadata = {
            "provider": "CBE2008", "model": SOURCE["model"],
            "execution_device": "cpu", "thread_safety": "process_serial",
            "native_batch": True, "dtype": "float64", "atomic_numbers": SOURCE["atomic_numbers"],
            "source_commit": SOURCE["commit"], "library_sha256": digest,
            "legacy_constants": SOURCE["legacy_constants"],
            "energy_zero": SOURCE["energy_zero"],
            "validation_scope": "electronic PES/adapter only; no thermal rate",
        }

    def evaluate(self, x):
        x = coordinates(x)
        if x.shape[1:] != (6, 3):
            raise PotentialError("CBE/PES-2008 requires six atoms in [H,C,H,H,H,H] order")
        if len(x) > np.iinfo(np.int32).max:
            raise PotentialError("CBE/PES-2008 batch is too large")
        energy = np.empty(len(x), dtype=np.float64)
        forces = np.empty_like(x)
        status = np.zeros(len(x), dtype=np.int32)
        with self._lock:
            self.batch(len(x), x, energy, forces, status)
        messages = tuple("" if s == 0 else ("Numerical guard: every atom pair must be >=0.2 bohr" if s == 1
                         else "Upstream CBE/PES-2008 returned an error or nonfinite values") for s in status)
        return validate_result(PESResult(energy, forces, status, messages), x.shape, require_success=False)


def build(library=None):
    """Factory usable with the Python-potential provider without importing PES files."""
    return CBE2008Potential(library)
