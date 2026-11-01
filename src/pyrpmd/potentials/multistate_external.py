"""Published molecular DPEMs and an explicit optional ChemPotPy bridge.

Only a complete matrix in a fixed diabatic basis, with its positive Cartesian
derivatives, satisfies this contract. Adiabatic energies plus NACs are not a
fixed-basis DPEM and are deliberately not converted to one here.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import importlib
from pathlib import Path
import threading

import numpy as np

from ..constants import BOHR_ANGSTROM, EV_TO_AU
from .base import Potential, PotentialError, BackendUnavailableError, coordinates
from .nah2_leps import NaH2LEPS5G, _SOURCE


__all__ = ["NaH2LEPS5G", "molecular_potential_from_dict", "ChemPotPyDPEM"]


def molecular_potential_from_dict(value):
    """Reconstruct a whitelisted published PES, checking its full provenance.

    Unknown fields, modified source hashes and unrecognized schema versions
    fail closed. No Python import string or executable serialization is used.
    """
    expected = {"schema", "version", "model", "parameters", "source"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("molecular potential identity must contain exactly schema, version, model, parameters, source")
    if (value["schema"] != "pyrpmd.molecular_potential" or type(value["version"]) is not int
            or value["version"] != 1 or value["model"] != "nah2_leps_5g_1992"):
        raise ValueError("unsupported molecular potential schema/version/model")
    if (not isinstance(value["parameters"], Mapping) or set(value["parameters"]) != {"symbols"}
            or not isinstance(value["source"], Mapping) or dict(value["source"]) != _SOURCE):
        raise ValueError("molecular potential parameters or pinned source identity mismatch")
    return NaH2LEPS5G(symbols=value["parameters"]["symbols"])


# ChemPotPy modules use saved Fortran state; calls across adapter instances must
# be serialized too. Batch safety means independent frames, not reentrancy.
_CHEMPOTPY_LOCK = threading.RLock()


class ChemPotPyDPEM(Potential):
    """Optional ChemPotPy bridge for its published NaH2 5G DPEM.

    ``symbols`` fixes the input atom order. The upstream ``u`` API accepts
    labeled angstrom geometries and returns eV matrices in that fixed diabatic
    convention. This energy-only upstream entry has no analytic derivatives;
    this bridge therefore uses an explicitly selected five-point Cartesian
    difference, with ``difference_step_bohr`` recorded in metadata. For the
    direct differentiated port without numerical force error, use NaH2LEPS5G.

    The installed Fortran source must match the pinned source hash. Users must
    build their corresponding upstream extension; matching a source file does
    not independently authenticate a precompiled binary. No downloads or
    compilation happen implicitly. Only this verified surface is exposed here;
    the package's general DiabaticPythonPotential handles other explicit DPEMs.
    """
    natoms = 3
    nstates = 2
    ncoordinates = 9
    coordinates = tuple(range(9))

    def __init__(self, *, symbols=("Na", "H", "H"), difference_step_bohr=1e-4):
        checker = NaH2LEPS5G(symbols=symbols)
        step = np.asarray(difference_step_bohr)
        if (step.shape != () or step.dtype.kind not in "fiu" or not np.isfinite(step)
                or step <= 0):
            raise ValueError("difference_step_bohr must be a positive finite real scalar")
        self.difference_step_bohr = float(step)
        if not np.isfinite(self.difference_step_bohr) or self.difference_step_bohr <= 0:
            raise ValueError("difference_step_bohr is outside the float64 range")
        self.symbols = checker.symbols
        try:
            self._api = importlib.import_module("chempotpy")
        except ImportError as exc:
            raise BackendUnavailableError("ChemPotPy is optional; install and compile its pinned NaH2 5G entry, "
                                          "or use the directly available NaH2LEPS5G provider") from exc
        if not callable(getattr(self._api, "u", None)):
            raise BackendUnavailableError("installed ChemPotPy does not provide its DPEM u() API")
        source = Path(self._api.__file__).resolve().parent/"NaH2"/"NaH2_LEPS_5G_1992_DPEM.f"
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != _SOURCE["sha256"]:
            raise BackendUnavailableError("installed ChemPotPy NaH2 5G Fortran source does not match the pinned hash")
        self._identity = {"surface": "NaH2_LEPS_5G_1992_DPEM", "system": "NaH2",
                          "symbols": list(self.symbols), "source": dict(_SOURCE),
                          "gradient": "five_point_cartesian_difference",
                          "difference_step_bohr": self.difference_step_bohr,
                          "input_bohr_to_angstrom": BOHR_ANGSTROM,
                          "output_ev_to_hartree": EV_TO_AU,
                          "wrapper_sha256": hashlib.sha256(Path(self._api.__file__).read_bytes()).hexdigest()}
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("external DPEM configuration is immutable")
        object.__setattr__(self, name, value)

    @property
    def metadata(self):
        return {"provider": "chempotpy_dpem", "natoms": 3, "nstates": 2,
                "charge": 0, "multiplicity": 2,
                "symbols": list(self.symbols), "execution_device": "cpu", "periodic": False,
                "thread_safety": "context_serial", "trajectory_batch_safe": True,
                "electronic_basis": "fixed_diabatic", "derivative": "gradient",
                "energy_unit": "hartree", "length_unit": "bohr",
                "source_identity": copy.deepcopy(self._identity)}

    def _matrix(self, q):
        geom = [[symbol, *xyz] for symbol, xyz in zip(self.symbols, q*BOHR_ANGSTROM)]
        try:
            result = np.asarray(self._api.u("NaH2", "NaH2_LEPS_5G_1992_DPEM", geom))
        except SystemExit as exc:
            raise PotentialError("ChemPotPy rejected the supplied NaH2 geometry") from exc
        if result.shape != (2, 2) or result.dtype.kind not in "fciu" or not np.isfinite(result).all():
            raise PotentialError("ChemPotPy must return the complete finite 2x2 diabatic matrix")
        return result*EV_TO_AU

    def evaluate(self, q):
        from ..nonadiabatic import DiabaticResult, validate_diabatic_result

        q = coordinates(q)
        if q.shape[1] != 3:
            raise PotentialError("ChemPotPy NaH2 requires three atoms")
        # The original Fortran STOPs at the H2 midpoint. Check every displaced
        # geometry first, so an invalid finite-difference stencil cannot exit
        # the host interpreter. This also checks finite/noncoincident nuclei.
        check = NaH2LEPS5G(symbols=self.symbols)
        step = self.difference_step_bohr
        matrix, gradient = [], []
        with _CHEMPOTPY_LOCK:
            for frame in q:
                stencil = [frame.copy()]
                for atom in range(3):
                    for axis in range(3):
                        for displacement in (2., 1., -1., -2.):
                            moved = frame.copy()
                            moved[atom, axis] += displacement*step
                            stencil.append(moved)
                check.evaluate(np.array(stencil))
                values = [self._matrix(point) for point in stencil]
                matrix.append(values[0])
                terms = np.asarray(values[1:]).reshape(3, 3, 4, 2, 2)
                fd = (-terms[:, :, 0]+8*terms[:, :, 1]-8*terms[:, :, 2]+terms[:, :, 3])/(12*step)
                gradient.append(fd.transpose(2, 3, 0, 1))
        return validate_diabatic_result(DiabaticResult(np.asarray(matrix), np.asarray(gradient)), q.shape, 2)
