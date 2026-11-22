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


_PINNED_SURFACE = ("NaH2", "NaH2_LEPS_5G_1992_DPEM")
_GRADIENT_MODES = ("auto", "analytic", "difference")
_SOURCE_SUFFIXES = (".f", ".f90", ".F", ".F90")


def _identifier(value, name):
    if not isinstance(value, str) or not value or not all(c.isalnum() or c == "_" for c in value):
        raise ValueError(f"{name} must be a ChemPotPy name made of letters, digits and underscores")
    return value


class ChemPotPyDPEM(Potential):
    """Optional bridge to the diabatic potential energy matrices (DPEMs) of ChemPotPy.

    ``system`` and ``surface`` name an entry of ChemPotPy's ``multi_state_list``
    (for example ``("OH3", "OH3_PIP_FFW1_2019_DPEM")`` or
    ``("NH3", "NH3_GEN_FFW1_2006_DPEM")``); the default is the pinned NaH2 5G
    surface.  ``symbols`` fixes the caller's atom order; ChemPotPy reorders
    atoms internally and returns the matrix in the order given.  Its public
    units are angstrom and eV (gradients eV/angstrom); the bridge converts to
    bohr and hartree with the coordinate chain rule.

    ``gradient`` selects the matrix derivatives:

    ``"analytic"``  upstream ``ug``.  Many entries compute analytic DPEM
                    gradients, but some are energy-only and then return an
                    all-zero matrix and gradient without an error, and the
                    coupling gradients of others disagree with their own
                    matrix.  The first evaluation therefore checks ``ug``
                    against ``u`` and a five-point difference at the first
                    frame and at ``check_points - 1`` deterministic random
                    displacements of it (``check_displacement_bohr``), so a
                    symmetric frame where a coupling and its error vanish
                    cannot pass alone; a mismatch raises.
    ``"difference"`` five-point Cartesian differences of ``u`` with
                    ``difference_step_bohr`` (12N+1 calls per frame).
    ``"auto"``      ``"analytic"`` when that check passes, else ``"difference"``;
                    the mode used is recorded in ``metadata``.

    The default is ``"difference"`` for the pinned NaH2 5G entry (energy-only
    upstream) and ``"auto"`` otherwise.  :meth:`matrices` returns the matrices
    alone, one ``u`` call per frame.

    The installed Fortran source is hashed into the identity; the pinned NaH2
    5G entry, or any entry given ``source_sha256``, must match.  A matching
    source does not authenticate a binary built elsewhere.  Calls are
    serialized because upstream Fortran routines keep saved state.  Upstream
    routines may print to standard output, and an invalid geometry can make
    the Fortran ``STOP`` the interpreter; only the pinned NaH2 5G entry is
    screened for its known singularity first.  Nothing is downloaded or
    compiled here: compile the upstream extensions for the running Python.
    """

    def __init__(self, *, system="NaH2", surface="NaH2_LEPS_5G_1992_DPEM", symbols=None, gradient=None,
                 difference_step_bohr=1e-4, nstates=None, source_sha256=None, gradient_check_tolerance=1e-5,
                 check_points=4, check_displacement_bohr=0.05):
        step = np.asarray(difference_step_bohr)
        if (step.shape != () or step.dtype.kind not in "fiu" or not np.isfinite(step)
                or step <= 0):
            raise ValueError("difference_step_bohr must be a positive finite real scalar")
        step = float(step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("difference_step_bohr is outside the float64 range")
        system, surface = _identifier(system, "system"), _identifier(surface, "surface")
        pinned = (system, surface) == _PINNED_SURFACE
        if pinned:
            symbols = NaH2LEPS5G(symbols=("Na", "H", "H") if symbols is None else symbols).symbols
        else:
            if symbols is None:
                raise ValueError(f"symbols are required for ChemPotPy surface {surface}: one element per atom, "
                                 "in the order of the coordinates")
            if (not isinstance(symbols, (list, tuple)) or not symbols
                    or any(not isinstance(x, str) or not x for x in symbols)):
                raise ValueError("symbols must be a nonempty list of element symbols")
            symbols = tuple(symbols)
        gradient = ("difference" if pinned else "auto") if gradient is None else gradient
        if gradient not in _GRADIENT_MODES:
            raise ValueError(f"gradient must be one of {_GRADIENT_MODES}")
        if nstates is not None and (isinstance(nstates, bool) or int(nstates) != nstates or int(nstates) < 2):
            raise ValueError("nstates must be an integer >= 2")
        tolerance = float(gradient_check_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("gradient_check_tolerance must be finite and positive")
        if isinstance(check_points, bool) or int(check_points) != check_points or int(check_points) < 1:
            raise ValueError("check_points must be a positive integer")
        displacement = float(check_displacement_bohr)
        if not np.isfinite(displacement) or displacement < 0:
            raise ValueError("check_displacement_bohr must be finite and nonnegative")
        try:
            api = importlib.import_module("chempotpy")
        except ImportError as exc:
            raise BackendUnavailableError(
                "ChemPotPy is optional: install it and compile the extension of the requested DPEM for this "
                "Python" + (", or use the directly available NaH2LEPS5G provider" if pinned else "")) from exc
        if not callable(getattr(api, "u", None)):
            raise BackendUnavailableError("installed ChemPotPy does not provide its DPEM u() API")
        if gradient == "analytic" and not callable(getattr(api, "ug", None)):
            raise BackendUnavailableError("installed ChemPotPy does not provide its DPEM ug() API")
        if gradient == "auto" and not callable(getattr(api, "ug", None)):
            gradient = "difference"
        listed = getattr(api, "multi_state_list", None)
        if listed is not None and surface not in listed:
            raise BackendUnavailableError(f"{surface} is not a ChemPotPy DPEM (not in chempotpy.multi_state_list)")
        package = Path(api.__file__).resolve().parent
        source = next((package/system/(surface + suffix) for suffix in _SOURCE_SUFFIXES
                       if (package/system/(surface + suffix)).is_file()), None)
        expected = _SOURCE["sha256"] if pinned and source_sha256 is None else source_sha256
        digest = None if source is None else hashlib.sha256(source.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise BackendUnavailableError(f"installed ChemPotPy {surface} Fortran source does not match the pinned hash")
        if nstates is None and source is not None:
            import re

            found = re.search(r"nstates\s*=\s*(\d+)", source.read_text(errors="ignore"), re.IGNORECASE)
            nstates = int(found.group(1)) if found else None
        self._api, self.system, self.surface, self.symbols = api, system, surface, symbols
        self.difference_step_bohr, self.gradient_check_tolerance = step, tolerance
        self.check_points, self.check_displacement_bohr = int(check_points), displacement
        self.natoms = len(symbols)
        self.ncoordinates = 3*self.natoms
        self.coordinates = tuple(range(self.ncoordinates))
        self._pinned = pinned
        self._state = {"nstates": None if nstates is None else int(nstates), "requested": gradient,
                       "mode": None if gradient == "auto" else gradient, "verified": gradient == "difference",
                       "check": None}
        self._identity = {"surface": surface, "system": system, "symbols": list(symbols),
                          "source": dict(_SOURCE) if pinned else
                          {"path": None if source is None else f"chempotpy/{system}/{source.name}",
                           "sha256": digest, "pinned": expected is not None},
                          "gradient": "five_point_cartesian_difference" if gradient == "difference" else gradient,
                          "difference_step_bohr": step,
                          "input_bohr_to_angstrom": BOHR_ANGSTROM,
                          "output_ev_to_hartree": EV_TO_AU,
                          "chempotpy_version": getattr(api, "__version__", None),
                          "wrapper_sha256": hashlib.sha256(Path(api.__file__).read_bytes()).hexdigest()}
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("external DPEM configuration is immutable")
        object.__setattr__(self, name, value)

    @property
    def nstates(self):
        return self._state["nstates"]

    @property
    def metadata(self):
        info = {"provider": "chempotpy_dpem", "natoms": self.natoms, "nstates": self.nstates,
                "symbols": list(self.symbols), "system": self.system, "surface": self.surface,
                "execution_device": "cpu", "periodic": False,
                "thread_safety": "context_serial", "trajectory_batch_safe": True,
                "electronic_basis": "fixed_diabatic", "derivative": "gradient",
                "energy_unit": "hartree", "length_unit": "bohr",
                "gradient_mode": self._state["mode"] or "auto (decided at the first evaluation)",
                "gradient_check": copy.deepcopy(self._state["check"]),
                "source_identity": copy.deepcopy(self._identity)}
        if self._pinned:
            info.update(charge=0, multiplicity=2)
        return info

    def _geometry(self, frame):
        return [[symbol, *xyz] for symbol, xyz in zip(self.symbols, frame*BOHR_ANGSTROM)]

    def _shape(self, matrix):
        nstates = self._state["nstates"]
        if nstates is None and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0] >= 2:
            nstates = self._state["nstates"] = int(matrix.shape[0])
        if (nstates is None or matrix.shape != (nstates, nstates) or matrix.dtype.kind not in "fciu"
                or not np.isfinite(matrix).all()):
            size = "SxS" if nstates is None else f"{nstates}x{nstates}"
            raise PotentialError(f"ChemPotPy must return the complete finite {size} diabatic matrix")
        return nstates

    def _matrix(self, frame):
        try:
            result = np.asarray(self._api.u(self.system, self.surface, self._geometry(frame)))
        except SystemExit as exc:
            raise PotentialError(f"ChemPotPy rejected the supplied {self.system} geometry") from exc
        self._shape(result)
        return result*EV_TO_AU

    def _matrix_gradient(self, frame):
        try:
            matrix, gradient = self._api.ug(self.system, self.surface, self._geometry(frame))
        except SystemExit as exc:
            raise PotentialError(f"ChemPotPy rejected the supplied {self.system} geometry") from exc
        matrix, gradient = np.asarray(matrix), np.asarray(gradient)
        nstates = self._shape(matrix)
        if gradient.shape != (nstates, nstates, self.natoms, 3) or not np.isfinite(gradient).all():
            raise PotentialError(f"ChemPotPy ug must return finite [{nstates},{nstates},{self.natoms},3] gradients")
        return matrix*EV_TO_AU, gradient*(EV_TO_AU*BOHR_ANGSTROM)

    def _stencil(self, frame):
        step, stencil = self.difference_step_bohr, [frame.copy()]
        for atom in range(self.natoms):
            for axis in range(3):
                for displacement in (2., 1., -1., -2.):
                    moved = frame.copy()
                    moved[atom, axis] += displacement*step
                    stencil.append(moved)
        if self._pinned:
            # The original Fortran STOPs at the H2 midpoint. Check every displaced
            # geometry first, so an invalid stencil cannot exit the host interpreter.
            NaH2LEPS5G(symbols=self.symbols).evaluate(np.array(stencil))
        return stencil

    def _difference(self, frame):
        values = [self._matrix(point) for point in self._stencil(frame)]
        nstates = values[0].shape[0]
        terms = np.asarray(values[1:]).reshape(self.natoms, 3, 4, nstates, nstates)
        step = self.difference_step_bohr
        fd = (-terms[:, :, 0]+8*terms[:, :, 1]-8*terms[:, :, 2]+terms[:, :, 3])/(12*step)
        return values[0], fd.transpose(2, 3, 0, 1)

    def _verify(self, frame):
        """Check upstream analytic gradients once; decide the mode for ``auto``."""
        rng = np.random.default_rng(20261123)
        points = [frame] + [frame + self.check_displacement_bohr*rng.standard_normal(frame.shape)
                            for _ in range(self.check_points - 1)]
        worst_gradient = worst_matrix = 0.0
        passed, reason = True, None
        for point in points:
            reference, numerical = self._difference(point)
            try:
                matrix, analytic = self._matrix_gradient(point)
            except PotentialError as exc:
                passed, reason = False, str(exc)
                break
            matrix_scale = max(1.0, float(np.max(np.abs(reference))))
            matrix_error = float(np.max(np.abs(matrix - reference)))
            error = float(np.max(np.abs(analytic - numerical)))
            worst_matrix, worst_gradient = max(worst_matrix, matrix_error), max(worst_gradient, error)
            if matrix_error > 1e-10*matrix_scale:
                passed, reason = False, "ug returned a different matrix than u (an energy-only entry returns zeros)"
                break
            if error > self.gradient_check_tolerance*max(1.0, float(np.max(np.abs(numerical)))):
                bad = np.unravel_index(int(np.argmax(np.abs(analytic - numerical))), analytic.shape)
                passed = False
                reason = (f"analytic gradient of element ({bad[0]},{bad[1]}) differs from the five-point "
                          f"difference of u by {error:.3g} hartree/bohr")
                break
        self._state["check"] = {"passed": passed, "max_gradient_difference": worst_gradient,
                                "max_matrix_difference": worst_matrix, "reason": reason,
                                "points": len(points), "displacement_bohr": self.check_displacement_bohr,
                                "tolerance": self.gradient_check_tolerance,
                                "difference_step_bohr": self.difference_step_bohr}
        if not passed and self._state["requested"] == "analytic":
            raise BackendUnavailableError(f"ChemPotPy {self.surface} does not provide usable analytic DPEM "
                                          f"gradients ({reason}); use gradient='difference'")
        self._state["mode"] = "analytic" if passed else "difference"
        self._state["verified"] = True

    def matrices(self, q):
        """Diabatic matrices ``[B,S,S]`` (hartree) only: one ``u`` call per frame."""
        q = coordinates(q)
        if q.shape[1] != self.natoms:
            raise PotentialError(f"ChemPotPy {self.system} requires {self.natoms} atoms")
        if self._pinned:
            NaH2LEPS5G(symbols=self.symbols).evaluate(q)
        with _CHEMPOTPY_LOCK:
            return np.asarray([self._matrix(frame) for frame in q])

    def evaluate(self, q):
        from ..nonadiabatic import DiabaticResult, validate_diabatic_result

        q = coordinates(q)
        if q.shape[1] != self.natoms:
            raise PotentialError(f"ChemPotPy {self.system} requires {self.natoms} atoms")
        matrix, gradient = [], []
        with _CHEMPOTPY_LOCK:
            for frame in q:
                if not self._state["verified"]:
                    self._verify(frame)
                value, derivative = (self._matrix_gradient(frame) if self._state["mode"] == "analytic"
                                     else self._difference(frame))
                matrix.append(value)
                gradient.append(derivative)
        return validate_diabatic_result(DiabaticResult(np.asarray(matrix), np.asarray(gradient)), q.shape,
                                        self.nstates)
