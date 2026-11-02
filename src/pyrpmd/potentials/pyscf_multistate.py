"""Validated, real SA-CASSCF electronic data from the optional PySCF package.

This is an :class:`~pyrpmd.adiabatic.AdiabaticPythonPotential` callback, not
a scalar PES. Coordinates are bohr; energies include nuclear repulsion and
are hartree; gradients are **positive** hartree/bohr; ``d[i,j]`` means
``<i|grad j>`` in bohr**-1. Overlaps are ``<old_i|new_j>``.

The supported model is an equally weighted, all-electron, real singlet
SA-CASSCF calculation with analytic state-specific gradients and full
derivative couplings. The electron-translation-factor approximation is not
used: it would disagree with derivatives of the physical overlaps supplied
here. PySCF is imported only when the provider is constructed.

Every geometry is solved from an independent RHF guess. The bounded cache
stores untracked solutions, and the explicit previous-frame context supplies
all state permutations and phases. Interleaving trajectories therefore does
not change their electronic histories. No thread count or numerical precision
is changed by this module.
"""

from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
import hashlib
import importlib
import json
import math
import re
import threading

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from .base import coordinates


class PySCFStatus(IntEnum):
    """Per-frame failures; a failed frame never supplies usable dynamics data."""

    SUCCESS = 0
    SCF_NOT_CONVERGED = 1
    CASSCF_NOT_CONVERGED = 2
    RESPONSE_NOT_CONVERGED = 3
    DEGENERATE_STATES = 4
    STATE_TRACKING_FAILED = 5
    ELECTRONIC_STRUCTURE_ERROR = 6
    SPIN_CONTAMINATION = 7


class _CalculationFailure(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _immutable(value, dtype=float):
    array = np.array(value, dtype=dtype, order="C", copy=True)
    # Bytes backing also prevents setflags(write=True) from making a cached
    # wavefunction writable through a returned context.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _real_array(value, shape, label):
    """Validate the external quantum program before any NumPy assignment/cast."""
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise _CalculationFailure(
            PySCFStatus.ELECTRONIC_STRUCTURE_ERROR,
            f"PySCF {label} must be a finite real numerical array with shape {shape}",
        )
    return array


@dataclass(frozen=True)
class _Wavefunction:
    """A geometry's unphased solution, never modified by state tracking."""

    q: np.ndarray
    molecule: str
    mo: np.ndarray
    ci: np.ndarray
    energies: np.ndarray
    gradients: np.ndarray
    couplings: np.ndarray
    ncore: int
    ncas: int
    nelecas: tuple
    spin_square: tuple

    @property
    def mol(self):
        # A fresh Mole prevents a caller from mutating the cached AO basis or
        # geometry through a process-local electronic context.
        from pyscf import gto
        return gto.loads(self.molecule)


@dataclass(frozen=True)
class _FrameContext:
    """Process-local wavefunction plus the reported electronic gauge.

    ``order[i]`` is the raw root associated with reported state i; ``phases``
    contains real signs. Contexts intentionally contain no PySCF solver or
    RNG. They are opaque to the dynamics engine and are not checkpoints.
    """

    fingerprint: str
    q: np.ndarray
    wavefunction: _Wavefunction
    order: np.ndarray
    phases: np.ndarray


def _check_context(context, *, q=None, fingerprint=None):
    if not isinstance(context, _FrameContext):
        raise ValueError("previous_context must contain PySCF electronic frame contexts")
    wavefunction = context.wavefunction
    if not isinstance(wavefunction, _Wavefunction):
        raise ValueError("invalid PySCF wavefunction in previous_context")
    nstates = len(wavefunction.energies)
    order, phases = np.asarray(context.order), np.asarray(context.phases)
    if (order.shape != (nstates,) or order.dtype.kind not in "iu"
            or not np.array_equal(np.sort(order), np.arange(nstates))):
        raise ValueError("PySCF context state order must be a permutation")
    if (phases.shape != (nstates,) or phases.dtype.kind not in "fiu"
            or not np.all(np.isin(phases, (-1, 1)))):
        raise ValueError("PySCF context phases must be real signs")
    if not np.array_equal(context.q, wavefunction.q):
        raise ValueError("PySCF context geometry differs from its wavefunction")
    if q is not None and not np.array_equal(context.q, q):
        raise ValueError("previous_q differs from the PySCF previous_context geometry")
    if fingerprint is not None and context.fingerprint != fingerprint:
        raise ValueError("previous_context belongs to a different PySCF scientific identity")
    return context


@lru_cache(maxsize=32)
def _occupations(ncas, nelec):
    # PySCF's CI coefficient axes use precisely this determinant ordering.
    from pyscf.fci import cistring
    return _immutable(cistring.gen_occslst(range(ncas), nelec), dtype=int)


def _determinant_overlaps(overlap, occupations):
    """Exterior power of an orbital overlap, with bounded temporary storage."""
    nstrings, nelec = occupations.shape
    if nelec == 0:
        return np.ones((1, 1))
    result = np.empty((nstrings, nstrings))
    for index, occupied in enumerate(occupations):
        minors = overlap[occupied[None, :, None], occupations[:, None, :]]
        result[index] = np.linalg.det(minors)
    return result


def _raw_overlap(old, new, *, max_memory_mb):
    """Exact core+active many-electron overlap without a full-space FCI tensor.

    A Schur complement eliminates the doubly occupied orbitals. Its squared
    core determinant and active-space determinant minors retain core-active
    cross terms, unlike multiplying independent bare core/active overlaps.
    A nearly singular core block is explicitly rejected.
    """
    from pyscf import gto

    if (old.ncore != new.ncore or old.ncas != new.ncas
            or old.nelecas != new.nelecas or old.ci.shape != new.ci.shape):
        raise ValueError("overlaps require identical CASSCF core and active spaces")
    nocc = old.ncore + old.ncas
    ao_overlap = gto.intor_cross("int1e_ovlp", old.mol, new.mol)
    orbital = old.mo[:, :nocc].T @ ao_overlap @ new.mo[:, :nocc]
    ncore = old.ncore
    prefactor = 1.0
    if ncore:
        core = orbital[:ncore, :ncore]
        if not np.isfinite(core).all() or np.linalg.cond(core) > 1e12:
            raise _CalculationFailure(
                PySCFStatus.STATE_TRACKING_FAILED,
                "inactive-orbital overlap is singular; reduce the geometry displacement",
            )
        prefactor = float(np.linalg.det(core)) ** 2
        active = orbital[ncore:, ncore:] - orbital[ncore:, :ncore] @ np.linalg.solve(
            core, orbital[:ncore, ncore:]
        )
    else:
        active = orbital

    na, nb = old.ci.shape[1:]
    # Include determinant matrices, row-batched minors, and transformed CI.
    nea, neb = old.nelecas
    estimated_bytes = 8 * (na*na + nb*nb + max(na*nea*nea, nb*neb*neb)
                           + 2 * new.ci.size)
    if estimated_bytes > max_memory_mb * 1e6:
        raise _CalculationFailure(
            PySCFStatus.STATE_TRACKING_FAILED,
            "exact CASSCF overlap exceeds max_memory_mb; increase its explicit limit",
        )
    da = _determinant_overlaps(active, _occupations(old.ncas, nea))
    db = da if nea == neb else _determinant_overlaps(active, _occupations(old.ncas, neb))
    transformed = np.einsum("ac,scd,bd->sab", da, new.ci, db, optimize=True)
    overlap = prefactor * (old.ci.reshape(len(old.ci), -1)
                           @ transformed.reshape(len(new.ci), -1).T)
    if not np.isfinite(overlap).all():
        raise _CalculationFailure(PySCFStatus.STATE_TRACKING_FAILED,
                                  "nonfinite CASSCF wavefunction overlap")
    return overlap


def casscf_overlap(old_context, new_context, *, max_memory_mb=2000):
    """Return ``<old_i|new_j>`` including inactive orbitals and tracked signs.

    The inputs are individual entries of an evaluation's ``context`` tuple.
    This helper is useful for independent overlap finite differences. It
    returns the physical overlap, without projecting it onto a unitary matrix.
    """
    old = _check_context(old_context)
    new = _check_context(new_context, fingerprint=old.fingerprint)
    limit = _positive(max_memory_mb, "max_memory_mb")
    raw = _raw_overlap(old.wavefunction, new.wavefunction, max_memory_mb=limit)
    return raw[np.ix_(old.order, new.order)] * old.phases[:, None] * new.phases[None, :]


class PySCFMultistatePotential:
    """Optional CPU SA-CASSCF callback for on-the-fly nonadiabatic dynamics.

    Wrap this instance in ``AdiabaticPythonPotential(provider)``. ``symbols``
    fixes atom order; every evaluation accepts float64 ``[B,N,3]`` bohr.
    ``state_order='energy'`` keeps ascending energies, as required by RPSH.
    ``'overlap'`` follows the maximum-overlap permutation and is intended for
    QD-NRPMD; its energies need not remain sorted.

    Only equally weighted singlets are supported. Nonzero spin, unequal
    weights, TDDFT, ECPs, SOC, frozen orbitals and GPU execution are not
    substituted automatically. Near-degenerate retained states fail before
    dividing by an energy gap. PySCF/SCF/CASSCF/response failures have explicit
    :class:`PySCFStatus` values and per-frame messages.

    The provider honors the caller's existing OpenMP/BLAS thread configuration.
    Set these environment variables before importing PySCF when reproducible
    single-thread timing is wanted. Cache entries are exact-geometry results;
    ``cache_size=0`` disables the cache without changing the model.
    """

    def __init__(
        self, symbols, *, basis="sto-3g", ncas=2, nelecas=2, nstates=2,
        charge=0, spin=0, state_order="energy", scf_conv_tol=1e-12,
        casscf_conv_tol=1e-10, casscf_conv_tol_grad=1e-6,
        fci_conv_tol=1e-12, response_atol=1e-10, response_rtol=1e-8,
        max_scf_cycles=100, max_casscf_cycles=100, max_fci_cycles=200,
        max_response_cycles=100, gap_tolerance=1e-7, spin_tolerance=1e-6,
        spin_penalty=1.0, phase_overlap_tolerance=1e-8, cache_size=16,
        max_memory_mb=2000,
    ):
        try:
            self._pyscf = importlib.import_module("pyscf")
            self._nac = importlib.import_module("pyscf.nac.sacasscf")
        except ImportError as exc:
            raise ImportError(
                "PySCFMultistatePotential requires PySCF with analytic SA-CASSCF NACs; "
                "install the optional 'pyscf' package"
            ) from exc
        if not callable(getattr(self._nac, "NonAdiabaticCouplings", None)):
            raise ImportError("this PySCF version lacks analytic SA-CASSCF NACs")
        if str(self._pyscf.__version__).split(".")[0] != "2":
            raise ImportError("PySCF multistate bridge requires PySCF 2.x; "
                              "a new major release requires an NAC convention review")
        from pyscf import gto

        if isinstance(symbols, (str, bytes)):
            raise ValueError("symbols must be a nonempty sequence of element symbols")
        try:
            symbols = tuple(symbols)
        except TypeError as exc:
            raise ValueError("symbols must be a nonempty sequence of element symbols") from exc
        if not symbols or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z][a-z]?", s)
                              for s in symbols):
            raise ValueError("symbols must contain standard element symbols, without ghosts")
        try:
            nuclear_charges = [gto.charge(s) for s in symbols]
        except (KeyError, RuntimeError) as exc:
            raise ValueError("symbols contains an unknown element") from exc
        if any(z <= 0 for z in nuclear_charges):
            raise ValueError("ghost atoms are not supported")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("basis must be a nonempty PySCF basis name")
        try:
            # PySCF accepts basis filenames as well as built-in aliases. Freeze
            # the resolved numerical basis so later external file changes can
            # neither alter a solve nor evade its scientific fingerprint.
            resolved_basis = {symbol: gto.basis.load(basis, symbol)
                              for symbol in dict.fromkeys(symbols)}
            self._basis_json = json.dumps(resolved_basis, sort_keys=True,
                                          separators=(",", ":"), allow_nan=False)
        except Exception as exc:
            raise ValueError(f"cannot resolve the all-electron PySCF basis {basis!r}: {exc}") from exc
        if isinstance(charge, (bool, np.bool_)) or not isinstance(charge, (int, np.integer)):
            raise ValueError("charge must be an integer")
        spin = _integer(spin, "spin")
        if spin != 0:
            raise ValueError("PySCF multistate bridge currently supports spin=0 singlets only")
        ncas = _integer(ncas, "ncas", 1)
        nstates = _integer(nstates, "nstates", 2)
        if isinstance(nelecas, (int, np.integer)) and not isinstance(nelecas, (bool, np.bool_)):
            active_electrons = _integer(nelecas, "nelecas", 2)
            if active_electrons % 2:
                raise ValueError("singlet nelecas must be even")
            nelecas = (active_electrons // 2,) * 2
        else:
            try:
                nelecas = tuple(nelecas)
            except TypeError as exc:
                raise ValueError("nelecas must be even or an equal (alpha, beta) pair") from exc
            if len(nelecas) != 2:
                raise ValueError("nelecas must be an (alpha, beta) pair")
            nelecas = tuple(_integer(n, "nelecas", 1) for n in nelecas)
        if nelecas[0] != nelecas[1] or max(nelecas) > ncas:
            raise ValueError("singlet active electrons must fit equally in ncas orbitals")
        nalpha = nelecas[0]
        singlet_dimension = math.comb(ncas, nalpha)**2
        if nalpha < ncas:
            singlet_dimension -= math.comb(ncas, nalpha+1) * math.comb(ncas, nalpha-1)
        if nstates > singlet_dimension:
            raise ValueError("nstates exceeds the number of singlet active-space states")
        nelectron = sum(nuclear_charges) - int(charge)
        ncore_electrons = nelectron - sum(nelecas)
        if ncore_electrons < 0 or ncore_electrons % 2:
            raise ValueError("molecule electron count is incompatible with the singlet active space")
        if state_order not in ("energy", "overlap"):
            raise ValueError("state_order must be 'energy' or 'overlap'")

        self._settings = {
            "symbols": list(symbols), "basis": basis, "ncas": ncas,
            "nelecas": list(nelecas), "nstates": nstates, "charge": int(charge),
            "spin": spin, "state_order": state_order,
        }
        for name, value in {
            "scf_conv_tol": scf_conv_tol, "casscf_conv_tol": casscf_conv_tol,
            "casscf_conv_tol_grad": casscf_conv_tol_grad, "fci_conv_tol": fci_conv_tol,
            "response_atol": response_atol, "response_rtol": response_rtol,
            "gap_tolerance": gap_tolerance, "spin_tolerance": spin_tolerance,
            "spin_penalty": spin_penalty, "phase_overlap_tolerance": phase_overlap_tolerance,
            "max_memory_mb": max_memory_mb,
        }.items():
            self._settings[name] = _positive(value, name)
        if self._settings["phase_overlap_tolerance"] >= 1:
            raise ValueError("phase_overlap_tolerance must be below one")
        for name, value in {
            "max_scf_cycles": max_scf_cycles, "max_casscf_cycles": max_casscf_cycles,
            "max_fci_cycles": max_fci_cycles, "max_response_cycles": max_response_cycles,
        }.items():
            self._settings[name] = _integer(value, name, 1)
        self._settings["cache_size"] = _integer(cache_size, "cache_size")
        self._identity = {
            "provider": "pyscf_sacasscf", "implementation": 1,
            "pyscf_version": self._pyscf.__version__, "numpy_version": np.__version__,
            "scipy_version": scipy.__version__, "units": "atomic",
            "dtype": "float64", "device": "cpu", "weights": [1/nstates] * nstates,
            "use_etfs": False, "natorb": True, "symmetry": False, "cart": False,
            "resolved_basis": json.loads(self._basis_json),
            **self._settings,
        }
        self._fingerprint = hashlib.sha256(
            json.dumps(self._identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._counts = {"hits": 0, "misses": 0, "evictions": 0, "solves": 0, "failures": 0}
        self.last_diagnostics = ()
        self.metadata = {
            "provider": "pyscf_sacasscf", "execution_device": "cpu",
            "thread_safety": "context_serial", "precision": "float64",
            "state_order": state_order, "units": "atomic", "native_batch": False,
            "requires_atomic_units": True,
        }

    @property
    def nstates(self):
        return self._settings["nstates"]

    @property
    def symbols(self):
        return tuple(self._settings["symbols"])

    def to_dict(self):
        """Detached, JSON-compatible scientific identity and explicit options."""
        return json.loads(json.dumps(self._identity))

    def cache_info(self):
        """Return detached counters; they do not affect the scientific identity."""
        with self._lock:
            return {**self._counts, "size": len(self._cache),
                    "maxsize": self._settings["cache_size"]}

    def clear_cache(self):
        """Drop cached solutions; existing explicit frame contexts remain usable."""
        with self._lock:
            self._cache.clear()

    def close(self):
        self.clear_cache()

    def export_context(self, context):
        """Export explicit electronic history as a versioned JSON-safe snapshot.

        This contains no pickle, executable callbacks, solver instances, or
        global cache state. It preserves MO/CI arrays and reported phases and
        permutations exactly through ordinary JSON float64 round-tripping.
        It is **not** a dynamics checkpoint: callers must separately preserve
        nuclear/electronic dynamics and RNG state and write their complete
        checkpoint atomically.
        """
        if not isinstance(context, (tuple, list)) or not context:
            raise ValueError("context must be a nonempty batch of PySCF frame contexts")
        frames = []
        for item in context:
            frame = _check_context(item, fingerprint=self._fingerprint)
            wavefunction = frame.wavefunction
            frames.append({
                "q": frame.q.tolist(), "order": frame.order.tolist(),
                "phases": frame.phases.tolist(), "mo": wavefunction.mo.tolist(),
                "ci": wavefunction.ci.tolist(), "energies": wavefunction.energies.tolist(),
                "gradients": wavefunction.gradients.tolist(),
                "couplings": wavefunction.couplings.tolist(),
            })
        payload = {"schema": "pyrpmd.pyscf_multistate.context", "version": 1,
                   "scientific_identity": self.to_dict(), "frames": frames}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {**payload, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}

    def import_context(self, snapshot):
        """Validate and restore an :meth:`export_context` snapshot.

        A fresh provider with the identical scientific identity can restore
        it. Hashes detect accidental data corruption; structural, AO-metric,
        CI normalization, spin, gap, and antisymmetry checks reject invalid
        electronic history. Import does not populate or change the cache.
        """
        from pyscf import fci

        required = {"schema", "version", "scientific_identity", "frames", "sha256"}
        if not isinstance(snapshot, dict) or set(snapshot) != required:
            raise ValueError("invalid PySCF context snapshot fields")
        if (snapshot["schema"] != "pyrpmd.pyscf_multistate.context"
                or type(snapshot["version"]) is not int or snapshot["version"] != 1):
            raise ValueError("unsupported PySCF context snapshot schema or version")
        if snapshot["scientific_identity"] != self.to_dict():
            raise ValueError("PySCF context snapshot scientific identity mismatch")
        payload = {key: value for key, value in snapshot.items() if key != "sha256"}
        try:
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("PySCF context snapshot must contain finite JSON data") from exc
        if hashlib.sha256(encoded.encode()).hexdigest() != snapshot["sha256"]:
            raise ValueError("PySCF context snapshot checksum mismatch")
        if not isinstance(snapshot["frames"], list) or not snapshot["frames"]:
            raise ValueError("PySCF context snapshot must contain frames")

        def read_array(value, shape, label, dtype=float):
            array = np.asarray(value)
            if (array.shape != shape or array.dtype.kind not in "fiu"
                    or not np.isfinite(array).all()):
                raise ValueError(f"invalid {label} in PySCF context snapshot")
            if dtype is int and (array.dtype.kind not in "iu"):
                raise ValueError(f"{label} must be integer in PySCF context snapshot")
            return _immutable(array, dtype)

        options = self._settings
        natoms, nstates, ncas = len(self.symbols), self.nstates, options["ncas"]
        nelecas = tuple(options["nelecas"])
        ci_shape = (nstates, math.comb(ncas, nelecas[0]), math.comb(ncas, nelecas[1]))
        restored = []
        frame_fields = {"q", "order", "phases", "mo", "ci", "energies", "gradients", "couplings"}
        for raw in snapshot["frames"]:
            if not isinstance(raw, dict) or set(raw) != frame_fields:
                raise ValueError("invalid frame fields in PySCF context snapshot")
            q = read_array(raw["q"], (natoms, 3), "geometry")
            mol = self._molecule(q)
            ncore = (mol.nelectron-sum(nelecas)) // 2
            mo_raw = np.asarray(raw["mo"])
            if (mo_raw.ndim != 2 or mo_raw.shape[0] != mol.nao_nr()
                    or not ncore+ncas <= mo_raw.shape[1] <= mol.nao_nr()):
                raise ValueError("invalid MO shape in PySCF context snapshot")
            mo = read_array(raw["mo"], mo_raw.shape, "MO coefficients")
            orbital_metric = mo.T @ mol.intor_symmetric("int1e_ovlp") @ mo
            if not np.allclose(orbital_metric, np.eye(mo.shape[1]), rtol=0, atol=1e-8):
                raise ValueError("context snapshot MOs are not orthonormal in the AO metric")
            ci = read_array(raw["ci"], ci_shape, "CI coefficients")
            ci_flat = ci.reshape(nstates, -1)
            if not np.allclose(ci_flat @ ci_flat.T, np.eye(nstates), rtol=0, atol=1e-8):
                raise ValueError("context snapshot CI roots are not orthonormal")
            spin_square = tuple(float(fci.spin_op.spin_square(c, ncas, nelecas)[0]) for c in ci)
            if max(abs(value) for value in spin_square) > options["spin_tolerance"]:
                raise ValueError("context snapshot roots are not singlets")
            energies = read_array(raw["energies"], (nstates,), "energies")
            if np.min(np.diff(np.sort(energies))) <= options["gap_tolerance"]:
                raise ValueError("context snapshot has unsupported degenerate energies")
            gradients = read_array(raw["gradients"], (nstates, natoms, 3), "gradients")
            couplings = read_array(raw["couplings"], (nstates, nstates, natoms, 3), "couplings")
            if not np.allclose(couplings, -couplings.swapaxes(0, 1), rtol=0, atol=1e-10):
                raise ValueError("context snapshot NACs are not antisymmetric")
            order = read_array(raw["order"], (nstates,), "state order", int)
            phases = read_array(raw["phases"], (nstates,), "phases")
            wavefunction = _Wavefunction(q, mol.dumps(), mo, ci, energies, gradients, couplings,
                                        ncore, ncas, nelecas, spin_square)
            context = _FrameContext(self._fingerprint, q, wavefunction, order, phases)
            restored.append(_check_context(context))
        return tuple(restored)

    def _molecule(self, q):
        from pyscf import gto

        options = self._settings
        return gto.M(
            atom=list(zip(self.symbols, q.tolist())), unit="Bohr", basis=json.loads(self._basis_json),
            charge=options["charge"], spin=0, symmetry=False, cart=False, verbose=0,
            max_memory=options["max_memory_mb"],
        )

    def _solve(self, q):
        from pyscf import fci, mcscf, scf

        options = self._settings
        self._counts["solves"] += 1
        mol = self._molecule(q)
        ncore = (mol.nelectron - sum(options["nelecas"])) // 2
        if ncore + options["ncas"] > mol.nao_nr():
            raise _CalculationFailure(PySCFStatus.ELECTRONIC_STRUCTURE_ERROR,
                                      "basis has too few orbitals for the requested active space")
        mf = scf.RHF(mol)
        mf.chkfile = None
        mf.conv_tol = options["scf_conv_tol"]
        mf.conv_tol_grad = math.sqrt(options["scf_conv_tol"])
        mf.max_cycle = options["max_scf_cycles"]
        mf.kernel()
        if not mf.converged:
            raise _CalculationFailure(PySCFStatus.SCF_NOT_CONVERGED,
                                      "PySCF RHF did not converge")
        mc = mcscf.CASSCF(mf, options["ncas"], tuple(options["nelecas"]))
        mc.chkfile = None
        mc.fix_spin_(ss=0, shift=options["spin_penalty"])
        mc = mc.state_average([1/self.nstates] * self.nstates)
        mc.natorb = True
        mc.conv_tol = options["casscf_conv_tol"]
        mc.conv_tol_grad = options["casscf_conv_tol_grad"]
        mc.max_cycle_macro = options["max_casscf_cycles"]
        mc.fcisolver.conv_tol = options["fci_conv_tol"]
        mc.fcisolver.max_cycle = options["max_fci_cycles"]
        mc.kernel()
        if not mc.converged or not np.all(getattr(mc.fcisolver, "converged", False)):
            raise _CalculationFailure(PySCFStatus.CASSCF_NOT_CONVERGED,
                                      "PySCF SA-CASSCF or its FCI roots did not converge")
        if (mc.ncore != ncore or mc.ncas != options["ncas"]
                or tuple(mc.nelecas) != tuple(options["nelecas"])):
            raise _CalculationFailure(PySCFStatus.ELECTRONIC_STRUCTURE_ERROR,
                                      "PySCF changed the requested core or active space")
        energies = _real_array(mc.e_states, (self.nstates,), "state energies")
        ci_shape = (self.nstates, math.comb(mc.ncas, mc.nelecas[0]),
                    math.comb(mc.ncas, mc.nelecas[1]))
        ci = _real_array(mc.ci, ci_shape, "CI coefficients")
        mo = np.asarray(mc.mo_coeff)
        if (mo.ndim != 2 or mo.shape[0] != mol.nao_nr()
                or not ncore+mc.ncas <= mo.shape[1] <= mol.nao_nr()):
            raise _CalculationFailure(PySCFStatus.ELECTRONIC_STRUCTURE_ERROR,
                                      "PySCF returned an invalid MO coefficient shape")
        mo = _real_array(mo, mo.shape, "MO coefficients")
        ci_flat = ci.reshape(self.nstates, -1)
        if not np.allclose(ci_flat @ ci_flat.T, np.eye(self.nstates), rtol=0, atol=1e-8):
            raise _CalculationFailure(PySCFStatus.ELECTRONIC_STRUCTURE_ERROR,
                                      "PySCF CI roots are incomplete or not orthonormal")
        orbital_metric = mo.T @ mol.intor_symmetric("int1e_ovlp") @ mo
        if not np.allclose(orbital_metric, np.eye(mo.shape[1]), rtol=0, atol=1e-8):
            raise _CalculationFailure(PySCFStatus.ELECTRONIC_STRUCTURE_ERROR,
                                      "PySCF MOs are not orthonormal in the AO metric")
        spin_square = tuple(float(fci.spin_op.spin_square(c, mc.ncas, mc.nelecas)[0]) for c in ci)
        if max(abs(value) for value in spin_square) > options["spin_tolerance"]:
            raise _CalculationFailure(PySCFStatus.SPIN_CONTAMINATION,
                                      f"PySCF roots are not pure singlets: S^2={spin_square}")
        gaps = np.diff(np.sort(energies))
        if np.min(gaps) <= options["gap_tolerance"]:
            raise _CalculationFailure(
                PySCFStatus.DEGENERATE_STATES,
                f"retained energy gap {np.min(gaps):.6g} hartree is at or below gap_tolerance; "
                "degenerate-subspace derivatives are not supported",
            )

        eris = mc.ao2mo(mc.mo_coeff)
        gradient_method = mc.nuc_grad_method()
        coupling_method = self._nac.NonAdiabaticCouplings(mc, use_etfs=False)
        for method in (gradient_method, coupling_method):
            method.conv_atol = options["response_atol"]
            method.conv_rtol = options["response_rtol"]
            method.max_cycle = options["max_response_cycles"]
        gradients = np.empty((self.nstates, len(self.symbols), 3))
        couplings = np.zeros((self.nstates, self.nstates, len(self.symbols), 3))
        for state in range(self.nstates):
            raw_gradient = gradient_method.kernel(state=state, eris=eris)
            if not gradient_method.converged:
                raise _CalculationFailure(
                    PySCFStatus.RESPONSE_NOT_CONVERGED,
                    f"SA-CASSCF gradient response did not converge for state {state}",
                )
            gradients[state] = _real_array(raw_gradient, (len(self.symbols), 3),
                                           f"gradient for state {state}")
        for bra in range(self.nstates):
            for ket in range(bra+1, self.nstates):
                # PySCF 2.x's actual sign is <state[0]|grad state[1]>,
                # opposite to its docstring: its Hamiltonian response is
                # divided by e_states[state[1]]-e_states[state[0]]. Physical
                # AO+core+CI overlap derivatives verify this sign (2.14.0).
                # Request the reversed pair and negate it: the first index
                # is then always >0, also avoiding 2.14's offs_bra conditional
                # on ket>0 in get_wfn_response. No trajectory-dependent sign
                # inference or finite-difference derivative is used here.
                raw_coupling = coupling_method.kernel(state=(ket, bra), eris=eris, use_etfs=False)
                if not coupling_method.converged:
                    raise _CalculationFailure(
                        PySCFStatus.RESPONSE_NOT_CONVERGED,
                        f"SA-CASSCF NAC response did not converge for states {bra},{ket}",
                    )
                value = -_real_array(raw_coupling, (len(self.symbols), 3),
                                     f"NAC for states {bra},{ket}")
                couplings[bra, ket] = value
                couplings[ket, bra] = -value
        if not np.isfinite(gradients).all() or not np.isfinite(couplings).all():
            raise _CalculationFailure(PySCFStatus.ELECTRONIC_STRUCTURE_ERROR,
                                      "PySCF returned nonfinite analytic derivatives")
        return _Wavefunction(
            _immutable(q), mol.dumps(), _immutable(mo), _immutable(ci),
            _immutable(energies), _immutable(gradients), _immutable(couplings),
            int(mc.ncore), int(mc.ncas), tuple(mc.nelecas), spin_square,
        )

    def _solution(self, q):
        key = q.tobytes(order="C")
        if key in self._cache:
            self._counts["hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._counts["misses"] += 1
        result = self._solve(q)
        if self._settings["cache_size"]:
            self._cache[key] = result
            while len(self._cache) > self._settings["cache_size"]:
                self._cache.popitem(last=False)
                self._counts["evictions"] += 1
        return result

    def _track(self, solution, previous):
        order = np.argsort(solution.energies, kind="stable")
        overlap = None
        if previous is None:
            flat = solution.ci[order].reshape(self.nstates, -1)
            phases = np.sign(flat[np.arange(self.nstates), np.argmax(np.abs(flat), axis=1)])
        else:
            raw = _raw_overlap(previous.wavefunction, solution,
                               max_memory_mb=self._settings["max_memory_mb"])
            raw = previous.phases[:, None] * raw[previous.order]
            if self._settings["state_order"] == "overlap":
                rows, columns = linear_sum_assignment(-np.abs(raw))
                order = columns[np.argsort(rows)]
            diagonal = raw[np.arange(self.nstates), order]
            if np.any(np.abs(diagonal) <= self._settings["phase_overlap_tolerance"]):
                raise _CalculationFailure(
                    PySCFStatus.STATE_TRACKING_FAILED,
                    "state overlap is too small for unambiguous phase tracking; "
                    "reduce displacement or use overlap ordering with QD-NRPMD",
                )
            phases = np.sign(diagonal)
            overlap = raw[:, order] * phases[None, :]
        context = _FrameContext(self._fingerprint, solution.q, solution,
                                _immutable(order, int), _immutable(phases))
        coupling = solution.couplings[order][:, order] * phases[:, None, None, None] \
            * phases[None, :, None, None]
        return solution.energies[order], solution.gradients[order], coupling, overlap, context

    def evaluate(self, q, *, previous_q=None, previous_context=None):
        """Evaluate a batch with explicit previous geometries and wavefunctions.

        Input/context mistakes raise ``ValueError`` immediately. Quantum
        chemistry failures are reported for each frame through ``status`` and
        ``messages``; the standard adiabatic adapter raises ``PotentialError``
        on any failed frame. Successful siblings remain diagnosable, but no
        dynamics step may consume a partially failed batch.
        """
        from ..adiabatic import AdiabaticResult

        q = coordinates(q)
        if q.shape[1] != len(self.symbols):
            raise ValueError("PySCF coordinates do not match the configured symbols")
        if (previous_q is None) != (previous_context is None):
            raise ValueError("previous_q and previous_context must be supplied together")
        b = len(q)
        previous = [None] * b
        if previous_q is not None:
            old_q = coordinates(previous_q)
            if old_q.shape != q.shape:
                raise ValueError("previous_q must have the same batch and atom shape")
            if not isinstance(previous_context, (list, tuple)) or len(previous_context) != b:
                raise ValueError("previous_context must contain one frame context per batch entry")
            previous = [_check_context(item, q=old_q[index], fingerprint=self._fingerprint)
                        for index, item in enumerate(previous_context)]
        energies = np.full((b, self.nstates), np.nan)
        gradients = np.full((b, self.nstates, len(self.symbols), 3), np.nan)
        couplings = np.full((b, self.nstates, self.nstates, len(self.symbols), 3), np.nan)
        overlaps = None if previous_q is None else np.full((b, self.nstates, self.nstates), np.nan)
        status = np.zeros(b, dtype=int)
        contexts, messages, diagnostics = [], [], []
        with self._lock:
            for index, geometry in enumerate(q):
                try:
                    solution = self._solution(geometry)
                    e, g, d, overlap, context = self._track(solution, previous[index])
                    energies[index], gradients[index], couplings[index] = e, g, d
                    if overlaps is not None:
                        overlaps[index] = overlap
                    contexts.append(context)
                    messages.append("")
                    diagnostics.append({
                        "status": int(PySCFStatus.SUCCESS), "scf_converged": True,
                        "casscf_converged": True, "response_converged": True,
                        "spin_square": list(solution.spin_square),
                        "minimum_gap": float(np.min(np.diff(np.sort(e)))),
                        "state_order": context.order.tolist(), "phases": context.phases.tolist(),
                    })
                except Exception as exc:
                    code = exc.status if isinstance(exc, _CalculationFailure) \
                        else PySCFStatus.ELECTRONIC_STRUCTURE_ERROR
                    status[index] = int(code)
                    message = f"{type(exc).__name__}: {exc}"
                    contexts.append(None)
                    messages.append(message)
                    diagnostics.append({"status": int(code), "message": message})
                    self._counts["failures"] += 1
            self.last_diagnostics = tuple(diagnostics)
        return AdiabaticResult(energies, gradients, couplings, overlaps, status,
                               tuple(contexts), tuple(messages))


__all__ = ["PySCFMultistatePotential", "PySCFStatus", "casscf_overlap"]
