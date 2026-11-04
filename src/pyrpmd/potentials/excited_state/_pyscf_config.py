"""Configuration and molecule construction for the PySCF TDA/TDDFT backend."""

from __future__ import annotations

import copy
import importlib
from typing import Sequence

import numpy as np

from ._util import canonical_json, finite_positive, positive_integer


class _PySCFConfigurationMixin:
    """Restricted closed-shell PySCF TDA/TDDFT backend.

    ``nstates`` includes the ground state.  The first frame obtains full NAC
    vectors from centered finite differences of wavefunction overlaps when
    ``initial_nac='finite_difference'``.  Later frames can use the provider's
    HST/NPI path coupling.  No zero-coupling fallback is used.
    """

    name = "pyscf-tddft"

    def __init__(
        self,
        symbols: Sequence[str],
        nstates: int,
        *,
        basis: str = "sto-3g",
        ecp: str | None = None,
        xc: str | None = "b3lyp",
        tda: bool = True,
        charge: int = 0,
        spin: int = 0,
        scf_conv_tol: float = 1e-10,
        td_conv_tol: float = 1e-8,
        max_scf_cycles: int = 100,
        max_td_cycles: int = 100,
        grid_level: int = 3,
        max_memory_mb: int = 2000,
        transition_properties: bool = True,
        initial_nac: str = "finite_difference",
        nac_finite_difference_step: float = 2e-4,
        finite_difference_subspace_tolerance: float = 0.1,
        max_determinant_pairs: int = 2_000_000,
    ):
        try:
            self._pyscf = importlib.import_module("pyscf")
            from pyscf import gto
        except ImportError as exc:
            raise ImportError(
                "PySCFTDDFTBackend requires the optional PySCF package; install PyRPMD with '.[pyscf]'"
            ) from exc
        self.program_version = str(self._pyscf.__version__)
        if self.program_version.split(".", 1)[0] != "2":
            raise ImportError(
                "PySCF TDA/TDDFT backend requires PySCF 2.x; a new major release "
                "requires an overlap and gradient convention review"
            )
        try:
            self.symbols = tuple(symbols)
        except TypeError as exc:
            raise ValueError("symbols must be a nonempty sequence") from exc
        if not self.symbols or any(not isinstance(symbol, str) or not symbol for symbol in self.symbols):
            raise ValueError("symbols must be nonempty element symbols")
        self.nstates = positive_integer(nstates, "nstates", minimum=2)
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError("basis must be a nonempty PySCF basis name")
        if ecp is not None and (not isinstance(ecp, str) or not ecp.strip()):
            raise ValueError("ecp must be a nonempty PySCF ECP name or None")
        if xc is not None and (not isinstance(xc, str) or not xc.strip()):
            raise ValueError("xc must be a nonempty PySCF functional name or None")
        if not isinstance(tda, (bool, np.bool_)):
            raise ValueError("tda must be boolean")
        if isinstance(charge, (bool, np.bool_)) or not isinstance(charge, (int, np.integer)):
            raise ValueError("charge must be an integer")
        if isinstance(spin, (bool, np.bool_)) or not isinstance(spin, (int, np.integer)):
            raise ValueError("spin must be an integer")
        if int(spin) != 0:
            raise ValueError("PySCF TDDFT backend currently supports closed-shell spin=0 only")
        if not isinstance(transition_properties, (bool, np.bool_)):
            raise ValueError("transition_properties must be boolean")
        if initial_nac not in ("finite_difference", "unavailable"):
            raise ValueError("initial_nac must be 'finite_difference' or 'unavailable'")
        self.basis = basis.strip()
        self.ecp = None if ecp is None else ecp.strip()
        self.xc = None if xc is None else xc.strip()
        self.tda = bool(tda)
        self.charge = int(charge)
        self.spin = int(spin)
        self.scf_conv_tol = finite_positive(scf_conv_tol, "scf_conv_tol")
        self.td_conv_tol = finite_positive(td_conv_tol, "td_conv_tol")
        self.max_scf_cycles = positive_integer(max_scf_cycles, "max_scf_cycles")
        self.max_td_cycles = positive_integer(max_td_cycles, "max_td_cycles")
        self.grid_level = positive_integer(grid_level, "grid_level", minimum=0)
        self.max_memory_mb = positive_integer(max_memory_mb, "max_memory_mb")
        self.transition_properties = bool(transition_properties)
        self.initial_nac = initial_nac
        self.nac_finite_difference_step = finite_positive(
            nac_finite_difference_step, "nac_finite_difference_step"
        )
        self.finite_difference_subspace_tolerance = finite_positive(
            finite_difference_subspace_tolerance,
            "finite_difference_subspace_tolerance",
        )
        if self.finite_difference_subspace_tolerance >= 1:
            raise ValueError("finite_difference_subspace_tolerance must be below one")
        self.max_determinant_pairs = positive_integer(
            max_determinant_pairs, "max_determinant_pairs"
        )
        try:
            self._resolved_basis = {
                symbol: gto.basis.load(self.basis, symbol)
                for symbol in dict.fromkeys(self.symbols)
            }
            self._basis_json = canonical_json(self._resolved_basis)
            if self.ecp is None:
                self._resolved_ecp = None
                self._ecp_json = None
            else:
                self._resolved_ecp = {
                    symbol: gto.basis.load_ecp(self.ecp, symbol)
                    for symbol in dict.fromkeys(self.symbols)
                }
                self._ecp_json = canonical_json(self._resolved_ecp)
        except Exception as exc:
            raise ValueError(f"cannot resolve PySCF basis/ECP: {exc}") from exc

    def _molecule(self, geometry: np.ndarray):
        from pyscf import gto

        return gto.M(
            atom=[(symbol, tuple(vector)) for symbol, vector in zip(self.symbols, geometry)],
            unit="Bohr",
            basis=copy.deepcopy(self._resolved_basis),
            ecp=None if self._resolved_ecp is None else copy.deepcopy(self._resolved_ecp),
            charge=self.charge,
            spin=self.spin,
            verbose=0,
            max_memory=self.max_memory_mb,
        )
