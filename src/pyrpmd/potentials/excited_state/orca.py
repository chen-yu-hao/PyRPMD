"""ORCA TDDFT/TDA file interface (parser tested; executable not bundled)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base import BackendFailure, ExcitedStateStatus
from .external import FileProgramBackend, ParsedExternalResult, parse_gradient_blocks, parse_transition_properties


class ORCAExcitedStateBackend(FileProgramBackend):
    name = "orca"
    input_filename = "job.inp"
    output_filename = "job.out"
    wavefunction_filenames = ("job.gbw", "job.cis")

    DEFAULT_TEMPLATE = """! {method} {basis} TightSCF EnGrad Bohrs
%tddft
  nroots {nroots}
  tda {tda}
end
* xyz {charge} {multiplicity}
{geometry}
*
"""

    def __init__(
        self,
        symbols: Sequence[str],
        nstates: int,
        *,
        program_version: str,
        method: str = "B3LYP",
        basis: str = "def2-SVP",
        tda: bool = True,
        charge: int = 0,
        multiplicity: int = 1,
        command: Sequence[str] = ("orca", "{input}"),
        input_template: str = DEFAULT_TEMPLATE,
        **kwargs,
    ):
        if not isinstance(tda, (bool, np.bool_)):
            raise ValueError("tda must be boolean")
        self.tda = bool(tda)
        super().__init__(
            symbols,
            nstates,
            program_version=program_version,
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            command=command,
            input_template=input_template,
            **kwargs,
        )

    def render_input(self, geometry: np.ndarray) -> str:
        lines = [f"{symbol:2s} {x: .16f} {y: .16f} {z: .16f}" for symbol, (x, y, z) in zip(self.symbols, geometry)]
        return self.input_template.format(
            method=self.method,
            basis=self.basis,
            nroots=self.nstates - 1,
            tda=str(self.tda).lower(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            geometry="\n".join(lines),
        )

    def parse_output(self, text: str) -> ParsedExternalResult:
        lower = text.lower()
        if "scf not converged" in lower or "scf did not converge" in lower:
            raise BackendFailure(ExcitedStateStatus.SCF_NOT_CONVERGED, "ORCA SCF did not converge")
        if "orca terminated normally" not in lower:
            raise BackendFailure(ExcitedStateStatus.EXTERNAL_PROGRAM_FAILED, "ORCA output lacks normal termination")
        ground_matches = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+0-9.DEde]+)", text)
        if not ground_matches:
            raise BackendFailure(ExcitedStateStatus.OUTPUT_PARSE_ERROR, "ORCA ground-state energy not found")
        ground = float(ground_matches[-1].replace("D", "E"))
        excitations = {}
        for state, value in re.findall(r"STATE\s+(\d+)\s*:\s*E\s*=\s*([-+0-9.DEde]+)\s*au", text):
            excitations[int(state)] = float(value.replace("D", "E"))
        required = set(range(1, self.nstates))
        missing = sorted(required.difference(excitations))
        if missing:
            raise BackendFailure(
                ExcitedStateStatus.STATE_COUNT_MISMATCH,
                f"ORCA is missing excited-state roots " + ", ".join(map(str, missing)),
            )
        energies = np.array([ground] + [ground + excitations[state] for state in range(1, self.nstates)])
        gradients = parse_gradient_blocks(text, nstates=self.nstates, natoms=len(self.symbols))
        strengths, dipoles = parse_transition_properties(text, nstates=self.nstates)
        return ParsedExternalResult(energies, gradients, strengths, dipoles, {"parser": "orca-v1"})

    def scientific_identity(self) -> Mapping[str, Any]:
        identity = dict(super().scientific_identity())
        identity["tda"] = self.tda
        return identity
