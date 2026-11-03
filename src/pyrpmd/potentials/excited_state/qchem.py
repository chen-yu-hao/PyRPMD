"""Q-Chem TDDFT/TDA file interface (parser tested; executable not bundled)."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

from .base import BackendFailure, ExcitedStateStatus
from .external import FileProgramBackend, ParsedExternalResult, parse_gradient_blocks, parse_transition_properties

EV_TO_HARTREE = 1.0 / 27.211386245988


class QChemExcitedStateBackend(FileProgramBackend):
    name = "qchem"
    input_filename = "job.in"
    output_filename = "job.out"
    wavefunction_filenames = ("save/53.0", "save/132.0")

    DEFAULT_TEMPLATE = """$molecule
{charge} {multiplicity}
{geometry}
$end

$rem
JOBTYPE FORCE
METHOD {method}
BASIS {basis}
CIS_N_ROOTS {nroots}
RPA {rpa}
SCF_CONVERGENCE 10
INPUT_BOHR TRUE
$end
"""

    def __init__(
        self,
        symbols: Sequence[str],
        nstates: int,
        *,
        program_version: str,
        method: str = "B3LYP",
        basis: str = "6-31G*",
        tda: bool = True,
        charge: int = 0,
        multiplicity: int = 1,
        command: Sequence[str] = ("qchem", "{input}", "{output}"),
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
            rpa="FALSE" if self.tda else "TRUE",
            charge=self.charge,
            multiplicity=self.multiplicity,
            geometry="\n".join(lines),
        )

    def parse_output(self, text: str) -> ParsedExternalResult:
        lower = text.lower()
        if "scf failed to converge" in lower or "scf convergence failure" in lower:
            raise BackendFailure(ExcitedStateStatus.SCF_NOT_CONVERGED, "Q-Chem SCF did not converge")
        if "thank you very much for using q-chem" not in lower:
            raise BackendFailure(ExcitedStateStatus.EXTERNAL_PROGRAM_FAILED, "Q-Chem output lacks normal termination")
        matches = re.findall(r"Total energy in the final basis set\s*=\s*([-+0-9.DEde]+)", text)
        if not matches:
            raise BackendFailure(ExcitedStateStatus.OUTPUT_PARSE_ERROR, "Q-Chem ground-state energy not found")
        ground = float(matches[-1].replace("D", "E"))
        excitation = {}
        for state, value in re.findall(
            r"Excited state\s+(\d+)\s*:\s*excitation energy \(eV\)\s*=\s*([-+0-9.DEde]+)",
            text,
            flags=re.IGNORECASE,
        ):
            excitation[int(state)] = float(value.replace("D", "E")) * EV_TO_HARTREE
        required = set(range(1, self.nstates))
        missing = sorted(required.difference(excitation))
        if missing:
            raise BackendFailure(
                ExcitedStateStatus.STATE_COUNT_MISMATCH,
                f"Q-Chem is missing excited-state roots " + ", ".join(map(str, missing)),
            )
        energies = np.array([ground] + [ground + excitation[state] for state in range(1, self.nstates)])
        gradients = parse_gradient_blocks(text, nstates=self.nstates, natoms=len(self.symbols))
        strengths, dipoles = parse_transition_properties(text, nstates=self.nstates)
        return ParsedExternalResult(energies, gradients, strengths, dipoles, {"parser": "qchem-v1"})

    def scientific_identity(self) -> Mapping[str, Any]:
        identity = dict(super().scientific_identity())
        identity["tda"] = self.tda
        return identity
