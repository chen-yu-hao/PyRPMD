"""Public nonadiabatic mapping API with the auxiliary-core NRPMD adapter."""

from ._nonadiabatic_legacy import (
    DiabaticResult,
    DiagonalPotential,
    NRPMDState,
    _mint,
    _positive_integer,
    validate_diabatic_result,
)
from .nrpmd_auxiliary import (
    NRPMD_COMPATIBILITY,
    NRPMDAuxiliaryOperator,
    NRPMDDynamics,
)

# Preserve historical public module paths for introspection and pickle names.
for _kind in (
    DiabaticResult,
    DiagonalPotential,
    NRPMDState,
    NRPMDAuxiliaryOperator,
    NRPMDDynamics,
):
    _kind.__module__ = __name__

__all__ = [
    "DiabaticResult",
    "validate_diabatic_result",
    "DiagonalPotential",
    "NRPMDState",
    "NRPMDAuxiliaryOperator",
    "NRPMDDynamics",
    "NRPMD_COMPATIBILITY",
]
