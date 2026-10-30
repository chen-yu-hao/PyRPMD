"""Electrode models built on PyRPMD's existing PES, dynamics and rate APIs.

The adiabatic effective PES, shared-electron-number equilibrium sampler, and
mean-field HER postprocessor are distinct, explicitly normalized models.
"""

from .reservoir import ElectronReservoir
from .potentials import (
    ElectronicPESResult, FixedElectronNumberPotential, AdiabaticElectrodeResult,
    AdiabaticElectrodePotential, build_adiabatic_electrode, evaluate_electronic,
)
from .kinetics import HERRates, HERFluxes, her_fluxes, steady_state_her


def __getattr__(name):
    if name in ("GrandCanonicalState", "GrandCanonicalSampler"):
        from . import sampling
        return getattr(sampling, name)
    raise AttributeError(name)


__all__ = [
    "ElectronReservoir", "ElectronicPESResult", "FixedElectronNumberPotential",
    "AdiabaticElectrodeResult", "AdiabaticElectrodePotential",
    "build_adiabatic_electrode", "evaluate_electronic",
    "GrandCanonicalState", "GrandCanonicalSampler",
    "HERRates", "HERFluxes", "her_fluxes", "steady_state_her",
]
