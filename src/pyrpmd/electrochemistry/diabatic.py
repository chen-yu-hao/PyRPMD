"""Restricted constant-potential matrix interface for existing NRPMD drivers.

Assumed model: H(q,N) = V_rel(q) + E_el(q,N) I, where V_rel excludes
EVERY electrode term. The charging coordinate N is independent of the local
mapping states. Eliminating N gives V_eff(q;mu)=V_rel(q)+Omega_el(q;mu) I.
This is a fast, state-independent adiabatic electrode approximation, NOT a
state-dependent charge-transfer reservoir or shared-N grand-canonical dynamics.
"""
from dataclasses import dataclass
import numpy as np

from ..nonadiabatic import DiabaticResult, validate_diabatic_result
from ..potentials.base import coordinates
from .potentials import AdiabaticElectrodePotential, AdiabaticElectrodeResult


@dataclass(frozen=True)
class ElectrodeDiabaticEvaluation:
    diabatic: DiabaticResult
    electrode: AdiabaticElectrodeResult


class StateIndependentElectrodeDiabaticPotential:
    """Compose disjoint local matrix and scalar electrode terms in atomic units.

    ``relative_diabatic`` implements evaluate(q)->DiabaticResult. Its matrix
    must NOT already contain the electrode energy or reservoir work. ``electrode``
    must be an AdiabaticElectrodePotential with an interior, convex number minimum.
    Each bead is minimized independently, as required by the existing PES batch
    contract. Use the SAME instance for thermal preparation and propagation.

    Both focused NRPMDDynamics and ThermalNRPMDDynamics can consume this adapter;
    thermal parents require the latter's scalar/traceless split. No classical
    Boltzmann distribution of an unbounded mapping Hamiltonian is introduced.
    This adapter has no electronic-number dynamics, no reservoir-induced
    transitions, no charge-state-specific capacitance and no new rate estimator.
    Providers remain caller-owned. Only finite Cartesian systems are covered.
    """

    def __init__(self, relative_diabatic, electrode):
        if not callable(getattr(relative_diabatic, "evaluate", None)):
            raise TypeError("relative_diabatic must implement evaluate(q)")
        if not isinstance(electrode, AdiabaticElectrodePotential):
            raise TypeError("electrode must be an AdiabaticElectrodePotential")
        self.relative_diabatic = relative_diabatic
        self.electrode = electrode
        self.metadata = {
            "electrode_approximation": "state_independent_independent_frame_adiabatic_minimum",
            "requires_atomic_units": True,
            "supports_pbc": False,
            "execution_device": "cpu",
            "thread_safety": "context_serial",
        }

    def evaluate_with_electrode(self, q):
        q = coordinates(q)
        local = validate_diabatic_result(self.relative_diabatic.evaluate(q), q.shape)
        solved = self.electrode.solve(q)
        matrix = local.matrix.copy()
        gradient = local.gradients.copy()
        index = np.arange(matrix.shape[1])
        matrix[:, index, index] += solved.pes.energy[:, None]
        # Matrix protocol uses positive gradients; scalar PES returns forces.
        gradient[:, index, index] -= solved.pes.forces[:, None, :, :]
        result = validate_diabatic_result(DiabaticResult(matrix, gradient), q.shape)
        return ElectrodeDiabaticEvaluation(result, solved)

    def evaluate(self, q):
        return self.evaluate_with_electrode(q).diabatic


__all__ = ["StateIndependentElectrodeDiabaticPotential", "ElectrodeDiabaticEvaluation"]
