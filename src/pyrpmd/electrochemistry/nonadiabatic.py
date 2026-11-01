"""Common scalar constant-potential charging on a diabatic matrix PES.

This adapter reuses the per-configuration adiabatic electrode solver.  It does
not minimize different electron numbers on different electronic states, and it
does not introduce a reservoir transition kernel or an electronic response
time.  All quantities at this boundary are atomic units.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..geometry import validate_cell_pbc
from ..nonadiabatic import DiabaticResult, validate_diabatic_result
from ..potentials.base import Potential, PESResult, PotentialError, coordinates
from .potentials import AdiabaticElectrodePotential, AdiabaticElectrodeResult


def _component_metadata(provider, name):
    if not callable(getattr(provider, "evaluate", None)):
        raise TypeError(f"{name} must implement evaluate")
    metadata = getattr(provider, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{name} metadata must be a mapping")
    capabilities = metadata.get("capabilities", {}) or {}
    if not isinstance(capabilities, Mapping):
        raise ValueError(f"{name} capabilities must be a mapping")
    if (metadata.get("batch_independent") is False
            or metadata.get("ring_polymer_coupled") or metadata.get("coupled_frames")):
        raise ValueError(f"{name} must evaluate independent physical configurations")
    if (any(getattr(provider, key, False) or metadata.get(key, False)
            for key in ("stateful", "time_dependent", "history_dependent"))
            or capabilities.get("stateful", False)):
        raise ValueError(f"{name} must be deterministic and time independent")
    return metadata


@dataclass(frozen=True)
class ConstantPotentialDiabaticResult:
    """Detached matrix evaluation and its scalar electrode diagnostics.

    ``diabatic`` is the full grand-potential matrix and its positive nuclear
    gradients.  ``electrode.pes`` contains only the common scalar grand
    potential and its negative nuclear gradients (forces).  Arrays are owned
    by this result, not by reusable work buffers of either provider.
    """

    diabatic: DiabaticResult
    electrode: AdiabaticElectrodeResult
    reservoir_chemical_potential_au: float

    @property
    def electron_numbers(self):
        return self.electrode.electron_numbers

    @property
    def chemical_potentials_au(self):
        return self.electrode.chemical_potentials_au

    @property
    def chemical_potential_residuals_au(self):
        """Per-configuration dE_charge/dN - mu, in hartree."""
        return self.chemical_potentials_au - self.reservoir_chemical_potential_au


class ConstantPotentialDiabaticPotential(Potential):
    """Add a state-independent electrode grand potential to every diagonal.

    The explicitly separable model is

    ``V_total(R,N) = V_bare(R) + E_charge(R,N) I`` and
    ``V_mu(R) = V_bare(R) + min_N[E_charge(R,N) - mu N] I``.

    ``matrix_potential.evaluate(q)`` supplies ``DiabaticResult`` for V_bare.
    ``electrode_potential`` must be an ``AdiabaticElectrodePotential`` for
    E_charge.  The charging energy must be absent from V_bare: adding a second
    full PES would double count it.  The two components must use identical
    nuclear coordinates, energy zero and atomic units.  A batch comprises
    independent configurations; each ring bead has its own electronic optimum.

    This model gives all diabatic states the same charge response.  It does
    not describe state-dependent charging, bias-dependent electronic couplings,
    electron-transfer jumps, shared-N grand-canonical fluctuations, or TP-RPMD.
    Such models need their own consistent matrix potential and dynamics.

    Use ``ThermalNRPMDDynamics`` for thermally sampled mapping variables: its
    scalar/traceless split applies the common charging force independently of
    the fluctuating mapping norm.  Focused ``NRPMDDynamics`` remains applicable
    when every bead has the normalized focused mapping norm.

    Both providers are borrowed.  ``bind_system`` forwards system information
    to both, while callers retain ownership and close the providers themselves.
    No native context from either component describes the combined matrix.
    """

    def __init__(self, matrix_potential, electrode_potential):
        if not isinstance(electrode_potential, AdiabaticElectrodePotential):
            raise TypeError("electrode_potential must be an AdiabaticElectrodePotential")
        matrix_metadata = _component_metadata(matrix_potential, "matrix_potential")
        electrode_metadata = _component_metadata(electrode_potential, "electrode_potential")
        # The scalar wrapper intentionally exports conservative metadata, so
        # also inspect explicit flags on its underlying number-dependent model.
        _component_metadata(electrode_potential.model, "electrode model")
        self.matrix_potential = matrix_potential
        self.electrode_potential = electrode_potential
        self.metadata = {
            "provider": "constant_potential_diabatic",
            "thread_safety": "context_serial", "execution_device": "cpu",
            "requires_atomic_units": True, "derivative": "gradient",
            "batch_independent": True, "trajectory_batch_safe": False,
            "electrode_approximation": "common_scalar_independent_frame_adiabatic_minimum",
            "constant_potential_matrix_model": "state_independent_charging",
        }
        if all(metadata.get("supports_pbc") is True
               and callable(getattr(provider, "bind_system", None))
               for provider, metadata in ((matrix_potential, matrix_metadata),
                                          (electrode_potential, electrode_metadata))):
            self.metadata["supports_pbc"] = True

    def bind_system(self, system):
        get = system.get if isinstance(system, dict) else lambda key, default=None: getattr(system, key, default)
        _, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        if pbc.any() and self.metadata.get("supports_pbc") is not True:
            raise PotentialError("periodic matrix and electrode models both require supports_pbc=True and bind_system")
        for provider in (self.matrix_potential, self.electrode_potential):
            bind = getattr(provider, "bind_system", None)
            if callable(bind):
                bind(system)

    def solve(self, q):
        """Evaluate the full matrix and retain the converged charge diagnostics."""
        q = coordinates(q)
        bare = validate_diabatic_result(self.matrix_potential.evaluate(q), q.shape)
        # Copy before evaluating the other component: providers may share or
        # reuse external work buffers, and the result must outlive later calls.
        matrix, gradients = bare.matrix.copy(), bare.gradients.copy()
        status, matrix_messages = bare.status.copy(), tuple(bare.messages)
        mu = self.electrode_potential.reservoir.chemical_potential_au
        result = self.electrode_potential.solve(q)
        pes = result.pes
        electrode = AdiabaticElectrodeResult(
            PESResult(pes.energy.copy(), pes.forces.copy(), pes.status.copy(),
                      tuple(pes.messages), None if pes.virial is None else pes.virial.copy()),
            result.electron_numbers.copy(), result.chemical_potentials_au.copy(),
            result.iterations,
        )
        index = np.arange(matrix.shape[1])
        matrix[:, index, index] += electrode.pes.energy[:, None]
        # The matrix contract uses +dV/dR; the scalar contract supplies -dV/dR.
        gradients[:, index, index] -= electrode.pes.forces[:, None]
        messages = tuple("; ".join(filter(None, (left, right)))
                         for left, right in zip(matrix_messages, electrode.pes.messages))
        full = validate_diabatic_result(DiabaticResult(matrix, gradients, status, messages), q.shape)
        return ConstantPotentialDiabaticResult(full, electrode, mu)

    def evaluate(self, q):
        return self.solve(q).diabatic


__all__ = ["ConstantPotentialDiabaticPotential", "ConstantPotentialDiabaticResult"]
