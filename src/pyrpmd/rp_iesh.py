"""Gardner's centroid ring-polymer independent-electron surface hopping.

This research variant follows J. J. Gardner, University of Warwick PhD thesis
(2023), section VI.1.1, equations (VI.1)--(VI.2), together with the centroid
surface-hopping prescription. All beads share one occupied Slater determinant;
one-electron amplitudes and hopping rates are evaluated at the centroid.

An accepted hop shifts every bead momentum by the same vector, using the
centroid NAC and centroid orbital energy gap. It conserves centroid kinetic
plus centroid electronic energy at the hop, generally NOT the sum-bead ring
Hamiltonian. Per-hop and cumulative full-ring changes are explicit diagnostics.
This heuristic extension does not establish preservation of the quantum
Boltzmann distribution, detailed balance, or physically converged metal-surface
dynamics. Complex SOC, decoherence and interacting electrons are unsupported.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .dynamics import State
from .iesh import IESHDynamics, IESHState, _occupied, _orbitals, _real
from .nonadiabatic import _positive_integer


def _hop_accumulators(signed, absolute):
    signed = _real(signed, "hop_energy_change")
    absolute = _real(absolute, "hop_energy_change_abs", nonnegative=True)
    tolerance = 64*np.finfo(float).eps*max(abs(signed), absolute)
    if absolute < abs(signed)-tolerance:
        raise ValueError("hop_energy_change_abs must bound abs(hop_energy_change)")
    return signed, absolute


@dataclass
class RPIESHState(IESHState):
    """A nuclear ring with one common electronic determinant and diagnostics.

    ``coefficients[S,Ne]`` are orthonormal fixed-diabatic orbitals, and
    ``occupied[Ne]`` is one ordered set of occupied adiabatic orbitals for the
    entire ring. The two energy accumulators record accepted-hop changes of
    the full sum-bead Hamiltonian, in hartree. They do not modify propagation.
    Save the dynamics RNG state alongside this state for exact continuation.
    """
    hop_energy_change: float = 0.
    hop_energy_change_abs: float = 0.

    def __post_init__(self):
        State.__post_init__(self)
        self.coefficients = _orbitals(self.coefficients)
        self.occupied = _occupied(self.occupied, *self.coefficients.shape)
        self.hop_energy_change, self.hop_energy_change_abs = _hop_accumulators(
            self.hop_energy_change, self.hop_energy_change_abs)

    def copy(self):
        return RPIESHState(self.q, self.p, self.step, self.time, self.coefficients,
                           self.occupied, self.hop_energy_change, self.hop_energy_change_abs)

    def to_dict(self):
        return {**super().to_dict(), "hop_energy_change": float(self.hop_energy_change),
                "hop_energy_change_abs": float(self.hop_energy_change_abs)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {
            "q", "p", "step", "time", "coefficients_real", "coefficients_imag", "occupied",
            "hop_energy_change", "hop_energy_change_abs",
        }:
            raise ValueError("invalid RP-IESH state fields")
        base = {key: entry for key, entry in value.items()
                if key not in {"hop_energy_change", "hop_energy_change_abs"}}
        state = super().from_dict(base)
        state.hop_energy_change, state.hop_energy_change_abs = _hop_accumulators(
            value["hop_energy_change"], value["hop_energy_change_abs"])
        return state


@dataclass(frozen=True)
class RPIESHHopResult:
    """Energy changes of a centroid hop, with all energies in hartree.

    ``centroid_energy_error`` is the actual change of centroid kinetic energy
    plus the centroid orbital gap. ``ring_energy_change`` is the actual
    full-ring kinetic change plus the sum of bead orbital gaps; the spring and
    bare potentials do not change instantaneously. Both are zero for a
    rejected hop. The compatibility ``energy_error`` property always denotes
    the centroid residual, never a claim of full-ring energy conservation.
    """
    electron: int
    source: int
    target: int
    accepted: bool
    frustrated: bool
    centroid_energy_error: float
    ring_energy_change: float

    @property
    def energy_error(self):
        return self.centroid_energy_error


class RPIESHDynamics(IESHDynamics):
    """Centroid RP-IESH with beadwise forces and a common electronic state.

    The constructor has the same parameters as :class:`IESHDynamics`, except
    that any positive integer bead count is supported. Nuclear propagation is
    unthermostatted with physical masses and unscaled bead forces. The
    electronic quadrature and at-most-one-hop timing are inherited from IESH.
    Frustrated hops keep all momenta unchanged; there is no alternative gap,
    rescaling, bead-local electronic state, or frustrated-hop option.

    P=1 uses the existing IESH numerical path, including its RNG consumption.
    P>1 uses the fixed ``gardner-2023-centroid`` research variant. See
    :class:`RPIESHHopResult` before interpreting the reported ring energy drift.
    """
    _state_type = RPIESHState
    method_variant = "gardner-2023-centroid"

    @staticmethod
    def _validate_nbeads(nbeads):
        _positive_integer(nbeads, "nbeads")

    def _electronic_coordinates(self, q):
        # Preserve every operation of the original P=1 path.
        return q if self.nbeads == 1 else q.mean(axis=0, keepdims=True)

    def _endpoint_adiabatic(self, q):
        if self.nbeads == 1:
            return super()._endpoint_adiabatic(q)
        # Endpoint bead forces need no bead NACs. A selected hop evaluates its
        # NAC at the centroid, rather than using an arbitrarily chosen bead.
        return (*self._adiabatic(q), None)

    def _check(self, state):
        super()._check(state)
        _hop_accumulators(state.hop_energy_change, state.hop_energy_change_abs)

    def _commit_state(self, state, trial):
        super()._commit_state(state, trial)
        state.hop_energy_change = trial.hop_energy_change
        state.hop_energy_change_abs = trial.hop_energy_change_abs

    def energy(self, state):
        """Full sum-bead active-determinant Hamiltonian, including springs.

        This quantity generally jumps at accepted centroid hops. The signed
        cumulative jump is available as ``state.hop_energy_change``.
        """
        if self.nbeads == 1:
            return super().energy(state)
        self._check(state)
        energies, _, _ = self._adiabatic(state.q)
        with np.errstate(over="ignore", invalid="ignore"):
            kinetic = np.sum(state.p**2/(2*self.masses[None, :, None]))
            dq = state.q-np.roll(state.q, -1, axis=0)
            spring = .5*(self.nbeads/self.beta)**2*np.sum(self.masses[None, :, None]*dq*dq)
            energy = kinetic+spring+self._bare(state.q).energy.sum()+energies[:, state.occupied].sum()
        return _real(energy, "RP-IESH ring energy")

    @staticmethod
    def _accumulate_hop(state, result):
        if result.accepted:
            signed, absolute = _hop_accumulators(
                state.hop_energy_change+result.ring_energy_change,
                state.hop_energy_change_abs+abs(result.ring_energy_change))
            state.hop_energy_change, state.hop_energy_change_abs = signed, absolute
        return result

    def _hop(self, state, electron, target, *, adiabatic=None):
        if self.nbeads == 1:
            hop = super()._hop(state, electron, target, adiabatic=adiabatic)
            result = RPIESHHopResult(hop.electron, hop.source, hop.target, hop.accepted,
                                     hop.frustrated, hop.energy_error, hop.energy_error)
            return self._accumulate_hop(state, result)
        if (isinstance(electron, (bool, np.bool_)) or not isinstance(electron, (int, np.integer))
                or not 0 <= electron < self.nelectrons):
            raise ValueError("electron must be an in-range integer")
        if (isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer))
                or not 0 <= target < self.nstates):
            raise ValueError("target must be an in-range orbital integer")
        source = int(state.occupied[electron])
        if target in state.occupied:
            raise ValueError("IESH hop target is already occupied (Pauli forbidden)")
        centroid = self._electronic_coordinates(state.q)
        energies, _, _, coupling = self._adiabatic(centroid, couplings=True)
        gap = float(energies[0, target]-energies[0, source])
        sqrt_mass = np.sqrt(self.masses[:, None])
        normal = coupling[0, source, target]/sqrt_mass
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm):
            raise ValueError("RP-IESH mass-weighted hop direction overflowed")
        if norm == 0.:
            return RPIESHHopResult(int(electron), source, int(target), False, True, 0., 0.)
        normal /= norm
        if normal.flat[np.argmax(np.abs(normal))] < 0.:
            normal *= -1.
        centroid_p = state.p.mean(axis=0)
        projection = float(np.sum(centroid_p/sqrt_mass*normal))
        discriminant = projection*projection-2*gap
        if not np.isfinite(discriminant):
            raise ValueError("RP-IESH momentum rescaling overflowed")
        if discriminant < 0.:
            return RPIESHHopResult(int(electron), source, int(target), False, True, 0., 0.)
        root = np.copysign(np.sqrt(discriminant), projection if projection != 0. else 1.)
        denominator = root+projection
        shift = -2*gap/denominator if denominator != 0. else 0.
        bead_energies = self._adiabatic(state.q)[0] if adiabatic is None else adiabatic[0]
        with np.errstate(over="ignore", invalid="ignore"):
            ring_gap = np.sum(bead_energies[:, target]-bead_energies[:, source])
            before_ring = np.sum(state.p**2/(2*self.masses[None, :, None]))
            before_centroid = np.sum(centroid_p**2/(2*self.masses[:, None]))
            state.p += (sqrt_mass*shift*normal)[None]
            after_ring = np.sum(state.p**2/(2*self.masses[None, :, None]))
            after_centroid = np.sum(state.p.mean(axis=0)**2/(2*self.masses[:, None]))
            centroid_error = _real(after_centroid-before_centroid+gap,
                                    "RP-IESH centroid hop energy error")
            ring_change = _real(after_ring-before_ring+ring_gap,
                                "RP-IESH full-ring hop energy change")
        state.occupied[electron] = target
        result = RPIESHHopResult(int(electron), source, int(target), True, False,
                                 centroid_error, ring_change)
        return self._accumulate_hop(state, result)
