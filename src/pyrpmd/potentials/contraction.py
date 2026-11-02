"""Explicit fast/slow PES splitting and ring-polymer contraction in atomic units.

RPC changes the potential part of the ring-polymer Hamiltonian.  These wrappers
therefore act on ONE complete ring, not on an arbitrary batch of independent
frames.  Forces differentiate the *sum* of the returned energies.  They are not
independent per-frame forces, and the metadata makes this distinction explicit.
"""

from __future__ import annotations

import numpy as np

from ..validation import positive_integer as _positive_integer
from .base import PESResult, Potential, PotentialError, coordinates, validate_result


def _set_component_cells(owner, providers, cell):
    """Forward a common cell to borrowed providers, retaining a rollback cell.

    Providers must implement the same explicit AU cell-update contract as an
    ordinary NPT potential. Check all callbacks before changing any component.
    A provider remains responsible for side effects of its own failed callback.
    """
    from ..geometry import validate_cell_pbc
    cell, _ = validate_cell_pbc(cell, [True, True, True])
    if cell is None:
        raise PotentialError("composite NPT potential requires a finite periodic cell")
    unique = list({id(provider): provider for provider in providers}.values())
    if any(not callable(getattr(provider, "set_cell", None)) for provider in unique):
        raise PotentialError("every composite NPT component must implement set_cell(cell_bohr)")
    previous = getattr(owner, "_cell", None)
    changed = []
    try:
        for provider in unique:
            changed.append(provider)
            provider.set_cell(cell.copy())
    except BaseException:
        if previous is not None:
            for provider in reversed(changed):
                provider.set_cell(previous.copy())
        raise
    owner._cell = cell.copy()


def _component(provider, name, *, _allow_callable=False):
    if not (callable(getattr(provider, "evaluate", None))
            or _allow_callable and callable(provider)):
        raise TypeError(f"{name} must implement evaluate(positions)")
    metadata = getattr(provider, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{name} potential metadata must be a mapping")
    capabilities = metadata.get("capabilities", {}) or {}
    if not isinstance(capabilities, dict):
        raise ValueError(f"{name} potential capabilities must be a mapping")
    if (metadata.get("batch_independent") is False
            or metadata.get("ring_polymer_coupled") or metadata.get("coupled_frames")):
        raise ValueError(f"{name} must evaluate independent physical configurations")
    # Composite providers publish their own metadata and may cache results at
    # an unchanged geometry.  Reject every supported history/time marker at
    # the leaf boundary, before wrapping can hide it from callers such as
    # replica exchange.  Attributes and metadata describe the same contract.
    dynamic_markers = ("history_dependent", "time_dependent", "stateful")
    if any(getattr(provider, key, False) or metadata.get(key, False)
           or capabilities.get(key, False) for key in dynamic_markers):
        raise ValueError(f"{name} must be a deterministic position-only potential")
    from .builtin import HarmonicPotential
    if type(provider) is HarmonicPotential and (
        provider.k.ndim > 2 or provider.center.ndim > 2
    ):
        raise ValueError(f"{name} harmonic parameters cannot depend on the frame axis")
    from .python import PythonPotential
    if type(provider) is PythonPotential:
        # Unit/sign conversion does not remove a coupling or frame-dependent
        # parameter in the wrapped physical provider, including plain callbacks.
        _component(provider.obj, name, _allow_callable=True)
    return provider


class RingPolymerContraction:
    """Low-frequency normal-mode projection from P to Pc beads.

    ``contract(q) = T @ q`` preserves the centroid and removes high-frequency
    modes.  ``lift_forces(f) = (P/Pc) T.T @ f`` is the exact chain rule for
    ``(P/Pc) sum_k V(contract(q)[k])``.  For even Pc < P the final retained mode
    is the cosine member of the degenerate pair, as in the i-PI convention.
    Pc=P is implemented as an exact identity rather than a numerical transform.
    """

    def __init__(self, nbeads, contracted_beads):
        self.nbeads = _positive_integer(nbeads, "nbeads")
        self.contracted_beads = _positive_integer(contracted_beads, "contracted_beads")
        if self.contracted_beads > self.nbeads:
            raise ValueError("contracted_beads cannot exceed nbeads")
        p, c = self.nbeads, self.contracted_beads
        if c == p:
            matrix = np.eye(p)
        else:
            # Direct trigonometric form of sqrt(Pc/P) C_Pc S C_P.T.
            # Real sine/cosine pairs combine into cos(theta_c - theta_P).
            source = np.arange(p, dtype=float)[None, :] / p
            target = np.arange(c, dtype=float)[:, None] / c
            matrix = np.ones((c, p), dtype=float)
            for k in range(1, (c + 1) // 2):
                matrix += 2 * np.cos(2 * np.pi * k * (target - source))
            if c % 2 == 0:
                matrix += np.sqrt(2.0) * np.cos(np.pi * np.arange(c))[:, None] * np.cos(np.pi * c * source)
            matrix /= p
        matrix.flags.writeable = False
        self._matrix = matrix

    @property
    def matrix(self):
        """Read-only [Pc,P] coordinate projection (do not change its scaling)."""
        return self._matrix

    def contract(self, positions):
        q = coordinates(positions)
        if len(q) != self.nbeads:
            raise ValueError(f"expected one complete ring with {self.nbeads} beads")
        if self.nbeads == self.contracted_beads:
            return q.copy()
        return np.ascontiguousarray(np.einsum("cj,jna->cna", self._matrix, q))

    def lift_forces(self, forces):
        f = coordinates(forces)
        if len(f) != self.contracted_beads:
            raise ValueError(f"expected forces on {self.contracted_beads} contracted beads")
        if self.nbeads == self.contracted_beads:
            return f.copy()
        return np.ascontiguousarray(
            (self.nbeads / self.contracted_beads) * np.einsum("cj,cna->jna", self._matrix, f)
        )


class DifferencePotential(Potential):
    """Physical correction ``target - reference`` on independent configurations.

    The two providers are borrowed: callers retain ownership and close them.
    Use with ``SplitPotential(reference, DifferencePotential(target, reference),
    ...)`` to avoid counting the reference potential twice.
    """

    def __init__(self, target, reference):
        self.target = _component(target, "target")
        self.reference = _component(reference, "reference")
        self.metadata = {
            "provider": "difference", "thread_safety": "context_serial",
            "batch_independent": True, "trajectory_batch_safe": False,
            "units": {"energy": "hartree", "length": "bohr"},
        }

    def set_cell(self, cell):
        """Update both physical providers; ownership remains with the caller."""
        _set_component_cells(self, (self.target, self.reference), cell)

    def evaluate(self, positions):
        q = coordinates(positions)
        high = validate_result(self.target.evaluate(q), q.shape)
        # Components may reuse their own output arrays, including shared native
        # storage.  Consume/copy all needed values before the next provider call.
        energy, force = high.energy.copy(), high.forces.copy()
        virial = None if high.virial is None else high.virial.copy()
        low = validate_result(self.reference.evaluate(q), q.shape)
        energy -= low.energy
        force -= low.forces
        if virial is not None and low.virial is not None:
            virial -= low.virial
        else:
            virial = None
        return validate_result(PESResult(energy, force, virial=virial), q.shape)


class SplitPotential(Potential):
    """``sum Vfast(q_j) + (P/Pc) sum Vslow((Tq)_k)`` for one complete ring.

    ``slow`` must be the additive correction, not another full potential unless
    that is intentionally the Hamiltonian being modelled.  Pc defaults to P
    (no contraction). Providers remain owned by the caller. If both components
    provide virials, the returned sum is the affine cell derivative of this
    contracted Hamiltonian. Contracted energy/virial rows are bookkeeping, not
    bead-resolved physical observables of the uncontracted target potential.
    """

    def __init__(self, fast, slow, nbeads, contracted_beads=None):
        self.fast = _component(fast, "fast")
        self.slow = _component(slow, "slow")
        self.contraction = RingPolymerContraction(
            nbeads, nbeads if contracted_beads is None else contracted_beads
        )
        self.nbeads = self.contraction.nbeads
        self.contracted_beads = self.contraction.contracted_beads
        self.metadata = {
            "provider": "ring-polymer-split", "thread_safety": "context_serial",
            "batch_independent": False, "trajectory_batch_safe": False,
            "ring_polymer_coupled": True, "nbeads": self.nbeads,
            "contracted_beads": self.contracted_beads,
            "approximate_hamiltonian": self.contracted_beads != self.nbeads,
            "energy_convention": "sum_beads; contracted_energy_distributed_uniformly",
            "virial_requires": "both_components",
            "virial_scope": "rpc_contracted_hamiltonian",
            "virial_convention": "sum_beads; contracted_virial_distributed_uniformly",
        }

    def set_cell(self, cell):
        """Bind both components to the same cell before a pressure evaluation."""
        _set_component_cells(self, (self.fast, self.slow), cell)

    def _coordinates(self, positions):
        q = coordinates(positions)
        if len(q) != self.nbeads:
            raise ValueError(f"SplitPotential requires one complete {self.nbeads}-bead ring")
        return q

    def evaluate_fast(self, positions):
        q = self._coordinates(positions)
        return validate_result(self.fast.evaluate(q), q.shape)

    def evaluate_slow(self, positions):
        q = self._coordinates(positions)
        qc = self.contraction.contract(q)
        result = validate_result(self.slow.evaluate(qc), qc.shape)
        forces = self.contraction.lift_forces(result.forces)
        if self.nbeads == self.contracted_beads:
            energy = result.energy.copy()
            virial = None if result.virial is None else result.virial.copy()
        else:
            # Only the sum has a physical meaning for the coupled Hamiltonian.
            energy = np.full(self.nbeads, float(np.mean(result.energy)))
            # T is independent of the cell and commutes with affine strain.
            # Consequently sum Wslow_RPC = (P/Pc) sum Wslow(Tq). There is
            # no force-lifting transform on a cell derivative. Uniform rows
            # preserve that sum without claiming individual physical virials.
            virial = (None if result.virial is None else np.broadcast_to(
                result.virial.mean(axis=0), (self.nbeads, 3, 3)).copy())
        return validate_result(PESResult(energy, forces, virial=virial), q.shape)

    def evaluate(self, positions):
        q = self._coordinates(positions)
        fast = self.evaluate_fast(q)
        energy, forces = fast.energy.copy(), fast.forces.copy()
        virial = None if fast.virial is None else fast.virial.copy()
        slow = self.evaluate_slow(q)
        energy += slow.energy
        forces += slow.forces
        virial = None if virial is None or slow.virial is None else virial + slow.virial
        return validate_result(PESResult(energy, forces, virial=virial), q.shape)


__all__ = ["RingPolymerContraction", "DifferencePotential", "SplitPotential"]
