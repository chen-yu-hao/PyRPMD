"""Exact thermal rates on coupled electronic states in a 1-3D nuclear subspace.

Research-level extension of :mod:`pyrpmd.analysis.subspace_quantum` to several
electronic states.  The subspace Hamiltonian is

    H_0 = sum_a T_a (x) 1_S + V(s),

where ``V(s)`` is an ``S x S`` Hermitian *diabatic* matrix on a tensor
Colbert--Miller sinc-DVR grid (complex entries, e.g. spin--orbit couplings, are
allowed).  The basis index is ``grid_index * S + state``.  The rate follows
Miller, Schwartz and Tromp with absorbing potentials exactly as the
single-surface dense route:

    k Q_r = Re Tr[F_beta X],     H^dag X - X H = i F,

with ``F = i[H_0, h]``, ``h = theta(s_0 - s_0*) (x) 1_S`` the nuclear product
projector summed over electronic states and ``H = H_0 - i eps`` with an
absorbing potential ``eps (x) 1_S`` at both ends of axis 0.

Product electronic channels are resolved with the identity

    int_0^inf exp(i H^dag t) F exp(-i H t) dt = -h + sum_c Y_c,
    H^dag Y_c - Y_c H = i B_c,    B_c = 2 eps_P^(1/2) P_c eps_P^(1/2),

where ``eps_P`` is the product-end absorber and ``P_c`` projects onto
electronic channel ``c`` (adiabatic or diabatic) at every product-absorber grid
point.  ``k_c Q_r = Re Tr[F_beta Y_c]`` is the thermal flux absorbed in channel
``c``; the channel sum reproduces the total up to ``Tr[F_beta h]``, which is
reported as a numerical check.  All channel solves share one Schur form.

Normalization is thermal over *all modelled* electronic states:

``reactant='asymptotic'``
    ``Q_r = sqrt(m_0/(2 pi beta)) Tr_{perp,el} exp(-beta H_perp(s_ref))``.
    Asymptotic electronic states that are not in the model (for example
    unreactive spin components) must be added by the caller.
``reactant='well'``
    ``Q_r = Tr[exp(-beta H_0)(1 - h)]`` on the grid.

Only the dense Sylvester route exists.  Grid spacing, box, absorbing potential
and the retained electronic states remain numerical parameters to converge.

:func:`propagation_diabatization` builds the diabatic matrix along a 1D path
from adiabatic energies and adjacent-frame wavefunction overlaps (the
convention of :mod:`pyrpmd.potentials.excited_state`).  It is exact for a
complete retained electronic subspace along one coordinate.  It does not
remove the non-removable (curl) part of the derivative coupling on a 2D or 3D
grid.

:func:`nonadiabatic_subspace_factor` compares the coupled-state rate with the
exact rate on one Born--Oppenheimer adiabatic surface of the same subspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.linalg import eigh, schur

from .subspace_model import MAX_DIMENSION, SubspaceModel, _axis, _positive_vector
from .subspace_quantum import (
    _json_safe,
    _pair,
    _triangular_sylvester,
    absorbing_potential,
    exact_quantum_rate,
    sinc_dvr_kinetic,
)

_REACTANT_KINDS = ("asymptotic", "well")
_CHANNEL_BASES = ("adiabatic", "diabatic", None)
_HERMITIAN_RTOL = 1e-10
_MAX_STATES = 32


def _hermitian_grid(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(values)
    dimension = len(shape)
    if (raw.ndim != dimension + 2 or raw.shape[:dimension] != shape or raw.shape[-1] != raw.shape[-2]
            or not 1 <= raw.shape[-1] <= _MAX_STATES or raw.dtype.kind not in "fciu"):
        raise ValueError(f"{name} must have shape {shape} + (S, S) with 1 <= S <= {_MAX_STATES}")
    is_complex = np.iscomplexobj(raw)
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.array(raw, dtype=complex if is_complex else float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    adjoint = np.conj(np.swapaxes(out, -1, -2))
    scale = max(1.0, float(np.max(np.abs(out))))
    if float(np.max(np.abs(out - adjoint))) > _HERMITIAN_RTOL * scale:
        raise ValueError(f"{name} must be Hermitian in the electronic indices at every grid point")
    out = 0.5 * (out + adjoint)
    if is_complex and not np.any(out.imag):
        out = np.ascontiguousarray(out.real)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class MultistateSubspaceModel:
    """Gridded 1-3D Hamiltonian ``-sum_a d^2/(2 m_a ds_a^2) (x) 1_S + V(s)``.

    ``diabatic`` holds the Hermitian diabatic matrix ``V(s)`` with shape
    ``grid_shape + (S, S)`` (``indexing='ij'``).  ``function``, when given, is a
    vectorized callable ``points[..., d] -> [..., S, S]`` used off the grid
    (the asymptotic reactant slice and :meth:`adiabatic_surface`); otherwise
    the reactant slice is interpolated linearly along axis 0.
    """

    axes: tuple
    diabatic: np.ndarray
    masses: np.ndarray
    dividing_surface: float
    reactant: str = "asymptotic"
    reactant_reference: float | None = None
    state_labels: tuple | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    function: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self):
        axes = tuple(_axis(a, f"axes[{i}]") for i, a in enumerate(self.axes))
        if not 1 <= len(axes) <= MAX_DIMENSION:
            raise ValueError("a subspace model must have one to three coordinates")
        shape = tuple(len(a) for a in axes)
        diabatic = _hermitian_grid(self.diabatic, shape, "diabatic")
        masses = _positive_vector(self.masses, len(axes), "masses")
        surface = float(self.dividing_surface)
        axis0 = axes[0]
        if not axis0[1] < surface < axis0[-2]:
            raise ValueError("dividing_surface must lie inside the reaction-coordinate grid")
        if self.reactant not in _REACTANT_KINDS:
            raise ValueError(f"reactant must be one of {_REACTANT_KINDS}")
        reference = self.reactant_reference
        if reference is None:
            reference = float(axis0[0] + 0.25 * (axis0[-1] - axis0[0]))
        reference = float(reference)
        if not axis0[0] <= reference < surface:
            raise ValueError("reactant_reference must lie on the reactant side of the dividing surface")
        if self.function is not None and not callable(self.function):
            raise TypeError("function must be callable or None")
        nstates = diabatic.shape[-1]
        labels = self.state_labels
        if labels is None:
            labels = tuple(f"state{i}" for i in range(nstates))
        labels = tuple(str(label) for label in labels)
        if len(labels) != nstates or len(set(labels)) != nstates:
            raise ValueError("state_labels must give one distinct label per electronic state")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "diabatic", diabatic)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "dividing_surface", surface)
        object.__setattr__(self, "reactant_reference", reference)
        object.__setattr__(self, "state_labels", labels)
        object.__setattr__(self, "provenance", dict(self.provenance or {}))

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self.axes)

    @property
    def nstates(self) -> int:
        return int(self.diabatic.shape[-1])

    @property
    def grid_size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def basis_size(self) -> int:
        return self.grid_size * self.nstates

    @property
    def is_complex(self) -> bool:
        return bool(np.iscomplexobj(self.diabatic))

    def grid_points(self) -> np.ndarray:
        return np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1)

    def matrix(self, points: np.ndarray) -> np.ndarray:
        """Diabatic matrices at off-grid ``points[..., d]`` (requires ``function``)."""
        if self.function is None:
            raise ValueError("off-grid diabatic matrices need a model function")
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != self.dimension:
            raise ValueError("points must end with the subspace dimension")
        values = np.asarray(self.function(points))
        expected = points.shape[:-1] + (self.nstates, self.nstates)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"model function must return finite matrices of shape {expected}")
        return values

    def adiabatic(self) -> np.ndarray:
        """Adiabatic energies ``grid_shape + (S,)`` in ascending order."""
        return np.linalg.eigvalsh(self.diabatic)

    def adiabatic_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Adiabatic energies and eigenvectors (columns) at every grid point."""
        return np.linalg.eigh(self.diabatic)

    def reference_slice(self) -> tuple[np.ndarray, str]:
        """Diabatic matrices on the axis-0 line ``s_0 = reactant_reference``."""
        transverse = self.shape[1:]
        if self.function is not None:
            if self.dimension == 1:
                points = np.array([self.reactant_reference])
            else:
                mesh = np.stack(np.meshgrid(*self.axes[1:], indexing="ij"), axis=-1)
                points = np.concatenate([np.full(mesh.shape[:-1] + (1,), self.reactant_reference), mesh], axis=-1)
            values = self.matrix(points)
            return values.reshape(transverse + (self.nstates, self.nstates)), "model function"
        axis0 = self.axes[0]
        upper = int(np.clip(np.searchsorted(axis0, self.reactant_reference, side="right"), 1, len(axis0) - 1))
        lower = upper - 1
        weight = (self.reactant_reference - axis0[lower]) / (axis0[upper] - axis0[lower])
        values = (1.0 - weight) * self.diabatic[lower] + weight * self.diabatic[upper]
        return values, "linear interpolation along axis 0"

    def diabatic_surface(self, index: int) -> SubspaceModel:
        """Single-surface model on the diagonal diabatic element ``V_ii``."""
        index = self._state_index(index)
        values = np.real(self.diabatic[..., index, index])
        function = None
        if self.function is not None:
            def function(points, _model=self, _i=index):
                return np.real(_model.matrix(points)[..., _i, _i])
        return self._single_surface(values, function, {"kind": "diabatic_surface", "state": index})

    def adiabatic_surface(self, index: int = 0) -> SubspaceModel:
        """Born--Oppenheimer model on adiabat ``index`` (no diagonal correction)."""
        index = self._state_index(index)
        values = self.adiabatic()[..., index]
        function = None
        if self.function is not None:
            def function(points, _model=self, _i=index):
                return np.linalg.eigvalsh(_model.matrix(points))[..., _i]
        return self._single_surface(values, function, {"kind": "adiabatic_surface", "state": index})

    def _state_index(self, index) -> int:
        if isinstance(index, (bool, np.bool_)) or int(index) != index or not 0 <= int(index) < self.nstates:
            raise ValueError(f"state index must be an integer in [0, {self.nstates})")
        return int(index)

    def _single_surface(self, values, function, info) -> SubspaceModel:
        provenance = {**info, "parent": dict(self.provenance), "state_label": self.state_labels[info["state"]]}
        return SubspaceModel(self.axes, values, self.masses, self.dividing_surface, self.reactant,
                             self.reactant_reference, provenance, function)

    def electronic_diagnostics(self) -> dict:
        """Adiabatic gaps over the grid and at the dividing-surface line."""
        info: dict[str, Any] = {"nstates": self.nstates, "complex_couplings": self.is_complex}
        if self.nstates == 1:
            return info
        energies = self.adiabatic()
        gaps = np.diff(energies, axis=-1)
        points = self.grid_points()
        line = int(np.argmin(np.abs(self.axes[0] - self.dividing_surface)))
        pairs = []
        for k in range(self.nstates - 1):
            gap = gaps[..., k]
            flat = int(np.argmin(gap))
            where = np.unravel_index(flat, self.shape)
            pairs.append({
                "states": [k, k + 1],
                "minimum_gap": float(gap.ravel()[flat]),
                "minimum_gap_point": points[where].tolist(),
                "minimum_gap_at_dividing_surface": float(np.min(gap[line])),
            })
        off = np.array(self.diabatic)
        index = np.arange(self.nstates)
        off[..., index, index] = 0.0
        info.update({
            "adiabatic_gaps": pairs,
            "maximum_diabatic_coupling": float(np.max(np.abs(off))),
        })
        return info

    def describe(self) -> dict:
        """JSON-safe summary of the Hamiltonian, without the grid values."""
        return {
            "dimension": self.dimension,
            "shape": list(self.shape),
            "nstates": self.nstates,
            "state_labels": list(self.state_labels),
            "basis_size": self.basis_size,
            "bounds": [[float(a[0]), float(a[-1])] for a in self.axes],
            "masses": self.masses.tolist(),
            "dividing_surface": self.dividing_surface,
            "reactant": self.reactant,
            "reactant_reference": self.reactant_reference,
            "lowest_adiabatic_energy": float(self.adiabatic()[..., 0].min()),
            "complex_couplings": self.is_complex,
            "potential_source": "callable" if self.function is not None else "diabatic grid",
            "construction": dict(self.provenance),
        }


def multistate_model_from_function(
    function: Callable[[np.ndarray], np.ndarray],
    axes: Sequence[Any],
    *,
    masses: Any = 1.0,
    dividing_surface: float = 0.0,
    reactant: str = "asymptotic",
    reactant_reference: float | None = None,
    state_labels: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> MultistateSubspaceModel:
    """Wrap a vectorized diabatic matrix ``V(points[..., d]) -> [..., S, S]``."""
    if not callable(function):
        raise TypeError("function must be callable")
    grids = tuple(_axis(a, f"axes[{i}]") for i, a in enumerate(axes))
    points = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
    values = np.asarray(function(points))
    info = {"kind": "function"}
    info.update(dict(provenance or {}))
    return MultistateSubspaceModel(grids, values, masses, dividing_surface, reactant, reactant_reference,
                                   None if state_labels is None else tuple(state_labels), info, function)


def _grid_kinetic(axes, masses) -> np.ndarray:
    shape = tuple(len(a) for a in axes)
    size = int(np.prod(shape))
    kinetic = np.zeros((size, size))
    for index, (axis, mass) in enumerate(zip(axes, masses)):
        left = int(np.prod(shape[:index]))
        right = int(np.prod(shape[index + 1:]))
        kinetic += np.kron(np.kron(np.eye(left), sinc_dvr_kinetic(axis, mass)), np.eye(right))
    return kinetic


def _block_hamiltonian(kinetic: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """``kinetic (x) 1_S + blockdiag(blocks)`` with basis index ``grid * S + state``."""
    size, nstates = len(kinetic), blocks.shape[-1]
    dtype = complex if np.iscomplexobj(blocks) else float
    hamiltonian = np.kron(kinetic, np.eye(nstates)).astype(dtype, copy=False)
    view = hamiltonian.reshape(size, nstates, size, nstates)
    index = np.arange(size)
    view[index, :, index, :] += blocks.reshape(size, nstates, nstates)
    return hamiltonian


def multistate_hamiltonian(model: MultistateSubspaceModel) -> np.ndarray:
    """Dense ``H_0`` in the basis ``grid_index * S + state``."""
    return _block_hamiltonian(_grid_kinetic(model.axes, model.masses), model.diabatic)


@dataclass(frozen=True)
class MultistateRateResult:
    """Exact coupled-state thermal rate of a subspace Hamiltonian.

    ``rate = flux / reactant_partition``; both carry the same energy shift
    ``exp(beta * energy_shift)``.  ``channel_rates`` (and ``branching``) are
    keyed by product channel: adiabatic index labels ``"adiabatic{k}"`` or
    the model's diabatic state labels.
    """

    beta: float
    rate: float
    flux: float
    reactant_partition: float
    method: str
    channel_basis: str | None
    channel_rates: Mapping[str, float]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "channel_rates", dict(self.channel_rates))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def branching(self) -> dict:
        total = sum(self.channel_rates.values())
        if not self.channel_rates or total == 0:
            return {}
        return {key: value / total for key, value in self.channel_rates.items()}

    def to_dict(self) -> dict:
        return {"beta": self.beta, "rate": self.rate, "flux": self.flux,
                "reactant_partition": self.reactant_partition, "method": self.method,
                "channel_basis": self.channel_basis, "channel_rates": _json_safe(self.channel_rates),
                "branching": _json_safe(self.branching), "diagnostics": _json_safe(self.diagnostics)}


class MultistateRouteUnavailable(ValueError):
    """The dense coupled-state solve exceeds the configured size or memory budget."""

    def __init__(self, plan: Mapping[str, Any]):
        self.plan = dict(plan)
        super().__init__(
            f"coupled-state DVR basis has {plan['basis_size']} functions "
            f"(limit {plan['dense_limit']}, estimated {plan['dense_memory_mb']:.0f} MiB); "
            "only the dense Sylvester route exists for several electronic states -- "
            "coarsen the grid, retain fewer states or raise dense_limit"
        )


def _dense_memory_mb(basis_size: int) -> float:
    # Schur pair, projected thermal flux, right-hand side and solution (complex),
    # plus the real Hamiltonian/commutator kept during assembly.
    return (6 * 16 + 2 * 8) * float(basis_size) ** 2 / 2**20


def _transverse_partition(model, beta, shift):
    blocks, source = model.reference_slice()
    if model.dimension == 1:
        levels = np.linalg.eigvalsh(blocks.reshape(model.nstates, model.nstates))
    else:
        kinetic = _grid_kinetic(model.axes[1:], model.masses[1:])
        if len(kinetic) * model.nstates > 6000:
            raise ValueError("transverse reactant slice exceeds 6000 basis functions; coarsen the transverse grid")
        levels = np.linalg.eigvalsh(_block_hamiltonian(kinetic, blocks))
    weights = np.exp(-beta * (levels - shift))
    return float(weights.sum()), {
        "reference_energy_levels": levels[:8].tolist(),
        "reference_slice": source,
        "transverse_tail_weight": float(weights[-1] / weights.sum()),
    }


def _channel_blocks(model, eps_product, basis):
    """``B_c`` blocks ``[grid, S, S]`` for every product channel, and diagnostics."""
    nstates, size = model.nstates, model.grid_size
    eps = eps_product.reshape(size)
    if basis == "diabatic":
        labels = list(model.state_labels)
        blocks = []
        for c in range(nstates):
            block = np.zeros((size, nstates, nstates))
            block[:, c, c] = 2.0 * eps
            blocks.append(block)
        return labels, blocks, {}
    energies, vectors = model.adiabatic_states()
    energies = energies.reshape(size, nstates)
    vectors = vectors.reshape(size, nstates, nstates)
    labels = [f"adiabatic{k}" for k in range(nstates)]
    blocks = [2.0 * eps[:, None, None] * np.einsum("xi,xj->xij", vectors[:, :, k], vectors[:, :, k].conj())
              for k in range(nstates)]
    absorbing = eps > 0
    info: dict[str, Any] = {}
    if nstates > 1 and np.any(absorbing):
        gap = float(np.min(np.diff(energies[absorbing], axis=-1)))
        info["minimum_adiabatic_gap_in_product_absorber"] = gap
        # Degenerate adiabats make individual adiabatic projectors arbitrary
        # (their sum is not); the total rate is unaffected.
        info["channel_resolution_ambiguous"] = bool(gap < 1e-8)
    return labels, blocks, info


def _apply_blocks(blocks: np.ndarray, matrix: np.ndarray, nstates: int) -> np.ndarray:
    size = blocks.shape[0]
    shaped = matrix.reshape(size, nstates, matrix.shape[1])
    return np.einsum("xij,xjk->xik", blocks, shaped).reshape(matrix.shape)


def exact_multistate_rate(
    model: MultistateSubspaceModel,
    beta: float,
    *,
    cap_width: Any = None,
    cap_strength: Any = None,
    cap_power: int = 2,
    channels: str | None = "adiabatic",
    energy_floor: float | None = None,
    dense_limit: int = 1600,
    memory_limit_mb: float | None = None,
) -> MultistateRateResult:
    """Exact coupled-state thermal rate of ``model`` at inverse temperature ``beta``.

    The absorbing potential, its defaults (20% of axis 0 at each end, strength
    ``30 v_th / width``), ``cap_width``/``cap_strength`` pairs and
    ``energy_floor`` follow :func:`~pyrpmd.analysis.subspace_quantum.exact_quantum_rate`;
    the absorber is the same on every electronic state.  ``channels``
    resolves the flux absorbed at the product end into ``"adiabatic"`` states
    (energy-ordered at each grid point), ``"diabatic"`` states, or nothing
    (``None``).  ``dense_limit`` counts DVR basis functions, grid points times
    electronic states.
    """
    if not isinstance(model, MultistateSubspaceModel):
        raise TypeError("model must be a MultistateSubspaceModel")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    if channels not in _CHANNEL_BASES:
        raise ValueError("channels must be 'adiabatic', 'diabatic' or None")
    size, nstates, basis = model.grid_size, model.nstates, model.basis_size
    plan = {"route": "sylvester", "basis_size": basis, "grid_points": size, "nstates": nstates,
            "dense_limit": int(dense_limit), "dense_memory_mb": _dense_memory_mb(basis)}
    if memory_limit_mb is not None and (not np.isfinite(memory_limit_mb) or memory_limit_mb <= 0):
        raise ValueError("memory_limit_mb must be finite and positive")
    if basis > int(dense_limit) or (memory_limit_mb is not None and plan["dense_memory_mb"] > memory_limit_mb):
        raise MultistateRouteUnavailable(plan)

    axis0 = model.axes[0]
    length = float(axis0[-1] - axis0[0])
    widths = _pair(0.2 * length if cap_width is None else cap_width)
    if not all(0 < w < 0.5 * length for w in widths):
        raise ValueError("cap_width must be positive and below half the reaction-coordinate range")
    if not axis0[0] + widths[0] < model.dividing_surface < axis0[-1] - widths[1]:
        raise ValueError("the dividing surface must lie between the absorbing regions")
    velocity = np.sqrt(2.0 / (beta * model.masses[0]))
    strengths = tuple(30.0 * velocity / w for w in widths) if cap_strength is None else _pair(cap_strength)
    if not all(np.isfinite(v) and v > 0 for v in strengths):
        raise ValueError("cap_strength must be positive")
    eps_axis = absorbing_potential(axis0, width=widths, strength=strengths, power=int(cap_power))
    product_axis = (axis0 > model.dividing_surface).astype(float)
    broadcast = (-1,) + (1,) * (model.dimension - 1)
    eps_grid = np.broadcast_to(eps_axis.reshape(broadcast), model.shape)
    product_grid = np.broadcast_to(product_axis.reshape(broadcast), model.shape)
    eps_product = eps_grid * product_grid
    eps_basis = np.repeat(eps_grid.ravel(), nstates)
    product = np.repeat(product_grid.ravel(), nstates)
    shift = float(model.adiabatic()[..., 0].min())

    started = perf_counter()
    hamiltonian = multistate_hamiltonian(model)
    energies, vectors = eigh(hamiltonian)
    diagonalized = perf_counter()
    if model.reactant == "asymptotic":
        transverse, partition_info = _transverse_partition(model, beta, shift)
        partition = float(np.sqrt(model.masses[0] / (2.0 * np.pi * beta)) * transverse)
        partition_info["normalization"] = "per unit length of axis 0 at reactant_reference, all modelled states"
    else:
        weights = np.exp(-beta * (energies - shift))
        populations = np.sum((vectors.conj() * vectors).real * (1.0 - product)[:, None], axis=0)
        partition = float(np.sum(weights * populations))
        partition_info = {"normalization": "Tr[exp(-beta H0)(1-h)] on the grid, all modelled states",
                          "boltzmann_tail_weight": float(weights[-1] / weights.sum())}
    if not np.isfinite(partition) or partition <= 0:
        raise FloatingPointError("reactant partition function must be finite and positive")
    half = np.exp(-0.5 * beta * (energies - shift))
    if energy_floor is not None:
        energy_floor = float(energy_floor)
        if not np.isfinite(energy_floor):
            raise ValueError("energy_floor must be finite")
        half = np.where(energies >= energy_floor, half, 0.0)
    lowest_levels = energies[:6].tolist()
    states_below_floor = int(np.sum(energies < energy_floor)) if energy_floor is not None else 0
    # K = [H0, h]; F = iK.  F_beta = i * thermal.
    commutator = hamiltonian * product[None, :]
    commutator -= product[:, None] * hamiltonian
    boltzmann_half = (vectors * half[None, :]) @ vectors.conj().T
    del vectors, energies
    thermal = boltzmann_half @ commutator @ boltzmann_half
    del boltzmann_half
    absorbing = np.array(hamiltonian, dtype=complex, order="F")
    del hamiltonian
    absorbing[np.diag_indices_from(absorbing)] -= 1j * eps_basis
    schur_started = perf_counter()
    triangular, unitary = schur(absorbing, output="complex", overwrite_a=True)
    schur_finished = perf_counter()
    del absorbing
    projected = unitary.conj().T @ thermal @ unitary
    side = float(np.real(1j * np.sum(np.diag(thermal) * product)))
    del thermal

    def solve(rhs):
        solution, scale, info, solver = _triangular_sylvester(triangular, rhs)
        if info < 0:
            raise RuntimeError(f"ztrsyl failed with info={info}")
        return float(np.real(1j * np.einsum("ij,ji->", projected, solution)) / scale), solver, float(scale), int(info)

    # Right-hand side i U^+ F U = -U^+ K U.
    rhs = -(unitary.conj().T @ commutator @ unitary)
    del commutator
    flux, solver, scale, info = solve(rhs)
    del rhs
    channel_flux: dict[str, float] = {}
    channel_info: dict[str, Any] = {}
    if channels is not None:
        labels, blocks, channel_info = _channel_blocks(model, eps_product, channels)
        for label, block in zip(labels, blocks):
            rhs = 1j * (unitary.conj().T @ _apply_blocks(block, unitary, nstates))
            channel_flux[label] = solve(rhs)[0]
            del rhs
    sylvester_finished = perf_counter()
    if not np.isfinite(flux):
        raise FloatingPointError("exact coupled-state rate produced a non-finite flux")
    diagnostics: dict[str, Any] = {
        "grid_points": size,
        "nstates": nstates,
        "basis_size": basis,
        "sylvester_scale": scale,
        "sylvester_info": info,
        "sylvester_solver": solver,
        "timing_seconds": {"assembly_and_eigh": diagonalized - started,
                           "schur": schur_finished - schur_started,
                           "sylvester": sylvester_finished - schur_finished,
                           "dense_total": perf_counter() - started},
        "flux_side_t0": side,
        "lowest_levels": lowest_levels,
        "energy_floor": energy_floor,
        "states_below_floor": states_below_floor,
        "partition": partition_info,
        "energy_shift": shift,
        "cap_width": widths[0] if widths[0] == widths[1] else list(widths),
        "cap_strength": strengths[0] if strengths[0] == strengths[1] else list(strengths),
        "cap_power": int(cap_power),
        "solver_plan": plan,
        "electronic": model.electronic_diagnostics(),
        "model": model.describe(),
        "units": "axis-0 length per time" if model.reactant == "asymptotic" else "per time",
    }
    if channels is not None:
        total = sum(channel_flux.values())
        # sum_c Y_c = X + h exactly, so sum_c flux_c = flux + Tr[F_beta h].
        diagnostics["channel_sum_residual"] = float((total - flux - side) / max(abs(flux), 1e-300))
        diagnostics["channel_flux"] = dict(channel_flux)
        diagnostics.update(channel_info)
    return MultistateRateResult(beta, float(flux / partition), float(flux), partition, "sylvester", channels,
                                {k: v / partition for k, v in channel_flux.items()}, diagnostics)


def nonadiabatic_subspace_factor(model: MultistateSubspaceModel, beta: float, *, state: int = 0,
                                 **options) -> dict:
    """Coupled-state rate over the exact rate on Born--Oppenheimer adiabat ``state``.

    Both rates use the same grid, absorber and ``energy_floor``.  The ratio
    measures nonadiabatic transitions, the diagonal Born--Oppenheimer
    correction and thermal population of the other modelled states inside
    this subspace only.  Combined with a single-surface RPMD subspace
    correction it assumes that the electronic coupling acts within the
    subspace and that the remaining modes follow adiabat ``state``.
    """
    allowed = {"cap_width", "cap_strength", "cap_power", "channels", "energy_floor", "dense_limit",
               "memory_limit_mb"}
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(f"unknown options {sorted(unknown)}")
    coupled = exact_multistate_rate(model, beta, **options)
    single = {key: options[key] for key in ("cap_width", "cap_strength", "cap_power", "energy_floor",
                                             "dense_limit", "memory_limit_mb") if key in options}
    born_oppenheimer = exact_quantum_rate(model.adiabatic_surface(state), beta, method="sylvester", **single)
    return {
        "factor": coupled.rate / born_oppenheimer.rate,
        "state": int(state),
        "multistate": coupled,
        "born_oppenheimer": born_oppenheimer,
    }


@dataclass(frozen=True)
class PropagationDiabatization:
    """Diabatic matrices along a 1D path from adiabatic data.

    ``transformations[n]`` holds the diabatic states as columns in the
    adiabatic basis of frame ``n`` (``chi_n = Phi_n A_n``);
    ``diabatic[n] = A_n^dag diag(E_n) A_n``.  ``singular_values[n]`` are those
    of the raw overlap between frames ``n`` and ``n + 1``; values below one
    measure leakage out of the retained electronic subspace.
    """

    diabatic: np.ndarray
    transformations: np.ndarray
    singular_values: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"diabatic": _json_safe(self.diabatic), "singular_values": _json_safe(self.singular_values),
                "diagnostics": _json_safe(self.diagnostics)}


def propagation_diabatization(energies: Any, overlaps: Any, *, anchor: int = 0,
                              min_singular_value: float = 0.9) -> PropagationDiabatization:
    """Propagation (overlap) diabatization of ``S`` adiabatic states along a path.

    ``energies[N, S]`` are adiabatic energies in any per-frame order and
    ``overlaps[N-1, S, S]`` the raw wavefunction overlaps
    ``O_n[i, j] = <phi_i(s_n)|phi_j(s_{n+1})>`` in the same orders, with
    arbitrary phases (the :class:`~pyrpmd.potentials.excited_state.BackendResult`
    convention).  Each overlap is replaced by its polar unitary factor ``W_n``
    and ``A_{n+1} = W_n^dag A_n`` with ``A_anchor = 1``, so the diabatic states
    equal the adiabatic states of frame ``anchor``.  State reordering and sign
    or phase changes are absorbed exactly.  The construction is exact along one
    coordinate when the retained states span a complete electronic subspace;
    a singular value below ``min_singular_value`` is rejected as leakage.
    """
    energies = np.asarray(energies, dtype=float)
    raw = np.asarray(overlaps)
    if energies.ndim != 2 or energies.shape[0] < 2 or not 1 <= energies.shape[1] <= _MAX_STATES:
        raise ValueError("energies must have shape [frames >= 2, states]")
    frames, nstates = energies.shape
    if raw.shape != (frames - 1, nstates, nstates) or raw.dtype.kind not in "fciu":
        raise ValueError(f"overlaps must have shape {(frames - 1, nstates, nstates)}")
    if not np.isfinite(energies).all() or not np.isfinite(raw).all():
        raise ValueError("energies and overlaps must be finite")
    if isinstance(anchor, (bool, np.bool_)) or int(anchor) != anchor or not 0 <= int(anchor) < frames:
        raise ValueError("anchor must be a frame index")
    if not 0 < float(min_singular_value) <= 1:
        raise ValueError("min_singular_value must lie in (0, 1]")
    is_complex = np.iscomplexobj(raw)
    dtype = complex if is_complex else float
    transformations = np.empty((frames, nstates, nstates), dtype=dtype)
    singular = np.empty((frames - 1, nstates))
    current = np.eye(nstates, dtype=dtype)
    transformations[0] = current
    for n in range(frames - 1):
        left, values, right = np.linalg.svd(raw[n])
        singular[n] = values
        if values.min() < min_singular_value:
            raise ValueError(
                f"overlap between frames {n} and {n + 1} has singular value {values.min():.3g} "
                f"< {min_singular_value}: the retained electronic subspace is incomplete or the step "
                "is too large; retain more states or refine the path")
        unitary = left @ right
        current = unitary.conj().T @ current
        transformations[n + 1] = current
    reference = transformations[int(anchor)].copy()
    transformations = transformations @ reference.conj().T
    diabatic = np.einsum("nia,ni,nib->nab", transformations.conj(), energies, transformations)
    diabatic = 0.5 * (diabatic + np.conj(np.swapaxes(diabatic, -1, -2)))
    if is_complex and not np.any(diabatic.imag):
        diabatic = diabatic.real
    off = np.array(diabatic)
    index = np.arange(nstates)
    off[:, index, index] = 0.0
    coupling = np.max(np.abs(off), axis=(1, 2))
    diagnostics = {
        "frames": frames,
        "nstates": nstates,
        "anchor": int(anchor),
        "minimum_singular_value": float(singular.min()),
        "maximum_singular_value": float(singular.max()),
        "coupling_at_ends": [float(coupling[0]), float(coupling[-1])],
        "maximum_coupling": float(coupling.max()),
        "scope": "1D propagation diabatization; exact only for a complete retained electronic subspace",
    }
    return PropagationDiabatization(diabatic, transformations, singular, diagnostics)


__all__ = [
    "MultistateRateResult",
    "MultistateRouteUnavailable",
    "MultistateSubspaceModel",
    "PropagationDiabatization",
    "exact_multistate_rate",
    "multistate_hamiltonian",
    "multistate_model_from_function",
    "nonadiabatic_subspace_factor",
    "propagation_diabatization",
]
