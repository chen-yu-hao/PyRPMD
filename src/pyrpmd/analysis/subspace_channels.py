"""Local transverse channels of a two-dimensional Cartesian Hamiltonian.

``sigma`` is the fixed Cartesian longitudinal coordinate, not Frenet arc
length. Curved valleys remain in V(sigma,Q). The local transverse basis gives
U^dagger H_cartesian U, including all kinetic overlaps; its full-Q limit is
unitarily equivalent to the original DVR. RPMD uses that original Cartesian
model, while finite-channel truncation remains a separate convergence error.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

import numpy as np
from scipy.linalg import eigh

from .subspace_model import SubspaceModel, path_curvature
from .subspace_metric import constant_metric_model
from .subspace_quantum import (_dense_rate_matrix, _truncation_margin, _truncation_threshold,
                               absorbing_potential, sinc_dvr_kinetic)


def _as_axis(value: Any, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=float)
    if axis.ndim != 1 or axis.size < 8 or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a finite one-dimensional grid with at least eight points")
    if not np.all(np.diff(axis) > 0) or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-9, atol=0):
        raise ValueError(f"{name} must be strictly increasing and uniformly spaced")
    return axis.copy()


def _evaluate_potential(potential: Callable, sigma: np.ndarray, q: np.ndarray) -> np.ndarray:
    points = np.stack(np.meshgrid(sigma, q, indexing="ij"), axis=-1)
    try:
        evaluator = potential.energy if hasattr(potential, "energy") else potential
        values = np.asarray(evaluator(points), dtype=float)
        if values.shape != points.shape[:-1]:
            raise ValueError
    except Exception:
        values = np.empty(points.shape[:-1], dtype=float)
        for i, s in enumerate(sigma):
            for j, normal in enumerate(q):
                try:
                    values[i, j] = float(evaluator(np.array([s, normal])))
                except Exception:
                    values[i, j] = float(evaluator(float(s), float(normal)))
    if not np.isfinite(values).all():
        raise ValueError("potential is not finite on the channel grid")
    return values


def _curvature_values(curvature: Any, sigma: np.ndarray) -> np.ndarray:
    if curvature is None:
        return np.zeros_like(sigma)
    values = np.asarray(curvature(sigma) if callable(curvature) else curvature, dtype=float)
    if values.ndim == 0:
        values = np.full_like(sigma, float(values))
    if values.shape != sigma.shape or not np.isfinite(values).all():
        raise ValueError("curvature must be a scalar, a callable, or one value per sigma grid point")
    return values


def _align_phases(states: np.ndarray) -> np.ndarray:
    """Change phases only: a general polar rotation changes eigenstate energies.

    A small overlap at a crossing is not evidence of a degenerate eigenspace.
    The Galerkin assembly below is invariant to independent local phases and
    does not need to mix distinct eigenstates to make a derivative smooth.
    """
    aligned = np.array(states, dtype=complex, copy=True)
    for i in range(1, len(aligned)):
        overlap = np.diag(aligned[i - 1].conj().T @ aligned[i])
        phase = np.ones_like(overlap)
        nonzero = np.abs(overlap) > 1e-14
        phase[nonzero] = overlap[nonzero] / np.abs(overlap[nonzero])
        aligned[i] *= phase.conj()[None, :]
    return aligned


def _path_jacobian(curvature, q):
    """Jacobian of r(sigma)+Q*N with the inward Frenet normal N=T'/kappa."""
    jacobian = 1.0 - np.asarray(curvature)[:, None] * np.asarray(q)[None, :]
    # J is affine in Q, so checking its positive sign at the endpoints also
    # excludes a singular surface between nodes. Squaring J would lose this.
    if np.any(jacobian[:, (0, -1)] <= 1e-8):
        raise ValueError("Frenet tube crosses a singular metric surface 1-kappa*Q=0")
    return jacobian


def _project_channel_hamiltonian(kinetic, energies, states):
    """U^dagger (T_sigma + T_Q + V) U without allocating the full 2D matrix.

    Local overlaps retain the full derivative norm, including the positive
    Born--Huang contribution from derivatives outside the retained channels.
    Replacing them by the within-channel connection A alone loses that term.
    """
    ns, _, nc = states.shape
    projected = np.einsum("ij,iqn,jqm->injm", kinetic, states.conj(), states,
                          optimize=True).reshape(ns * nc, ns * nc)
    projected += _block_diag(np.array([np.diag(e) for e in energies], complex))
    return 0.5 * (projected + projected.conj().T)


def _block_diag(blocks: np.ndarray) -> np.ndarray:
    n, m, _ = blocks.shape
    out = np.zeros((n * m, n * m), dtype=blocks.dtype)
    for i, block in enumerate(blocks):
        out[i * m : (i + 1) * m, i * m : (i + 1) * m] = block
    return out


@dataclass(frozen=True)
class SubspaceChannels:
    """A finite adiabatic-channel representation of a reaction-path model."""

    sigma: np.ndarray
    q: np.ndarray
    energies: np.ndarray
    states: np.ndarray
    derivative_couplings: np.ndarray
    curvature_couplings: np.ndarray
    metric_couplings: np.ndarray
    curvature: np.ndarray
    hamiltonian: np.ndarray
    potential_values: np.ndarray
    dividing_surface: float
    reactant_reference: float
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_model: SubspaceModel | None = field(default=None, repr=False, compare=False)
    channel_counts: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for name in ("sigma", "q", "energies", "states", "derivative_couplings",
                     "curvature_couplings", "metric_couplings", "curvature", "hamiltonian",
                     "potential_values"):
            value = np.array(getattr(self, name), copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.hamiltonian.shape[0] != self.hamiltonian.shape[1]:
            raise ValueError("channel Hamiltonian must be square")
        if not np.allclose(self.hamiltonian, self.hamiltonian.conj().T, atol=2e-10, rtol=1e-9):
            raise ValueError("channel Hamiltonian must be Hermitian")
        counts = (np.full(self.sigma.size, self.energies.shape[1], dtype=int)
                  if self.channel_counts is None else np.array(self.channel_counts, dtype=int, copy=True))
        counts.flags.writeable = False
        object.__setattr__(self, "channel_counts", counts)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def n_sigma(self) -> int:
        return int(self.sigma.size)

    @property
    def n_channels(self) -> int:
        return int(self.energies.shape[1])

    @property
    def size(self) -> int:
        return int(self.hamiltonian.shape[0])

    @property
    def dense_matrix(self) -> np.ndarray:
        return self.hamiltonian

    def describe(self) -> dict:
        """Return a JSON-safe summary without storing the channel vectors."""
        return {
            "sigma_points": self.n_sigma,
            "normal_points": int(self.q.size),
            "channels": self.n_channels,
            "bounds": [[float(self.sigma[0]), float(self.sigma[-1])],
                       [float(self.q[0]), float(self.q[-1])]],
            "dividing_surface": float(self.dividing_surface),
            "reactant_reference": float(self.reactant_reference),
            "construction": dict(self.provenance),
        }

    @property
    def active_channels(self):
        return np.arange(self.n_channels)[None, :] < self.channel_counts[:, None]

    def product_projector(self) -> np.ndarray:
        return np.repeat((self.sigma > self.dividing_surface).astype(float), self.n_channels)[self.active_channels.ravel()]

    def channel_model(self, *, reactant: str | None = None) -> SubspaceModel:
        """The underlying Cartesian model, not a fitted channel potential.

        The numerator is a Galerkin approximation to this same two-dimensional
        Hamiltonian; increasing n_channels to len(q) recovers its full DVR.
        """
        self._require_shared_hamiltonian()
        if self.source_model is not None:
            if reactant is not None and reactant != self.source_model.reactant:
                raise ValueError("reactant normalization must match the source model")
            return self.source_model
        masses = np.array([self.provenance.get("sigma_mass", 1.0),
                           self.provenance.get("normal_mass", 1.0)])
        return SubspaceModel((self.sigma, self.q), self.potential_values, masses,
                             self.dividing_surface, reactant or "asymptotic",
                             self.reactant_reference,
                             {**self.provenance, "kind": "cartesian_channels_rpmd"})

    def _require_shared_hamiltonian(self):
        if np.any(self.curvature != 0):
            raise NotImplementedError(
                "Nonzero Frenet curvature has no validated shared quantum/RPMD "
                "Hamiltonian; construct channels_from_model on a Cartesian plane."
            )
        if not self.provenance.get("include_nonadiabatic", True):
            raise NotImplementedError(
                "Uncoupled channel matrices are diagnostic only and cannot be "
                "paired with the coupled Cartesian RPMD model."
            )

    rpmd_model = channel_model

    def correction(self, beta, nbeads, *, quantum_options=None, rpmd_options=None, rpmd_model=None):
        """C_S for the shared Cartesian model, with channel error unresolved.

        Channel truncation is a quantum discretization error. It must be
        converged separately from RPMD sampling and bead-number errors before
        this finite-basis estimate can be used as a production correction.
        """
        from .subspace_correction import SubspaceCorrection
        from .subspace_rpmd import ring_polymer_rate

        model = self.channel_model()
        if rpmd_model is not None:
            rpmd_model = constant_metric_model(rpmd_model)
            _check_same_cartesian_model(model, rpmd_model)
        quantum = self.exact_rate(beta, **dict(quantum_options or {}))
        rpmd = ring_polymer_rate(model if rpmd_model is None else rpmd_model,
                                 beta, nbeads, **dict(rpmd_options or {}))
        if quantum.rate <= 0 or rpmd.rate <= 0:
            raise FloatingPointError("subspace rates must be positive to form C_S")
        return SubspaceCorrection(
            float(beta), 2, quantum.rate / rpmd.rate,
            rpmd.log_rate_standard_error, quantum.rate, rpmd.rate, int(nbeads),
            model.describe(), quantum, rpmd,
            {"hamiltonian": "shared Cartesian SubspaceModel",
             "n_channels": self.n_channels, "normal_grid_points": int(self.q.size),
             "channel_basis_complete": quantum.diagnostics["channel_basis_complete"],
             "channel_truncation_converged": quantum.diagnostics["channel_basis_complete"],
             "channel_truncation_error_in_standard_error": False,
             "physical_converged": False},
        )

    def exact_rate(self, beta: float, *, cap_width: float | None = None,
                   cap_strength: float | None = None, energy_floor: float | None = None,
                   cap_power: int = 2, absorber=None, normalization="default",
                   truncation_energy=None, dense_limit=1600, method="sylvester"):
        """Exact thermal rate from the dense channel Hamiltonian."""
        self._require_shared_hamiltonian()
        if method not in ("auto", "sylvester"):
            raise ValueError("channel rates currently require the dense Sylvester route")
        if normalization not in ("default", "slice"):
            raise ValueError("normalization must be 'default' or 'slice'")
        if truncation_energy is not None:
            truncated = channels_from_model(self.channel_model(), n_channels=self.n_channels,
                                             truncation_energy=truncation_energy)
            return truncated.exact_rate(beta, cap_width=cap_width, cap_strength=cap_strength,
                cap_power=cap_power, energy_floor=energy_floor, absorber=absorber,
                normalization=normalization, dense_limit=dense_limit, method=method)
        if self.size > max(dense_limit, 6000):
            raise ValueError("channel grid too large for the dense Sylvester route")
        beta = float(beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive")
        length = float(self.sigma[-1] - self.sigma[0])
        width = 0.2 * length if cap_width is None else float(cap_width)
        if not 0 < width < 0.5 * length:
            raise ValueError("cap_width must be positive and below half the sigma range")
        sigma_mass = float(self.provenance.get("sigma_mass", 1.0))
        if not np.isfinite(sigma_mass) or sigma_mass <= 0:
            raise ValueError("channel sigma mass must be positive")
        strength = (30.0 * np.sqrt(2.0 / (beta * sigma_mass)) / width
                    if cap_strength is None else float(cap_strength))
        eps_sigma = absorbing_potential(self.sigma, width=width, strength=strength, power=int(cap_power))
        eps = np.repeat(eps_sigma, self.n_channels)[self.active_channels.ravel()]
        if absorber is not None:
            points = self.channel_model().grid_points()
            eps_grid = np.asarray(absorber(points.reshape(-1, 2)), dtype=float).reshape(self.potential_values.shape)
            if not np.isfinite(eps_grid).all() or np.any(eps_grid < 0):
                raise ValueError("absorber must return finite nonnegative values")
            blocks = np.einsum("iqn,iq,iqm->inm", self.states.conj(), eps_grid, self.states)
            eps = _block_diag(blocks)[np.ix_(self.active_channels.ravel(), self.active_channels.ravel())]
        product = self.product_projector()
        model = self.channel_model()
        reference_points = np.column_stack((np.full(len(self.q), self.reactant_reference), self.q))
        transverse = (sinc_dvr_kinetic(self.q, float(model.masses[1]))
                      + np.diag(model.energy(reference_points)))
        reference_levels = np.linalg.eigvalsh(transverse)[:self.n_channels]
        shift = float(np.min(self.energies[self.active_channels]))

        def partition(energies, vectors):
            if normalization == "slice":
                ref = int(np.argmin(abs(self.sigma - self.reactant_reference)))
                on_line = np.repeat(np.arange(self.n_sigma) == ref, self.n_channels)[self.active_channels.ravel()]
                population = np.sum(np.abs(vectors[on_line])**2, axis=0)
                z = np.sum(np.exp(-beta * (energies-shift))*population)/(self.sigma[1]-self.sigma[0])
                return float(z), {"normalization": "quantum line density on the reference slice",
                                  "reference_line": float(self.sigma[ref]), "slice_points": int(self.channel_counts[ref])}
            z = float(np.sum(np.exp(-beta * (reference_levels - shift))))
            return np.sqrt(sigma_mass / (2.0 * np.pi * beta)) * z, {
                "normalization": "channel sum at reactant reference per unit sigma",
                "reference_sigma": float(self.reactant_reference), "channels": self.n_channels,
            }

        flux, partition_value, info = _dense_rate_matrix(
            self.hamiltonian, eps, product, beta, shift,
            partition=partition if normalization == "slice" or model.reactant == "asymptotic" else None,
            reactant_weights=1.0 - product,
            energy_floor=energy_floor,
        )
        from .subspace_quantum import QuantumRateResult
        # The normal DVR is a finite box.  Report the smallest energy gap from
        # each local transverse valley floor to either normal-coordinate wall;
        # this is the channel analogue of ``truncation_margin`` in the tensor
        # product solver and must be several transverse quanta for a reliable
        # rate.  Keeping the minimum is conservative when the valley bends.
        normal_floor = np.min(self.potential_values, axis=1)
        normal_wall = np.min(self.potential_values[:, (0, -1)], axis=1)
        normal_margin = float(np.min(normal_wall - normal_floor))
        normal_margins = normal_wall - normal_floor
        reference_index = int(np.argmin(abs(self.sigma - self.reactant_reference)))
        surface_index = int(np.argmin(abs(self.sigma - self.dividing_surface)))
        normal_frequency = (float(np.max(self.energies[:, 1] - self.energies[:, 0]))
                            if self.n_channels > 1 else float("nan"))
        info.update({"channel_count": self.n_channels, "sigma_points": self.n_sigma,
                     "cap_width": width, "cap_strength": strength,
                     "normal_grid_points": int(self.q.size),
                     "truncation_margin": normal_margin,
                     "truncation_margin_reference": float(normal_margins[reference_index]),
                     "truncation_margin_dividing_surface": float(normal_margins[surface_index]),
                     "truncation_margin_minimum": normal_margin,
                     "normal_frequency_estimate": normal_frequency,
                     "six_quanta_margin": (6.0 * normal_frequency if np.isfinite(normal_frequency) else None),
                     "truncation_converged": (bool(normal_margin >= 6.0 * normal_frequency)
                                               if np.isfinite(normal_frequency) else False),
                     # A channel solve is a numerical result until the user
                     # checks the normal-box and sigma-grid ladders.
                     "physical_converged": False,
                     "position_dependent_metric": False,
                     "hamiltonian": "Cartesian T_sigma + T_Q + V; local Galerkin projection",
                     "channel_basis_complete": bool(self.provenance.get("channel_basis_complete", self.n_channels == self.q.size)),
                     "channel_truncation_converged": bool(self.provenance.get("channel_basis_complete", self.n_channels == self.q.size)),
                     "channel_counts": self.channel_counts.tolist(),
                     "active_channel_dimension": self.size})
        info.update(self.provenance.get("energy_truncation_diagnostics", {}))
        return QuantumRateResult(beta, flux / partition_value, flux, partition_value,
                                 "sylvester", info)


def build_subspace_channels(
    potential: Callable,
    sigma: Any,
    q: Any,
    *,
    n_channels: int = 4,
    masses: Any = (1.0, 1.0),
    curvature: Any = None,
    method: str = "sinc",
    include_nonadiabatic: bool = True,
    include_coupling: bool | None = None,
    dividing_surface: float = 0.0,
    reactant_reference: float | None = None,
    provenance: Mapping[str, Any] | None = None,
    retained_points: Any = None,
) -> SubspaceChannels:
    """Construct transverse eigenchannels on a Cartesian longitudinal grid.

    ``potential(points)`` receives ``[..., 2]`` points ``(sigma,Q)``.  The
    normal kinetic mass is ``masses[1]`` and the reaction-coordinate mass is
    ``masses[0]``.  ``method='harmonic'`` is accepted as a provenance label for
    callers that use harmonic-cut grids; the same sinc diagonalization is used
    so the stored energies remain variationally consistent with the DVR.
    """
    sigma = _as_axis(sigma, "sigma")
    q = _as_axis(q, "q")
    if isinstance(n_channels, bool) or int(n_channels) != n_channels or not 1 <= int(n_channels) <= q.size:
        raise ValueError("n_channels must be an integer between one and len(q)")
    n_channels = int(n_channels)
    if method not in ("sinc", "harmonic"):
        raise ValueError("method must be 'sinc' or 'harmonic'")
    if not isinstance(include_nonadiabatic, (bool, np.bool_)):
        raise ValueError("include_nonadiabatic must be boolean")
    if include_coupling is not None:
        include_nonadiabatic = bool(include_coupling)
    masses = np.asarray(masses, dtype=float)
    if masses.shape != (2,) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("masses must contain two positive finite values")
    if not sigma[0] < dividing_surface < sigma[-1]:
        raise ValueError("dividing_surface must lie inside sigma")
    reference = (sigma[0] + 0.25 * (sigma[-1] - sigma[0])
                 if reactant_reference is None else float(reactant_reference))
    if not sigma[0] <= reference < dividing_surface:
        raise ValueError("reactant_reference must lie on the reactant side")

    values = _evaluate_potential(potential, sigma, q)
    kappa = _curvature_values(curvature, sigma)
    _path_jacobian(kappa, q)
    if np.any(kappa != 0):
        raise NotImplementedError(
            "Nonzero Frenet curvature requires a consistent metric quantum "
            "Hamiltonian and RPMD measure. Use channels_from_model with the "
            "original two-dimensional Cartesian potential instead."
        )
    tq = sinc_dvr_kinetic(q, masses[1])
    retained = np.ones_like(values, dtype=bool) if retained_points is None else np.asarray(retained_points, dtype=bool)
    if retained.shape != values.shape or not np.any(retained):
        raise ValueError("retained_points must select at least one point on the channel grid")
    counts = np.minimum(n_channels, retained.sum(axis=1))
    energies = np.zeros((sigma.size, n_channels), dtype=float)
    states = np.zeros((sigma.size, q.size, n_channels), dtype=complex)
    for i in range(sigma.size):
        keep = np.flatnonzero(retained[i])
        if not len(keep):
            continue
        e, u = eigh(tq[np.ix_(keep, keep)] + np.diag(values[i, keep]))
        energies[i, :counts[i]] = e[:counts[i]]
        states[i][np.ix_(keep, np.arange(counts[i]))] = u[:, :counts[i]]
    states = _align_phases(states)

    derivative_couplings = np.empty((sigma.size, n_channels, n_channels), dtype=complex)
    for i in range(sigma.size):
        if i == 0:
            derivative = (states[1] - states[0]) / (sigma[1] - sigma[0])
        elif i == sigma.size - 1:
            derivative = (states[-1] - states[-2]) / (sigma[-1] - sigma[-2])
        else:
            derivative = (states[i + 1] - states[i - 1]) / (sigma[i + 1] - sigma[i - 1])
        derivative_couplings[i] = 0.5 * (
            states[i].conj().T @ derivative - derivative.conj().T @ states[i]
        )
    curvature_couplings = np.zeros_like(derivative_couplings)
    metric_couplings = np.broadcast_to(np.eye(n_channels), derivative_couplings.shape).copy()
    kinetic = sinc_dvr_kinetic(sigma, masses[0])
    if include_nonadiabatic:
        hamiltonian = _project_channel_hamiltonian(kinetic, energies, states)
    else:
        derivative_couplings[...] = 0.0
        hamiltonian = np.kron(kinetic, np.eye(n_channels)) + _block_diag(
            np.array([np.diag(e) for e in energies], complex))
    active = (np.arange(n_channels)[None, :] < counts[:, None]).ravel()
    hamiltonian = hamiltonian[np.ix_(active, active)]
    info = {
        "kind": "adiabatic_channels", "method": method,
        "sigma_points": int(sigma.size), "normal_points": int(q.size),
        "sigma_mass": float(masses[0]), "normal_mass": float(masses[1]),
        "n_channels": n_channels, "curvature_max": float(np.max(np.abs(kappa))),
        "phase_aligned": True, "metric": "Cartesian identity",
        "kinetic_projection": "U_dagger_H_cartesian_U",
        "include_nonadiabatic": bool(include_nonadiabatic),
        "channel_basis_complete": bool(np.all(counts == retained.sum(axis=1))),
    }
    info = {**dict(provenance or {}), **info}
    return SubspaceChannels(sigma, q, energies, states, derivative_couplings,
                            curvature_couplings, metric_couplings, kappa,
                            hamiltonian, values, float(dividing_surface), float(reference), info,
                            channel_counts=counts)


def _check_same_cartesian_model(coarse, fine):
    if (fine.dimension != 2 or fine.reactant != coarse.reactant
            or fine.dividing_surface != coarse.dividing_surface
            or fine.reactant_reference != coarse.reactant_reference
            or not np.array_equal(fine.masses, coarse.masses)):
        raise ValueError("rpmd_model must have the same masses, surface and reactant normalization")
    for nodes, refined in zip(coarse.axes, fine.axes):
        indices = np.searchsorted(refined, nodes)
        indices = np.clip(indices, 0, len(refined)-1)
        if not np.allclose(refined[indices], nodes, rtol=0, atol=1e-10):
            raise ValueError("rpmd_model must contain every coarse Cartesian grid node")
    if not np.allclose(fine.energy(coarse.grid_points()), coarse.values, rtol=1e-10, atol=1e-11):
        raise ValueError("rpmd_model potential differs at the shared Cartesian nodes")


def channels_from_model(model: SubspaceModel, *, n_channels: int = 4, truncation_energy=None) -> SubspaceChannels:
    """Project an existing 2D Cartesian H_S into local transverse eigenstates.

    Curved valleys remain in V(x,y); sigma is the fixed Cartesian reaction
    axis, not Frenet arc length. No inverse tube map or variable-mass RPMD is
    needed. The same model object supplies the denominator and normalization.
    """
    if not isinstance(model, SubspaceModel) or model.dimension != 2:
        raise ValueError("channels_from_model requires a two-dimensional Cartesian SubspaceModel")
    model = constant_metric_model(model)
    retained, margin = None, {}
    if truncation_energy is not None:
        threshold = _truncation_threshold(model, truncation_energy)
        retained = model.values <= threshold
        margin = _truncation_margin(model, model.values, threshold)
    channels = build_subspace_channels(
        model.energy, model.axes[0], model.axes[1], n_channels=n_channels,
        masses=model.masses, dividing_surface=model.dividing_surface,
        reactant_reference=model.reactant_reference,
        provenance={"coordinate_system": "Cartesian", "source_model": model.describe(),
                    "energy_truncation_diagnostics": margin},
        retained_points=retained,
    )
    return replace(channels, source_model=model)


def assemble_channel_hamiltonian(channels: SubspaceChannels) -> np.ndarray:
    """Return a fresh copy of the assembled channel matrix."""
    if not isinstance(channels, SubspaceChannels):
        raise TypeError("channels must be a SubspaceChannels object")
    return np.array(channels.hamiltonian, copy=True)


construct_subspace_channels = build_subspace_channels


def project_path_channels(potential: Callable, gradient: Any, path: Any, masses: Any,
                          axis: Any, *, q_axis: Any = None, normal: Any = None,
                          n_channels: int = 4, curvature: Any = None, **kwargs):
    """Build channels from a straight Cartesian reaction path.

    ``axis`` is the mass-weighted arc-length grid and ``q_axis`` is the normal
    grid.  A two-dimensional path uses its Frenet normal; higher-dimensional
    paths require an explicit ``normal`` array. Curved Frenet models are
    rejected; use channels_from_model with their original Cartesian potential.
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[0] < 4 or not np.isfinite(points).all():
        raise ValueError("path must be a finite [points, coordinates] array with at least four points")
    mass = np.asarray(masses, dtype=float)
    if mass.shape != (points.shape[1],) or np.any(mass <= 0) or not np.isfinite(mass).all():
        raise ValueError("masses must contain one positive value per path coordinate")
    sigma = _as_axis(axis, "axis")
    q = _as_axis(q_axis, "q_axis") if q_axis is not None else None
    if q is None:
        raise ValueError("q_axis is required for a path channel model")
    weighted = points * np.sqrt(mass)[None, :]
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(weighted, axis=0), axis=1))]
    if np.any(np.diff(chord) <= 0) or sigma[0] < chord[0] or sigma[-1] > chord[-1]:
        raise ValueError("axis must lie within the path arc length")
    path_sigma = np.column_stack([np.interp(sigma, chord, points[:, j]) for j in range(points.shape[1])])
    tangent = np.gradient(weighted, chord, axis=0, edge_order=2)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    if normal is None:
        normal_points = np.gradient(tangent, chord, axis=0, edge_order=2)
        norm = np.linalg.norm(normal_points, axis=1)
        normal_points = normal_points / np.maximum(norm[:, None], 1e-300)
        if np.any(norm <= 1e-10):
            if points.shape[1] != 2:
                raise ValueError("normal must be supplied for a straight path in dimension greater than two")
            normal_points[norm <= 1e-10] = np.column_stack((-tangent[norm <= 1e-10, 1],
                                                               tangent[norm <= 1e-10, 0]))
    else:
        normal_points = np.array(normal, dtype=float, copy=True)
        if normal_points.shape != points.shape or not np.isfinite(normal_points).all():
            raise ValueError("normal must have the same shape as path")
        lengths = np.linalg.norm(normal_points, axis=1)
        if np.any(lengths <= 1e-12) or np.any(np.abs(np.sum(normal_points * tangent, axis=1)) > 1e-8 * lengths):
            raise ValueError("normal must be nonzero and perpendicular to the path tangent")
        normal_points /= lengths[:, None]
    normal_sigma = np.column_stack([np.interp(sigma, chord, normal_points[:, j]) for j in range(points.shape[1])])
    normal_sigma /= np.linalg.norm(normal_sigma, axis=1)[:, None]

    def cut_potential(grid):
        grid = np.asarray(grid, dtype=float)
        flat = grid.reshape(-1, 2)
        centres = np.column_stack([np.interp(flat[:, 0], sigma, path_sigma[:, j]) for j in range(points.shape[1])])
        directions = np.column_stack([np.interp(flat[:, 0], sigma, normal_sigma[:, j]) for j in range(points.shape[1])])
        cartesian = centres + flat[:, 1, None] * directions / np.sqrt(mass)[None, :]
        try:
            value = np.asarray(potential(cartesian), dtype=float)
            if value.shape != (len(flat),):
                raise ValueError
        except Exception:
            value = np.array([float(potential(x)) for x in cartesian])
        return value.reshape(grid.shape[:-1])

    if curvature is None:
        _, curvature_samples = path_curvature(points, mass, samples=max(401, len(sigma)))
        curvature = np.interp(sigma, np.linspace(chord[0], chord[-1], len(curvature_samples)), curvature_samples)
    dividing_surface = kwargs.pop("dividing_surface", float(sigma[len(sigma) // 2]))
    reference = kwargs.pop("reactant_reference", None)
    return build_subspace_channels(cut_potential, sigma, q, n_channels=n_channels,
                                   masses=(1.0, 1.0), curvature=curvature,
                                   dividing_surface=dividing_surface,
                                   reactant_reference=reference, **kwargs)


project_path_subspace_channels = project_path_channels


def exact_channel_rate(channels: SubspaceChannels, beta: float, **options):
    """Convenience wrapper for :meth:`SubspaceChannels.exact_rate`."""
    if not isinstance(channels, SubspaceChannels):
        raise TypeError("channels must be a SubspaceChannels object")
    return channels.exact_rate(beta, **options)


def channels_to_model(channels: SubspaceChannels, **kwargs) -> SubspaceModel:
    """Convert channels to the shared two-dimensional RPMD potential model."""
    if not isinstance(channels, SubspaceChannels):
        raise TypeError("channels must be a SubspaceChannels object")
    return channels.channel_model(**kwargs)


__all__ = [
    "SubspaceChannels", "build_subspace_channels", "construct_subspace_channels",
    "project_path_channels", "project_path_subspace_channels", "assemble_channel_hamiltonian", "channels_to_model",
    "exact_channel_rate", "channels_from_model",
]
