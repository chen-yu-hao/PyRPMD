"""Adiabatic transverse channels for a curved reaction-path subspace.

The path coordinate ``sigma`` is retained as a DVR coordinate while a small
set of eigenstates of the normal Hamiltonian in ``Q`` is retained at every
point.  The resulting matrix is a variational ``sigma x channel`` Hamiltonian
and is deliberately kept as a dense object: the exact rate solver already has
an accurately tested dense Sylvester implementation for this size of model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.linalg import eigh

from .subspace_model import SubspaceModel, model_from_function, path_curvature
from .subspace_quantum import _dense_rate_matrix, absorbing_potential, sinc_dvr_kinetic


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


def _derivative_matrix(axis: np.ndarray) -> np.ndarray:
    """First derivative matrix paired with the sinc-DVR kinetic matrix."""
    difference = np.arange(axis.size)[:, None] - np.arange(axis.size)[None, :]
    out = np.zeros((axis.size, axis.size), dtype=float)
    off_diagonal = difference != 0
    out[off_diagonal] = ((-1.0) ** difference[off_diagonal]) / (
        difference[off_diagonal] * (axis[1] - axis[0])
    )
    return 0.5 * (out - out.T)


def _align_phases(states: np.ndarray) -> np.ndarray:
    """Align adjacent channel phases by maximizing their overlap."""
    aligned = np.array(states, dtype=complex, copy=True)
    for i in range(1, len(aligned)):
        overlap = aligned[i - 1].conj().T @ aligned[i]
        if aligned.shape[-1] > 1 and np.min(np.abs(np.diag(overlap))) < 1e-7:
            left, _, right = np.linalg.svd(overlap)
            aligned[i] = aligned[i] @ (right.conj().T @ left.conj().T)
        else:
            phase = np.diag(overlap)
            phase = np.where(np.abs(phase) > 1e-14, phase / np.abs(phase), 1.0)
            aligned[i] *= phase.conj()[None, :]
    return aligned


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

    def product_projector(self) -> np.ndarray:
        return np.repeat((self.sigma > self.dividing_surface).astype(float), self.n_channels)

    def channel_model(self, *, reactant: str = "asymptotic") -> SubspaceModel:
        """Return the same ``V(sigma,Q)`` for the RPMD half."""
        sigma, q, values = self.sigma, self.q, self.potential_values
        kappa = self.curvature.copy()

        from scipy.interpolate import RegularGridInterpolator
        interpolant = RegularGridInterpolator((sigma, q), values, bounds_error=False, fill_value=None)

        def energy(points):
            points = np.asarray(points, dtype=float)
            clipped = np.column_stack((np.clip(points[..., 0].reshape(-1), sigma[0], sigma[-1]),
                                       np.clip(points[..., 1].reshape(-1), q[0], q[-1])))
            return interpolant(clipped).reshape(points.shape[:-1])

        def metric(points):
            points = np.asarray(points, dtype=float)
            s = np.clip(points[..., 0], sigma[0], sigma[-1])
            normal = np.clip(points[..., 1], q[0], q[-1])
            kap = np.interp(s, sigma, kappa)
            return 1.0 / (1.0 + kap * normal) ** 2

        masses = [float(self.provenance.get("sigma_mass", 1.0)),
                  float(self.provenance.get("normal_mass", 1.0))]
        return model_from_function(
            energy, (sigma, q), masses=masses,
            dividing_surface=self.dividing_surface, reactant=reactant,
            reactant_reference=self.reactant_reference,
            provenance={**self.provenance, "kind": "adiabatic_channels_rpmd",
                         "position_dependent_inverse_metric": True},
            metric=metric,
        )

    rpmd_model = channel_model

    def exact_rate(self, beta: float, *, cap_width: float | None = None,
                   cap_strength: float | None = None, energy_floor: float | None = None):
        """Exact thermal rate from the dense channel Hamiltonian."""
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
        eps_sigma = absorbing_potential(self.sigma, width=width, strength=strength)
        eps = np.repeat(eps_sigma, self.n_channels)
        product = self.product_projector()
        ref = int(np.argmin(abs(self.sigma - self.reactant_reference)))
        shift = float(np.min(np.linalg.eigvalsh(self.hamiltonian)))

        def partition(energies, vectors):
            z = float(np.sum(np.exp(-beta * (self.energies[ref] - shift))))
            return np.sqrt(sigma_mass / (2.0 * np.pi * beta)) * z, {
                "normalization": "channel sum at reactant reference per unit sigma",
                "reference_sigma": float(self.sigma[ref]), "channels": self.n_channels,
            }

        flux, partition_value, info = _dense_rate_matrix(
            self.hamiltonian, eps, product, beta, shift, partition=partition,
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
                     "position_dependent_metric": bool(np.any(abs(self.curvature) > 0))})
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
) -> SubspaceChannels:
    """Construct ``sigma``-dependent normal eigenchannels.

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
    tq = sinc_dvr_kinetic(q, masses[1])
    energies = np.empty((sigma.size, n_channels), dtype=float)
    states = np.empty((sigma.size, q.size, n_channels), dtype=complex)
    for i in range(sigma.size):
        e, u = eigh(tq + np.diag(values[i]))
        energies[i] = e[:n_channels]
        states[i] = u[:, :n_channels]
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
    q_operator = states.conj().transpose(0, 2, 1) @ (q[:, None] * states)
    curvature_couplings = kappa[:, None, None] * q_operator
    metric_couplings = np.empty_like(curvature_couplings)
    for i, kap in enumerate(kappa):
        denominator = (1.0 + kap * q) ** 2
        if np.min(denominator) <= 1e-8:
            raise ValueError("curvature makes the reaction-path metric singular on the Q grid")
        metric_couplings[i] = states[i].conj().T @ ((1.0 / denominator)[:, None] * states[i])
        metric_couplings[i] = 0.5 * (metric_couplings[i] + metric_couplings[i].conj().T)

    if not include_nonadiabatic:
        derivative_couplings[...] = 0.0
        metric_couplings[...] = np.eye(n_channels)[None, :, :]
    d0 = _derivative_matrix(sigma)
    n = n_channels
    covariant = np.kron(d0, np.eye(n, dtype=complex)) + _block_diag(derivative_couplings)
    metric = _block_diag(metric_couplings)
    base = np.kron(sinc_dvr_kinetic(sigma, masses[0]), np.eye(n))
    correction = 0.5 * (
        covariant.conj().T @ metric @ covariant
        - np.kron(d0.conj().T @ d0, np.eye(n))
    )
    hamiltonian = base + correction + _block_diag(
        np.array([np.diag(e) for e in energies], complex)
    )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    info = {
        "kind": "adiabatic_channels", "method": method,
        "sigma_points": int(sigma.size), "normal_points": int(q.size),
        "sigma_mass": float(masses[0]), "normal_mass": float(masses[1]),
        "n_channels": n_channels, "curvature_max": float(np.max(np.abs(kappa))),
        "phase_aligned": True, "metric": "(1+kappa*Q)^-2",
        "include_nonadiabatic": bool(include_nonadiabatic),
    }
    info.update(dict(provenance or {}))
    return SubspaceChannels(sigma, q, energies, states, derivative_couplings,
                            curvature_couplings, metric_couplings, kappa,
                            hamiltonian, values, float(dividing_surface), float(reference), info)


def assemble_channel_hamiltonian(channels: SubspaceChannels) -> np.ndarray:
    """Return a fresh copy of the assembled channel matrix."""
    if not isinstance(channels, SubspaceChannels):
        raise TypeError("channels must be a SubspaceChannels object")
    return np.array(channels.hamiltonian, copy=True)


construct_subspace_channels = build_subspace_channels


def project_path_channels(potential: Callable, gradient: Any, path: Any, masses: Any,
                          axis: Any, *, q_axis: Any = None, normal: Any = None,
                          n_channels: int = 4, curvature: Any = None, **kwargs):
    """Build channels from a Cartesian reaction path.

    ``axis`` is the mass-weighted arc-length grid and ``q_axis`` is the normal
    grid.  A two-dimensional path uses its Frenet normal; higher-dimensional
    paths require an explicit ``normal`` array.
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
        normal_points = np.asarray(normal, dtype=float)
        if normal_points.shape != points.shape or not np.isfinite(normal_points).all():
            raise ValueError("normal must have the same shape as path")
        normal_points /= np.linalg.norm(normal_points, axis=1)[:, None]
    normal_sigma = np.column_stack([np.interp(sigma, chord, normal_points[:, j]) for j in range(points.shape[1])])

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
    "exact_channel_rate",
]
