"""RPMD rate constants for 1-3D subspace Hamiltonians.

This is the denominator of the subspace correction ``C_S = k_QM/k_RPMD``.
It uses the *same* ``V_S`` and the same reactant normalization as
:func:`pyrpmd.analysis.subspace_quantum.exact_quantum_rate`, so ``C_S`` only
carries the RPMD method error inside ``S``.

Bennett--Chandler factorization with the ring-polymer centroid on axis 0:

``k = k_QTST * kappa``.  ``k_QTST`` comes from thermodynamic integration of
the exact centroid mean force ``dW/ds = <(1/P) sum_j dV/ds_{j,0}>`` (the
constraint is linear in a Cartesian coordinate, so no Jacobian or Fixman
term appears).  Constrained ensembles are sampled by Hamiltonian Monte Carlo
in normal-mode coordinates with exact free ring-polymer propagation and a
Metropolis test, so the only sampling bias is statistical.  ``kappa(t)`` is
the transmission coefficient from unconstrained RPMD trajectories started
on the dividing surface with paired ``+p``/``-p`` momenta.

Normalization, identical to the quantum solver:

``asymptotic``  ``k_QTST = exp(-beta [W(s_dagger) - W(s_ref)]) / sqrt(2 pi beta m_0)``
``well``        ``k_QTST = 1 / (sqrt(2 pi beta m_0) int_{s_lo}^{s_dagger} exp(-beta [W(s) - W(s_dagger)]) ds)``

Atomic units with hbar = 1 and ``beta_P = beta / P`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .subspace_model import SubspaceModel


def normal_mode_matrix(nbeads: int) -> np.ndarray:
    """Real orthonormal bead-to-normal-mode matrix; column 0 is the centroid."""
    p = int(nbeads)
    j = np.arange(p)
    matrix = np.empty((p, p))
    matrix[:, 0] = 1.0 / np.sqrt(p)
    for column in range(1, p):
        if 2 * column < p:
            matrix[:, column] = np.sqrt(2.0 / p) * np.cos(2 * np.pi * j * column / p)
        elif 2 * column == p:
            matrix[:, column] = np.where(j % 2, -1.0, 1.0) / np.sqrt(p)
        else:
            matrix[:, column] = np.sqrt(2.0 / p) * np.sin(2 * np.pi * j * column / p)
    return matrix


def free_ring_frequencies(nbeads: int, beta: float) -> np.ndarray:
    """``omega_k = 2 (P/beta) sin(pi k / P)`` matching :func:`normal_mode_matrix`."""
    p = int(nbeads)
    return 2.0 * (p / float(beta)) * np.sin(np.pi * np.arange(p) / p)


@dataclass(frozen=True)
class RingPolymerRateResult:
    """RPMD rate of a subspace model with its statistical decomposition."""

    beta: float
    nbeads: int
    rate: float
    rate_standard_error: float
    qtst_rate: float
    qtst_log_standard_error: float
    transmission: float
    transmission_standard_error: float
    windows: np.ndarray
    mean_force: np.ndarray
    mean_force_standard_error: np.ndarray
    free_energy: np.ndarray
    time: np.ndarray
    kappa: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("windows", "mean_force", "mean_force_standard_error", "free_energy", "time", "kappa"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def log_rate_standard_error(self) -> float:
        return float(self.rate_standard_error / self.rate) if self.rate > 0 else float("inf")

    def to_dict(self) -> dict:
        from .subspace_quantum import _json_safe

        return _json_safe({
            "beta": self.beta, "nbeads": self.nbeads, "rate": self.rate,
            "rate_standard_error": self.rate_standard_error, "qtst_rate": self.qtst_rate,
            "qtst_log_standard_error": self.qtst_log_standard_error,
            "transmission": self.transmission, "transmission_standard_error": self.transmission_standard_error,
            "windows": self.windows, "mean_force": self.mean_force,
            "mean_force_standard_error": self.mean_force_standard_error,
            "free_energy": self.free_energy, "diagnostics": self.diagnostics,
        })


class _RingPolymer:
    """Vectorized ring polymers ``[batch, bead, dimension]`` in normal modes."""

    def __init__(self, model: SubspaceModel, beta: float, nbeads: int):
        self.model = model
        self.beta = float(beta)
        self.p = int(nbeads)
        self.d = model.dimension
        self.beta_p = self.beta / self.p
        self.mass = np.asarray(model.masses, dtype=float)
        self.c = normal_mode_matrix(self.p)
        self.omega = free_ring_frequencies(self.p, self.beta)
        self.momentum_scale = np.sqrt(self.mass / self.beta_p)

    def beads(self, modes):
        return np.einsum("jk,bkd->bjd", self.c, modes)

    def modes(self, beads):
        return np.einsum("jk,bjd->bkd", self.c, beads)

    def potential(self, modes):
        energy, gradient = self.model.energy_and_gradient(self.beads(modes))
        if not np.isfinite(energy).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("subspace potential returned non-finite values")
        return energy.sum(axis=1), gradient

    def spring(self, modes):
        return 0.5 * np.sum(self.mass[None, None, :] * (self.omega[None, :, None] * modes) ** 2, axis=(1, 2))

    def kinetic(self, momenta):
        return 0.5 * np.sum(momenta**2 / self.mass[None, None, :], axis=(1, 2))

    def free_propagator(self, dt):
        w = self.omega[:, None] * np.ones((1, self.d))
        m = np.ones((self.p, 1)) * self.mass[None, :]
        cos = np.cos(w * dt)
        sin = np.sin(w * dt)
        with np.errstate(divide="ignore", invalid="ignore"):
            q_from_p = np.where(w > 0, sin / (m * w), dt / m)
        p_from_q = -m * w * sin
        return cos, q_from_p, p_from_q

    def step(self, modes, momenta, force, dt, propagator, constrained):
        momenta = momenta + 0.5 * dt * force
        cos, q_from_p, p_from_q = propagator
        modes, momenta = cos * modes + q_from_p * momenta, p_from_q * modes + cos * momenta
        energy, gradient = self.potential(modes)
        force = -self.modes(gradient)
        if constrained:
            force[:, 0, 0] = 0.0
        momenta = momenta + 0.5 * dt * force
        return modes, momenta, force, energy, gradient

    def sample_momenta(self, rng, batch, constrained):
        momenta = rng.normal(size=(batch, self.p, self.d)) * self.momentum_scale[None, None, :]
        if constrained:
            momenta[:, 0, 0] = 0.0
        return momenta


def _thermal_frequency(model: SubspaceModel, beta: float) -> float:
    values = model.values
    accessible = values <= min(values.max(), np.min(values[np.searchsorted(model.axes[0], model.dividing_surface)])
                               + 30.0 / beta)
    accessible = accessible | (values <= values.min() + 30.0 / beta)
    curvature = 0.0
    for axis, mass in enumerate(model.masses):
        if len(model.axes[axis]) < 3:
            continue
        h = model.axes[axis][1] - model.axes[axis][0]
        second = (np.roll(values, -1, axis) - 2 * values + np.roll(values, 1, axis)) / (h * h)
        interior = [slice(None)] * values.ndim
        interior[axis] = slice(1, -1)
        mask = accessible[tuple(interior)]
        if mask.any():
            curvature = max(curvature, float(np.max(np.abs(second[tuple(interior)][mask]))) / mass)
    return float(np.sqrt(max(curvature, 1e-30)))


def _integrate_profile(windows, mean_force):
    from scipy.interpolate import CubicSpline

    spline = CubicSpline(windows, mean_force).antiderivative()
    return spline


def ring_polymer_rate(
    model: SubspaceModel,
    beta: float,
    nbeads: int,
    *,
    windows: Any = 48,
    walkers: int = 32,
    burn_in: int = 200,
    sweeps: int = 400,
    leapfrog_steps: int = 8,
    hmc_dt: float | None = None,
    trajectories: int = 512,
    md_dt: float | None = None,
    md_time: float | None = None,
    plateau_fraction: float = 0.3,
    bootstrap: int = 400,
    seed: int = 0,
) -> RingPolymerRateResult:
    """RPMD rate of ``model`` by centroid TI and recrossing trajectories.

    ``windows`` is a count or explicit increasing axis-0 grid ending at the
    dividing surface; the default starts at ``reactant_reference``
    (asymptotic) or at the first grid point (well).  ``walkers`` independent
    ring polymers per window run ``burn_in + sweeps`` HMC trajectories of
    ``leapfrog_steps`` steps; the window standard error comes from walker
    means.  ``trajectories`` configurations from the dividing-surface window
    start paired ``+p``/``-p`` RPMD trajectories of length ``md_time``.
    Default time steps follow the largest thermally accessible frequency of
    ``V_S``; free ring-polymer motion is integrated exactly.
    """
    if not isinstance(model, SubspaceModel):
        raise TypeError("model must be a SubspaceModel")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    if isinstance(nbeads, bool) or int(nbeads) != nbeads or nbeads < 1:
        raise ValueError("nbeads must be a positive integer")
    for name, value, low in (("walkers", walkers, 2), ("sweeps", sweeps, 2), ("burn_in", burn_in, 0),
                             ("leapfrog_steps", leapfrog_steps, 1), ("trajectories", trajectories, 2)):
        if isinstance(value, bool) or int(value) != value or value < low:
            raise ValueError(f"{name} must be an integer of at least {low}")
    if not 0 < plateau_fraction <= 1:
        raise ValueError("plateau_fraction must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    surface = model.dividing_surface
    start = model.reactant_reference if model.reactant == "asymptotic" else float(model.axes[0][0])
    if np.ndim(windows) == 0:
        count = int(windows)
        if count < 4:
            raise ValueError("at least four TI windows are required")
        grid = np.linspace(start, surface, count)
    else:
        grid = np.asarray(windows, dtype=float)
        if grid.ndim != 1 or len(grid) < 4 or np.any(np.diff(grid) <= 0) or not np.isclose(grid[-1], surface):
            raise ValueError("windows must be increasing, contain four points and end at the dividing surface")
        if model.reactant == "asymptotic" and not np.isclose(grid[0], model.reactant_reference):
            raise ValueError("asymptotic windows must start at reactant_reference")
    ring = _RingPolymer(model, beta, int(nbeads))
    frequency = _thermal_frequency(model, beta)
    hmc_step = 0.3 / frequency if hmc_dt is None else float(hmc_dt)
    md_step = 0.1 / frequency if md_dt is None else float(md_dt)
    if md_time is None:
        length = float(model.axes[0][-1] - model.axes[0][0])
        md_time = 0.3 * length * np.sqrt(beta * model.masses[0])
    md_steps = max(10, int(np.ceil(float(md_time) / md_step)))

    # --- constrained-centroid HMC for every window in one batch ---------
    nw, nwalk = len(grid), int(walkers)
    batch = nw * nwalk
    centers = np.repeat(grid, nwalk)
    positions = np.zeros((batch, ring.p, ring.d))
    positions[:, :, 0] = centers[:, None]
    if ring.d > 1:
        # Start transverse coordinates at the lowest grid energy of the slice.
        for i, value in enumerate(grid):
            index = int(np.clip(np.searchsorted(model.axes[0], value), 0, model.shape[0] - 1))
            flat = np.unravel_index(int(np.argmin(model.values[index])), model.shape[1:])
            positions[i * nwalk:(i + 1) * nwalk, :, 1:] = [model.axes[a + 1][flat[a]] for a in range(ring.d - 1)]
    modes = ring.modes(positions)
    energy, gradient = ring.potential(modes)
    force = -ring.modes(gradient)
    force[:, 0, 0] = 0.0
    propagator = ring.free_propagator(hmc_step)
    accepted_total, proposals = 0, 0
    samples = np.zeros(batch)
    sample_count = 0
    for sweep in range(int(burn_in) + int(sweeps)):
        momenta = ring.sample_momenta(rng, batch, True)
        start_h = ring.kinetic(momenta) + ring.spring(modes) + energy
        trial = (modes, momenta, force, energy, gradient)
        steps = max(1, int(round(leapfrog_steps * rng.uniform(0.5, 1.5))))
        for _ in range(steps):
            trial = ring.step(trial[0], trial[1], trial[2], hmc_step, propagator, True)
        new_modes, new_momenta, new_force, new_energy, new_gradient = trial
        delta = ring.kinetic(new_momenta) + ring.spring(new_modes) + new_energy - start_h
        accept = np.log(rng.uniform(size=batch)) < -ring.beta_p * np.nan_to_num(delta, nan=np.inf)
        modes = np.where(accept[:, None, None], new_modes, modes)
        force = np.where(accept[:, None, None], new_force, force)
        energy = np.where(accept, new_energy, energy)
        gradient = np.where(accept[:, None, None], new_gradient, gradient)
        rate = float(accept.mean())
        if sweep < burn_in:
            # Robbins--Monro step-size adaptation towards 75% acceptance.
            hmc_step *= float(np.exp(0.5 * (rate - 0.75)))
            propagator = ring.free_propagator(hmc_step)
        else:
            accepted_total += int(accept.sum())
            proposals += batch
            samples += gradient[:, :, 0].mean(axis=1)
            sample_count += 1
    walker_means = (samples / sample_count).reshape(nw, nwalk)
    mean_force = walker_means.mean(axis=1)
    mean_force_error = walker_means.std(axis=1, ddof=1) / np.sqrt(nwalk)

    # --- free energy and QTST with parametric bootstrap -----------------
    def qtst_log(force_profile, grid=grid):
        antiderivative = _integrate_profile(grid, force_profile)
        free = antiderivative(grid) - antiderivative(grid[0])
        prefactor = -0.5 * np.log(2 * np.pi * beta * model.masses[0])
        if model.reactant == "asymptotic":
            return prefactor - beta * (free[-1] - free[0]), free
        fine = np.linspace(grid[0], grid[-1], 2001)
        relative = antiderivative(fine) - antiderivative(grid[-1])
        weights = np.exp(-beta * relative)
        integral = float(np.sum(0.5 * (weights[1:] + weights[:-1]) * np.diff(fine)))
        return prefactor - np.log(integral), free

    log_qtst, free_energy = qtst_log(mean_force)
    draws = [qtst_log(mean_force + mean_force_error * rng.normal(size=nw))[0] for _ in range(int(bootstrap))]
    log_qtst_statistical = float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan")
    # Quadrature error of the cubic-spline TI: compare with every second
    # window (error ~ h^4, so the fine-grid error is about |difference|/15).
    coarse = slice(None, None, 2) if (nw - 1) % 2 == 0 else np.r_[0:nw:2, nw - 1]
    if len(grid[coarse]) >= 4:
        log_coarse = qtst_log(mean_force[coarse], grid[coarse])[0]
        quadrature_error = abs(log_qtst - log_coarse) / 15.0
    else:
        quadrature_error = float("nan")
    log_qtst_error = float(np.hypot(log_qtst_statistical, np.nan_to_num(quadrature_error)))

    # --- dividing-surface configurations and recrossing ----------------
    first = (nw - 1) * nwalk
    surf_modes = modes[first:].copy()
    surf_force = force[first:].copy()
    surf_energy = energy[first:].copy()
    surf_gradient = gradient[first:].copy()
    collected = []
    while sum(len(c) for c in collected) < trajectories:
        momenta = ring.sample_momenta(rng, nwalk, True)
        start_h = ring.kinetic(momenta) + ring.spring(surf_modes) + surf_energy
        trial = (surf_modes, momenta, surf_force, surf_energy, surf_gradient)
        for _ in range(max(1, int(round(leapfrog_steps * rng.uniform(0.5, 1.5))))):
            trial = ring.step(trial[0], trial[1], trial[2], hmc_step, propagator, True)
        delta = ring.kinetic(trial[1]) + ring.spring(trial[0]) + trial[3] - start_h
        accept = np.log(rng.uniform(size=nwalk)) < -ring.beta_p * np.nan_to_num(delta, nan=np.inf)
        surf_modes = np.where(accept[:, None, None], trial[0], surf_modes)
        surf_force = np.where(accept[:, None, None], trial[2], surf_force)
        surf_energy = np.where(accept, trial[3], surf_energy)
        surf_gradient = np.where(accept[:, None, None], trial[4], surf_gradient)
        collected.append(surf_modes.copy())
    starts = np.concatenate(collected)[: int(trajectories)]
    ntraj = len(starts)
    momenta = ring.sample_momenta(rng, ntraj, False)
    velocity = momenta[:, 0, 0] / (np.sqrt(ring.p) * model.masses[0])
    md_modes = np.concatenate([starts, starts])
    md_momenta = np.concatenate([momenta, -momenta])
    md_energy, md_gradient = ring.potential(md_modes)
    md_force = -ring.modes(md_gradient)
    md_propagator = ring.free_propagator(md_step)
    initial_h = ring.kinetic(md_momenta) + ring.spring(md_modes) + md_energy
    side = np.empty((md_steps + 1, 2 * ntraj), dtype=bool)
    side[0] = md_modes[:, 0, 0] / np.sqrt(ring.p) > surface
    for step in range(1, md_steps + 1):
        md_modes, md_momenta, md_force, md_energy, _ = ring.step(
            md_modes, md_momenta, md_force, md_step, md_propagator, False)
        side[step] = md_modes[:, 0, 0] / np.sqrt(ring.p) > surface
    final_h = ring.kinetic(md_momenta) + ring.spring(md_modes) + md_energy
    energy_error = float(np.max(np.abs(final_h - initial_h)) * ring.beta_p)
    forward, backward = side[:, :ntraj].astype(float), side[:, ntraj:].astype(float)
    numerator = velocity[None, :] * (forward - backward)
    denominator = float(np.sum(np.abs(velocity)))
    kappa = numerator.sum(axis=1) / denominator
    time = np.arange(md_steps + 1) * md_step
    tail = max(2, int(np.ceil(plateau_fraction * len(time))))
    plateau = float(kappa[-tail:].mean())
    boot = []
    for _ in range(int(bootstrap)):
        index = rng.integers(0, ntraj, size=ntraj)
        boot.append(numerator[-tail:, index].sum(axis=1).mean() / np.sum(np.abs(velocity[index])))
    kappa_error = float(np.std(boot, ddof=1)) if len(boot) > 1 else float("nan")
    plateau_drift = float(abs(kappa[-tail] - kappa[-1]))

    qtst = float(np.exp(log_qtst))
    rate = qtst * plateau
    relative = np.hypot(log_qtst_error, kappa_error / plateau if plateau > 0 else np.inf)
    diagnostics = {
        "nbeads": int(nbeads),
        "windows": int(nw),
        "walkers_per_window": nwalk,
        "hmc_sweeps": int(sweeps),
        "hmc_burn_in": int(burn_in),
        "hmc_time_step": float(hmc_step),
        "hmc_acceptance": float(accepted_total / max(proposals, 1)),
        "trajectories": int(ntraj),
        "md_time_step": float(md_step),
        "md_time": float(time[-1]),
        "max_energy_error_beta_p": energy_error,
        "plateau_fraction": float(plateau_fraction),
        "plateau_drift": plateau_drift,
        "thermal_frequency": frequency,
        "qtst_log_statistical_error": log_qtst_statistical,
        "ti_quadrature_log_error": float(quadrature_error),
        "barrier_free_energy": float(free_energy[-1] - free_energy[0]) if model.reactant == "asymptotic" else float(free_energy[-1] - free_energy.min()),
        "spline_outside_fraction": model.interpolant.outside_fraction if model.function is None else 0.0,
        "normalization": model.reactant,
        "units": "axis-0 length per time" if model.reactant == "asymptotic" else "per time",
        "physical_converged": False,
    }
    return RingPolymerRateResult(beta, int(nbeads), rate, float(rate * relative), qtst, log_qtst_error,
                                 plateau, kappa_error, grid, mean_force, mean_force_error,
                                 free_energy, time, kappa, diagnostics)


__all__ = ["RingPolymerRateResult", "free_ring_frequencies", "normal_mode_matrix", "ring_polymer_rate"]
