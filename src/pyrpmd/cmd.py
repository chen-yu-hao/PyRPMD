"""Reference partially adiabatic CMD and conservative centroid force matching.

Atomic units and the sum-bead Hamiltonian at beta/P are used throughout.
PA-CMD changes the *kinetic* normal-mode masses; spring and physical forces
remain unchanged. Its internal modes are thermostatted, its production
centroid is not. Finite adiabatic separation and timestep convergence remain
system-dependent approximations, not a guarantee of exact quantum dynamics.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .dynamics import State, _state_clock
from .potentials.base import Potential, PESResult, coordinates, evaluate_checked


def _real(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(raw, dtype=float, copy=True)


def _scalar(value, name, *, positive=True):
    a = _real(value, name)
    if a.ndim or (a <= 0 if positive else a < 0):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} scalar")
    return float(a)


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _normal_modes(beads, beta):
    j = np.arange(beads)
    rows = [np.ones(beads) / np.sqrt(beads)]
    omega = [0.0]
    for k in range(1, (beads + 1) // 2):
        rows.extend([np.sqrt(2 / beads) * np.cos(2 * np.pi * k * j / beads),
                     np.sqrt(2 / beads) * np.sin(2 * np.pi * k * j / beads)])
        omega.extend([2 * beads / beta * np.sin(np.pi * k / beads)] * 2)
    if beads % 2 == 0:
        rows.append((-1.0) ** j / np.sqrt(beads))
        omega.append(2 * beads / beta)
    return np.asarray(rows), np.asarray(omega)


class PACMDDynamics:
    """PA-CMD reference dynamics with exact free-mode BAOAB splitting.

    ``target_frequency`` is the common *angular* frequency Omega of all
    noncentroid free-ring modes, in inverse atomic time. Their fictitious
    masses are ``m_i * (omega_k / Omega)**2``; the centroid mass stays physical.
    ``State.p`` stores the bead-basis representation of these canonical
    momenta, so internal bead velocities are **not** p/m. Use ``velocities``.

    ``step`` applies gamma_k=2*friction_scale*Omega to internal modes only.
    ``sample_step`` additionally thermostats the centroid for equilibration.
    ``hamiltonian_step`` disables thermostats for numerical diagnostics.
    The ordinary native RPMD kernels cannot propagate these fictitious masses.
    This implementation deliberately uses its own NumPy stepper and never
    silently calls those kernels.

    See Hone, Rossky and Voth, JCP 124, 154103 (2006),
    https://doi.org/10.1063/1.2186636.
    """

    def __init__(self, potential, masses, beta, nbeads, dt, *, target_frequency,
                 friction_scale=1.0, seed=0):
        self.potential = potential
        self.masses = _real(masses, "masses")
        if self.masses.ndim != 1 or not len(self.masses) or np.any(self.masses <= 0):
            raise ValueError("masses must be a nonempty positive vector")
        self.beta = _scalar(beta, "beta")
        self.nbeads = _integer(nbeads, "nbeads")
        self.dt = _scalar(dt, "dt")
        self.target_frequency = _scalar(target_frequency, "target_frequency")
        self.friction_scale = _scalar(friction_scale, "friction_scale", positive=False)
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self.shape = (self.nbeads, len(self.masses), 3)
        self.transform, self.ring_frequencies = _normal_modes(self.nbeads, self.beta)
        factors = np.ones(self.nbeads)
        factors[1:] = (self.ring_frequencies[1:] / self.target_frequency) ** 2
        self.mode_masses = factors[:, None, None] * self.masses[None, :, None]
        self.free_frequencies = np.full(self.nbeads, self.target_frequency)
        self.free_frequencies[0] = 0.0
        if (not np.isfinite(self.mode_masses).all() or np.any(self.mode_masses <= 0)
                or not np.isfinite(self.ring_frequencies).all()
                or not np.isfinite(self.dt * self.target_frequency)):
            raise ValueError("PA-CMD mass/frequency scale overflows; check parameters")
        for a in (self.masses, self.transform, self.ring_frequencies,
                  self.mode_masses, self.free_frequencies):
            a.flags.writeable = False
        self._identity = (self.beta, self.nbeads, self.dt, self.target_frequency, self.friction_scale)

    def _check(self, state):
        if (self.beta, self.nbeads, self.dt, self.target_frequency, self.friction_scale) != self._identity:
            raise ValueError("PA-CMD parameters are immutable; create a new dynamics object")
        if not isinstance(state, State) or state.q.shape != self.shape or state.p.shape != self.shape:
            raise ValueError(f"PA-CMD requires a State with shape {self.shape}")
        for a in (state.q, state.p):
            if (a.dtype != np.float64 or not a.flags.c_contiguous or not a.flags.writeable
                    or not np.isfinite(a).all()):
                raise ValueError("state arrays must be writable contiguous finite float64")
        if np.shares_memory(state.q, state.p):
            raise ValueError("state q and p cannot share memory")
        return _state_clock(state.step, state.time)

    def _modes(self, value):
        return np.einsum("kb,bac->kac", self.transform, value)

    def _beads(self, value):
        return np.ascontiguousarray(np.einsum("kb,kac->bac", self.transform, value))

    def sample_momenta(self):
        """Draw canonical fictitious momenta at beta/P, including the centroid."""
        pk = self.rng.normal(size=self.shape) * np.sqrt(self.mode_masses * self.nbeads / self.beta)
        return self._beads(pk)

    def velocities(self, momenta):
        p = _real(momenta, "momenta")
        if p.shape != self.shape:
            raise ValueError(f"momenta must have shape {self.shape}")
        return self._beads(self._modes(p) / self.mode_masses)

    def _free(self, qk, pk, h):
        phase = self.free_frequencies * h
        cosine = np.cos(phase)[:, None, None]
        drift = (h * np.sinc(phase / np.pi))[:, None, None]
        force = (self.free_frequencies ** 2)[:, None, None]
        old_q = qk.copy()
        qk[:] = cosine * old_q + drift * pk / self.mode_masses
        pk[:] = cosine * pk - self.mode_masses * force * drift * old_q

    def _force(self, q, bias):
        force = evaluate_checked(self.potential, q).forces.copy()
        if bias is not None:
            energy, extra = bias.energy_force(q)
            extra = _real(extra, "bias force")
            if extra.shape != q.shape or not np.isfinite(energy):
                raise ValueError("bias must return a finite energy and forces matching q")
            force += extra
        return force

    def _advance(self, state, bias, gamma_centroid, internal_thermostat):
        step, time = self._check(state)
        next_time = time + self.dt
        if not math.isfinite(next_time):
            raise FloatingPointError("nonfinite integration clock")
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            q = state.q.copy()
            p = state.p.copy()
            p += 0.5 * self.dt * self._force(q, bias)
            qk, pk = self._modes(q), self._modes(p)
            self._free(qk, pk, self.dt / 2)
            gamma = np.zeros(self.nbeads)
            if internal_thermostat:
                gamma[1:] = 2 * self.friction_scale * self.target_frequency
            gamma[0] = gamma_centroid
            active = np.flatnonzero(gamma > 0)
            if len(active):
                decay = np.exp(-gamma[active] * self.dt)[:, None, None]
                variance = self.mode_masses[active] * self.nbeads / self.beta
                amplitude = np.sqrt(-np.expm1(-2 * gamma[active] * self.dt))[:, None, None]
                noise = self.rng.normal(size=(len(active), len(self.masses), 3))
                pk[active] = decay * pk[active] + amplitude * np.sqrt(variance) * noise
            self._free(qk, pk, self.dt / 2)
            q, p = self._beads(qk), self._beads(pk)
            p += 0.5 * self.dt * self._force(q, bias)
            if not np.isfinite(q).all() or not np.isfinite(p).all():
                raise FloatingPointError("nonfinite PA-CMD integration state")
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        state.q[:] = q
        state.p[:] = p
        state.step = step + 1
        state.time = next_time
        return state

    def step(self, state, bias=None):
        """Production PA-CMD step; the physical centroid is unthermostatted."""
        return self._advance(state, bias, 0.0, True)

    def sample_step(self, state, bias=None, *, gamma_centroid):
        """Equilibrate at beta/P with a positive centroid thermostat friction."""
        gamma = _scalar(gamma_centroid, "gamma_centroid")
        return self._advance(state, bias, gamma, True)

    def hamiltonian_step(self, state, bias=None):
        """Thermostat-free reversible map of the fictitious-mass Hamiltonian."""
        return self._advance(state, bias, 0.0, False)

    def energy(self, state, bias=None):
        """Sum-bead fictitious Hamiltonian (not a quantum energy estimator)."""
        self._check(state)
        qk, pk = self._modes(state.q), self._modes(state.p)
        energy = float(np.sum(evaluate_checked(self.potential, state.q).energy))
        energy += float(np.sum(pk * pk / (2 * self.mode_masses)))
        energy += float(0.5 * np.sum(self.masses[None, :, None]
                                     * self.ring_frequencies[:, None, None] ** 2 * qk ** 2))
        if bias is not None:
            extra = _real(bias.energy_force(state.q)[0], "bias energy")
            if extra.ndim:
                raise ValueError("bias energy must be a scalar")
            energy += float(extra)
        return energy

    def metadata(self):
        """Scientific settings for recording/checkpoint fingerprinting."""
        return {"method": "pa-cmd", "integrator": "exact-normal-mode-baoab",
                "beta_au": self.beta, "beads": self.nbeads, "dt_au": self.dt,
                "target_angular_frequency_au": self.target_frequency,
                "internal_friction_scale": self.friction_scale,
                "mass_convention": "m_0=m; m_k=m*(omega_k/Omega)^2",
                "production_centroid_thermostat": False,
                "implementation": "numpy_reference"}


def centroid_force_training_data(positions, forces):
    """Return Cartesian centroids and mean physical forces from PIMD frames.

    Both inputs have shape [frame,bead,atom,3], in bohr and hartree/bohr.
    Forces must be *physical* forces, excluding ring springs and bias forces.
    Frames must sample the unbiased canonical quantum distribution at the
    intended temperature. Averaging gives noisy conditional centroid-force
    targets; the returned arrays are suitable for an external conservative ML
    fit. No claim of statistical independence is made for saved frames.
    """
    q, f = _real(positions, "positions"), _real(forces, "forces")
    if q.ndim != 4 or q.shape[-1] != 3 or min(q.shape) < 1 or f.shape != q.shape:
        raise ValueError("positions and forces need identical [frame,bead,atom,3] shapes")
    return {"centroids": q.mean(axis=1), "mean_forces": f.mean(axis=1),
            "beads": q.shape[1], "frames": q.shape[0],
            "force_convention": "negative gradient of centroid PMF; mean physical bead force"}


@dataclass
class CentroidForceMatch:
    """A conservative quadratic fit and its explicitly in-sample diagnostics."""

    potential: "QuadraticCentroidPotential"
    force_rmse: float
    rank: int
    parameters: int
    frames: int


class QuadraticCentroidPotential(Potential):
    """Cartesian quadratic centroid PMF for f-CMD model tests/local fitting.

    This finite basis is suitable only for a local harmonic basin. It is not
    translation/rotation/permutation invariant and must not be presented as a
    general molecular ML force field. Use a conservative invariant ML model
    fitted to ``centroid_force_training_data`` for general systems.
    """

    metadata = {"provider": "quadratic_centroid_pmf", "thread_safety": "reentrant",
                "execution_device": "cpu", "model_scope": "local_harmonic_centroid_pmf"}

    def __init__(self, origin, linear, hessian, *, beta):
        self.origin = _real(origin, "origin")
        if self.origin.ndim != 2 or self.origin.shape[1] != 3 or not len(self.origin):
            raise ValueError("origin must be [atom,3]")
        size = self.origin.size
        self.linear = _real(linear, "linear")
        self.hessian = _real(hessian, "hessian")
        if self.linear.shape != (size,) or self.hessian.shape != (size, size):
            raise ValueError("linear/hessian shapes must match the flattened origin")
        if not np.allclose(self.hessian, self.hessian.T, rtol=1e-12, atol=1e-14):
            raise ValueError("hessian must be symmetric for a conservative force")
        self.beta = _scalar(beta, "beta")
        self.metadata = {**type(self).metadata, "centroid_pmf_beta_au": self.beta}
        for a in (self.origin, self.linear, self.hessian):
            a.flags.writeable = False

    def evaluate(self, x):
        q = coordinates(x)
        if q.shape[1:] != self.origin.shape:
            raise ValueError("geometry does not match the fitted centroid PMF")
        dx = (q - self.origin).reshape(len(q), -1)
        hd = dx @ self.hessian
        energy = dx @ self.linear + 0.5 * np.sum(dx * hd, axis=1)
        force = -(hd + self.linear).reshape(q.shape)
        return PESResult(energy, force)


def fit_quadratic_centroid_pmf(positions, forces, *, beta, ridge=0.0):
    """Conservative force matching for a *local harmonic* f-CMD potential.

    Fits the independent coefficients of a symmetric Hessian jointly, rather
    than symmetrizing an unconstrained force regression afterwards. ``ridge``
    penalizes Hessian coefficients, not the constant force. In-sample RMSE is
    only a diagnostic; held-out trajectories and production validation remain
    necessary. The resulting potential is temperature-specific and must be
    propagated with P=1 and physical masses, not path-integral quantized again.

    General force-matching f-CMD: Hone, Izvekov and Voth, JCP 122, 054105
    (2005), https://doi.org/10.1063/1.1836731.
    """
    data = centroid_force_training_data(positions, forces)
    beta = _scalar(beta, "beta")
    regularization = _scalar(ridge, "ridge", positive=False)
    q, f = data["centroids"], data["mean_forces"]
    origin = q.mean(axis=0)
    x = (q - origin).reshape(len(q), -1)
    y = -f.reshape(len(q), -1)
    d = x.shape[1]
    # This intentionally bounded reference basis must not allocate a giant
    # dense least-squares problem when given a molecular training dataset.
    if d > 96:
        raise ValueError("local quadratic fit supports at most 32 atoms; use an external invariant ML fit")
    pairs = [(i, j) for i in range(d) for j in range(i, d)]
    nparams = d + len(pairs)
    if len(q) * d * nparams > 20_000_000:
        raise ValueError("quadratic design matrix is too large; subsample or use an external fit")
    design = np.zeros((len(q), d, nparams))
    for i in range(d):
        design[:, i, i] = 1.0
    for col, (i, j) in enumerate(pairs, d):
        design[:, i, col] = x[:, j]
        if i != j:
            design[:, j, col] = x[:, i]
    a, target = design.reshape(-1, nparams), y.reshape(-1)
    rank = int(np.linalg.matrix_rank(a))
    if rank < nparams and regularization == 0:
        raise ValueError("centroid training geometries do not identify the quadratic PMF; add data or explicit ridge")
    if regularization:
        penalty = np.zeros((len(pairs), nparams))
        penalty[:, d:] = np.eye(len(pairs)) * np.sqrt(regularization)
        solve_a, solve_b = np.vstack([a, penalty]), np.r_[target, np.zeros(len(pairs))]
    else:
        solve_a, solve_b = a, target
    coefficients = np.linalg.lstsq(solve_a, solve_b, rcond=None)[0]
    hessian = np.zeros((d, d))
    for value, (i, j) in zip(coefficients[d:], pairs):
        hessian[i, j] = hessian[j, i] = value
    fitted = QuadraticCentroidPotential(origin, coefficients[:d], hessian, beta=beta)
    rmse = float(np.sqrt(np.mean((a @ coefficients - target) ** 2)))
    return CentroidForceMatch(fitted, rmse, rank, nparams, len(q))


def ibi_update(correction, target_density, sampled_density, *, beta,
               damping=1.0, regularization=0.0):
    """One IBI update for precomputed CMD/QCMD coordinate distributions.

    Densities must use the same coordinate, grid, normalization, and measure.
    For f-CMD, intramolecular coordinates are evaluated at Cartesian centroids;
    f-QCMD instead averages internal coordinates over beads. h-CMD combines
    explicit selections. This utility does not reconstruct quasi-centroids,
    generate distributions, run the IBI iterations, or implement constrained
    adiabatic QCMD. No automatic QCMD/h-CMD method claim follows from using it.
    The explicit bent-triatomic f-QCMD geometry, distribution, model and
    production APIs are provided separately in :mod:`pyrpmd.qcmd`.
    """
    value = _real(correction, "correction")
    target, sampled = _real(target_density, "target_density"), _real(sampled_density, "sampled_density")
    if value.ndim != 1 or len(value) < 2 or target.shape != value.shape or sampled.shape != value.shape:
        raise ValueError("correction and densities must be equal vectors of length >= 2")
    if np.any(target < 0) or np.any(sampled < 0) or target.max() <= 0 or sampled.max() <= 0:
        raise ValueError("densities must be nonnegative with positive support")
    inverse_temperature = _scalar(beta, "beta")
    damping = _scalar(damping, "damping")
    regularization = _scalar(regularization, "regularization", positive=False)
    if damping > 1:
        raise ValueError("damping must lie in (0,1]")
    floor = regularization * max(float(target.max()), float(sampled.max()))
    if floor == 0 and (np.any(target == 0) or np.any(sampled == 0)):
        raise ValueError("zero-density bins need explicit regularization or a common supported grid")
    update = value + (damping / inverse_temperature) * (np.log(sampled + floor) - np.log(target + floor))
    if not np.isfinite(update).all():
        raise ValueError("IBI update overflowed")
    return update
