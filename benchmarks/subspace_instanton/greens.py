"""Exact thermal rates from the cumulative reaction probability N(E).

The reaction coordinate is discretized on a uniform grid (second-order finite
differences, hopping ``t = 1 / (2 ds^2)``) and the transverse coordinate is
expanded in harmonic-oscillator states, so the Hamiltonian is a block
tridiagonal chain with channel matrices ``W(s)``.  The two ends continue as
semi-infinite uniform leads, whose effect is an exact energy-dependent
self-energy; no absorbing potential is involved.  Then (Landauer, Fisher--Lee)

    N(E) = Tr[Gamma_R G Gamma_L G^dagger],     k Q_r = (1 / 2 pi) int N(E) exp(-beta E) dE,

with ``G`` the retarded Green's function between the first and the last slice,
obtained by one forward recursion.  One N(E) gives the rate at every
temperature.  This reference is independent of the DVR / absorbing-potential
flux solver of ``pyrpmd.analysis.subspace_quantum``; the method follows the
package ``subspace_rpmd_code`` supplied for comparison, re-implemented here.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid


def _lead_self_energy(energies, onsite, t):
    """Retarded self-energy and broadening of a uniform semi-infinite lead, ``[E, M, M]``.

    ``onsite`` is the channel matrix of the lead.  In its eigenbasis each
    channel is a 1D chain with ``E - eps = 2 t (1 - cos ka)``: open channels
    have ``Sigma = -t exp(i ka)``, closed ones the decaying real root.
    """
    eps, vectors = np.linalg.eigh(onsite)
    c = 1.0 - (energies[:, None] - eps[None, :]) / (2.0 * t)
    open_channel = np.abs(c) <= 1.0
    z = np.where(open_channel, c + 1j * np.sqrt(np.clip(1.0 - c * c, 0.0, None)),
                 c - np.sign(c) * np.sqrt(np.clip(c * c - 1.0, 0.0, None)))
    sigma = -t * z
    self_energy = np.einsum("an,en,bn->eab", vectors, sigma, vectors)
    broadening = np.einsum("an,en,bn->eab", vectors, -2.0 * sigma.imag, vectors)
    return self_energy, broadening


def cumulative_probability(energies, channel_matrix, grid):
    """``N(E)`` for the chain with slices ``channel_matrix(s) -> [M, M]`` on the uniform ``grid``."""
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    ds = float(grid[1] - grid[0])
    t = 1.0 / (2.0 * ds * ds)
    first, last = np.atleast_2d(channel_matrix(grid[0])), np.atleast_2d(channel_matrix(grid[-1]))
    size = first.shape[0]
    eye = np.eye(size)
    sigma_left, gamma_left = _lead_self_energy(energies, first, t)
    sigma_right, gamma_right = _lead_self_energy(energies, last, t)
    ez = energies[:, None, None] * eye
    # g: Green's function of the last slice of the chain grown so far; G: its
    # block between that slice and the first one.
    g = np.linalg.inv(ez - (2.0 * t * eye + first) - sigma_left)
    transfer = g.copy()
    for i in range(1, len(grid)):
        block = ez - (2.0 * t * eye + np.atleast_2d(channel_matrix(grid[i]))) - t * t * g
        if i == len(grid) - 1:
            block = block - sigma_right
        g = np.linalg.inv(block)
        transfer = -t * (g @ transfer)
    return np.einsum("eab,ebc,ecd,ead->e", gamma_right, transfer, gamma_left, transfer.conj()).real


def oscillator_matrices(size, frequency):
    """``Q``, ``Q^2`` (exact, not a truncated product) and ``H_0`` in the oscillator basis of ``frequency``."""
    n = np.arange(size)
    q = np.zeros((size, size))
    off = np.sqrt((n[:-1] + 1) / (2.0 * frequency))
    q[n[:-1], n[:-1] + 1] = q[n[:-1] + 1, n[:-1]] = off
    q2 = np.diag((2 * n + 1) / (2.0 * frequency))
    off2 = np.sqrt((n[:-2] + 1) * (n[:-2] + 2)) / (2.0 * frequency)
    q2[n[:-2], n[:-2] + 2] = q2[n[:-2] + 2, n[:-2]] = off2
    return q, q2, np.diag(frequency * (n + 0.5))


def valley_channels(model, size):
    """Channel matrix ``W(s)`` of a :class:`models.Valley` in the oscillator basis of ``omega_inf``."""
    q, q2, h0 = oscillator_matrices(size, model.omega_inf)
    eye = np.eye(size)

    def matrix(s):
        w, g = float(model.omega(s)), float(model.floor(s))
        return h0 + 0.5 * (w * w - model.omega_inf**2) * q2 - w * w * g * q \
            + (0.5 * w * w * g * g + float(model.barrier(s))) * eye

    return matrix


def thermal_rate(energies, probability, beta, threshold, reactant_partition):
    """``k = (1 / 2 pi) int N(E) exp(-beta (E - threshold)) dE / Q_r`` with ``Q_r`` measured from ``threshold``."""
    weight = probability * np.exp(-beta * (energies - threshold))
    return float(trapezoid(weight, x=energies) / (2.0 * np.pi) / reactant_partition)


def valley_rates(model, betas, *, size=30, half_length=14.0, ds=0.02, points=600):
    """Exact rates (length / time) of a 2D valley model, reactants in the left lead.

    ``Q_r = sqrt(1 / (2 pi beta)) / (1 - exp(-beta omega_inf))`` per unit length,
    measured from the ground channel threshold.
    """
    grid = np.arange(-half_length, half_length + 0.5 * ds, ds)
    channels = valley_channels(model, size)
    threshold = float(np.linalg.eigvalsh(channels(grid[0]))[0])
    top = model.V0 + 0.5 * model.omega_inf + 1.0
    energies = threshold + np.linspace(1e-4, top + 28.0 / min(betas), points)
    probability = cumulative_probability(energies, channels, grid)
    rates = {}
    for beta in betas:
        partition = np.sqrt(1.0 / (2.0 * np.pi * beta)) / (1.0 - np.exp(-beta * model.omega_inf))
        rates[float(beta)] = thermal_rate(energies, probability, beta, threshold, partition)
    return rates, energies - threshold, probability


def barrier_rates(x, potential, betas, *, points=3000):
    """Exact rates of a 1D barrier tabulated on the uniform grid ``x`` (tending to zero at both ends)."""
    potential = np.asarray(potential, dtype=float)
    energies = np.linspace(1e-5, max(float(potential.max()), 0.5) + 30.0 / min(betas), points)
    probability = cumulative_probability(energies, lambda s, x0=x[0], h=x[1] - x[0]:
                                         np.array([[potential[int(round((s - x0) / h))]]]), x)
    return {float(b): thermal_rate(energies, probability, b, 0.0, np.sqrt(1.0 / (2.0 * np.pi * b))) for b in betas}
