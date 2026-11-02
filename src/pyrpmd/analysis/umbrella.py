"""Umbrella bias energies, density histograms, and Gaussian integration."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..bias import displacement


def umbrella_reduced_potentials(samples, centers, stiffness, beta, periods=None):
    x = np.asarray(samples, float)
    centers = np.asarray(centers, float)
    stiffness = np.asarray(stiffness, float)
    if stiffness.ndim == 2:
        stiffness = np.broadcast_to(stiffness, (len(centers),) + stiffness.shape)
    values = []
    for center, k in zip(centers, stiffness):
        delta = displacement(x, center, periods)
        values.append(beta * 0.5 * np.einsum("ni,ij,nj->n", delta, k, delta))
    return np.asarray(values)


def _histogram(x, weights, edges, beta, min_count=1):
    mass, _ = np.histogramdd(x, bins=edges, weights=weights)
    count, _ = np.histogramdd(x, bins=edges)
    volumes = np.ones(tuple(len(e) - 1 for e in edges))
    for d, e in enumerate(edges):
        shape = [1] * len(edges)
        shape[d] = len(e) - 1
        volumes *= np.diff(e).reshape(shape)
    density = mass / volumes
    mask = (count < min_count) | (mass <= 0) | ~np.isfinite(density) | (density <= 0)
    f = np.full_like(mass, np.nan)
    f[~mask] = -np.log(density[~mask]) / beta
    if np.any(~mask):
        f[~mask] -= np.nanmin(f)
    return mass, density, count, f, mask


def umbrella_integration(samples, centers, kappa, beta, grid):
    """RPMDrate 1D Gaussian umbrella integration, using sampled mean/variance.

    Mathematical port of RPMDrate main.py computePotentialOfMeanForce,
    Copyright (c) 2012 Allen, Green, Suleimanov, MIT. Gaussian-window
    approximation is explicit; MBAR overlap must be inspected separately.
    """
    arrays = []
    for sample in samples:
        a = np.asarray(sample, float)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError("UI samples must be finite scalar streams with >=2 points")
        arrays.append(a)
    if not arrays:
        raise ValueError("at least one umbrella stream is required")
    if np.ndim(beta) != 0 or not np.isfinite(beta) or beta <= 0:
        raise ValueError("UI beta must be finite and positive")
    centers = np.asarray(centers, float)
    if centers.shape not in ((len(arrays),), (len(arrays), 1)):
        raise ValueError("one scalar center per umbrella stream is required")
    centers = centers.reshape(-1)
    kappa = np.asarray(kappa, float)
    if kappa.size == 1 and kappa.ndim <= 2:
        kappa = np.full(len(arrays), float(kappa.reshape(-1)[0]))
    elif kappa.shape in ((len(arrays),), (len(arrays), 1)):
        kappa = kappa.reshape(-1)
    else:
        raise ValueError("UI kappa must be scalar or one scalar per stream")
    grid = np.asarray(grid, float)
    if (
        not np.isfinite(centers).all()
        or not np.isfinite(kappa).all()
        or np.any(kappa <= 0)
        or grid.ndim != 1
        or len(grid) < 2
        or not np.isfinite(grid).all()
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("invalid umbrella centers, stiffness, or increasing UI grid")
    means = np.array([x.mean() for x in arrays])
    variances = np.array([x.var() for x in arrays])
    # An absolute variance cutoff depends on the chosen CV unit: the same
    # physical distribution must remain valid after a linear coordinate change.
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("zero/nonfinite-variance umbrella stream")
    logs = (
        np.log([len(x) for x in arrays])[None]
        - 0.5 * np.log(2 * np.pi * variances)[None]
    )
    logs = logs - 0.5 * (grid[:, None] - means[None]) ** 2 / variances[None]
    weights = np.exp(logs - logsumexp(logs, axis=1)[:, None])
    gradients = (grid[:, None] - means[None]) / (beta * variances[None]) - kappa[
        None
    ] * (grid[:, None] - centers[None])
    derivative = np.sum(weights * gradients, axis=1)
    f = np.zeros_like(grid)
    f[1:] = np.cumsum(0.5 * np.diff(grid) * (derivative[:-1] + derivative[1:]))
    f -= f.min()
    if not np.isfinite(f).all() or not np.isfinite(derivative).all():
        raise FloatingPointError("nonfinite Gaussian UI curve")
    return f, derivative
