"""H3 bond-diagonal RPMD surface: exact centroid mass metric and flux velocity.

The geometrical dividing surface is sigma=r(0,1)-r(1,2). This provides the
configuration-dependent positive Maxwell moment but does not supply the free
reactant normalization, constrained canonical ensemble or recrossing dynamics.
"""
from __future__ import annotations
import numpy as np


def bond_difference_gradient(centroid, bonds=((0, 1), (1, 2))):
    """Value/Cartesian gradient of the two bond-length difference in bohr."""
    q = np.asarray(centroid, float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("centroid must be finite [N,3]")
    if len(bonds) != 2:
        raise ValueError("expected two bonds")
    g = np.zeros_like(q)
    d = []
    for sign, bond in zip((1., -1.), bonds):
        a, b = map(int, bond)
        if a == b or min(a,b) < 0 or max(a,b) >= len(q):
            raise ValueError("invalid bond indices")
        vector = q[b]-q[a]
        length = float(np.linalg.norm(vector))
        if length <= 1e-12 or not np.isfinite(length):
            raise ValueError("undefined bond-length gradient")
        unit = vector/length
        g[a] -= sign*unit
        g[b] += sign*unit
        d.append(length)
    return d[0]-d[1], g


def bond_diagonal_flux_metric(q, masses, beta):
    """Return sigma and exact positive *half*-Maxwell normal flux velocity.

    Coordinates q in bohr: [P,N,3] or [N,3]. Masses in electron masses;
    beta in inverse hartree. No spurious bead-count factor.

    The unit normal in bond-distance CV space is (1,-1)/sqrt(2);
    G_nn=0.5*sum_a |d(r01-r12)/dqa|^2/m_a.
    Conditional speed = sqrt(G_nn/(2*pi*beta)).
    The angle-dependent G_nn MUST be averaged over the canonical dividing
    surface before replacing it by an effective constant. A single q is not
    a thermal velocity model for the entire surface.
    """
    q = np.asarray(q,float)
    if q.ndim == 3:
        if q.shape[0] < 1:
            raise ValueError("empty bead axis")
        q = q.mean(axis=0)
    if q.ndim != 2:
        raise ValueError("q must have [P,N,3] or [N,3] shape")
    m = np.asarray(masses,float)
    if m.shape != (len(q),) or np.any(m <= 0) or not np.isfinite(m).all():
        raise ValueError("masses must be finite positive values per atom")
    if not np.isscalar(beta) or isinstance(beta,(bool,np.bool_)) or not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite scalar")
    sigma, grad = bond_difference_gradient(q)
    metric = float(np.sum(grad**2/m[:,None])/2)
    return {
        "sigma_bohr":float(sigma),
        "inverse_normal_mass_au":metric,
        "positive_normal_speed_bohr_per_au_time":float(np.sqrt(metric/(2*np.pi*beta))),
    }
