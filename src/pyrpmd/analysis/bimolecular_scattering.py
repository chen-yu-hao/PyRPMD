"""Auditable 3D gas-phase rate aggregation, not a new exact quantum dynamics.

All input quantities use atomic units. Unknown/failed trajectories are retained
as rate bounds. Partial waves, electronic weights and reactant normalization
are explicit. The functions do not infer convergence from a small error bar.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp

KB_AU = 3.166811563e-6
BOHR_CM = 5.29177210903e-9
TIME_S = 2.4188843265864e-17
RATE_AU_TO_CM3_S = BOHR_CM ** 3 / TIME_S


def _positive(x, name):
    if isinstance(x, (bool, np.bool_)) or not np.isscalar(x) or not np.isfinite(x) or x <= 0:
        raise ValueError(f'{name} must be a positive finite scalar')
    return float(x)


def _array(x, name):
    a = np.asarray(x)
    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain finite real values')
    return np.asarray(a, dtype=float)


@dataclass(frozen=True)
class CollisionFlux:
    energy: np.ndarray
    impact_parameter: np.ndarray
    importance_weight: np.ndarray
    mean_speed: float
    area: float
    beta: float
    proposal_beta: float


def sample_collision_flux(n, temperature, reduced_mass, bmax, *, seed=0,
                          proposal_temperature_ratio=1.0):
    """Sample beta^2 E exp(-beta E), and b db uniformly in an impact disk.

    k(T) = sqrt(8/(pi beta mu)) pi*bmax^2 E_proposal[w * I_react].
    Unlike the collinear flux ensemble, the energy density includes E.
    """
    if isinstance(n, (bool, np.bool_)) or not np.isscalar(n) or not np.isfinite(n) or int(n) != n or n < 1:
        raise ValueError('n must be a positive integer')
    T = _positive(temperature, 'temperature'); mu = _positive(reduced_mass, 'reduced_mass')
    bmax = _positive(bmax, 'bmax'); ratio = _positive(proposal_temperature_ratio, 'proposal_temperature_ratio')
    beta = 1 / (KB_AU * T); bp = beta / ratio
    rng = np.random.default_rng(seed)
    E = rng.gamma(2., 1 / bp, int(n)); b = bmax * np.sqrt(rng.random(int(n)))
    w = (beta / bp)**2 * np.exp(-(beta - bp) * E)
    if not np.isfinite(w).all():
        raise ValueError('Importance weights overflow; use a warmer proposal')
    return CollisionFlux(E, b, w, float(np.sqrt(8 / (np.pi * beta * mu))),
                         float(np.pi * bmax*bmax), beta, bp)


def thermal_impact_rate(outcomes, weights, *, temperature, reduced_mass, bmax,
                        electronic_weight=1.0, symmetry_factor=1.0,
                        impact_parameters=None):
    """Rates for independent flux samples; outcomes {-2,-1,0,1}.

    -2=invalid and -1=unfinished are both unknown. The lower/upper difference
    concerns only the sampled impact disk, NOT the uncomputed b>bmax tail.
    A zero-observed-event sampling SE is not a precision certificate.
    """
    y = _array(outcomes, 'outcomes'); w = _array(weights, 'weights')
    if y.ndim != 1 or w.shape != y.shape or len(y) < 2 or np.any(w < 0):
        raise ValueError('Need matched independent outcome/weight vectors of length >=2')
    if not np.isin(y, [-2, -1, 0, 1]).all() or not np.any(w > 0):
        raise ValueError('Invalid outcome code or all-zero weights')
    T = _positive(temperature, 'temperature'); mu = _positive(reduced_mass, 'reduced_mass')
    bmax = _positive(bmax, 'bmax')
    factor = _positive(electronic_weight, 'electronic_weight') * _positive(symmetry_factor, 'symmetry_factor')
    pref = np.sqrt(8 * KB_AU * T / (np.pi * mu)) * np.pi*bmax*bmax * factor
    lo = w * (y == 1); hi = w * ((y == 1) | (y < 0))
    se = float(pref * lo.std(ddof=1) / np.sqrt(len(y)))
    result = {'rate_lower_au': float(pref * lo.mean()), 'rate_upper_au': float(pref * hi.mean()),
              'rate_lower_cm3_molecule_s': float(pref * lo.mean() * RATE_AU_TO_CM3_S),
              'rate_upper_cm3_molecule_s': float(pref * hi.mean() * RATE_AU_TO_CM3_S),
              'sampling_se_au': se, 'sampling_se_cm3_molecule_s': se * RATE_AU_TO_CM3_S,
              'sample_count': len(y), 'reactive': int((y == 1).sum()),
              'invalid': int((y == -2).sum()), 'unfinished': int((y == -1).sum()),
              'weight_ess': float(w.sum()**2 / (w @ w)),
              'reactive_weight_ess': float(lo.sum()**2 / (lo @ lo)) if np.any(lo) else 0.,
              'mean_importance_weight': float(w.mean()), 'bmax_bohr': bmax,
              'temperature_K': T, 'reduced_mass_au': mu,
              'electronic_weight': float(electronic_weight), 'symmetry_factor': float(symmetry_factor),
              'unit': 'cm^3 molecule^-1 s^-1', 'spatial_dimension': 3,
              'uncertainty_scope': 'independent trajectory sampling only; excludes PES, preparation and numerical bias',
              'impact_tail_beyond_bmax': 'not bounded', 'physical_converged': False}
    if impact_parameters is not None:
        b = _array(impact_parameters, 'impact_parameters')
        if b.shape != y.shape or np.any((b < 0) | (b > bmax*(1+1e-12))):
            raise ValueError('Impact parameters must be within sampled disk')
        outer = b >= .8*bmax
        result['outer_annulus'] = {'fraction_of_area': .36, 'n': int(outer.sum()),
            'reactive': int(((y == 1) & outer).sum()),
            'unknown': int(((y < 0) & outer).sum()),
            'rate_contribution_au': float(pref * (lo*outer).mean())}
    return result


def rovibrational_partition(levels, degeneracies, temperature, *, reference_energy=0.0):
    """Dimensionless bound-reactant internal partition; continuum not inferred."""
    E = _array(levels, 'levels'); g = _array(degeneracies, 'degeneracies')
    if E.ndim != 1 or g.shape != E.shape or not len(E) or np.any(g <= 0):
        raise ValueError('Matched levels and positive degeneracies required')
    if not np.isfinite(reference_energy): raise ValueError('Invalid energy reference')
    beta = 1 / (KB_AU * _positive(temperature, 'temperature'))
    return float(np.exp(logsumexp(np.log(g) - beta*(E-reference_energy))))


def partial_wave_thermal_rate(energies, crp_by_J, Js, *, temperature, reduced_mass,
                              internal_partition, reference_energy=0.0,
                              parity_summed=False, electronic_weight=1.0,
                              symmetry_factor=1.0):
    """Thermally integrate supplied TOTAL-energy CRPs, including (2J+1).

    N_J must already sum both parity blocks and all open entrance/product states,
    without (2J+1). No J-shifting, high-J interpolation or uncomputed tail is
    inserted. Supports arbitrary internal molecular dimension through supplied
    CRPs and internal partition; it does NOT solve those CRPs automatically.
    """
    E = _array(energies, 'energies'); N = _array(crp_by_J, 'crp_by_J'); J = _array(Js, 'Js')
    if E.ndim != 1 or len(E) < 2 or np.any(np.diff(E) <= 0):
        raise ValueError('Increasing total-energy grid required')
    if J.ndim != 1 or not len(J) or not np.array_equal(J, np.arange(len(J))):
        raise ValueError('Require contiguous integer J=0,...,Jmax; missing partial waves rejected')
    if N.shape != (len(J), len(E)) or np.any(N < 0) or not parity_summed:
        raise ValueError('Nonnegative matched CRPs and explicit parity_summed=True required')
    T = _positive(temperature, 'temperature'); mu = _positive(reduced_mass, 'reduced_mass')
    zint = _positive(internal_partition, 'internal_partition')
    if not np.isfinite(reference_energy): raise ValueError('Invalid energy zero')
    beta = 1/(KB_AU*T); ztrans = (mu/(2*np.pi*beta))**1.5
    boltz = np.exp(-beta*(E-reference_energy))
    if not np.isfinite(boltz).all(): raise ValueError('Bad energy zero causes overflow')
    contributions = (2*J+1)*trapezoid(N*boltz[None], E, axis=1)/(2*np.pi*ztrans*zint)
    factor = _positive(electronic_weight,'electronic_weight')*_positive(symmetry_factor,'symmetry_factor')
    contributions *= factor
    rate = float(contributions.sum())
    return {'rate_au': rate, 'rate_cm3_molecule_s': rate*RATE_AU_TO_CM3_S,
            'J_contributions_au': contributions.tolist(), 'Jmax': int(J[-1]),
            'last_J_fraction': float(contributions[-1]/rate) if rate>0 else None,
            'energy_min_au': float(E[0]), 'energy_max_au': float(E[-1]),
            'internal_partition': zint, 'translation_partition_per_bohr3': ztrans,
            'temperature_K': T, 'reference_energy_au': float(reference_energy),
            'electronic_weight': float(electronic_weight), 'symmetry_factor': float(symmetry_factor),
            'unit': 'cm^3 molecule^-1 s^-1', 'physical_converged': False,
            'uncomputed_J_and_energy_tails': 'not bounded',
            'rate_kind': 'finite-energy, finite-J thermal integral; not a certified full rate'}


def matched_bimolecular_rate_defect(full_rp, active_rp, active_q):
    """Experimental rate-level defect, enforcing identical molecular conventions.

    Does not establish the transfer hypothesis or repair missing partial waves.
    Collinear length/time rates and unmatched observables are rejected.
    """
    keys = ('reaction', 'temperature_K', 'unit', 'spatial_dimension',
            'reactant_normalization', 'electronic_weight', 'symmetry_factor', 'observable',
            'pes_identity', 'angular_momentum_coverage', 'state_population_convention')
    for item in (full_rp, active_rp, active_q):
        if not isinstance(item, Mapping) or any(k not in item for k in keys):
            raise ValueError('Missing rate identity metadata')
        if item['unit'] != 'cm^3 molecule^-1 s^-1' or item['spatial_dimension'] != 3:
            raise ValueError('Only matched three-dimensional bimolecular rates accepted')
        if not np.isfinite(item.get('rate', np.nan)) or item['rate'] < 0:
            raise ValueError('Finite nonnegative input rate required')
    if any(any(item[k] != full_rp[k] for k in keys) for item in (active_rp, active_q)):
        raise ValueError('Rate identities do not match')
    value = float(full_rp['rate'] + active_q['rate'] - active_rp['rate'])
    return {**{k:full_rp[k] for k in keys}, 'rate': value, 'negative_rate':value<0,
            'approximation':'matched-bimolecular-additive-defect',
            'physical_converged':False, 'exact_general_rate_claim':False}
