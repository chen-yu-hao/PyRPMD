"""Independent Jacobi-coordinate equilibrium reference for classical H + H2.

The physical Cartesian measure is integrated explicitly. This is benchmark
tooling, not a replacement for PyRPMD dynamics. Both equivalent H-exchange
channels are represented by u = abs(cos(theta)) in [0, 1].
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import logsumexp, roots_legendre

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.classical_tst import MASS_AU, quadrature
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.constants import AU_RATE_TO_CM3_S, KB_AU


def cartesian(r_com, bond, cosine):
    """One orientation representative; overall rotations integrate to 8pi²."""
    r_com, bond, cosine = np.broadcast_arrays(r_com, bond, cosine)
    q = np.zeros(r_com.shape + (3, 3))
    q[..., 0, 0] = r_com * np.sqrt(np.maximum(0., 1-cosine*cosine))
    q[..., 0, 2] = r_com * cosine
    q[..., 1, 2] = -bond/2
    q[..., 2, 2] = bond/2
    return q


def coordinate_metric(r_com, bond, cosine, *, rinf=30., mass=MASS_AU):
    """Independent xi, dxi/dR and full Cartesian inverse-mass metric.

    Reduced masses are mu_R=2m/3 and mu_r=m/2. The angular derivative
    contributes (1-u²)(1/(mu_R R²)+1/(mu_r r²)) (dxi/du)².
    """
    r_com, bond, cosine = np.broadcast_arrays(r_com, bond, cosine)
    near = np.sqrt(r_com*r_com + bond*bond/4-r_com*bond*cosine)
    s0 = rinf-r_com
    s1 = bond-near
    denominator = s0-s1
    d_near_r = (r_com-bond*cosine/2)/near
    d_near_bond = (bond/4-r_com*cosine/2)/near
    d_xi_r = (s1-s0*d_near_r)/denominator**2
    d_xi_bond = s0*(1-d_near_bond)/denominator**2
    d_xi_u = s0*r_com*bond/(2*near*denominator**2)
    metric = (d_xi_r*d_xi_r/(2*mass/3)
              + d_xi_bond*d_xi_bond/(mass/2)
              + (1-cosine*cosine)*(1/((2*mass/3)*r_com*r_com)
                                    + 1/((mass/2)*bond*bond))*d_xi_u*d_xi_u)
    return s0/denominator, d_xi_r, metric


def radial_roots(xi, bond, cosine, *, rinf=30.):
    """All physical R roots of xi at fixed bond/u, plus validity masks.

    Squaring xi*d_near = xi*r+(1-xi)*(Rinf-R) introduces spurious roots;
    the unsquared equation, positive denominator, and positive R are checked.
    A second physical root can occur beyond the transition state and is kept.
    """
    bond, cosine = np.broadcast_arrays(bond, cosine)
    xi = float(xi)
    if abs(xi) < 1e-12:
        r = np.full((2,)+bond.shape, rinf)
        valid = np.zeros_like(r, dtype=bool)
        valid[0] = True
        return r, valid
    a = 1-xi
    b = xi*bond+a*rinf
    aa = 2*xi-1
    bb = 2*a*b-xi*xi*bond*cosine
    cc = xi*xi*bond*bond/4-b*b
    discriminant = bb*bb-4*aa*cc
    root_disc = np.sqrt(np.maximum(discriminant, 0.))
    with np.errstate(divide='ignore', invalid='ignore'):
        if abs(aa) < 1e-13:
            candidates = np.stack((-cc/bb, np.full_like(bond, np.nan)))
        else:
            # Stable quadratic formula followed by c/(a*r) for the other root.
            q = -.5*(bb+np.copysign(root_disc, bb))
            candidates = np.stack((q/aa, cc/q))
        near = np.sqrt(candidates*candidates+bond*bond/4-candidates*bond*cosine)
        residual = xi*near-(b-a*candidates)
        denominator = rinf-candidates-bond+near
    valid = (np.isfinite(candidates) & (candidates > 0) & (discriminant >= 0)
             & (denominator > 0) & (np.abs(residual) < 1e-9))
    valid[1] &= ~np.isclose(candidates[0], candidates[1], rtol=1e-12, atol=1e-12)
    return candidates, valid


def flux_density(potential, xi, *, temperature=300., mass=MASS_AU,
                 rinf=30., radial_order=256, angular_order=128,
                 radial_domain=(.5, 8.), minimum_pair_distance=.5,
                 energy_origin=0.):
    """Unnormalized 1D rate-flux density divided by constant 16pi².

    rho(xi) = integral R²*r²*|dR/dxi|*sqrt(g/(2pi beta))*exp(-beta V) dr du.
    The factor two from folding cos(theta) cancels in endpoint ratios. Every
    positive physical R root is included; no assumed channel prefactor is fit.
    """
    if (temperature <= 0 or mass <= 0 or rinf <= radial_domain[1]
            or not .4 <= radial_domain[0] < radial_domain[1]
            or radial_order < 8 or angular_order < 8):
        raise ValueError('invalid positive physical-channel quadrature settings')
    beta = 1/(KB_AU*temperature)
    rn, rw = roots_legendre(radial_order)
    un, uw = roots_legendre(angular_order)
    low, high = radial_domain
    r = low+(rn+1)*(high-low)/2
    rw = rw*(high-low)/2
    u, uw = (un+1)/2, uw/2
    rr, uu = np.meshgrid(r, u, indexing='ij')
    weights = rw[:, None]*uw[None, :]
    density = []
    second_roots = []
    for x in np.atleast_1d(xi):
        candidates, valid = radial_roots(x, rr, uu, rinf=rinf)
        integrals = []
        for candidate, admissible in zip(candidates, valid):
            candidate = np.where(admissible, candidate, rinf)
            near = np.sqrt(candidate*candidate+rr*rr/4-candidate*rr*uu)
            far = np.sqrt(candidate*candidate+rr*rr/4+candidate*rr*uu)
            admissible &= (near >= minimum_pair_distance) & (far >= minimum_pair_distance)
            idx = np.flatnonzero(admissible)
            if not len(idx):
                integrals.append(0.)
                continue
            rc, rb, uc = candidate.ravel()[idx], rr.ravel()[idx], uu.ravel()[idx]
            q = cartesian(rc, rb, uc)
            result = potential.evaluate(q)
            if np.any(result.status):
                raise ValueError('PES failed on a physical-channel quadrature point')
            actual_xi, dxi_dr, metric = coordinate_metric(rc, rb, uc, rinf=rinf, mass=mass)
            if not np.allclose(actual_xi, x, atol=1e-10, rtol=0):
                raise ValueError('conditional geometry did not match xi')
            integrand = (rc*rc*rb*rb*np.sqrt(metric/(2*np.pi*beta))/np.abs(dxi_dr)
                         *np.exp(-beta*(result.energy-energy_origin)))
            integrals.append(float(np.dot(weights.ravel()[idx], integrand)))
        density.append(sum(integrals))
        second_roots.append(min(integrals)/sum(integrals) if sum(integrals) else 0.)
    return np.asarray(density), np.asarray(second_roots)


def umbrella_moments(grid, density, centers, *, beta, stiffness=30.):
    """Exact biased-coordinate moments from a converged conditional density."""
    grid, density, centers = map(np.asarray, (grid, density, centers))
    dx = grid[:, None]-centers[None, :]
    logweight = np.log(density)[:, None]-.5*beta*stiffness*dx*dx
    logweight -= logweight.max(axis=0)
    weights = np.exp(logweight)
    normalizer = trapezoid(weights, grid, axis=0)
    shift = trapezoid(weights*dx, grid, axis=0)/normalizer
    means = centers+shift
    variances = trapezoid(weights*(grid[:, None]-means)**2, grid, axis=0)/normalizer
    return means, variances


def gaussian_ui_from_moments(means, variances, centers, grid, *, beta, stiffness=30.):
    """Gaussian UI's infinite-sampling limit, not a sampled trajectory."""
    means, variances, centers, grid = map(np.asarray, (means, variances, centers, grid))
    logs = -.5*np.log(2*np.pi*variances)[None, :]-.5*(grid[:, None]-means)**2/variances
    weights = np.exp(logs-logsumexp(logs, axis=1)[:, None])
    gradient = np.sum(weights*((grid[:, None]-means)/(beta*variances)
                               -stiffness*(grid[:, None]-centers)), axis=1)
    free = np.zeros_like(grid)
    free[1:] = np.cumsum(.5*np.diff(grid)*(gradient[:-1]+gradient[1:]))
    return free-free[0], gradient


def reference(output, library, *, radial_order=256, angular_order=128, grid_step=.001):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    potential = BKMP2Potential(library)
    origin = potential.radial([1.4014719755, 99., 99.])[0]
    beta = 1/(KB_AU*300.)
    # Soft umbrellas require their full tails; a narrow product-side cutoff
    # can otherwise create a spurious Gaussian-UI bias.
    grid = np.linspace(-.2, 1.25, round(1.45/grid_step)+1)
    started = time.perf_counter()
    rho, secondary = flux_density(potential, grid, radial_order=radial_order,
                                  angular_order=angular_order, energy_origin=origin)
    centers = np.linspace(-.05, 1.05, 111)
    means, variances = umbrella_moments(grid, rho, centers, beta=beta)
    analysis_grid = np.linspace(0, 1.02, 1021)
    exact = -np.log(np.interp(analysis_grid, grid, rho)/np.interp(0., grid, rho))/beta
    ui, derivative = gaussian_ui_from_moments(means, variances, centers, analysis_grid, beta=beta)
    prefactor = 4*np.pi*30**2/np.sqrt(2*np.pi*beta*(2*MASS_AU/3))
    direct = quadrature(potential, energy_origin=origin)
    rate_at_one = prefactor*np.interp(1., grid, rho)/np.interp(0., grid, rho)*AU_RATE_TO_CM3_S
    arrays = dict(xi=grid, density=rho, secondary_branch_fraction=secondary, centers=centers,
                  means=means, variances=variances, analysis_xi=analysis_grid,
                  exact_pmf_hartree=exact, gaussian_ui_pmf_hartree=ui,
                  gaussian_ui_derivative_hartree=derivative)
    stiffness_checks = []
    for stiffness, spacing in ((2., .02), (2., .01), (30., .01)):
        trial_centers = np.arange(-.06,1.061,spacing) if spacing == .02 else centers
        trial_means, trial_variances = umbrella_moments(grid,rho,trial_centers,beta=beta,stiffness=stiffness)
        trial_ui,_ = gaussian_ui_from_moments(trial_means,trial_variances,trial_centers,analysis_grid,
                                             beta=beta,stiffness=stiffness)
        trial_rate = prefactor*np.exp(-beta*trial_ui.max())*AU_RATE_TO_CM3_S
        key = f'k{stiffness:g}_spacing{spacing:g}'
        arrays[key+'_centers'] = trial_centers
        arrays[key+'_means'] = trial_means
        arrays[key+'_variances'] = trial_variances
        arrays[key+'_ui_pmf'] = trial_ui
        stiffness_checks.append({'stiffness_hartree':stiffness,'center_spacing':spacing,
                                 'maximum_xi':float(analysis_grid[trial_ui.argmax()]),
                                 'infinite_sample_ui_rate_cm3_molecule_s':float(trial_rate),
                                 'ratio_to_direct_tst':float(trial_rate/direct['classical_tst_cm3_molecule_s'])})
    np.savez_compressed(output/'conditional-reference.npz', **arrays)
    np.savetxt(output/'conditional-reference.csv', np.column_stack((analysis_grid, exact, ui)),
               delimiter=',', header='xi,exact_pmf_hartree,gaussian_ui_pmf_hartree', comments='')
    np.savetxt(output/'umbrella-moments.csv', np.column_stack((centers, means, variances)),
               delimiter=',', header='center,mean,variance', comments='')
    report = {'schema':'pyrpmd.h3.classical-conditional-reference.v1',
              'created_utc':datetime.now(timezone.utc).isoformat(), 'temperature_K':300.,
              'radial_order':radial_order, 'angular_order':angular_order, 'grid_step':grid_step,
              'xi_integration_range':[float(grid[0]),float(grid[-1])],
              'radial_domain_bohr':[.5,8.], 'minimum_pair_distance_bohr':.5,
              'rate_at_xi_one_cm3_molecule_s':rate_at_one,
              'independent_tst_quadrature_cm3_molecule_s':direct['classical_tst_cm3_molecule_s'],
              'ratio_to_independent_tst_quadrature':rate_at_one/direct['classical_tst_cm3_molecule_s'],
              'exact_maximum_xi':float(analysis_grid[exact.argmax()]),
              'exact_maximum_rate_cm3_molecule_s':float(prefactor*np.exp(-beta*exact.max())*AU_RATE_TO_CM3_S),
              'gaussian_ui_maximum_xi':float(analysis_grid[ui.argmax()]),
              'gaussian_ui_infinite_sampling_rate_cm3_molecule_s':float(prefactor*np.exp(-beta*ui.max())*AU_RATE_TO_CM3_S),
              'stiffness_checks':stiffness_checks,
              'elapsed_seconds':time.perf_counter()-started,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'potential_metadata':potential.metadata,
              'scientific_scope':'Independent classical conditional quadrature; not a trajectory or quantum RPMD validation.'}
    (output/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--radial-order',type=int,default=256)
    parser.add_argument('--angular-order',type=int,default=128)
    parser.add_argument('--grid-step',type=float,default=.001)
    args=parser.parse_args()
    reference(args.output,args.library,radial_order=args.radial_order,
              angular_order=args.angular_order,grid_step=args.grid_step)


if __name__=='__main__':
    main()
