"""Independent checks of the shipped separable cosine model; no real PES claims.

Run with the PyRPMD virtualenv Python, from any directory. This computes a
finite-P primitive path-integral centroid QTST reference independently of MD,
umbrella sampling, histograms, and the package MBAR solver.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import i0, ndtri
from scipy.stats import qmc
from pyrpmd.input import JobSpec
from pyrpmd.constants import ATOMIC_TIME_S
from pyrpmd.analysis.surface import basin_qtst
from pyrpmd.analysis.recrossing import summarize_recrossing


def reference(beta, mass, period, amplitude, beads, power, seed):
    # beta/P * spring = m*P/(2*beta) q.T L q. Eliminate the centroid
    # null mode and sample the remaining exact free-ring Gaussian modes.
    a = beta * amplitude / 2
    prefactor = np.sqrt(1 / (2 * np.pi * beta * mass)) / ATOMIC_TIME_S
    if beads == 1:
        return prefactor * np.exp(-a) / (period * i0(a))
    difference = np.eye(beads) - np.roll(np.eye(beads), 1, axis=1)
    values, vectors = np.linalg.eigh(difference.T @ difference)
    transform = vectors[:, 1:] * np.sqrt(beta / (mass * beads * values[1:]))
    uniform = qmc.Sobol(beads - 1, scramble=True, seed=seed).random_base2(power)
    internal = ndtri(uniform) @ transform.T
    phase = 2 * np.pi / period * internal
    c, s = np.cos(phase).mean(axis=1), np.sin(phase).mean(axis=1)
    # rho(L/2) / integral_{-L/2}^{L/2} rho(s) ds, with exp(-a) canceled.
    ratio = np.exp(-a * c).mean() / (period * i0(a * np.hypot(c, s)).mean())
    return prefactor * ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, system = JobSpec.from_file(args.repo / 'examples/surface/diffusion.inp').resolve()
    beta, mass, period, amplitude = config['beta'], system.masses[0], 6., .003
    records = []
    for beads in (1, 2, 4, 8, 16, 32, 64):
        for power in (14, 17):
            rates = [reference(beta, mass, period, amplitude, beads, power, seed)
                     for seed in (71, 211, 318, 417)]
            records.append(dict(beads=beads, points_per_scramble=2**power,
                                scrambles=4, k_qtst_s_inv=float(np.mean(rates)),
                                scramble_std_s_inv=float(np.std(rates, ddof=1)),
                                scramble_rates=rates))
    # Compare the package's basin integration against exact classical quadrature.
    grid_checks = []
    classical = reference(beta, mass, period, amplitude, 1, 0, 0)
    for bins in (20, 40, 80, 160, 320, 16001):
        edges = np.linspace(-4., 4., bins + 1)
        grid = (edges[:-1] + edges[1:]) / 2
        potential = amplitude / 2 * (1 - np.cos(2*np.pi*grid/period))
        rate = basin_qtst(grid, potential, beta, 1/mass, [-3., 3.], 3.) / ATOMIC_TIME_S
        grid_checks.append(dict(bins=bins, k_qtst_s_inv=rate,
                                relative_error=rate/classical-1))
    # Realizable per-pair indicators: repeated synchronized oscillations.
    nsteps = 400
    cycle = .5 + .4 * np.cos(2*np.pi*np.arange(25)/25)
    levels = np.tile(cycle, nsteps//25)
    children = [dict(chain=i//10, parent=i%10, pair=0,
                     numerator=(levels > (i+.5)/40).astype(float), denominator=1.)
                for i in range(40)]
    stats = summarize_recrossing(children, child_steps=nsteps, plateau_fraction=.25,
                                plateau_tolerance=.1, bootstrap=200,
                                rng=np.random.default_rng(17))
    if stats['plateau_detected']:
        raise AssertionError('Persistent transmission oscillations were labeled a plateau')
    result = dict(model='Shipped separable rigid-substrate cosine potential ONLY',
                  formula='sqrt(1/(2*pi*beta*m))*E[exp(-a*C)]/(L*E[I0(a*sqrt(C^2+S^2))]) / atomic_time',
                  path_integral='Primitive finite-P unwrapped ring, linear centroid, no winding/exchange',
                  beta_au=beta, mass_au=float(mass), period_bohr=period,
                  barrier_hartree=amplitude, classical_exact_qtst_s_inv=classical,
                  finite_P_reference=records, integration_checks=grid_checks,
                  oscillatory_negative_control=dict(plateau_detected=stats['plateau_detected'],
                      kappa=stats['kappa'], drift=stats['drift'],
                      drift_standard_error=stats['drift_standard_error'],
                      tail_min=float(stats['curve'][stats['start']:].min()),
                      tail_max=float(stats['curve'][stats['start']:].max())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
