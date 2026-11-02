#!/usr/bin/env python3
"""Independent PI references and actual main-engine trajectories.

Feldman & Hirshberg, JCP 159, 154107 (2023), Appendix B, Fig. 6:
two-dimensional ideal bosons, P=36, beta*hbar*omega=2 or 6.
Kapil et al., JCP 145, 234103 (2016): fourth-order Suzuki--Chin action.
Sarsa et al., JCP 113, 1366 (2000): open-chain ground-state projection.

The latter two are analytical method checks, not reproductions of water/helium
results. All computed reference numbers are independent calculations, not
digitized paper data. See the generated manifest and accompanying report.
Run with single-threaded BLAS and PYTHONPATH=src. Jobs save resumable snapshots
outside the tracked CSV/JSON results; a completed job is never silently rerun.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.pimd_methods import (SuzukiChinPotential, BosonicExchangePotential,
    PIGSPotential, GaussianTrialWavefunction, bosonic_exchange)
from pyrpmd.potentials.base import PESResult
from pyrpmd.potentials.builtin import HarmonicPotential

SEEDS = (48153, 275011, 910731)
ROOT = Path(__file__).resolve().parents[2]


def source_provenance():
    """Fingerprint the actual driver and the Python/native sources it uses.

    A changed driver or core requires a fresh output directory. In particular,
    old completed results must never acquire a new driver's identity just
    because ``--mode sample`` was invoked again.
    """
    paths = [Path(__file__), *(ROOT/name for name in (
        'src/pyrpmd/dynamics.py', 'src/pyrpmd/pimd_methods.py',
        'src/pyrpmd/path_integral.py', 'src/pyrpmd/potentials/base.py',
        'src/pyrpmd/potentials/builtin.py', 'src/pyrpmd/potentials/contraction.py'))]
    paths += [path for path in sorted((ROOT/'native').glob('*'))
              if path.suffix in {'.cpp', '.h', '.hpp'}]
    return dict(schema=1, source_sha256={
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths})


def job_key(case, seed):
    return f"{case['method']}_n{case['particles']}_b{case['beta']:g}_p{case['beads']}_dt{case['dt']:g}_s{seed}"


def check_completed(path, case, seed, provenance):
    """Read a completed result only if settings and source identity agree."""
    if not path.exists():
        return None
    existing = json.loads(path.read_text())
    if existing.get('case') != case or existing.get('seed') != seed:
        raise ValueError(f'Completed job parameters differ: {path}; use a fresh output directory')
    if existing.get('provenance') != provenance:
        raise ValueError(f'Completed job source provenance differs or is missing: {path}; use a fresh output directory')
    return existing


def check_checkpoint(path, case, seed, provenance):
    """Check metadata before restoring any arrays, RNG state or clock."""
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as saved:
        if not {'case_json', 'seed', 'provenance_json'} <= set(saved.files):
            raise ValueError(f'Checkpoint provenance is missing: {path}; use a fresh output directory')
        if json.loads(str(saved['case_json'])) != case or int(saved['seed']) != seed:
            raise ValueError(f'Checkpoint parameters differ: {path}; use a fresh output directory')
        if json.loads(str(saved['provenance_json'])) != provenance:
            raise ValueError(f'Checkpoint source provenance differs: {path}; use a fresh output directory')


def preflight_sampling(output, definitions, provenance):
    """Reject incompatible evidence before modifying the existing manifest."""
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get('provenance') != provenance:
            raise ValueError('Sampling manifest source provenance differs or is missing; '
                             'preserve the old evidence and use a fresh output directory')
        if existing.get('cases') != definitions or existing.get('seeds') != list(SEEDS):
            raise ValueError('Sampling manifest parameters differ; use a fresh output directory')
    elif (any((output/'jobs').glob('*.json')) or any((output/'jobs').glob('*.npz'))):
        raise ValueError('Sampling caches have no manifest; use a fresh output directory')
    for case in definitions:
        for seed in SEEDS:
            key = job_key(case, seed)
            check_completed(output/'jobs'/f'{key}.json', case, seed, provenance)
            check_checkpoint(output/'jobs'/f'{key}.npz', case, seed, provenance)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boson_energy(n, beta, beads=None, dimension=2, omega=1.):
    """Canonical ideal-boson partition recursion, independent of force code.

    A length-k permutation cycle has Z1(k beta). For the primitive action,
    Z1 becomes [2 sinh(k P asinh(beta*omega/(2P)))]**(-d). Differentiating
    log Z_N gives total energy. The harmonic scaling identity E=2<V>
    holds for both continuum and discrete actions.
    """
    if beads is None:
        a, derivative = beta*omega/2, omega/2
    else:
        x = beta*omega/(2*beads)
        a, derivative = beads*np.arcsinh(x), omega/(2*np.sqrt(1+x*x))
    log_z = np.zeros(n+1)
    energy = np.zeros(n+1)
    for count in range(1, n+1):
        k = np.arange(1, count+1)
        y = k*a
        log_single = -dimension*(y + np.log(-np.expm1(-2*y)))
        log_terms = log_single + log_z[count-k]
        normalization = logsumexp(log_terms)
        weights = np.exp(log_terms-normalization)
        log_z[count] = normalization - np.log(count)
        cycle_energy = dimension*k*derivative/np.tanh(y)
        energy[count] = np.dot(weights, energy[count-k]+cycle_energy)
    return float(energy[n])


def sc_variance(beta, beads, *, fourth_order):
    """Integrate the finite Gaussian action, using paper alpha=0 coefficients."""
    p = beads
    eye = np.eye(p)
    laplacian = 2*eye-np.roll(eye, 1, axis=0)-np.roll(eye, -1, axis=0)
    if fourth_order:
        weights = np.where(np.arange(p) % 2 == 0, 2/3, 4/3)
        correction = np.where(np.arange(p) % 2 == 0, 0., (beta/p)**2/9)
    else:
        weights, correction = np.ones(p), np.zeros(p)
    stiffness = (p/beta)**2*laplacian + np.diag(weights+2*correction)
    variance = np.diag(np.linalg.inv(stiffness)*p/beta)
    return float(variance[::2].mean() if fourth_order else variance.mean())


def pigs_variance(projection, beads, alpha=.25):
    tau = projection/(beads-1)
    laplacian = 2*np.eye(beads)-np.eye(beads, k=1)-np.eye(beads, k=-1)
    laplacian[0, 0] = laplacian[-1, -1] = 1
    weights = np.ones(beads)
    weights[[0, -1]] = .5
    precision = laplacian/tau + np.diag(tau*weights)
    precision[0, 0] += alpha
    precision[-1, -1] += alpha
    finite = np.linalg.inv(precision)[beads//2, beads//2]
    propagated_alpha = (alpha + np.tanh(projection/2))/(1+alpha*np.tanh(projection/2))
    return float(finite), float(1/(2*propagated_alpha))


class Quartic:
    def evaluate(self, q):
        return PESResult(np.sum(.5*q*q + .1*q**4, axis=(1, 2)), -q-.4*q**3)


class DimensionalNoise:
    """Restrict the driver to d Cartesian directions before every OU step.

    Dynamics and its native/free/force paths are unchanged. Inactive noise and
    initial positions/momenta are zero, so inactive coordinates stay zero at
    every substep (no post-step projection changes exchange probabilities).
    """
    def __init__(self, generator, dimension):
        self.generator, self.dimension = generator, dimension

    @property
    def bit_generator(self):
        return self.generator.bit_generator

    def normal(self, *args, **kwargs):
        values = self.generator.normal(*args, **kwargs)
        values[..., self.dimension:] = 0
        return values


def deterministic(output):
    output.mkdir(parents=True, exist_ok=True)
    rows, fits = [], []
    for beta in (1., 4., 10.):
        exact = .5/np.tanh(beta/2)
        for method in ('primitive', 'suzuki_chin'):
            points = []
            for p in (4, 8, 16, 32, 64, 128):
                value = sc_variance(beta, p, fourth_order=method == 'suzuki_chin')
                error = abs(value-exact)
                row = dict(method=method, beta=beta, beads=p, variance=value,
                           exact=exact, absolute_error=error, relative_error=error/exact)
                rows.append(row)
                if p >= 16:
                    points.append((p, error))
            slope = -np.polyfit(np.log([x[0] for x in points]), np.log([x[1] for x in points]), 1)[0]
            fits.append(dict(method=method, beta=beta, fitted_order=float(slope), fit_beads='16,32,64,128'))
    write_csv(output/'suzuki_chin_convergence.csv', rows)
    write_json(output/'suzuki_chin_orders.json', fits)
    q = np.random.default_rng(513).normal(scale=.7, size=(8, 2, 3))
    rows = []
    for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        model = SuzukiChinPotential(Quartic(), [1., 2.], 4., 8, fd_step=h)
        force = -q-.4*q**3
        expected = model.weights[:, None, None]*force + 2*model.coefficients[:, None, None]*(1+1.2*q*q)*force/np.array([1., 2.])[None, :, None]
        got = model.evaluate(q).forces
        rows.append(dict(fd_step=h, maximum_force_error=float(np.max(abs(got-expected))),
                         rms_force_error=float(np.sqrt(np.mean((got-expected)**2)))))
    write_csv(output/'suzuki_chin_finite_difference.csv', rows)
    rows = []
    for alpha in (.25, 2.):
        for projection in (2., 4., 8., 12.):
            for beads in (17, 33, 65, 129):
                finite, projected = pigs_variance(projection, beads, alpha)
                rows.append(dict(trial_alpha=alpha, projection_time=projection,
                    beads=beads, tau=projection/(beads-1), finite_bead_variance=finite,
                    continuum_projected_variance=projected, ground_variance=.5,
                    discretization_error=finite-projected, projection_error=projected-.5))
    write_csv(output/'pigs_convergence.csv', rows)
    rows = []
    for beta, n in itertools.chain(((6., n) for n in (2, 3, 4, 16, 32, 64)),
                                    ((b, 16) for b in (2., 3., 4., 5.))):
        continuum, discrete = boson_energy(n, beta), boson_energy(n, beta, 36)
        rows.append(dict(particles=n, beta=beta, beads=36, continuum_energy=continuum,
                         finite_bead_energy=discrete, relative_discretization_error=(discrete-continuum)/continuum))
    write_csv(output/'boson_paper_parameter_references.csv', rows)
    # NVE time-step and FD checks expose signs/factors beyond a static action.
    rows = []
    for h in (1e-3, 1e-5):
        for dt in (.04, .02, .01):
            potential = SuzukiChinPotential(Quartic(), [1.], 4., 8, fd_step=h)
            dyn = Dynamics(potential, [1.], 4., 8, dt, seed=151, native=True)
            state = State(np.random.default_rng(143).normal(scale=.2, size=(8, 1, 3)), dyn.sample_momenta())
            initial_q, initial_p = state.q.copy(), state.p.copy()
            initial_energy = dyn.energy(state)
            drift = 0.
            steps = round(20/dt)
            for step in range(steps):
                dyn.step(state)
                drift = max(drift, abs(dyn.energy(state)-initial_energy))
            state.p *= -1
            for step in range(steps):
                dyn.step(state)
            reversibility = max(np.max(abs(state.q-initial_q)), np.max(abs(state.p+initial_p)))
            rows.append(dict(fd_step=h, timestep=dt, propagation_time=20.,
                             max_relative_energy_drift=drift/abs(initial_energy),
                             max_reversibility_error=float(reversibility)))
    write_csv(output/'suzuki_chin_nve.csv', rows)


def cases(steps):
    common = dict(steps=steps, burn=steps//5, stride=10, thermostat='PILE-L',
                  integrator='cayley-bcb', gamma_centroid=1.)
    definitions = []
    for n, beta in ((2, 6.), (4, 6.), (16, 6.), (16, 2.)):
        definitions.append(dict(common, method='bosons', particles=n, beta=beta, beads=36,
                                dt=.02, dimension=2, reference=boson_energy(n, beta, 36),
                                continuum=boson_energy(n, beta)))
    # Paired seeds and equal simulation time for the explicit time-step check.
    definitions.append(dict(definitions[1], dt=.01, steps=2*steps, burn=2*(steps//5), stride=20))
    for dt in (.04, .02):
        definitions.append(dict(common, method='suzuki_chin', particles=1, beta=4., beads=8,
            dt=dt, dimension=1, reference=sc_variance(4., 8, fourth_order=True), continuum=.5/np.tanh(2)))
    definitions.append(dict(common, method='pigs', particles=1, beta=33*8/32, beads=33,
        dt=.02, dimension=1, projection=8., alpha=.25, reference=pigs_variance(8., 33)[0],
        continuum=pigs_variance(8., 33)[1]))
    return definitions


def run_job(case, seed, output, provenance=None):
    output = Path(output)
    current = source_provenance()
    if provenance is None:
        provenance = current
    elif provenance != current:
        raise ValueError('Benchmark sources changed after sampling preflight')
    key = job_key(case, seed)
    result_path = output/'jobs'/f'{key}.json'
    existing = check_completed(result_path, case, seed, provenance)
    checkpoint = output/'jobs'/f'{key}.npz'
    check_checkpoint(checkpoint, case, seed, provenance)
    if existing is not None:
        return existing
    masses, p, beta = np.ones(case['particles']), case['beads'], case['beta']
    physical = HarmonicPotential()
    if case['method'] == 'bosons':
        potential = BosonicExchangePotential(physical, masses, beta, p)
    elif case['method'] == 'suzuki_chin':
        potential = SuzukiChinPotential(physical, masses, beta, p)
    else:
        potential = PIGSPotential(physical, masses, p, case['projection'], GaussianTrialWavefunction(case['alpha']))
    dyn = Dynamics(potential, masses, beta, p, case['dt'], seed=seed, native=True, threads=1)
    dyn.rng = DimensionalNoise(dyn.rng, case['dimension'])
    state = State(np.zeros((p, len(masses), 3)), dyn.sample_momenta())
    values = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    elapsed = 0.
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            state = State(saved['q'], saved['p'], int(saved['step']), float(saved['time']))
            dyn.rng.bit_generator.state = json.loads(str(saved['rng']))
            values = list(saved['values'])
            elapsed = float(saved['elapsed'])
    start = time.perf_counter()
    for step in range(state.step+1, case['steps']+1):
        dyn.sample_step(state, gamma_centroid=case['gamma_centroid'])
        if step > case['burn'] and step % case['stride'] == 0:
            if np.any(state.q[..., case['dimension']:] != 0) or np.any(state.p[..., case['dimension']:] != 0):
                raise AssertionError('inactive Cartesian coordinate moved')
            if case['method'] == 'bosons':
                value = np.sum(state.q**2)/p  # 2<V>, total harmonic energy
            elif case['method'] == 'suzuki_chin':
                value = np.mean(state.q[::2, 0, 0]**2)
            else:
                value = state.q[p//2, 0, 0]**2
            values.append(float(value))
        if step % 5000 == 0 or step == case['steps']:
            if source_provenance() != provenance:
                raise ValueError('Benchmark sources changed during sampling; old checkpoint retained')
            temporary = checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, q=state.q, p=state.p, step=state.step, time=state.time,
                rng=json.dumps(dyn.rng.bit_generator.state), values=values,
                elapsed=elapsed+time.perf_counter()-start, seed=seed,
                case_json=json.dumps(case, sort_keys=True),
                provenance_json=json.dumps(provenance, sort_keys=True))
            temporary.replace(checkpoint)
    elapsed += time.perf_counter()-start
    blocks = [float(np.mean(block)) for block in np.array_split(values, 16)]
    result = dict(case=case, seed=seed, mean=float(np.mean(values)),
                  block_means=blocks, samples=len(values), seconds=elapsed,
                  native_free=dyn.native, key=key, provenance=provenance)
    write_json(result_path, result)
    write_csv(output/'jobs'/f'{key}.csv',
        (dict(sample_step=case['burn']+(i+1)*case['stride'], observable=x) for i, x in enumerate(values)))
    return result


def summarize(output):
    jobs = [json.loads(path.read_text()) for path in sorted((output/'jobs').glob('*.json'))]
    groups = {}
    for job in jobs:
        key = job['key'].rsplit('_s', 1)[0]
        groups.setdefault(key, []).append(job)
    rows = []
    for key, members in groups.items():
        c = members[0]['case']
        means = np.array([x['mean'] for x in members])
        if len(means) < 2:
            continue
        sem = float(means.std(ddof=1)/np.sqrt(len(means)))
        halfwidth = float(student_t.ppf(.975, len(means)-1)*sem)
        mean = float(means.mean())
        rows.append(dict(case=key, method=c['method'], particles=c['particles'], beta=c['beta'],
            beads=c['beads'], timestep=c['dt'], independent_seeds=len(members),
            steps_per_seed=c['steps'], mean=mean, seed_sem=sem, ci95_halfwidth=halfwidth,
            finite_bead_reference=c['reference'], continuum_reference=c['continuum'],
            relative_error_to_finite_beads=(mean-c['reference'])/c['reference'],
            relative_error_to_continuum=(mean-c['continuum'])/c['continuum'],
            reference_in_ci95=bool(abs(mean-c['reference']) <= halfwidth),
            total_cpu_wall_seconds=sum(x['seconds'] for x in members)))
    if rows:
        write_csv(output/'trajectory_summary.csv', rows)
        write_json(output/'trajectory_summary.json', rows)
    return rows


def timing(output):
    """Same exchange-only scope as paper Appendix B1; machine-specific times."""
    rows = []
    rng = np.random.default_rng(988)
    for n in (16, 32, 64, 128, 256, 512, 1024, 2048):
        q = rng.normal(scale=.4, size=(32, n, 3))
        bosonic_exchange(q, 1., 6.)
        times = []
        for repeat in range(7):
            start = time.perf_counter()
            result = bosonic_exchange(q, 1., 6.)
            times.append(time.perf_counter()-start)
        rows.append(dict(particles=n, beads=32, median_seconds=float(np.median(times)),
                         min_seconds=min(times), max_seconds=max(times),
                         row_sum_error=float(np.max(abs(result.connections.sum(axis=1)-1))),
                         column_sum_error=float(np.max(abs(result.connections.sum(axis=0)-1)))))
    write_csv(output/'boson_exchange_timing.csv', rows)
    fit = rows[-4:]
    slope = np.polyfit(np.log([r['particles'] for r in fit]), np.log([r['median_seconds'] for r in fit]), 1)[0]
    write_json(output/'boson_exchange_scaling.json', dict(fitted_slope=float(slope),
        fit_particles=[r['particles'] for r in fit], paper_ipi_slope=1.9,
        note='Single-process exchange energy and forces only, P=32. Different hardware and Python overhead; no absolute speedup claim.'))


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    with (output/'suzuki_chin_convergence.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for method, label in (('primitive', 'Primitive'), ('suzuki_chin', 'Suzuki–Chin')):
        selected = [r for r in rows if r['method'] == method and float(r['beta']) == 4.]
        axes[0].loglog([int(r['beads']) for r in selected], [float(r['relative_error']) for r in selected], 'o-', label=label)
    axes[0].set(xlabel='Beads P', ylabel='Relative variance error', title='Harmonic action, βω = 4')
    axes[0].legend()
    with (output/'pigs_convergence.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for projection in (2., 4., 8., 12.):
        selected = [r for r in rows if float(r['trial_alpha']) == .25 and float(r['projection_time']) == projection]
        axes[1].loglog([float(r['tau']) for r in selected], [abs(float(r['discretization_error'])) for r in selected], 'o-', label=f'Θ={projection:g}')
    axes[1].set(xlabel='Imaginary-time step τ', ylabel='Absolute midpoint variance error', title='PIGS: separate projection from τ error')
    axes[1].legend()
    summary = output/'trajectory_summary.json'
    if summary.exists():
        selected = [r for r in json.loads(summary.read_text()) if r['method'] == 'bosons']
        labels = [f"N={r['particles']}, β={r['beta']:g}\ndt={r['timestep']:g}" for r in selected]
        error = [100*r['relative_error_to_continuum'] for r in selected]
        ci = [100*r['ci95_halfwidth']/r['continuum_reference'] for r in selected]
        axes[2].errorbar(np.arange(len(selected)), error, yerr=ci, fmt='o', capsize=3, label='MD, seed t 95% CI')
        axes[2].plot(np.arange(len(selected)), [100*(r['finite_bead_reference']/r['continuum_reference']-1) for r in selected], 'x', label='Exact P=36 action')
        axes[2].set_xticks(np.arange(len(selected)), labels, fontsize=7)
        axes[2].axhline(0, color='gray', linewidth=.8)
        axes[2].axhspan(-.5, .5, color='gray', alpha=.12, label='Paper ±0.5% envelope')
        axes[2].set(ylabel='Energy error vs continuum (%)', title='2D bosons: published model')
        axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=.2)
    fig.savefig(output/'path_integral_comparison.png', dpi=180)
    fig.savefig(output/'path_integral_comparison.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/validation/tier2-literature-2026-11-01/path_integral')
    parser.add_argument('--mode', choices=('deterministic', 'sample', 'summarize', 'timing', 'plots', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--case-limit', type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in ('sample', 'all'):
        definitions = cases(args.steps)
        if args.case_limit:
            definitions = definitions[:args.case_limit]
        provenance = source_provenance()
        preflight_sampling(args.output, definitions, provenance)
    if args.mode in ('deterministic', 'all'):
        deterministic(args.output)
    if args.mode in ('sample', 'all'):
        manifest = dict(python=platform.python_version(), numpy=np.__version__,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seeds=SEEDS, cases=definitions,
            provenance=provenance,
            analysis_script_sha256=hashlib.sha256((ROOT/'benchmarks/tier2_literature/path_integral_analysis.py').read_bytes()).hexdigest(),
            boson_paper_doi='10.1063/5.0173749',
            author_input_repository='https://github.com/Hirshberg-Lab/quadratic-bosonic-pimd-jcp23-data',
            author_input_commit='c3939a9d718dcdd8bd2f3ff718c60b22d3fa6609',
            paper_scope='Appendix B2/Fig6 ideal 2D harmonic bosons, P36; reduced m=hbar=omega=1.',
            differences=f'PILE-L/Cayley instead of paper NHC; {args.steps} (refinement {2*args.steps}) steps instead of 1 million; no raw paper curve arrays were available.',
            estimator='Boson E=2<V> by harmonic scaling; SC even-bead x²; PIGS midpoint x².',
            uncertainty='Independent-seed standard error and Student-t 95% CI; 16 time-block means per seed are retained.',
            suzuki_chin_doi='10.1063/1.4971438', pigs_doi='10.1063/1.481926')
        if not (args.output/'manifest.json').exists():
            write_json(args.output/'manifest.json', manifest)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, case, seed, args.output, provenance)
                       for case in definitions for seed in SEEDS]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps({k: result[k] for k in ('key', 'mean', 'seconds')}), flush=True)
                summarize(args.output)
    if args.mode in ('summarize', 'all'):
        summarize(args.output)
    if args.mode in ('timing', 'all'):
        timing(args.output)
    if args.mode in ('plots', 'all'):
        plots(args.output)


if __name__ == '__main__':
    main()
