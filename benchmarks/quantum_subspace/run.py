"""Run from repository root: PYTHONPATH=src python benchmarks/quantum_subspace/run.py --case strong

Requires optional numba. All errors concern a common q response, not a rate.
Independent full quantum data are audit references only, never selector inputs.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import numba
from scipy.integrate import trapezoid
from pyrpmd.analysis.quantum_subspace import select_subspace, memory_embed
from doublewell import Model, sample_hmc, quantum, exact_response, exact_correlation, propagate, reduced_model


def save_json(path, value):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x))
    path.write_text(json.dumps(value, indent=2, default=encode), encoding='utf-8')


def transform(time_grid, correlation, beta, frequencies, damping):
    # Integration by parts: do not differentiate a noisy correlation curve.
    z = 1j*frequencies-damping
    integral = trapezoid(np.exp(np.outer(z, time_grid))*correlation[None], time_grid, axis=1)
    return beta*(correlation[0]-np.exp(z*time_grid[-1])*correlation[-1]+z*integral)


def error_metrics(predicted, reference, frequencies):
    band = (frequencies >= .03) & (frequencies <= .6)
    peak = lambda spectrum: float(frequencies[np.flatnonzero(band)[np.argmax(spectrum[band])]])
    spectrum = predicted.imag
    return {
        'relative_complex_L2_full': float(np.linalg.norm(predicted-reference)/np.linalg.norm(reference)),
        'relative_spectrum_L2_tunneling': float(np.linalg.norm(spectrum[band]-reference.imag[band])/np.linalg.norm(reference.imag[band])),
        'peak_tunneling': peak(spectrum), 'reference_peak': peak(reference.imag),
        'negative_spectral_fraction': float(trapezoid(np.maximum(-spectrum, 0), frequencies)
                                           /max(trapezoid(np.abs(spectrum), frequencies), 1e-30)),
    }


def run_rpmd(model, beta, beads, count, seed, tmax, dt, output, label):
    q, p, sampling = sample_hmc(model, beta, beads, count, seed)
    stride, steps = round(.04/dt), round(tmax/dt)
    blocks = propagate(q, p, model.stiffness(), model.quartic, beta, dt, steps, stride)
    times = np.arange(blocks.shape[1])*stride*dt
    np.savez_compressed(output/(label+'_rpmd.npz'), t=times, blocks=blocks)
    save_json(output/(label+'_sampling.json'), sampling)
    return times, blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['weak', 'strong', 'residual', 'collective', 'warm'], default='strong')
    parser.add_argument('--beads', type=int, default=64)
    parser.add_argument('--n', type=int, default=2048)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--max-dim', type=int, default=3)
    parser.add_argument('--output-dir', default='output/quantum-subspace')
    args = parser.parse_args()
    if args.n < 64 or args.n % 64 or args.beads < 1:
        parser.error('n must be a positive multiple of 64 and beads must be positive')
    settings = {'weak': ((.03, .02), 8.), 'strong': ((.8, .02), 8.),
                'residual': ((.8, .05), 8.), 'collective': ((.8, .5), 8.),
                'warm': ((.8, .02), 2.)}
    couplings, beta = settings[args.case]
    dt = args.dt if args.dt is not None else (.005 if args.case == 'warm' else .02)
    if dt <= 0 or not np.isclose(round(.04/dt)*dt, .04):
        parser.error('dt must be positive and divide the .04 output interval')
    output = Path(args.output_dir)/args.case
    output.mkdir(parents=True, exist_ok=True)
    numba.set_num_threads(min(4, numba.get_num_threads()))
    started = time.perf_counter()
    model = Model(displacements=couplings)
    frequencies = np.linspace(.03, 3.5, 700)
    damping, tmax = .04, 200.
    pilot, _, pilot_record = sample_hmc(model, beta, args.beads, 2048, 10103)
    positions = pilot[:, 0, :]
    selection = select_subspace(positions, model.force(positions),
        max_dimension=args.max_dim, target=np.eye(3)[0])
    full_quantum = quantum(model, beta, [36, 12, 10], 80)
    Q_full = exact_response(full_quantum, frequencies, damping)
    static, _, static_record = sample_hmc(model, beta, args.beads, 32768, 20103, chains=128)
    centroids = static.mean(1)
    covariance = centroids.T@centroids/len(centroids)
    times, full_blocks = run_rpmd(model, beta, args.beads, args.n, 30103, tmax, dt, output, 'full')
    R_full = transform(times, full_blocks.mean(0), beta, frequencies, damping)
    report = {'model': asdict(model), 'beta': beta, 'beads': args.beads, 'dt': dt,
        'n_initial_paths': args.n, 'tmax': tmax, 'damping': damping,
        'selection': selection, 'quantum_reference': full_quantum['metadata'],
        'pilot': pilot_record, 'static_sampling': static_record,
        'rpmd': error_metrics(R_full, Q_full, frequencies), 'methods': {},
        'physical_converged': False,
        'scope': 'Closed 3D double-well q response and tunneling doublet, NOT a thermal rate',
        'uncertainty_unit': 'Independent HMC chains, not snapshots within a chain'}
    curves = {'omega': frequencies, 'quantum': Q_full, 'rpmd': R_full,
        'time': times, 'rpmd_C': full_blocks.mean(0),
        'quantum_C': exact_correlation(full_quantum, beta, times)}
    for dimension in sorted(selection['nested_bases']):
        U = selection['nested_bases'][dimension]
        reduced, rotation, complement, cross, bath = reduced_model(model, U, selection['stiffness'])
        spec = quantum(reduced, beta, [36, 12, 10][:dimension], 80)
        Q_active = exact_response(spec, frequencies, damping)
        embedded, diagnostics = memory_embed(spec, frequencies, damping, cross, bath)
        matched, static_diagnostics = memory_embed(spec, frequencies, damping, cross, bath,
            rotation.T@covariance@rotation, beta)
        if dimension == model.dimension:
            # Same Hamiltonian and same q observable: reuse data for exact cancellation.
            R_active = R_full
        else:
            sub_times, sub_blocks = run_rpmd(reduced, beta, args.beads, args.n,
                40103+dimension, tmax, dt, output, 'sub'+str(dimension))
            R_active = transform(sub_times, sub_blocks.mean(0), beta, frequencies, damping)
        defect = 1/(1/R_full+1/Q_active-1/R_active)
        report['methods'][str(dimension)] = {
            'reference_quantum': spec['metadata'], 'submodel': asdict(reduced),
            'isolated_active_quantum': error_metrics(Q_active, Q_full, frequencies),
            'matched_inverse_defect': error_metrics(defect, Q_full, frequencies),
            'memory_only': error_metrics(embedded[:, 0, 0], Q_full, frequencies),
            'optional_static_matched_memory': error_metrics(matched[:, 0, 0], Q_full, frequencies),
            'memory_diagnostics': diagnostics, 'static_diagnostics': static_diagnostics,
            'is_full_quantum_limit': dimension == model.dimension}
        curves['memory_d'+str(dimension)] = embedded[:, 0, 0]
        curves['isolated_d'+str(dimension)] = Q_active
        curves['defect_d'+str(dimension)] = defect
        curves['static_memory_d'+str(dimension)] = matched[:, 0, 0]
    report['elapsed_seconds'] = time.perf_counter()-started
    save_json(output/'summary.json', report)
    np.savez_compressed(output/'curves.npz', **curves)
    print(json.dumps({'case': args.case, 'selected_dimension': selection['selected_dimension'],
                     'summary': str(output/'summary.json'), 'physical_converged': False}, indent=2))


if __name__ == '__main__':
    main()
