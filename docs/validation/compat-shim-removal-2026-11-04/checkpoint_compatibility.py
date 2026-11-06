"""Observe main@801953d checkpoints; resume them with a second checkout.

Run with PYTHONPATH pointing to the desired checkout. No implementation is
mocked: a profile callback copies already committed checkpoints after the real
atomic writer returns, and the reference trajectory continues uninterrupted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib

import numpy as np

from pyrpmd.checkpoint import atomic_savez, load_checkpoint, ENVELOPE
from pyrpmd.input import JobSpec
from pyrpmd.workflows.fes_dispatch import run_fes
from pyrpmd.workflows.common import _fingerprint


def cases():
    result = {}
    for method in ('metadynamics', 'opes'):
        modes = ('standard', 'well_tempered') if method == 'metadynamics' else ('metad', 'explore')
        for mode in modes:
            for bath in ('andersen', 'pile-l'):
                name = f'{method}-{mode}-{bath}'
                result[name] = (method, mode, bath, 'serial', False, 2)
        if method == 'opes':
            for schema in (1, 2):
                for mode in modes:
                    result[f'opes-{mode}-adaptive-schema{schema}'] = (method, mode, 'andersen', 'serial', True, schema)
                result[f'opes-metad-fixed-schema{schema}'] = (method, 'metad', 'andersen', 'serial', False, schema)
        else:
            for mode in modes:
                for bath in ('andersen', 'pile-l'):
                    result[f'shared-{mode}-{bath}'] = (method, mode, bath, 'shared', False, 2)
        for transport in ('async-files', 'mpi-sync'):
            for bath in ('andersen', 'pile-l'):
                result[f'{method}-{transport}-{bath}'] = (method, modes[-1], bath, transport, False, 2)
    return result


def job_for(case, examples):
    if case.startswith('example-'):
        relative = case.removeprefix('example-').replace('__', '/') + '.inp'
        return tomllib.loads((examples / relative).read_text())
    method, mode, bath, transport, adaptive, schema = cases()[case]
    mapping = {
        'task': 'fes', 'fes': {'method': method}, 'temperature': '300 K',
        'beads': 3, 'seed': 321,
        'system': {'symbols': ['H'], 'geometry': [[0.1, -0.05, 0.0]]},
        'potential': {'provider': 'builtin', 'name': 'harmonic', 'params': {'k': 0.002}},
        'dynamics': {'timestep': '0.001 fs', 'native': False},
        'sampling': {'equilibration_steps': 2, 'production_steps': 24, 'sample_interval': 2,
                     'andersen_interval': 4, 'replicas': 2, 'thermostat': bath},
        'cvs': [{'name': x, 'kind': 'cartesian', 'atom': 0, 'axis': x} for x in ('x', 'y')],
        method: {'mode': mode, 'sigma': [0.3, 0.4], 'pace': 2},
        'analysis': {'bins': [6, 5], 'bootstrap': 0},
        'output': {'checkpoint_steps': 10},
    }
    if bath == 'pile-l':
        mapping['sampling']['gamma_centroid'] = '0.01 fs^-1'
    if method == 'metadynamics':
        mapping[method]['height'] = '0.02 kcal/mol'
        mapping['output']['store_configurations'] = True
        if mode == 'well_tempered':
            mapping[method]['bias_factor'] = 5.0
        if transport != 'serial':
            mapping[method]['shared_walkers'] = True
    else:
        mapping[method].update(barrier='2 kcal/mol', bias_factor=5.0)
        if adaptive:
            mapping[method].update(adaptive_sigma=True, sigma_min=[0.03, 0.04])
    if transport in ('mpi-sync', 'async-files'):
        mapping[method]['walkers_mode'] = transport
        # At step 10 the baseline already loses clock bits on distributed
        # restart (elapsed * dt vs accumulated time). Step 8 has nonempty
        # history and an exactly reconstructible clock; keep the defect in
        # the separate audit, rather than changing the numerical contract.
        mapping['output']['checkpoint_steps'] = 8
    return mapping


def payload(path):
    with load_checkpoint(path, require_integrity=True) as data:
        return {key: data[key].copy() for key in data.files if key != ENVELOPE}


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and not p.name.endswith('.lock')}


def run_case(case, stage, root, examples, checkpoint_steps=None):
    mapping = job_for(case, examples)
    if checkpoint_steps is not None:
        mapping.setdefault('output', {})['checkpoint_steps'] = checkpoint_steps
    config, system = JobSpec.from_mapping(mapping).resolve()
    schema = cases()[case][-1] if not case.startswith('example-') else 2
    mode = config.get('metadynamics', config.get('opes', {})).get('walkers_mode', 'serial')
    rank = 0
    if mode == 'mpi-sync':
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    directory = root / case
    reference, prefix, resumed = (directory / name for name in ('reference', 'prefix', 'resumed'))
    if rank == 0 and stage == 'generate':
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'input.json').write_text(json.dumps(mapping, indent=2) + '\n')
        (directory / 'fingerprint.txt').write_text(_fingerprint(config, system, {}))
    if rank == 0 and stage == 'resume':
        assert _fingerprint(config, system, {}) == (directory / 'fingerprint.txt').read_text()
        shutil.copytree(prefix, resumed)
    if mode == 'mpi-sync':
        MPI.COMM_WORLD.Barrier()
    target = reference if stage == 'generate' else resumed
    stop = config['output'].get('checkpoint_steps', 1000)
    captured = set()

    def observe(frame, event, arg):
        if event != 'return' or frame.f_code is not atomic_savez.__code__:
            return
        path = Path(frame.f_locals['path'])
        arrays = frame.f_locals['arrays']
        if path.parent.name != 'checkpoints' or int(arrays.get('elapsed', -1)) != stop:
            return
        if path.name in captured:
            return
        if mode == 'async-files':
            # A fixed, repeatable publication schedule: walker 0 finishes, then
            # walker 1 runs. Save after publication, never reorder events.
            if not path.name.endswith('0001.npz') or json.loads(str(arrays['pending_record'])) is not None:
                return
            shutil.copytree(reference, prefix)
        else:
            (prefix / 'checkpoints').mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, prefix / 'checkpoints' / path.name)
        captured.add(path.name)

    if stage == 'generate' and rank == 0:
        sys.setprofile(observe)
    try:
        if mode == 'async-files':
            for walker in ((0, 1) if stage == 'generate' else (1,)):
                os.environ['PYRPMD_WALKER_ID'] = str(walker)
                os.environ['PYRPMD_WALKERS'] = '2'
                run_fes(config, system, target)
        elif schema == 1:
            # The baseline worker's existing legacy task flag constructs the
            # historical empty bias before the first MD step; no patching.
            from pyrpmd.workflows.opes import _sample_replica, _analyze
            streams = [_sample_replica({'config': config, 'system': system, 'output': str(target),
                                       'replica': i, 'initialization_schema': 1}) for i in range(2)]
            _analyze(streams, config, system, target, False)
        else:
            run_fes(config, system, target)
    finally:
        sys.setprofile(None)
        os.environ.pop('PYRPMD_WALKER_ID', None)
        os.environ.pop('PYRPMD_WALKERS', None)
    if rank != 0:
        return
    if stage == 'generate':
        assert captured, f'no checkpoint captured: {case}'
        result = {'case': case, 'prefix_sha256': inventory(prefix), 'reference_sha256': inventory(reference)}
        (directory / 'generated.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        expected = inventory(reference)
        actual = inventory(resumed)
        assert expected.keys() == actual.keys(), (case, expected.keys() ^ actual.keys())
        checked_arrays = 0
        for relative in expected:
            first, second = reference / relative, resumed / relative
            if first.suffix == '.npz':
                a, b = payload(first), payload(second)
                assert a.keys() == b.keys(), (case, relative)
                for key in a:
                    left, right = a[key], b[key]
                    if left.dtype.kind in 'US':
                        left = np.asarray([str(x).replace(str(reference), '<OUTPUT>') for x in left.flat]).reshape(left.shape)
                        right = np.asarray([str(x).replace(str(resumed), '<OUTPUT>') for x in right.flat]).reshape(right.shape)
                    assert left.shape == right.shape and left.dtype == right.dtype, (case, relative, key)
                    assert left.tobytes() == right.tobytes(), (case, relative, key)
                    checked_arrays += 1
            else:
                a = first.read_text().replace(str(reference), '<OUTPUT>')
                b = second.read_text().replace(str(resumed), '<OUTPUT>')
                assert a == b, (case, relative)
        result = {'case': case, 'bitwise_equal': True, 'arrays': checked_arrays,
                  'files': len(expected), 'resumed_sha256': actual}
        (directory / 'compared.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'case': case, 'stage': stage, 'ok': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['generate', 'resume', 'list'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--checkpoint-steps', type=int, help='reproduce the pre-existing clock boundary')
    args = parser.parse_args()
    selected = [name for name in cases() if ('mpi-sync' in name) == args.mpi]
    if not args.mpi:
        selected += ['example-' + str(p.relative_to(args.examples)).removesuffix('.inp').replace('/', '__')
                     for subdir in ('metadynamics', 'opes') for p in sorted((args.examples / subdir).glob('*.inp'))]
    if args.case:
        selected = [args.case]
    if args.stage == 'list':
        print('\n'.join(selected))
        return
    for name in selected:
        run_case(name, args.stage, args.root.resolve(), args.examples.resolve(), args.checkpoint_steps)


if __name__ == '__main__':
    main()
