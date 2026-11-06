"""Sampling dispatch and checkpoint state do not depend on module import order."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pyrpmd.checkpoint import load_checkpoint


MODULES = [
    'pyrpmd.workflows.metadynamics', 'pyrpmd.workflows.shared_metadynamics',
    'pyrpmd.workflows.opes', 'pyrpmd.workflows._distributed_meta_async',
    'pyrpmd.workflows._distributed_meta_async_io', 'pyrpmd.workflows._distributed_meta_mpi',
    'pyrpmd.workflows._distributed_opes_common', 'pyrpmd.workflows._distributed_opes_async',
    'pyrpmd.workflows._distributed_opes_async_io', 'pyrpmd.workflows._distributed_opes_mpi',
]
SCRIPT = '''
import importlib, json, sys, warnings
from pathlib import Path
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
from pyrpmd import run, resume_run
from pyrpmd.workflows import metadynamics, opes
assert metadynamics.run_metadynamics.__module__ == metadynamics.__name__
assert opes.run_opes.__module__ == opes.__name__
assert opes._sample_replica.__module__ == opes.__name__
from pyrpmd.opes import OPESBias
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', DeprecationWarning)
    from pyrpmd._opes_serial import OPESBias as OldOPESBias
assert OldOPESBias is OPESBias
assert len(caught) == 1 and caught[0].category is DeprecationWarning
for method in ('metadynamics', 'opes'):
    job = {
        'task': 'fes', 'fes': {'method': method}, 'temperature': '300 K',
        'beads': 2, 'seed': 431,
        'system': {'symbols': ['H'], 'geometry': [[0.1, 0., 0.]]},
        'potential': {'provider': 'builtin', 'name': 'harmonic', 'params': {'k': .002}},
        'dynamics': {'timestep': '0.001 fs', 'native': False},
        'sampling': {'equilibration_steps': 0, 'production_steps': 8,
                     'sample_interval': 2, 'andersen_interval': 3},
        'cvs': [{'name': 'x', 'kind': 'cartesian', 'atom': 0, 'axis': 'x'}],
        method: {'sigma': [.3], 'pace': 2, 'bias_factor': 5.},
        'analysis': {'bins': [4], 'bootstrap': 0},
        'output': {'checkpoint_steps': 4},
    }
    job[method].update({'height': .0001} if method == 'metadynamics' else {'barrier': .001})
    output = Path(sys.argv[2]) / method
    result = run(job, output=output)
    resumed = resume_run(output)
    assert result['samples'] == resumed['samples'] == 4
'''


def _run(output, order):
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-c', SCRIPT, json.dumps(order), str(output)],
                   env=env, check=False, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('order', [MODULES[::-1], MODULES[3:] + MODULES[:3]])
def test_public_fes_input_and_resume_are_independent_of_import_order(tmp_path, order):
    _run(tmp_path / 'reference', MODULES)
    _run(tmp_path / 'reordered', order)
    for method in ('metadynamics', 'opes'):
        relative = Path(method) / 'checkpoints' / f'{method}_0000.npz'
        with load_checkpoint(tmp_path / 'reference' / relative) as left, \
                load_checkpoint(tmp_path / 'reordered' / relative) as right:
            assert left.files == right.files
            for key in left.files:
                a, b = left[key], right[key]
                assert a.dtype == b.dtype and a.shape == b.shape, key
                assert a.tobytes() == b.tobytes(), key
        left = json.loads((tmp_path / 'reference' / method / 'manifest.json').read_text())
        right = json.loads((tmp_path / 'reordered' / method / 'manifest.json').read_text())
        assert left['physics_fingerprint'] == right['physics_fingerprint']
