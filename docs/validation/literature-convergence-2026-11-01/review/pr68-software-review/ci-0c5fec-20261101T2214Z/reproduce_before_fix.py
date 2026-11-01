"""Reproduce the two CI fixture defects without loading a scientific PES."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

BASE = Path('/workspace/scratch/a5e4820daedd')
ROOT = BASE / 'pyrpmd-convergence-integration'
OUT = ROOT / 'docs/validation/literature-convergence-2026-11-01/review/pr68-software-review/ci-0c5fec-20261101T2214Z'
WORK = BASE / 'ci-fixture-reproduction-0c5fec'
os.sched_setaffinity(0, {8})
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import pytest
from benchmarks.hch4 import efficiency_pilot as pilot
from benchmarks.h3 import p256_native_calibration as calibration

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def module(path):
    spec = importlib.util.spec_from_file_location('old_hch4_fixture', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

WORK.mkdir(exist_ok=False)
fake_root = WORK / 'projects/project'
fake_tmp = WORK / 'tmp/pytest-of-runner/case'
fake_tmp.mkdir(parents=True)
test_source = ROOT / 'tests/test_hch4_efficiency_pilot.py'
old_test = module(test_source)
results = {'schema': 'pyrpmd.ci-fixture-defect-reproduction.v1',
           'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
               (test_source, ROOT / 'tests/test_h3_p256_native_calibration.py', pilot.SELF, Path(calibration.__file__))},
           'real_PES_evaluations': 0, 'real_dynamics_steps': 0}
with pytest.MonkeyPatch.context() as mp:
    for name in ('SELF', 'TESTS'):
        source = getattr(pilot, name)
        target = fake_root / source.relative_to(pilot.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        mp.setattr(pilot, name, target)
    mp.setattr(pilot, 'ROOT', fake_root)
    plan, _ = old_test.plan_fixture(fake_tmp, mp)
    plan_path = fake_tmp / 'plan.json'
    pilot.write_new_json(plan_path, plan)
    try:
        pilot.load_plan(plan_path)
    except ValueError as exc:
        assert str(exc) == 'live pilot output must be a separate adopted workspace directory'
        results['path_fixture'] = {'reproduced': True, 'project': str(fake_root),
            'temporary_directory': str(fake_tmp), 'error': str(exc)}
    else:
        raise AssertionError('old external-temporary-directory fixture unexpectedly passed')

origin = 13.000000000000014
wall = 115.0
with pytest.MonkeyPatch.context() as mp:
    mp.setattr(calibration, 'ENTRY_WALL', origin)
    mp.setattr(calibration, 'cpu_seconds', lambda: 1.0)
    mp.setattr(calibration, 'time', SimpleNamespace(monotonic=lambda: origin + wall))
    calibration.admission()
    results['clock_fixture'] = {'reproduced': True, 'clock_origin': origin,
        'requested_elapsed': wall, 'computed_clock': origin + wall,
        'subtracted_elapsed': (origin + wall) - origin,
        'old_fixture_failed_to_raise': True}
path = OUT / 'before-fix-reproduction.json'
with path.open('x') as stream:
    json.dump(results, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({'output': str(path), 'sha256': sha(path), 'both_reproduced': True,
                  'real_PES_evaluations': 0, 'real_dynamics_steps': 0}))
