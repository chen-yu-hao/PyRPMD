"""Changing the projected Hamiltonian must not reuse a same-temperature factor."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_correction import SubspaceCorrection
from pyrpmd.system import System


@pytest.fixture
def setup_pipeline(tmp_path, monkeypatch):
    run = SimpleNamespace(root=tmp_path, beta=100., nbeads=8, xi_star=1., reaction={},
                          config={'task': 'rate', 'beta': 100., 'beads': 8,
                                  'potential': {'provider': 'free'}},
                          system=System(('H',), np.array([1837.]), np.zeros((1, 3))))
    path = np.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    mean = SimpleNamespace(curvature_width=lambda *a, **k: {'maximum': 0.})
    monkeypatch.setattr(pipeline, 'load_rate_run', lambda _: run)
    monkeypatch.setattr(pipeline, 'build_path', lambda *args: (mean, None, path, {}))
    monkeypatch.setattr(pipeline, '_static_curvature_width', lambda *args: {'maximum': 0.})
    monkeypatch.setattr(pipeline, 'path_model', lambda *args: (None, {}, {}))
    computed = []
    def correction(model, beta, nbeads, **kwargs):
        computed.append((beta, nbeads))
        return SubspaceCorrection(beta, 1, 2., .1, 2., 1., nbeads, {})
    monkeypatch.setattr(pipeline, 'subspace_correction', correction)
    monkeypatch.setattr(pipeline, 'correct_rate_run', lambda *a, **k: {})
    spec = {'source': {'rate_run': str(tmp_path)}, 'subspace': {'dimension': 1},
            'output': {'directory': str(tmp_path / 'correction'), 'reuse': True}}
    return spec, run, path, computed


def test_identical_run_reuses_factor_and_keeps_its_identity(setup_pipeline):
    spec, run, path, computed = setup_pipeline
    first = pipeline.run_subspace_pipeline(spec)
    second = pipeline.run_subspace_pipeline(spec)
    assert len(computed) == 1
    assert first['pipeline_identity'] == second['pipeline_identity']
    assert first['corrections']['1'] == second['corrections']['1']


@pytest.mark.parametrize('change', ['curvature', 'path', 'path_source', 'beads', 'potential', 'quantum'])
def test_changed_same_temperature_calculation_recomputes(setup_pipeline, change):
    spec, run, path, computed = setup_pipeline
    first = pipeline.run_subspace_pipeline(spec)
    if change == 'curvature':
        spec['subspace']['curvature'] = 'small'
    elif change == 'path':
        path[1, 1] = .2
    elif change == 'path_source':
        spec['path'] = {'source': 'least_action'}
    elif change == 'beads':
        spec['rpmd'] = {'nbeads': 16}
    elif change == 'potential':
        run.config['potential'] = {'provider': 'harmonic', 'k': .1}
    else:
        spec['quantum'] = {'spacing_1d': .05}
    second = pipeline.run_subspace_pipeline(spec)
    assert len(computed) == 2
    assert first['pipeline_identity'] != second['pipeline_identity']


def test_legacy_temperature_only_cache_is_not_reused(setup_pipeline):
    spec, run, path, computed = setup_pipeline
    directory = run.root / 'correction'
    directory.mkdir()
    SubspaceCorrection(run.beta, 1, 999., .1, 999., 1., run.nbeads, {}).write(directory / 'C_S_1d.json')
    report = pipeline.run_subspace_pipeline(spec)
    assert len(computed) == 1
    assert report['corrections']['1']['factor'] == 2.
