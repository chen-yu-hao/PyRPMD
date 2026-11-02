"""Run the input-file/CLI path with the real optional PLUMED shared library.

PYRPMD_REQUIRE_PLUMED=1 turns absence into a failure in the pinned CI job.
"""
import importlib
import os

import numpy as np
import pytest

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.cli import main
from pyrpmd.constants import ATOMIC_TIME_S, KB_AU


@pytest.fixture
def live_plumed():
    required = os.environ.get('PYRPMD_REQUIRE_PLUMED') == '1'
    try:
        module = importlib.import_module('plumed')
        with module.Plumed() as kernel:
            api = np.zeros(1, dtype=np.intc)
            kernel.cmd('getApiVersion', api)
            assert api[0] >= 8
    except (ImportError, RuntimeError) as error:
        if required:
            pytest.fail(f'Required real PLUMED runtime unavailable: {error}', pytrace=False)
        pytest.skip(f'Optional real PLUMED runtime unavailable: {error}')
    return module


def rows(path):
    return np.loadtxt(path, comments='#', ndmin=2)


@pytest.mark.parametrize('beads', [1, 4])
def test_real_kernel_input_cli_metad_clock_and_observation_invariance(tmp_path, live_plumed, beads, capsys):
    script = ('UNITS TIME=fs\n'
              'd: DISTANCE ATOMS=1,2 NOPBC\n'
              'm: METAD ARG=d SIGMA=0.02 HEIGHT=0.1 PACE=2 FILE=HILLS\n'
              'PRINT ARG=d,m.bias STRIDE=1 FILE=COLVAR\n')
    (tmp_path/'bias.dat').write_text(script)
    states, histories = [], []
    for interval in (1, 4):
        data = {
            'task': 'plumed', 'temperature': 1/(4*KB_AU), 'beads': beads,
            'seed': 51, 'native': False,
            'system': {'symbols': ['H', 'H'], 'masses': [2., 3.],
                       'geometry': [[0., 0., 0.], [3., 0., 0.]]},
            'potential': {'provider': 'harmonic', 'params': {'k': .2}},
            'dynamics': {'timestep': .02},
            'sampling': {'equilibration_steps': 2, 'production_steps': 12,
                         'sample_interval': interval},
            'plumed': {'file': 'bias.dat'}, 'output': {'checkpoint_steps': 3},
        }
        job = tmp_path/f'job-{interval}.py'
        job.write_text('def build():\n    return ' + repr(data) + '\n')
        output = tmp_path/f'run-{interval}'
        assert main(['validate', str(job), '--check-pes', '--output', str(output)]) == 0
        assert not (output/'plumed/HILLS').exists()
        assert main(['run', str(job), '--output', str(output)]) == 0
        capsys.readouterr()
        hills, colvar = rows(output/'plumed/HILLS'), rows(output/'plumed/COLVAR')
        assert hills.shape[0] == 7
        assert colvar.shape[0] == 14
        assert np.isfinite(hills).all() and np.isfinite(colvar).all()
        # fs resolves each atomic-time step at the HILLS default print precision.
        np.testing.assert_allclose(hills[:, 0], np.arange(2, 15, 2)*.02*ATOMIC_TIME_S/1e-15,
                                   atol=5.1e-7, rtol=0)
        with load_checkpoint(output/'checkpoints/plumed-md-state.npz', require_integrity=True) as saved:
            assert int(saved['step']) == 14
            assert len(saved['sample_steps']) == 12//interval
            assert np.isfinite(saved['physical_bias']).all()
            states.append((saved['q'].copy(), saved['p'].copy()))
        histories.append((hills, colvar))
        before = (output/'plumed/HILLS').read_bytes()
        assert main(['resume', str(output)]) != 0
        assert 'resume is unsupported' in capsys.readouterr().err
        assert (output/'plumed/HILLS').read_bytes() == before
    # Diagnostic sample frequency changes neither the trajectory nor history.
    for first, second in zip(states[0], states[1]):
        np.testing.assert_array_equal(first, second)
    for first, second in zip(histories[0], histories[1]):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize('well_tempered', [False, True])
@pytest.mark.parametrize('checkpoint_steps', [3, 7])
def test_real_kernel_file_restart_matches_uninterrupted_history(tmp_path, live_plumed, monkeypatch, well_tempered,
                                                               checkpoint_steps):
    from pyrpmd.plumed import PlumedBias
    from pyrpmd.workflows.plumed import resolve_plumed_job, run_plumed

    tempering = ' BIASFACTOR=4 TEMP=300' if well_tempered else ''
    script = ('UNITS LENGTH=Bohr ENERGY=Ha TIME=atomic\n'
              'd: DISTANCE ATOMS=1,2 NOPBC\n'
              'm: METAD ARG=d SIGMA=0.2 HEIGHT=0.0001 PACE=2 FILE=HILLS' + tempering + '\n'
              'PRINT ARG=d,m.bias STRIDE=1 FILE=COLVAR\n')
    data = {
        'task': 'plumed', 'temperature': 1/(4*KB_AU), 'beads': 4,
        'seed': 51, 'native': False,
        'system': {'symbols': ['H', 'H'], 'masses': [2., 3.],
                   'geometry': [[0., 0., 0.], [3., 0., 0.]]},
        'potential': {'provider': 'harmonic', 'params': {'k': .2}},
        'dynamics': {'timestep': .02},
        'sampling': {'equilibration_steps': 2, 'production_steps': 12, 'sample_interval': 2},
        'plumed': {'input': script, 'restart': 'files'}, 'output': {'checkpoint_steps': checkpoint_steps},
    }
    config, system = resolve_plumed_job(data, base_directory=tmp_path)
    complete, split = tmp_path/'complete', tmp_path/'split'
    run_plumed(config, system, complete)
    original = PlumedBias.step

    def interrupt(self, dynamics, state, **kwargs):
        result = original(self, dynamics, state, **kwargs)
        if state.step == 8:
            raise RuntimeError('interrupt after real PLUMED writes a hill')
        return result

    with monkeypatch.context() as patch:
        patch.setattr(PlumedBias, 'step', interrupt)
        with pytest.raises(RuntimeError, match='interrupt after real'):
            run_plumed(config, system, split)
    result = run_plumed({**config, 'resume': True}, system, split)
    assert result['restart_supported'] and result['steps'] == 14
    for name, count in [('HILLS', 7), ('COLVAR', 14)]:
        expected, actual = rows(complete/'plumed'/name), rows(split/'plumed'/name)
        assert len(actual) == count
        # HILLS decimal round trips and WT rescaling can differ at the last
        # binary bit; this is a floating-point consistency, not bitwise claim.
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)
    with load_checkpoint(complete/'checkpoints/plumed-md-state.npz', require_integrity=True) as full:
        expected = {name: full[name] for name in ['q', 'p', 'positions', 'physical_bias', 'rng']}
    with load_checkpoint(split/'checkpoints/plumed-md-state.npz', require_integrity=True) as resumed:
        for name in ['q', 'p', 'positions', 'physical_bias']:
            np.testing.assert_allclose(resumed[name], expected[name], rtol=2e-12, atol=2e-14)
        np.testing.assert_array_equal(resumed['rng'], expected['rng'])
