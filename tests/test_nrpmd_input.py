"""NRPMD public input and matrix-provider contracts."""
import numpy as np
import pytest

from pyrpmd.input import JobSpec, InputError, validate_nonadiabatic
from pyrpmd.potentials.diabatic import DiabaticPythonPotential, check_diabatic_potential
from pyrpmd.runner import prepare, run_resolved
from test_input_cli import small_collision


def job():
    d = small_collision()
    d['potential'] = {'provider': 'diabatic_python', 'factory': 'matrix_pes.py:make'}
    d['nonadiabatic'] = {'nstates': 2}
    return d


def test_normalized_input_and_identity():
    c, _ = JobSpec.from_mapping(job()).resolve()
    assert c['nonadiabatic'] == {
        'method': 'nrpmd', 'nstates': 2, 'initial_state': 0,
        'state_labels': ['0', '1'], 'initialization': 'focused',
        'asymptotic_coupling_tolerance': 1e-8,
    }
    assert any('Experimental NRPMD' in x for x in c['_warnings'])
    assert validate_nonadiabatic(c) == c['nonadiabatic']


@pytest.mark.parametrize('field,value', [
    ('nstates', 0), ('nstates', True), ('nstates', 2.5),
    ('initial_state', -1), ('initial_state', 2), ('initial_state', True),
    ('method', 'surface-hopping'), ('initialization', 'thermal'),
    ('state_labels', ['a', 'a']), ('state_labels', ['a']), ('state_labels', ['', 'b']),
    ('asymptotic_coupling_tolerance', -1), ('asymptotic_coupling_tolerance', float('nan')),
])
def test_bad_electronic_input(field, value):
    d = job(); d['nonadiabatic'][field] = value
    with pytest.raises((ValueError, TypeError)):
        JobSpec.from_mapping(d).resolve()


@pytest.mark.parametrize('update', [
    {'task': 'rate'}, {'task': 'fes'},
    {'potential': {'provider': 'free'}},
    {'potential': {'provider': 'diabatic_python', 'factory': 'm:f', 'derivative': 'force'}},
    {'parallel': {'backend': 'mpi'}},
    {'parallel': {'bead_workers': 2}},
    {'parallel': {'mode': 'gpu', 'devices': [0]}},
    {'dynamics': {'loop_backend': 'native'}},
])
def test_unsupported_method_or_backend(update):
    d = job(); d.update(update)
    with pytest.raises(ValueError):
        JobSpec.from_mapping(d).resolve()


def test_matrix_requires_electronic_section():
    d = job(); del d['nonadiabatic']
    with pytest.raises(InputError, match='requires'):
        JobSpec.from_mapping(d).resolve()


def test_resumed_backend_revalidated_before_output(tmp_path):
    c, s = JobSpec.from_mapping(job()).resolve()
    c['parallel']['backend'] = 'mpi'
    with pytest.raises(InputError, match='trajectory-process'):
        run_resolved(c, s, output=tmp_path/'not-created', resume=True)
    assert not (tmp_path/'not-created').exists()


def test_provider_units_and_all_matrix_gradient_check():
    from pyrpmd.nonadiabatic import DiabaticResult
    def harmonic(q):
        v = np.einsum('bai,bai->b', q, q)
        w = np.array([[1., .2], [.2, 2.]])
        return DiabaticResult(v[:, None, None]*w, 2*q[:, None, None, :, :]*w[None,:,:,None,None])
    p = DiabaticPythonPotential(harmonic, units={'length': 'angstrom', 'energy': 'eV'})
    q = np.arange(6.).reshape(1,2,3)/10
    result = p.evaluate(q)
    lf, ef = p.length_factor, p.energy_factor
    expected = harmonic(q/lf)
    np.testing.assert_allclose(result.matrix, expected.matrix*ef)
    np.testing.assert_allclose(result.gradients, expected.gradients*ef/lf)
    assert check_diabatic_potential(p, q, 2)['max_absolute_gradient_error'] < 1e-10


def test_public_prepare_importable_matrix(tmp_path):
    source = tmp_path/'matrix_pes.py'
    source.write_text('''import numpy as np
from pyrpmd.nonadiabatic import DiabaticResult
class PES:
 def evaluate(self,q):
  b,n,_=q.shape
  return DiabaticResult(np.zeros((b,2,2)),np.zeros((b,2,2,n,3)))
def make(): return PES()
''')
    d = job()
    d['potential']['factory'] = str(source)+':make'
    result = prepare(d, numerical=True)
    assert result['pes_check']['matrix_hartree'] == [[[0.,0.],[0.,0.]]]
    assert result['pes_check']['max_absolute_gradient_error'] == 0
