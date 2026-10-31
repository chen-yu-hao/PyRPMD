"""Offline dataset, real JaxPIP fit, and optional real PySCF regression tests."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pyrpmd.pesgen.reference import ReferencePoint, PySCFSettings
from pyrpmd.pesgen.formats import write_xyz
from pyrpmd.pesgen.jaxpip_data import (check_points, collect_dataset, geometry_key,
                                     load_dataset, member, write_checked)
from pyrpmd.pesgen.jaxpip_fit import FitSettings, fit_jaxpip, load_fit

HAVE_JAXPIP = importlib.util.find_spec("jaxpip") is not None
jaxpip_test = pytest.mark.skipif(not HAVE_JAXPIP, reason="optional real JaxPIP environment")


def point(r, offset=0.):
    q = np.array([[0., 0., 0.], [r, 0., 0.]])
    energy = -1.2 + .3*np.exp(-r) + .05*np.exp(-2*r) + offset
    derivative = -.3*np.exp(-r) - .1*np.exp(-2*r)
    gradient = np.array([[-derivative, 0., 0.], [derivative, 0., 0.]])
    # Synthetic numerical fixture, never claimed to be an electronic calculation.
    return ReferencePoint(("H", "H"), q, energy, gradient, provenance={
        "engine": "PySCF", "version": "synthetic-test-fixture", "scf_converged": True,
        "settings": asdict(PySCFSettings(basis="sto-3g")), "test_fixture": True})


def dataset(tmp_path, *, heldout=True, offset=0.):
    tmp_path.mkdir(parents=True, exist_ok=True)
    splits = {"train": np.linspace(1., 2., 7)}
    if heldout:
        splits.update(validation=[1.075, 1.575], test=[1.125, 1.875])
    text = 'schema="pyrpmd.jaxpip-data/1"\n[references]\n'
    for split, radii in splits.items():
        names = []
        for i, r in enumerate(radii):
            name = f"{split}-{i}.json"
            point(r, offset=offset if split != "train" else 0.).save(tmp_path/name)
            names.append(name)
        text += f'{split} = {json.dumps(names)}\n'
    spec = tmp_path/"data.toml"
    spec.write_text(text)
    out = tmp_path/"data"
    collect_dataset(spec, out)
    return out, spec


def fit_input(tmp_path, data, *, kind="linear", units=("bohr", "hartree"), bounds=False, extra=""):
    basis = tmp_path/"basis.json"
    basis.write_text(json.dumps([[[0]], [[1]], [[2]], [[3]]]))
    spec = tmp_path/"fit.toml"
    spec.write_text(f'''schema="pyrpmd.jaxpip-fit/1"
dataset={json.dumps(str(data))}
basis="basis.json"
basis_symbols=["H", "H"]
[fit]
kind="{kind}"
alpha={1.0 if units[0] == "bohr" else 0.529177210903}
position_unit="{units[0]}"
energy_unit="{units[1]}"
hidden_layers=[4]
max_steps=60
{extra}
''' + ('[acceptance]\nenergy_rmse_max=1e-8\nforce_rmse_max=1e-8\n' if bounds else ''))
    return spec


def test_reference_import_units_sign_resume_and_corruption(tmp_path):
    data, spec = dataset(tmp_path)
    meta, split = load_dataset(data)
    assert len(split['train']) == 7 and len(split['test']) == 2
    assert meta['contract']['atomic_numbers'] == [1, 1]
    assert np.allclose(split['train'][0].gradient, point(1.).gradient)
    assert collect_dataset(spec, data, resume=True) == meta
    with pytest.raises(ValueError, match='resume'):
        collect_dataset(spec, data)
    path = data/meta['records'][0]['file']
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='checksum'):
        load_dataset(data)


def test_changed_input_rejects_resume(tmp_path):
    data, spec = dataset(tmp_path)
    spec.write_text(spec.read_text()+'\n# changed declaration\n')
    # Semantically identical TOML comments are intentionally irrelevant.
    collect_dataset(spec, data, resume=True)
    point(1.11).save(tmp_path/'train-0.json')
    with pytest.raises(ValueError, match='identical'):
        collect_dataset(spec, data, resume=True)


def test_no_optional_import_for_existing_reference_data(tmp_path):
    data, _ = dataset(tmp_path)
    result = subprocess.run([sys.executable, '-c',
        "import sys; sys.modules['pyscf']=None; sys.modules['jax']=None; "
        "from pyrpmd.pesgen.jaxpip_data import load_dataset; "
        f"load_dataset({str(data)!r})"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_rigid_and_permutation_duplicate_rejected():
    p, other = point(1.), point(1.)
    other.positions = other.positions[::-1] @ np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]]) + 3.
    assert geometry_key(p.symbols,p.positions) == geometry_key(other.symbols,other.positions)
    with pytest.raises(ValueError, match='duplicate'):
        check_points({'train':[p,point(2.)], 'test':[other]})


@pytest.mark.parametrize('change', ['symbols', 'settings', 'convergence', 'version'])
def test_inconsistent_reference_contract(change):
    bad=point(1.4)
    if change=='symbols': bad.symbols=('He','He')
    elif change=='settings': bad.provenance['settings']['basis']='def2-svp'
    elif change=='convergence': bad.provenance['scf_converged']=False
    else: bad.provenance['version']='other'
    with pytest.raises(ValueError):
        check_points({'train':[point(1.), bad]})


@pytest.mark.parametrize('name', ['../escape', '/tmp/absolute', '.'])
def test_manifest_paths_confined(tmp_path,name):
    with pytest.raises(ValueError): member(tmp_path,name)


@pytest.mark.parametrize('kwargs', [dict(kind='wrong'),dict(alpha=0),dict(force_weight=-1),
    dict(seed=True),dict(seed=-1),dict(max_steps=0),dict(hidden_layers=[0]),
    dict(ridge=float('nan')),dict(position_unit='nm'),dict(energy_unit='kcal/mol')])
def test_settings_reject_invalid(kwargs):
    with pytest.raises(ValueError): FitSettings(**kwargs)


@jaxpip_test
@pytest.mark.parametrize('units', [('bohr','hartree'),('bohr','eV'),('angstrom','hartree'),('angstrom','eV')])
def test_real_linear_jaxpip_fit_units_forces_and_checkpoint(tmp_path,units):
    import jax
    from jaxpip.model import PolynomialLinearModel
    data,_=dataset(tmp_path)
    spec=fit_input(tmp_path,data,units=units,bounds=True)
    out=tmp_path/'model'
    manifest=fit_jaxpip(spec,out)
    assert manifest['within_requested_test_bounds'] is True
    metrics=json.loads((out/'metrics.json').read_text())['metrics']
    for split in metrics.values():
        assert split['energy_rmse_hartree'] < 1e-8
        assert split['force_rmse_hartree_per_bohr'] < 1e-8
    assert fit_jaxpip(spec,out,resume=True)==load_fit(out)
    model=PolynomialLinearModel.from_file(str(out/'basis.json'),str(out/'model.eqx'))
    assert str(model.dtype)=='float64' or np.dtype(model.dtype)==np.float64
    q=np.load(out/'probe.npy')
    e,f=jax.vmap(model.get_energy_and_forces)(q)
    h=1e-5; qp=q[0].copy(); qm=q[0].copy(); qp[1,0]+=h; qm[1,0]-=h
    fd=-(float(model.get_energy(qp))-float(model.get_energy(qm)))/(2*h)
    assert float(f[0,1,0]) == pytest.approx(fd,rel=1e-6,abs=1e-8)
    with pytest.raises(ValueError,match='resume'):
        fit_jaxpip(spec,out)
    (out/'model.eqx').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        load_fit(out)


@jaxpip_test
@pytest.mark.parametrize('kind',['linear','nn'])
def test_test_data_never_changes_fit_or_scaler(tmp_path,kind):
    from jaxpip.model import PolynomialLinearModel,PolynomialNeuralNetwork
    manifests=[]; models=[]
    for name,offset in [('a',0.),('b',10.)]:
        root=tmp_path/name; data,_=dataset(root,offset=offset)
        spec=fit_input(root,data,kind=kind,bounds=True)
        out=root/'model'; manifests.append(fit_jaxpip(spec,out))
        cls=PolynomialLinearModel if kind=='linear' else PolynomialNeuralNetwork
        models.append(cls.from_file(str(out/'basis.json'),str(out/'model.eqx')))
        if kind=='nn':
            optimizer=json.loads((out/'metrics.json').read_text())['optimizer']
            assert optimizer['final_loss'] < optimizer['initial_loss']*.01
    q=point(1.5).positions
    np.testing.assert_array_equal(models[0].get_energy(q),models[1].get_energy(q))
    if kind=='nn':
        np.testing.assert_array_equal(models[0].scaler.V_min,models[1].scaler.V_min)
        np.testing.assert_array_equal(models[0].scaler.p_max,models[1].scaler.p_max)
    assert not manifests[1]['within_requested_test_bounds']


@jaxpip_test
def test_no_test_set_no_accuracy_claim(tmp_path):
    data,_=dataset(tmp_path,heldout=False)
    spec=fit_input(tmp_path,data,bounds=True)
    assert not fit_jaxpip(spec,tmp_path/'model')['within_requested_test_bounds']


@jaxpip_test
def test_basis_identity_and_memory_limit(tmp_path):
    data,_=dataset(tmp_path)
    spec=fit_input(tmp_path,data)
    spec.write_text(spec.read_text().replace('["H", "H"]','["He", "He"]'))
    with pytest.raises(ValueError,match='basis_symbols'):
        fit_jaxpip(spec,tmp_path/'bad-order')
    spec=fit_input(tmp_path,data,extra='max_design_elements=1')
    with pytest.raises(ValueError,match='max_design_elements'):
        fit_jaxpip(spec,tmp_path/'too-large')


@pytest.mark.skipif(importlib.util.find_spec('pyscf') is None,reason='optional real PySCF')
def test_actual_pyscf_collection_reuses_cache(tmp_path,monkeypatch):
    from pyrpmd.pesgen.reference import PySCFReference
    q=np.zeros((4,2,3)); q[:,1,0]=[1.2,1.4,1.6,1.8]
    write_xyz(tmp_path/'h2.xyz',('H','H'),q)
    spec=tmp_path/'real.toml'
    spec.write_text('schema="pyrpmd.jaxpip-data/1"\n[geometries]\ntrain="h2.xyz"\n[qm]\nmethod="RHF"\nbasis="sto-3g"\n')
    out=tmp_path/'real'; collect_dataset(spec,out)
    meta,points=load_dataset(out)
    assert meta['contract']['version']!='synthetic-test-fixture'
    assert all(p.hessian is None and p.hirshfeld is None for p in points['train'])
    def forbidden(*a,**k): raise AssertionError('unexpected SCF on completed resume')
    monkeypatch.setattr(PySCFReference,'calculate',forbidden)
    collect_dataset(spec,out,resume=True)


@pytest.mark.parametrize("basis", [[[[0]], [[256]]], [[[0]], [[-1]]], [[[0]], [[True]]], [[[1]], [[2]]], [[[0]]]])
def test_invalid_basis_exponents_and_constant(tmp_path, basis):
    from pyrpmd.pesgen.jaxpip_fit import validate_basis
    path = tmp_path / "basis.json"
    path.write_text(json.dumps(basis))
    with pytest.raises(ValueError):
        validate_basis(path, 2)
