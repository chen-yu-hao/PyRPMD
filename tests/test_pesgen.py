"""Offline PES contracts; real QM tests skip only when optional PySCF is absent."""
import json
from pathlib import Path
import numpy as np
import pytest

from pyrpmd.pesgen.reference import ReferencePoint, PySCFSettings, PySCFReference, mayer_bond_orders
from pyrpmd.pesgen.formats import read_xyz, write_xyz, export_qmdff_reference, validate_coordinates, write_grad_hess
from pyrpmd.pesgen.build import source_digest, SOURCE_DIGEST, patch_source


def point():
    return ReferencePoint(("H", "H"), [[0,0,0],[0,0,1.4]], -1., np.zeros((2,3)),
                          np.eye(6), [0,0], [[0,1],[1,0]], {"test": True})


def test_reference_integrity_and_units(tmp_path):
    p = point(); file = tmp_path / "reference.json"; p.save(file)
    loaded = ReferencePoint.load(file)
    np.testing.assert_array_equal(loaded.hessian, p.hessian)
    outer = json.loads(file.read_text()); outer["data"]["energy"] += .1
    file.write_text(json.dumps(outer))
    with pytest.raises(ValueError, match="checksum"):
        ReferencePoint.load(file)


@pytest.mark.parametrize("key,value", [("method","CCSD"),("method","RKS"),("spin",True),
    ("spin",1),("charge",.5),("max_cycle",0),("grid_level",10),("conv_tol",0),
    ("xc","pbe"),("hessian","magic"),("basis","")])
def test_settings_reject_invalid(key, value):
    with pytest.raises(ValueError):
        PySCFSettings(**{key:value})


def test_reference_validates_derivatives():
    with pytest.raises(ValueError, match="finite"):
        ReferencePoint(("H",), [[np.nan,0,0]], 0, [[0,0,0]])
    with pytest.raises(ValueError, match="symmetric"):
        ReferencePoint(("H",), [[0,0,0]], 0, [[0,0,0]], [[1,2,3],[0,1,0],[0,0,1]])


def test_mayer_spin_reduction():
    dm = np.ones((2,2)); s=np.eye(2); slices=[(0,1),(1,2)]
    restricted = mayer_bond_orders(dm,s,slices)
    np.testing.assert_array_equal(restricted,[[0,1],[1,0]])
    np.testing.assert_allclose(mayer_bond_orders(np.stack([dm/2,dm/2]),s,slices),restricted)
    with pytest.raises(ValueError):
        mayer_bond_orders(np.zeros((3,2,2)),s,slices)


def test_formats_are_explicit_and_ordered(tmp_path):
    p=point(); xyz=tmp_path/'m.xyz';write_xyz(xyz,p.symbols,[p.positions,p.positions+.1])
    symbols, frames=read_xyz(xyz); assert symbols==p.symbols
    np.testing.assert_allclose(frames[0],p.positions,atol=1e-14)
    export_qmdff_reference(p,tmp_path/'reference')
    text=(tmp_path/'reference.out').read_text()
    assert 'NOT an ORCA calculation' in text
    assert '1.0000000000000000E+00' in text  # upstream readl only understands uppercase E
    assert 'HIRSHFELD ANALYSIS' in text
    write_grad_hess([p],tmp_path/'grad_hess.dat')
    assert 'NPOINTS 1' in (tmp_path/'grad_hess.dat').read_text()
    with pytest.raises(ValueError):
        export_qmdff_reference(p,tmp_path/'bad-name')


def test_coordinate_and_xyz_rejections(tmp_path):
    file=tmp_path/'coord.inp';file.write_text('1 2\n1 3\n2 1 3\n')
    assert len(validate_coordinates(file,3))==3
    file.write_text('1 2\n1 1\n2 1 3\n')
    with pytest.raises(ValueError): validate_coordinates(file,3)
    xyz=tmp_path/'bad.xyz';xyz.write_text('2\ntruncated\nH 0 0 0\n')
    with pytest.raises(ValueError): read_xyz(xyz)


def test_native_header_is_current():
    root=Path(__file__).resolve().parents[1]
    assert (root/'native/include/pyrpmd/pes_api_v1.h').read_bytes()==(root/'src/pyrpmd/pesgen/_native/pes_api_v1.h').read_bytes()


@pytest.fixture(scope='module')
def real_reference():
    pytest.importorskip('pyscf')
    return PySCFReference(PySCFSettings(basis='sto-3g',conv_tol=1e-11))


def test_real_pyscf_hessian_force_and_populations(real_reference):
    engine=real_reference;q=np.array([[0.,0,0],[0,0,1.4]])
    p=engine.calculate(('H','H'),q,hessian=True,populations=True)
    assert abs(sum(p.hirshfeld))<5e-4
    assert abs(p.mayer[0,1]-1)<1e-6
    h=1e-3; plus=q.copy();plus[1,2]+=h;minus=q.copy();minus[1,2]-=h
    pp=engine.calculate(('H','H'),plus,hessian=False);pm=engine.calculate(('H','H'),minus,hessian=False)
    assert abs((pp.energy-pm.energy)/(2*h)-p.gradient[1,2])<1e-6
    np.testing.assert_allclose((pp.gradient-pm.gradient).ravel()/(2*h),p.hessian[:,5],atol=3e-6)
    np.testing.assert_allclose(p.hessian,p.hessian.T,atol=1e-12)
    assert p.provenance['scf_converged'] is True


@pytest.mark.parametrize('method,spin,xc',[('UHF',2,None),('RKS',0,'pbe'),('UKS',2,'pbe')])
def test_real_pyscf_other_scf_methods(method,spin,xc):
    pytest.importorskip('pyscf')
    engine=PySCFReference(PySCFSettings(method=method,basis='sto-3g',spin=spin,xc=xc,grid_level=2))
    q=np.array([[0.,0,0],[0,0,1.7]])
    p=engine.calculate(('H','H'),q,hessian=False)
    plus=q.copy();minus=q.copy();plus[1,2]+=1e-4;minus[1,2]-=1e-4
    ep=engine.calculate(('H','H'),plus,hessian=False).energy
    em=engine.calculate(('H','H'),minus,hessian=False).energy
    assert abs((ep-em)/2e-4-p.gradient[1,2])<2e-5


def test_real_pyscf_finite_difference_hessian(real_reference):
    q=np.array([[0.,0,0],[0,0,1.4]])
    expected=real_reference.calculate(('H','H'),q).hessian
    engine=PySCFReference(PySCFSettings(basis='sto-3g',hessian='finite_difference',conv_tol=1e-11))
    got=engine.calculate(('H','H'),q).hessian
    np.testing.assert_allclose(got,expected,atol=3e-6)


def test_scf_nonconvergence_is_not_a_reference():
    pytest.importorskip('pyscf')
    engine=PySCFReference(PySCFSettings(basis='sto-3g',max_cycle=1,conv_tol=1e-14))
    with pytest.raises(RuntimeError,match='converge'):
        engine.calculate(('O','H','H'),[[0,0,0],[0,1.4,1.1],[0,-1.4,1.1]])
