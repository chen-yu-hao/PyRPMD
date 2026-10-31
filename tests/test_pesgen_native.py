"""Real pinned Caracal/PySCF integration, enabled by explicit local build paths."""
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest

from pyrpmd.pesgen.build import bundle_native, load_build, run_command, source_digest, SOURCE_DIGEST
from pyrpmd.pesgen.formats import read_xyz, write_xyz
from pyrpmd.pesgen.reference import PySCFReference, PySCFSettings, file_hash
from pyrpmd.pesgen.workflow import generate, run_generated_job, verify_bundle
from pyrpmd.potentials.cabi import CABIPotential

BUILD = os.environ.get('PYRPMD_PESGEN_BUILD')
SOURCE = os.environ.get('PYRPMD_CARACAL_SOURCE')
pytestmark = pytest.mark.skipif(not BUILD or not SOURCE,reason='explicit pinned native Caracal build/source required')


@pytest.fixture(scope='module')
def build():
    path=Path(BUILD).resolve(); load_build(path)
    assert source_digest(Path(SOURCE))==SOURCE_DIGEST
    return path


@pytest.fixture(scope='module')
def water(tmp_path_factory, build):
    pytest.importorskip('pyscf')
    root=tmp_path_factory.mktemp('water-qmdff')
    write_xyz(root/'water.xyz',('O','H','H'),[[0,0,0],[0,1.4,1.1],[0,-1.4,1.1]])
    inp=root/'pes.toml'
    inp.write_text(f'''schema="pyrpmd.pesgen/1"
method="qmdff"
endpoints=["water.xyz"]
native_build={json.dumps(str(build))}
optimize_endpoints=true
[qm]
method="RHF"
basis="sto-3g"
''')
    result=generate(inp,root/'run')
    symbols,q=read_xyz(root/'run/bundle/model/start.xyz')
    return root,inp,result,symbols,q[0]


def test_real_qmdff_pipeline_resume_and_gate(water):
    root,inp,result,symbols,q=water
    assert result['training']['energy_rmse_hartree']<1e-9
    assert result['training']['force_rmse_hartree_per_bohr']<1e-5
    assert not result['validated_physical_accuracy']
    assert not result['within_requested_validation_bounds']
    hashes={p:file_hash(p) for p in (root/'run').rglob('*.json')}
    assert generate(inp,root/'run',resume=True)==result
    assert all(file_hash(p)==v for p,v in hashes.items())
    with pytest.raises(ValueError,match='unvalidated'):
        run_generated_job(root/'run',{},output=root/'unused')
    changed=inp.read_text().replace('sto-3g','6-31g');inp.write_text(changed)
    with pytest.raises(ValueError,match='changed'):
        generate(inp,root/'run',resume=True)
    inp.write_text(changed.replace('6-31g','sto-3g'))


def test_qmdff_native_forces_contexts_and_fused_steps(water):
    from pyrpmd.dynamics import Dynamics,State,native_capabilities
    from pyrpmd.system import _masses_for_symbols
    root,_,_,symbols,q=water;lib=root/'run/bundle/libpes.so'
    p=CABIPotential(lib);other=CABIPotential(lib)
    try:
        q=q.copy();q[1,2]+=.03
        frames=np.stack([q,q+.1,q-.2]);value=p.evaluate(frames)
        np.testing.assert_allclose(value.energy,value.energy[0],atol=1e-10)
        np.testing.assert_allclose(other.evaluate(frames[::-1].copy()).forces[::-1],value.forces,atol=1e-12)
        h=1e-4;plus=q.copy();minus=q.copy();plus[1,2]+=h;minus[1,2]-=h
        e=p.evaluate(np.stack([plus,minus])).energy
        assert abs(-(e[0]-e[1])/(2*h)-value.forces[0,1,2])<1e-7
        with pytest.raises(Exception): CABIPotential(lib,config={'natoms':99})
        if native_capabilities()['available']:
            state=State(frames[:1].copy(),np.zeros((1,3,3)))
            dyn=Dynamics(p,_masses_for_symbols(symbols),1000.,1,.1,loop_backend='native')
            calls=[]
            def profile(frame,event,arg):
                if event=='call' and frame.f_code is CABIPotential.evaluate.__code__:
                    calls.append(1)
            sys.setprofile(profile)
            try: dyn.propagate(state,30,backend='native')
            finally: sys.setprofile(None)
            assert dyn.propagation_stats['native_steps']==30
            assert len(calls)==0  # not even one per-step Python potential callback
    finally:
        p.close();other.close()


def test_zero_hirshfeld_h2_is_not_missing(tmp_path,build):
    pytest.importorskip('pyscf')
    write_xyz(tmp_path/'h2.xyz',('H','H'),[[0,0,0],[0,0,1.4]])
    inp=tmp_path/'pes.toml';inp.write_text(f'''schema="pyrpmd.pesgen/1"
method="qmdff"
endpoints=["h2.xyz"]
native_build={json.dumps(str(build))}
optimize_endpoints=true
[qm]
basis="sto-3g"
''')
    result=generate(inp,tmp_path/'run')
    assert result['training']['energy_rmse_hartree']<1e-10


@pytest.mark.parametrize('method',['de_evb','treq'])
def test_original_models_and_force_consistency(tmp_path,build,method):
    source=Path(SOURCE)
    rel='examples/explore/egrad_norbornene' if method=='de_evb' else 'examples/calc_rate/cf3o+ch4_treq'
    folder=tmp_path/'original';shutil.copytree(source/rel,folder)
    pathfile='path.xyz' if method=='de_evb' else 'irc.xyz'
    keyfile='scan.key' if method=='de_evb' else 'rate.key'
    symbols,q=read_xyz(folder/pathfile)
    write_xyz(folder/'start.xyz',symbols,q[0])
    key='\n'.join(line for line in (folder/keyfile).read_text().splitlines()
                  if not line.strip().lower().startswith('xyzstart'))+'\nxyzstart start.xyz\n'
    files={'model.key':key,'start.xyz':(folder/'start.xyz').read_text()}
    names=(['min1.qmdff','min2.qmdff','evb_pars.dat','path.xyz'] if method=='de_evb' else
           ['educts.qmdff','products.qmdff','grad_hess.dat','coord_def.inp','irc.xyz','irc_ens.dat'])
    files.update({name:(folder/name).read_text() for name in names})
    lib=bundle_native(build,files,symbols,tmp_path/'bundle',force_mode='fd' if method=='treq' else 'analytic')
    p=CABIPotential(lib)
    try:
        frames=q[np.linspace(len(q)//4,3*len(q)//4,3,dtype=int)]
        v=p.evaluate(frames);rev=p.evaluate(frames[::-1].copy())
        np.testing.assert_allclose(v.energy,rev.energy[::-1],atol=1e-12)
        np.testing.assert_allclose(v.forces,rev.forces[::-1],atol=1e-12)
        h=2e-5;plus=frames[1].copy();minus=plus.copy();plus[0,0]+=h;minus[0,0]-=h
        e=p.evaluate(np.stack([plus,minus])).energy
        assert abs(-(e[0]-e[1])/(2*h)-v.forces[1,0,0])<(2e-5 if method=='treq' else 1e-7)
        if method=='treq':
            assert p.metadata['capabilities']['force_kind']=='finite_difference'
        # compare the same compiled upstream energy with the analytic mode.
        lib2=bundle_native(build,files,symbols,tmp_path/'analytic',force_mode='analytic')
        p2=CABIPotential(lib2)
        try: np.testing.assert_allclose(p2.evaluate(frames).energy,v.energy,atol=1e-12)
        finally:p2.close()
    finally:p.close()
    verify_bundle(tmp_path/'bundle')
    with open(tmp_path/'bundle/caracal_worker.x','ab') as out:out.write(b'corruption')
    with pytest.raises(ValueError,match='worker'): verify_bundle(tmp_path/'bundle')
    with pytest.raises(Exception,match='integrity'): CABIPotential(lib)


@pytest.mark.skipif(os.environ.get('PYRPMD_PESGEN_FULL')!='1',reason='full MPI installation for original optimizer')
def test_original_evb_optimizer(tmp_path,build):
    folder=tmp_path/'fit';shutil.copytree(Path(SOURCE)/'examples/evbopt/dE-EVB',folder)
    run_command([str(build/'Caracal/bin/evbopt.x'),'evbopt.key'],folder,folder/'run.log',timeout=120)
    assert (folder/'evb_pars.dat').stat().st_size>10
    # Inspect fitted values, not merely process success.
    assert np.isfinite(np.loadtxt(folder/'evb_pars.dat',skiprows=1)).all()


def test_real_pyscf_treq_generation(tmp_path,build):
    import importlib.util
    file=Path(__file__).resolve().parents[1]/'examples/pesgen/nh3_inversion.py'
    spec=importlib.util.spec_from_file_location('nh3_pesgen_example',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    inp=module.prepare(tmp_path/'nh3',build)
    summary=generate(inp,tmp_path/'run')
    assert summary['method']=='treq' and summary['experimental']
    assert summary['force_mode']=='fourth_order_energy_difference'
    assert summary['training']['samples']==9
    assert np.isfinite(summary['training']['energy_rmse_hartree'])
    assert not summary['validated_physical_accuracy']
    assert not summary['within_requested_validation_bounds']
    assert generate(inp,tmp_path/'run',resume=True)==summary


@pytest.mark.skipif(os.environ.get('PYRPMD_PESGEN_FULL')!='1',reason='full MPI installation for original optimizer')
def test_real_pyscf_evb_generation(tmp_path,build):
    import importlib.util
    file=Path(__file__).resolve().parents[1]/'examples/pesgen/nh3_inversion.py'
    spec=importlib.util.spec_from_file_location('nh3_evb_example',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    inp=module.prepare(tmp_path/'nh3',build,method='de_evb')
    summary=generate(inp,tmp_path/'run')
    assert summary['method']=='de_evb' and not summary['experimental']
    assert summary['force_mode']=='analytic'
    assert summary['training']['samples']==9
    assert np.isfinite(summary['training']['energy_rmse_hartree'])
    assert not summary['validated_physical_accuracy']
    assert not summary['within_requested_validation_bounds']
    assert generate(inp,tmp_path/'run',resume=True)==summary
