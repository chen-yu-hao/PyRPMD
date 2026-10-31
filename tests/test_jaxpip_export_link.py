"""Wrapper contracts only; the dedicated CI also exports actual JaxPIP graphs."""
import json
from pathlib import Path
import sys
import subprocess
from dataclasses import asdict

import pytest

from pyrpmd.pesgen.jaxpip_data import write_checked
from pyrpmd.pesgen.jaxpip_fit import FitSettings
from pyrpmd.pesgen.jaxpip_export import export_jaxpip, load_export, native_potential_spec
from pyrpmd.pesgen.reference import file_hash


def fixture_fit(tmp_path, accepted=False):
    root=tmp_path/'fit'; root.mkdir()
    for name in ('basis.json','model.eqx','probe.npy','metrics.json'):
        (root/name).write_text('test-only wrapper fixture; not a trained model')
    manifest={'schema':'pyrpmd.jaxpip-fit/1','basis':'basis.json','checkpoint':'model.eqx',
              'settings':asdict(FitSettings()), 'contract':{'atomic_numbers':[1,1], 'charge':0,'multiplicity':1},
              'artifacts':{p.name:file_hash(p) for p in root.iterdir()},
              'within_requested_test_bounds':accepted}
    write_checked(root/'model.json',manifest)
    return root


def fixture_exporter(tmp_path, mode='ok'):
    path=tmp_path/'exporter.py'
    # Stand-in checks the subprocess contract, not ONNX numerical accuracy.
    path.write_text('''import sys,json,hashlib
from pathlib import Path
args=sys.argv[1:]
def value(key): return args[args.index(key)+1]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
output=Path(value('--output')); output.write_bytes(b'wrapper-test-only')
metadata={'atomic_numbers':[1,1], 'state':{'charge':0,'multiplicity':1},
          'position_unit':value('--position-unit'), 'energy_unit':value('--energy-unit'),
          'source_sha256':{'basis':digest(value('--basis')), 'checkpoint':digest(value('--checkpoint'))}}
''' + ("sys.exit(3)\n" if mode=='fail' else '') +
        ("metadata['state']['multiplicity']=3\n" if mode=='mismatch' else '') +
        "print(json.dumps({'sha256':digest(output),'metadata':metadata,'batches_checked':[1,3,32],'model':str(output)}))\n")
    return path


def test_export_wrapper_provenance_portability_and_no_optional_import(tmp_path):
    fit=fixture_fit(tmp_path); exporter=fixture_exporter(tmp_path); out=tmp_path/'export'
    export_jaxpip(fit,exporter,out)
    manifest,_=load_export(out)
    assert manifest['exporter_sha256']==file_hash(exporter)
    assert json.loads((out/'export-report.json').read_text())['model']=='model.onnx'
    lib=tmp_path/'library.so'; lib.write_bytes(b'path-contract-only')
    with pytest.raises(ValueError,match='allow_unvalidated'):
        native_potential_spec(out,lib)
    potential=native_potential_spec(out,lib,allow_unvalidated=True)
    assert potential['params']['model']==str(out/'model.onnx')
    code="import sys; sys.modules['jax']=None; sys.modules['pyscf']=None; "
    code+="from pyrpmd.pesgen.jaxpip_export import native_potential_spec; "
    code+=f"native_potential_spec({str(out)!r},{str(lib)!r},allow_unvalidated=True)"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with pytest.raises(ValueError,match='never overwritten'):
        export_jaxpip(fit,exporter,out)
    (out/'model.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        load_export(out)


@pytest.mark.parametrize('mode',['fail','mismatch'])
def test_failed_export_is_not_published(tmp_path,mode):
    fit=fixture_fit(tmp_path); out=tmp_path/'failed'
    with pytest.raises((ValueError,RuntimeError)):
        export_jaxpip(fit,fixture_exporter(tmp_path,mode),out)
    assert not (out/'export.json').exists()
    assert (out/'export.stderr.log').exists()


@pytest.mark.parametrize('kwargs',[{'allow_unvalidated':'yes'}, {'intra_op_threads':0},
    {'intra_op_threads':True},{'device':'gpu'}, {'device':'cuda:-1'}])
def test_native_config_rejects_invalid(tmp_path,kwargs):
    fit=fixture_fit(tmp_path,accepted=True); out=tmp_path/'out'
    export_jaxpip(fit,fixture_exporter(tmp_path),out)
    lib=tmp_path/'plugin.so'; lib.write_bytes(b'not-loaded')
    with pytest.raises(ValueError): native_potential_spec(out,lib,**kwargs)


def test_missing_exporter_and_overlapping_output(tmp_path):
    fit=fixture_fit(tmp_path)
    with pytest.raises(ValueError): export_jaxpip(fit,tmp_path/'missing',tmp_path/'out')
    with pytest.raises(ValueError): export_jaxpip(fit,fixture_exporter(tmp_path),fit)
