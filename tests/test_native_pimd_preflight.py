"""Public preparation validates the selected PIMD engine without propagation."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd import prepare
from pyrpmd.cli import main
from pyrpmd.potentials import make_potential


ROOT = Path(__file__).resolve().parents[1]


def job(engine="native", npt=False):
    result = {
        "task": "pimd", "beads": 4, "temperature": "300 K",
        "system": {"symbols": ["H", "H"], "geometry": [[.1, .2, .3], [1., 1., 1.]],
                   "cell": (np.eye(3) * 5).tolist(), "pbc": [True, True, True]},
        "potential": {"provider": "free"},
        "dynamics": {"timestep": "0.1 fs", "loop_backend": "native" if engine == "native" else "python"},
        "sampling": {"engine": engine, "thermostat": "pile-l", "equilibration_steps": 0,
                     "production_steps": 4, "sample_interval": 1, "replicas": 1},
    }
    if npt:
        result["barostat"] = ({"mode": "bzp", "pressure": "1 bar", "tau": "10 fs"}
                              if engine == "native" else
                              {"method": "bzp", "pressure": "1 bar", "time_constant": "10 fs"})
    return result


@pytest.mark.parametrize("engine", ["legacy", "native"])
@pytest.mark.parametrize("npt", [False, True])
@pytest.mark.parametrize("numerical", [False, True])
def test_prepare_distinguishes_engine_contracts_without_running(tmp_path, monkeypatch, engine, npt, numerical):
    config = job(engine, npt)
    output = tmp_path / "not-created"
    config["output"] = {"directory": str(output)}
    before = copy.deepcopy(config)
    if engine == "native":
        from pyrpmd.workflows import native_pimd

        def unexpected(*args, **kwargs):
            raise AssertionError("preparation must not propagate or require legacy BZP")

        if numerical:
            pytest.importorskip("pyrpmd._thermal")
            from pyrpmd.native_ensemble import NativeEnsemble
            from pyrpmd import barostat
            monkeypatch.setattr(NativeEnsemble, "run", unexpected)
            monkeypatch.setattr(barostat, "BZPBarostat", unexpected)
        else:
            monkeypatch.setattr(native_pimd, "_engine", unexpected)
    result = prepare(config, numerical=numerical)
    metadata = result["materials_preparation"]
    assert result["ready"]
    assert metadata["ensemble"] == ("npt" if npt else "fixed_cell")
    assert metadata["trajectory_storage"] == ("native_pimd_journal" if engine == "native" else "stream")
    assert (metadata["barostat"] is not None) == npt
    if numerical:
        checked = result["pes_check"]
        assert checked["energy_hartree"] == [0.] * (4 if engine == "native" else 1)
        assert checked["status"] == [0] * (4 if engine == "native" else 1)
        if engine == "native":
            assert checked["step"] == 0
            assert checked["force_evaluations"] == 1
            assert "initial_configurational_pressure_au" not in checked
        elif npt:
            assert checked["initial_configurational_pressure_au"] == 0.
    else:
        assert "pes_check" not in result
    assert config == before
    assert not output.exists()


def test_native_fixed_slab_cosine_preflight_checks_full_ring(tmp_path):
    pytest.importorskip("pyrpmd._thermal")
    config = job()
    config["system"].update(pbc=[True, True, False], fixed_atoms=[1])
    config["potential"] = {"provider": "builtin", "name": "periodic_cosine",
                           "params": {"height": [.1, .2, 0.]}}
    config["output"] = {"directory": str(tmp_path / "not-created")}
    result = prepare(config, numerical=True)
    assert result["materials_preparation"]["fixed_atoms"] == [1]
    from pyrpmd.input import JobSpec
    resolved, system = JobSpec.from_mapping(config).resolve()
    provider = make_potential(resolved["potential"], system)
    expected = provider.evaluate(np.repeat(system.geometry[None], config["beads"], axis=0))
    np.testing.assert_allclose(result["pes_check"]["energy_hartree"], expected.energy, atol=2e-16)
    assert result["pes_check"]["step"] == 0
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("filename", ["pimd_native.inp", "pimd_native_npt.inp"])
def test_documented_native_validate_command_succeeds_without_output(tmp_path, monkeypatch, capsys, filename):
    pytest.importorskip("pyrpmd._thermal")
    monkeypatch.chdir(tmp_path)
    assert main(["validate", str(ROOT / "examples" / "ensembles" / filename), "--check-pes"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["pes_check"]["step"] == 0
    assert result["pes_check"]["force_evaluations"] == 1
    assert not list(tmp_path.iterdir())


def test_native_preflight_rejects_unavailable_binding_and_closes_provider(monkeypatch, tmp_path):
    import pyrpmd.potentials as potentials

    class Unsupported:
        closed = False

        def evaluate(self, q):
            raise AssertionError("native preflight must not use a Python fallback")

        def close(self):
            self.closed = True

    provider = Unsupported()
    config = job()
    config["potential"] = {"provider": "native_periodic", "library": str(tmp_path / "unused.so")}
    monkeypatch.setattr(potentials, "make_potential", lambda *args: provider)
    with pytest.raises(ValueError, match="no callback-free native binding"):
        prepare(config, numerical=True)
    assert provider.closed
    assert not list(tmp_path.iterdir())


def test_native_constructor_failure_closes_provider(monkeypatch):
    pytest.importorskip("pyrpmd._thermal")
    import pyrpmd.potentials as potentials
    from pyrpmd.potentials.builtin import FreePotential

    provider = FreePotential()
    closed = []
    provider.close = lambda: closed.append(True)
    monkeypatch.setattr(potentials, "make_potential", lambda *args: provider)
    config = job()
    config["sampling"]["thermostat_params"] = {"gamma_centroid": -1.}
    with pytest.raises(ValueError, match="gamma_centroid"):
        prepare(config, numerical=True)
    assert closed == [True]


@pytest.fixture
def periodic_library(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for the native periodic ABI check")
    source = tmp_path / "volume.c"
    source.write_text(r'''
#include <stdint.h>
#include <math.h>
int32_t evaluate(void*ctx,int64_t P,int64_t N,const double*q,const double*h,
  const int32_t*pbc,double*e,double*f,double*v,char*err,int64_t cap){
    double volume=h[0]*h[4]*h[8];
    for(int64_t b=0;b<P;++b){
        e[b]=0.001*volume;
        for(int64_t i=0;i<3*N;++i)f[b*3*N+i]=0;
        for(int i=0;i<9;++i)v[b*9+i]=(i%4==0)?-e[b]:0;
    }
    return 0;
}
int32_t nonfinite(void*ctx,int64_t P,int64_t N,const double*q,const double*h,
  const int32_t*pbc,double*e,double*f,double*v,char*err,int64_t cap){
    evaluate(ctx,P,N,q,h,pbc,e,f,v,err,cap);
    f[0]=NAN;
    return 0;
}
''')
    library = tmp_path / "volume.so"
    subprocess.run([compiler, "-shared", "-O2", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


def test_native_periodic_npt_uses_compiled_contract_without_set_cell_or_close(tmp_path, monkeypatch, periodic_library):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd.native_ensemble import NativePeriodicPES

    def unexpected(*args, **kwargs):
        raise AssertionError("compiled preflight must not call the Python evaluation adapter")

    monkeypatch.setattr(NativePeriodicPES, "evaluate", unexpected)
    config = job(npt=True)
    config["potential"] = {"provider": "native_periodic", "library": str(periodic_library)}
    output = tmp_path / "not-created"
    config["output"] = {"directory": str(output)}
    result = prepare(config, numerical=True)
    checked = result["pes_check"]
    np.testing.assert_allclose(checked["energy_hartree"], [.125] * 4)
    np.testing.assert_allclose(checked["initial_virial_hartree"], np.repeat((-.125 * np.eye(3))[None], 4, axis=0))
    assert checked["step"] == 0
    assert checked["force_evaluations"] == 1
    assert not output.exists()


def test_native_nonfinite_force_releases_lease_before_closing(monkeypatch, periodic_library, tmp_path):
    pytest.importorskip("pyrpmd._thermal")
    import pyrpmd.potentials as potentials

    lifetime = {"leases": 0, "closed": False}

    def instrumented_factory(config, system):
        provider = make_potential(config, system)
        original_acquire = provider.acquire

        @contextmanager
        def acquire():
            with original_acquire() as descriptor:
                lifetime["leases"] += 1
                try:
                    yield descriptor
                finally:
                    lifetime["leases"] -= 1

        def close():
            assert lifetime["leases"] == 0
            lifetime["closed"] = True

        provider.acquire, provider.close = acquire, close
        return provider

    monkeypatch.setattr(potentials, "make_potential", instrumented_factory)
    config = job(npt=True)
    config["potential"] = {"provider": "native_periodic", "library": str(periodic_library),
                           "function": "nonfinite"}
    output = tmp_path / "not-created"
    config["output"] = {"directory": str(output)}
    with pytest.raises(ValueError, match="nonfinite energy, force or virial"):
        prepare(config, numerical=True)
    assert lifetime == {"leases": 0, "closed": True}
    assert not output.exists()
