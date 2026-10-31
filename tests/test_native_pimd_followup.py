"""Fused maps, streamed observations, and standard-input native PIMD."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.native_ensemble import NativeEnsemble
from pyrpmd.input import JobSpec, InputError
from pyrpmd.runner import run, resume_run
from pyrpmd.workflows import native_pimd as workflow


def engine(beads=8, mode="pile-l", **kwargs):
    t = {"mode": mode}
    if "gle" in mode or mode == "piglet":
        t["gle"] = {"A": [[1., .3], [-.3, 1.]], "C": [[1., 0.], [0., 1.]]}
        if mode == "piglet":
            t["gle"].update(A=np.repeat(np.array(t["gle"]["A"])[None], beads, axis=0).tolist(),
                            beta=2., beads=beads, provenance="test-only canonical matrices")
    options = dict(q=np.random.default_rng(3).normal(size=(beads, 2, 3))*.1,
                   masses=[1., 2.], beta=2., dt=.03,
                   potential={"kind": "harmonic", "k": .7}, seed=21, thermostat=t)
    options.update(kwargs)
    return NativeEnsemble(**options)


@pytest.mark.parametrize("mode", ["none", "pile-l", "pile-g", "nm-gle", "nm-gle-g", "piglet", "trpmd"])
@pytest.mark.parametrize("beads", [1, 3, 8, 32])
@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
def test_fused_map_matches_unfused_reference(mode, beads, integrator):
    fused = engine(beads, mode, integrator=integrator)
    reference = engine(beads, mode, integrator=integrator, fuse_modes=False)
    a, b = fused.run(40, stride=7), reference.run(40, stride=7)
    np.testing.assert_allclose(a["samples"], b["samples"], rtol=5e-12, atol=5e-12)
    for key in ("q", "p", "aux"):
        np.testing.assert_allclose(fused.state()[key], reference.state()[key], rtol=1e-10, atol=5e-12)
    assert fused.state()["rng"] == reference.state()["rng"]
    # Rounding path is fingerprinted: exact-restart promises are not overstated.
    with pytest.raises(ValueError, match="physics"):
        fused.restore(reference.state())


def test_native_moments_and_offset_clock(tmp_path):
    a, b = engine(), engine()
    expected = a.run(37, stride=3, sample_start=7, record_centroids=True, record_moments=True)
    chunks = [b.run(n, stride=3, sample_start=7, record_centroids=True, record_moments=True,
                    trajectory=tmp_path/f"part-{i}.npy") for i, n in enumerate((7, 11, 5, 14))]
    np.testing.assert_array_equal(expected["samples"][:, 0], np.arange(10, 38, 3))
    for key in ("samples", "centroids", "velocities", "radii_gyration_squared"):
        np.testing.assert_array_equal(np.concatenate([c[key] for c in chunks]), expected[key])
    state = a.state()
    np.testing.assert_allclose(expected["centroids"][-1], state["q"].mean(axis=0), atol=2e-14)
    np.testing.assert_allclose(expected["velocities"][-1], state["p"].mean(axis=0)/np.array([1., 2.])[:, None], atol=2e-14)
    radius = np.mean(np.sum((state["q"]-state["q"].mean(0))**2, axis=2), axis=0)
    np.testing.assert_allclose(expected["radii_gyration_squared"][-1], radius, atol=2e-14)
    assert a.state()["digest"] == b.state()["digest"]


@pytest.mark.parametrize("failure", ["link", "load", "fsync"])
def test_publication_failure_rolls_back_state_and_does_not_leave_file(tmp_path, monkeypatch, failure):
    import pyrpmd.native_ensemble as module
    e = engine()
    before = e.state()["digest"]
    def fail(*args, **kwargs):
        raise OSError("injected publication failure")
    target = module.np if failure == "load" else module.os
    with monkeypatch.context() as patch:
        patch.setattr(target, failure, fail)
        with pytest.raises(OSError, match="injected publication"):
            e.run(25, trajectory=tmp_path/"records.npy")
    assert e.state()["digest"] == before
    assert not list(tmp_path.iterdir())
    reference = engine()
    e.run(25)
    reference.run(25)
    assert e.state()["digest"] == reference.state()["digest"]


def test_concurrent_output_is_not_overwritten_or_removed(tmp_path, monkeypatch):
    import pyrpmd.native_ensemble as module
    e = engine()
    before = e.state()["digest"]
    def other_writer(source, target):
        Path(target).write_text("another process owns this file")
        raise FileExistsError(target)
    monkeypatch.setattr(module.os, "link", other_writer)
    with pytest.raises(FileExistsError):
        e.run(15, trajectory=tmp_path/"records.npy")
    assert (tmp_path/"records.npy").read_text() == "another process owns this file"
    assert e.state()["digest"] == before


def config(*, blocks=11, mode="pile-l", replicas=1, barostat=None):
    c = {"task": "pimd", "temperature": "300 K", "beads": 4, "seed": 45,
         "system": {"symbols": ["H", "H"], "geometry": [[0., 0., 0.], [.1, .2, .3]], "geometry_unit": "bohr"},
         "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .01}},
         "dynamics": {"timestep": "0.1 autime", "loop_backend": "native"},
         "sampling": {"engine": "native", "thermostat": mode, "equilibration_steps": 7,
                      "production_steps": 30, "sample_interval": 3, "replicas": replicas},
         "output": {"checkpoint_steps": blocks}}
    if barostat is not None:
        c["potential"] = {"provider": "free"}
        c["system"].update(cell=np.eye(3).tolist(), cell_unit="bohr", pbc=[True]*3)
        c["barostat"] = {"mode": barostat, "pressure": "1 bar", "tau": "10 fs", "gamma": "0.01 fs^-1"}
        if barostat == "rgb":
            c["barostat"].pop("pressure")
            c["barostat"].update(stress=[["1 bar" if i==j else "0 bar" for j in range(3)] for i in range(3)],
                                  reference_cell=np.eye(3).tolist(), reference_cell_unit="bohr")
        if barostat == "scr":
            c["beads"] = 1
            c["barostat"].pop("gamma")
            c["barostat"]["compressibility"] = "0.001 bar^-1"
    return c


def read_records(path):
    journal = json.loads(Path(path).read_text())
    arrays = [np.load(Path(path).parent / b["directory"] / "samples.npy") for b in journal["blocks"] if b["samples_sha256"]]
    return np.concatenate(arrays)


@pytest.mark.parametrize("mode", ["pile-l", "pile-g", "andersen", "langevin", "svr", "ffl", "gle", "nm-gle", "nm-gle-g"])
def test_standard_workflow_is_native_and_chunk_independent(tmp_path, mode):
    c = config(blocks=11, mode=mode)
    if "gle" in mode:
        c["sampling"]["thermostat_params"] = {"gle": {"A": [[1., .2], [-.2, 1.]], "A_unit": "fs^-1"}}
    a = run(c, output=tmp_path/"a")
    c["output"]["checkpoint_steps"] = 2
    b = run(c, output=tmp_path/"b")
    assert a["python_step_callbacks"] == 0
    assert a["samples_per_replica"] == [10]
    assert a["propagation"][0]["native_blocks"] == 4
    np.testing.assert_array_equal(read_records(a["files"]["journals"][0]), read_records(b["files"]["journals"][0]))
    assert a["estimators"] == b["estimators"]


@pytest.mark.parametrize("barostat", ["bzp", "mttk", "rgb", "scr"])
def test_standard_npt_workflow(tmp_path, barostat):
    c = config(barostat=barostat)
    a = run(c, output=tmp_path/barostat)
    records = read_records(a["files"]["journals"][0])
    assert np.all(records[:, 6] > 0)
    assert a["barostat"] == barostat
    assert a["pressure_mean_au"] == pytest.approx(records[:, 7].mean())
    assert resume_run(tmp_path/barostat)["estimators"] == a["estimators"]


def test_native_workflow_committed_prefix_recovery(tmp_path, monkeypatch):
    c = config()
    reference = run(c, output=tmp_path/"reference")
    original = workflow.write_json
    def interrupted(path, data):
        if Path(path).name == "journal.json" and data["blocks"][-1]["end"] == 29:
            raise OSError("injected journal failure")
        return original(path, data)
    with monkeypatch.context() as patch:
        patch.setattr(workflow, "write_json", interrupted)
        with pytest.raises(OSError, match="journal failure"):
            run(c, output=tmp_path/"resumed")
    path = tmp_path/"resumed"/"samples"/"native_pimd_0000"/"journal.json"
    saved = json.loads(path.read_text())
    assert saved["blocks"][-1]["end"] == 18
    assert len(list(path.parent.glob("chunk-*"))) == len(saved["blocks"])
    # Execution/checkpoint changes do not alter the RNG or absolute sample clock.
    resolved = tmp_path/"resumed"/"input"/"resolved.json"
    data = json.loads(resolved.read_text());data["output"]["checkpoint_steps"] = 4
    resolved.write_text(json.dumps(data))
    resumed = resume_run(tmp_path/"resumed")
    np.testing.assert_array_equal(read_records(reference["files"]["journals"][0]), read_records(resumed["files"]["journals"][0]))
    assert resumed["estimators"] == reference["estimators"]
    # Completed restart must not construct/evaluate any potential.
    with monkeypatch.context() as patch:
        patch.setattr(workflow, "make_task_potential", lambda *args: pytest.fail("completed resume evaluated PES"))
        assert resume_run(tmp_path/"resumed")["estimators"] == resumed["estimators"]


def test_native_workflow_corrupt_record_rejected(tmp_path):
    result = run(config(), output=tmp_path/"a")
    path = Path(result["files"]["journals"][0])
    journal = json.loads(path.read_text())
    file = next(path.parent/b["directory"]/"samples.npy" for b in journal["blocks"] if b["samples_sha256"])
    with file.open("r+b") as stream:
        stream.seek(-1, 2);stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        resume_run(tmp_path/"a")


def test_native_workflow_process_replicas(tmp_path):
    c = config(replicas=2)
    sequential = run(c, output=tmp_path/"sequential")
    c["parallel"] = {"backend": "process", "trajectory_workers": 2}
    parallel = run(c, output=tmp_path/"parallel")
    assert sequential["estimators"] == parallel["estimators"]
    for a, b in zip(sequential["files"]["journals"], parallel["files"]["journals"]):
        np.testing.assert_array_equal(read_records(a), read_records(b))


@pytest.mark.parametrize("change", [
    {"task": "diffusion"}, {"task": "rate"},
    {"potential": {"provider": "ase", "factory": "anything:calculator"}},
    {"potential": {"provider": "builtin", "name": "harmonic", "params": {"center": 1.}}},
    {"parallel": {"mode": "bead", "bead_workers": 2}},
    {"dynamics": {"loop_backend": "python"}},
    {"sampling": {"engine": "native", "thermostat": "trpmd"}},
])
def test_no_unsupported_native_workflow_fallback(change):
    c = config();c.update(change)
    with pytest.raises(InputError):
        JobSpec.from_mapping(c).resolve()


def test_non_native_workflow_cannot_silently_ignore_barostat():
    c = config(barostat="bzp")
    c["sampling"].pop("engine")
    with pytest.raises(InputError, match="barostat"):
        JobSpec.from_mapping(c).resolve()


def test_compiled_periodic_provider_preflight_and_native_workflow(tmp_path):
    import ctypes
    import shutil
    import subprocess
    from pyrpmd.potentials import make_potential, check_potential
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required")
    source = tmp_path/"volume.c"
    source.write_text(r'''
#include <stdint.h>
static int64_t calls=0;
int64_t get_calls(void){return calls;}
int32_t evaluate(void*ctx,int64_t P,int64_t N,const double*q,const double*h,
  const int32_t*pbc,double*e,double*f,double*v,char*err,int64_t cap){
    ++calls;
    double volume=h[0]*(h[4]*h[8]-h[5]*h[7])-h[1]*(h[3]*h[8]-h[5]*h[6])+h[2]*(h[3]*h[7]-h[4]*h[6]);
    for(int64_t b=0;b<P;++b){e[b]=0.001*volume;
        for(int64_t j=0;j<3*N;++j)f[b*3*N+j]=0;
        for(int j=0;j<9;++j)v[b*9+j]=(j%4==0)?-e[b]:0;
    }
    return 0;
}
''')
    library = tmp_path/"volume.so"
    subprocess.run([compiler, "-shared", "-O2", "-fPIC", str(source), "-o", str(library)], check=True)
    c = config(barostat="bzp")
    c["potential"] = {"provider": "native_periodic", "library": str(library), "function": "evaluate"}
    resolved, system = JobSpec.from_mapping(c).resolve()
    provider = make_potential(resolved["potential"], system)
    q = np.repeat(system.geometry[None], c["beads"], axis=0)
    result = provider.evaluate(q)
    np.testing.assert_array_equal(result.energy, np.full(c["beads"], .001))
    np.testing.assert_array_equal(result.virial, np.repeat((-.001*np.eye(3))[None], c["beads"], axis=0))
    counter = ctypes.CDLL(str(library));counter.get_calls.restype = ctypes.c_int64
    before = counter.get_calls()
    summary = run(c, output=tmp_path/"run")
    # One initial evaluation and one per timestep; Python never calls evaluate.
    assert counter.get_calls()-before == 38
    assert summary["python_step_callbacks"] == 0
    records = read_records(summary["files"]["journals"][0])
    np.testing.assert_allclose(records[:, 2], .001*records[:, 6], rtol=1e-13)


def test_fixed_atoms_work_only_in_native_nvt(tmp_path):
    c = config()
    c["system"]["fixed_atoms"] = [1]
    summary = run(c, output=tmp_path/"nvt")
    records = read_records(summary["files"]["journals"][0])
    # 18 scalars; first 6 centroid coordinates, then velocities, then radii.
    np.testing.assert_array_equal(records[:, 21:24], np.broadcast_to([.1, .2, .3], (len(records),3)))
    np.testing.assert_array_equal(records[:, 27:30], 0)
    np.testing.assert_array_equal(records[:, -1], 0)
    c = config(barostat="bzp");c["system"]["fixed_atoms"] = [1]
    with pytest.raises(InputError, match="all atoms mobile"):
        JobSpec.from_mapping(c).resolve()


def test_standard_native_workflow_calls_once_per_checkpoint_not_save(tmp_path, monkeypatch):
    calls = []
    original = NativeEnsemble.run
    def counted(self, steps, **kwargs):
        calls.append(steps)
        return original(self, steps, **kwargs)
    monkeypatch.setattr(NativeEnsemble, "run", counted)
    c = config(blocks=100000)
    c["sampling"].update(production_steps=1000, sample_interval=1)
    result = run(c, output=tmp_path/"native")
    assert calls == [7, 1000]
    assert result["samples_per_replica"] == [1000]


def test_sampling_type_error_precedes_native_dispatch():
    from pyrpmd.input import JobSpec, InputError
    with pytest.raises(InputError, match="sampling must be a table"):
        JobSpec({"task": "rate", "sampling": []}).resolve()
