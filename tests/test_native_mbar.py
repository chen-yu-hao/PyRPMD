"""Native reductions, bounded umbrella views, and parallel rate estimators."""
from concurrent.futures import ThreadPoolExecutor
import pickle

import numpy as np
import pytest

from pyrpmd.analysis.mbar import solve_mbar, _reweighting_signature
from pyrpmd.analysis.mbar_problem import ReducedPotentials, umbrella_problem
from pyrpmd.analysis.fes import mbar_rate_pmf, reconstruct_fes
from pyrpmd.analysis.umbrella import umbrella_reduced_potentials


def problem(dimension=1):
    rng=np.random.default_rng(724)
    centers=rng.normal(0,.4,(5,dimension));centers[3]=centers[0]
    counts=np.array([113,99,121,87,100])
    samples=np.concatenate([rng.normal(center,.5,(n,dimension)) for center,n in zip(centers,counts)])
    matrices=np.repeat(np.eye(dimension)[None],5,axis=0)*2
    if dimension>1:
        matrices[:,0,1]=matrices[:,1,0]=.25
    periods=[None]*dimension
    if dimension==3:
        periods[1]=2*np.pi
    return umbrella_problem(samples,centers,matrices,1.3,periods),counts


@pytest.mark.parametrize("dimension",[1,2,3])
@pytest.mark.parametrize("threads",[1,2,4])
def test_native_matches_dense_legacy_and_blocked_numpy(dimension,threads):
    pytest.importorskip("pyrpmd._mbar")
    u,counts=problem(dimension)
    target=np.linspace(-2,3,u.shape[1])
    reference=solve_mbar(np.asarray(u),counts,target,tolerance=1e-10,method="legacy")
    for backend in ("native","numpy"):
        result=solve_mbar(u,counts,target,tolerance=1e-10,backend=backend,threads=threads)
        assert result.converged and result.residual<1e-10
        for key in ("free_energies","weights","state_weights","overlap","log_weights"):
            np.testing.assert_allclose(getattr(result,key),getattr(reference,key),atol=3e-9,rtol=3e-8)
        np.testing.assert_allclose(result.overlap.sum(axis=1),1,atol=1e-9)
        if backend=="native":
            assert result.diagnostics["native_calls"]>0


def test_dense_offsets_disconnected_components_and_omitted_weights():
    pytest.importorskip("pyrpmd._mbar")
    x=np.r_[np.full(30,-10.),np.full(40,10.)]
    u=100*(x[None]-np.array([-10,10])[:,None])**2
    fit=solve_mbar(u,[30,40],return_state_weights=False,backend="native",threads=2)
    assert fit.converged and len(np.unique(fit.components))==2
    assert fit.state_weights is None
    shifted=solve_mbar(u+np.array([1e5,-1e5])[:,None],[30,40],
                       initial_free_energies=fit.free_energies+np.array([1e5,-1e5]),backend="native")
    np.testing.assert_allclose(shifted.weights,fit.weights,atol=1e-10)


def test_lazy_problem_never_materializes_matrix(monkeypatch):
    u,counts=problem(2)
    reference=umbrella_reduced_potentials(u.values,u.centers,u.stiffness,u.beta)
    assert _reweighting_signature(u,counts,1e-6)==_reweighting_signature(reference,counts,1e-6)
    def forbidden(*args,**kwargs):
        raise AssertionError("K x N matrix materialization")
    monkeypatch.setattr(ReducedPotentials,"__array__",forbidden)
    for backend in ("auto","numpy"):
        fit=solve_mbar(u,counts,backend=backend,return_state_weights=False)
        assert fit.converged
    rng=np.random.default_rng(1)
    indices=rng.integers(0,u.shape[1],u.shape[1])
    resampled=u[:,indices]
    assert np.shares_memory(resampled.values,u.values)
    assert np.shares_memory(resampled.centers,u.centers)
    fit=solve_mbar(pickle.loads(pickle.dumps(resampled)),counts,return_state_weights=False)
    assert fit.converged
    streams=np.split(u.values,np.cumsum(counts)[:-1])
    reconstructed=reconstruct_fes(streams,u.centers,u.stiffness,u.beta,bins=6,bootstrap=2,
                                  return_reweighting=True)
    assert reconstructed["connected"]


def test_native_threads_are_actually_used():
    native=pytest.importorskip("pyrpmd._mbar")
    if not native.OPENMP:
        pytest.skip("build without OpenMP")
    import os
    if hasattr(os,"sched_getaffinity") and len(os.sched_getaffinity(0))<2:
        pytest.skip("one CPU affinity")
    u,counts=problem()
    fit=solve_mbar(u,counts,threads=2,backend="native",return_state_weights=False)
    assert fit.diagnostics["threads_used"]==2


def test_external_executor_uses_compact_draws_and_matches_serial(monkeypatch):
    rng=np.random.default_rng(23)
    centers=[.2,.5,.8]
    samples=[rng.normal(c,.2,240) for c in centers]
    options=dict(centers=centers,stiffness=8.,beta=1.,bins=np.linspace(-.5,1.5,12),
                 reference=.2,xi_star=.8,bootstrap=6,seed=77,min_count=1,
                 min_weight_ess=None,min_endpoint_ess=None,max_center_displacement_sigma=None,
                 compare_ui=True,solver_options={"threads":1})
    serial=mbar_rate_pmf(samples,bootstrap_workers=1,**options)
    class CheckedExecutor:
        calls=0
        def map(self,fn,tasks):
            self.calls+=1
            assert len(tasks)<=2
            assert all(isinstance(t[1],dict) for t in tasks)
            with ThreadPoolExecutor(max_workers=2) as pool:
                return list(pool.map(fn,tasks))
    executor=CheckedExecutor()
    parallel=mbar_rate_pmf(samples,bootstrap_workers=2,bootstrap_executor=executor,**options)
    assert executor.calls==3
    np.testing.assert_array_equal(serial["pmf_bootstrap_barriers"],parallel["pmf_bootstrap_barriers"])
    assert serial["bootstrap"]["draws"]==parallel["bootstrap"]["draws"]
    assert parallel["bootstrap"]["backend"]=="executor"


def test_native_rejects_malformed_buffers_and_indices():
    native=pytest.importorskip("pyrpmd._mbar")
    u,counts=problem()
    args=[u.values,u.centers,u.stiffness,u.periods,u.rows,u.indices,
          np.zeros(5),counts.astype(float),np.zeros(u.shape[1]),u.beta,2,2]
    for position,bad in ((0,u.values.astype(np.float32)),(4,np.array([999]*5,dtype=np.int64)),
                         (5,np.array([999999]*u.shape[1],dtype=np.int64)),(6,np.zeros(6))):
        broken=args.copy();broken[position]=bad
        with pytest.raises(ValueError):native.reduce(*broken)


@pytest.mark.parametrize("kwargs",[{"threads":0},{"threads":True},{"threads":1.5},{"backend":"gpu"}])
def test_invalid_solver_execution_options(kwargs):
    with pytest.raises(ValueError):solve_mbar(np.zeros((1,2)),[2],**kwargs)


def test_missing_extension_falls_back_to_bounded_numpy(monkeypatch):
    import sys
    import pyrpmd
    monkeypatch.delattr(pyrpmd,"_mbar",raising=False)
    monkeypatch.setitem(sys.modules,"pyrpmd._mbar",None)
    u,counts=problem()
    result=solve_mbar(u,counts,return_state_weights=False)
    assert result.converged and result.diagnostics["backend"]=="numpy_blocked"
    with pytest.raises(RuntimeError,match="unavailable"):
        solve_mbar(u,counts,backend="native")


def test_grouped_fes_parallel_bootstrap_matches_serial():
    u,counts=problem(2)
    streams=np.split(u.values,np.cumsum(counts)[:-1])
    options=dict(centers=u.centers,stiffness=u.stiffness,beta=u.beta,bins=5,
                 bootstrap=4,seed=912)
    serial=reconstruct_fes(streams,bootstrap_workers=1,**options)
    parallel=reconstruct_fes(streams,bootstrap_workers=2,**options)
    for key in ("free_energy","standard_error","bootstrap_valid_per_bin"):
        np.testing.assert_array_equal(serial[key],parallel[key])
    assert serial["bootstrap_diagnostics"]==parallel["bootstrap_diagnostics"]


def test_existing_process_executor_can_deserialize_bootstrap_tasks():
    import subprocess
    import sys
    import textwrap
    script=textwrap.dedent("""
        import numpy as np
        from pyrpmd.analysis.fes import mbar_rate_pmf
        from pyrpmd.parallel import TaskExecutor
        rng=np.random.default_rng(53)
        centers=[.2,.5,.8]
        samples=[rng.normal(c,.2,200) for c in centers]
        options=dict(centers=centers,stiffness=8.,beta=1.,bins=np.linspace(-.5,1.5,12),
                     reference=.2,xi_star=.8,bootstrap=4,seed=42,min_count=1,
                     min_weight_ess=None,min_endpoint_ess=None,max_center_displacement_sigma=None)
        serial=mbar_rate_pmf(samples,bootstrap_workers=1,**options)
        executor=TaskExecutor({'backend':'process','trajectory_workers':2,'threads_per_rank':1})
        result=mbar_rate_pmf(samples,bootstrap_workers=2,bootstrap_executor=executor,**options)
        np.testing.assert_array_equal(serial['pmf_bootstrap_barriers'],result['pmf_bootstrap_barriers'])
        assert serial['bootstrap']['draws']==result['bootstrap']['draws']
        assert result['bootstrap']['backend']=='executor'
    """)
    subprocess.run([sys.executable,"-c",script],check=True,capture_output=True,text=True,timeout=60)
