"""Complete native EF blocks preserve physical OU statistics and restart RNG."""
import copy
import cProfile
from pathlib import Path
import pstats
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd import _native
from pyrpmd.dynamics import INTEGRATORS, State
from pyrpmd.electronic_friction import ConstantElectronicFriction, ElectronicFrictionDynamics
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.potentials.cabi import CABIPotential
from pyrpmd.propagation import NativeUnavailableError

pytestmark = pytest.mark.skipif(getattr(_native, "CENTROID_FRICTION_API", 0) < 1,
                                reason="new centroid-friction native kernel must be built")
ROOT = Path(__file__).resolve().parents[1]


def driver(beads=4, rates=.3, native=True, integrator="cayley-bcb", fixed=False, potential=None):
    options = {"fixed_atoms": [1], "fixed_positions": np.zeros((2,3))} if fixed else {}
    return ElectronicFrictionDynamics(potential or HarmonicPotential(.02), [2., 3.], 4., beads, .03,
        ConstantElectronicFriction.from_rates(rates, [2., 3.]), seed=451,
        native=native, integrator=integrator, **options)


def initial(beads=4, fixed=False):
    rng = np.random.default_rng(92)
    state = State(rng.normal(scale=.1, size=(beads,2,3)), rng.normal(scale=.3, size=(beads,2,3)))
    if fixed:
        state.q[:,1] = 0
        state.p[:,1] = 0
    return state


@pytest.mark.parametrize("beads", [1,4,16])
@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("rates", [0., .3, [0., .5, .2, .7, .1, .4]])
def test_seeded_native_blocks_match_independent_python_steps(beads, integrator, rates):
    native, reference = driver(beads,rates,integrator=integrator), driver(beads,rates,native=False,integrator=integrator)
    a,b = initial(beads),initial(beads)
    profile = cProfile.Profile()
    with profile:
        block = native.propagate_block(a, 23, check_energy=True)
    for _ in range(23):
        reference.step(b)
    np.testing.assert_allclose(a.q,b.q,atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(a.p,b.p,atol=2e-13,rtol=2e-13)
    assert native.rng.bit_generator.state == reference.rng.bit_generator.state
    assert native.loop_stats == {"native_steps":23,"python_steps":0,"fallback_reasons":{}}
    assert block.backend == "native" and block.completed == 23
    callbacks = sum(row[1] for (filename,_,function),row in pstats.Stats(profile).stats.items()
                    if function in ("step","evaluate") and filename.endswith(("electronic_friction.py","builtin.py")))
    assert callbacks == 0


def test_chunking_and_prefinal_rollback_preserve_the_random_stream():
    whole, split = driver(fixed=True),driver(fixed=True)
    a,b = initial(fixed=True),initial(fixed=True)
    result = whole.propagate_block(a,17,record_previous=True)
    split.propagate_block(b,5)
    split.propagate_block(b,11)
    np.testing.assert_array_equal(result.previous_q,b.q)
    np.testing.assert_array_equal(result.previous_p,b.p)
    assert result.previous_rng_state == split.rng.bit_generator.state
    split.propagate_block(b,1)
    np.testing.assert_array_equal(a.q,b.q)
    np.testing.assert_array_equal(a.p,b.p)
    assert whole.rng.bit_generator.state == split.rng.bit_generator.state
    np.testing.assert_array_equal(a.p[:,1],0.)


def test_free_centroid_stationary_variance_is_mass_over_beta():
    native = driver(potential=FreePotential())
    state = State(np.zeros((4,2,3)),np.zeros((4,2,3)))
    values=[]
    mask = np.arange(2048)%8 == 0
    for _ in range(30):
        result=native.propagate_block(state,2048,observe=mask,record_energy=False)
        values.extend(result.centroid_momenta)
    variance=np.asarray(values)[1000:].var(axis=0)
    np.testing.assert_allclose(variance,np.broadcast_to(np.array([2.,3.])[:,None]/4.,(2,3)),rtol=.16)
    # No stochastic kick is applied to the internal modes.
    np.testing.assert_allclose(state.p-state.p.mean(axis=0),0.,atol=5e-11)


@pytest.fixture
def domain_library(tmp_path):
    compiler=shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    source=(ROOT/"native/examples/harmonic_pes.c").read_text().replace(
        'if(!isfinite(q))','if(!isfinite(q)||fabs(q)>.6)')
    c=tmp_path/"domain.c"; library=tmp_path/"domain.so"
    c.write_text(source)
    subprocess.run([compiler,"-shared","-fPIC","-O2","-std=c11","-I",str(ROOT/"native/include"),str(c),"-o",str(library)],check=True)
    return library


def test_failed_step_keeps_only_the_successful_noise_prefix(domain_library):
    provider=CABIPotential(domain_library)
    try:
        def make(native):
            return ElectronicFrictionDynamics(provider,[1.],1e6,1,.1,
                ConstantElectronicFriction.from_rates(.01,[1.]),native=native,seed=11)
        native,reference=make(True),make(False)
        a=State(np.zeros((1,1,3)),np.array([[[1.,0.,0.]]]))
        b=a.copy()
        result=native.propagate_block(a,20,raise_on_error=False,check_energy=True)
        count=0
        for _ in range(20):
            try:
                reference.step(b)
                count+=1
            except Exception:
                break
        assert result.error and 0<count==result.completed<20
        np.testing.assert_allclose(a.q,b.q,atol=1e-14)
        np.testing.assert_allclose(a.p,b.p,atol=1e-14)
        assert native.rng.bit_generator.state==reference.rng.bit_generator.state
    finally:
        provider.close()


@pytest.mark.parametrize("kind", ["tensor","override","subclass"])
def test_unsupported_friction_is_rejected_before_state_or_rng_changes(kind):
    native=driver()
    if kind=="tensor":
        tensor=np.eye(6)*.3;tensor[0,1]=tensor[1,0]=.1
        native=ElectronicFrictionDynamics(FreePotential(),[2.,3.],4.,4,.03,ConstantElectronicFriction(tensor),seed=451)
    elif kind=="override":
        native.friction.evaluate=lambda q: None
    else:
        class Custom(ElectronicFrictionDynamics):
            pass
        native=Custom(FreePotential(),[2.,3.],4.,4,.03,ConstantElectronicFriction.from_rates(.3,[2.,3.]))
    state=initial(); before=state.copy(); rng=copy.deepcopy(native.rng.bit_generator.state)
    assert not native.can_fuse()
    with pytest.raises(NativeUnavailableError):
        native.propagate_block(state,3)
    np.testing.assert_array_equal(state.q,before.q)
    np.testing.assert_array_equal(state.p,before.p)
    assert native.rng.bit_generator.state==rng


def test_diffusion_uses_full_native_blocks_and_restores_rng_after_observation_failure(tmp_path,monkeypatch):
    from pyrpmd.workflows import materials
    from pyrpmd.system import System
    system=System(("H",),[2.],[[0.,0.,0.]])
    config={"task":"diffusion","beta":4.,"beads":4,"dt_au":.03,"seed":617,
        "potential":{"provider":"free"},"dynamics":{"loop_backend":"native","block_steps":64},
        "electronic_friction":{"gamma":.3},
        "sampling":{"thermostat":"pile-l","gamma_centroid":.3,"equilibration_steps":4,
                    "production_steps":49,"sample_interval":7,"replicas":1},
        "diffusion":{"axes":[0],"bootstrap":0},"output":{"checkpoint_steps":17}}
    def task(where):
        return {"config":config,"system":system,"output":where,"replica":0}
    reference=materials._sample_replica(task(tmp_path/"full"))
    original=materials.quantum_observables
    def fail(state,*args,**kwargs):
        if state.step==11:
            raise RuntimeError("observation interrupted")
        return original(state,*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(materials,"quantum_observables",fail)
        with pytest.raises(RuntimeError,match="observation interrupted"):
            materials._sample_replica(task(tmp_path/"resume"))
    restored=materials._sample_replica(task(tmp_path/"resume"))
    for name in ("positions","velocities","observables","sample_steps"):
        np.testing.assert_array_equal(reference[name],restored[name])
    assert reference["rng"]==restored["rng"]
    assert reference["propagation"]==restored["propagation"]
    stats=restored["propagation"]
    assert stats["production"]=={"native_steps":49,"python_steps":0,"fallback_reasons":{}}
    assert stats["sampling"]["native_steps"]==4 and stats["sampling"]["python_steps"]==0


def test_fallback_counters_name_the_unsupported_friction_scope():
    tensor=np.eye(6)*.3;tensor[0,1]=tensor[1,0]=.1
    dynamic=ElectronicFrictionDynamics(FreePotential(),[2.,3.],4.,4,.03,ConstantElectronicFriction(tensor),seed=451)
    result=dynamic.propagate(initial(),3,record_centroids=False)
    assert result.backend=="python"
    assert "off-diagonal electronic friction" in result.fallback_reason
    assert dynamic.loop_stats["python_steps"]==3
    assert dynamic.loop_stats["native_steps"]==0
    assert dynamic.loop_stats["fallback_reasons"]=={result.fallback_reason:3}


@pytest.mark.parametrize("scope", ["constant","offdiag","python"])
def test_static_diffusion_plan_matches_actual_native_scope(scope):
    from pyrpmd.runner import propagation_backend_plan
    from pyrpmd.system import System
    system=System(("H",),[2.],[[0.,0.,0.]])
    config={"task":"diffusion","beta":4.,"beads":4,"dt_au":.03,
            "potential":{"provider":"free"}, "electronic_friction":{"gamma":.3},
            "sampling":{"production_steps":49,"equilibration_steps":4}}
    if scope=="offdiag":
        config["electronic_friction"]={"tensor":[[.4,.1,0.],[.1,.4,0.],[0.,0.,.4]]}
    elif scope=="python":
        config["dynamics"]={"loop_backend":"python"}
    before=copy.deepcopy(config)
    plan=propagation_backend_plan(config,system)
    production=next(stage for stage in plan if stage["stage"]=="production")
    assert config==before
    if scope=="constant":
        assert production["backend"]=="native" and production["reason"] is None
    else:
        assert production["backend"]=="python"
        reason="off-diagonal electronic friction" if scope=="offdiag" else "explicit Python loop backend"
        assert reason in production["reason"]
