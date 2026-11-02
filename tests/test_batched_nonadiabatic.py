import numpy as np
import pytest

from pyrpmd.batched_nonadiabatic import BatchedNRPMDState, BatchedNRPMDDynamics
from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import NRPMDState, NRPMDDynamics
from pyrpmd.potentials.nonadiabatic_models import LinearVibronicCoupling


def fixture(beads=4, count=5):
    potential=LinearVibronicCoupling(
        [[.1,.2j],[-.2j,-.1]], [[[.2,.03],[.03,-.2]]],harmonic=[.4])
    rng=np.random.default_rng(772)
    parents=[NRPMDState.from_nuclear(State(rng.normal(size=(beads,1,3))*.2,
                                          rng.normal(size=(beads,1,3))*.1),2,0,rng)
             for _ in range(count)]
    return potential,parents


@pytest.mark.parametrize("backend",["numpy","native"])
@pytest.mark.parametrize("beads",[1,2,5])
@pytest.mark.parametrize("scheme",["cayley-bcb","normal-mode-bab"])
def test_batch_matches_independent_scalar_trajectories(backend,beads,scheme):
    if backend=="native":pytest.importorskip("pyrpmd._electronic")
    potential,parents=fixture(beads)
    scalar=NRPMDDynamics(potential,[2.],3.,beads,.013,2,native=False,integrator=scheme)
    batch=BatchedNRPMDDynamics(potential,[2.],3.,beads,.013,2,native=False,
                              integrator=scheme,electronic_backend=backend)
    state=BatchedNRPMDState.from_states(parents)
    originals=[s.copy() for s in parents]
    for _ in range(9):
        batch.step(state)
        for s in parents:scalar.step(s)
    assert batch.pes_calls==18
    assert batch.pes_frames==18*len(parents)*beads
    for expected,actual in zip(parents,state.to_states()):
        for key in ("q","p","mapping_q","mapping_p"):
            np.testing.assert_allclose(getattr(actual,key),getattr(expected,key),atol=2e-13,rtol=2e-13)
        assert actual.step==expected.step
        assert actual.time==expected.time
    np.testing.assert_allclose(batch.energy(state),[scalar.energy(s) for s in parents],atol=2e-13)
    np.testing.assert_allclose(batch.populations(state),[scalar.populations(s) for s in parents],atol=2e-13)
    assert not any(np.shares_memory(state.q,s.q) for s in originals)


def test_failed_second_evaluation_rolls_back_entire_batch():
    potential,parents=fixture()
    class Failing:
        metadata={"trajectory_batch_safe":True}
        calls=0
        def evaluate(self,q):
            self.calls+=1
            if self.calls==2:raise ValueError("failed bead")
            return potential.evaluate(q)
    driver=BatchedNRPMDDynamics(Failing(),[2.],3.,4,.013,2,native=False)
    state=BatchedNRPMDState.from_states(parents)
    before=state.copy()
    with pytest.raises(ValueError,match="failed bead"):driver.step(state)
    for key in ("q","p","mapping_q","mapping_p","step","time"):
        np.testing.assert_array_equal(getattr(state,key),getattr(before,key))


def test_unknown_provider_requires_explicit_batch_contract():
    class Unknown:pass
    with pytest.raises(ValueError,match="trajectory_batch_safe"):
        BatchedNRPMDDynamics(Unknown(),[1.],2.,2,.1,2,native=False)


def test_single_trajectory_native_electronic_backend():
    pytest.importorskip("pyrpmd._electronic")
    potential,parents=fixture()
    a,b=parents[0].copy(),parents[0].copy()
    x=NRPMDDynamics(potential,[2.],3.,4,.013,2,native=False)
    y=NRPMDDynamics(potential,[2.],3.,4,.013,2,native=False,electronic_backend="native")
    for _ in range(15):x.step(a);y.step(b)
    for key in ("q","p","mapping_q","mapping_p"):
        np.testing.assert_allclose(getattr(a,key),getattr(b,key),atol=2e-13)


@pytest.mark.parametrize("clock,source", [("time", "q"), ("step", "p"), ("step", "time")])
def test_clock_alias_is_rejected_before_propagation(clock, source):
    potential, parents = fixture(count=2)
    driver = BatchedNRPMDDynamics(potential, [2.], 3., 4, .013, 2, native=False)
    state = BatchedNRPMDState.from_states(parents)
    storage = getattr(state, source).reshape(-1)[:2]
    storage[:] = 0
    setattr(state, clock, storage.view(np.int64) if clock == "step" else storage)
    before = state.q.copy(), state.p.copy()
    with pytest.raises(ValueError, match="must not alias"):
        driver.step(state)
    assert driver.pes_calls == 0
    np.testing.assert_array_equal(state.q, before[0])
    np.testing.assert_array_equal(state.p, before[1])
