"""Physics and failure-semantics checks for explicit RPC / r-RESPA APIs."""

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.multitime import MTSDynamics
from pyrpmd.potentials import FreePotential, HarmonicPotential, PESResult, PotentialError
from pyrpmd.potentials.contraction import DifferencePotential, RingPolymerContraction, SplitPotential


class QuarticPotential:
    def evaluate(self, q):
        return PESResult(np.sum(0.01 * q**4 + 0.03 * q**2, axis=(1, 2)), -0.04 * q**3 - 0.06 * q)


class CountingPotential:
    def __init__(self, potential):
        self.potential = potential
        self.calls = 0
        self.frames = 0

    def evaluate(self, q):
        self.calls += 1
        self.frames += len(q)
        return self.potential.evaluate(q)


@pytest.mark.parametrize("p,c", [(1, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 8), (9, 4), (9, 7), (9, 9)])
def test_contraction_is_normalized_projection(p, c):
    rpc = RingPolymerContraction(p, c)
    q = np.random.default_rng(51).normal(size=(p, 2, 3))
    contracted = rpc.contract(q)
    np.testing.assert_allclose(contracted.mean(axis=0), q.mean(axis=0), atol=5e-16)
    np.testing.assert_allclose(rpc.matrix @ rpc.matrix.T, (c / p) * np.eye(c), atol=2e-15)
    if c == p:
        np.testing.assert_array_equal(contracted, q)
        np.testing.assert_array_equal(rpc.lift_forces(q), q)
        assert not np.shares_memory(contracted, q)
    if c == 1:
        np.testing.assert_allclose(contracted[0], q.mean(axis=0), atol=5e-16)
        np.testing.assert_allclose(rpc.lift_forces(contracted), np.broadcast_to(contracted, q.shape))


@pytest.mark.parametrize("p,c", [(16, 3), (16, 5), (16, 4), (17, 8)])
def test_retained_fourier_pairs_and_removed_high_modes(p, c):
    rpc = RingPolymerContraction(p, c)
    for k in range(1, (c + 1) // 2):
        for trig in (np.cos, np.sin):
            q = np.zeros((p, 1, 3))
            q[:, 0, 0] = trig(2 * np.pi * k * np.arange(p) / p)
            expected = trig(2 * np.pi * k * np.arange(c) / c)
            np.testing.assert_allclose(rpc.contract(q)[:, 0, 0], expected, atol=3e-15)
    q = np.zeros((p, 1, 3))
    q[:, 0, 0] = np.cos(2 * np.pi * (c // 2 + 1) * np.arange(p) / p)
    np.testing.assert_allclose(rpc.contract(q), 0.0, atol=3e-15)


def test_even_cutoff_convention_uses_one_cosine_mode():
    p, c = 16, 4
    rpc = RingPolymerContraction(p, c)
    q = np.zeros((p, 1, 3))
    q[:, 0, 0] = np.cos(2 * np.pi * (c // 2) * np.arange(p) / p)
    np.testing.assert_allclose(rpc.contract(q)[:, 0, 0], (-1.0)**np.arange(c) / np.sqrt(2), atol=2e-15)
    q[:, 0, 0] = np.sin(2 * np.pi * (c // 2) * np.arange(p) / p)
    np.testing.assert_allclose(rpc.contract(q), 0.0, atol=2e-15)


@pytest.mark.parametrize("p,c", [(8, 1), (8, 2), (8, 3), (8, 4), (8, 8), (9, 5)])
def test_rpc_force_is_derivative_of_total_contracted_energy(p, c):
    split = SplitPotential(HarmonicPotential(0.2), QuarticPotential(), p, c)
    q = np.random.default_rng(82).normal(size=(p, 2, 3))
    result = split.evaluate(q)
    eps = 1e-5
    numerical = np.empty_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical[index] = -(split.evaluate(plus).energy.sum() - split.evaluate(minus).energy.sum()) / (2 * eps)
    np.testing.assert_allclose(result.forces, numerical, rtol=2e-7, atol=1e-9)


def test_energy_normalization_and_explicit_difference():
    class Constant:
        def evaluate(self, q):
            return PESResult(np.full(len(q), 2.5), np.zeros_like(q))

    q = np.random.default_rng(1).normal(size=(16, 1, 3))
    for c in (1, 2, 3, 8, 16):
        split = SplitPotential(FreePotential(), Constant(), 16, c)
        assert split.evaluate(q).energy.sum() == 40.0
    target, reference = HarmonicPotential(0.8), HarmonicPotential(0.2)
    split = SplitPotential(reference, DifferencePotential(target, reference), 16, 16)
    np.testing.assert_allclose(split.evaluate(q).energy, target.evaluate(q).energy, atol=2e-15)
    np.testing.assert_allclose(split.evaluate(q).forces, target.evaluate(q).forces, atol=2e-15)


def test_shared_provider_output_buffers_do_not_alias_component_results():
    energies, forces = np.zeros(5), np.zeros((5, 1, 3))

    class Borrowed:
        def __init__(self, k):
            self.k = k

        def evaluate(self, q):
            energies[:] = 0.5 * self.k * np.sum(q*q, axis=(1, 2))
            forces[:] = -self.k * q
            return PESResult(energies, forces)

    q = np.random.default_rng(7).normal(size=(5, 1, 3))
    reference, target = Borrowed(0.2), Borrowed(0.8)
    correction = DifferencePotential(target, reference)
    result = correction.evaluate(q)
    np.testing.assert_allclose(result.forces, -0.6 * q)
    split = SplitPotential(reference, correction, 5)
    result = split.evaluate(q)
    np.testing.assert_allclose(result.forces, -0.8 * q)
    target.evaluate(q + 3)
    np.testing.assert_allclose(result.forces, -0.8 * q)


def test_failed_contracted_frame_aborts_the_whole_ring():
    class Fail:
        def evaluate(self, q):
            status = np.zeros(len(q), dtype=int)
            status[-1] = 9
            return PESResult(np.zeros(len(q)), np.zeros_like(q), status)

    split = SplitPotential(FreePotential(), Fail(), 8, 3)
    with pytest.raises(PotentialError, match="status=9"):
        split.evaluate(np.ones((8, 1, 3)))


def test_rpc_input_guards_and_batch_metadata():
    for p, c in [(False, 1), (1, True), (2.0, 1), (2, 0), (2, 3)]:
        with pytest.raises(ValueError):
            RingPolymerContraction(p, c)
    rpc = RingPolymerContraction(8, 3)
    with pytest.raises(ValueError, match="complete ring"):
        rpc.contract(np.ones((16, 1, 3)))
    with pytest.raises(ValueError, match="contracted beads"):
        rpc.lift_forces(np.ones((4, 1, 3)))
    split = SplitPotential(FreePotential(), FreePotential(), 8, 3)
    assert split.metadata["ring_polymer_coupled"] is True
    assert split.metadata["trajectory_batch_safe"] is False
    with pytest.raises(ValueError, match="independent physical"):
        SplitPotential(split, FreePotential(), 8)


@pytest.mark.parametrize("scheme", ["normal-mode-bab", "cayley-bcb", "velocity-verlet"])
def test_mts_one_inner_step_equals_existing_dynamics(scheme):
    split = SplitPotential(HarmonicPotential(0.4), QuarticPotential(), 8)
    ordinary = Dynamics(split, [2, 3], 4, 8, 0.01, native=False, integrator=scheme)
    mts = MTSDynamics(split, [2, 3], 4, 8, 0.01, inner_steps=1, native=False, integrator=scheme)
    rng = np.random.default_rng(23)
    a = State(rng.normal(size=(8, 2, 3)), rng.normal(size=(8, 2, 3)))
    b = a.copy()
    for _ in range(40):
        ordinary.step(a)
    mts.propagate(b, 40)
    np.testing.assert_allclose(a.q, b.q, atol=3e-14)
    np.testing.assert_allclose(a.p, b.p, atol=3e-14)
    assert a.step == b.step == 40
    assert a.time == pytest.approx(b.time)


@pytest.mark.parametrize("c", [1, 3, 8])
@pytest.mark.parametrize("native", [False] + ([True] if native_capabilities()["available"] else []))
def test_mts_reversibility_and_energy(c, native):
    split = SplitPotential(HarmonicPotential(0.4), QuarticPotential(), 8, c)
    mts = MTSDynamics(split, [2, 3], 4, 8, 0.01, inner_steps=4, native=native)
    rng = np.random.default_rng(33)
    state = State(rng.normal(size=(8, 2, 3)), rng.normal(size=(8, 2, 3)))
    initial = state.copy()
    e0 = mts.energy(state)
    mts.propagate(state, 100)
    assert abs((mts.energy(state) - e0) / e0) < 5e-6
    state.p *= -1
    mts.propagate(state, 100)
    np.testing.assert_allclose(state.q, initial.q, atol=3e-12)
    np.testing.assert_allclose(state.p, -initial.p, atol=3e-12)


def test_mts_second_order_outer_timestep_convergence():
    rng = np.random.default_rng(49)
    original = State(rng.normal(size=(4, 1, 3)), rng.normal(size=(4, 1, 3)))
    split = SplitPotential(HarmonicPotential(0.7), QuarticPotential(), 4)
    reference = original.copy()
    MTSDynamics(split, [2], 5, 4, 0.0005, inner_steps=4, native=False).propagate(reference, 800)
    errors = []
    for dt, steps in [(0.04, 10), (0.02, 20), (0.01, 40)]:
        state = original.copy()
        MTSDynamics(split, [2], 5, 4, dt, inner_steps=4, native=False).propagate(state, steps)
        errors.append(np.linalg.norm(state.q - reference.q) + np.linalg.norm(state.p - reference.p))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_expensive_evaluation_reduction_and_no_stale_cache_between_blocks():
    fast = CountingPotential(HarmonicPotential(0.4))
    slow = CountingPotential(QuarticPotential())
    split = SplitPotential(fast, slow, 16, 3)
    mts = MTSDynamics(split, [2], 4, 16, 0.01, inner_steps=4, native=False)
    state = State(np.ones((16, 1, 3)), np.zeros((16, 1, 3)))
    mts.propagate(state, 10)
    assert (fast.calls, fast.frames) == (41, 41 * 16)
    assert (slow.calls, slow.frames) == (11, 11 * 3)
    assert mts.force_evaluations == {"fast": 41, "slow": 11}
    state.q += 0.2
    changed = state.copy()
    fresh = MTSDynamics(split, [2], 4, 16, 0.01, inner_steps=4, native=False)
    mts.step(state)
    fresh.step(changed)
    np.testing.assert_array_equal(state.q, changed.q)
    np.testing.assert_array_equal(state.p, changed.p)


@pytest.mark.parametrize("keep_prefix", [False, True])
def test_failed_outer_step_preserves_requested_transaction_boundary(keep_prefix):
    class FailOnThirdSlow:
        calls = 0

        def evaluate(self, q):
            self.calls += 1
            if self.calls == 3:
                raise PotentialError("failed second outer endpoint")
            return HarmonicPotential(0.2).evaluate(q)

    initial = State(np.ones((4, 1, 3)), np.full((4, 1, 3), 0.2), step=7, time=0.3)
    expected = initial.copy()
    reference = MTSDynamics(SplitPotential(HarmonicPotential(0.4), HarmonicPotential(0.2), 4, 1), [2], 3, 4, 0.02, inner_steps=3, native=False)
    if keep_prefix:
        reference.step(expected)
    state = initial.copy()
    mts = MTSDynamics(SplitPotential(HarmonicPotential(0.4), FailOnThirdSlow(), 4, 1), [2], 3, 4, 0.02, inner_steps=3, native=False)
    advance = mts.propagate_block if keep_prefix else mts.propagate
    with pytest.raises(PotentialError, match="second outer"):
        advance(state, 5)
    np.testing.assert_array_equal(state.q, expected.q)
    np.testing.assert_array_equal(state.p, expected.p)
    assert state.step == expected.step
    assert state.time == expected.time


def test_mts_shape_settings_and_state_guards():
    split = SplitPotential(HarmonicPotential(0.3), FreePotential(), 4)
    with pytest.raises(TypeError, match="SplitPotential"):
        MTSDynamics(HarmonicPotential(), [2], 3, 4, 0.01)
    with pytest.raises(ValueError, match="bead counts"):
        MTSDynamics(split, [2], 3, 8, 0.01)
    for inner in (False, 0, 1.5):
        with pytest.raises(ValueError, match="inner_steps"):
            MTSDynamics(split, [2], 3, 4, 0.01, inner_steps=inner)
    mts = MTSDynamics(split, [2], 3, 4, 0.01, inner_steps=2, native=False)
    state = State(np.ones((4, 1, 3)), np.zeros((4, 1, 3)))
    for steps in (-1, True, 0.5):
        with pytest.raises(ValueError, match="steps"):
            mts.propagate(state, steps)
    assert mts.propagate(state, 0) is state
    assert mts.force_evaluations == {"fast": 0, "slow": 0}
    mts.inner_steps = 3
    with pytest.raises(ValueError, match="immutable"):
        mts.step(state)


@pytest.mark.parametrize("field,value", [("step", True), ("step", -1), ("step", 1.5), ("time", np.nan), ("time", np.inf), ("time", 1j)])
def test_mutated_state_clock_is_rejected(field, value):
    split = SplitPotential(FreePotential(), FreePotential(), 1)
    mts = MTSDynamics(split, [1], 1, 1, 0.01, native=False)
    state = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    setattr(state, field, value)
    with pytest.raises(ValueError, match="state clock"):
        mts.step(state)


@pytest.mark.parametrize("keep_prefix", [False, True])
def test_time_overflow_preserves_transaction_boundary(keep_prefix):
    split = SplitPotential(FreePotential(), FreePotential(), 1)
    mts = MTSDynamics(split, [1], 1, 1, 1e308, native=False)
    state = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    advance = mts.propagate_block if keep_prefix else mts.propagate
    with pytest.raises(FloatingPointError, match="state clock"):
        advance(state, 2)
    assert state.step == int(keep_prefix)
    assert state.time == (1e308 if keep_prefix else 0.0)
    np.testing.assert_array_equal(state.q, 0)
    np.testing.assert_array_equal(state.p, 0)
