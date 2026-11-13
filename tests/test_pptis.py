"""Local path measure, overlap swap balance, exact recurrence and restart tests."""

import copy
import numpy as np
import pytest

from pyrpmd.checkpoint import CheckpointError, atomic_savez, load_checkpoint
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, PESResult
from pyrpmd.pptis import (
    PPTISSampler, ReplicaExchangePPTIS, combine_pptis_rate, pptis_crossing_probability,
)
from pyrpmd.transition_interface import PathGenerationError, TISPath, TransitionInterfaceSampler


def make_sampler(cls=ReplicaExchangePPTIS, *, seed=7, nbeads=1, interfaces=(-0.5, 0.5),
                 potential=None, dt=0.1, max_path_steps=100, identity=None):
    potential = FreePotential() if potential is None else potential
    return cls(Dynamics(potential, [1.], 2., nbeads, dt, native=False), Cartesian(0),
               basin_a=-1., basin_b=1., interfaces=interfaces, max_path_steps=max_path_steps,
               seed=seed, identity=identity if identity is not None else
               {"potential": "free-unit-mass", "coordinate": "centroid-x0"})


def point(x=0., velocity=1., nbeads=1):
    q = np.zeros((nbeads, 1, 3)); p = np.zeros_like(q)
    q[:, 0, 0], p[:, 0, 0] = x, velocity
    return State(q, p)


def seeds(sampler):
    return [sampler.generate_path(point(x=x, nbeads=sampler.dynamics.nbeads), i)
            for i, x in enumerate(sampler.interfaces)]


@pytest.mark.parametrize("cls", [PPTISSampler, ReplicaExchangePPTIS])
@pytest.mark.parametrize("nbeads", [1, 4])
def test_partial_paths_have_local_endpoints_both_origins_and_replay(cls, nbeads):
    sampler = make_sampler(cls, nbeads=nbeads)
    for i, x in enumerate(sampler.interfaces):
        forward = sampler.generate_path(point(x=x, nbeads=nbeads), i)
        backward = sampler.generate_path(point(x=x, velocity=-1, nbeads=nbeads), i)
        sampler.validate_path(forward, i)
        sampler.validate_path(backward, i)
        assert sampler._record(forward, i) == (-1, 1)
        assert sampler._record(backward, i) == (1, -1)
        assert forward.q.shape[1:] == (nbeads, 1, 3)
        # A local path need not reach either physical stable basin.
        if i == 0:
            assert forward.values[-1] < sampler.basin_b


def test_local_shooting_preserves_canonical_length_ratio_and_time_reversal():
    sampler = make_sampler()
    slow, fast = point(x=-.5, velocity=.8), point(x=-.5, velocity=2.)
    old = sampler.generate_path(slow, 0)
    new = sampler.generate_path(fast, 0)
    ratio = sampler._shooting_log_ratio(old, new, slow, fast)
    assert ratio == pytest.approx(np.log(len(old) / len(new)), abs=1e-14)
    assert sampler._shooting_log_ratio(new, old, fast, slow) == pytest.approx(-ratio)
    sampler.validate_path(old.reversed(), 0)


def test_invalid_local_path_seeds_are_not_accepted():
    sampler = make_sampler()
    path = seeds(sampler)[0]
    q = path.q.copy(); q[len(q) // 2, 0, 0, 1] = .2
    with pytest.raises(ValueError, match="not a trajectory"):
        sampler.validate_path(TISPath(q, path.p, path.values, path.dt), 0)
    with pytest.raises(ValueError, match="local outer"):
        sampler.generate_path(point(x=.8), 0)
    with pytest.raises(ValueError, match="strict|centers"):
        make_sampler(interfaces=(-1., .5))


def test_missing_middle_crossing_is_rejected_even_with_valid_endpoints():
    sampler = make_sampler()
    values = np.array([-1.1, -.8, -1.1])
    q = np.zeros((3, 1, 1, 3)); q[:, 0, 0, 0] = values
    path = TISPath(q, np.zeros_like(q), values, .1)
    assert not sampler._member(path, 0)
    values = np.array([.6, .2, .6])
    q[:, 0, 0, 0] = values
    assert not sampler._member(TISPath(q, np.zeros_like(q), values, .1), 0)


def test_local_rejected_moves_and_callback_keep_path_multiplicity():
    sampler = make_sampler(PPTISSampler, max_path_steps=25, seed=22)
    path = sampler.generate_path(point(x=-.5, velocity=2.), 0)
    retained = []
    result = sampler.sample(path, 0, n_samples=60,
                            on_sample=lambda i, p, m: retained.append((p, m)))
    for current, move in retained:
        if not move.accepted:
            assert current is path
        path = current
    assert len(retained) == 60
    assert result.length_rejections > 0
    assert not result.accepted.all()


def test_overlap_swap_direction_combinations_and_inverse_are_balanced():
    sampler = make_sampler()
    initial = seeds(sampler)
    supported = []
    for directions in [(False, False), (False, True), (True, False), (True, True)]:
        try:
            proposed = sampler._swap_trial(*initial, 0, directions)
        except PathGenerationError as error:
            assert error.reason == "incompatible_swap_direction"
            continue
        supported.append(directions)
        reverse_directions = (not directions[1], not directions[0])
        back = sampler._swap_trial(*proposed, 0, reverse_directions)
        for i, (expected, actual) in enumerate(zip(initial, back)):
            np.testing.assert_allclose(actual.q, expected.q, atol=2e-15)
            np.testing.assert_allclose(actual.p, expected.p, atol=2e-15)
            sampler.validate_path(proposed[i], i)
        assert sampler._swap_log_ratio(initial, proposed) == pytest.approx(0)
        assert sampler._swap_log_ratio(proposed, initial) == pytest.approx(0)
    # For two LMR trajectories only one of four equiprobable choices works.
    # Choosing a direction conditional on the path type would accept every time.
    assert supported == [(False, True)]
    draws = [sampler.exchange_paths(initial.copy(), 0, 1) for _ in range(1200)]
    assert np.mean(draws) == pytest.approx(.25, abs=.04)


def test_swap_uses_overlap_and_regrowth_with_no_uniform_shooting_length_factor():
    sampler = make_sampler()
    a = sampler.generate_path(point(x=-.5, velocity=2.), 0)
    b = sampler.generate_path(point(x=.5, velocity=.6), 1)
    new = sampler._swap_trial(a, b, 0, (False, True))
    assert new[0] is not b and new[1] is not a
    for i, p in enumerate(new):
        sampler.validate_path(p, i)
    expected = -sampler.dynamics.beta * (
        sum(sampler.dynamics.energy(p.state(0)) for p in new)
        - sum(sampler.dynamics.energy(p.state(0)) for p in (a, b)))
    assert sampler._swap_log_ratio((a, b), new) == pytest.approx(expected, abs=1e-14)
    assert sampler._swap_log_ratio(new, (a, b)) == pytest.approx(-expected, abs=1e-14)


def test_failed_regrowth_keeps_both_paths():
    sampler = make_sampler(max_path_steps=20)
    initial = [sampler.generate_path(point(x=x, velocity=2.), i)
               for i, x in enumerate(sampler.interfaces)]
    with pytest.raises(ValueError, match="adjacent"):
        sampler.exchange_paths(initial, 1, 0)
    for _ in range(20):
        old = initial.copy()
        accepted = sampler.exchange_paths(initial, 0, 1)
        if not accepted:
            assert initial[0] is old[0] and initial[1] is old[1]


def independent_hitting_probability(lr, rl):
    """Solve the direction-aware absorbing Markov chain, independent of recursion."""
    n = len(lr)
    transition = np.zeros((2 * n, 2 * n))
    product = np.zeros(2 * n)
    for i in range(n):
        for direction, right_probability in enumerate((lr[i], 1 - rl[i])):
            row = 2 * i + direction
            if i == n - 1:
                product[row] += right_probability
            else:
                transition[row, 2 * (i + 1)] += right_probability
            if i:
                transition[row, 2 * (i - 1) + 1] += 1 - right_probability
    return np.linalg.solve(np.eye(2 * n) - transition, product)[0]


def test_pptis_recursion_matches_exact_directional_first_passage_solution():
    rng = np.random.default_rng(1741)
    for n in range(1, 8):
        lr, rl = rng.uniform(.1, .9, size=(2, n))
        actual = pptis_crossing_probability(lr, rl)
        assert actual == pytest.approx(independent_hitting_probability(lr, rl), rel=2e-14)
    assert pptis_crossing_probability([.5] * 5, [.5] * 5) == pytest.approx(1 / 6)
    assert pptis_crossing_probability([.5] * 5, [.5] * 5) != pytest.approx(.5 ** 5)


@pytest.mark.parametrize("lr,rl", [([], []), ([.3], [.2, .4]), ([True], [.2]),
    ([np.nan], [.5]), ([-.1], [.3]), ([1.1], [.3]), ([.3], [np.inf])])
def test_recursion_rejects_invalid_probabilities(lr, rl):
    with pytest.raises(ValueError):
        pptis_crossing_probability(lr, rl)


def test_singular_closed_local_chain_is_not_regularized():
    with pytest.raises(ValueError, match="singular"):
        pptis_crossing_probability([1., 0.], [0., 1.])


@pytest.mark.parametrize("cls", [PPTISSampler, ReplicaExchangePPTIS])
def test_checkpoint_resume_is_exact_for_rng_paths_observations_and_labels(tmp_path, cls):
    full = make_sampler(cls, seed=214)
    initial = seeds(full)
    expected = full.sample_interfaces(initial, n_samples=55, burn_in=7)
    split = make_sampler(cls, seed=214)
    first = split.sample_interfaces(initial, n_samples=23, burn_in=7)
    filename = tmp_path / "partial.npz"
    split.save_checkpoint(filename, first)
    continued = make_sampler(cls, seed=987)
    loaded = continued.load_checkpoint(filename)
    actual = continued.sample_interfaces(n_samples=32, resume=loaded)
    for a, b in zip(actual.interfaces, expected.interfaces):
        for field in ("starts", "ends", "accepted", "path_frames"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        np.testing.assert_array_equal(a.final_path.q, b.final_path.q)
        np.testing.assert_array_equal(a.final_path.p, b.final_path.p)
        assert a.reasons == b.reasons
        assert a.length_rejections == b.length_rejections
    for field in ("walker_labels", "swap_attempts", "swap_accepts", "swap_length_rejections"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.rng_state == expected.rng_state
    assert actual.burn_in == expected.burn_in
    assert actual.summary(block_size=11) == expected.summary(block_size=11)


def test_checkpoint_identity_and_payload_damage_are_rejected_without_rng_mutation(tmp_path):
    sampler = make_sampler()
    result = sampler.sample_interfaces(seeds(sampler), n_samples=10)
    filename = tmp_path / "chain.npz"
    sampler.save_checkpoint(filename, result)
    with pytest.raises(CheckpointError, match="identity"):
        make_sampler(dt=.2).load_checkpoint(filename)
    with pytest.raises(CheckpointError, match="identity"):
        make_sampler(identity={"potential": "different", "coordinate": "centroid-x0"}).load_checkpoint(filename)
    with load_checkpoint(filename) as data:
        payload = {key: data[key].copy() for key in data.files}
    payload["starts_0"][0] = 0
    atomic_savez(filename, kind="PyRPMD.pptis.v1", **payload)
    before = copy.deepcopy(sampler.rng.bit_generator.state)
    with pytest.raises(CheckpointError, match="endpoint history"):
        sampler.load_checkpoint(filename)
    assert sampler.rng.bit_generator.state == before


def test_persistent_checkpoint_requires_explicit_opaque_identity(tmp_path):
    sampler = ReplicaExchangePPTIS(Dynamics(FreePotential(), [1.], 2., 1, .1, native=False),
        Cartesian(0), basin_a=-1., basin_b=1., interfaces=[-.5, .5])
    result = sampler.sample_interfaces(seeds(sampler), n_samples=3)
    with pytest.raises(ValueError, match="explicit"):
        sampler.save_checkpoint(tmp_path / "anonymous.npz", result)


@pytest.mark.parametrize("field,value", [("max_path_steps", 200), ("basin_a", -2.),
    ("basin_b", 2.), ("interfaces", np.array([-.4, .5]))])
def test_changed_sampler_configuration_cannot_reuse_stale_identity(field, value, tmp_path):
    sampler = make_sampler()
    result = sampler.sample_interfaces(seeds(sampler), n_samples=3)
    setattr(sampler, field, value)
    with pytest.raises(ValueError, match="configuration changed"):
        sampler.save_checkpoint(tmp_path / "changed.npz", result)


def test_overlap_regrowth_length_rejections_are_reported_separately():
    sampler = ReplicaExchangePPTIS(Dynamics(FreePotential(), [1.], 2., 1, .1, native=False),
        Cartesian(0), basin_a=-2., basin_b=.6, interfaces=[-1., .2], max_path_steps=20, seed=32)
    initial = [sampler.generate_path(point(x=-1., velocity=2.), 0),
               sampler.generate_path(point(x=.2, velocity=1.), 1)]
    with pytest.raises(PathGenerationError, match="max_path_steps"):
        sampler._swap_trial(*initial, 0, (False, True))
    reasons = []
    for _ in range(40):
        current = initial.copy()
        assert not sampler.exchange_paths(current, 0, 1)
        assert current[0] is initial[0] and current[1] is initial[1]
        reasons.append(sampler._last_swap_reason)
    assert "max_path_steps" in reasons
    result = sampler.sample_interfaces(initial, n_samples=60)
    assert result.summary()["swap_length_rejections"] == result.swap_length_rejections.tolist()
    assert np.all(result.swap_length_rejections <= result.swap_attempts - result.swap_accepts)


def test_joint_block_error_does_not_discard_blocks_missing_origin_support():
    sampler = make_sampler()
    result = sampler.sample_interfaces(seeds(sampler), n_samples=20)
    summary = result.summary(block_size=1)
    assert summary["supported_blocks"] == 0
    assert summary["crossing_probability_standard_error_block"] is None
    assert summary["memory_loss_approximation"]
    assert summary["requires_independent_flux"]


def test_pptis_flux_matches_first_center_and_does_not_double_count_entrance():
    sampler = make_sampler()
    result = sampler.sample_interfaces(seeds(sampler), n_samples=40)
    flux = {"basin_a": -1., "basin_b": 1., "first_interface": -.5,
            "dt_au": .1, "flux_au": .2}
    rate = combine_pptis_rate(result, flux, block_size=10)
    assert rate["rate_estimate_au"] == pytest.approx(.2)
    assert rate["rate_s_inv"] is None and rate["converged"] is False
    for key, value in [("first_interface", -1.), ("dt_au", .2), ("basin_a", -2.)]:
        with pytest.raises(ValueError, match="flux must match"):
            combine_pptis_rate(result, {**flux, key: value})


class SeparableDoubleWell:
    """Quartic x double well plus harmonic y/z, with exactly separable flux."""

    def evaluate(self, q):
        energy = np.sum((q[:, :, 0] ** 2 - 1) ** 2, axis=1)
        energy += .5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * q[:, :, 0] * (q[:, :, 0] ** 2 - 1)
        return PESResult(energy, force)


def test_repptis_double_well_rate_matches_exact_flux_and_full_tis():
    import time
    from scipy.integrate import quad
    start_time = time.perf_counter()
    beta = 1.
    potential = SeparableDoubleWell()
    dynamics = Dynamics(potential, [1.], beta, 1, .04, native=False)
    centers = [-.8, -.4]
    sampler = ReplicaExchangePPTIS(dynamics, Cartesian(0), basin_a=-1., basin_b=1.,
        interfaces=centers, max_path_steps=500, seed=325,
        identity={"potential": "separable-double-well-v1", "coordinate": "x0"})
    initial = [sampler.generate_path(point(x=x, velocity=1.5), i) for i, x in enumerate(centers)]
    result = sampler.sample_interfaces(initial, n_samples=1800, burn_in=200)
    partition = quad(lambda x: np.exp(-beta * (x*x-1)**2), -np.inf, np.inf)[0]
    exact_flux = 2 * np.exp(-beta * (centers[0]**2-1)**2) / (np.sqrt(2*np.pi*beta) * partition)
    exact_rate = 2 * np.exp(-beta) / (np.sqrt(2*np.pi*beta) * partition)
    flux = {"basin_a": -1., "basin_b": 1., "first_interface": centers[0],
            "dt_au": dynamics.dt, "flux_au": exact_flux}
    estimate = combine_pptis_rate(result, flux, block_size=100)
    # All centers are uphill of A. Local R-to-L probabilities are one and
    # conserved energy carries no additional relevant memory at these windows.
    for local in result.interfaces:
        assert local.summary()["p_RL"] == pytest.approx(1.)
    error = estimate["rate_standard_error_block_conditional_on_flux_au"]
    assert error is not None and error > 0
    assert abs(estimate["rate_estimate_au"] - exact_rate) < 4 * error
    assert result.swap_accepts[0] > 0 and result.swap_accepts[0] < result.swap_attempts[0]
    assert all(local.length_rejections == 0 for local in result.interfaces)
    # Independent full-TIS path chain gives the A-origin conditional reference.
    tis = TransitionInterfaceSampler(dynamics, Cartesian(0), basin_a=-1., basin_b=1.,
                                    interfaces=[centers[0]], max_path_steps=500, seed=109)
    tis_result = tis.sample(tis.generate_path(point(x=-.6, velocity=1.5)), n_samples=1200, burn_in=150)
    tis_rate = exact_flux * tis_result.summary()["crossing_probability"]
    tis_error = exact_flux * tis_result.summary()["standard_error_diagnostic"]
    assert abs(tis_rate - estimate["rate_estimate_au"]) < 4 * np.hypot(error, tis_error)
    print("REPPTIS_REFERENCE " + repr({
        "exact_rate_au": exact_rate, "flux_au": exact_flux,
        "repptis_rate_au": estimate["rate_estimate_au"], "repptis_block_se_au": error,
        "tis_rate_au": tis_rate, "tis_diagnostic_se_au": tis_error,
        "repptis_samples": 1800, "tis_samples": 1200,
        "swap_attempts": result.swap_attempts.tolist(), "swap_accepts": result.swap_accepts.tolist(),
        "swap_length_rejections": result.swap_length_rejections.tolist(),
        "elapsed_seconds": time.perf_counter() - start_time, "threads": 1,
    }))
