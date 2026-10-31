"""Physical/convention checks for canonical transition-interface path sampling."""

import numpy as np
import pytest

from pyrpmd.constants import ATOMIC_TIME_S
from pyrpmd.cv import Cartesian
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, PESResult
from pyrpmd.transition_interface import (
    PathGenerationError, TISPath, TransitionInterfaceSampler,
    combine_tis_rate, estimate_effective_flux,
)


class DoubleWell:
    """U(x,y,z) = (x*x-1)**2 + (y*y+z*z)/2; unit masses."""

    def evaluate(self, q):
        energy = np.sum((q[:, :, 0] ** 2 - 1) ** 2, axis=1)
        energy += 0.5 * np.sum(q[:, :, 1:] ** 2, axis=(1, 2))
        force = -q.copy()
        force[:, :, 0] = -4 * q[:, :, 0] * (q[:, :, 0] ** 2 - 1)
        return PESResult(energy, force)


def sampler(*, seed=7, nbeads=1, max_path_steps=100, potential=None, dt=0.1, interfaces=(-0.5, 0.5)):
    dyn = Dynamics(FreePotential() if potential is None else potential, [1.0], 2.0, nbeads, dt, native=False)
    return TransitionInterfaceSampler(
        dyn, Cartesian(0), basin_a=-1, basin_b=1, interfaces=interfaces,
        max_path_steps=max_path_steps, seed=seed,
    )


def shooting(nbeads=1, velocity=1.0, x=0.0):
    q, p = np.zeros((nbeads, 1, 3)), np.zeros((nbeads, 1, 3))
    q[:, 0, 0], p[:, 0, 0] = x, velocity
    return State(q, p, step=23, time=17.0)


def test_seed_is_reversible_dynamics_and_owns_snapshots():
    tis = sampler(nbeads=4)
    state = shooting(4)
    before = state.copy()
    path = tis.generate_path(state, 1)
    assert path.q.shape[1:] == (4, 1, 3)
    assert path.values[0] <= -1 and path.values[-1] >= 1
    assert np.all(np.abs(path.values[1:-1]) < 1)
    tis.validate_path(path, 1)
    np.testing.assert_array_equal(state.q, before.q)
    np.testing.assert_array_equal(state.p, before.p)
    assert state.step == 23 and state.time == 17
    with pytest.raises(ValueError):
        path.q[0, 0, 0, 0] = 3
    copied = path.state(0)
    copied.q[:] = 42
    assert path.q[0, 0, 0, 0] < 0
    reverse = path.reversed()
    np.testing.assert_array_equal(reverse.p[0], -path.p[-1])


def test_arbitrary_scalar_coordinate_needs_no_derivative():
    dyn = sampler().dynamics
    tis = TransitionInterfaceSampler(
        dyn, lambda q: 3 * np.mean(q[:, 0, 0]), basin_a=-3, basin_b=3,
        interfaces=(-1.5, 1.5), seed=8,
    )
    path = tis.generate_path(shooting(), 1)
    np.testing.assert_allclose(path.values, 3 * path.q[:, :, 0, 0].mean(axis=1))


def test_coordinate_cannot_accidentally_modify_the_dynamics_state():
    def mutating(q):
        q[:] = 1
        return 0.0
    tis = TransitionInterfaceSampler(
        sampler().dynamics, mutating, basin_a=-1, basin_b=1, interfaces=[0],
    )
    initial = shooting()
    with pytest.raises(ValueError, match="read-only"):
        tis.generate_path(initial)
    assert np.all(initial.q == 0)


def test_generated_path_failures_and_numerical_errors_are_distinct():
    tis = sampler(max_path_steps=10)
    with pytest.raises(PathGenerationError, match="max_path_steps"):
        tis.generate_path(shooting())
    with pytest.raises(ValueError, match="outside both basins"):
        tis.generate_path(shooting(x=-2))
    with pytest.raises(PathGenerationError, match="backward_not_A"):
        sampler().generate_path(shooting(velocity=-1))

    class FailingPotential:
        def evaluate(self, q):
            raise ArithmeticError("PES failed")

    bad = sampler(potential=FailingPotential())
    with pytest.raises(ArithmeticError, match="PES failed"):
        bad.generate_path(shooting())


def test_invalid_or_fabricated_seed_is_rejected():
    tis = sampler()
    path = tis.generate_path(shooting())
    q = path.q.copy()
    q[len(q) // 2, 0, 0, 1] = 0.3  # CV is unchanged, but dynamics is broken.
    broken = TISPath(q, path.p, path.values, path.dt)
    with pytest.raises(ValueError, match="not a trajectory"):
        tis.sample(broken, n_samples=1)
    values = path.values.copy()
    values[len(values) // 2] = 0.2
    with pytest.raises(ValueError, match="order parameter"):
        tis.validate_path(TISPath(path.q, path.p, values, path.dt))
    with pytest.raises(ValueError, match="shape/timestep"):
        tis.validate_path(TISPath(path.q, path.p, path.values, 2 * path.dt))


def test_shooting_acceptance_uses_selectable_frame_count():
    tis = sampler()
    old_shoot, new_shoot = shooting(velocity=2), shooting(velocity=1)
    old, new = tis.generate_path(old_shoot), tis.generate_path(new_shoot)
    log_ratio = tis._shooting_log_ratio(old, new, old_shoot, new_shoot)
    assert log_ratio == pytest.approx(np.log(len(old) / len(new)), abs=1e-14)
    reverse_ratio = tis._shooting_log_ratio(new, old, new_shoot, old_shoot)
    assert log_ratio + reverse_ratio == pytest.approx(0, abs=1e-14)
    assert abs(log_ratio - np.log(old.interior_frames / new.interior_frames)) > 0.01


def test_endpoint_canonical_correction_matches_proposal_density():
    tis = sampler(potential=DoubleWell(), dt=0.08, max_path_steps=200)
    old_shoot, new_shoot = shooting(velocity=1.1, x=-0.1), shooting(velocity=1.7, x=-0.1)
    old, new = tis.generate_path(old_shoot), tis.generate_path(new_shoot)
    dyn = tis.dynamics
    beta_p = dyn.beta / dyn.nbeads
    def kinetic(state):
        return np.sum(state.p ** 2 / (2 * dyn.masses[None, :, None]))
    target_ratio = -beta_p * (dyn.energy(new.state(0)) - dyn.energy(old.state(0)))
    proposal_ratio = -beta_p * (kinetic(old_shoot) - kinetic(new_shoot))
    expected = target_ratio + proposal_ratio + np.log(len(old) / len(new))
    assert tis._shooting_log_ratio(old, new, old_shoot, new_shoot) == pytest.approx(expected)


def test_rejections_are_repeated_in_chain_and_callback_and_seeded_replay():
    def run():
        tis = sampler(seed=35, max_path_steps=45)
        initial = tis.generate_path(shooting(velocity=2))
        retained = []
        result = tis.sample(initial, n_samples=80, on_sample=lambda i, p, m: retained.append((p, m)))
        previous = initial
        for path, move in retained:
            if not move.accepted:
                assert path is previous
            previous = path
        assert len(retained) == len(result.crossing) == 80
        assert not result.accepted.all()
        assert result.length_rejections > 0
        return result
    first, second = run(), run()
    np.testing.assert_array_equal(first.crossing, second.crossing)
    np.testing.assert_array_equal(first.accepted, second.accepted)
    np.testing.assert_array_equal(first.path_frames, second.path_frames)
    np.testing.assert_array_equal(first.final_path.q, second.final_path.q)
    assert first.rng_state == second.rng_state
    summary = first.summary()
    assert summary["constant_crossing_indicator"]
    assert summary["standard_error_diagnostic"] is None
    assert summary["converged"] is False


def test_resume_with_saved_rng_and_final_path_is_exact():
    full = sampler(seed=42)
    path = full.generate_path(shooting())
    expected = full.sample(path, n_samples=70)
    split = sampler(seed=42)
    first = split.sample(path, n_samples=30)
    continued = sampler(seed=0)
    continued.rng.bit_generator.state = first.rng_state
    second = continued.sample(first.final_path, n_samples=40)
    np.testing.assert_array_equal(np.r_[first.accepted, second.accepted], expected.accepted)
    np.testing.assert_array_equal(np.r_[first.path_frames, second.path_frames], expected.path_frames)
    np.testing.assert_array_equal(second.final_path.q, expected.final_path.q)


def test_ring_polymer_maxwell_variance_and_fixed_atoms():
    dyn = Dynamics(
        FreePotential(), [2.0, 5.0], 3, 4, 0.1, native=False,
        fixed_atoms=(1,), fixed_positions=np.zeros((2, 3)),
    )
    tis = TransitionInterfaceSampler(dyn, Cartesian(0), basin_a=-1, basin_b=1, interfaces=[0], seed=221)
    momenta = np.array([tis._momenta() for _ in range(4000)])
    assert np.var(momenta[:, :, 0]) == pytest.approx(2 * 4 / 3, rel=0.025)
    assert np.all(momenta[:, :, 1] == 0)
    assert np.var(momenta[:, :, 0].mean(axis=1)) == pytest.approx(2 / 3, rel=0.04)


def test_two_frame_direct_crossings_are_in_the_path_ensemble():
    tis = sampler(dt=1.0, max_path_steps=1, interfaces=(0,))
    start, end = shooting(x=-1.1, velocity=3), shooting(x=1.9, velocity=3)
    direct = TISPath([start.q, end.q], [start.p, end.p], [-1.1, 1.9], 1.0)
    tis.validate_path(direct)
    from_start = tis._generate_path(start, 0, "start")
    from_end = tis._generate_path(end, 0, "end")
    np.testing.assert_allclose(from_start.q, direct.q)
    np.testing.assert_allclose(from_end.q, direct.q)
    assert from_start.interior_frames == 0
    # All attempts are recorded even if a slower Maxwell draw cannot cross
    # the entire transition region in this deliberately coarse one-step limit.
    result = tis.sample(direct, n_samples=12)
    assert np.all(result.crossing) and len(result.crossing) == 12
    flux = estimate_effective_flux(
        direct.values, basin_a=-1, basin_b=1, first_interface=0, dt=1.0,
    )
    assert flux["effective_crossings"] == 1
    assert combine_tis_rate([result], flux)["rate_estimate_au"] == pytest.approx(1.0)
    longer_sampler = sampler(dt=1.0, max_path_steps=10, interfaces=(0,))
    slower = shooting(x=-1.1, velocity=0.4)
    longer = longer_sampler._generate_path(slower, 0, "start")
    assert longer_sampler._shooting_log_ratio(longer, direct, slower, start) == pytest.approx(
        np.log(len(longer) / 2), abs=1e-14,
    )


def test_guards_reject_changed_overridden_or_stateful_dynamics():
    tis = sampler()
    tis.dynamics.potential = FreePotential()
    with pytest.raises(ValueError, match="changed"):
        tis.generate_path(shooting())
    dyn = sampler().dynamics
    dyn.step = dyn.sample_step
    with pytest.raises(TypeError, match="overridden/thermostatted"):
        TransitionInterfaceSampler(dyn, Cartesian(0), basin_a=-1, basin_b=1, interfaces=[0])
    potential = FreePotential()
    potential.history_dependent = True
    with pytest.raises(ValueError, match="time-independent"):
        sampler(potential=potential)


@pytest.mark.parametrize("interfaces", [[], [0, 0], [0.5, -0.5], [-2], [1], [np.nan], [True]])
def test_invalid_interfaces(interfaces):
    with pytest.raises(ValueError):
        sampler(interfaces=interfaces)


def test_flux_counts_first_crossings_and_history_population():
    trace = [-2, -0.5, 0.5, -0.2, 0.7, -2, -0.4, 0.3, 2, 0.2, -2, 0.2]
    flux = estimate_effective_flux(trace, basin_a=-1, basin_b=1, first_interface=0, dt=0.2)
    assert flux["effective_crossings"] == 3
    assert flux["a_history_time_au"] == pytest.approx(9 * 0.2)
    assert flux["flux_au"] == pytest.approx(3 / (9 * 0.2))
    assert flux["flux_s_inv"] == pytest.approx(flux["flux_au"] / ATOMIC_TIME_S)
    # Unknown initial history is excluded, not guessed to have come from A.
    unknown = estimate_effective_flux([0.2, 0.4, -2, 0.2], basin_a=-1, basin_b=1, first_interface=0, dt=0.2)
    assert unknown["effective_crossings"] == 1
    assert unknown["discarded_unknown_history_time_au"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="no known A-history"):
        estimate_effective_flux([0, 2, 0], basin_a=-1, basin_b=1, first_interface=0, dt=1)


def test_product_of_conditionals_requires_matching_complete_ensembles():
    tis = sampler()
    path = tis.generate_path(shooting(), 1)
    results = tis.sample_interfaces([path, path], n_samples=10)
    flux = estimate_effective_flux([-2, 0, 2], basin_a=-1, basin_b=1, first_interface=-0.5, dt=0.1)
    rate = combine_tis_rate(results, flux)
    assert rate["rate_estimate_au"] == pytest.approx(flux["flux_au"])
    assert rate["rate_s_inv"] is None and rate["converged"] is False
    assert rate["max_path_steps"] == tis.max_path_steps
    with pytest.raises(ValueError, match="every adjacent"):
        combine_tis_rate(results[:1], flux)
    with pytest.raises(ValueError, match="same basins"):
        combine_tis_rate(results, {**flux, "first_interface": 0})
    with pytest.raises(ValueError, match="same basins"):
        combine_tis_rate(results, {**flux, "dt_au": 0.2})


def imported_tis_summaries():
    shared = {
        "basin_a": -1.0, "basin_b": 1.0, "dt_au": 0.1, "beta_au": 2.0,
        "nbeads": 4, "max_path_steps": 100, "crossing_probability": 0.5,
    }
    return [
        {**shared, "interface_index": 0, "interface": -0.5, "target": 0.5, "target_is_product": False},
        {**shared, "interface_index": 1, "interface": 0.5, "target": 1.0, "target_is_product": True},
    ], {"basin_a": -1.0, "basin_b": 1.0, "first_interface": -0.5, "dt_au": 0.1, "flux_au": 0.2}


def test_imported_tis_summaries_require_identical_truncated_path_support():
    summaries, flux = imported_tis_summaries()
    candidate = combine_tis_rate(summaries, flux)
    assert candidate["rate_estimate_au"] == pytest.approx(0.05)
    assert candidate["max_path_steps"] == 100
    summaries[1]["max_path_steps"] = 200
    with pytest.raises(ValueError, match="same max_path_steps"):
        combine_tis_rate(summaries, flux)


@pytest.mark.parametrize("field,value", [
    ("basin_a", np.nan), ("basin_b", np.inf), ("interface", -2),
    ("target", np.inf), ("target", -0.75), ("dt_au", 0),
    ("beta_au", -1), ("beta_au", np.nan), ("nbeads", 0), ("nbeads", 1.5),
    ("nbeads", True), ("interface_index", 0.0), ("interface_index", False),
    ("max_path_steps", 0), ("max_path_steps", 1.5), ("max_path_steps", True),
    ("target_is_product", 0), ("crossing_probability", np.nan),
    ("length_rejections", -1),
])
def test_invalid_imported_tis_summary_metadata_is_rejected(field, value):
    summaries, flux = imported_tis_summaries()
    summaries[0][field] = value
    with pytest.raises(ValueError):
        combine_tis_rate(summaries, flux)


def test_decreasing_but_adjacent_imported_tis_interfaces_are_rejected():
    summaries, flux = imported_tis_summaries()
    summaries[0]["interface"] = flux["first_interface"] = 0.5
    summaries[0]["target"] = summaries[1]["interface"] = -0.5
    with pytest.raises(ValueError, match="ordered"):
        combine_tis_rate(summaries, flux)


def test_double_well_conditional_probability_matches_analytic_flux():
    # For Hamiltonian 1D motion up a monotone barrier, positive-flux energy
    # has an exponential tail, hence P(s_j|s_i)=exp[-beta(U_j-U_i)].  The
    # transverse harmonic coordinates integrate out.  This exercises actual
    # rejected/accepted variable-length path sampling, not independent shots.
    dyn = Dynamics(DoubleWell(), [1], 1.0, 1, 0.04, native=False)
    tis = TransitionInterfaceSampler(
        dyn, Cartesian(0), basin_a=-0.95, basin_b=0.95,
        interfaces=[-0.8, -0.4], max_path_steps=500, seed=832,
    )
    path = tis.generate_path(shooting(x=-0.6, velocity=1.5), 1)
    result = tis.sample(path, 0, n_samples=1500, burn_in=200)
    expected = np.exp(-(((-0.4) ** 2 - 1) ** 2 - ((-0.8) ** 2 - 1) ** 2))
    assert result.summary()["crossing_probability"] == pytest.approx(expected, abs=0.10)
    assert result.accepted.any() and not result.accepted.all()
    assert np.any(result.crossing) and not np.all(result.crossing)
    assert result.length_rejections == 0
