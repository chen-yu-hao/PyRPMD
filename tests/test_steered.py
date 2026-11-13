"""Independent work identities, CV derivatives and exact checkpoint replay."""

import copy
import json

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.special import logsumexp

from pyrpmd.analysis.mbar import solve_mbar
from pyrpmd.checkpoint import load_checkpoint, save_metadata_npz
from pyrpmd.cv import Cartesian, CVStack, Distance
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.builtin import DoubleWellPotential, HarmonicPotential
from pyrpmd.steered import SteeredBias, SteeredMD, crooks, jarzynski


def simulation(*, beads=1, seed=17, gamma=0.7, equilibration=0, thermostat="langevin",
               centers=None, dt=0.04, state=None, potential=None, identity="harmonic:k=1"):
    dynamics = Dynamics(HarmonicPotential() if potential is None else potential, [1.0], 1.3,
                        beads, dt, seed=seed, native=False, potential_identity=identity)
    if state is None:
        q = np.linspace(-0.2, 0.2, 3 * beads).reshape(beads, 1, 3)
        state = State(q, dynamics.sample_momenta())
    if centers is None:
        centers = np.linspace(-0.3, 0.8, 21)[:, None]
    return SteeredMD(dynamics, state, SteeredBias(Cartesian(0, 0), centers, 2.0),
                     gamma=gamma, thermostat=thermostat, equilibration_steps=equilibration)


@pytest.mark.parametrize("beads", [1, 4, 16])
def test_steered_cv_jacobian_force_and_bead_scaling(beads):
    cv = CVStack([Distance([0, 1]), Cartesian(0, "y")])
    q = np.broadcast_to([[[0.2, 0.4, 0.1], [1.3, -0.2, 0.1]]], (beads, 2, 3)).copy()
    bias = SteeredBias(cv, [[1.0, 0.0], [1.2, 0.4]], [[2.0, 0.3], [0.3, 1.4]])
    energy, force = bias.energy_force(q)
    value = cv.value(q)
    assert energy == pytest.approx(beads * bias.physical_energy(value))
    numeric = np.zeros_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numeric[index] = -(bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / 2e-6
    np.testing.assert_allclose(force, numeric, atol=5e-9)
    np.testing.assert_allclose(force, np.broadcast_to(force[:1], force.shape))


def test_exact_discrete_crooks_work_distribution_and_unequal_sample_counts():
    # p(+log(2))=2/3, p(-log(2))=1/3 satisfies Jarzynski exactly;
    # its reverse distribution is identical. Add a known endpoint offset.
    base = np.log(2) * np.array([1, 1, -1])
    forward = np.tile(base, 200) + 3.25
    reverse = np.tile(base, 71) - 3.25
    for bootstrap in (0, 35):
        result = crooks(forward, reverse, 1.0, bootstrap=bootstrap, seed=10)
        assert result.delta_free_energy == pytest.approx(3.25, abs=1e-12)
        assert result.diagnostics["overlap"] > 0.8
        assert result.standard_error > 0
        assert jarzynski(forward, 1.0, bootstrap=bootstrap).delta_free_energy == pytest.approx(3.25)
    # Independent equilibrium MBAR on the same two-state work differences.
    joined = np.concatenate((forward, -reverse))
    fit = solve_mbar(np.stack((np.zeros(len(joined)), joined)), [len(forward), len(reverse)])
    assert fit.free_energies[1] == pytest.approx(result.delta_free_energy, abs=1e-8)


def test_work_estimators_stable_offsets_and_diagnostic_semantics():
    works = np.array([-0.4, 0.2, 0.2, 0.6, 1.0])
    a, b = jarzynski(works, 1.7), jarzynski(works + 1e6, 1.7)
    assert b.delta_free_energy - 1e6 == pytest.approx(a.delta_free_energy, abs=1e-10)
    assert b.standard_error == pytest.approx(a.standard_error, abs=1e-10)
    assert a.diagnostics["leading_order_positive_bias"] >= 0
    assert a.diagnostics["work_variance"] == pytest.approx(works.var(ddof=1))
    assert a.diagnostics["largest_normalized_weight"] <= 1
    one = jarzynski([2.0], 1.0)
    assert np.isnan(one.standard_error)
    assert one.to_dict()["standard_error"] is None
    absent_overlap = crooks([1000.0, 1001.0], [1000.0, 1001.0], 1.0, bootstrap=5)
    assert np.isfinite(absent_overlap.delta_free_energy)
    assert not absent_overlap.diagnostics["overlap_resolved"]
    assert np.isnan(absent_overlap.standard_error)
    impossible = crooks([-1000.0, -1000.0], [-1000.0, -1000.0], 1.0, bootstrap=5)
    assert not impossible.diagnostics["overlap_resolved"]
    assert not impossible.converged
    assert np.isnan(impossible.standard_error)


@pytest.mark.parametrize("method", [jarzynski, lambda w, b, **kw: crooks(w, [1.0], b, **kw)])
@pytest.mark.parametrize("work,beta,options", [
    ([], 1, {}), ([[1]], 1, {}), ([True], 1, {}), ([1j], 1, {}), ([np.nan], 1, {}),
    ([1], 0, {}), ([1], True, {}), ([1], 1, {"bootstrap": 1}),
    ([1], 1, {"bootstrap": 1.5}),
])
def test_work_estimators_reject_invalid_inputs(method, work, beta, options):
    with pytest.raises(ValueError):
        method(work, beta, **options)


@pytest.mark.parametrize("thermostat", ["none", "langevin"])
@pytest.mark.parametrize("beads", [1, 4])
def test_first_law_separates_ou_heat_and_shadow_work(thermostat, beads):
    sim = simulation(thermostat=thermostat, beads=beads)
    initial = sim.dynamics.energy(sim.state, sim.bias) / beads
    result = sim.run()
    final = sim.dynamics.energy(sim.state, sim.bias) / beads
    assert final - initial == pytest.approx(result["total_work"][-1] + result["heat"][-1], abs=2e-13)
    np.testing.assert_array_equal(result["total_work"], result["protocol_work"] + result["shadow_work"])
    if thermostat == "none":
        np.testing.assert_array_equal(result["heat"], 0)
    else:
        assert np.max(np.abs(result["heat"])) > 0


def test_time_reversed_protocol_retraces_hamiltonian_pull():
    sim = simulation(thermostat="none", beads=4, dt=0.07)
    start = sim.state.copy()
    forward = sim.run()
    reverse_state = State(sim.state.q, -sim.state.p)
    reverse = simulation(thermostat="none", beads=4, dt=0.07, state=reverse_state,
                         centers=sim.bias.centers[::-1]).run()
    np.testing.assert_allclose(reverse["state"].q, start.q, atol=3e-15)
    np.testing.assert_allclose(reverse["state"].p, -start.p, atol=4e-15)
    assert forward["total_work"][-1] == pytest.approx(-reverse["total_work"][-1], abs=3e-14)
    assert forward["protocol_work"][-1] == pytest.approx(-reverse["protocol_work"][-1], abs=3e-14)


def test_finite_step_shadow_work_recovers_exact_harmonic_endpoint_free_energy():
    # Gauss-Hermite integration over independently canonical initial q,p.
    # A large reversible timestep intentionally makes protocol-only work biased.
    beta, k, stiffness, end = 1.3, 1.0, 2.0, 1.0
    nodes, weights = np.polynomial.hermite.hermgauss(18)
    logs, protocol_logs, log_weights = [], [], []
    for i, x in enumerate(nodes):
        for j, p in enumerate(nodes):
            q = np.zeros((1, 1, 3))
            momenta = q.copy()
            q[0, 0, 0] = np.sqrt(2 / (beta * (k + stiffness))) * x
            momenta[0, 0, 0] = np.sqrt(2 / beta) * p
            sim = simulation(thermostat="none", dt=0.6, state=State(q, momenta),
                             centers=np.linspace(0, end, 5)[:, None])
            result = sim.run()
            logs.append(-beta * result["total_work"][-1])
            protocol_logs.append(-beta * result["protocol_work"][-1])
            log_weights.append(np.log(weights[i] * weights[j] / np.pi))
    exact = 0.5 * k * stiffness / (k + stiffness) * end**2
    corrected = -logsumexp(np.asarray(logs) + log_weights) / beta
    uncorrected = -logsumexp(np.asarray(protocol_logs) + log_weights) / beta
    assert corrected == pytest.approx(exact, abs=2e-8)
    assert abs(uncorrected - exact) > 0.01


@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("cut", [3, 7, 13])
def test_exact_resume_in_equilibration_at_boundary_and_during_pull(tmp_path, beads, cut):
    whole = simulation(beads=beads, equilibration=7)
    reference = whole.run()
    path = tmp_path / "steered.npz"
    prefix = simulation(beads=beads, equilibration=7)
    prefix.run(cut, checkpoint=path, checkpoint_interval=3)
    continued = simulation(beads=beads, equilibration=7)
    continued.load(path)
    result = continued.run()
    for field in ("time", "cv_values", "centers", "protocol_work", "shadow_work", "total_work", "heat"):
        np.testing.assert_array_equal(result[field], reference[field])
    np.testing.assert_array_equal(result["state"].q, reference["state"].q)
    np.testing.assert_array_equal(result["state"].p, reference["state"].p)
    assert continued.dynamics.rng.bit_generator.state == whole.dynamics.rng.bit_generator.state
    assert result["equilibration_accepted"] == reference["equilibration_accepted"]


def test_checkpoint_rejects_changed_physics_and_validates_before_restoring_rng(tmp_path):
    path = tmp_path / "pull.npz"
    sim = simulation(equilibration=7)
    sim.run(13, checkpoint=path)
    with pytest.raises(ValueError, match="identity"):
        simulation(gamma=0.71, equilibration=7).load(path)
    restored = simulation(equilibration=7)
    before = restored.dynamics.operator_state_dict()
    with load_checkpoint(path) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: archive[name] for name in ("q", "p", "records")}
    metadata["step"] += 1
    save_metadata_npz(path, metadata, **arrays)
    with pytest.raises(ValueError, match="clock"):
        restored.load(path)
    assert before == restored.dynamics.operator_state_dict()


def test_failed_force_rolls_back_rng_protocol_clock_and_work():
    class FailOnDemand(HarmonicPotential):
        broken = False

        def evaluate(self, q):
            if self.broken:
                raise RuntimeError("intentional force error")
            return super().evaluate(q)

    provider = FailOnDemand()
    sim = simulation(potential=provider)
    state, rng = sim.state.copy(), copy.deepcopy(sim.dynamics.rng.bit_generator.state)
    provider.broken = True
    with pytest.raises(RuntimeError, match="intentional"):
        sim.step()
    np.testing.assert_array_equal(sim.state.q, state.q)
    np.testing.assert_array_equal(sim.state.p, state.p)
    np.testing.assert_array_equal(sim.bias.center, sim.bias.centers[0])
    assert sim.state.step == state.step and sim.state.time == state.time
    assert sim.dynamics.rng.bit_generator.state == rng
    assert sim.elapsed == 0 and len(sim.result()["total_work"]) == 1


def test_checkpoint_requires_explicit_potential_identity_before_any_steps(tmp_path):
    sim = simulation(identity=None)
    with pytest.raises(ValueError, match="potential_identity"):
        sim.run(checkpoint=tmp_path / "checkpoint.npz")
    assert sim.elapsed == 0
    assert not (tmp_path / "checkpoint.npz").exists()


def test_changed_protocol_is_not_silently_reused():
    sim = simulation()
    sim.bias.stiffness = np.eye(1) * 3
    with pytest.raises(ValueError, match="parameters changed"):
        sim.run()


@pytest.mark.parametrize("cv,centers,stiffness", [
    (Cartesian(0, 0, estimator="bead_average"), [[0], [1]], 1),
    (Cartesian(0, 0), [[0]], 1),
    (Cartesian(0, 0), [[0], [np.nan]], 1),
    (Cartesian(0, 0), [[0], [1]], -1),
    (Cartesian(0, 0), [[0], [1]], True),
])
def test_steered_bias_rejects_unsupported_parameters(cv, centers, stiffness):
    with pytest.raises(ValueError):
        SteeredBias(cv, centers, stiffness)


def test_double_well_actual_pulls_agree_with_quadrature_and_umbrella_mbar():
    beta, stiffness = 1.3, 1.5
    start, end = -0.7, 0.3
    grid = np.linspace(-3.5, 3.5, 16001)
    potential = 0.4 * (grid**2 - 1)**2
    rng = np.random.default_rng(441)
    works, initials, z = [], [], []
    for center, target in ((start, end), (end, start)):
        density = np.exp(-beta * (potential + 0.5 * stiffness * (grid - center)**2))
        z.append(simpson(density, x=grid))
        cdf = cumulative_trapezoid(density, grid, initial=0)
        cdf /= cdf[-1]
        positions = np.interp(rng.random(350), cdf, grid)
        initials.append(positions)
        direction = []
        for x in positions:
            dynamics = Dynamics(DoubleWellPotential(height=0.4, confinement=1.0), [1.0], beta,
                                1, 0.12, seed=0, native=False)
            q = np.zeros((1, 1, 3))
            q[0, 0, 0] = x
            # The transverse modes decouple and cancel from the endpoint ratio.
            p = np.zeros_like(q)
            p[0, 0, 0] = rng.normal() / np.sqrt(beta)
            driver = SteeredMD(dynamics, State(q, p),
                               SteeredBias(Cartesian(0, 0), np.linspace(center, target, 7)[:, None], stiffness),
                               thermostat="none", initial_equilibrium=True)
            direction.append(driver.run()["total_work"][-1])
        works.append(np.asarray(direction))
    exact = -np.log(z[1] / z[0]) / beta
    result = crooks(*works, beta, bootstrap=50, seed=88)
    assert abs(result.delta_free_energy - exact) < 4 * result.standard_error
    joined = np.concatenate(initials)
    reduced = beta * 0.5 * stiffness * (joined[None] - np.asarray([start, end])[:, None])**2
    mbar = solve_mbar(reduced, [350, 350])
    mbar_delta = (mbar.free_energies[1] - mbar.free_energies[0]) / beta
    assert abs(result.delta_free_energy - mbar_delta) < 4 * result.standard_error
    assert result.diagnostics["overlap_resolved"]
