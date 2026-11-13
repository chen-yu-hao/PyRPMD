"""Paper DAC reference and analytically equivalent 2-state spin-flow tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.dynamics import State
from pyrpmd.nonadiabatic import NRPMDDynamics, NRPMDState
from pyrpmd.spin_mapping import SMNRPMDDynamics
from pyrpmd.potentials.nonadiabatic_models import TullyModelTwo

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/paper_validation/dac_spin_statistical.py"
spec = importlib.util.spec_from_file_location("paper_dac_spin_statistical", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("momentum, end, reference_end", [
    (5, 6000, 0.00069145052342991684),
    (15, 2800, 0.12901143457710037565),
    (30, 1200, 0.65991740127597031673),
])
def test_source_pinned_quantum_anchors_are_normalized(momentum, end, reference_end):
    data, path = benchmark.reference(momentum)
    assert path.is_file()
    assert data[-1, 0] == end
    np.testing.assert_allclose(data[:, 1] + data[:, 2], 1., rtol=0, atol=2e-7)
    assert abs(data[-1, 2] - reference_end) < 1e-14
    assert data.shape[0] == end//40 + 1


@pytest.mark.parametrize("method", ["nrpmd", "sm-nrpmd"])
def test_vectorized_exact_spin_matches_original_scalar_dynamics(method):
    potential = TullyModelTwo()
    x = np.array([-9., -4., -.7, .1, 2., 7.], dtype=float)
    p = np.array([5., 15., 5., 15., 5., 15.], dtype=float)
    dt, steps = 2., 120
    states = []
    spins = []
    for index, (q0, p0) in enumerate(zip(x, p)):
        q = np.zeros((1, 1, 3))
        momenta = np.zeros_like(q)
        q[0, 0, 0], momenta[0, 0, 0] = q0, p0
        nuclear = State(q, momenta)
        if method == "nrpmd":
            driver = NRPMDDynamics(potential, [benchmark.MASS], 100., 1, dt, 2,
                                   native=False, integrator="velocity-verlet")
            state = NRPMDState.from_nuclear(nuclear, 2, 0, np.random.default_rng(90+index))
            z = state.mapping_q[0] + 1j*state.mapping_p[0]
            spin = np.array([(z[0].conj()*z[1]).real,
                             (z[0].conj()*z[1]).imag,
                             (abs(z[0])**2-abs(z[1])**2)/2])
        else:
            driver = SMNRPMDDynamics(potential, [benchmark.MASS], 100., 1, dt,
                                     native=False, integrator="velocity-verlet")
            state = driver.initialize(nuclear, 0, np.random.default_rng(90+index))
            spin = np.sqrt(3.)*state.spin[0]
        states.append((driver, state))
        spins.append(spin)
    x = x.copy()
    p = p.copy()
    spins = np.asarray(spins)
    np.testing.assert_allclose(
        benchmark.energy(x, p, spins),
        [driver.energy(state) for driver, state in states], rtol=0, atol=1e-13)
    for _ in range(steps):
        benchmark.one_step(x, p, spins, dt)
        for driver, state in states:
            driver.step(state)
    expected_q = np.array([state.q[0,0,0] for _,state in states])
    expected_p = np.array([state.p[0,0,0] for _,state in states])
    expected_spin = []
    for _, state in states:
        if method == "nrpmd":
            z = state.mapping_q[0]+1j*state.mapping_p[0]
            expected_spin.append([(z[0].conj()*z[1]).real,
                                  (z[0].conj()*z[1]).imag,
                                  (abs(z[0])**2-abs(z[1])**2)/2])
        else:
            expected_spin.append(np.sqrt(3.)*state.spin[0])
    np.testing.assert_allclose(x, expected_q, rtol=0, atol=2e-10)
    np.testing.assert_allclose(p, expected_p, rtol=0, atol=2e-10)
    np.testing.assert_allclose(spins, expected_spin, rtol=0, atol=2e-10)
    np.testing.assert_allclose(
        benchmark.energy(x, p, spins),
        [driver.energy(state) for driver,state in states], rtol=0, atol=2e-10)


def test_fullsphere_w_representation_has_correct_initial_operator_trace():
    # Eq. (19–20) for two states, W gamma=(sqrt(3)-1)/2.
    # Integrate analytically-polynomial Bloch projectors over uniform sphere.
    nodes, weights = np.polynomial.legendre.leggauss(16)
    sz = np.sqrt(3.)*nodes
    initial = (1.+sz)/2.
    final0 = (1.+sz)/2.
    final1 = (1.-sz)/2.
    assert abs(np.sum(weights*2*initial*final0)/2.-1.) < 1e-13
    assert abs(np.sum(weights*2*initial*final1)/2.) < 1e-13


@pytest.mark.parametrize("momentum", [5, 15, 30])
def test_seeded_statistical_smoke_never_clips_signed_population(momentum):
    methods=("mmst-focused","w-focused","spm-w-fullsphere","ehrenfest")
    a,_=benchmark.simulate(momentum, 32, 40., 17, 2, methods)
    b,_=benchmark.simulate(momentum, 32, 40., 17, 2, methods)
    assert a == b
    assert a["claims"]["paper_level_method_agreement_certified"] is False
    assert a["reference"]["times"][-1] == ({5: 6000, 15: 2800, 30: 1200}[momentum])
    for name, result in a["methods"].items():
        assert len(result["mean"]) == len(a["reference"]["times"])
        assert result["max_spin_norm_error"] < 1e-10
        assert np.isfinite(result["max_energy_drift"])
        assert result["max_energy_drift"] < 1.
        if name != "spm-w-fullsphere":
            assert abs(result["initial_population_mean"]) < 1e-14


def test_invalid_reference_and_unknown_electronic_sampling_are_rejected():
    with pytest.raises(ValueError, match="only pinned"):
        benchmark.reference(99)
    with pytest.raises(ValueError, match="unknown method"):
        benchmark.starting_spin("fake", np.array([0.0]), np.array([0.0]))
