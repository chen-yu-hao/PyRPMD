"""Restricted electrode/matrix coupling; exact limits, not physical-rate tests."""
import numpy as np
import pytest

from pyrpmd.electrochemistry import ElectronReservoir, ElectronicPESResult, AdiabaticElectrodePotential
from pyrpmd.electrochemistry.diabatic import StateIndependentElectrodeDiabaticPotential
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDState
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics, ThermalNRPMDSampler, thermal_weight


class LocalMatrix:
    def __init__(self, coupling=.2):
        self.coupling = coupling
    def evaluate(self, q):
        x = q[:, 0, 0]
        v = np.zeros((len(q), 2, 2))
        g = np.zeros((len(q), 2, 2, *q.shape[1:]))
        common = .35*np.sum(q*q, axis=(1, 2))
        for s in range(2):
            sign = 1-2*s
            v[:, s, s] = common + sign*(.15*x + .06)
            g[:, s, s] = .7*q
            g[:, s, s, 0, 0] += sign*.15
        v[:, 0, 1] = v[:, 1, 0] = self.coupling
        return DiabaticResult(v, g)


class Charging:
    def evaluate(self, q, electron_numbers):
        n = electron_numbers
        force = np.zeros_like(q)
        force[:, 0, 0] = .4*n
        return ElectronicPESResult(.65*n*n-.4*q[:, 0, 0]*n,
                                   force, 1.3*n-.4*q[:, 0, 0])


def combined(voltage=0., local=None):
    bath = ElectronReservoir(voltage, 0., .23, "synthetic-model")
    electrode = AdiabaticElectrodePotential(Charging(), bath,
        electron_bounds=(-20, 20), chemical_potential_tolerance_au=1e-12)
    return StateIndependentElectrodeDiabaticPotential(local or LocalMatrix(), electrode)


def state():
    rng = np.random.default_rng(812)
    return NRPMDState(rng.normal(size=(4, 1, 3))*.2,
        rng.normal(size=(4, 1, 3))*.2,
        mapping_q=rng.normal(size=(4, 2)), mapping_p=rng.normal(size=(4, 2)))


@pytest.mark.parametrize("voltage", [-.2, 0., .2])
def test_exact_elimination_and_all_matrix_gradients(voltage):
    potential = combined(voltage)
    q = state().q
    r = potential.evaluate_with_electrode(q)
    mu = potential.electrode.reservoir.chemical_potential_au
    number = (.4*q[:, 0, 0]+mu)/1.3
    omega = -.5*1.3*number**2
    np.testing.assert_allclose(r.electrode.electron_numbers, number, atol=1e-14)
    np.testing.assert_allclose(r.electrode.chemical_potentials_au, mu, atol=1e-14)
    bare = potential.relative_diabatic.evaluate(q)
    np.testing.assert_allclose(r.diabatic.matrix, bare.matrix+omega[:, None, None]*np.eye(2), atol=1e-14)
    eps = 1e-5
    for a in range(q.shape[1]):
        for c in range(3):
            qp, qm = q.copy(), q.copy()
            qp[:, a, c] += eps
            qm[:, a, c] -= eps
            numerical = (potential.evaluate(qp).matrix-potential.evaluate(qm).matrix)/(2*eps)
            np.testing.assert_allclose(numerical, r.diabatic.gradients[:, :, :, a, c], atol=2e-10)


def test_batch_independence_and_provider_buffers_not_mutated():
    local = LocalMatrix()
    q = state().q
    original = local.evaluate(q)
    class Buffered:
        def evaluate(self, q):
            return original
    before = original.matrix.copy()
    combined(local=Buffered()).evaluate(q)
    np.testing.assert_array_equal(original.matrix, before)
    potential = combined()
    all_frames = potential.evaluate(q)
    frames = [potential.evaluate(frame[None]) for frame in q]
    np.testing.assert_allclose(np.concatenate([r.matrix for r in frames]), all_frames.matrix)
    np.testing.assert_allclose(np.concatenate([r.gradients for r in frames]), all_frames.gradients)


def test_scalar_thermal_weight_exact_shift_even_with_fluctuating_mapping_norm():
    s = state()
    beta = 3.
    potential = combined()
    w0 = thermal_weight(potential.relative_diabatic, s, [1.], beta)
    w1 = thermal_weight(potential, s, [1.], beta)
    omega = potential.electrode.evaluate(s.q).energy
    assert w1.sign == w0.sign
    assert w1.log_density-w0.log_density == pytest.approx(-beta*np.mean(omega), abs=1e-12)
    np.testing.assert_allclose(w1.electronic_populations, w0.electronic_populations)


def test_thermal_propagation_reversibility_energy_and_second_order():
    potential = combined()
    initial = state()
    drifts = []
    for dt, steps in ((.02, 100), (.01, 200)):
        dynamics = ThermalNRPMDDynamics(potential, [1.], 3., 4, dt,
                                        nstates=2, native=False)
        s = initial.copy()
        e0 = dynamics.energy(s)
        errors = []
        for _ in range(steps):
            dynamics.step(s)
            errors.append(abs(dynamics.energy(s)-e0))
        drifts.append(max(errors))
        s.p *= -1
        s.mapping_p *= -1
        for _ in range(steps):
            dynamics.step(s)
        np.testing.assert_allclose(s.q, initial.q, atol=2e-12)
        np.testing.assert_allclose(s.p, -initial.p, atol=2e-12)
        np.testing.assert_allclose(s.mapping_q, initial.mapping_q, atol=2e-12)
        np.testing.assert_allclose(s.mapping_p, -initial.mapping_p, atol=2e-12)
    assert drifts[1] < .3*drifts[0]


def test_existing_thermal_sampler_and_driver_end_to_end():
    potential = combined()
    sampler = ThermalNRPMDSampler(potential, [1.], 3., state().q, nstates=2, seed=41)
    parents = sampler.sample(6, burn_in=10, stride=2)
    dynamics = ThermalNRPMDDynamics(potential, [1.], 3., 4, .01, nstates=2, native=False)
    assert parents.potential is potential
    for parent in parents.states:
        child = parent.copy()
        dynamics.step(child)
        assert np.isfinite(dynamics.energy(child))
    assert set(parents.signs) <= {-1., 1.}


def test_complex_local_matrix_is_preserved_by_adapter():
    class Complex(LocalMatrix):
        def evaluate(self, q):
            r = super().evaluate(q)
            r.matrix = r.matrix.astype(complex)
            r.matrix[:, 0, 1] += .07j
            r.matrix[:, 1, 0] -= .07j
            return r
    q = state().q
    r = combined(local=Complex()).evaluate(q)
    np.testing.assert_allclose(r.matrix[:, 0, 1].imag, .07)
    np.testing.assert_allclose(r.matrix, r.matrix.conj().swapaxes(1, 2))


@pytest.mark.parametrize("args", [(None, None), (LocalMatrix(), None)])
def test_reject_invalid_provider_contract(args):
    with pytest.raises(TypeError):
        StateIndependentElectrodeDiabaticPotential(*args)
