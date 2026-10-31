"""NHC ensemble scaling, independent-reference and extended-energy checks."""
import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.thermostats import NoseHooverChain


@pytest.mark.parametrize("beads", [1, 4])
def test_nhc_agrees_with_independent_ase_chain_and_excludes_frozen_dofs(beads):
    ase = pytest.importorskip("ase")
    from ase.md.nose_hoover_chain import NoseHooverChainThermostat
    from ase.units import kB
    masses = np.array([2., 3., 100.])
    dyn = Dynamics(FreePotential(), masses, 2., beads, .01, fixed_atoms=[2],
                   fixed_positions=np.zeros((3, 3)), native=False, seed=13)
    bath = NoseHooverChain(dyn, tdamp=.3, chain_length=3, substeps=2)
    reference = NoseHooverChainThermostat(num_atoms_global=2 * beads,
        masses=np.tile(masses[:2], beads)[:, None], temperature_K=(beads / 2) / kB,
        tdamp=.3, tchain=3, tloop=2)
    state = State(np.zeros((beads, 3, 3)), dyn.sample_momenta())
    refp = state.p[:, :2].reshape(-1, 3).copy()
    refq = state.q.copy()
    assert bath.dof == 6 * beads
    assert bath.Q[0] == pytest.approx(6 * beads * (beads / 2) * .3**2)
    for _ in range(40):
        bath.step(state)
        refp = reference.integrate_nhc(refp, .005)
        # Supply the same deterministic free-ring map between the two
        # independently implemented bath half steps (P>1 has spring motion).
        ringp = np.zeros_like(state.p)
        ringp[:, :2] = refp.reshape(beads, 2, 3)
        dyn.integrator.free_step(refq, ringp)
        refp = ringp[:, :2].reshape(-1, 3).copy()
        refp = reference.integrate_nhc(refp, .005)
    np.testing.assert_allclose(state.p[:, :2].reshape(-1, 3), refp, atol=2e-13)
    np.testing.assert_allclose(bath.eta, reference._eta, atol=2e-13)
    np.testing.assert_allclose(bath.momentum, reference._p_eta, atol=2e-13)
    np.testing.assert_array_equal(state.p[:, 2], 0)
    assert bath.energy() == pytest.approx(reference.get_thermostat_energy(), abs=3e-13)


def test_nhc_harmonic_extended_energy_and_time_reversal():
    def integrate(dt, steps):
        dyn = Dynamics(HarmonicPotential(k=.7), [2., 3.], 2., 1, dt, seed=5,
                       integrator="velocity-verlet", native=False)
        state = State(np.array([[[.3, .6, -.5], [.1, -.2, .7]]]), dyn.sample_momenta())
        bath = NoseHooverChain(dyn, tdamp=.7)
        initial = state.copy()
        e0 = dyn.energy(state) + bath.energy()
        error = 0.
        for _ in range(steps):
            bath.step(state)
            error = max(error, abs(dyn.energy(state) + bath.energy() - e0))
        state.p *= -1
        bath.momentum *= -1
        for _ in range(steps):
            bath.step(state)
        np.testing.assert_allclose(state.q, initial.q, atol=2e-12)
        np.testing.assert_allclose(state.p, -initial.p, atol=2e-12)
        np.testing.assert_allclose(bath.eta, 0, atol=2e-12)
        return error
    coarse, fine = integrate(.02, 200), integrate(.01, 400)
    assert coarse < 1e-3
    assert 3.5 < coarse / fine < 4.5


def test_bath_failure_restores_state_and_chain(monkeypatch):
    dyn = Dynamics(FreePotential(), [2.], 1., 1, .1, native=False)
    state = State(np.zeros((1, 1, 3)), dyn.sample_momenta())
    bath = NoseHooverChain(dyn, tdamp=.5)
    saved, old = state.copy(), bath.state_dict()
    original = dyn.step
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("failed force or observer")
    monkeypatch.setattr(dyn, "step", fail)
    with pytest.raises(RuntimeError):
        bath.step(state)
    np.testing.assert_array_equal(state.q, saved.q)
    np.testing.assert_array_equal(state.p, saved.p)
    assert state.step == saved.step and state.time == saved.time
    assert bath.state_dict() == old
    with pytest.raises(ValueError, match="parameters"):
        bath.load_state_dict(None)
