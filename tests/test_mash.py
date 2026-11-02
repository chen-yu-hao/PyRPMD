import numpy as np
import pytest

from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticResult
from pyrpmd.dynamics import State
from pyrpmd.mash import MASHDynamics


class RotatingSurfaces:
    """Constant eigenvalues but changing eigenvectors, with exact overlaps."""
    def __init__(self, gap=.002, coupling=1.):
        self.gap, self.coupling = gap, coupling

    def evaluate(self, q, *, previous_q=None, previous_context=None):
        angle = self.coupling*q[:, 0, 0]
        u = np.empty((len(q), 2, 2))
        u[:, 0, 0] = u[:, 1, 1] = np.cos(angle)
        u[:, 0, 1], u[:, 1, 0] = -np.sin(angle), np.sin(angle)
        energy = np.tile([0., self.gap], (len(q), 1))
        gradient = np.zeros((len(q), 2, 1, 3))
        nac = np.zeros((len(q), 2, 2, 1, 3))
        nac[:, 0, 1, 0, 0], nac[:, 1, 0, 0, 0] = -self.coupling, self.coupling
        overlap = None if previous_context is None else previous_context.swapaxes(1, 2)@u
        return AdiabaticResult(energy, gradient, nac, overlap, context=u)


def dynamics(gap=.002, beads=1, dt=.1):
    return MASHDynamics(AdiabaticPythonPotential(RotatingSurfaces(gap)), [1.], 2., beads, dt,
                         native=False, seed=17)


def incoming(d, active=0):
    q = np.zeros((d.nbeads, 1, 3))
    p = np.zeros_like(q)
    p[:, 0, 0] = .2
    c = np.array([np.sqrt(.55), -np.sqrt(.45)])
    if active:
        c = c[::-1]
    return d.initialize(State(q, p), active, adiabatic_coefficients=c)


def test_mash_hemisphere_sampling_has_correct_population_weight():
    d = dynamics()
    n = State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3)))
    z = np.array([abs(d.initialize(n).coefficients[0])**2*2-1 for _ in range(3000)])
    # p(z)=2z on 0<z<1; E[z]=2/3 and E[z^2]=1/2.
    assert np.min(z) > 0 and np.max(z) <= 1
    assert z.mean() == pytest.approx(2/3, abs=.015)
    assert np.mean(z*z) == pytest.approx(.5, abs=.015)


@pytest.mark.parametrize("beads", [1, 4])
def test_mash_crossing_is_deterministic_and_conserves_ring_energy(beads):
    d = dynamics(beads=beads)
    s = incoming(d)
    e = d.energy(s)
    for _ in range(8):
        d.step(s)
        if d.last_hop is not None:
            break
    assert d.last_hop is not None and d.last_hop.accepted
    assert s.active_surface == 1
    assert abs(s.coefficients[1])**2 > .5
    np.testing.assert_array_equal(d.populations(s), [0., 1.])
    assert d.energy(s) == pytest.approx(e, abs=2e-14)
    np.testing.assert_allclose(s.p[:, 0, 0], np.sqrt(.2**2-2*.002), atol=2e-14)


def test_mash_frustration_reflects_momentum_and_spin_hemisphere():
    d = dynamics(gap=.04)
    s = incoming(d)
    before = d.energy(s)
    for _ in range(8):
        d.step(s)
        if d.last_hop is not None:
            break
    assert d.last_hop is not None and d.last_hop.frustrated
    assert s.active_surface == 0
    assert abs(s.coefficients[0])**2 > .5
    np.testing.assert_allclose(s.p[:, 0, 0], -.2, atol=2e-14)
    assert d.energy(s) == pytest.approx(before, abs=2e-14)


def test_reflection_preserves_transverse_spin_and_only_flips_z():
    d = dynamics(gap=.04)
    s = incoming(d)
    # Exercise the actual transition decision with a crossed spin. This checks
    # the phase-sensitive reflection, not just the norm or active label.
    s.coefficients[:] = [np.sqrt(.4)*np.exp(.7j), np.sqrt(.6)*np.exp(-.2j)]
    coherence = s.coefficients[0].conj()*s.coefficients[1]
    z = abs(s.coefficients[0])**2-abs(s.coefficients[1])**2
    hop = d._select_hop(s, np.zeros(2), [0, 1])
    assert hop.frustrated
    assert s.coefficients[0].conj()*s.coefficients[1] == pytest.approx(coherence)
    assert abs(s.coefficients[0])**2-abs(s.coefficients[1])**2 == pytest.approx(-z)


def test_mash_rejects_wrong_method_options_and_inconsistent_spin():
    provider = AdiabaticPythonPotential(RotatingSurfaces())
    with pytest.raises(ValueError, match="two states"):
        MASHDynamics(provider, [1.], 2., 1, .1, nstates=3)
    with pytest.raises(ValueError, match="reflection"):
        MASHDynamics(provider, [1.], 2., 1, .1, frustrated_hops="keep")
    d = dynamics()
    with pytest.raises(ValueError, match="hemisphere"):
        d.initialize(State(np.zeros((1, 1, 3)), np.zeros((1, 1, 3))), 0,
                     adiabatic_coefficients=[0., 1.])
    with pytest.raises(ValueError, match="spin crossings"):
        d.attempt_hop(incoming(d), 1)


def test_mash_rejects_unresolved_multiple_crossings():
    d = dynamics()
    s = incoming(d)
    before = s.copy()
    with pytest.raises(ValueError, match="multiple MASH"):
        d._select_hop(s, np.zeros(2), [0, 1, 0])
    np.testing.assert_array_equal(s.p, before.p)
