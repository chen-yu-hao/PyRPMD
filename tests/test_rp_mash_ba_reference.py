"""Independent overlap, event and Fourier references for two-state BA-RP-MASH.

The wavepacket check is a controlled high-kinetic-energy, small-gap limit.
It does not assert that approximate ring-polymer trajectories solve general
coupled electronic/nuclear quantum dynamics.
"""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.optimize import brentq

from pyrpmd.analysis.quantum_reference import SplitOperator1D
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.nonadiabatic_models import LinearVibronicCoupling
from pyrpmd.rp_mash import RPMASHDynamics, RPMASHState
from pyrpmd.rp_mash_ba import RPMASHBADynamics, RPMASHBAState


class RotatingSurfaces:
    """Constant adiabatic energies +/-gap and analytically rotating vectors."""

    def __init__(self, gap=.2, direction=None, curvature=None):
        self.gap = float(gap)
        self.direction = np.array([[1., 0., 0.]] if direction is None else direction)
        self.curvature = np.zeros_like(self.direction) if curvature is None else np.array(curvature)

    def evaluate(self, q):
        angle = np.sum(self.direction*q + .5*self.curvature*q*q, axis=(1, 2))
        derivative = self.direction + self.curvature*q
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.empty((len(q), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = self.gap*cosine, -self.gap*cosine
        matrix[:, 0, 1] = matrix[:, 1, 0] = self.gap*sine
        angle_gradient = np.empty_like(matrix)
        angle_gradient[:, 0, 0], angle_gradient[:, 1, 1] = -self.gap*sine, self.gap*sine
        angle_gradient[:, 0, 1] = angle_gradient[:, 1, 0] = self.gap*cosine
        return DiabaticResult(matrix, angle_gradient[..., None, None]*derivative[:, None, None])


def rotation(spin, omega, time):
    """Independent matrix exponential of the real antisymmetric generator."""
    x, y, z = omega
    generator = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return expm(time*generator)@spin


def eigensystem(model, q):
    energies, vectors = np.linalg.eigh(model.evaluate(q).matrix)
    # Ordered upper/lower columns and determinant +1 fix the relative sign.
    # The remaining simultaneous column sign does not change <upper|d lower>.
    frame = vectors[..., ::-1].copy()
    frame[..., 1] *= np.linalg.det(frame)[:, None]
    return energies, frame


def overlap_derivative(model, q, step=2e-5):
    """Central differences of aligned eigenvector overlaps, not PES gradients."""
    energies, frame = eigensystem(model, q)
    coupling = np.empty_like(q)
    gradients = np.empty((len(q), 2, *q.shape[1:]))
    for atom in range(q.shape[1]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += step
            minus[:, atom, axis] -= step
            ep, up = eigensystem(model, plus)
            em, um = eigensystem(model, minus)
            up *= np.sign(np.einsum("bij,bij->bj", frame, up))[:, None, :]
            um *= np.sign(np.einsum("bij,bij->bj", frame, um))[:, None, :]
            coupling[:, atom, axis] = np.einsum(
                "bi,bi->b", frame[:, :, 0], (up[:, :, 1]-um[:, :, 1])/(2*step))
            gradients[:, :, atom, axis] = (ep-em)/(2*step)
    return energies, coupling, gradients


def lvc_example():
    return LinearVibronicCoupling(
        [[.17, .23], [.23, -.09]],
        [[[.31, .07], [.07, -.22]], [[-.05, .13], [.13, .18]]],
        harmonic=[.17, .11], coordinates=[0, 4], natoms=2)


@pytest.mark.parametrize("active", [0, 1])
def test_bead_generator_and_forces_match_finite_difference_overlaps(active):
    model = lvc_example()
    masses = np.array([1.3, 3.7])
    q = np.zeros((4, 2, 3))
    q[:, 0, 0], q[:, 1, 1] = [-.7, .1, .45, .8], [.4, -.6, .2, -.3]
    p = np.random.default_rng(8).normal(size=q.shape)
    spin = np.array([.6, -.3, (2*active-1)*np.sqrt(.55)])
    driver = RPMASHBADynamics(model, masses, 5., 4, .02)
    packed = np.concatenate((q.ravel(), p.ravel(), spin))
    observed = driver._rhs(0., packed, active)
    energies, nac, gradients = overlap_derivative(model, q)
    velocity = p/masses[None, :, None]
    omega = [0., 2*np.sum(nac*velocity)/len(q), np.mean(energies[:, 1]-energies[:, 0])]
    spring = (4/5.)**2*masses[None, :, None]*(2*q-np.roll(q, 1, 0)-np.roll(q, -1, 0))
    expected = np.concatenate((velocity.ravel(), (-gradients[:, active]-spring).ravel(),
                               np.cross(omega, spin)))
    np.testing.assert_allclose(observed, expected, rtol=0., atol=2e-9)

    # This geometry explicitly distinguishes a bead average from the
    # centroid Hamiltonian/NAC, including unequal nuclear masses.
    ec, dc, _ = overlap_derivative(model, q.mean(axis=0, keepdims=True))
    centroid_omega = [0., 2*np.sum(dc[0]*velocity.mean(axis=0)), ec[0, 1]-ec[0, 0]]
    assert np.linalg.norm(np.cross(omega, spin)-np.cross(centroid_omega, spin)) > .01


@pytest.mark.parametrize("source,scale", [(0, 1.), (1, 1.), (0, .003)])
def test_full_bead_impulse_matches_overlap_direction_and_energy(source, scale):
    model = lvc_example()
    masses = np.array([1.3, 3.7])
    q = np.zeros((4, 2, 3))
    q[:, 0, 0], q[:, 1, 1] = [-.7, .1, .45, .8], [.4, -.6, .2, -.3]
    energies, nac, _ = overlap_derivative(model, q)
    # Large longitudinal momentum ensures the first upward case is accepted;
    # the third case is energetically frustrated on exactly the same ring.
    p = scale*(8*nac + np.array([[[0., .17, .23], [.13, 0., -.11]]]))
    before = p.copy()
    driver = RPMASHBADynamics(model, masses, 5., 4, .02)
    updated, event = driver._impulse(q, p, source, .04)
    np.testing.assert_array_equal(p, before)
    gap = (1-2*source)*np.sum(energies[:, 1]-energies[:, 0])
    np.testing.assert_allclose(event.ring_energy_gap, gap, rtol=0., atol=3e-14)
    change = updated-p
    factor = np.vdot(change, nac)/np.vdot(nac, nac)
    np.testing.assert_allclose(change, factor*nac, rtol=0., atol=2e-8)
    old_projection = np.sum(p*nac/masses[None, :, None])
    new_projection = np.sum(updated*nac/masses[None, :, None])
    kinetic_change = np.sum((updated**2-p**2)/(2*masses[None, :, None]))
    if scale < .01:
        assert event.frustrated and not event.accepted
        np.testing.assert_allclose(new_projection, -old_projection, rtol=0., atol=2e-10)
        np.testing.assert_allclose(kinetic_change, 0., rtol=0., atol=2e-13)
    else:
        assert event.accepted and not event.frustrated
        assert old_projection*new_projection > 0.
        np.testing.assert_allclose(kinetic_change+gap, 0., rtol=0., atol=3e-13)
    # A bead-dependent direction must change internal momenta. This detects
    # accidental reuse of CA's identical impulse on every bead.
    assert np.max(np.abs(change-change.mean(axis=0))) > 1e-4


@pytest.mark.parametrize("beads,source,momentum,z", [
    (1, 0, 2., -.2), (2, 0, 2., -.2), (8, 0, 2., -.2),
    (4, 0, .4, -.05), (3, 1, 2., .2),
])
def test_resolved_events_match_independent_piecewise_solution(beads, source, momentum, z):
    # Reverse Sx for the downward case so the crossing is incoming.
    sx = (1-2*source)*np.sqrt(1-z*z)
    spin = np.array([sx, 0., z])
    q, p = np.zeros((beads, 1, 3)), np.zeros((beads, 1, 3))
    p[:, 0, 0] = momentum
    state = RPMASHBAState(q, p, spin=spin, active_surface=source)
    driver = RPMASHBADynamics(RotatingSurfaces(), [1.], 3., beads, .2,
                             rtol=1e-11, atol=1e-13)
    initial_energy = driver.energy(state)
    event_time = brentq(lambda t: rotation(spin, [0., -momentum, .4], t)[2], 0., .2)
    event_spin = rotation(spin, [0., -momentum, .4], event_time)
    gap = .4*(1-2*source)
    accepted = momentum**2 >= 2*gap
    outgoing = np.sqrt(momentum**2-2*gap) if accepted else -momentum
    expected_spin = rotation(event_spin, [0., -outgoing, .4], .2-event_time)
    expected_q = momentum*event_time + outgoing*(.2-event_time)
    driver.step(state)
    assert len(driver.last_hops) == 1
    event = driver.last_hops[0]
    assert event.accepted is accepted and event.frustrated is (not accepted)
    assert state.active_surface == (1-source if accepted else source)
    np.testing.assert_allclose(event.time, event_time, rtol=0., atol=2e-11)
    np.testing.assert_allclose(event.ring_energy_gap, beads*gap, rtol=0., atol=3e-14)
    np.testing.assert_allclose(state.q[:, 0, 0], expected_q, rtol=0., atol=3e-11)
    np.testing.assert_allclose(state.p[:, 0, 0], outgoing, rtol=0., atol=3e-11)
    np.testing.assert_allclose(state.spin, expected_spin, rtol=0., atol=3e-11)
    np.testing.assert_allclose(driver.energy(state), initial_energy, rtol=0., atol=3e-11)


@pytest.mark.parametrize("momentum,z", [(2., -.2), (.4, -.05)])
def test_one_bead_matches_centroid_mash_after_basis_transformation(momentum, z):
    q = np.array([[[.37, 0., 0.]]])
    p = np.array([[[momentum, 0., 0.]]])
    spin = np.array([np.sqrt(1-z*z), 0., z])
    ca = RPMASHDynamics(RotatingSurfaces(), [1.], 3., 1, .2, rtol=1e-11, atol=1e-13)
    ba = RPMASHBADynamics(RotatingSurfaces(), [1.], 3., 1, .2, rtol=1e-11, atol=1e-13)
    ca_state = RPMASHState(q, p, spin=rotation(spin, [0., 1., 0.], q[0, 0, 0]))
    ba_state = RPMASHBAState(q, p, spin=spin)
    ca.step(ca_state)
    ba.step(ba_state)
    assert len(ca.last_hops) == len(ba.last_hops) == 1
    assert ca.last_hop.accepted == ba.last_hop.accepted
    assert ca_state.active_surface == ba_state.active_surface
    np.testing.assert_allclose(ca.last_hop.time, ba.last_hop.time, rtol=0., atol=3e-11)
    np.testing.assert_allclose(ca_state.q, ba_state.q, rtol=0., atol=3e-11)
    np.testing.assert_allclose(ca_state.p, ba_state.p, rtol=0., atol=3e-11)
    expected_fixed = rotation(ba_state.spin, [0., 1., 0.], ba_state.q[0, 0, 0])
    np.testing.assert_allclose(ca_state.spin, expected_fixed, rtol=0., atol=3e-11)


def fourier_adiabatic_population(gap, duration, *, points=1024, dt=.1):
    """Propagate a genuine finite-mass nuclear packet and project locally."""
    mass, momentum, width, center = 1000., 100., .2, -1.
    grid = np.linspace(-6., 6., points, endpoint=False)
    dx = grid[1]-grid[0]
    q = np.zeros((points, 1, 3))
    q[:, 0, 0] = grid
    model = RotatingSurfaces(gap)
    # Smooth local upper/lower eigenvectors in the fixed diabatic basis.
    upper = np.column_stack((np.cos(grid/2), np.sin(grid/2)))
    lower = np.column_stack((-np.sin(grid/2), np.cos(grid/2)))
    packet = np.exp(-(grid-center)**2/(4*width**2) + 1j*momentum*(grid-center))
    psi = packet[:, None]*lower
    psi /= np.sqrt(dx*np.sum(abs(psi)**2))
    steps = int(np.ceil(duration/dt))
    reference = SplitOperator1D(grid, model.evaluate(q).matrix, mass, duration/steps)
    result = reference.propagate(psi, steps, sample_stride=steps)
    final = result.final_wavefunctions[0]
    population = dx*np.sum(abs(np.einsum("xi,xi->x", upper, final))**2)
    return float(population), result


def test_nontrivial_population_matches_fourier_in_small_gap_high_kinetic_limit():
    # At zero gap a pi/2 adiabatic-frame rotation transfers exactly half the
    # population for any momentum distribution symmetric about this velocity.
    # A positive, resolved gap avoids using a degenerate driver. Its finite
    # effect and the nuclear back reaction are much smaller than the bound.
    gap, mass, momentum = 1e-5, 1000., 100.
    duration = np.pi/(2*(momentum/mass))
    exact, reference = fourier_adiabatic_population(gap, duration)
    refined, refined_reference = fourier_adiabatic_population(gap, duration, points=2048, dt=.05)
    np.testing.assert_allclose(exact, refined, rtol=0., atol=2e-8)
    assert .49 < refined < .51
    assert np.max(refined_reference.edge_population) < 1e-15
    np.testing.assert_allclose(reference.norm, 1., rtol=0., atol=2e-12)

    beads = 4
    q, p = np.zeros((beads, 1, 3)), np.zeros((beads, 1, 3))
    q[:, 0, 0] = -1. + np.array([-.12, .03, .05, .04])
    p[:, 0, 0] = momentum + np.array([-1., .2, .3, .5])
    driver = RPMASHBADynamics(RotatingSurfaces(gap), [mass], 100., beads, duration,
                             rtol=1e-10, atol=1e-12, max_step=duration/16)
    values = []
    # Midpoints integrate the population importance measure: |Sz|=sqrt(u),
    # phi uniform. The symmetric azimuth grid avoids equatorial endpoints.
    for u in (.25, .75):
        z = -np.sqrt(u)
        for phi in 2*np.pi*(np.arange(8)+.5)/8:
            spin = [np.sqrt(1-u)*np.cos(phi), np.sqrt(1-u)*np.sin(phi), z]
            state = RPMASHBAState(q, p, spin=spin, sampling="population")
            driver.step(state)
            values.append(driver.populations(state)[1])
    np.testing.assert_allclose(np.mean(values), refined, rtol=0., atol=2e-6)
