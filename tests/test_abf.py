"""Independent geometry, quadrature and force-contract checks for ABF/eABF."""

import copy
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from pyrpmd.abf import ABFBias, ExtendedABF, ForceGrid, MeanForceEstimator, integrate_mean_force
from pyrpmd.analysis.fes_result import FESResult, write_fes_result
from pyrpmd.cv import Cartesian, CollectiveVariable, COMDistance, CVStack, Distance


@pytest.mark.parametrize("beads", [1, 4, 16])
def test_cartesian_uses_mean_unscaled_physical_forces(beads):
    q = np.random.default_rng(81).normal(size=(beads, 2, 3))
    cv = CVStack([Cartesian(0, "x"), Cartesian(1, "y")])
    value, gradient = MeanForceEstimator(cv, [2, 5], 3).evaluate(q, -2.4 * q)
    np.testing.assert_allclose(gradient, 2.4 * value, atol=2e-15)


@pytest.mark.parametrize("beads", [1, 8])
@pytest.mark.parametrize("kind", ["distance", "com"])
def test_radial_jacobian_matches_exact_spherical_integral(beads, kind):
    # The relative vector has V=k*r^2/2, hence A=k*r^2/2-2*log(r)/beta.
    masses = [2., 3.]
    cv = Distance([0, 1]) if kind == "distance" else COMDistance([[0], [1]], masses)
    centroid = np.array([[0., 0., 0.], [1.2, .3, -.2]])
    q = np.repeat(centroid[None], beads, axis=0)
    offsets = np.random.default_rng(7).normal(scale=.05, size=q.shape)
    q += offsets - offsets.mean(axis=0)
    delta = q[:, 1] - q[:, 0]
    forces = np.stack((1.7 * delta, -1.7 * delta), axis=1)
    radius = np.linalg.norm(centroid[1])
    exact = 1.7 * radius - 2 / (2.3 * radius)
    analytic = MeanForceEstimator(cv, masses, 2.3).evaluate(q, forces)[1]
    numerical = MeanForceEstimator(cv, masses, 2.3, derivative="finite_difference").evaluate(q, forces)[1]
    np.testing.assert_allclose(analytic, exact, atol=1e-13)
    np.testing.assert_allclose(numerical, exact, atol=5e-10)


class SquaredX(CollectiveVariable):
    def centroid_value_gradient(self, q):
        gradient = np.zeros_like(q)
        gradient[0, 0] = 2 * q[0, 0]
        return q[0, 0] ** 2, gradient


class SineX(CollectiveVariable):
    def centroid_value_gradient(self, q):
        gradient = np.zeros_like(q)
        gradient[0, 0] = 20 * np.cos(20 * q[0, 0])
        return np.sin(20 * q[0, 0]), gradient


class ScaledY(CollectiveVariable):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def centroid_value_gradient(self, q):
        gradient = np.zeros_like(q)
        gradient[0, 1] = self.scale
        return self.scale * q[0, 1], gradient


@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("second", [Cartesian(0), SineX()], ids=["duplicate", "curved-dependent"])
@pytest.mark.parametrize("operation", ["energy_force", "observe"])
def test_eabf_rejects_rank_deficient_cvs_before_changing_history(beads, second, operation):
    # A curved rank-one image may occupy neighboring bins in both axes; its
    # histogram stencil does not establish the existence of a 2D CV density.
    cv = CVStack([Cartesian(0), second])
    extended = ExtendedABF(cv, [np.linspace(-1, 1, 10)] * 2, 2., [.1, -.2],
                           momentum=[.3, -.4], coupling=3., mass=5., friction=.2)
    before = copy.deepcopy(extended.state_dict())
    q = np.broadcast_to([.21, -.3, .4], (beads, 1, 3)).copy()
    coordinates = q.copy()
    with pytest.raises(ValueError, match="rank|singular|metric"):
        getattr(extended, operation)(q)
    assert extended.state_dict() == before
    np.testing.assert_array_equal(q, coordinates)


@pytest.mark.parametrize("beads", [1, 4])
@pytest.mark.parametrize("scale", [1e-15, 1e15])
def test_eabf_full_rank_check_is_independent_of_cv_scale(beads, scale):
    # Changing the second CV's units by 30 orders of magnitude across cases
    # must preserve both its independent direction and the physical coupling.
    cv = CVStack([Cartesian(0), ScaledY(scale)])
    edges = [np.linspace(-1, 1, 5), scale * np.linspace(-1, 1, 5)]
    extended = ExtendedABF(cv, edges, 2., [.1, scale * .2],
                           coupling=[2., 3. / scale**2], mass=[5., 7. / scale**2], friction=.2)
    q = np.broadcast_to([.3, -.4, .2], (beads, 1, 3)).copy()
    energy, force = extended.energy_force(q)
    assert energy == pytest.approx(beads * .5 * (2. * .2**2 + 3. * .6**2))
    np.testing.assert_allclose(force, np.broadcast_to([-.4, 1.8, 0.], q.shape), atol=1e-14)
    extended.observe(q)
    assert extended.bias.observations == 1
    assert extended.bias.grid.count.sum() == 1
    assert extended.physical_grid.count.sum() == 1


def test_two_dimensional_nonlinear_metric_includes_log_jacobian():
    cv = CVStack([SquaredX(), Cartesian(0, "y")])
    q = np.array([[[1.4, -.3, 0.]], [[1.2, .1, 0.]]])
    value, derivative = MeanForceEstimator(cv, [4.2], 2., derivative="finite_difference").evaluate(q, -q)
    # rho(s,t) proportional to exp[-beta*(s+t^2)/2]/sqrt(s).
    np.testing.assert_allclose(derivative, [.5 + 1 / (4 * value[0]), value[1]], rtol=2e-9)
    with pytest.raises(ValueError, match="explicitly"):
        MeanForceEstimator(cv, [4.2], 2.)


def test_singular_and_noncentroid_combinations_rejected():
    with pytest.raises(ValueError, match="centroid"):
        ABFBias(Cartesian(0, estimator="bead_average"), [[-1, 0, 1]], 1)
    estimator = MeanForceEstimator(CVStack([Cartesian(0), Cartesian(0)]), [1], 1)
    with pytest.raises(ValueError, match="metric"):
        estimator.evaluate(np.ones((2, 1, 3)), np.ones((2, 1, 3)))
    cv = Cartesian(0)
    cv.centroid_value_gradient = lambda q: (q[0, 0] ** 2, 2 * q)
    with pytest.raises(ValueError, match="unmodified"):
        MeanForceEstimator(cv, [1], 1)


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("beads", [1, 7])
def test_projected_bias_force_is_negative_energy_derivative(dimension, beads):
    cv = CVStack([Cartesian(0, d) for d in range(dimension)])
    edges = [np.linspace(-1, 1, 15)] * dimension
    bias = ABFBias(cv, edges, 1, full_samples=1, wall_stiffness=3)
    centers = [(e[1:] + e[:-1]) / 2 for e in edges]
    for index in np.ndindex(bias.grid.shape):
        point = np.array([centers[d][index[d]] for d in range(dimension)])
        bias.observe(point, point)
    bias.refresh()
    for xyz in ([.14, -.23, 0], [1.25, -.12, 0]):
        q = np.broadcast_to(xyz, (beads, 1, 3)).copy()
        force = bias.energy_force(q)[1]
        numerical = np.zeros_like(q)
        for index in np.ndindex(q.shape):
            plus, minus = q.copy(), q.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical[index] = -(bias.energy_force(plus)[0] - bias.energy_force(minus)[0]) / 2e-6
        np.testing.assert_allclose(force, numerical, atol=1e-8)
    old = copy.deepcopy(bias.state_dict())
    bias.energy_force(q)
    assert bias.state_dict() == old


def test_two_dimensional_projection_matches_nonseparable_exact_surface():
    edges = [np.linspace(-1, 1, 21), np.linspace(-.5, 1.5, 18)]
    x, y = np.meshgrid(*[(e[1:] + e[:-1]) / 2 for e in edges], indexing="ij")
    gradient = np.stack((2*x + .4*y, 3*y + .4*x), axis=-1)
    free, mask, diagnostics = integrate_mean_force(edges, gradient)
    exact = x*x + 1.5*y*y + .4*x*y
    np.testing.assert_allclose(free, exact - exact[0, 0], atol=3e-10)
    assert not mask.any()
    assert diagnostics["edge_residual_rms"] < 1e-10


def test_disconnected_support_never_invents_relative_offsets():
    edges = [np.linspace(-1, 1, 11)]
    count = np.array([5, 4, 0, 0, 2, 2, 2, 2, 0, 3])
    _, mask, diagnostics = integrate_mean_force(edges, np.ones((10, 1)), counts=count)
    np.testing.assert_array_equal(~mask, [False]*4 + [True]*4 + [False]*2)
    assert diagnostics["components"] == 3
    assert diagnostics["discarded_bins"] == 3


def test_double_well_abf_matches_exact_bin_quadrature():
    edges = np.linspace(-1.5, 1.5, 251)
    centers = (edges[1:] + edges[:-1]) / 2
    bias = ABFBias(Cartesian(0), [edges], 2., full_samples=1)
    for x in centers:
        bias.observe([x], [4*x*(x*x-1)])
    result = bias.fes_result()
    nodes, weights = leggauss(16)
    points = centers[:, None] + .5*np.diff(edges)[:, None]*nodes
    density = np.exp(-2 * (points*points-1)**2) @ weights / 2
    exact = -.5*np.log(density)
    exact -= exact[0]
    np.testing.assert_allclose(result.free_energy, exact, atol=.002)
    assert np.isnan(result.uncertainty).all()


def test_abf_double_well_agrees_with_existing_umbrella_mbar_within_error():
    from pyrpmd.analysis.fes import reconstruct_fes

    beta = 2.
    rng = np.random.default_rng(9871)
    grid = np.linspace(-2., 2., 40001)
    centers = np.array([[-1.], [0.], [1.]])
    samples = []
    for center in centers[:, 0]:
        density = np.exp(-beta*((grid*grid-1)**2 + 1.5*(grid-center)**2))
        density /= density.sum()
        samples.append(rng.choice(grid, size=12000, p=density)[:, None])
    mbar = reconstruct_fes(samples, centers, np.array([[3.]]), beta,
                           bins=30, ranges=[[-1.5, 1.5]], bootstrap=24, seed=418)
    edges = mbar["edges"][0]
    x = (edges[1:] + edges[:-1]) / 2
    bias = ABFBias(Cartesian(0), [edges], beta)
    for point in x:
        bias.observe([point], [4*point*(point*point-1)])
    abf = bias.fes_result().free_energy
    reference = np.nanargmin(mbar["free_energy"])
    difference = (abf-abf[reference]) - (mbar["free_energy"]-mbar["free_energy"][reference])
    # The separate O(h^2) allowance is integration/bin averaging error, not a
    # fabricated sampling SEM for the force oracle.
    supported = (~mbar["mask"]) & np.isfinite(mbar["standard_error"])
    assert supported.sum() >= 25
    assert np.all(np.abs(difference[supported]) < 4*mbar["standard_error"][supported] + .08)


def _muller_brown(x, y):
    amplitude = np.array([-200., -100., -170., 15.]) * .01
    a, b, c = np.array([-1., -1., -6.5, .7]), np.array([0., 0., 11., .6]), np.array([-10., -10., -6.5, .7])
    dx, dy = x[..., None] - [1., 0., -.5, -1.], y[..., None] - [0., .5, 1.5, 1.]
    terms = amplitude * np.exp(a*dx*dx + b*dx*dy + c*dy*dy)
    energy = terms.sum(-1)
    gradient = np.stack(((terms*(2*a*dx+b*dy)).sum(-1), (terms*(b*dx+2*c*dy)).sum(-1)), -1)
    return energy, gradient


def test_muller_brown_projection_matches_exact_numerical_integral():
    edges = [np.linspace(-1.25, .75, 71), np.linspace(0., 1.8, 64)]
    x, y = np.meshgrid(*[(e[1:] + e[:-1])/2 for e in edges], indexing="ij")
    _, gradient = _muller_brown(x, y)
    free, mask, _ = integrate_mean_force(edges, gradient)
    nodes, weights = leggauss(5)
    density = np.zeros(x.shape)
    for i, nx in enumerate(nodes):
        for j, ny in enumerate(nodes):
            energy = _muller_brown(x + nx*np.diff(edges[0])[0]/2, y + ny*np.diff(edges[1])[0]/2)[0]
            density += weights[i]*weights[j]*np.exp(-energy)/4
    exact = -np.log(density)
    difference = free - exact
    difference -= difference.mean()
    assert not mask.any()
    assert np.sqrt(np.mean(difference**2)) < .008
    assert np.max(np.abs(difference)) < .03


@pytest.mark.parametrize("dimension", [1, 2])
def test_czar_removes_auxiliary_bias_and_recovers_exact_gaussian(dimension, tmp_path):
    cv = CVStack([Cartesian(0, d) for d in range(dimension)])
    edges = [np.linspace(-1, 1, 61)] * dimension
    extended = ExtendedABF(cv, edges, 2., np.zeros(dimension), coupling=4., mass=3., friction=.1)
    axes = [(e[1:] + e[:-1])/2 for e in edges]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    stiffness = np.array([[1.3]]) if dimension == 1 else np.array([[1.3, .25], [.25, .8]])
    tilt = np.arange(1, dimension+1) * .2
    exact = .5*np.einsum("...i,ij,...j->...", grid, stiffness, grid)
    # Integrating lambda analytically gives rho_tilde(z) = exp[-beta*(V(z)-b.z)].
    density = np.exp(-2*(exact-grid @ tilt))
    extended.physical_grid.count[:] = np.rint(density * 1e12).astype(np.int64)
    extended.physical_grid.mean[:] = tilt
    result = extended.fes_result()
    difference = result.free_energy[~result.mask] - exact[~result.mask]
    np.testing.assert_allclose(difference, difference[0], atol=2e-10)
    assert result.mask[(0,)*dimension]
    write_fes_result(tmp_path, result)
    restored = FESResult.read(tmp_path)
    assert restored.method == "eabf"
    assert restored.metadata["estimator"] == "czar"


def test_czar_retains_radial_entropy_and_masks_empty_stencils():
    edges = [np.linspace(.5, 2, 101)]
    extended = ExtendedABF(Distance([0,1]), edges, 2., [1.], coupling=3., mass=1., friction=1.)
    r = (edges[0][1:] + edges[0][:-1])/2
    density = r*r*np.exp(-2*(.5*r*r-.2*r))
    extended.physical_grid.count[:] = np.rint(density*1e12).astype(int)
    extended.physical_grid.mean[:] = .2
    result = extended.fes_result()
    np.testing.assert_allclose(result.extras["mean_force"][1:-1,0], r[1:-1]-1/r[1:-1], atol=.0006)
    extended.physical_grid.count[40] = 0
    result = extended.fes_result()
    assert result.mask[39:42].all()


def test_statistics_roundtrip_and_invalid_state_rejected():
    grid = ForceGrid([[-1, 0, 1]])
    for force in (1., 2., 3.):
        grid.update([-.2], [force])
    grid.update([2.], [9.])
    np.testing.assert_allclose(grid.variance[0], [1.])
    restored = ForceGrid(grid.edges)
    restored.load_state_dict(grid.state_dict())
    assert restored.state_dict() == grid.state_dict()
    broken = grid.state_dict()
    broken["count"][0] = 0
    with pytest.raises(ValueError, match="statistics"):
        restored.load_state_dict(broken)


@pytest.mark.parametrize("options", [{"full_samples": True}, {"wall_stiffness": False}, {"beta": 0.}, {"full_samples": 1.2}])
def test_invalid_bias_controls(options):
    with pytest.raises((ValueError, TypeError)):
        ABFBias(Cartesian(0), [[-1, 0, 1]], **{"beta": 1., **options})
