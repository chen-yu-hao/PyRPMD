"""Independent derivative and saddle-optimizer checks for the T09 workflow."""

import numpy as np
import pytest

from pyrpmd.constants import KB_AU
from pyrpmd.instanton import (
    InstantonError, RingPolymerAction, _bofill_update, _physical_hessians,
    _phase_projected_spectrum,
    crossover_temperature, instanton_seed, optimize_instanton,
)
from pyrpmd.potentials.base import PESResult


class CurvedBarrier:
    """Analytic three-dimensional quartic with a curved transverse valley."""

    def __init__(self, bend=0.5, slope=0.3):
        self.bend, self.slope = bend, slope
        self.calls = 0
        self.frames = 0

    def evaluate(self, q):
        self.calls += 1
        self.frames += len(q)
        x, y, z = q[:, 0, :].T
        u, v = y-self.bend*x*x, z-self.slope*x
        f = np.zeros_like(q)
        f[:, 0, 0] = -0.04*x*(x*x-1)+0.06*self.bend*x*u+0.02*self.slope*v
        f[:, 0, 1], f[:, 0, 2] = -0.03*u, -0.02*v
        return PESResult(0.01*(x*x-1)**2+0.015*u*u+0.01*v*v, f)


class AnalyticCurvedBarrier(CurvedBarrier):
    def hessian(self, q):
        x, y, _ = q[:, 0, :].T
        u = y-self.bend*x*x
        h = np.zeros((len(q), 3, 3), dtype=np.float64)
        h[:, 0, 0] = (0.04*(3*x*x-1)-0.06*self.bend*u
                       +0.12*self.bend**2*x*x+0.02*self.slope**2)
        h[:, 0, 1] = h[:, 1, 0] = -0.06*self.bend*x
        h[:, 0, 2] = h[:, 2, 0] = -0.02*self.slope
        h[:, 1, 1], h[:, 2, 2] = 0.03, 0.02
        return h


def curved_setup(p=32):
    masses = np.array([1822.])
    potential = CurvedBarrier()
    tc = crossover_temperature(potential, np.zeros((1, 3)), masses)
    path = instanton_seed(np.zeros((1, 3)), [[1., 0., 0.3]], p, 0.9)
    path[:, 0, 1] = potential.bend*path[:, 0, 0]**2
    return potential, masses, 0.6*tc, path


def test_analytic_provider_matches_force_differences_in_full_and_active_space():
    rng = np.random.default_rng(119)
    q = rng.normal(scale=0.4, size=(8, 1, 3))
    analytic, numerical = AnalyticCurvedBarrier(), CurvedBarrier()
    for mask in (np.ones((1, 3), dtype=bool), np.array([[True, False, True]])):
        observed = _physical_hessians(analytic, q, mask, 1e-4)
        reference = _physical_hessians(numerical, q, mask, 1e-5)
        np.testing.assert_allclose(observed, reference, rtol=2e-8, atol=2e-10)
    assert analytic.calls == 0
    a = RingPolymerAction(analytic, [1822.], 80.)
    b = RingPolymerAction(numerical, [1822.], 80., hessian_step_bohr=1e-5)
    np.testing.assert_allclose(a.hessian(q), b.hessian(q), atol=1e-7)


def test_cartesian_five_dimensional_hessian_layout_and_reused_buffer():
    class TensorProvider(AnalyticCurvedBarrier):
        def hessian(self, q):
            h = super().hessian(q)
            if not hasattr(self, "buffer"):
                self.buffer = np.empty_like(h)
            self.buffer[:] = h
            return self.buffer.reshape(len(q), 1, 3, 1, 3)

    pes, masses, temperature, path = curved_setup()
    result = optimize_instanton(TensorProvider(), masses, temperature, path,
                                method="bofill", retain_hessian=True)
    assert result.converged
    np.testing.assert_allclose(result.action_hessian,
                               RingPolymerAction(AnalyticCurvedBarrier(), masses, temperature).hessian(result.path_bohr),
                               atol=1e-12)
    assert result.optimizer_diagnostics["finite_difference_hessian_evaluations"] == 0


def test_analytic_hessian_slices_multiple_atoms_in_cartesian_boolean_order():
    rng = np.random.default_rng(281)
    a = rng.normal(size=(6, 6))
    matrix = a.T @ a

    class Quadratic:
        def evaluate(self, q):
            flat = q.reshape(len(q), 6)
            gradient = flat @ matrix
            return PESResult(0.5*np.sum(flat*gradient, axis=1), -gradient.reshape(q.shape))

    class AnalyticQuadratic(Quadratic):
        def hessian(self, q):
            return np.broadcast_to(matrix, (len(q), 6, 6))

    q = rng.normal(size=(8, 2, 3))
    mask = np.array([[False, True, True], [True, False, True]])
    analytic = _physical_hessians(AnalyticQuadratic(), q, mask, 1e-5)
    numerical = _physical_hessians(Quadratic(), q, mask, 1e-5)
    np.testing.assert_allclose(analytic, numerical, atol=2e-9)


@pytest.mark.parametrize("problem", ["dtype", "shape", "nonsymmetric", "nonfinite", "noncallable", "exception"])
def test_invalid_analytic_provider_is_never_silently_replaced(problem):
    class BadProvider(AnalyticCurvedBarrier):
        def hessian(self, q):
            if problem == "exception":
                raise RuntimeError("provider unavailable")
            h = super().hessian(q)
            if problem == "dtype":
                return h.astype(np.float32)
            if problem == "shape":
                return h[0]
            if problem == "nonsymmetric":
                h[:, 0, 1] += 0.1
            if problem == "nonfinite":
                h[:, 0, 0] = np.nan
            return h

    pes = BadProvider()
    if problem == "noncallable":
        pes.hessian = None
    expected = RuntimeError if problem == "exception" else ValueError
    with pytest.raises(expected):
        RingPolymerAction(pes, [1822.], 80.).hessian(np.zeros((8, 1, 3)))
    assert pes.calls == 0


def test_bofill_update_is_symmetric_and_satisfies_independent_secant_equation():
    h = np.array([[-2., 0.5, 0.2], [0.5, 3., -0.1], [0.2, -0.1, 1.]])
    displacement = np.array([0.3, -0.1, 0.25])
    change = np.array([-0.7, 0.2, 0.4])
    updated, changed = _bofill_update(h, displacement, change)
    assert changed
    np.testing.assert_allclose(updated, updated.T, atol=1e-15)
    np.testing.assert_allclose(updated @ displacement, change, atol=1e-15)
    # Orthogonal s and secant error must use the finite PSB limit.
    s = np.array([1., 0., 0.])
    y = h @ s + np.array([0., 1., 0.])
    updated, _ = _bofill_update(h, s, y)
    np.testing.assert_allclose(updated @ s, y, atol=1e-15)
    assert np.isfinite(updated).all()


def test_phase_householder_spectrum_matches_independent_qr_projection():
    rng = np.random.default_rng(612)
    a = rng.normal(size=(11, 11))
    hessian = 0.5*(a+a.T)
    tangent = rng.normal(size=11)
    complement = np.linalg.qr(tangent[:, None], mode="complete")[0][:, 1:]
    expected = np.linalg.eigvalsh(complement.T @ hessian @ complement)
    np.testing.assert_allclose(_phase_projected_spectrum(hessian, tangent), expected, atol=3e-14)


def test_molecular_negative_discrete_phase_is_removed_only_when_resolved():
    class RadialAsymmetricBarrier:
        """Translation/rotation invariant Eckart barrier in a bond distance."""

        def evaluate(self, q):
            vector = q[:, 1]-q[:, 0]
            radius = np.linalg.norm(vector, axis=1)
            t = np.tanh(0.5*(radius-10.))
            radial_force = 0.5*(1-t*t)*(0.02*t-0.003)
            force = np.zeros_like(q)
            force[:, 1] = radial_force[:, None]*vector/radius[:, None]
            force[:, 0] = -force[:, 1]
            return PESResult(0.01*(1-t*t)+0.003*t, force)

    masses = [3644., 3644.]  # Bond reduced mass = 1822 electron masses.
    radius_ts = 10.+2*np.arctanh(0.15)
    ts = np.zeros((2, 3))
    ts[:, 0] = [-radius_ts/2, radius_ts/2]
    potential = RadialAsymmetricBarrier()
    tc = crossover_temperature(potential, ts, masses)
    radius = radius_ts+2.5*np.cos(2*np.pi*np.arange(10)/10)
    seed = np.zeros((10, 2, 3))
    seed[:, 0, 0], seed[:, 1, 0] = -radius/2, radius/2
    result = optimize_instanton(potential, masses, 0.9*tc, seed, method="bofill")
    diag = result.optimizer_diagnostics
    assert result.converged and result.negative_modes == 1
    assert diag["raw_negative_modes"] == 2
    assert diag["raw_near_zero_modes"] == 5
    assert diag["phase_projection_applied"]
    assert diag["projected_negative_modes"] == 1
    assert 0.01 < result.phase_mode_relative_residual < 0.05
    assert np.count_nonzero(result.hessian_eigenvalues < -1e-5*abs(result.hessian_eigenvalues[0])) == 2
    # Tightening the phase guard must reject this same stationary finite grid.
    with pytest.raises(InstantonError, match="negative mode"):
        optimize_instanton(potential, masses, 0.9*tc, result.path_bohr,
                           method="bofill", phase_tolerance=0.01)
    # An already stationary path needs only its final true Hessian.
    warm = optimize_instanton(potential, masses, 0.9*tc, result.path_bohr, method="bofill")
    assert warm.optimizer_diagnostics["iterations"] == 0
    assert warm.optimizer_diagnostics["physical_hessian_evaluations"] == 1


def test_phase_projection_cannot_hide_a_second_physical_instability():
    class UnstableBath:
        def evaluate(self, q):
            x, y = q[:, 0, 0], q[:, 0, 1]
            t = np.tanh(0.5*x)
            force = np.zeros_like(q)
            force[:, 0, 0] = 0.01*(1-t*t)*t
            force[:, 0, 1] = 1e-5*y
            return PESResult(0.01*(1-t*t)-0.5e-5*y*y, force)

    potential, masses = UnstableBath(), [1822.]
    mask = np.array([[True, True, False]])
    # The independent inverted bath deliberately makes the TS second order;
    # omit transition_state and let the final orbit index check detect it.
    temperature = 0.5*0.5*np.sqrt(0.02/masses[0])/(2*np.pi*KB_AU)
    seed = instanton_seed(np.zeros((1, 3)), [[1., 0., 0.]], 64, 2.5)
    result = optimize_instanton(potential, masses, temperature, seed, active_mask=mask,
                                method="bofill", raise_on_failure=False)
    assert result.gradient_max < 1e-8
    assert result.phase_mode_relative_residual < 0.05
    assert result.optimizer_diagnostics["phase_projection_applied"]
    assert result.optimizer_diagnostics["projected_negative_modes"] == 2
    assert result.negative_modes == 2 and not result.converged


def test_curved_three_dimensional_saddle_agrees_with_root_and_saves_force_calls():
    pes, masses, temperature, path = curved_setup(64)
    legacy = optimize_instanton(pes, masses, temperature, path, method="root")
    new_pes = CurvedBarrier()
    updated = optimize_instanton(new_pes, masses, temperature, path,
                                 method="bofill", retain_hessian=True)
    assert updated.converged and legacy.converged
    assert updated.gradient_max <= 1e-8 and legacy.gradient_max <= 1e-8
    assert updated.negative_modes == legacy.negative_modes == 1
    assert updated.action_hbar == pytest.approx(legacy.action_hbar, abs=1e-10)
    np.testing.assert_allclose(updated.path_bohr, legacy.path_bohr, atol=1e-7)
    assert updated.optimizer_diagnostics["force_calls"] == new_pes.calls
    assert updated.optimizer_diagnostics["force_configurations"] == new_pes.frames
    assert updated.optimizer_diagnostics["force_calls"] < legacy.optimizer_diagnostics["force_calls"]
    assert updated.optimizer_diagnostics["hessian_updates"] > 0
    assert updated.to_dict()["physical_converged"] is False
    # The accepted path follows a curved valley, not a frozen straight line.
    assert np.ptp(updated.path_bohr[:, 0, 1]) > 0.1


def test_analytic_optimizer_avoids_all_force_difference_calls():
    _, masses, temperature, path = curved_setup()
    analytic = optimize_instanton(AnalyticCurvedBarrier(), masses, temperature, path,
                                  method="bofill")
    numerical = optimize_instanton(CurvedBarrier(), masses, temperature, path,
                                   method="bofill")
    assert analytic.converged
    assert analytic.action_hbar == pytest.approx(numerical.action_hbar, abs=1e-10)
    report = analytic.optimizer_diagnostics
    assert report["hessian_provider_calls"] == report["physical_hessian_evaluations"]
    assert report["finite_difference_hessian_evaluations"] == 0
    assert report["force_calls"] < numerical.optimizer_diagnostics["force_calls"]


def test_scheduled_refreshes_and_failed_optimization_keep_honest_diagnostics():
    pes, masses, temperature, path = curved_setup()
    refreshed = optimize_instanton(pes, masses, temperature, path, method="bofill",
                                   hessian_update_interval=2)
    assert refreshed.converged
    assert refreshed.optimizer_diagnostics["hessian_refreshes"] > 0
    failed = optimize_instanton(pes, masses, temperature, path, method="bofill",
                                max_iterations=1, raise_on_failure=False)
    assert failed.converged is False
    assert failed.optimizer_diagnostics["termination"] == "evaluation budget"
    assert failed.gradient_max > 1e-8
    with pytest.raises(InstantonError, match="residual"):
        optimize_instanton(pes, masses, temperature, path, method="bofill", max_iterations=1)


@pytest.mark.parametrize("bead_reflection", [False, True])
def test_bofill_asymmetric_orbit_is_not_centroid_pinned(bead_reflection):
    class AsymmetricEckart:
        def evaluate(self, q):
            t = np.tanh(0.5*q[:, 0, 0])
            f = np.zeros_like(q)
            f[:, 0, 0] = 0.5*(1-t*t)*(0.02*t-0.003)
            return PESResult(0.01*(1-t*t)+0.003*t, f)

    ts = np.array([[2*np.arctanh(0.15), 0., 0.]])
    mask = np.array([[True, False, False]])
    masses = [1822.]
    potential = AsymmetricEckart()
    tc = crossover_temperature(potential, ts, masses, active_mask=mask)
    seed = instanton_seed(ts, [[1., 0., 0.]], 64, 2.)
    if bead_reflection:
        seed[:, 0, 0] = ts[0, 0]+2*np.cos(2*np.pi*np.arange(64)/64)
    result = optimize_instanton(potential, masses, 0.65*tc, seed,
                                active_mask=mask, transition_state=ts, method="bofill")
    assert result.converged
    assert abs(np.mean(result.path_bohr[:, 0, 0])-ts[0, 0]) > 0.1
    assert result.optimizer_diagnostics["unique_beads"] == 32+int(bead_reflection)
    reference = optimize_instanton(potential, masses, 0.65*tc, seed,
                                   active_mask=mask, method="root")
    assert result.action_hbar == pytest.approx(reference.action_hbar, abs=1e-10)


@pytest.mark.parametrize("options", [
    {"method": "lbfgs"}, {"method": "bofill", "hessian_update_interval": True},
    {"method": "bofill", "hessian_update_interval": -1},
    {"method": "bofill", "hessian_update_interval": 1.5},
    {"method": "bofill", "trust_radius_bohr": 0.},
    {"method": "bofill", "trust_radius_bohr": np.nan},
    {"method": "root", "hessian_update_interval": 2},
    {"method": "root", "trust_radius_bohr": 0.5},
    {"phase_tolerance": 0.}, {"phase_tolerance": np.nan},
    {"raise_on_failure": 1},
])
def test_optimizer_option_validation(options):
    pes, masses, temperature, path = curved_setup()
    with pytest.raises(ValueError):
        optimize_instanton(pes, masses, temperature, path, **options)
