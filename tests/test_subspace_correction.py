"""Exact 1-3D subspace rates, matched RPMD rates and the correction factor."""

import json

import numpy as np
import pytest
from scipy.integrate import quad

from pyrpmd.analysis.subspace_correction import (
    SubspaceCorrection,
    apply_subspace_correction,
    correct_rate_run,
    nested_subspace_sensitivity,
    subspace_correction,
)
from pyrpmd.analysis.subspace_model import (
    curvature_width_criterion,
    model_from_function,
    path_curvature,
    project_linear_subspace,
    project_path_subspace,
    ring_polymer_mean_path,
)
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate, sinc_dvr_kinetic
from pyrpmd.analysis.subspace_rpmd import normal_mode_matrix, ring_polymer_rate

KELVIN = 3.166811563e-6
MASS, V0, A = 1060.0, 0.425 / 27.211386, 0.66


def eckart(points):
    return V0 / np.cosh(points[..., 0] / A) ** 2


def eckart_gradient(points):
    out = np.zeros_like(points)
    x = points[..., 0]
    out[..., 0] = -2.0 * V0 / A * np.tanh(x / A) / np.cosh(x / A) ** 2
    return out


def eckart_analytic_rate(beta):
    """Analytic symmetric-Eckart transmission, per unit reactant length."""
    barrier = np.cosh(0.5 * np.pi * np.sqrt(8.0 * MASS * V0 * A * A - 1.0))

    def transmission(energy):
        x = np.pi * A * np.sqrt(2.0 * MASS * energy)
        return 0.0 if x == 0 else 1.0 / (1.0 + (barrier / np.sinh(x)) ** 2)

    flux = quad(lambda e: transmission(e) * np.exp(-beta * e), 0.0, 60.0 / beta, limit=400, points=[V0])[0]
    return flux / (2.0 * np.pi) / np.sqrt(MASS / (2.0 * np.pi * beta))


def eckart_model(n=321, half=8.0):
    return model_from_function(eckart, [np.linspace(-half, half, n)], gradient=eckart_gradient,
                               masses=MASS, dividing_surface=0.0)


def test_sinc_dvr_harmonic_levels():
    x = np.linspace(-6.0, 6.0, 81)
    levels = np.linalg.eigvalsh(sinc_dvr_kinetic(x, 1.0) + np.diag(0.5 * x * x))
    assert np.allclose(levels[:4], [0.5, 1.5, 2.5, 3.5], atol=1e-9)


@pytest.mark.parametrize("temperature, tolerance", [(1000.0, 5e-4), (300.0, 3e-3)])
def test_dense_exact_rate_matches_analytic_eckart(temperature, tolerance):
    beta = 1.0 / (temperature * KELVIN)
    result = exact_quantum_rate(eckart_model(), beta)
    assert result.method == "sylvester"
    assert result.rate / eckart_analytic_rate(beta) == pytest.approx(1.0, abs=tolerance)
    assert abs(result.diagnostics["flux_side_t0"]) < 1e-12 * result.flux / beta


def test_iterative_flux_eigenstates_agree_with_dense_route():
    beta = 1.0 / (1000.0 * KELVIN)
    model = eckart_model(241)
    dense = exact_quantum_rate(model, beta, method="sylvester")
    iterative = exact_quantum_rate(model, beta, method="flux_eigenstates", max_time=1.5e4)
    assert iterative.rate / dense.rate == pytest.approx(1.0, abs=2e-3)
    assert iterative.diagnostics["relative_extrapolation_spread"] < 5e-3
    assert iterative.diagnostics["flux_spectrum_converged"]


def test_separable_two_dimensional_rate_equals_one_dimensional_rate():
    beta = 1.0 / (500.0 * KELVIN)
    s = np.linspace(-6.0, 6.0, 81)
    y = np.linspace(-0.9, 0.9, 12)
    omega = 0.01
    model_2d = model_from_function(lambda p: eckart(p) + 0.5 * MASS * omega**2 * p[..., 1] ** 2,
                                   [s, y], masses=[MASS, MASS], dividing_surface=0.0)
    model_1d = model_from_function(eckart, [s], masses=MASS, dividing_surface=0.0)
    two = exact_quantum_rate(model_2d, beta)
    one = exact_quantum_rate(model_1d, beta)
    assert two.method == "sylvester"
    assert two.rate / one.rate == pytest.approx(1.0, rel=1e-8)


def test_normal_mode_matrix_is_orthonormal():
    for beads in (1, 2, 5, 8):
        c = normal_mode_matrix(beads)
        assert np.allclose(c.T @ c, np.eye(beads), atol=1e-13)
        assert np.allclose(c[:, 0], 1.0 / np.sqrt(beads))


def test_single_bead_rpmd_reproduces_classical_tst():
    beta = 1.0 / (1000.0 * KELVIN)
    result = ring_polymer_rate(eckart_model(481, 12.0), beta, 1, walkers=4, burn_in=2, sweeps=4,
                               trajectories=64, seed=3)
    classical = np.exp(-beta * V0) / np.sqrt(2.0 * np.pi * beta * MASS)
    assert result.rate / classical == pytest.approx(1.0, abs=5e-3)
    assert result.transmission == pytest.approx(1.0, abs=1e-12)
    assert result.diagnostics["ti_quadrature_log_error"] < 2e-3


def test_rpmd_beads_raise_the_tunnelling_rate_towards_exact():
    beta = 1.0 / (600.0 * KELVIN)
    model = eckart_model(481, 12.0)
    exact = eckart_analytic_rate(beta)
    classical = ring_polymer_rate(model, beta, 1, walkers=4, burn_in=2, sweeps=4, trajectories=64, seed=1)
    quantum = ring_polymer_rate(model, beta, 12, walkers=16, sweeps=200, trajectories=256, seed=1)
    assert classical.rate < 0.5 * exact
    assert quantum.rate / exact == pytest.approx(0.9, abs=0.15)
    assert quantum.diagnostics["max_energy_error_beta_p"] < 0.1
    assert 0.5 < quantum.diagnostics["hmc_acceptance"] < 0.95


def test_full_dimensional_subspace_correction_recovers_exact_rate():
    beta = 1.0 / (400.0 * KELVIN)
    correction = subspace_correction(eckart_model(), beta, 8,
                                     rpmd_options={"walkers": 8, "sweeps": 60, "trajectories": 64, "seed": 2})
    applied = apply_subspace_correction(correction.rpmd_rate, correction, standard_error=0.0, beta=beta)
    assert applied["corrected_rate"] == pytest.approx(correction.quantum_rate, rel=1e-12)
    assert correction.factor > 1.0
    assert correction.log_standard_error > 0
    with pytest.raises(ValueError, match="beta"):
        apply_subspace_correction(1.0, correction, beta=2 * beta)


def test_linear_projection_of_separable_potential_adds_bath_zero_point():
    omega = 0.01
    masses = np.array([MASS, 2.0 * MASS])

    def potential(x):
        return float(eckart(np.array([x[0]])) + 0.5 * masses[1] * omega**2 * (x[1] - 0.1) ** 2)

    def gradient(x):
        return np.array([eckart_gradient(np.array([x[0]]))[0], masses[1] * omega**2 * (x[1] - 0.1)])

    axis = np.sqrt(MASS) * np.linspace(-4.0, 4.0, 41)
    model = project_linear_subspace(potential, gradient, [0.0, 0.0], masses, [1.0, 0.0], [axis],
                                    environment="adiabatic", dividing_surface=0.0)
    expected = eckart(axis[:, None] / np.sqrt(MASS)) + 0.5 * omega
    assert np.allclose(model.values, expected, atol=1e-9)
    assert model.provenance["unstable_complement_points"] == 0
    frozen = project_linear_subspace(potential, gradient, [0.0, 0.0], masses, [1.0, 0.0], [axis],
                                     environment="frozen", dividing_surface=0.0)
    assert np.allclose(frozen.values - model.values, 0.5 * masses[1] * omega**2 * 0.01 - 0.5 * omega, atol=1e-9)


def test_straight_path_model_matches_linear_projection():
    omega = 0.02

    def potential(x):
        return float(eckart(np.array([x[0]])) + 0.5 * MASS * omega**2 * x[1] ** 2)

    def gradient(x):
        return np.array([eckart_gradient(np.array([x[0]]))[0], MASS * omega**2 * x[1]])

    path = np.column_stack([np.linspace(-4.0, 4.0, 9), np.zeros(9)])
    length = 8.0 * np.sqrt(MASS)
    axis = np.linspace(0.0, length, 41)
    model = project_path_subspace(potential, gradient, path, [MASS, MASS], axis)
    expected = eckart(((axis - 0.5 * length) / np.sqrt(MASS))[:, None]) + 0.5 * omega
    assert np.allclose(model.values, expected, atol=1e-8)
    assert model.dividing_surface == pytest.approx(0.5 * length)
    xi, curvature = path_curvature(path, [MASS, MASS])
    assert np.max(curvature) < 1e-10


def test_mean_path_and_curvature_width_criterion():
    q = np.linspace(-2.0, 2.0, 21)
    smooth, slope = ring_polymer_mean_path(q, 0.3 * q + 0.1, bandwidth=0.4)
    assert np.allclose(smooth[:, 0], 0.3 * q + 0.1)
    assert np.allclose(slope[:, 0], 0.3)
    verdict = curvature_width_criterion([0.1, 4.0], [0.2, 0.25])
    assert verdict["include_transverse_mode"] and verdict["maximum"] == pytest.approx(1.0)
    assert not curvature_width_criterion([0.1], [0.5])["include_transverse_mode"]


def _record(dimension, factor, error=0.01, beta=1000.0):
    return SubspaceCorrection(beta, dimension, factor, error, factor, 1.0, 16, {"dimension": dimension})


def test_nested_sensitivity_selects_smallest_settled_dimension():
    result = nested_subspace_sensitivity([_record(1, 1.30), _record(2, 1.90), _record(3, 1.92)])
    assert result["recommended_dimension"] == 2
    assert not result["steps"][0]["settled"] and result["steps"][1]["settled"]
    settled = nested_subspace_sensitivity([_record(1, 0.97), _record(2, 0.98)])
    assert settled["recommended_dimension"] == 1
    with pytest.raises(ValueError):
        nested_subspace_sensitivity([_record(1, 1.0), _record(2, 1.0, beta=2.0)])


def test_correction_record_round_trip(tmp_path):
    record = _record(2, 1.5, 0.04)
    path = record.write(tmp_path / "c.json")
    loaded = SubspaceCorrection.read(path)
    assert loaded.factor == 1.5 and loaded.dimension == 2 and loaded.log_standard_error == 0.04
    with pytest.raises(ValueError):
        SubspaceCorrection.from_dict({"schema": "other"})


def _fake_rate_run(root, beta, publishable=True):
    (root / "input").mkdir(parents=True)
    (root / "input" / "resolved.json").write_text(json.dumps({"beta": beta, "task": "rate"}))
    summary = {"task": "rate", "status": "plateau_estimate" if publishable else "pilot_requires_convergence",
               "rate_cm3_s": 2.0e-12 if publishable else None, "rate_estimate_cm3_s": 2.0e-12,
               "rate_standard_error_cm3_s": 1.0e-13}
    (root / "summary.json").write_text(json.dumps(summary))
    return summary


def test_correct_rate_run_writes_provenance_and_refuses_mismatches(tmp_path):
    beta = 1000.0
    _fake_rate_run(tmp_path, beta)
    before = (tmp_path / "summary.json").read_bytes()
    record = _record(1, 1.5, 0.04, beta=beta)
    result = correct_rate_run(tmp_path, record)
    assert (tmp_path / "summary.json").read_bytes() == before
    written = json.loads((tmp_path / "rate_quantum_corrected.json").read_text())
    assert written == result
    assert result["corrected_rate_cm3_s"] == pytest.approx(3.0e-12)
    expected_error = 3.0e-12 * np.hypot(0.05, 0.04)
    assert result["corrected_rate_standard_error_cm3_s"] == pytest.approx(expected_error)
    assert result["physical_converged"] is False and result["source_files"]
    # Re-running overwrites its own output; the wrong temperature is refused.
    correct_rate_run(tmp_path, record)
    with pytest.raises(ValueError, match="beta"):
        correct_rate_run(tmp_path, _record(1, 1.5, beta=900.0))
    with pytest.raises(ValueError, match="overwrite"):
        correct_rate_run(tmp_path, record, output=tmp_path / "summary.json")
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match="unrelated"):
        correct_rate_run(tmp_path, record, output=tmp_path / "other.json")


def test_correct_rate_run_keeps_unpublishable_rates_unpublished(tmp_path):
    _fake_rate_run(tmp_path, 1000.0, publishable=False)
    result = correct_rate_run(tmp_path, _record(1, 2.0, beta=1000.0))
    assert result["corrected_rate_cm3_s"] is None
    assert result["corrected_rate_estimate_cm3_s"] == pytest.approx(4.0e-12)


def test_model_validation():
    s = np.linspace(-1.0, 1.0, 16)
    with pytest.raises(ValueError, match="dividing_surface"):
        model_from_function(eckart, [s], dividing_surface=5.0)
    with pytest.raises(ValueError, match="one to three"):
        model_from_function(lambda p: p.sum(-1), [s] * 4)
    with pytest.raises(ValueError, match="reactant"):
        model_from_function(eckart, [s], reactant="bimolecular")
    with pytest.raises(ValueError, match="uniformly"):
        model_from_function(eckart, [np.r_[s[:-1], 3.0]])


def test_grid_potential_interpolates_values_and_gradients():
    from pyrpmd.analysis.subspace_model import GridPotential

    axes = [np.linspace(-2.0, 2.0, 41), np.linspace(-1.0, 1.5, 31)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    grid = GridPotential(axes, np.sin(points[..., 0]) * np.cos(2 * points[..., 1]))
    sample = np.random.default_rng(0).uniform([-2.0, -1.0], [2.0, 1.5], size=(200, 2))
    energy, gradient = grid.energy_and_gradient(sample)
    x, y = sample[:, 0], sample[:, 1]
    assert np.allclose(energy, np.sin(x) * np.cos(2 * y), atol=5e-5)
    assert np.allclose(gradient, np.column_stack([np.cos(x) * np.cos(2 * y), -2 * np.sin(x) * np.sin(2 * y)]), atol=2e-3)
    grid.energy_and_gradient(np.array([[3.0, 0.0]]))
    assert grid.outside_fraction == pytest.approx(1 / 201)


def test_three_dimensional_separable_iterative_rate_equals_one_dimensional_rate():
    beta = 1.0 / (1000.0 * KELVIN)
    s = np.linspace(-6.0, 6.0, 61)
    y = np.linspace(-0.6, 0.6, 8)
    z = np.linspace(-0.5, 0.5, 8)
    omegas = (0.012, 0.018)

    def potential(p):
        return eckart(p) + 0.5 * MASS * (omegas[0] ** 2 * p[..., 1] ** 2 + omegas[1] ** 2 * p[..., 2] ** 2)

    model_3d = model_from_function(potential, [s, y, z], masses=[MASS] * 3, dividing_surface=0.0)
    model_1d = model_from_function(eckart, [s], masses=MASS, dividing_surface=0.0)
    three = exact_quantum_rate(model_3d, beta, max_time=1.5e4)
    one = exact_quantum_rate(model_1d, beta)
    assert three.method == "flux_eigenstates"
    assert three.rate / one.rate == pytest.approx(1.0, abs=3e-3)


def _separable_2d(omega=0.01):
    s = np.linspace(-6.0, 6.0, 81)
    y = np.linspace(-0.9, 0.9, 12)
    return model_from_function(lambda p: eckart(p) + 0.5 * MASS * omega**2 * p[..., 1] ** 2,
                               [s, y], masses=[MASS, MASS], dividing_surface=0.0)


def test_dense_route_options_are_consistent():
    from pyrpmd.analysis.subspace_quantum import absorbing_potential

    beta = 1.0 / (500.0 * KELVIN)
    model = _separable_2d()
    base = exact_quantum_rate(model, beta)
    high = exact_quantum_rate(model, beta, truncation_energy=10.0)
    assert high.rate == pytest.approx(base.rate, rel=1e-10)
    floor = exact_quantum_rate(model, beta, energy_floor=-1.0)
    assert floor.rate == pytest.approx(base.rate, rel=1e-10)
    axis0 = model.axes[0]
    width = 0.2 * (axis0[-1] - axis0[0])
    strength = base.diagnostics["cap_strength"]
    absorber = lambda p: absorbing_potential(np.r_[axis0[0], p[:, 0], axis0[-1]], width=width,
                                             strength=strength)[1:-1]
    custom = exact_quantum_rate(model, beta, absorber=absorber)
    assert custom.rate == pytest.approx(base.rate, rel=1e-10)
    # For a channel perpendicular to the slice, the quantum line density is
    # the free translational density times the transverse partition function.
    sliced = exact_quantum_rate(model, beta, normalization="slice")
    # Finite-box and grid effects in the slice density are a few 1e-3 here.
    assert sliced.rate == pytest.approx(base.rate, rel=5e-3)
    truncated = exact_quantum_rate(model, beta, truncation_energy=float(model.values.min()) + 60.0 / beta)
    assert truncated.diagnostics["retained_points"] < model.values.size
    assert truncated.rate == pytest.approx(base.rate, rel=1e-3)


def test_energy_floor_is_exact_below_the_open_channel_threshold():
    """Exothermic 1D model: states below the reactant threshold carry no net flux."""
    beta = 1.0 / (300.0 * KELVIN)
    drop = 0.02

    def potential(p):
        x = p[..., 0]
        return V0 / np.cosh(x / A) ** 2 - drop * 0.5 * (1.0 + np.tanh(x / (2 * A)))

    model = model_from_function(potential, [np.linspace(-10.0, 10.0, 401)], masses=MASS, dividing_surface=0.0)
    one = exact_quantum_rate(model, beta, energy_floor=-1.0 / beta)
    three = exact_quantum_rate(model, beta, energy_floor=-3.0 / beta)
    assert one.diagnostics["states_below_floor"] > three.diagnostics["states_below_floor"] > 0
    # In this short 1D box, absorber reflection of slow sub-threshold states
    # is a ~2% effect either way; the floor margin must not matter beyond it.
    assert one.rate == pytest.approx(three.rate, rel=3e-2)
    full = exact_quantum_rate(model, beta)
    assert one.rate == pytest.approx(full.rate, rel=3e-2)


def test_relaxed_path_follows_a_curved_valley():
    from pyrpmd.analysis.subspace_model import relaxed_path

    radius, k, slope = 10.0, 0.5, 0.01

    def potential(x):
        r, theta = np.hypot(x[0], x[1]), np.arctan2(x[1], x[0])
        return 0.5 * k * (r - radius) ** 2 - slope * theta

    def gradient(x):
        r = np.hypot(x[0], x[1])
        return k * (r - radius) * x / r - slope * np.array([-x[1], x[0]]) / r**2

    points = relaxed_path(potential, gradient, [radius, 0.0], [0.0, 1.0], [1.0, 1.0], step=0.5,
                          max_steps=40, rigid_body=False)
    radii = np.hypot(points[:, 0], points[:, 1])
    assert np.max(np.abs(radii - radius)) < 0.05
    assert np.arctan2(points[-1, 1], points[-1, 0]) > 1.5


def test_rigid_body_complement_keeps_only_vibrations():
    masses = np.array([16.0, 1.0, 1.0]) * 1822.888
    a, theta0, kb, ka = 1.8, 1.82, 0.5, 0.15

    def internals(x):
        g = x.reshape(3, 3)
        u, v = g[1] - g[0], g[2] - g[0]
        return np.linalg.norm(u), np.linalg.norm(v), np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))

    def potential(x):
        r1, r2, t = internals(x)
        return 0.5 * kb * ((r1 - a) ** 2 + (r2 - a) ** 2) + 0.5 * ka * (t - theta0) ** 2

    def gradient(x, h=1e-6):
        return np.array([(potential(x + h * e) - potential(x - h * e)) / (2 * h) for e in np.eye(9)])

    eq = np.array([[0.0, 0.0, 0.0], [a, 0.0, 0.0], [a * np.cos(theta0), a * np.sin(theta0), 0.0]]).ravel()
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    hess = np.array([(gradient(eq + 1e-4 * e) - gradient(eq - 1e-4 * e)) / 2e-4 for e in np.eye(9)])
    values, vectors = np.linalg.eigh(hess / sqrt_m[:, None] / sqrt_m[None, :])
    vib = values[values > 1e-8]
    mode = vectors[:, np.argmax(values)]
    model = project_linear_subspace(potential, gradient, eq, np.repeat(masses, 3), mode,
                                    [np.linspace(-1.0, 1.0, 9)], environment="adiabatic", rigid_body=True)
    assert model.provenance["complement_dimension"] == 2
    expected = 0.5 * float(np.sum(np.sqrt(np.sort(vib)[:2])))
    assert model.values[4] == pytest.approx(expected, rel=1e-3)
