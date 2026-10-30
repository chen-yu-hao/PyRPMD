"""Independent transport references and finite-record scientific qualifiers."""

import json
import numpy as np
import pytest

from pyrpmd.analysis.diffusion import analyze_diffusion, DIFFUSION_AU_TO_M2_S
from pyrpmd.constants import BOHR_M, ATOMIC_TIME_S


def brownian(seed=73, replicas=48, frames=1025, atoms=2, dt=0.1):
    tensor = np.array([[0.3, 0.08, -0.02], [0.08, 0.5, 0.04], [-0.02, 0.04, 0.2]])
    rng = np.random.default_rng(seed)
    increments = rng.multivariate_normal(np.zeros(3), 2 * dt * tensor,
                                         size=(replicas, frames - 1, atoms))
    q = np.concatenate((np.zeros((replicas, 1, atoms, 3)), increments.cumsum(axis=1)), axis=1)
    return tensor, q, np.arange(frames) * dt


def test_anisotropic_brownian_reference_and_independent_replica_errors():
    expected, q, time = brownian()
    result = analyze_diffusion(q, time, max_lag=64, fit_range=(1.0, 5.0), bootstrap=300, seed=55)
    assert result["status"] == "diffusive_regime_candidate"
    assert result["physical_converged"] is False
    np.testing.assert_allclose(result["diffusion_tensor_au"], expected, atol=0.025, rtol=0.1)
    assert result["diffusion_au"] == pytest.approx(np.trace(expected) / 3, rel=0.08)
    np.testing.assert_allclose(result["diffusion_tensor_m2_s"],
                               np.asarray(result["diffusion_tensor_au"]) * BOHR_M**2 / ATOMIC_TIME_S)
    assert result["diffusion_m2_s"] == pytest.approx(result["diffusion_au"] * DIFFUSION_AU_TO_M2_S)
    uncertainty = result["uncertainty"]
    assert uncertainty["method"] == "independent_replicas"
    assert uncertainty["independent_units"] == len(q)
    replica_values = np.array([item["diffusion_au"] for item in result["per_replica_fits"]])
    assert uncertainty["standard_error_au"] == pytest.approx(replica_values.std(ddof=1) / np.sqrt(len(q)))
    lo, hi = uncertainty["confidence_interval_au"]
    assert lo < result["diffusion_au"] < hi
    # Every nested result can be saved by an ordinary strict JSON writer.
    json.dumps(result, allow_nan=False)


def test_fft_displacements_match_direct_tensor_average_for_all_lags():
    _, q, time = brownian(replicas=3, frames=39, atoms=3)
    result = analyze_diffusion(q, time, axes=[2, 0], atom_indices=[1, 2], max_lag=16, bootstrap=0)
    x = q[:, :, [1, 2], :][..., [2, 0]]
    expected = [np.zeros((2, 2))]
    for lag in range(1, 17):
        delta = x[:, lag:] - x[:, :-lag]
        expected.append(np.einsum("rfna,rfnb->ab", delta, delta) / np.prod(delta.shape[:3]))
    np.testing.assert_allclose(result["msd_tensor_au"], expected, atol=1e-13, rtol=1e-12)
    assert result["input"]["axes"] == [2, 0]
    assert result["input"]["atom_indices"] == [1, 2]


def test_ballistic_motion_has_correct_msd_and_vacf_but_no_diffusion_claim():
    time = np.arange(101) * 0.2
    speed = np.array([0.3, -0.1, 0.2])
    q = (time[:, None] * speed)[None, :, None, :]
    v = np.broadcast_to(speed, q.shape)
    result = analyze_diffusion(q, time, velocities=v, max_lag=25, bootstrap=0)
    lag_time = np.asarray(result["lag_time_au"])
    expected_msd = lag_time[:, None, None]**2 * np.outer(speed, speed)
    np.testing.assert_allclose(result["msd_tensor_au"], expected_msd, atol=2e-14)
    np.testing.assert_allclose(result["vacf_tensor_au"],
                               np.broadcast_to(np.outer(speed, speed), expected_msd.shape), atol=1e-16)
    np.testing.assert_allclose(result["vacf_integral_tensor_au"],
                               lag_time[:, None, None] * np.outer(speed, speed), atol=1e-16)
    assert result["status"] == "ballistic_or_drifting"
    assert result["diffusion_m2_s"] is None
    assert result["raw_fit"]["log_msd_slope"] == pytest.approx(2.0)
    assert result["uncertainty"]["method"] == "unavailable"
    assert result["uncertainty"]["standard_error_au"] is None


def test_stationary_record_does_not_establish_zero_diffusivity():
    q = np.ones((2, 81, 3, 3)) * 1e6
    result = analyze_diffusion(q, np.arange(81), bootstrap=0)
    assert result["status"] == "no_observed_motion"
    assert result["diffusion_au"] is None
    assert result["raw_fit"]["diffusion_au"] == 0
    json.dumps(result, allow_nan=False)


def test_caged_oscillator_is_not_reported_as_diffusion():
    time = np.arange(2049) * 0.1
    phases = np.arange(12) * 2 * np.pi / 12
    q = np.zeros((12, len(time), 1, 3))
    q[:, :, 0, 0] = np.cos(time[None, :] + phases[:, None])
    result = analyze_diffusion(q, time, axes=(0,), max_lag=500,
                               fit_range=(25.0, 50.0), bootstrap=0)
    assert result["status"] == "non_diffusive_window"
    assert result["diffusion_au"] is None


def test_mass_weighted_reference_removes_lattice_motion_for_both_estimators():
    _, q, time = brownian(replicas=8, frames=513, atoms=1)
    rng = np.random.default_rng(4)
    # Values need not be derivatives in this algebra-only frame test.
    velocities = rng.normal(size=q.shape)
    drift_velocity = np.array([0.15, -0.2, 0.1])
    drift = time[None, :, None, None] * drift_velocity[None, None, None, :]
    host = np.zeros((8, len(time), 2, 3))
    host[:, :, 0, 0] = -2
    host[:, :, 1, 0] = 1
    host += drift
    host_velocity = np.broadcast_to(drift_velocity, host.shape)
    shifted = analyze_diffusion(q + drift, time, velocities=velocities + drift_velocity,
                                reference_positions=host, reference_velocities=host_velocity,
                                reference_masses=[1, 2], max_lag=32, bootstrap=0)
    original = analyze_diffusion(q, time, velocities=velocities, max_lag=32, bootstrap=0)
    np.testing.assert_allclose(shifted["msd_tensor_au"], original["msd_tensor_au"], atol=5e-13)
    np.testing.assert_allclose(shifted["vacf_tensor_au"], original["vacf_tensor_au"], atol=2e-15)
    assert shifted["input"]["reference_atom_count"] == 2
    assert shifted["raw_fit"]["diffusion_au"] == pytest.approx(original["raw_fit"]["diffusion_au"])


def test_multiple_origins_do_not_create_independent_error_samples():
    _, q, time = brownian(replicas=1, frames=2049, atoms=4)
    result = analyze_diffusion(q, time, max_lag=16)
    assert result["uncertainty"]["method"] == "unavailable"
    assert result["uncertainty"]["standard_error_au"] is None
    batches = analyze_diffusion(q, time, max_lag=16, block_length=256, bootstrap=50)
    assert batches["uncertainty"]["method"] == "conditional_nonoverlapping_blocks"
    assert batches["uncertainty"]["independent_units"] == 8
    assert batches["uncertainty"]["standard_error_au"] > 0
    assert "correlation time" in batches["uncertainty"]["reason"]
    assert batches["physical_converged"] is False


def test_short_saved_record_returns_diagnostic_without_a_fit():
    result = analyze_diffusion(np.zeros((3, 1, 3)), [0.0, 0.2, 0.4])
    assert result["status"] == "insufficient_data"
    assert result["raw_fit"] is None
    assert result["diffusion_au"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change,match", [
    ({"axes": [0, 0]}, "axes"),
    ({"axes": [True]}, "integer"),
    ({"atom_indices": []}, "atom_indices"),
    ({"atom_indices": [1]}, "atom_indices"),
    ({"max_lag": 100}, "max_lag"),
    ({"max_lag": 1.5}, "integer"),
    ({"bootstrap": 1}, "bootstrap"),
    ({"bootstrap": -1}, "bootstrap"),
    ({"fit_range": [2.0, 1.0]}, "fit_range"),
    ({"fit_range": [0.0, 100.0]}, "fit_range"),
    ({"block_length": 10}, "block_length"),
    ({"reference_masses": [1.0]}, "reference"),
    ({"velocities": np.zeros((3, 1, 3))}, "match positions"),
])
def test_invalid_controls_fail_before_misleading_estimation(change, match):
    with pytest.raises(ValueError, match=match):
        analyze_diffusion(np.zeros((81, 1, 3)), np.arange(81), **change)


@pytest.mark.parametrize("time", [np.arange(81)[::-1], np.arange(81)**2,
                                  np.zeros(81), np.full(81, np.nan)])
def test_invalid_times_rejected(time):
    with pytest.raises(ValueError, match="times"):
        analyze_diffusion(np.zeros((81, 1, 3)), time)


def test_nonfinite_complex_and_mismatched_reference_records_rejected():
    q = np.zeros((81, 1, 3))
    time = np.arange(81)
    for invalid in (q + np.nan, q + 1j):
        with pytest.raises(ValueError, match="positions"):
            analyze_diffusion(invalid, time)
    with pytest.raises(ValueError, match="reference_positions"):
        analyze_diffusion(q, time, reference_positions=q[:2])
    with pytest.raises(ValueError, match="reference_velocities"):
        analyze_diffusion(q, time, velocities=q, reference_positions=q)
    with pytest.raises(ValueError, match="positive"):
        analyze_diffusion(q, time, reference_positions=q, reference_masses=[0])


def test_time_units_rescale_diffusivity_and_leave_position_statistics_unchanged():
    _, q, time = brownian(replicas=8, frames=513, atoms=1)
    one = analyze_diffusion(q, time, max_lag=24, bootstrap=0)
    five = analyze_diffusion(q, 5 * time, max_lag=24, bootstrap=0)
    np.testing.assert_allclose(one["msd_tensor_au"], five["msd_tensor_au"])
    assert five["raw_fit"]["diffusion_au"] == pytest.approx(one["raw_fit"]["diffusion_au"] / 5)
