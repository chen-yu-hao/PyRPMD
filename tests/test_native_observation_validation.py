"""Overflow in a native observation must roll back the whole run block."""

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.native_ensemble import NativeEnsemble


def _free(**kwargs):
    arguments = dict(q=np.zeros((1, 1, 3)), p=np.zeros((1, 1, 3)),
                     masses=[1.], beta=1., dt=1e-5,
                     potential={"kind": "free"}, thermostat="none", seed=42)
    arguments.update(kwargs)
    return NativeEnsemble(**arguments)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thermostat", ["none", "langevin"])
def test_overflowing_energy_rolls_back_state_rng_and_stream(tmp_path, stream, thermostat):
    engine = _free(p=np.full((1, 1, 3), 1e200), thermostat=thermostat)
    before = engine.state()
    target = tmp_path / "trajectory.npy"
    with pytest.raises(ValueError, match="nonfinite ensemble observation"):
        engine.run(2, trajectory=target if stream else None)
    assert engine.state()["digest"] == before["digest"]
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("stream", [False, True])
def test_overflowing_ring_radius_is_not_a_successful_moment_record(tmp_path, stream):
    # An almost free two-bead ring can have finite H_P but an unrepresentable
    # squared radius. This exercises the moment check independently of energy.
    q = np.zeros((2, 1, 3))
    q[:, 0, 0] = [1e160, -1e160]
    engine = _free(q=q, p=np.zeros_like(q), beta=1e200)
    before = engine.state()
    target = tmp_path / "moments.npy"
    with pytest.raises(ValueError, match="nonfinite centroid observation"):
        engine.run(1, record_centroids=True, record_moments=True,
                   trajectory=target if stream else None)
    assert engine.state()["digest"] == before["digest"]
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("stream", [False, True])
def test_late_observation_failure_discards_earlier_rows_in_the_block(tmp_path, stream):
    p = np.zeros((2, 1, 3))
    p[:, 0, 0] = [1e150, -1e150]
    arguments = dict(q=np.zeros_like(p), p=p, beta=1e200, dt=5e3)
    accepted = _free(**arguments).run(2, record_centroids=True, record_moments=True)
    assert np.isfinite(accepted["radii_gyration_squared"]).all()
    engine = _free(**arguments)
    before = engine.state()
    target = tmp_path / "partial.npy"
    with pytest.raises(ValueError, match="nonfinite centroid observation"):
        engine.run(3, record_centroids=True, record_moments=True,
                   trajectory=target if stream else None)
    assert engine.state()["digest"] == before["digest"]
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("stream", [False, True])
def test_native_time_overflow_rejected_even_without_an_observation(tmp_path, stream):
    engine = _free(dt=1e308)
    before = engine.state()
    target = tmp_path / "unsampled.npy"
    with pytest.raises(ValueError, match="trajectory time overflow"):
        engine.run(2, stride=3, trajectory=target if stream else None)
    assert engine.state()["digest"] == before["digest"]
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("periodic", [False, True])
def test_normal_records_preserve_defined_and_undefined_observables(periodic):
    engine = _free(**({"cell": np.eye(3) * 2} if periodic else {}))
    result = engine.run(2, record_centroids=True, record_moments=True)
    samples = result["samples"]
    assert np.isfinite(samples[:, :6]).all()
    assert np.isfinite(samples[:, 8:]).all()
    if periodic:
        assert np.isfinite(samples[:, 6:8]).all()
    else:
        assert np.isnan(samples[:, 6:8]).all()
    for key in ("centroids", "velocities", "radii_gyration_squared"):
        assert np.isfinite(result[key]).all()
