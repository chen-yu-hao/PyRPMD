"""Native path portability and exact SVR zero-energy transition."""
import numpy as np
import pytest
pytest.importorskip("pyrpmd._thermal")
from pyrpmd.native_ensemble import NativeEnsemble


def test_svr_zero_energy_uses_finite_time_transition():
    atoms = 20000
    q = np.zeros((1, atoms, 3))
    engine = NativeEnsemble(q, np.ones(atoms), 2., .1, p=q,
                            potential={"kind": "free"},
                            thermostat={"mode": "svr", "tau": 1.}, seed=971)
    engine.run(1)
    p = engine.state()["p"]
    expected = .5*(-np.expm1(-.1))
    assert np.mean(p*p) == pytest.approx(expected, rel=.025)


def test_native_streaming_unicode_path(tmp_path):
    engine = NativeEnsemble(np.zeros((4, 2, 3)), [1., 2.], 2., .025,
                            potential={"kind": "harmonic", "k": .7}, seed=41)
    result = engine.run(12, stride=3, trajectory=tmp_path / "量子轨迹.npy")
    np.testing.assert_array_equal(np.load(result["trajectory"]), result["samples"])
