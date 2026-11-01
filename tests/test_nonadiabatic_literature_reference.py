"""Independent controls for the published-model benchmark, not fitted targets."""

import numpy as np
import pytest
import json

from benchmarks.tier2_literature.nonadiabatic import (
    CHANNELS, TullySimpleAvoidedCrossing, quantum_scattering, wilson,
)


def test_open_discrete_leads_transmit_a_free_channel_without_reflection():
    model = TullySimpleAvoidedCrossing(a=0., c=0.)
    result = quantum_scattering(15., dx=.01, boundary=2., model=model)
    # This degenerate free control enters and exits diabatic channel zero;
    # the benchmark's right-hand channel labels call that channel 'upper'.
    assert result["transmitted_upper"] == pytest.approx(1., abs=5e-13)
    assert result["reflected_lower"] < 1e-24
    assert result["transmitted_lower"] == 0.
    assert result["flux_error"] < 5e-13


def test_uncoupled_closed_channel_reflects_all_incoming_flux():
    result = quantum_scattering(8., dx=.005, boundary=8.,
                                model=TullySimpleAvoidedCrossing(c=0.))
    assert result["reflected_lower"] == pytest.approx(1., abs=5e-12)
    assert result["transmitted_lower"] == 0.
    assert result["transmitted_upper"] == 0.


def test_coupled_quantum_reference_conserves_flux_and_converges_quadratically():
    values = [quantum_scattering(15., dx=dx, boundary=8.) for dx in (.01, .005, .0025)]
    for result in values:
        assert result["flux_error"] < 5e-12
        assert all(result[channel] >= 0. for channel in CHANNELS[:4])
    coarse = values[0]["transmitted_upper"]-values[1]["transmitted_upper"]
    fine = values[1]["transmitted_upper"]-values[2]["transmitted_upper"]
    assert coarse/fine == pytest.approx(4., rel=.015)


def test_tully_direct_qc_gradients_and_nac_match_operator_finite_differences():
    model = TullySimpleAvoidedCrossing(sign_flips=True)
    q = np.zeros((5, 1, 3)); q[:, 0, 0] = [-2., -.3, .2, .8, 2.]
    result = model.evaluate(q)
    gradient = np.zeros((5, 2, 2))
    gradient[:, 0, 0], gradient[:, 1, 1] = result.gradients[:, 0, 0, 0], result.gradients[:, 1, 0, 0]
    gap = result.energies[:, None, :]-result.energies[:, :, None]
    gradient += gap*result.couplings[:, :, :, 0, 0]
    u = result.context
    fixed = u@gradient@u.swapaxes(1, 2)
    plus, minus = q.copy(), q.copy()
    plus[:, 0, 0] += 1e-6; minus[:, 0, 0] -= 1e-6
    finite = (model.diabatic(plus).matrix-model.diabatic(minus).matrix)/2e-6
    np.testing.assert_allclose(fixed, finite, atol=5e-12, rtol=1e-8)
    following = model.evaluate(plus, previous_q=q, previous_context=u)
    np.testing.assert_allclose(following.overlaps, u.swapaxes(1, 2)@following.context, atol=1e-15)


def test_zero_event_interval_does_not_claim_zero_uncertainty():
    low, high = wilson(0, 512)
    assert low == pytest.approx(0.)
    assert .007 < high < .008
    low, high = wilson(256, 512)
    assert low < .5 < high
    assert low == pytest.approx(1-high)


def test_trajectory_job_resumes_real_completed_prefix_with_identical_physics(tmp_path, monkeypatch):
    import benchmarks.tier2_literature.nonadiabatic as benchmark

    spec = {"method": "mash", "momentum": 25., "dt": 8., "seed": 731921,
            "count": 2, "boundary": 8., "electronic_substeps": 4,
            "code_identity": "test_identity", "path": str(tmp_path/"resume.json")}
    real = benchmark.run_trajectory
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ArithmeticError("injected interruption between trajectories")
        return real(*args, **kwargs)
    monkeypatch.setattr(benchmark, "run_trajectory", fail_second)
    with pytest.raises(ArithmeticError, match="interruption"):
        benchmark._run_job(spec)
    prefix = json.loads((tmp_path/"resume.json").read_text())
    assert not prefix["complete"] and len(prefix["trajectories"]) == 1
    monkeypatch.setattr(benchmark, "run_trajectory", real)
    benchmark._run_job(spec)
    benchmark._run_job(dict(spec, path=str(tmp_path/"full.json")))
    resumed = json.loads((tmp_path/"resume.json").read_text())
    full = json.loads((tmp_path/"full.json").read_text())
    assert resumed["complete"]
    assert resumed["trajectories"][0] == prefix["trajectories"][0]
    for actual, expected in zip(resumed["trajectories"], full["trajectories"]):
        assert {k: v for k, v in actual.items() if k != "seconds"} == {
            k: v for k, v in expected.items() if k != "seconds"}
    with pytest.raises(ValueError, match="identity differs"):
        benchmark._run_job(dict(spec, dt=4.))
