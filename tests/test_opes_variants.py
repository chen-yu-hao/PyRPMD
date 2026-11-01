"""Literal-density/force formulas and exact restart for OPES variants."""

import copy
import numpy as np
import pytest
from scipy.integrate import trapezoid

from pyrpmd.cv import Cartesian
from pyrpmd.opes import OPESBias, OPESExploreBias, normalized_weights
from pyrpmd.system import System
from pyrpmd.workflows.opes import run_opes, reanalyze_opes


@pytest.mark.parametrize("mode", ["metad", "explore"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_variant_probability_and_z_match_independent_gaussian_sum(mode, adaptive):
    beta, gamma, barrier = 2.0, 5.0, 3.0
    bias = OPESBias(Cartesian(0), [.4], beta, barrier=barrier, bias_factor=gamma,
                    mode=mode, adaptive_sigma=adaptive)
    points = np.linspace(-4, 4, 8001)
    centers = np.array([-.4, .2, 1.3, .7, -1.5])
    for index, center in enumerate(centers, 1):
        bias.update([center], step=index, time=index * .1)
        state = bias.state_dict()
        widths = np.asarray(state["kernel_sigmas"])[:, 0] if adaptive else np.full(index, .4)
        weights = normalized_weights(state["log_weights"])
        c = centers[:index]
        matrix = np.exp(-.5 * ((points[:, None] - c) / widths)**2) / (np.sqrt(2 * np.pi) * widths)
        probability = matrix @ weights
        center_matrix = np.exp(-.5 * ((c[:, None] - c) / widths)**2) / (np.sqrt(2 * np.pi) * widths)
        z = np.mean(center_matrix @ weights)
        np.testing.assert_allclose(np.exp(bias.log_probability(points[:, None])), probability, rtol=5e-13, atol=2e-15)
        assert bias.log_z == pytest.approx(np.log(z), abs=1e-14)
        prefactor = (gamma - 1 if mode == "explore" else 1 - 1 / gamma) / beta
        expected = prefactor * np.log(probability / z + np.exp(-barrier / prefactor))
        np.testing.assert_allclose(bias.physical_energy(points[:, None]), expected, atol=1e-14, rtol=5e-13)
    if mode == "explore":
        np.testing.assert_array_equal(state["log_weights"], 0)
        assert bias.effective_sample_size == pytest.approx(len(centers))


@pytest.mark.parametrize("mode", ["metad", "explore"])
def test_adaptive_width_force_normalization_and_resume(mode):
    cv = Cartesian(0)
    cv.periods = (2 * np.pi,)
    options = dict(barrier=1.5, mode=mode, adaptive_sigma=True, sigma_min=[.03])
    bias = OPESBias(cv, [.4], 3., **options)
    for step, x in enumerate([3.0, -3., -2.8, -2.9, -2.5], 1):
        bias.update([x], step=step, time=step / 10)
    state = bias.state_dict()
    assert abs(state["adaptive_mean"][0] - np.pi) < .7  # seam-unwrapped moment
    point = np.linspace(-np.pi, np.pi, 20001)[:, None]
    probability = np.exp(bias.log_probability(point))
    assert trapezoid(probability, point[:, 0]) == pytest.approx(1, abs=2e-12)
    x = point[::1000]
    energy, gradient = bias.physical_energy_gradient(x)
    numeric = (bias.physical_energy(x + 1e-6) - bias.physical_energy(x - 1e-6)) / 2e-6
    np.testing.assert_allclose(gradient[:, 0], numeric, atol=2e-9, rtol=2e-6)
    resumed = OPESBias(cv, [.4], 3., state=state, **options)
    assert resumed.state_dict() == state
    for model in (resumed, bias):
        model.update([-2.4], step=6, time=.6)
    assert resumed.state_dict() == bias.state_dict()
    corrupt = copy.deepcopy(state)
    corrupt["kernel_sigmas"][2][0] *= 1.3
    with pytest.raises(ValueError, match="width"):
        OPESBias(cv, [.4], 3., state=corrupt, **options)
    corrupt = copy.deepcopy(state)
    corrupt["adaptive_mean"][0] += 1
    with pytest.raises(ValueError, match="recurrence"):
        OPESBias(cv, [.4], 3., state=corrupt, **options)


def test_explore_convenience_and_state_identity():
    bias = OPESExploreBias(Cartesian(0), [.2], 4., barrier=1.)
    bias.update([.3], step=1, time=.1)
    with pytest.raises(ValueError, match="state"):
        OPESBias(Cartesian(0), [.2], 4., barrier=1., state=bias.state_dict())
    corrupt = bias.state_dict()
    corrupt["log_weights"][0] = 1
    with pytest.raises(ValueError, match="unweighted"):
        OPESExploreBias(Cartesian(0), [.2], 4., barrier=1., state=corrupt)


@pytest.mark.parametrize("mode", ["metad", "explore"])
def test_opes_variant_workflow_checkpoint_and_offline_reanalysis(tmp_path, mode):
    system = System(("H",), np.array([1.0]), np.array([[.1, 0., 0.]]))
    config = {"task": "opes", "beta": 2., "beads": 2, "dt_au": .02,
              "seed": 31, "native": False,
              "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .2}},
              "sampling": {"equilibration_steps": 0, "production_steps": 16, "sample_interval": 1,
                           "andersen_interval": 2, "replicas": 1},
              "cvs": [{"name": "x", "kind": "cartesian", "atom": 0, "axis": "x"}],
              "opes": {"sigma": [.3], "barrier": 2., "pace": 2, "bias_factor": 5,
                       "mode": mode, "adaptive_sigma": True, "sigma_min": [.05]},
              "analysis": {"bins": [6], "bootstrap": 0}, "output": {"checkpoint_steps": 3}}
    result = run_opes(config, system, tmp_path, return_data=True)
    again = reanalyze_opes(config, system, tmp_path, return_data=True)
    assert result["summary"]["method"] == "opes_" + mode + "_adaptive_bandwidth"
    np.testing.assert_array_equal(result["data"]["free_energy"], again["data"]["free_energy"])
    assert result["streams"][0]["bias_state"] == again["streams"][0]["bias_state"]
    completed = run_opes(config, system, tmp_path, return_data=True)
    np.testing.assert_array_equal(result["streams"][0]["samples"], completed["streams"][0]["samples"])
