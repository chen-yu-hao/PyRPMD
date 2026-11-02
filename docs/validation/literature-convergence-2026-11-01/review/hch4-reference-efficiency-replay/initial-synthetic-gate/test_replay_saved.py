"""Small independent analytic checks; no actual dataset, PES or RNG."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("hch4_saved_calculation", Path(__file__).with_name("replay_saved.py"))
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def constants():
    return {"masses_electron_mass": [1., 12., 1., 1., 1., 1.],
            "beta_hartree_inverse": 2., "reference_difference_bohr": -.5}


def tetrahedron():
    return np.array([2., 2., -1/3, -math.sqrt(8)/3, math.sqrt(8/3), -2/3,
                     -math.sqrt(8)/3, -math.sqrt(8/3), -2/3])


@pytest.mark.parametrize("channel", replay.CHANNELS)
def test_cartesian_gradient_surface_measure_has_analytic_collinear_value(channel):
    z = tetrahedron()
    answer = replay.coordinate_measure(z, [0., 0., 1.], channel, 1., constants())
    expected = 4*math.log(2.)+2*math.log(1.5)+.5*math.log((5.+1/12)/(4*math.pi))
    assert answer["common_zero_reason"] is None
    assert answer["domains"] == {name: True for name in replay.DOMAINS}
    assert answer["log_geometric_measure"] == pytest.approx(expected, abs=1e-13)
    q = np.asarray(answer["coordinates_bohr"])
    assert np.linalg.norm(q[5]-q[channel]) == pytest.approx(1.5)


def test_entry_measure_uses_one_sphere_and_reduced_fragment_mass():
    answer = replay.coordinate_measure(tetrahedron(), [0., 0., 1.], 3, None, constants(), 30.)
    reduced_mass = 16/17
    expected = 4*math.log(2.)+2*math.log(30.)-.5*math.log(4*math.pi*reduced_mass)
    assert answer["log_geometric_measure"] == pytest.approx(expected, abs=1e-13)


def test_invalid_latent_shape_keeps_zero_instead_of_replacing_point():
    z = tetrahedron(); z[2] = 1.1
    answer = replay.coordinate_measure(z, [0., 0., 1.], 3, .999, constants())
    assert answer["common_zero_reason"] == "invalid_chart"
    assert answer["coordinates_bohr"] is None and not any(answer["domains"].values())


@pytest.mark.parametrize("entry", [False, True])
def test_full_mixture_center_matches_closed_form_normal_and_student_density(entry):
    models = {}
    for target in replay.TARGETS:
        dimension = 9 if target == "entry" else 11
        mean = np.zeros(dimension); mean[:2] = 2.
        models[target] = {"mean_x": mean.tolist(), "covariance_x": np.eye(dimension).tolist()}
    mixture = replay.FullMixture(models, entry)
    z = np.zeros(9); z[:2] = 2.
    d = 9 if entry else 11
    jac = 4*math.pi if entry else 4.
    uniform = 1/(8*(3.8**2*2*10**6)*4*math.pi)
    normal = (2*math.pi)**(-d/2)
    student = math.gamma((5+d)/2)/math.gamma(2.5)/(5*math.pi)**(d/2)/2.4**(d/2)
    expected = uniform+.75*normal/jac+.125*student/jac
    assert math.exp(mixture(z, [0., 0., 1.])) == pytest.approx(expected, rel=2e-14)
    assert math.exp(mixture(z, [0., 0., -1.])) >= uniform*(1-1e-14)


def test_stereographic_jacobian_matches_explicit_surface_derivatives():
    t = np.array([.3, -.7]); u, jac = replay.stereo(t)
    step = 1e-5
    derivatives = [(replay.stereo(t+step*np.eye(2)[i])[0]-replay.stereo(t-step*np.eye(2)[i])[0])/(2*step) for i in range(2)]
    assert np.dot(u, u) == pytest.approx(1.)
    assert np.linalg.norm(np.cross(*derivatives)) == pytest.approx(jac, rel=1e-9)


def test_unconditional_zeros_remain_in_mean_and_concentration_count():
    result = replay.summary([math.log(2), None, None, math.log(2)])
    assert result["draws"] == 4 and result["positive_weights"] == 2
    assert result["log_mean"] == pytest.approx(0.)
    assert result["empirical_ESS"] == pytest.approx(2.)
    assert result["maximum_normalized_weight"] == pytest.approx(.5)


def test_four_chart_means_and_one_entry_mean_have_no_extra_channel_multiplier():
    rows, weights = [], {}
    for target in replay.TARGETS:
        for channel in ([None] if target == "entry" else replay.CHANNELS):
            for repeat in range(2):
                index = len(rows)
                rows.append({"index": index, "target": target, "channel": channel})
                weights[index] = {"original": 0.}
    result = replay.integral_vector(rows, weights)
    assert result["entry"] == pytest.approx(0.)
    assert result["alpha_0.999"] == result["alpha_1.007"] == pytest.approx(math.log(4))


def test_seed_vector_ratio_uses_pooled_means_and_preserves_covariance():
    logs = np.log(np.arange(1., 9.))
    result = replay.covariance_ratio(logs, [0.]*8)
    expected_se = np.std(np.arange(1., 9.)/4.5, ddof=1)/math.sqrt(8)
    assert result["log_ratio_of_pooled_means"] == pytest.approx(math.log(4.5))
    assert result["log_ratio_of_pooled_means"] != pytest.approx(float(logs.mean()))
    assert result["nominal_delta_standard_error"] == pytest.approx(expected_se)
    paired = replay.covariance_ratio(logs, logs+math.log(3))
    assert paired["log_ratio_of_pooled_means"] == pytest.approx(-math.log(3))
    assert paired["nominal_delta_standard_error"] < 1e-15


def test_zero_integral_is_unresolved_without_epsilon():
    result = replay.covariance_ratio([None]*8, [0.]*8)
    assert result["resolved"] is False and result["log_ratio_of_pooled_means"] is None


def test_physical_gate_booleans_are_compared_exactly():
    with pytest.raises(AssertionError, match="exact value"):
        replay.Comparison().tree({"passed": True}, {"passed": False})
