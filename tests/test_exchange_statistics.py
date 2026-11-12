"""Joint sign/numerator statistics checked against explicit block arithmetic."""

import math

import numpy as np
import pytest

from pyrpmd.analysis.fes import statistical_inefficiency
from pyrpmd.quantum_statistics import exchange_thermodynamics


def columns(sign, numerator=None):
    sign = np.asarray(sign, dtype=float)
    numerator = 2 * sign if numerator is None else np.asarray(numerator, dtype=float)
    return {"sign": sign,
            "potential_energy_numerator_hartree": numerator,
            "kinetic_energy_numerator_hartree": 0.5 * numerator,
            "internal_energy_numerator_hartree": 1.5 * numerator}


def warning_codes(result):
    return {item["code"] for item in result["warnings"]}


def test_ratio_and_uncertainty_match_independent_joint_block_jackknife():
    sign = np.repeat([1.0, 0.7, 0.8, 0.9, 0.6, 0.95], 3)
    numerator = np.repeat([2.1, 3.7, 1.2, 4.0, 2.8, 1.5], 3)
    data = columns(sign, numerator)
    result = exchange_thermodynamics(data, block_size=3)
    block_sign = sign.reshape(6, 3).mean(axis=1)
    block_numerator = numerator.reshape(6, 3).mean(axis=1)
    # Independently form each retained sample set, rather than reuse the
    # subtraction formula in the implementation.
    leave = np.array([np.delete(block_numerator, i).mean()
                      / np.delete(block_sign, i).mean() for i in range(6)])
    expected_error = math.sqrt(5 / 6 * float(np.sum((leave - leave.mean())**2)))
    assert result["average_sign"] == pytest.approx(sign.mean())
    assert result["average_sign_standard_error"] == pytest.approx(block_sign.std(ddof=1) / math.sqrt(6))
    assert result["potential_energy_hartree"] == pytest.approx(numerator.sum() / sign.sum())
    assert result["potential_energy_standard_error_hartree"] == pytest.approx(expected_error)
    assert result["kinetic_energy_hartree"] == pytest.approx(0.5 * numerator.sum() / sign.sum())
    assert result["internal_energy_standard_error_hartree"] == pytest.approx(1.5 * expected_error)
    assert result["sign_resolved"]
    assert result["warnings"] == []


def test_numerator_denominator_covariance_cancels_for_constant_observable():
    sign = np.array([0.6, 1.0, 0.8, 0.9, 0.7, 0.95, 0.8, 1.0])
    result = exchange_thermodynamics(columns(sign), block_size=1)
    assert result["average_sign_standard_error"] > 0
    assert result["potential_energy_hartree"] == pytest.approx(2.0)
    assert result["potential_energy_standard_error_hartree"] < 1e-14
    assert result["kinetic_energy_standard_error_hartree"] < 1e-14
    assert result["internal_energy_standard_error_hartree"] < 1e-14


def test_signed_numerators_are_not_multiplied_by_sign_a_second_time():
    sign = np.array([1.0, 0.6, 0.8, 0.7, 0.9, 0.8])
    numerator = np.array([1.0, 3.0, 1.0, 0.5, 2.0, 4.0])
    result = exchange_thermodynamics(columns(sign, numerator), block_size=1)
    assert result["potential_energy_hartree"] == pytest.approx(numerator.mean() / sign.mean())
    assert abs(result["potential_energy_hartree"] - (sign * numerator).mean() / sign.mean()) > 0.1


def test_row_input_and_optional_observables_preserve_units_and_errors():
    data = columns(np.ones(8), np.arange(1.0, 9.0))
    data["virial_internal_energy_numerator_hartree"] = np.arange(1.0, 9.0) / 2
    data["pressure_numerator_au"] = np.arange(1.0, 9.0) * 1e-5
    rows = [{key: float(value[i]) for key, value in data.items()} for i in range(8)]
    from_rows = exchange_thermodynamics(rows, block_size=1)
    from_columns = exchange_thermodynamics(data, block_size=1)
    assert from_rows == from_columns
    assert from_rows["pressure_au"] == pytest.approx(4.5e-5)
    assert from_rows["pressure_standard_error_au"] == pytest.approx(np.std(data["pressure_numerator_au"], ddof=1) / math.sqrt(8))
    assert from_rows["virial_internal_energy_hartree"] == pytest.approx(2.25)
    assert from_rows["average_sign"] == 1.0
    assert from_rows["average_sign_standard_error"] == 0.0


def test_mapping_order_does_not_change_report_after_checkpoint_json_sorting():
    data = columns(np.ones(8), np.arange(1.0, 9.0))
    data["virial_internal_energy_numerator_hartree"] = np.arange(1.0, 9.0) / 2
    reverse = {key: data[key] for key in reversed(data)}
    sorted_rows = [{key: float(data[key][i]) for key in sorted(data)} for i in range(8)]
    expected = exchange_thermodynamics(data, block_size=1)
    assert exchange_thermodynamics(reverse, block_size=1) == expected
    assert exchange_thermodynamics(sorted_rows, block_size=1) == expected


def test_npt_volume_and_enthalpy_use_signed_numerators_and_joint_errors():
    sign = np.array([1.0, 0.7, 0.9, 0.8, 0.6, 0.95])
    volume = np.array([2.0, 5.0, 3.0, 4.0, 7.0, 3.0])
    pressure = 0.2
    data = columns(sign)
    data["volume_numerator_bohr3"] = sign * volume
    data["enthalpy_numerator_hartree"] = (
        data["internal_energy_numerator_hartree"] + pressure * sign * volume)
    result = exchange_thermodynamics(data, block_size=1)
    expected_volume = np.sum(sign * volume) / sign.sum()
    leave_volume = np.array([np.delete(sign * volume, i).sum()
                             / np.delete(sign, i).sum() for i in range(len(sign))])
    expected_error = math.sqrt(5 / 6 * np.sum((leave_volume - leave_volume.mean())**2))
    assert result["volume_bohr3"] == pytest.approx(expected_volume)
    assert result["volume_bohr3"] != pytest.approx(volume.mean())
    assert result["volume_standard_error_bohr3"] == pytest.approx(expected_error)
    assert result["enthalpy_hartree"] == pytest.approx(3.0 + pressure * expected_volume)
    assert result["enthalpy_standard_error_hartree"] == pytest.approx(pressure * expected_error)


def test_unresolved_sign_withholds_npt_volume_and_enthalpy():
    data = columns(np.full(8, 0.01))
    data["volume_numerator_bohr3"] = np.full(8, 0.05)
    data["enthalpy_numerator_hartree"] = np.full(8, 0.2)
    result = exchange_thermodynamics(data, block_size=1)
    assert result["volume_bohr3"] is None
    assert result["volume_standard_error_bohr3"] is None
    assert result["enthalpy_hartree"] is None
    assert result["enthalpy_standard_error_hartree"] is None
    assert "average_sign_below_threshold" in warning_codes(result)


def test_block_remainder_is_explicit_and_all_estimates_use_same_prefix():
    data = columns(np.r_[np.ones(8), -1.0], np.r_[np.arange(8.0), 1e6])
    result = exchange_thermodynamics(data, block_size=2)
    assert result["samples"] == 9
    assert result["diagnostics"]["samples_used"] == 8
    assert result["diagnostics"]["discarded_tail_frames"] == 1
    assert result["average_sign"] == 1.0
    assert result["potential_energy_hartree"] == pytest.approx(3.5)


def test_automatic_blocking_detects_slow_ratio_influence_hidden_by_large_offset():
    rng = np.random.default_rng(382)
    count = 4000
    sign = 0.8 + 0.1 * rng.choice([-1, 1], count)
    noise = rng.normal(size=count)
    slow = np.empty(count)
    slow[0] = noise[0]
    for i in range(1, count):
        slow[i] = 0.95 * slow[i - 1] + noise[i]
    numerator = 1000 * sign + 0.01 * slow
    data = columns(sign, numerator)
    result = exchange_thermodynamics(data)
    diagnostics = result["diagnostics"]
    assert statistical_inefficiency(numerator) < 2
    assert diagnostics["statistical_inefficiency"] > 15
    assert diagnostics["block_length_frames"] >= diagnostics["statistical_inefficiency"]
    assert "potential_energy_numerator_hartree_ratio_influence" in diagnostics["correlation_observables"]
    assert result["potential_energy_standard_error_hartree"] > 0


@pytest.mark.parametrize("sign,kwargs,code", [
    (np.zeros(8), {}, "nonpositive_average_sign"),
    (-np.ones(8), {}, "nonpositive_average_sign"),
    (np.full(8, 0.01), {}, "average_sign_below_threshold"),
    ([1.0, 1.0, 1.0, -1.0], {}, "sign_uncertainty_reaches_zero"),
    ([1.0, -0.1, -0.1, -0.1], {"sign_sigma": 0.1}, "nonpositive_jackknife_sign"),
])
def test_unresolved_sign_reports_warning_and_suppresses_all_physical_conclusions(sign, kwargs, code):
    data = columns(sign)
    data["pressure_numerator_au"] = np.ones(len(sign))
    data["virial_internal_energy_numerator_hartree"] = np.ones(len(sign))
    result = exchange_thermodynamics(data, block_size=1, **kwargs)
    assert code in warning_codes(result)
    assert result["sign_resolved"] is False
    assert result["average_sign_standard_error"] is not None
    for key in ("potential_energy_hartree", "kinetic_energy_hartree", "internal_energy_hartree",
                "virial_internal_energy_hartree", "pressure_au", "pressure_standard_error_au",
                "potential_energy_standard_error_hartree", "internal_energy_standard_error_hartree"):
        assert result[key] is None


@pytest.mark.parametrize("count", [1, 2, 3])
def test_too_few_blocks_keep_sign_mean_but_no_error_or_energy(count):
    result = exchange_thermodynamics(columns(np.ones(count)), block_size=1)
    assert result["average_sign"] == 1
    assert result["average_sign_standard_error"] is None
    assert result["internal_energy_hartree"] is None
    assert "fewer_than_four_complete_blocks" in warning_codes(result)


@pytest.mark.parametrize("change", [
    {"sign": []}, {"sign": [True] * 8}, {"sign": [1.1] * 8},
    {"sign": [float("nan")] * 8}, {"sign": [float("inf")] * 8},
    {"sign": [[1.0]] * 8}, {"sign": [1.0] * 7},
    {"potential_energy_numerator_hartree": [complex(1, 1)] * 8},
    {"typo": np.ones(8)},
])
def test_invalid_estimator_columns_fail_explicitly(change):
    data = columns(np.ones(8))
    data.update(change)
    with pytest.raises(ValueError):
        exchange_thermodynamics(data, block_size=1)


@pytest.mark.parametrize("data", [None, [], {}, {"sign": [1]}, [{"sign": 1}, {"other": 1}], [1, 2]])
def test_invalid_input_structure_fails_explicitly(data):
    with pytest.raises(ValueError):
        exchange_thermodynamics(data)


@pytest.mark.parametrize("kwargs", [
    {"minimum_average_sign": 0}, {"minimum_average_sign": 1.1},
    {"minimum_average_sign": True}, {"minimum_average_sign": float("nan")},
    {"sign_sigma": 0}, {"sign_sigma": True}, {"sign_sigma": float("inf")},
    {"block_size": 0}, {"block_size": 9}, {"block_size": 1.5}, {"block_size": True},
])
def test_invalid_diagnostic_controls_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        exchange_thermodynamics(columns(np.ones(8)), **kwargs)


def test_sign_roundoff_is_accepted_without_clipping_input():
    sign = np.full(8, 1 + np.finfo(float).eps)
    result = exchange_thermodynamics(columns(sign), block_size=1)
    assert result["average_sign"] > 1
    np.testing.assert_array_equal(sign, np.full(8, 1 + np.finfo(float).eps))


def test_large_finite_constant_offset_does_not_overflow_means_or_correlation():
    data = {key: np.full(8, 1e308) for key in (
        "potential_energy_numerator_hartree", "kinetic_energy_numerator_hartree",
        "internal_energy_numerator_hartree")}
    data["sign"] = np.ones(8)
    result = exchange_thermodynamics(data)
    assert result["internal_energy_hartree"] == 1e308
    assert result["internal_energy_standard_error_hartree"] == 0.0
    data["sign"] = np.full(8, 0.1)
    with pytest.raises(ValueError, match="overflows"):
        exchange_thermodynamics(data)
