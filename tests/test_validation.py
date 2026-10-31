"""Scientific boundary contracts shared by inputs, dynamics and analysis."""

from decimal import Decimal
from fractions import Fraction
import importlib

import numpy as np
import pytest

from pyrpmd import validation
from pyrpmd.input import InputError
from pyrpmd.trajectory import TrajectoryError


@pytest.mark.parametrize("value", [1, np.int8(1), np.int64(1), np.uint64(1)])
def test_bead_counts_preserve_integral_scalars(value):
    result = validation.checked_integer(value, "beads")
    assert type(result) is int
    assert result == 1


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), np.bool_(False), 1.0, np.float64(1),
     1.5, float("nan"), float("inf"), "1", None, np.array(1), [1]],
)
def test_bead_counts_never_truncate_or_treat_flags_as_counts(value):
    with pytest.raises(ValueError, match="^beads must be an integer >= 1$"):
        validation.checked_integer(value, "beads")


@pytest.mark.parametrize(
    "helper, accepted, rejected",
    [(validation.positive_integer, 1, 0),
     (validation.nonnegative_integer, 0, -1),
     (validation.integer_count, 0, -1)],
)
def test_positive_sizes_and_zero_allowed_counts_have_distinct_defaults(helper, accepted, rejected):
    assert helper(accepted, "count") == accepted
    with pytest.raises(ValueError):
        helper(rejected, "count")


def test_charge_can_be_negative_while_atom_indices_cannot():
    from pyrpmd.system import _integer

    assert _integer(-2, "Charge") == -2
    assert validation.integer_value(-2, "Charge") == -2
    with pytest.raises(ValueError, match="^Fixed atom index must be >= 0$"):
        _integer(-1, "Fixed atom index", 0)


@pytest.mark.parametrize("value", [2, 3, 4, np.int64(2), np.uint64(4)])
def test_inclusive_integer_bounds_accept_both_endpoints(value):
    assert validation.checked_integer(value, "states", 2, 4) == int(value)


@pytest.mark.parametrize("value", [1, 5, np.int64(1), np.uint64(5)])
def test_inclusive_integer_bounds_reject_outside_values(value):
    with pytest.raises(ValueError, match="^states must be an integer >= 2 and <= 4$"):
        validation.checked_integer(value, "states", 2, 4)


def test_native_counter_widths_are_checked_before_conversion_to_c():
    from pyrpmd.native_ensemble import _integer as native_integer
    from pyrpmd.plumed import _integer as plumed_integer
    from pyrpmd.trajectory import _integer as trajectory_integer

    uint64_max = 2**64 - 1
    assert native_integer(np.uint64(uint64_max), "seed") == uint64_max
    with pytest.raises(ValueError, match="^seed is out of range$"):
        native_integer(uint64_max + 1, "seed")

    int64_max = int(np.iinfo(np.int64).max)
    assert trajectory_integer(np.int64(int64_max), "frames") == int64_max
    for overflow in (int64_max + 1, np.uint64(int64_max + 1)):
        with pytest.raises(TrajectoryError, match="^frames must be an integer >= 0$"):
            trajectory_integer(overflow, "frames")

    c_int_max = int(np.iinfo(np.intc).max)
    assert plumed_integer(c_int_max, "step", 0) == c_int_max
    with pytest.raises(ValueError, match="^step exceeds the PLUMED setStep integer range$"):
        plumed_integer(c_int_max + 1, "step", 0)


@pytest.mark.parametrize(
    "module, name, value, options, message",
    [
        ("input", "_integer", True, {}, "beads must be an integer >= 1"),
        ("input", "_integer", 0, {}, "beads must be an integer >= 1"),
        ("input", "_positive", 0.0, {}, "beads must be positive and finite"),
        ("workflows.state_selected", "_positive_integer", 0, {},
         "beads must be a positive integer"),
        ("pcet_input", "_integer", 5, {"minimum": 1, "maximum": 4},
         "beads must be 1..4"),
        ("pcet_input", "_real", float("inf"), {"positive": True},
         "beads must be finite and positive"),
        ("pcet_input", "_real", "1", {}, "beads must be a finite real number"),
    ],
)
def test_input_boundaries_keep_inputerror_and_actionable_messages(module, name, value, options, message):
    helper = getattr(importlib.import_module("pyrpmd." + module), name)
    with pytest.raises(InputError) as caught:
        helper(value, "beads", **options)
    assert str(caught.value) == message


@pytest.mark.parametrize("value", [0.5, np.float32(0.5), np.float64(0.5), Fraction(1, 2)])
def test_finite_real_accepts_numeric_real_scalars(value):
    assert validation.finite_real(value, "mass", positive=True) == 0.5


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), "0.5", None, float("nan"), float("inf"),
     float("-inf"), 1 + 0j, 1 + 1j, np.complex128(1), np.complex128(1 + 1j)],
)
def test_real_physical_controls_reject_flags_nonfinite_and_complex_values(value):
    with pytest.raises(ValueError):
        validation.finite_real(value, "mass", positive=True)


@pytest.mark.parametrize("value", [np.complex64(1), np.complex128(1 + 2j)])
def test_float_coercion_never_silently_discards_an_imaginary_component(value):
    # NumPy scalar-to-float conversion can otherwise warn and retain only the
    # real component, admitting a complex mass/timestep into propagation.
    with pytest.raises(ValueError):
        validation.finite_positive(value, "mass")


def test_float_overflow_cannot_enter_a_physical_control():
    with pytest.raises(ValueError, match="mass must be a finite real number"):
        validation.finite_real(10**400, "mass")
    with pytest.raises(ValueError, match="mass must be finite and positive"):
        validation.finite_positive(10**400, "mass")


@pytest.mark.parametrize("zero", [0, 0.0, -0.0, np.float64(0)])
def test_positive_and_nonnegative_controls_distinguish_zero(zero):
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        validation.finite_positive(zero, "temperature")
    assert validation.finite_positive(zero, "friction", allow_zero=True) == 0.0
    assert validation.finite_real(zero, "friction", nonnegative=True) == 0.0


def test_small_real_values_do_not_get_an_artificial_zero_cutoff():
    tiny = np.nextafter(0.0, 1.0)
    assert validation.finite_positive(tiny, "timestep") == tiny
    with pytest.raises(ValueError, match="friction must be nonnegative"):
        validation.finite_real(-tiny, "friction", nonnegative=True)


@pytest.mark.parametrize("module, name", [("thermal_nrpmd", "_finite_positive"),
                                          ("analysis.rate_postprocess", "_positive")])
def test_zero_keyword_remains_compatible_for_existing_analysis_callers(module, name):
    helper = getattr(importlib.import_module("pyrpmd." + module), name)
    assert helper(0.0, "rate", zero=True) == 0.0
    assert helper(0.0, "rate", allow_zero=True) == 0.0
    with pytest.raises(ValueError, match="rate must be finite and positive"):
        helper(0.0, "rate")
    with pytest.raises(ValueError, match="rate must be finite and nonnegative"):
        helper(-1.0, "rate", zero=True)


@pytest.mark.parametrize("module, name", [("pcet", "_real"),
                                          ("path_integral", "_positive"),
                                          ("analysis.rate_io", "_positive")])
@pytest.mark.parametrize("value", ["0.5", Decimal("0.5")])
def test_existing_coercible_numerical_apis_keep_accepting_real_values(module, name, value):
    helper = getattr(importlib.import_module("pyrpmd." + module), name)
    assert helper(value, "mass") == 0.5


@pytest.mark.parametrize("module, name", [("barostat", "_real"),
                                          ("native_ensemble", "_scalar"),
                                          ("analysis.spectroscopy", "_positive"),
                                          ("pcet_input", "_real")])
def test_strict_numerical_apis_do_not_start_coercing_strings(module, name):
    helper = getattr(importlib.import_module("pyrpmd." + module), name)
    with pytest.raises(ValueError):
        helper("0.5", "mass")


@pytest.mark.parametrize("value", [Decimal("0.5"), np.array(0.5)])
def test_parsed_input_values_keep_numeric_conversion_support(value):
    from pyrpmd.input import _positive

    assert _positive(value, "mass") == 0.5


@pytest.mark.parametrize("value", ["1", b"1", np.str_("1"), np.bytes_(b"1")])
def test_raw_strings_do_not_bypass_input_quantity_parsing(value):
    from pyrpmd.input import _positive

    with pytest.raises(InputError, match="^mass must be positive and finite$"):
        _positive(value, "mass")


def test_scalar_shape_contracts_survive_shared_validation():
    from pyrpmd.instanton import _positive as instanton_positive
    from pyrpmd.plumed import _positive as plumed_positive

    assert instanton_positive(np.array(0.5), "mass") == 0.5
    with pytest.raises(ValueError, match="mass must be positive and finite"):
        plumed_positive(np.array(0.5), "mass")
    for vector in ([0.5], np.array([0.5]), np.array([0.5, 1.0])):
        with pytest.raises(ValueError, match="mass must be a positive finite scalar"):
            instanton_positive(vector, "mass")


def test_unit_bearing_initial_state_controls_still_convert_to_atomic_units():
    from pyrpmd.constants import EV_TO_AU
    from pyrpmd.initial_states import _positive

    assert _positive("2 eV", "energy", "energy") == pytest.approx(2 * EV_TO_AU)
    with pytest.raises(ValueError, match="energy must be positive and finite"):
        _positive("0 eV", "energy", "energy")
