"""Shared scalar validation at public configuration and numerical boundaries.

Integer validation never truncates floats or accepts booleans. Real validation
distinguishes numeric inputs from explicitly coercible inputs. Callers can keep
their domain-specific exception type and diagnostic wording without maintaining
another implementation of the type, finiteness and bounds checks.
"""

from __future__ import annotations

import math
import numbers
from functools import partial

import numpy as np


def checked_integer(
    value,
    name,
    minimum=1,
    maximum=None,
    *,
    error_type=ValueError,
    message=None,
    type_message=None,
    bounds_message=None,
    maximum_message=None,
):
    """Return an integer within inclusive bounds, rejecting bools and floats.

    Diagnostic templates may use ``{name}``, ``{minimum}`` and ``{maximum}``.
    ``error_type`` may also be an exception factory, for lazy input frontends.
    """
    fields = {
        "name": name,
        "minimum": minimum,
        "maximum": maximum,
        "bounds": f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}",
    }
    default = "{name} must be an integer"
    if minimum is not None:
        default += " >= {minimum}"
    if maximum is not None:
        default += " and <= {maximum}"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise error_type((type_message or message or default).format(**fields))
    result = int(value)
    if maximum is not None and result > maximum and maximum_message is not None:
        raise error_type(maximum_message.format(**fields))
    if (minimum is not None and result < minimum) or (
        maximum is not None and result > maximum
    ):
        raise error_type((bounds_message or message or default).format(**fields))
    return result


positive_integer = partial(checked_integer, message="{name} must be a positive integer")
nonnegative_integer = partial(
    checked_integer, minimum=0, message="{name} must be a nonnegative integer"
)


def integer_count(value, name, minimum=0, maximum=None, **options):
    """An integer count, allowing zero unless a caller requests another bound."""
    return checked_integer(value, name, minimum, maximum, **options)


def integer_value(value, name, minimum=None, maximum=None, **options):
    """An integer identity/index with no implicit sign restriction."""
    return checked_integer(value, name, minimum, maximum, **options)


def finite_real(
    value,
    name,
    *,
    positive=False,
    nonnegative=False,
    coerce=False,
    allow_bool=False,
    allow_strings=True,
    shape=None,
    error_type=ValueError,
    message=None,
    type_message=None,
    finite_message=None,
    bounds_message=None,
):
    """Return a finite real, with optional sign and input-type constraints.

    By default only real numeric scalars are accepted. ``coerce=True`` permits
    inputs understood by ``float`` (for legacy numerical APIs). ``shape`` can
    restrict those inputs to NumPy scalars (``"scalar"``) or also permit a
    zero-dimensional array (``"zero_dim"``). Booleans remain opt-in. Set
    ``allow_strings=False`` when a frontend has already parsed unit strings.
    """
    if shape not in (None, "scalar", "zero_dim"):
        raise ValueError("shape must be None, 'scalar' or 'zero_dim'")
    sign = "positive" if positive else "nonnegative" if nonnegative else "finite"
    fields = {
        "name": name,
        "sign": sign,
        "bound": " > 0" if positive else " >= 0" if nonnegative else "",
        "and_sign": " and " + sign if positive or nonnegative else "",
    }
    type_error = type_message or message or "{name} must be a finite real number"
    if (
        (not allow_bool and isinstance(value, (bool, np.bool_)))
        or (not allow_strings and isinstance(value, (str, bytes)))
        or np.iscomplexobj(value)
        or (not coerce and not isinstance(value, numbers.Real))
        or (shape == "scalar" and not np.isscalar(value))
        or (shape == "zero_dim" and np.ndim(value) != 0)
    ):
        raise error_type(type_error.format(**fields))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise error_type(type_error.format(**fields)) from error
    if not math.isfinite(result):
        template = finite_message or message or "{name} must be finite"
        raise error_type(template.format(**fields))
    if (positive and result <= 0) or (nonnegative and result < 0):
        default = "{name} must be " + ("positive" if positive else "nonnegative")
        raise error_type((bounds_message or message or default).format(**fields))
    return result


def finite_positive(
    value,
    name,
    allow_zero=False,
    *,
    error_type=ValueError,
    allow_bool=False,
    message=None,
    zero=None,
    **options,
):
    """Check a positive (or nonnegative) coercible finite real value.

    ``zero`` retains the keyword used by the thermal and rate-analysis helpers.
    New callers should use ``allow_zero``.
    """
    if zero is not None:
        allow_zero = zero
    options.setdefault("coerce", True)
    return finite_real(
        value,
        name,
        positive=not allow_zero,
        nonnegative=allow_zero,
        allow_bool=allow_bool,
        error_type=error_type,
        message=message
        or (
            "{name} must be finite and " + ("nonnegative" if allow_zero else "positive")
        ),
        **options,
    )


def positive_quantity(value, name, dimension=None):
    """Resolve a unit-bearing quantity before checking its positive value."""
    from .units import parse_quantity

    return finite_positive(
        parse_quantity(value, dimension),
        name,
        message="{name} must be positive and finite",
    )


def positive_number(value, name, *, integer=False):
    """Validate adapter controls that may require an integer or a real scalar."""
    if integer:
        return positive_integer(value, name)
    return finite_real(
        value,
        name,
        positive=True,
        shape="scalar",
        message="{name} must be a positive number",
    )
