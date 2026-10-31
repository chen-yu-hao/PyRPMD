"""Single-site mean-field Volmer--Heyrovsky--Tafel postprocessing.

This module consumes independently supplied elementary rate coefficients.  It
does not infer them from a PMF, an electrode potential, or local PCET rates.
Coefficients are effective per-site values in s^-1; activities and coverages
are dimensionless.  In particular, the Tafel coefficient already contains the
chosen pair-counting/geometrical factor: no further coordination number or
factor of one half is inserted here.

The reactions and positive directions are

    Volmer:    H+ + e- + *  -> H*
    Heyrovsky: H* + H+ + e- -> H2 + *
    Tafel:     2 H*         -> H2 + 2 *.

Reservoir activities are externally fixed.  The model assumes an ideal
well-mixed surface, one adsorbed H per site, and uncorrelated site occupations.
It includes neither transport nor lateral interactions.  A caller must check
thermodynamic consistency of independently supplied forward/reverse rates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import numbers

from scipy.constants import elementary_charge

from ..constants import N_A


FARADAY_C_MOL = N_A * elementary_charge


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return value


@dataclass(frozen=True)
class HERRates:
    """Effective per-site forward/reverse coefficients in s^-1.

    ``forward`` follows hydrogen evolution.  Proton activity multiplies the
    forward Volmer and Heyrovsky coefficients.  Hydrogen activity multiplies
    the reverse Heyrovsky and Tafel coefficients.  All activities are relative
    to the standard states used to define these coefficients.  Electron
    chemical potential is already represented in the coefficients, so there
    is no separate electron activity.
    """

    volmer_forward_s: float = 0.0
    volmer_reverse_s: float = 0.0
    heyrovsky_forward_s: float = 0.0
    heyrovsky_reverse_s: float = 0.0
    tafel_forward_s: float = 0.0
    tafel_reverse_s: float = 0.0

    def __post_init__(self):
        for field in fields(self):
            object.__setattr__(
                self, field.name, _nonnegative(getattr(self, field.name), field.name)
            )


@dataclass(frozen=True)
class HERFluxes:
    """Signed net elementary fluxes and observables for one surface state.

    All ``*_s`` values are per site per second, except ``coverage_rate_s``,
    which is d(theta)/dt in s^-1.  ``hydrogen_tof_s`` counts H2 molecules per
    site per second and may be negative for net hydrogen oxidation.
    ``current_density_A_m2`` is positive for anodic electron production and
    negative for cathodic electron consumption.  No absolute value is applied.

    ``vacancy`` is reported separately to retain a small vacancy fraction when
    a steady-state coverage rounds to exactly one in floating point.
    """

    coverage: float
    vacancy: float
    volmer_s: float
    heyrovsky_s: float
    tafel_s: float
    coverage_rate_s: float
    hydrogen_tof_s: float
    current_density_A_m2: float


def _effective(rates, proton_activity, hydrogen_activity, site_density_mol_m2):
    if not isinstance(rates, HERRates):
        raise TypeError("rates must be HERRates")
    proton = _nonnegative(proton_activity, "proton_activity")
    hydrogen = _nonnegative(hydrogen_activity, "hydrogen_activity")
    sites = _nonnegative(site_density_mol_m2, "site_density_mol_m2")
    if sites == 0:
        raise ValueError("site_density_mol_m2 must be positive")
    coefficients = (
        rates.volmer_forward_s * proton,
        rates.volmer_reverse_s,
        rates.heyrovsky_forward_s * proton,
        rates.heyrovsky_reverse_s * hydrogen,
        rates.tafel_forward_s,
        rates.tafel_reverse_s * hydrogen,
    )
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("activity-weighted rate coefficients exceed numerical range")
    if any(
        coefficient == 0 and rate > 0 and activity > 0
        for coefficient, rate, activity in (
            (coefficients[0], rates.volmer_forward_s, proton),
            (coefficients[2], rates.heyrovsky_forward_s, proton),
            (coefficients[3], rates.heyrovsky_reverse_s, hydrogen),
            (coefficients[5], rates.tafel_reverse_s, hydrogen),
        )
    ):
        raise ValueError("activity-weighted rate coefficients underflow numerical range")
    scale = max(coefficients)
    scaled = tuple(value / scale for value in coefficients) if scale else coefficients
    if any(value > 0 and reduced == 0 for value, reduced in zip(coefficients, scaled)):
        raise ValueError("activity-weighted rate ratios exceed numerical range")
    return scaled, scale, sites


def _fluxes(coverage, vacancy, scaled, scale, sites):
    a, b, c, d, e, f = scaled
    volmer = math.fsum((a * vacancy, -b * coverage))
    heyrovsky = math.fsum((c * coverage, -d * vacancy))
    tafel = math.fsum((e * coverage * coverage, -f * vacancy * vacancy))
    coverage_rate = math.fsum((volmer, -heyrovsky, -2 * tafel))
    hydrogen_tof = math.fsum((heyrovsky, tafel))
    electrons = math.fsum((volmer, heyrovsky))
    values = (
        volmer * scale,
        heyrovsky * scale,
        tafel * scale,
        coverage_rate * scale,
        hydrogen_tof * scale,
        -electrons * scale * sites * FARADAY_C_MOL,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("HER fluxes or current exceed numerical range")
    return HERFluxes(coverage, vacancy, *values)


def her_fluxes(
    coverage,
    rates,
    *,
    proton_activity,
    hydrogen_activity,
    site_density_mol_m2,
):
    """Evaluate net fluxes at an explicitly supplied H coverage.

    With ``v = 1-theta`` and dimensionless activities ``aH`` and ``aH2``:

    ``rV = kV+ aH v - kV- theta``;
    ``rH = kH+ aH theta - kH- aH2 v``;
    ``rT = kT+ theta**2 - kT- aH2 v**2``.

    The balance is ``dtheta/dt = rV-rH-2*rT``.  Current is
    ``j = -F * site_density_mol_m2 * (rV+rH)`` in A/m^2.  Thus at a steady
    state it equals ``-2 F * site_density_mol_m2 * hydrogen_tof_s``.  Away
    from a steady state, surface H accumulation prevents that replacement.
    """
    theta = _nonnegative(coverage, "coverage")
    if theta > 1:
        raise ValueError("coverage must be between zero and one")
    scaled, scale, sites = _effective(
        rates, proton_activity, hydrogen_activity, site_density_mol_m2
    )
    return _fluxes(theta, 1.0 - theta, scaled, scale, sites)


def steady_state_her(
    rates,
    *,
    proton_activity,
    hydrogen_activity,
    site_density_mol_m2,
):
    """Return the unique mean-field steady H coverage and signed observables.

    The coverage balance is continuous and monotonically decreasing on [0,1]
    for nonnegative coefficients.  The physical root is evaluated with a
    cancellation-free quadratic expression.  Both occupancy and vacancy are
    solved directly, retaining small boundary populations.  Exactly empty or
    full surfaces are valid solutions.

    If all activity-weighted reactions vanish, every coverage is stationary;
    this function raises ``ValueError`` because no unique coverage is defined.
    Use :func:`her_fluxes` to evaluate an explicitly known coverage instead.
    A stationary state may sustain a net current and is not necessarily a
    thermodynamic equilibrium.
    """
    scaled, scale, sites = _effective(
        rates, proton_activity, hydrogen_activity, site_density_mol_m2
    )
    if scale == 0:
        raise ValueError("all activity-weighted rates are zero; steady coverage is not unique")
    a, b, c, d, e, f = scaled
    adsorption = a + d
    desorption = b + c
    pair_forward = 2 * e
    pair_reverse = 2 * f
    # For g(theta)=A(1-theta)-B theta-E theta^2+F(1-theta)^2,
    # D=(A+B)^2+4(A E+B F+E F) is manifestly nonnegative.
    discriminant = math.sqrt(
        (adsorption + desorption) ** 2
        + 4 * (
            adsorption * pair_forward
            + desorption * pair_reverse
            + pair_forward * pair_reverse
        )
    )
    theta_numerator = 2 * (adsorption + pair_reverse)
    vacancy_numerator = 2 * (desorption + pair_forward)
    if theta_numerator == 0:
        theta, vacancy = 0.0, 1.0
    elif vacancy_numerator == 0:
        theta, vacancy = 1.0, 0.0
    else:
        theta = theta_numerator / (
            adsorption + desorption + 2 * pair_reverse + discriminant
        )
        vacancy = vacancy_numerator / (
            adsorption + desorption + 2 * pair_forward + discriminant
        )
    return _fluxes(theta, vacancy, scaled, scale, sites)
