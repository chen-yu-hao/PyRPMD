"""Independent balance, equilibrium, and limiting-case HER checks."""

from dataclasses import asdict
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyrpmd.electrochemistry.kinetics import (
    FARADAY_C_MOL,
    HERRates,
    her_fluxes,
    steady_state_her,
)


RESERVOIRS = {
    "proton_activity": 1.0,
    "hydrogen_activity": 1.0,
    "site_density_mol_m2": 2.5e-5,
}


def test_volmer_heyrovsky_irreversible_cycle_has_two_electrons_per_hydrogen():
    rates = HERRates(volmer_forward_s=3.0, heyrovsky_forward_s=7.0)
    state = steady_state_her(rates, **RESERVOIRS)
    assert state.coverage == pytest.approx(0.3)
    assert state.vacancy == pytest.approx(0.7)
    assert state.volmer_s == pytest.approx(2.1)
    assert state.heyrovsky_s == pytest.approx(2.1)
    assert state.tafel_s == 0
    assert state.hydrogen_tof_s == pytest.approx(2.1)
    assert state.coverage_rate_s == pytest.approx(0, abs=2e-15)
    assert state.current_density_A_m2 == pytest.approx(
        -2 * FARADAY_C_MOL * RESERVOIRS["site_density_mol_m2"] * 2.1
    )


def test_volmer_tafel_cycle_includes_two_adsorbates_per_tafel_event():
    # 2(1-theta)=2*3*theta^2 gives theta=(-1+sqrt(13))/6.
    rates = HERRates(volmer_forward_s=2.0, tafel_forward_s=3.0)
    state = steady_state_her(rates, **RESERVOIRS)
    expected = (math.sqrt(13) - 1) / 6
    assert state.coverage == pytest.approx(expected)
    assert state.volmer_s == pytest.approx(2 * state.tafel_s)
    assert state.hydrogen_tof_s == pytest.approx(3 * expected**2)
    assert state.heyrovsky_s == 0
    assert state.current_density_A_m2 == pytest.approx(
        -2 * FARADAY_C_MOL * RESERVOIRS["site_density_mol_m2"] * state.hydrogen_tof_s
    )


def test_activity_dependence_and_adsorption_equilibrium():
    rates = HERRates(volmer_forward_s=2.0, volmer_reverse_s=5.0)
    for activity in (0.0, 0.01, 1.0, 100.0):
        reservoir = dict(RESERVOIRS, proton_activity=activity)
        state = steady_state_her(rates, **reservoir)
        assert state.coverage == pytest.approx(2 * activity / (2 * activity + 5))
        assert state.current_density_A_m2 == pytest.approx(0, abs=2e-14)
        assert state.hydrogen_tof_s == 0


def test_all_three_reactions_obey_constructed_detailed_balance():
    # K_V=2, K_H=3, K_T=K_H/K_V=1.5; equilibrium aH2=6*aH^2.
    rates = HERRates(4.0, 2.0, 9.0, 3.0, 3.0, 2.0)
    reservoir = dict(RESERVOIRS, proton_activity=0.4, hydrogen_activity=0.96)
    state = steady_state_her(rates, **reservoir)
    assert state.coverage == pytest.approx(0.8 / 1.8)
    assert state.volmer_s == pytest.approx(0, abs=1e-14)
    assert state.heyrovsky_s == pytest.approx(0, abs=1e-14)
    assert state.tafel_s == pytest.approx(0, abs=1e-14)
    assert state.current_density_A_m2 == pytest.approx(0, abs=1e-13)


def test_reversible_hydrogen_oxidation_retains_signed_tof_and_anodic_current():
    rates = HERRates(volmer_reverse_s=5.0, heyrovsky_reverse_s=3.0)
    state = steady_state_her(rates, **dict(RESERVOIRS, hydrogen_activity=2.0))
    assert state.coverage == pytest.approx(6 / 11)
    assert state.volmer_s == pytest.approx(-30 / 11)
    assert state.heyrovsky_s == pytest.approx(-30 / 11)
    assert state.hydrogen_tof_s == pytest.approx(-30 / 11)
    assert state.current_density_A_m2 > 0
    assert state.current_density_A_m2 == pytest.approx(
        -2 * FARADAY_C_MOL * RESERVOIRS["site_density_mol_m2"] * state.hydrogen_tof_s
    )


def test_transient_current_includes_surface_hydrogen_accumulation():
    rates = HERRates(volmer_forward_s=3.0)
    state = her_fluxes(0.25, rates, **RESERVOIRS)
    assert state.coverage_rate_s == pytest.approx(2.25)
    assert state.hydrogen_tof_s == 0
    assert state.current_density_A_m2 == pytest.approx(
        -FARADAY_C_MOL * RESERVOIRS["site_density_mol_m2"] * 2.25
    )
    # This identity must hold even before the coverage reaches a steady state.
    assert -state.current_density_A_m2 / (
        FARADAY_C_MOL * RESERVOIRS["site_density_mol_m2"]
    ) == pytest.approx(state.coverage_rate_s + 2 * state.hydrogen_tof_s)


def test_pure_reversible_tafel_has_square_root_equilibrium_and_no_current():
    rates = HERRates(tafel_forward_s=2.0, tafel_reverse_s=8.0)
    state = steady_state_her(rates, **RESERVOIRS)
    assert state.coverage == pytest.approx(2 / 3)
    assert state.vacancy == pytest.approx(1 / 3)
    assert state.hydrogen_tof_s == pytest.approx(0, abs=1e-15)
    assert state.current_density_A_m2 == 0


@pytest.mark.parametrize(
    "rates,coverage",
    [
        (HERRates(volmer_forward_s=3), 1.0),
        (HERRates(heyrovsky_reverse_s=3), 1.0),
        (HERRates(tafel_reverse_s=3), 1.0),
        (HERRates(volmer_reverse_s=3), 0.0),
        (HERRates(heyrovsky_forward_s=3), 0.0),
        (HERRates(tafel_forward_s=3), 0.0),
    ],
)
def test_exact_boundary_steady_states(rates, coverage):
    state = steady_state_her(rates, **RESERVOIRS)
    assert state.coverage == coverage
    assert state.vacancy == 1 - coverage
    assert state.coverage_rate_s == 0
    assert state.current_density_A_m2 == 0


def test_zero_rates_and_zero_reactant_activities_do_not_invent_unique_coverage():
    for rates, reservoirs in (
        (HERRates(), RESERVOIRS),
        (HERRates(volmer_forward_s=2), dict(RESERVOIRS, proton_activity=0)),
        (HERRates(tafel_reverse_s=2), dict(RESERVOIRS, hydrogen_activity=0)),
    ):
        with pytest.raises(ValueError, match="not unique"):
            steady_state_her(rates, **reservoirs)
        state = her_fluxes(0.31, rates, **reservoirs)
        assert state.coverage == 0.31
        assert state.coverage_rate_s == state.hydrogen_tof_s == state.current_density_A_m2 == 0


def test_independent_ode_relaxation_agrees_with_algebraic_steady_state():
    rates = HERRates(1.7, 0.23, 0.39, 1.1, 4.2, 0.17)
    reservoir = dict(RESERVOIRS, proton_activity=0.37, hydrogen_activity=0.82)
    state = steady_state_her(rates, **reservoir)

    def equation(_time, y):
        theta = y[0]
        return [
            1.7 * 0.37 * (1 - theta) - 0.23 * theta
            - 0.39 * 0.37 * theta + 1.1 * 0.82 * (1 - theta)
            - 2 * 4.2 * theta**2 + 2 * 0.17 * 0.82 * (1 - theta)**2
        ]

    for initial in (0.0, 1.0):
        result = solve_ivp(equation, (0, 30), [initial], method="Radau", rtol=1e-11, atol=1e-13)
        assert result.success
        assert result.y[0, -1] == pytest.approx(state.coverage, abs=1e-12)
    assert state.coverage_rate_s == pytest.approx(0, abs=3e-15)


def test_global_rate_scaling_changes_flux_but_not_steady_coverage():
    rates = HERRates(1.3, 2.8, 3.7, 0.61, 0.17, 0.33)
    reference = steady_state_her(rates, **RESERVOIRS)
    for scale in (1e-200, 1e200):
        scaled = HERRates(**{key: value * scale for key, value in asdict(rates).items()})
        state = steady_state_her(scaled, **RESERVOIRS)
        assert state.coverage == pytest.approx(reference.coverage, rel=3e-15)
        assert state.current_density_A_m2 / scale == pytest.approx(reference.current_density_A_m2)
        assert state.hydrogen_tof_s / scale == pytest.approx(reference.hydrogen_tof_s)


def test_tiny_vacancy_is_preserved_when_coverage_rounds_to_one():
    rates = HERRates(volmer_forward_s=1e100, heyrovsky_forward_s=1e-100)
    state = steady_state_her(rates, **RESERVOIRS)
    assert state.coverage == 1.0
    assert state.vacancy == pytest.approx(1e-200, rel=1e-15, abs=0)
    assert state.volmer_s == pytest.approx(1e-100, rel=1e-15, abs=0)
    assert state.heyrovsky_s == pytest.approx(1e-100, rel=1e-15, abs=0)
    assert state.coverage_rate_s == 0
    assert state.hydrogen_tof_s == pytest.approx(1e-100, rel=1e-15, abs=0)


def test_small_pair_equilibrium_uses_square_root_without_cancellation():
    state = steady_state_her(HERRates(tafel_forward_s=1.0, tafel_reverse_s=1e-300), **RESERVOIRS)
    assert state.coverage == pytest.approx(1e-150, rel=1e-14, abs=0)
    assert state.vacancy == 1.0


def test_surface_site_density_changes_current_and_leaves_per_site_kinetics():
    rates = HERRates(volmer_forward_s=2, heyrovsky_forward_s=3)
    original = steady_state_her(rates, **RESERVOIRS)
    changed = steady_state_her(rates, **dict(RESERVOIRS, site_density_mol_m2=7.5e-5))
    assert changed.coverage == original.coverage
    assert changed.hydrogen_tof_s == original.hydrogen_tof_s
    assert changed.current_density_A_m2 == pytest.approx(3 * original.current_density_A_m2)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, 1j, [1.0], "2"])
@pytest.mark.parametrize("field", list(HERRates.__dataclass_fields__))
def test_invalid_rate_coefficients_raise(field, bad):
    with pytest.raises(ValueError, match=field):
        HERRates(**{field: bad})


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True, [0.2]])
def test_invalid_coverage_raises(bad):
    with pytest.raises(ValueError, match="coverage"):
        her_fluxes(bad, HERRates(), **RESERVOIRS)


@pytest.mark.parametrize("field", ["proton_activity", "hydrogen_activity", "site_density_mol_m2"])
@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, [1.0]])
def test_invalid_reservoir_inputs_raise(field, bad):
    with pytest.raises(ValueError, match=field):
        steady_state_her(HERRates(volmer_forward_s=1), **dict(RESERVOIRS, **{field: bad}))


def test_zero_site_density_is_not_a_valid_active_surface():
    with pytest.raises(ValueError, match="positive"):
        her_fluxes(0.5, HERRates(), **dict(RESERVOIRS, site_density_mol_m2=0))


def test_overflow_and_unrepresentable_rate_ratios_raise_instead_of_inf_or_nan():
    with pytest.raises(ValueError, match="numerical range"):
        steady_state_her(HERRates(volmer_forward_s=1e300), **dict(RESERVOIRS, proton_activity=1e300))
    with pytest.raises(ValueError, match="underflow"):
        steady_state_her(HERRates(volmer_forward_s=1e-300), **dict(RESERVOIRS, proton_activity=1e-300))
    with pytest.raises(ValueError, match="rate ratios"):
        steady_state_her(HERRates(volmer_forward_s=1e300, heyrovsky_forward_s=1e-300), **RESERVOIRS)
    with pytest.raises(ValueError, match="numerical range"):
        her_fluxes(0.0, HERRates(volmer_forward_s=1e308), **dict(RESERVOIRS, site_density_mol_m2=1e100))


def test_numpy_real_scalars_are_supported_and_boolean_is_rejected():
    rates = HERRates(volmer_forward_s=np.float64(2), heyrovsky_forward_s=np.int64(3))
    assert steady_state_her(rates, **RESERVOIRS).coverage == pytest.approx(0.4)
    with pytest.raises(ValueError):
        HERRates(volmer_forward_s=np.bool_(True))


def test_rates_type_is_explicit():
    with pytest.raises(TypeError, match="HERRates"):
        steady_state_her({"volmer_forward_s": 1}, **RESERVOIRS)
