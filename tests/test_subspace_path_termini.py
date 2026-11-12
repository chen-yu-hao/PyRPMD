"""Path termini in front of pre- and post-reaction complexes (issue #191).

An ion--molecule reaction such as H2 + NH2- -> NH3 + H- has a submerged
barrier between two complexes.  Steepest descent from the barrier ends in
those complexes; the old follower kept climbing out along the incoming
tangent, wandered for thousands of steps and reported
``asymptote_not_approached``.  A minimum is now a recorded terminus: accepted
on the product side once it is deep enough below the barrier to commit
products, never on the reactant side, whose normalization needs the free
reactant channel.
"""

import numpy as np
import pytest

from pyrpmd.analysis.subspace_correction import validity_gates
from pyrpmd.analysis.subspace_model import SubspaceModel, assemble_reaction_path, relaxed_path
from pyrpmd.analysis.subspace_pipeline import _product_cap
from pyrpmd.analysis.subspace_quantum import exact_quantum_rate

# Reaction profile along x: free reactants (V -> 0) at x -> -inf, a
# pre-reaction complex at x = -2, a submerged barrier at x = 0, a deep
# post-reaction complex at x = 2 and the separated products (V -> 0 again)
# beyond it.  The valley bends (y = 0.1 sin x) and is harmonic across.
WELLS = ((-4e-3, -2.0, 1.0), (2e-3, 0.0, 0.4), (-2.4e-2, 2.0, 0.8))


def profile(x):
    return sum(a * np.exp(-(x - c) ** 2 / w) for a, c, w in WELLS)


def profile_slope(x):
    return sum(-2.0 * a * (x - c) / w * np.exp(-(x - c) ** 2 / w) for a, c, w in WELLS)


def potential(q):
    x, y = q
    return profile(x) + 0.1 * (y - 0.1 * np.sin(x)) ** 2


def gradient(q):
    x, y = q
    d = y - 0.1 * np.sin(x)
    return np.array([profile_slope(x) - 0.02 * d * np.cos(x), 0.2 * d])


CORE = np.column_stack([np.linspace(-0.8, 0.8, 9), 0.1 * np.sin(np.linspace(-0.8, 0.8, 9))])
OPTIONS = dict(step=0.25, rigid_body=False)


def test_descent_into_a_complex_stops_at_its_minimum():
    points, info = relaxed_path(potential, gradient, CORE[-1], CORE[-1] - CORE[-2], [1.0, 1.0],
                                return_info=True, **OPTIONS)
    assert info["status"] == "minimum_reached" and not info["converged"]
    assert points[-1][0] == pytest.approx(2.0, abs=0.02)
    assert info["end_energy"] == pytest.approx(potential([2.0, 0.1 * np.sin(2.0)]), abs=1e-6)
    assert info["gradient_norm"] < 1e-3
    # It ends where the valley ends instead of wandering: few steps, no climb.
    assert info["steps"] < 40 and np.all(points[:, 0] <= 2.0 + 0.02)


def test_a_deep_product_well_is_an_accepted_terminus_and_a_reactant_complex_is_not():
    path, info = assemble_reaction_path(CORE, potential, gradient, [1.0, 1.0], product_well_drop=1e-2,
                                        **OPTIONS)
    reactant, product = info["reactant_termination"], info["product_termination"]
    assert reactant["status"] == product["status"] == "minimum_reached"
    assert product["converged"] and product["accepted"] == "product_well"
    assert product["drop_below_top"] == pytest.approx(product["barrier_top_energy"] - product["end_energy"])
    assert product["drop_below_top"] > 1e-2
    assert not reactant["converged"] and reactant["accepted"] is None
    assert "pre-reaction complex" in reactant["reason"]
    assert path[0][0] == pytest.approx(-2.0, abs=0.02) and path[-1][0] == pytest.approx(2.0, abs=0.02)
    assert not info["termini_accepted"] and not info["asymptotes_converged"]

    gates = validity_gates({"path_termination": {"reactant": reactant, "product": product}})
    assert not gates["valid"]
    (reason,) = gates["failures"]
    assert "reactant: minimum_reached" in reason and "pre-reaction complex" in reason
    assert "product" not in reason
    detail = gates["gates"][0]["detail"]
    assert detail["product"]["accepted"] == "product_well"
    assert detail["reactant"]["end_energy"] == pytest.approx(reactant["end_energy"])


def test_a_shallow_or_unrequested_product_well_is_not_accepted():
    _, deep = assemble_reaction_path(CORE, potential, gradient, [1.0, 1.0], product_well_drop=5e-2, **OPTIONS)
    product = deep["product_termination"]
    assert not product["converged"] and "not committed" in product["reason"]
    assert product["required_drop"] == pytest.approx(5e-2)
    _, plain = assemble_reaction_path(CORE, potential, gradient, [1.0, 1.0], **OPTIONS)
    assert not plain["product_termination"]["converged"] and plain["product_well_drop"] is None
    with pytest.raises(ValueError, match="product_well_drop"):
        assemble_reaction_path(CORE, potential, gradient, [1.0, 1.0], product_well_drop=-1.0, **OPTIONS)


def test_an_accepted_product_well_and_a_reactant_asymptote_pass_the_path_gate():
    reactant = {"status": "asymptote_converged", "converged": True, "accepted": "asymptote", "steps": 80,
                "arc_length": 20.0, "tail_energy_range": 5e-7, "asymptote_tolerance": 1e-6}
    product = {"status": "minimum_reached", "converged": True, "accepted": "product_well", "steps": 12,
               "arc_length": 3.0, "tail_energy_range": None, "asymptote_tolerance": 1e-6,
               "drop_below_top": 2.6e-2, "required_drop": 1e-2}
    gates = validity_gates({"path_termination": {"reactant": reactant, "product": product}})
    assert gates["valid"] and gates["gates"][0]["passed"]


# --- the 1D quantum box of a path that ends in a product well ---------------

def test_product_absorber_covers_only_the_committed_part_of_a_terminal_well():
    axis = np.linspace(0.0, 10.0, 201)
    # Long reactant channel, barrier at 7, product well bottom at the end.
    values = np.where(axis < 7.0, 0.01 * np.exp(-(axis - 7.0) ** 2), 0.01 - 0.05 * (axis - 7.0) / 3.0)
    cap = _product_cap(axis, values, 7.0, default=2.0, drop=0.02)
    assert cap["rule"] == "committed_product_well"
    assert cap["committed_arc"] == pytest.approx(8.2, abs=0.05) and cap["width"] == pytest.approx(1.8, abs=0.05)
    # A product channel that runs on keeps the default absorber.
    longer = np.linspace(0.0, 14.0, 281)
    values = np.where(longer < 7.0, 0.01 * np.exp(-(longer - 7.0) ** 2), 0.01 - 0.05 * np.minimum(longer - 7.0, 3.0) / 3.0)
    assert _product_cap(longer, values, 7.0, default=2.0, drop=0.02) == {
        "width": 2.0, "committed_arc": pytest.approx(8.2, abs=0.05), "rule": "default"}
    # A channel that never drops far enough keeps the default too.
    assert _product_cap(axis, np.zeros_like(axis), 7.0, default=2.0, drop=0.02)["rule"] == "default"


def eckart(low=-6.0, high=6.0, points=241):
    axis = np.linspace(low, high, points)
    values = 0.01 / np.cosh(axis / 0.6) ** 2 - 0.004 * (1 + np.tanh(axis / 0.6))
    return SubspaceModel((axis,), values, (1836.0,), 0.0, "asymptotic", low + 0.25 * (high - low))


def test_a_cap_width_pair_reproduces_the_scalar_cap_and_checks_each_end():
    model, beta = eckart(), 1.0 / 9.5e-4
    scalar = exact_quantum_rate(model, beta, cap_width=2.4)
    pair = exact_quantum_rate(model, beta, cap_width=(2.4, 2.4))
    assert pair.rate == pytest.approx(scalar.rate, rel=1e-12)
    assert pair.diagnostics["cap_width"] == scalar.diagnostics["cap_width"] == 2.4
    narrow = exact_quantum_rate(model, beta, cap_width=(2.4, 1.6))
    assert narrow.diagnostics["cap_width"] == [2.4, 1.6]
    assert narrow.rate == pytest.approx(scalar.rate, rel=0.05)
    # Each end is checked against the dividing surface with its own width.
    near = eckart(-8.0, 4.0)
    assert exact_quantum_rate(near, beta, cap_width=(5.5, 3.5)).rate > 0
    with pytest.raises(ValueError, match="between the absorbing regions"):
        exact_quantum_rate(near, beta, cap_width=(3.5, 4.5))
    with pytest.raises(ValueError, match="pair"):
        exact_quantum_rate(model, beta, cap_width=(1.0, 1.0, 1.0))


def test_path_model_gives_a_terminal_product_well_its_own_absorber(monkeypatch):
    from types import SimpleNamespace

    from pyrpmd.analysis import subspace_pipeline as pipeline

    axis = np.linspace(0.0, 10.0, 201)
    # Long reactant channel, barrier at 7, a product well whose bottom ends the path.
    values = np.where(axis < 7.0, 0.01 * np.exp(-(axis - 7.0) ** 2), 0.01 - 0.05 * (axis - 7.0) / 3.0)

    def projection(*args, provenance=None, **kwargs):
        return SubspaceModel((axis,), values, (1.0,), 7.0, "asymptotic", None, provenance)

    monkeypatch.setattr(pipeline, "project_path_subspace", projection)
    path = np.column_stack([axis, np.zeros((len(axis), 2))])
    beta = 1.0 / 1e-3
    run = SimpleNamespace(cartesian_masses=np.ones(3), beta=beta, xi_star=7.0,
                          energy=lambda x: float(np.interp(x[0], axis, values)), gradient=lambda x: np.zeros(3),
                          xi=lambda x: float(x[0]))
    options = {**pipeline.DEFAULTS["quantum"], "product_drop_kt": 20.0}
    model, quantum, info = pipeline.path_model(run, path, options)
    assert info["product_absorber"]["rule"] == "committed_product_well"
    assert quantum["cap_width"][0] == pytest.approx(0.2 * 10.0)
    assert quantum["cap_width"][1] == pytest.approx(10.0 - info["product_absorber"]["committed_arc"])
    assert info["product_absorber"]["committed_arc"] == pytest.approx(8.2, abs=0.05)
    # The product threshold is read where products are absorbed, inside the well.
    assert info["channel_thresholds"][1] <= 0.01 - 20.0 / beta
    # With the default drop (15 kT = 0.015) the committed part is wider than the
    # default absorber, which is then kept unchanged.
    model, quantum, info = pipeline.path_model(run, path, dict(pipeline.DEFAULTS["quantum"]))
    assert info["product_absorber"]["rule"] == "default" and quantum["cap_width"] == pytest.approx(2.0)
