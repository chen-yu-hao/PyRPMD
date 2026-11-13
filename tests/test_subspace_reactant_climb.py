"""Reactant continuations that climb out of a pre-reaction complex (issue #191).

Steepest descent from a submerged barrier ends in the pre-reaction complex,
but both rates of the correction are normalized in the *free* reactant
channel.  The reactant continuation therefore climbs out of the complex
along decreasing rate coordinate ``xi`` -- each point the minimum of ``V``
on the hyperplane one step further along ``-grad xi`` -- until ``xi = 0``,
where the fragments are ``R_inf`` apart: the reactant reference of the rate
itself.  The models put the complex at the corner of an L-shaped valley:
the path arrives along a transfer coordinate and must leave along the
fragment separation, so climbing on along the incoming tangent would climb
a wall.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_correction import validity_gates
from pyrpmd.analysis.subspace_model import SubspaceModel, assemble_reaction_path, relaxed_path
from pyrpmd.input_schema import InputError
from pyrpmd.input_subspace import rate_subspace_options

# Separation R: a Morse complex near R = 4 and an ion-like -C/R^4 tail.
D, A, C = 6e-3, 1.0, 2.0


def separation(R):
    return D * (np.exp(-2 * A * (R - 4)) - 2 * np.exp(-A * (R - 4))) - C / (R**4 + 50.0)


def separation_slope(R):
    return D * (-2 * A * np.exp(-2 * A * (R - 4)) + 2 * A * np.exp(-A * (R - 4))) + 4 * C * R**3 / (R**4 + 50.0) ** 2


# Transfer coordinate r: barrier at r = 0, reactant minimum near r = -1.
def transfer(r):
    return 0.02 * (r**2 - 1) ** 2 - 0.01 * r


def transfer_slope(r):
    return 0.08 * r * (r**2 - 1) - 0.01


def potential(q):
    return separation(q[0]) + transfer(q[1])


def gradient(q):
    return np.array([separation_slope(q[0]), transfer_slope(q[1])])


R_INF = 25.0


def xi(q):
    """Linear stand-in for the rate coordinate: 0 at R_inf, decreasing outwards."""
    return (R_INF - q[0]) / (R_INF - 4.0), np.array([-1.0 / (R_INF - 4.0), 0.0])


# The core runs along the transfer coordinate at the complex separation.
CORE = np.column_stack([np.full(9, 4.0), np.linspace(-0.5, 0.5, 9)])
OPTIONS = dict(step=0.25, rigid_body=False)


def test_the_reactant_continuation_climbs_out_of_the_complex_to_xi_zero():
    start, direction = CORE[0], CORE[0] - CORE[1]
    points, info = relaxed_path(potential, gradient, start, direction, [1.0, 1.0], exit_coordinate=xi,
                                return_info=True, **OPTIONS)
    assert info["status"] == "coordinate_target_reached" and not info["converged"]
    (complex_,) = info["minima"]
    k = complex_["point_index"]
    bottom = points[k]
    # The complex: the transfer minimum at the separation minimum.
    assert bottom[1] == pytest.approx(-0.93, abs=0.02) and complex_["gradient_norm"] < 1e-3
    assert complex_["energy"] == pytest.approx(potential(bottom))
    # Before it the path descends along r; after it, R grows monotonically
    # to R_inf while r stays relaxed -- it never turns back into the well.
    assert np.allclose(points[: k + 1, 0], 4.0, atol=0.4)
    climb = points[k:]
    assert np.all(np.diff(climb[:, 0]) > 0)
    assert np.allclose(climb[1:, 1], bottom[1], atol=1e-4)
    assert R_INF <= points[-1][0] < R_INF + 0.3
    coordinate = info["exit_coordinate"]
    assert coordinate["start"] == pytest.approx(xi(bottom)[0]) and coordinate["target"] == 0.0
    assert -0.02 < coordinate["end"] <= 0.0
    assert info["unresolved_turns"] == 0 and info["end_energy"] == pytest.approx(potential(points[-1]))


def test_a_flat_tail_before_xi_zero_ends_the_climb_as_an_asymptote():
    far = lambda q: ((60.0 - q[0]) / 56.0, np.array([-1.0 / 56.0, 0.0]))
    points, info = relaxed_path(potential, gradient, CORE[0], CORE[0] - CORE[1], [1.0, 1.0],
                                exit_coordinate=far, return_info=True, **OPTIONS)
    assert info["status"] == "asymptote_converged" and info["converged"]
    assert len(info["minima"]) == 1 and 20.0 < points[-1][0] < 60.0
    assert info["tail_energy_range"] < info["asymptote_tolerance"]


def test_a_failing_exit_coordinate_is_a_termination_status():
    def broken(q):
        if q[0] > 6.0:
            raise ValueError("RPMDrate umbrella left the regular branch")
        return xi(q)

    _, info = relaxed_path(potential, gradient, CORE[0], CORE[0] - CORE[1], [1.0, 1.0],
                           exit_coordinate=broken, return_info=True, **OPTIONS)
    assert info["status"] == "potential_failure" and "regular branch" in info["message"]
    with pytest.raises(TypeError, match="callable"):
        relaxed_path(potential, gradient, CORE[0], CORE[0] - CORE[1], [1.0, 1.0], exit_coordinate=1.0, **OPTIONS)


def test_assembled_path_accepts_the_reactant_reference_and_a_product_well():
    # Mirror the reactant side onto the product side, with a deeper complex
    # that commits products (the separated products are never reached).
    def full(q):
        return potential(q) - 0.03 * np.exp(-((q[1] - 1.0) ** 2) / 0.3)

    def full_gradient(q):
        g = gradient(q)
        g[1] += 0.03 * 2 * (q[1] - 1.0) / 0.3 * np.exp(-((q[1] - 1.0) ** 2) / 0.3)
        return g

    path, info = assemble_reaction_path(CORE, full, full_gradient, [1.0, 1.0], product_well_drop=1e-2,
                                        reactant_coordinate=xi, **OPTIONS)
    reactant, product = info["reactant_termination"], info["product_termination"]
    assert reactant["status"] == "coordinate_target_reached" and reactant["accepted"] == "reactant_reference"
    assert product["status"] == "minimum_reached" and product["accepted"] == "product_well"
    assert info["termini_accepted"] and info["reactant_target"] == 0.0
    assert "-grad(reactant coordinate)" in info["continuation"]
    assert path[0][0] >= R_INF
    gates = validity_gates({"path_termination": {"reactant": reactant, "product": product}})
    assert gates["valid"]
    # The product side never climbs: only the reactant side has a reference.
    assert product["exit_coordinate"] is None and len(reactant["minima"]) == 1


# --- a molecule in 3D: ion + diatomic, rigid-body modes removed ------------

MASSES = np.array([2.0, 1.0, 1.0])


def _internals(x):
    g = x.reshape(3, 3)
    centre = (g[1] + g[2]) / 2.0
    axis = g[2] - g[1]
    separation_vector = g[0] - centre
    R, bond = np.linalg.norm(separation_vector), np.linalg.norm(axis)
    return R, bond, separation_vector @ axis / (R * bond)


def ion_molecule(x):
    R, bond, cosine = _internals(x)
    # Complex near R = 4 with the ion on the bond axis; a stiff bond.
    return separation(R) + 0.3 * (bond - 1.4) ** 2 + 2e-3 * (1 - cosine**2) * np.exp(-(R - 4.0) / 3.0)


def numeric_gradient(f, h=1e-6):
    def g(x):
        x = np.asarray(x, dtype=float)
        return np.array([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(x.size)])
    return g


def ion_xi(x):
    """0 at R = 12, as the rate coordinate is at R_inf; gradient along the ion--COM line."""
    g = x.reshape(3, 3)
    vector = g[0] - (g[1] + g[2]) / 2.0
    R = np.linalg.norm(vector)
    u = vector / R
    derivative = np.concatenate([u, -u / 2.0, -u / 2.0]) * (-1.0 / 8.0)
    return (12.0 - R) / 8.0, derivative


def test_a_molecular_complex_is_left_along_xi_with_rigid_body_modes_removed():
    # Core on the inner wall of the complex, ion approaching along the bond
    # axis, centre of mass fixed at the origin (ion at R/2, bond centre at -R/2).
    core = np.array([[0.0, 0.0, R / 2, 0.0, 0.0, -R / 2 - 0.7, 0.0, 0.0, -R / 2 + 0.7] for R in (3.0, 2.8, 2.6)])
    masses = np.repeat(MASSES, 3)
    path, info = assemble_reaction_path(core, ion_molecule, numeric_gradient(ion_molecule), masses,
                                        step=0.25, rigid_body=True, reactant_coordinate=ion_xi,
                                        product_well_drop=0.0)
    reactant = info["reactant_termination"]
    assert reactant["status"] == "coordinate_target_reached" and reactant["accepted"] == "reactant_reference"
    (complex_,) = reactant["minima"]
    R_complex = _internals(path[info["core_slice"][0] - complex_["point_index"]])[0]
    assert R_complex == pytest.approx(3.65, abs=0.1)
    R_end, bond_end, cosine_end = _internals(path[0])
    assert 12.0 <= R_end < 12.3 and bond_end == pytest.approx(1.4, abs=1e-3) and abs(cosine_end) > 0.99
    # No rigid-body drift: the centre of mass stays where it was.
    com = lambda x: (x.reshape(3, 3) * MASSES[:, None]).sum(axis=0) / MASSES.sum()
    np.testing.assert_allclose(com(path[0]), com(core[0]), atol=1e-6)


# --- the 1D reference slice ---------------------------------------------------

def test_reactant_reference_stays_in_place_moves_into_the_free_channel_or_rejects():
    axis = np.linspace(0.0, 10.0, 201)
    beta, reference_kt = 1.0 / 1e-3, 0.3          # tolerance 3e-4
    flat_long = np.where(axis < 6.0, 0.0, -0.01 * np.exp(-(axis - 8.0) ** 2))
    kept = pipeline._reactant_reference(axis, flat_long, 8.0, 2.0, 0.5, beta, reference_kt)
    assert kept["rule"] == "default" and kept["reference"] == pytest.approx(2.5) and kept["width"] == 2.0
    # The channel is free only up to 1.8: the slice moves there, the absorber shrinks.
    short = np.where(axis < 1.8, 0.0, -0.01 * (axis - 1.8))
    moved = pipeline._reactant_reference(axis, short, 8.0, 2.0, 0.5, beta, reference_kt)
    assert moved["rule"] == "free_channel"
    assert moved["reference"] == pytest.approx(1.8, abs=0.05) and moved["width"] == pytest.approx(1.3, abs=0.05)
    assert moved["offset_kt"] <= reference_kt
    # A reactant end in a sloped well: no free channel at all.
    sloped = -0.01 * np.exp(-axis)
    rejected = pipeline._reactant_reference(axis, sloped, 8.0, 2.0, 0.5, beta, reference_kt)
    assert rejected["rule"] == "not_free" and rejected["offset_kt"] > reference_kt
    converged = {"status": "asymptote_converged", "converged": True}
    gates = validity_gates({"path_termination": {"reactant": converged, "product": converged},
                            "reactant_reference": rejected})
    assert not gates["valid"] and "no free reactant channel" in gates["failures"][0]
    assert gates["gates"][0]["detail"]["reactant_reference"]["rule"] == "not_free"
    assert validity_gates({"path_termination": {"reactant": converged, "product": converged},
                           "reactant_reference": moved})["valid"]


def test_path_model_moves_a_sloped_reference_slice_into_the_free_channel(monkeypatch):
    axis = np.linspace(0.0, 10.0, 201)
    # The free channel is only 1.8 long, then V falls into a complex well.
    values = np.where(axis < 1.8, 0.0, -0.004 * np.sin(np.pi * np.clip(axis - 1.8, 0, 4.2) / 4.2)) \
        + 0.01 * np.exp(-(axis - 7.0) ** 2) - np.where(axis > 7.0, 0.03 * (axis - 7.0) / 3.0, 0.0)

    def projection(*args, provenance=None, **kwargs):
        return SubspaceModel((axis,), values, (1.0,), 7.0, "asymptotic", None, provenance)

    monkeypatch.setattr(pipeline, "project_path_subspace", projection)
    path = np.column_stack([axis, np.zeros((len(axis), 2))])
    run = SimpleNamespace(cartesian_masses=np.ones(3), beta=1.0 / 1e-3, xi_star=7.0,
                          energy=lambda x: float(np.interp(x[0], axis, values)), gradient=lambda x: np.zeros(3),
                          xi=lambda x: float(x[0]))
    model, quantum, info = pipeline.path_model(run, path, dict(pipeline.DEFAULTS["quantum"]))
    reference = model.provenance["reactant_reference"]
    # The channel is flat to 0.3 kT a little past the start of the slope at
    # 1.8, but the kink there is not quantum-free: the slice stops before it.
    assert reference["rule"] == "free_channel" and 1.6 <= model.reactant_reference <= 1.8
    assert reference["offset_kt"] <= reference["reference_kt"]
    assert reference["density_mismatch"] <= reference["density_tolerance"]
    assert quantum["cap_width"][0] == pytest.approx(reference["width"])
    assert info["reactant_reference"] == model.reactant_reference
    assert info["model_construction"]["reactant_reference"]["rule"] == "free_channel"


# --- wiring -----------------------------------------------------------------

def test_build_path_climbs_along_the_rate_coordinate_to_reactant_xi(monkeypatch):
    seen = {}

    def assemble(core, *args, **kwargs):
        seen.update(kwargs)
        return np.asarray(core), {}

    monkeypatch.setattr(pipeline, "assemble_reaction_path", assemble)
    mean = SimpleNamespace(xi=np.linspace(0.6, 1.1, 6), locked=lambda factor: np.ones(6, bool),
                           mean_centroid=np.zeros((6, 2, 3)) + np.arange(6)[:, None, None],
                           orientation_spread=np.zeros(6))
    monkeypatch.setattr(pipeline, "mean_path_from_rate_run", lambda root: mean)
    calls = []
    run = SimpleNamespace(root=None, beta=2.0, system=SimpleNamespace(masses=np.ones(2)),
                          cartesian_masses=np.ones(6), energy=lambda x: 0.0, gradient=lambda x: np.zeros(6),
                          xi_value_gradient=lambda x: calls.append(x) or (0.5, np.ones(6)))
    pipeline.build_path(run, {"source": "centroid", "smoothing": None, "product_drop_kt": 15.0})
    assert seen["reactant_target"] == 0.0 and seen["product_well_drop"] == pytest.approx(7.5)
    assert seen["reactant_coordinate"](np.zeros(6)) == (0.5, pytest.approx(np.ones(6))) and len(calls) == 1
    pipeline.build_path(run, {"source": "centroid", "smoothing": None, "reactant_xi": -0.05})
    assert seen["reactant_target"] == -0.05 and seen["product_well_drop"] is None


def test_reactant_xi_input_is_checked():
    def config(value):
        return {"task": "rate", "rate": {"subspace_correction": {"enabled": False, "path": {"reactant_xi": value}}}}

    assert rate_subspace_options(config(-0.1))["path"]["reactant_xi"] == -0.1
    assert rate_subspace_options(config(0))["path"]["reactant_xi"] == 0.0
    with pytest.raises(InputError, match="reactant side"):
        rate_subspace_options(config(1.0))
    with pytest.raises(InputError, match="reactant_xi"):
        rate_subspace_options(config("0"))
