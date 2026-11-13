"""Reaction-path Hamiltonian 2D subspace on the MEP (``subspace.model_2d = "rph"``).

The ribbon ``x(s, q) = w(s) + q n(s)`` around the MEP, with a Bishop normal,
has the metric ``diag((1 - kappa q)^2, 1)`` and is unrolled into a plane in
which the reaction-path kinetic energy is flat.  For a two-dimensional system
the development is congruent to the system, so ``V_S`` must reproduce the
potential itself; in three dimensions the normal carries only part of the
curvature and the rest of the bath enters adiabatically.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_rph import develop_path, rph_bath
from pyrpmd.input_schema import InputError
from pyrpmd.input_subspace import rate_subspace_options

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.subspace_rph.models import CM, Curved, Embedded, model_run  # noqa: E402
from benchmarks.subspace_rph.reference import QUANTUM, build_path  # noqa: E402

GATING = {"validity": {}, "hessian_displacement": 5e-3}


def rph(run, path, info, **options):
    return pipeline.rph_model(run, path, {**QUANTUM, "environment": "adiabatic", **GATING, "path_info": info,
                                          **options})


# --- the development ------------------------------------------------------------

def test_a_planar_path_unrolls_onto_itself():
    x = np.linspace(-8, 8, 161)
    path = np.column_stack([x, 0.8 * np.exp(-x**2 / 4)])
    development = develop_path(path, [1.0, 1.0], rigid_body=False, extend=3.0)
    assert development.captured_curvature == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    s, q = rng.uniform(3.0, development.s[-1] - 3.0, 300), rng.uniform(-1.0, 1.0, 300)
    c, _, normal = development.point(s)
    points = c + q[:, None] * normal          # around the developed curve, inside its curvature radius
    configurations = development.configuration(*development.locate(points))
    developed = np.linalg.norm(points[:, None] - points[None], axis=2)
    original = np.linalg.norm(configurations[:, None] - configurations[None], axis=2)
    assert np.abs(developed - original).max() < 2e-3


def test_the_bishop_ribbon_has_the_rph_metric_on_a_twisted_path():
    # A helix-like path in 3D: curvature and torsion everywhere.
    t = np.linspace(0, 3 * np.pi, 400)
    path = np.column_stack([4 * np.cos(t), 4 * np.sin(t), 1.5 * t])
    development = develop_path(path, [1.0, 1.0, 1.0], rigid_body=False)
    k = len(development.s) // 2
    s, ds = development.s[k], 1e-3
    for q in (-1.0, 0.5, 1.5):
        a, b = development.configuration([s - ds, s + ds], q)
        speed = np.linalg.norm(b - a) / (2 * ds)
        assert speed == pytest.approx(1.0 - development.kappa[k] * q, rel=2e-3)
    # The normal stays a unit normal.  A helix twists: its curvature direction
    # turns about the tangent while a Bishop normal does not, so one normal
    # coordinate carries only part of the curvature (the rest stays adiabatic).
    assert np.allclose(np.linalg.norm(development.n, axis=1), 1.0)
    assert np.abs(np.einsum("ij,ij->i", development.n, development.t)).max() < 1e-8
    assert 0.3 < development.captured_curvature < 0.8


# --- the model on model potentials --------------------------------------------------

@pytest.fixture(scope="module")
def curved_2d():
    potential = Curved(2)
    run = model_run(potential, 500.0, 16)
    path, info = build_path(run)
    return potential, run, path, info


def test_rph_v_s_of_a_two_dimensional_system_is_its_potential(curved_2d):
    potential, run, path, info = curved_2d
    model, quantum, model_info, rpmd_model = rph(run, path, info, spacing_2d=[4.0, 3.0])
    assert model_info["captured_curvature"] == pytest.approx(1.0, abs=1e-6)
    assert model.provenance["model_2d"] == "rph" and model.provenance["bath_stable"]
    # Developed coordinates differ from (x, y) by a congruence that fixes the
    # saddle (0, 10) at the origin; the normal's sign may mirror the plane.
    X, Y = np.meshgrid(*rpmd_model.axes, indexing="ij")
    keep = rpmd_model.values < model_info["truncation_energy"]
    points = np.column_stack([X[keep], Y[keep]])
    best = np.inf
    for mirror in (1.0, -1.0):
        for a in np.radians(np.linspace(-3, 3, 61)):
            c, s_ = np.cos(a), np.sin(a)
            xy = (points * [1.0, mirror]) @ np.array([[c, s_], [-s_, c]]) + np.array([0.0, 10.0])
            best = min(best, float(np.abs(potential.energy(xy) - rpmd_model.values[keep]).max()))
    # Against values up to the truncation energy (~0.04 Eh): the residual is
    # the linear interpolation of the developed curve.
    assert best < 2e-4
    assert model.reactant_reference in model.axes[0]     # snapped onto a DVR node
    assert model.dividing_surface == pytest.approx(0.0, abs=1e-6)


def test_rph_on_a_non_planar_path_keeps_the_third_mode_adiabatic():
    potential = Curved(3, omega_z=1200.0 * CM)
    run = model_run(potential, 500.0, 16)
    path, info = build_path(run)
    model, quantum, model_info, rpmd_model = rph(run, path, info, spacing_2d=[5.0, 4.0])
    assert 0.5 < model_info["captured_curvature"] < 0.999
    # The bath is the z mode: half its frequency, softened at the barrier.
    development = develop_path(path, run.cartesian_masses, rigid_body=False)
    k = int(np.argmax([run.energy(p) for p in path]))
    arcs = [float(np.interp(k, np.arange(len(path)), pipeline._arc(path, run.cartesian_masses)))]
    bath = rph_bath(run.gradient, development, arcs, run.cartesian_masses, rigid_body=False)
    softened = potential.omega_z * (1 - potential.soften_z)
    assert bath["energy"][0] == pytest.approx(0.5 * softened, rel=0.05)
    assert bath["bath_stable"]


def test_rph_is_invariant_under_embedding_and_rotation():
    base = Curved(2)
    embedded = Embedded(base, (900 * CM, 2500 * CM))
    values = []
    for potential in (base, embedded):
        run = model_run(potential, 500.0, 16)
        path, info = build_path(run)
        model, _, model_info, _ = rph(run, path, info, spacing_2d=[4.0, 3.0])
        values.append((model.values - model.values.min(), model_info["captured_curvature"]))
    (a, ca), (b, cb) = values
    assert ca == pytest.approx(cb, abs=1e-3)
    shape = tuple(min(i, j) for i, j in zip(a.shape, b.shape))
    low = (a[: shape[0], : shape[1]] < 0.03) & (b[: shape[0], : shape[1]] < 0.03)
    assert np.abs(a[: shape[0], : shape[1]] - b[: shape[0], : shape[1]])[low].max() < 2e-4


# --- wiring -------------------------------------------------------------------------

def test_reference_slices_are_snapped_onto_dvr_nodes():
    axis = np.linspace(-10, 10, 81)                       # spacing 0.25
    assert pipeline._grid_reference(axis, -7.26) == pytest.approx(-7.25)
    assert pipeline._grid_reference(axis, -7.1) == pytest.approx(-7.0)


def test_model_2d_switch_defaults_to_rph():
    config = {"task": "rate", "rate": {"subspace_correction": {"enabled": False}}}
    assert rate_subspace_options(config)["model_2d"] == "rph"
    config["rate"]["subspace_correction"]["model_2d"] = "plane"
    assert rate_subspace_options(config)["model_2d"] == "plane"
    config["rate"]["subspace_correction"]["model_2d"] = "irc"
    with pytest.raises(InputError, match="model_2d"):
        rate_subspace_options(config)
    assert pipeline.DEFAULTS["subspace"]["model_2d"] == "rph"
    assert pipeline.DEFAULTS["path"]["source"] == "mep"


def test_saved_two_dimensional_factors_are_reused_only_for_the_same_model():
    saved = SimpleNamespace(dimension=2, model={"construction": {"environment": "adiabatic", "model_2d": "plane"}})
    assert not pipeline._saved_model_matches_options(saved, 2, {"environment": "adiabatic"})
    assert pipeline._saved_model_matches_options(saved, 2, {"environment": "adiabatic", "model_2d": "plane"})
    saved.model["construction"]["model_2d"] = "rph"
    assert pipeline._saved_model_matches_options(saved, 2, {"environment": "adiabatic"})


def test_the_channel_stays_open_beyond_the_box(curved_2d):
    """No wall across the channel ends: ring polymers that leave must not bounce back.

    The quantum rate absorbs products before the box ends, so a wall there was
    invisible to it but sent recrossing ring polymers back across the
    dividing surface (kappa 0.84 instead of 0.98 on this model).
    """
    potential, run, path, info = curved_2d
    model, quantum, model_info, rpmd_model = rph(run, path, info, spacing_2d=[4.0, 3.0])
    wall = rpmd_model.values.max()
    # Along the path, from one end of the grid to the other, nothing reaches the wall.
    X, Y = np.meshgrid(*rpmd_model.axes, indexing="ij")
    floor_line = rpmd_model.values.min(axis=1)
    assert floor_line.max() < 0.5 * wall
    # Both grid ends are flat channels near the asymptote.
    assert floor_line[0] == pytest.approx(floor_line[-1], abs=2e-3)
    assert floor_line[0] < model_info["energy_floor"] + 0.01


def test_the_reference_slice_lies_where_quantum_and_centroid_densities_agree():
    """On a slope the quantum and the centroid reactant density differ at O(hbar^2).

    The 0.3 kT energy rule alone put the 300 K reference slice at x = -71 on
    the curved-valley model, where beta^2 |V''| / 12 is ~0.15: C_S came out
    ~5% high for the RPH model, which is exact for a 2D system.
    """
    potential = Curved(2)
    run = model_run(potential, 300.0, 16)
    path, info = build_path(run)
    arc = pipeline._arc(path, run.cartesian_masses)
    energies = np.array([run.energy(p) for p in path])
    mismatch = pipeline._density_mismatch(arc, energies, run.beta)
    loose = pipeline._free_channel(energies, np.zeros_like(mismatch), run.beta, 0.3, np.inf)
    strict = pipeline._free_channel(energies, mismatch, run.beta, 0.3, 1e-2)
    edge = lambda free: path[int(np.argmin(free[: int(np.argmax(energies))])) - 1, 0]
    assert edge(loose) == pytest.approx(-71.0, abs=4.0)
    assert edge(strict) < -95.0
    # The mismatch of the analytic tail at x: beta^2 V''/12 + beta^3 V'^2/24.
    a, V0 = potential.a, potential.V0
    x = -80.0
    sech2, tanh = 1 / np.cosh(x / a) ** 2, np.tanh(x / a)
    d1 = -2 * V0 / a * sech2 * tanh
    d2 = 2 * V0 / a**2 * sech2 * (3 * tanh**2 - 1)
    exact = run.beta**2 * abs(d2) / 12 + run.beta**3 * d1**2 / 24
    k = int(np.argmin(abs(path[:, 0] - x)))
    assert mismatch[k] == pytest.approx(exact, rel=0.2)
