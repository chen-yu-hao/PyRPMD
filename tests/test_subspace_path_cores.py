"""Path cores on model potentials: centroid, centroid_all, mep and pmf_minimum.

A ring-polymer centroid is a thermal average, not a geometry: librating or
ring-opening coordinates have centroids shorter than any bond the polymer
visits, and freely rotating fragments have no mean geometry at all.  The
``mep`` and ``pmf_minimum`` cores are built on the potential instead.  They
are tested on a LEPS H + H2 surface, optionally with a damped ion--induced
dipole term that adds pre- and post-reaction complexes (the topology of an
ion--molecule reaction), with the real RPMDrate coordinate ``xi``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_paths import mass_weighted_hessian, refine_saddle
from pyrpmd.constants import KB_AU
from pyrpmd.cv import RPMDrateCoordinate
from pyrpmd.input_schema import InputError
from pyrpmd.input_subspace import rate_subspace_options

AMU = 1822.888486
MASSES = np.full(3, 1.007825 * AMU)
D, A, R0, S = 0.1744, 1.04435, 1.40083, 0.106     # LEPS H3 (hartree, bohr)
R_INF = 14.0


def _distances(x):
    g = np.asarray(x, dtype=float).reshape(3, 3)
    return np.array([np.linalg.norm(g[0] - g[1]), np.linalg.norm(g[1] - g[2]), np.linalg.norm(g[0] - g[2])])


def _leps(r):
    e = np.exp(-A * (r - R0))
    Q = D / (4 * (1 + S)) * ((3 + S) * e**2 - (2 + 6 * S) * e)
    J = D / (4 * (1 + S)) * ((1 + 3 * S) * e**2 - (6 + 2 * S) * e)
    return Q.sum() - np.sqrt(0.5 * ((J[0] - J[1]) ** 2 + (J[1] - J[2]) ** 2 + (J[2] - J[0]) ** 2))


def model(ion=0.0):
    """LEPS H + H2, plus -ion/(R^4 + 300) between each end atom and the far pair."""
    def energy(x):
        g = np.asarray(x, dtype=float).reshape(3, 3)
        value = _leps(_distances(x))
        if ion:
            for atom, pair in ((0, (1, 2)), (2, (0, 1))):
                R = np.linalg.norm(g[atom] - g[list(pair)].mean(axis=0))
                value -= ion / (R**4 + 300.0)
        return float(value)

    def gradient(x, h=1e-6):
        x = np.asarray(x, dtype=float).ravel()
        return np.array([(energy(x + h * e) - energy(x - h * e)) / (2 * h) for e in np.eye(9)])

    return energy, gradient


GUESS = np.array([0.0, 0.0, -1.8, 0.0, 0.0, 0.02, 0.05, 0.0, 1.75])   # bent, asymmetric


def make_run(ion=0.0, temperature=200.0):
    energy, gradient = model(ion)
    saddle, record = refine_saddle(gradient, GUESS, np.repeat(MASSES, 3))
    assert record["saddle"]
    coordinate = RPMDrateCoordinate([[0], [1, 2]], MASSES, saddle.reshape(3, 3), [[0, 1]], [[1, 2]], R_INF)

    def xi(x):
        value, derivative = coordinate.centroid_value_gradient(np.asarray(x, dtype=float).reshape(3, 3))[:2]
        return float(np.asarray(value)), np.asarray(derivative, dtype=float).ravel()

    # The rate run's transition-state geometry is only near the saddle.
    near = saddle + np.array([0.0, 0.03, 0.04, 0.0, 0.0, -0.02, 0.0, -0.03, 0.05])
    return SimpleNamespace(root=None, beta=1.0 / (KB_AU * temperature), nbeads=8,
                           system=SimpleNamespace(masses=MASSES, geometry=near.reshape(3, 3)),
                           cartesian_masses=np.repeat(MASSES, 3), energy=energy, gradient=gradient,
                           xi_value_gradient=xi, saddle=saddle)


class Windows:
    """The parts of a ring-polymer mean path that build_path reads."""

    def __init__(self, xi, centroids):
        self.xi = np.asarray(xi, dtype=float)
        self.mean_centroid = np.asarray(centroids, dtype=float).reshape(len(self.xi), 3, 3)
        self.orientation_spread = np.zeros(len(self.xi))

    def locked(self, factor=2.0):
        return self.xi > 0.75


def contracted(points, factor=0.92):
    """Centroids whose two bonds are shorter than the geometry they average (libration)."""
    out = []
    for x in np.asarray(points).reshape(len(points), 3, 3):
        centre = x.mean(axis=0)
        out.append(centre + factor * (x - centre))
    return np.asarray(out)


def build(run, monkeypatch, source, windows=None, **options):
    if windows is None:
        windows = Windows(np.linspace(0.8, 1.05, 6), np.repeat(run.saddle.reshape(1, 3, 3), 6, axis=0))
    monkeypatch.setattr(pipeline, "mean_path_from_rate_run", lambda root: windows)
    options = {"source": source, "step": 10.0, "smoothing": None, "product_drop_kt": 15.0, **options}
    _, _, path, info = pipeline.build_path(run, options)
    return path, info


# --- saddle refinement ------------------------------------------------------

def test_saddle_refinement_finds_the_collinear_leps_saddle_from_a_bent_guess():
    energy, gradient = model()
    saddle, record = refine_saddle(gradient, GUESS, np.repeat(MASSES, 3))
    assert record["saddle"] and record["negative_modes"] == 1 and record["iterations"] <= 10
    r12, r23, r13 = _distances(saddle)
    assert r12 == pytest.approx(r23, abs=1e-4) and r13 == pytest.approx(r12 + r23, abs=1e-4)  # collinear
    assert r12 == pytest.approx(1.770, abs=5e-3)
    assert 2000 < record["imaginary_frequency"] * 219474.63 < 2500
    # Its imaginary mode carries the central atom from one end atom to the
    # other: the end atoms move together, the centre twice as far the other way.
    mode = np.asarray(record["mode"]).reshape(3, 3)
    assert mode[0, 2] == pytest.approx(mode[2, 2], abs=1e-4)
    assert mode[1, 2] == pytest.approx(-2.0 * mode[0, 2], abs=1e-4)
    # A minimum is refused as a saddle.
    h2 = np.array([0.0, 0.0, -9.0, 0.0, 0.0, -0.7, 0.0, 0.0, 0.7])
    _, minimum = refine_saddle(gradient, h2 + 0.01, np.repeat(MASSES, 3), max_iterations=5)
    assert not minimum["saddle"]


def test_mass_weighted_hessian_is_symmetric_and_has_rigid_body_zeros():
    energy, gradient = model()
    saddle, _ = refine_saddle(gradient, GUESS, np.repeat(MASSES, 3))
    values = np.linalg.eigvalsh(mass_weighted_hessian(gradient, saddle, np.repeat(MASSES, 3)))
    # 3N = 9 for a linear molecule: 5 rigid-body zeros, one negative curvature, 3 positive.
    assert np.sum(np.abs(values) < 1e-7) == 5 and np.sum(values < -1e-7) == 1 and np.sum(values > 1e-7) == 3


# --- mep ---------------------------------------------------------------------

def test_mep_core_is_a_minimum_energy_path_through_the_saddle(monkeypatch):
    run = make_run()
    path, info = build(run, monkeypatch, "mep")
    record = info["mep"]
    assert record["saddle"] and record["saddle_shift_mass_weighted"] > 0
    k = info["core_slice"][0] + 1                     # the saddle within the assembled path
    # The same saddle as refined from the bent guess, up to a rigid-body motion.
    np.testing.assert_allclose(_distances(path[k]), _distances(run.saddle), atol=1e-4)
    energies = np.array([run.energy(p) for p in path])
    assert np.argmax(energies) == k
    # Steepest descent on both sides: monotone down to the asymptotes ...
    assert np.all(np.diff(energies[: k + 1]) > -1e-10) and np.all(np.diff(energies[k:]) < 1e-10)
    assert info["reactant_termination"]["accepted"] == info["product_termination"]["accepted"] == "asymptote"
    # ... and a minimum-energy path: the gradient lies along the tangent, up
    # to the chord error of the steps, which vanishes as the step shrinks.
    coarse = _transverse_gradient(run, path)
    fine_path, _ = build(run, monkeypatch, "mep", step=2.0)
    fine = _transverse_gradient(run, fine_path)
    assert np.median(coarse) < 1e-2 and np.median(fine) < 1e-3
    assert np.percentile(fine, 90) < 0.3 * np.percentile(coarse, 90) and fine.max() < 0.1


def _transverse_gradient(run, path):
    """|gradient across the path| / |gradient| at interior points (mass-weighted, central tangent)."""
    sqrt_m = np.sqrt(run.cartesian_masses)
    weighted = path * sqrt_m
    ratios = []
    for j in range(2, len(path) - 2):
        tangent = weighted[j + 1] - weighted[j - 1]
        tangent /= np.linalg.norm(tangent)
        g = run.gradient(path[j]) / sqrt_m
        if np.linalg.norm(g) > 1e-6:
            ratios.append(np.linalg.norm(g - (g @ tangent) * tangent) / np.linalg.norm(g))
    return np.asarray(ratios)


def test_mep_continues_through_ion_molecule_complexes(monkeypatch):
    run = make_run(ion=3.0)
    path, info = build(run, monkeypatch, "mep")
    reactant, product = info["reactant_termination"], info["product_termination"]
    # The reactant side records the complex and climbs out of it to xi = 0 (R_inf).
    assert reactant["status"] == "coordinate_target_reached" and reactant["accepted"] == "reactant_reference"
    (complex_,) = reactant["minima"]
    assert complex_["energy"] < run.energy(path[0]) - 7e-3
    g = path[0].reshape(3, 3)
    assert R_INF <= np.linalg.norm(g[0] - g[1:].mean(axis=0)) < R_INF + 0.3
    # The product side ends in the post-reaction complex, 15 kT below the barrier at 200 K.
    assert product["status"] == "minimum_reached" and product["accepted"] == "product_well"
    assert info["termini_accepted"]


def test_mep_refuses_a_geometry_that_is_not_near_a_saddle(monkeypatch):
    run = make_run()
    run.system.geometry = np.array([0.0, 0.0, -9.0, 0.0, 0.0, -0.7, 0.0, 0.0, 0.7]).reshape(3, 3)
    with pytest.raises(ValueError, match="first-order saddle"):
        build(run, monkeypatch, "mep", ts_gradient_tolerance=1e-5)


# --- pmf_minimum and centroid_all -------------------------------------------

def pmf_windows(run):
    """Umbrella window centres from near the reactants to past the dividing surface xi = 1."""
    return np.r_[np.linspace(0.05, 0.95, 11), 1.0, 1.04]


def test_pmf_minimum_has_physical_bonds_where_the_centroids_are_contracted(monkeypatch):
    run = make_run()
    xi = pmf_windows(run)
    from pyrpmd.analysis.subspace_paths import pmf_minimum_core

    exact, record = pmf_minimum_core(run, xi, {})
    assert record["failed_xi"] == [] and record["max_constraint_residual"] < 1e-8
    windows = Windows(xi, contracted(exact))
    _, minimum_info = build(run, monkeypatch, "pmf_minimum", windows=windows)
    _, centroid_info = build(run, monkeypatch, "centroid_all", windows=windows)
    assert minimum_info["pmf_minimum"]["windows"] == centroid_info["windows"] == len(xi)
    assert centroid_info["window_xi"] == pytest.approx(list(xi))
    # On every window surface the minimum lies below the contracted centroid ...
    for point, centroid in zip(exact, windows.mean_centroid):
        assert run.energy(point) <= run.energy(centroid.ravel()) + 1e-12
    # ... and near the reactants the H2 bond is the molecule's, not the centroid's.
    r12, r23, _ = _distances(exact[0])
    assert r23 == pytest.approx(1.40, abs=0.02) and r12 > 4.0
    assert _distances(windows.mean_centroid[0])[1] < 1.31
    # The minimum on the dividing surface xi = 1 is the saddle.
    on_surface = exact[int(np.argmin(abs(xi - 1.0)))]
    np.testing.assert_allclose(_distances(on_surface), _distances(run.saddle), atol=1e-3)


def test_pmf_minimum_path_is_connected_and_monotone_in_xi(monkeypatch):
    run = make_run(ion=3.0)
    xi = pmf_windows(run)
    windows = Windows(xi, np.repeat(run.saddle.reshape(1, 3, 3), len(xi), axis=0))
    path, info = build(run, monkeypatch, "pmf_minimum", windows=windows)
    lo, hi = info["core_slice"]
    core = path[lo:hi]
    values = [run.xi_value_gradient(p)[0] for p in core]
    assert np.all(np.diff(values) > 0)
    np.testing.assert_allclose(values, xi, atol=1e-7)
    steps = np.linalg.norm(np.diff(core * np.sqrt(run.cartesian_masses), axis=0), axis=1)
    assert steps.max() < 6.0 * np.median(steps)          # no jump into another basin
    assert info["pmf_minimum"]["start_xi"] == pytest.approx(xi[np.argmin(abs(xi - run.xi_value_gradient(
        run.system.geometry.ravel())[0]))])


def test_centroid_sources_are_unchanged(monkeypatch):
    run = make_run()
    xi = np.linspace(0.6, 1.04, 8)
    from pyrpmd.analysis.subspace_paths import pmf_minimum_core

    exact, _ = pmf_minimum_core(run, xi, {})
    windows = Windows(xi, exact)
    _, centroid = build(run, monkeypatch, "centroid", windows=windows, max_steps=3)
    assert centroid["locked_windows"] == int(np.sum(xi > 0.75)) and "window_xi" not in centroid
    assert centroid["core_points"] == centroid["locked_windows"]


# --- input -------------------------------------------------------------------

@pytest.mark.parametrize("source", ["centroid", "centroid_all", "bead_weighted", "least_action", "instanton",
                                    "mep", "pmf_minimum"])
def test_every_path_source_is_accepted_by_the_input(source):
    config = {"task": "rate", "rate": {"subspace_correction": {"enabled": False, "path": {
        "source": source, "mep_displacement": 0.5, "ts_gradient_tolerance": 1e-6,
        "pmf_constraint_tolerance": 1e-9}}}}
    path = rate_subspace_options(config)["path"]
    assert path["source"] == source and path["mep_displacement"] == 0.5


def test_unknown_path_source_is_rejected():
    config = {"task": "rate", "rate": {"subspace_correction": {"enabled": False, "path": {"source": "irc"}}}}
    with pytest.raises(InputError, match="mep, pmf_minimum"):
        rate_subspace_options(config)
    with pytest.raises(ValueError, match="path.source"):
        pipeline.build_path(SimpleNamespace(), {"source": "irc"})
