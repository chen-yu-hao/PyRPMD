"""Independent geometry/measure/finite-bin oracles for the classical reference."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import erf

from benchmarks.brhcl.reference import classical_pmf, common_wall_energy, distance_geometry, load_reference_cache, save_reference_cache
from pyrpmd.checkpoint import atomic_savez
from pyrpmd.constants import ANGSTROM_TO_AU, KB_AU, KCAL_MOL_TO_AU
from pyrpmd.potentials import PESResult, PotentialError


class ZeroPotential:
    def __init__(self, offset=0.0):
        self.offset = offset
        self.maximum_batch = 0

    def evaluate(self, q):
        self.maximum_batch = max(self.maximum_batch, len(q))
        return PESResult(np.full(len(q), self.offset), np.zeros_like(q))


class SeparablePotential:
    def __init__(self, ax, ay):
        self.ax, self.ay = ax, ay

    def evaluate(self, q):
        x, y = q[:, 0] - q[:, 1], q[:, 2] - q[:, 1]
        energy = self.ax * np.sum(x*x, axis=1) + self.ay * np.sum(y*y, axis=1)
        forces = np.zeros_like(q)
        forces[:, 0], forces[:, 2] = -2*self.ax*x, -2*self.ay*y
        forces[:, 1] = -(forces[:, 0] + forces[:, 2])
        return PESResult(energy, forces)


def _mean_square(edges):
    a, b = edges[:-1], edges[1:]
    return (a*a + a*b + b*b) / 3


def _square_gaussian_integral(x, alpha):
    return (np.sqrt(np.pi) * erf(np.sqrt(alpha)*x) / (4*alpha**1.5)
            - x*np.exp(-alpha*x*x)/(2*alpha))


def test_cartesian_mapping_preserves_all_three_distances_and_atom_order():
    x = np.array([1.1, 2.4, 4.0, 1.0])
    y = np.array([1.3, 1.7, 1.0, 1.0])
    z = np.array([0.8, 3.8, 3.0, 2.0])
    q = distance_geometry(x, y, z)
    assert q.shape == (4, 3, 3)
    np.testing.assert_array_equal(q[:, 1], 0.0)
    for expected, (left, right) in zip((x, y, z), ((0, 1), (1, 2), (0, 2))):
        np.testing.assert_allclose(np.linalg.norm(q[:, left] - q[:, right], axis=1),
                                   expected, atol=2e-14, rtol=0)


@pytest.mark.parametrize("distances", [(1, 1, 3), (3, 1, 1), (0, 1, 1), (1, np.nan, 1)])
def test_invalid_triangles_are_rejected(distances):
    with pytest.raises(ValueError):
        distance_geometry(*distances)


def test_zero_potential_has_exact_two_x_squared_y_squared_measure():
    x = np.array([0.0, 0.3, 1.1, 1.8])
    y = np.array([0.2, 0.7, 1.6])
    potential = ZeroPotential()
    result = classical_pmf(potential, x, y, beta=1.3, wall_angstrom=0,
                           xy_order=2, z_order=2, max_xy_order=4, max_z_order=4,
                           tolerance_kbt=1e-12, max_batch=7)
    # With V=0, integral_{|x-y|}^{x+y} z dz=2xy, leaving 2*x^2*y^2.
    exact = 2*_mean_square(x)[:, None]*_mean_square(y)[None, :]
    np.testing.assert_allclose(np.exp(result["log_density_unnormalized"]), exact,
                               rtol=2e-14, atol=2e-14)
    volume = np.diff(x)[:, None]*np.diff(y)[None, :]
    np.testing.assert_allclose(result["probability_mass"], exact*volume/(exact*volume).sum(), rtol=2e-14)
    assert result["free_energy"].shape == (3, 2)
    assert result["converged"]
    assert potential.maximum_batch <= 7
    assert len(result["history"]) == 2


def test_separable_radial_harmonics_agree_with_analytic_bin_integrals():
    x = np.array([0.2, 0.8, 1.4, 2.0])
    y = np.array([0.1, 0.4, 0.9, 1.7, 2.2])
    ax, ay, beta = 0.7, 1.3, 1.8
    result = classical_pmf(SeparablePotential(ax, ay), x, y, beta,
                           wall_angstrom=0, xy_order=3, z_order=2,
                           max_xy_order=24, max_z_order=16, tolerance_kbt=1e-10)
    ix = np.diff(_square_gaussian_integral(x, beta*ax))/np.diff(x)
    iy = np.diff(_square_gaussian_integral(y, beta*ay))/np.diff(y)
    exact = 2*ix[:, None]*iy[None, :]
    np.testing.assert_allclose(np.exp(result["log_density_unnormalized"]), exact, rtol=2e-10, atol=1e-14)
    assert result["converged"]
    assert result["quadrature_converged"].all()
    assert result["free_energy"].shape == (3, 4)


def test_logarithmic_integration_is_stable_under_extreme_energy_gauge():
    options = dict(x_edges=[0.5, 1, 1.8], y_edges=[0.7, 1.2, 2.1], beta=1.7,
                   wall_angstrom=0, xy_order=2, z_order=2,
                   max_xy_order=4, max_z_order=4, tolerance_kbt=1e-8)
    base = classical_pmf(ZeroPotential(), **options)
    for offset in [-1e5, 1e5]:
        result = classical_pmf(ZeroPotential(offset), **options)
        assert result["converged"]
        np.testing.assert_allclose(result["free_energy"], base["free_energy"], atol=8e-11, rtol=0)
        np.testing.assert_allclose(result["probability_mass"], base["probability_mass"], atol=5e-11, rtol=0)
        np.testing.assert_allclose(result["log_density_unnormalized"] + options["beta"]*offset,
                                   base["log_density_unnormalized"], atol=8e-11, rtol=0)


def test_common_wall_counts_every_pair_and_uses_physical_units():
    distances = np.array([6.5, 7.0, 7.5]) * ANGSTROM_TO_AU
    value = common_wall_energy(*distances, wall_angstrom=6., wall_k_kcal_mol_angstrom2=20.)
    assert value == pytest.approx(0.5*20*(0.5**2+1.0**2+1.5**2)*KCAL_MOL_TO_AU)
    assert common_wall_energy(*distances, wall_angstrom=0.) == 0
    assert common_wall_energy(*distances, wall_k_kcal_mol_angstrom2=0.) == 0


def test_provider_bare_entry_prevents_double_counting_common_confinement():
    class ConfinedProvider(ZeroPotential):
        def bare_evaluate(self, q):
            return super().evaluate(q)

        def evaluate(self, q):
            raise AssertionError("reference must use the bare PES")

    kwargs = dict(x_edges=np.array([0.3, 0.8, 1.4])*ANGSTROM_TO_AU,
                  y_edges=np.array([0.4, 0.9, 1.3])*ANGSTROM_TO_AU,
                  beta=20, wall_angstrom=0.7, wall_k_kcal_mol_angstrom2=40.,
                  xy_order=4, z_order=8, max_xy_order=8, max_z_order=16)
    explicit = classical_pmf(ZeroPotential(), **kwargs)
    bare = classical_pmf(ConfinedProvider(), **kwargs)
    np.testing.assert_array_equal(bare["log_density_unnormalized"], explicit["log_density_unnormalized"])
    assert bare["potential_entry"] == "bare_evaluate"


def test_one_order_cannot_claim_quadrature_convergence():
    result = classical_pmf(ZeroPotential(), [0, 1], [0, 1], 1,
                           wall_angstrom=0, xy_order=2, z_order=2,
                           max_xy_order=2, max_z_order=2)
    assert not result["converged"]
    assert not result["quadrature_converged"].any()
    assert np.isnan(result["quadrature_error_kbt"]).all()
    assert result["status"] == "quadrature_order_limit"


def test_declared_pair_domain_clips_all_distances_but_keeps_original_bin_area():
    class CoreExcludedPotential(ZeroPotential):
        metadata = {"minimum_pair_distance_bohr": .5}

        def evaluate(self, q):
            for a, b in ((0, 1), (1, 2), (0, 2)):
                assert np.all(np.linalg.norm(q[:, a] - q[:, b], axis=1) >= .5)
            return super().evaluate(q)

    x, y = np.array([0., .2, .6, .8]), np.array([.3, .55, .75])
    exact = np.zeros((3, 2))
    for i, (a, b) in enumerate(zip(x[:-1], x[1:])):
        for j, (c, d) in enumerate(zip(y[:-1], y[1:])):
            lo_x, lo_y = max(a, .5), max(c, .5)
            if b <= lo_x or d <= lo_y:
                continue
            ix = {n: (b**(n+1)-lo_x**(n+1))/(n+1) for n in (1, 2, 3)}
            iy = {n: (d**(n+1)-lo_y**(n+1))/(n+1) for n in (1, 2, 3)}
            # Throughout these valid rectangles |x-y|<.5, so the exact
            # z integral is ((x+y)^2-.5^2)/2, multiplied by x*y.
            exact[i, j] = .5*(ix[3]*iy[1]+2*ix[2]*iy[2]+ix[1]*iy[3]-.25*ix[1]*iy[1])/((b-a)*(d-c))
    result = classical_pmf(CoreExcludedPotential(), x, y, beta=1., wall_angstrom=0.,
                           xy_order=2, z_order=2, max_xy_order=4, max_z_order=4,
                           tolerance_kbt=1e-12, max_batch=7)
    np.testing.assert_allclose(np.exp(result["log_density_unnormalized"]), exact, atol=1e-16, rtol=2e-14)
    assert result["converged"]
    assert np.isposinf(result["free_energy"][0]).all()
    np.testing.assert_array_equal(result["probability_mass"][0], 0.)
    assert result["domain"]["minimum_pair_distance_bohr"] == .5
    assert result["domain"]["source"] == "provider_metadata"
    with pytest.raises(ValueError, match="no probability in the declared PES domain"):
        classical_pmf(CoreExcludedPotential(), [0., .3], [.1, .4], 1.)


def test_unrelated_pes_failure_is_never_interpreted_as_excluded_core():
    class FailingPotential(ZeroPotential):
        metadata = {"minimum_pair_distance_bohr": .2}

        def evaluate(self, q):
            return PESResult(np.zeros(len(q)), np.zeros_like(q),
                             status=np.ones(len(q), dtype=int),
                             messages=("independent numerical failure",)*len(q))

    with pytest.raises(PotentialError, match="independent numerical failure"):
        classical_pmf(FailingPotential(), [1., 2.], [1., 2.], 1.)


@pytest.mark.parametrize("overrides", [
    {"x_edges": [-1, 0, 1]}, {"y_edges": [0, 1, np.inf]},
    {"beta": 0}, {"xy_order": True}, {"max_z_order": 1},
    {"max_batch": 0}, {"tolerance_kbt": np.nan}, {"wall_angstrom": -1},
    {"minimum_pair_distance_bohr": -1}, {"minimum_pair_distance_bohr": True},
])
def test_invalid_quadrature_configuration_is_rejected(overrides):
    options = dict(x_edges=[0, 1], y_edges=[0, 1], beta=1, xy_order=2, z_order=2)
    options.update(overrides)
    with pytest.raises(ValueError):
        classical_pmf(ZeroPotential(), **options)


def _write_reference_cache(path, result, identity):
    checksum = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    summary = {key: value for key, value in result.items() if not isinstance(value, np.ndarray)}
    summary.update(identity=identity, identity_sha256=checksum)
    atomic_savez(path, **arrays, identity=json.dumps(identity, sort_keys=True), identity_sha256=checksum)
    path.with_suffix(".json").write_text(json.dumps(summary, allow_nan=False))


@pytest.fixture(scope="module")
def real_reference_cache_values():
    from benchmarks.brhcl.pes import BrHClPotential, DEFAULT_LIBRARY, SOURCE
    if not DEFAULT_LIBRARY.is_file():
        pytest.skip("build optional pinned BrHCl PES for actual-source cache checks")
    settings = {"model": "BrHCl_LEPS1", "beads": 1, "temperature_K": 2000.,
                "beta": 1/(KB_AU*2000.), "wall_angstrom": 6., "wall_k_kcal_mol_angstrom2": 20.,
                "atomic_order": ["Br", "H", "Cl"], "cv_names": ["r_HBr", "r_HCl"]}
    edges = [np.array([1.2, 1.4, 1.7])*ANGSTROM_TO_AU, np.array([1.1, 1.5])*ANGSTROM_TO_AU]
    library_hash = hashlib.sha256(DEFAULT_LIBRARY.read_bytes()).hexdigest()
    with BrHClPotential() as potential:
        result = classical_pmf(potential, *edges, settings["beta"], xy_order=4, z_order=32,
                               max_xy_order=8, max_z_order=64)
    identity = {**settings, "library_sha256": library_hash, "source_commit": SOURCE["commit"],
                "source_sha256": SOURCE["sha256"], "minimum_pair_distance_bohr": .2,
                "atomic_order": ["Br", "H", "Cl"], "cv_names": ["r_HBr", "r_HCl"],
                "x_edges_bohr": edges[0].tolist(), "y_edges_bohr": edges[1].tolist()}
    return settings, edges, library_hash, result, identity


def test_real_classical_reference_cache_reuses_only_the_same_physical_target(real_reference_cache_values, tmp_path):
    settings, edges, library_hash, result, identity = real_reference_cache_values
    path = tmp_path / "reference.npz"
    _write_reference_cache(path, result, identity)
    # Independent seed, trajectory length and sampling kernel are irrelevant
    # to the deterministic integral for this unchanged target Hamiltonian.
    repeated = {**settings, "seed": 99, "production_steps": 100000, "orientation_mc_trials": 32}
    loaded = load_reference_cache(path, settings=repeated, library_sha256=library_hash, edges=edges)
    np.testing.assert_array_equal(loaded["free_energy"], result["free_energy"])
    assert loaded["domain"]["source"] == "provider_metadata"
    assert loaded["domain"]["minimum_pair_distance_bohr"] == .2
    assert loaded["free_energy"].shape == (2, 1)


@pytest.mark.parametrize("change", ["library", "source", "temperature", "beta", "wall", "stiffness", "beads", "edges", "atomic_order", "cv_names", "core"])
def test_reference_cache_rejects_different_scientific_target(real_reference_cache_values, tmp_path, change):
    settings, edges, library_hash, result, identity = copy.deepcopy(real_reference_cache_values)
    path = tmp_path / "reference.npz"
    if change == "source":
        identity["source_sha256"] = "0"*64
    _write_reference_cache(path, result, identity)
    if change == "library":
        library_hash = "0"*64
    elif change == "temperature":
        settings["temperature_K"] = 2100.
        settings["beta"] = 1/(KB_AU*2100.)
    elif change == "beta":
        settings["beta"] *= 1.01
    elif change == "wall":
        settings["wall_angstrom"] = 7.
    elif change == "stiffness":
        settings["wall_k_kcal_mol_angstrom2"] = 21.
    elif change == "beads":
        settings["beads"] = 8
    elif change == "edges":
        edges[0][1] += .001
    elif change in ("atomic_order", "cv_names"):
        settings[change].reverse()
    elif change == "core":
        settings["minimum_pair_distance_bohr"] = .3
    with pytest.raises(ValueError, match="identity|temperature and beta|only a P=1"):
        load_reference_cache(path, settings=settings, library_sha256=library_hash, edges=edges)


@pytest.mark.parametrize("change", ["edge_array", "free_energy", "quadrature", "json_identity"])
def test_reference_cache_rejects_reencoded_inconsistent_arrays_or_metadata(real_reference_cache_values, tmp_path, change):
    settings, edges, library_hash, result, identity = copy.deepcopy(real_reference_cache_values)
    if change == "edge_array":
        result["x_edges"][1] += .001
    elif change == "free_energy":
        result["free_energy"] += .1
    elif change == "quadrature":
        result["quadrature_converged"][:] = True
        result["quadrature_error_kbt"][:] = 10.
    path = tmp_path / "reference.npz"
    _write_reference_cache(path, result, identity)
    if change == "json_identity":
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["identity_sha256"] = "0"*64
        path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inconsistent|metadata is invalid"):
        load_reference_cache(path, settings=settings, library_sha256=library_hash, edges=edges)


def test_public_cache_writer_roundtrips_an_analytic_grid_and_refuses_overwrite(tmp_path):
    settings = {"model": "BrHCl_LEPS1", "beads": 1, "temperature_K": 2000.,
                "beta": 1/(KB_AU*2000.), "wall_angstrom": 0., "wall_k_kcal_mol_angstrom2": 20.,
                "atomic_order": ["Br", "H", "Cl"], "cv_names": ["r_HBr", "r_HCl"]}
    potential = ZeroPotential()
    potential.metadata = {"minimum_pair_distance_bohr": .2}
    edges = [np.array([.4, .7, 1.]), np.array([.3, .8])]
    exact = classical_pmf(potential, *edges, settings["beta"], wall_angstrom=0.,
                          xy_order=4, z_order=4, max_xy_order=8, max_z_order=8)
    path = tmp_path/"analytic.npz"
    saved = save_reference_cache(path, exact, settings=settings, library_sha256="a"*64)
    loaded = load_reference_cache(path, settings=settings, library_sha256="a"*64, edges=edges)
    np.testing.assert_array_equal(loaded["free_energy"], exact["free_energy"])
    np.testing.assert_array_equal(saved["probability_mass"], exact["probability_mass"])
    assert loaded["array_file"] == path.name
    before = path.read_bytes(), path.with_suffix(".json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        save_reference_cache(path, exact, settings=settings, library_sha256="a"*64)
    assert before == (path.read_bytes(), path.with_suffix(".json").read_bytes())


def test_public_cache_writer_roundtrips_real_model_and_rejects_incompatible_domain(real_reference_cache_values, tmp_path):
    settings, edges, library_hash, result, _ = real_reference_cache_values
    path = tmp_path/"real.npz"
    save_reference_cache(path, result, settings=settings, library_sha256=library_hash)
    loaded = load_reference_cache(path, settings=settings, library_sha256=library_hash, edges=edges)
    np.testing.assert_array_equal(loaded["free_energy"], result["free_energy"])
    bad = copy.deepcopy(result)
    bad["domain"]["minimum_pair_distance_bohr"] = .3
    rejected = tmp_path/"incompatible.npz"
    with pytest.raises(ValueError, match="contradicts its scientific identity"):
        save_reference_cache(rejected, bad, settings=settings, library_sha256=library_hash)
    assert not rejected.exists() and not rejected.with_suffix(".json").exists()
    with pytest.raises(ValueError, match="core-domain identity"):
        save_reference_cache(rejected, result, settings={**settings, "minimum_pair_distance_bohr": .3},
                             library_sha256=library_hash)
    assert not rejected.exists()


def test_reference_cli_rebuilds_real_non_square_grid_and_checks_recorded_binary(real_reference_cache_values, tmp_path):
    from benchmarks.brhcl.pes import DEFAULT_LIBRARY
    settings, _, library_hash, _, _ = real_reference_cache_values
    settings = {**settings, "bins": [2, 1], "ranges_angstrom": [[1.2, 1.7], [1.1, 1.5]]}
    run = tmp_path/"run"
    run.mkdir()
    (run/"plan.json").write_text(json.dumps(settings))
    model = {"library": str(DEFAULT_LIBRARY), "library_sha256": library_hash}
    (run/"model.json").write_text(json.dumps(model))
    output = tmp_path/"rebuilt.npz"
    repository = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    command = [sys.executable, "benchmarks/brhcl/reference.py", "--run", str(run),
               "--output", str(output), "--tolerance-kbt", ".003"]
    completed = subprocess.run(command, cwd=repository, env=env, capture_output=True, text=True, check=True)
    report = json.loads(completed.stdout)
    assert report["total_bins"] == 2
    edges = [np.linspace(*bounds, count+1) for bounds, count in
             zip(np.asarray(settings["ranges_angstrom"])*ANGSTROM_TO_AU, settings["bins"])]
    loaded = load_reference_cache(output, settings=settings, library_sha256=library_hash, edges=edges)
    assert loaded["tolerance_kbt"] == .003 and loaded["free_energy"].shape == (2, 1)
    assert loaded["source_run"] == str(run.resolve())
    model["library_sha256"] = "0"*64
    (run/"model.json").write_text(json.dumps(model))
    bad_output = tmp_path/"bad-binary.npz"
    command[command.index("--output")+1] = str(bad_output)
    failed = subprocess.run(command, cwd=repository, env=env, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "PES binary no longer matches the recorded run" in failed.stderr
    assert not bad_output.exists() and not bad_output.with_suffix(".json").exists()
