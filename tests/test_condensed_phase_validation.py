"""Independent force/strain and thermodynamic checks for solid benchmarks."""
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "benchmarks" / "condensed_phase"
sys.path.insert(0, str(MODEL_PATH))
from models import HarmonicCrystal, LennardJonesSolid, build_lj, fcc, piglet_parameters
from validate import finite_bead_crystal, gaussian_estimator_moments, save_csv, save_provenance, load_csv
sys.path.pop(0)


def test_coupled_crystal_force_and_full_cell_derivative():
    model = HarmonicCrystal()
    rng = np.random.default_rng(771)
    q = model.centers[None]+rng.normal(scale=.1, size=(2, model.natoms, 3))
    got = model.evaluate(q)
    direction = rng.normal(size=q.shape)
    eps = 1e-5
    numerical = (model.evaluate(q+eps*direction).energy-model.evaluate(q-eps*direction).energy)/(2*eps)
    np.testing.assert_allclose(numerical, -np.sum(got.forces*direction, axis=(1, 2)), rtol=2e-8)
    h = model.cell.copy()
    for a, b in ((0, 0), (1, 1), (0, 2)):
        strain = np.zeros((3, 3)); strain[a, b] = eps
        energies = []
        for sign in (-1, 1):
            affine = np.eye(3)+sign*strain
            model.set_cell(h@affine.T)
            energies.append(model.evaluate(q@affine.T).energy)
        model.set_cell(h)
        np.testing.assert_allclose(-(energies[1]-energies[0])/(2*eps), got.virial[:, a, b], rtol=2e-6, atol=2e-7)


def test_crystal_partition_derivatives_and_sc_fourth_order():
    model = HarmonicCrystal()
    beta = 8.
    exact = model.quantum_reference(beta)
    errors = []
    for p in (8, 16, 32):
        current = finite_bead_crystal(model, beta, p)
        h = 1e-3
        plus = finite_bead_crystal(model, beta+h, p)
        minus = finite_bead_crystal(model, beta-h, p)
        assert current["energy"] == pytest.approx(-(plus["log_partition"]-minus["log_partition"])/(2*h), rel=1e-8)
        assert current["heat_capacity_kB"] == pytest.approx(-beta**2*(plus["energy"]-minus["energy"])/(2*h), rel=1e-6)
        errors.append(abs(current["energy"]-exact["energy"]))
    assert errors[0]/errors[1] > 10
    assert errors[1]/errors[2] > 10


def test_production_crystal_estimator_gaussian_moments_match_partition_derivatives():
    model = HarmonicCrystal()
    reference = finite_bead_crystal(model, 8., 8)
    production = gaussian_estimator_moments(model, 8., 8)
    assert production["energy"] == pytest.approx(reference["energy"], rel=1e-10)
    assert production["pressure"] == pytest.approx(reference["pressure"], rel=1e-10)
    assert production["heat_capacity_kB"] == pytest.approx(reference["heat_capacity_kB"], abs=1e-4)


@pytest.mark.skipif(shutil.which("c++") is None, reason="benchmark potential requires C++ compiler")
def test_lj_matches_independent_image_sum_force_virial_and_split(tmp_path):
    q, cell = fcc()
    q = q[None]+np.random.default_rng(314).normal(scale=.01, size=(2, len(q), 3))
    library = build_lj(tmp_path)
    potential = LennardJonesSolid(library, q.shape[1], cell)
    fast = LennardJonesSolid(library, q.shape[1], cell, component=1)
    slow = LennardJonesSolid(library, q.shape[1], cell, component=2)
    got = potential.evaluate(q)
    a, b = fast.evaluate(q), slow.evaluate(q)
    np.testing.assert_allclose(got.energy, a.energy+b.energy, rtol=1e-12)
    np.testing.assert_allclose(got.forces, a.forces+b.forces, rtol=2e-11, atol=1e-11)
    np.testing.assert_allclose(got.virial, a.virial+b.virial, rtol=2e-12, atol=2e-11)
    # Deliberately use all ordered pairs and half their sum, different from
    # production's unordered pair loop; include multiple images per pair.
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T@cell
    d = q[:, :, None]-q[:, None, :]
    d -= np.rint(d/np.diag(cell))*np.diag(cell)
    r = np.linalg.norm(d[:, :, :, None, :]+shifts, axis=-1)
    r[:, np.arange(q.shape[1]), np.arange(q.shape[1]), :] = np.inf
    inv = np.where(r<2.5, 1/r, 0)
    t = np.clip((r-2.)/.5, 0, 1)
    switch = 1-10*t**3+15*t**4-6*t**5
    energy = .5*np.sum(4*(inv**12-inv**6)*switch, axis=(1, 2, 3))
    np.testing.assert_allclose(got.energy, energy, rtol=1e-12)
    direction = np.random.default_rng(14).normal(size=q.shape)
    h = 1e-6
    derivative = (potential.evaluate(q+h*direction).energy-potential.evaluate(q-h*direction).energy)/(2*h)
    np.testing.assert_allclose(derivative, -np.sum(got.forces*direction, axis=(1, 2)), rtol=2e-7, atol=2e-6)
    es = []
    for sign in (-1, 1):
        scale = np.exp(sign*h)
        potential.set_cell(cell*scale)
        es.append(potential.evaluate(q*scale).energy)
    potential.set_cell(cell)
    np.testing.assert_allclose(-(es[1]-es[0])/(2*h), np.trace(got.virial, axis1=1, axis2=2), rtol=2e-7, atol=2e-5)


def test_authentic_piglet_fixture_normalization_and_positive_diffusion():
    from pyrpmd.native_ensemble import _gle_matrices
    parameters = piglet_parameters(5.)
    n, transitions, noises, roots = _gle_matrices(parameters, "piglet", 4, 5., .02)
    assert n == 9
    np.testing.assert_array_equal(parameters["C"][0], np.eye(9))
    assert not np.allclose(parameters["C"][1], np.eye(9))
    assert "87d27b7d" in parameters["provenance"]


def test_resume_requires_matching_source_protocol_and_raw_identity(tmp_path):
    path=tmp_path/"series.csv"
    rows=[{"step":float(n), "energy":.2*n} for n in (12,14,16)]
    save_csv(path, rows)
    options=dict(burn=10,steps=6,stride=2,identity={"source":"version-a","beta":5.})
    with pytest.raises(ValueError,match="without protocol/source"):
        load_csv(path,**options)
    save_provenance(path,options["identity"])
    assert load_csv(path,**options)==rows
    with pytest.raises(ValueError,match="identity differs"):
        load_csv(path,**{**options,"identity":{"source":"version-b","beta":5.}})
    path.write_text(path.read_text().replace("2.4000000000000004", "9.4000000000000004"))
    # Guarantee a byte change even if the platform's float spelling differs.
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError,match="identity differs"):
        load_csv(path,**options)


def test_numerical_identity_ignores_source_line_shifts_but_detects_physics_edits(monkeypatch):
    import validate
    source_path=Path(validate.__file__).resolve()
    source=source_path.read_text()
    baseline=validate.numerical_source_identity()
    original=Path.read_text
    replacement="\n# Report-only line shift while a live code object exists.\n"+source
    monkeypatch.setattr(Path,"read_text",lambda path,*a,**kw:
                        replacement if path.resolve()==source_path else original(path,*a,**kw))
    assert validate.numerical_source_identity()==baseline
    replacement=replacement.replace("for step in range(burn+steps):", "for step in range(burn+steps+1):")
    changed=validate.numerical_source_identity()
    key="validate.py::run_trajectory::AST"
    assert changed[key]!=baseline[key]
    assert {k:v for k,v in changed.items() if k!=key}=={k:v for k,v in baseline.items() if k!=key}
