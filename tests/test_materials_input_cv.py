"""Public inputs and periodic CV derivatives, independent of the PES model."""

import numpy as np
import pytest

from pyrpmd import JobSpec, System, prepare
from pyrpmd.cv import make_cv
from pyrpmd.input import InputError
from pyrpmd.constants import ANGSTROM_TO_AU, FS_TO_AU


def system():
    return System(("H", "O", "O"), np.array([1837., 29156., 29156.]),
                  np.array([[.3, .8, .6], [9.6, .2, .3], [.8, 2., .4]]),
                  cell=np.array([[10., 0., 0.], [2., 8., 0.], [0., 0., 12.]]),
                  pbc=[True, True, False])


@pytest.mark.parametrize("spec", [
    {"kind": "distance", "atoms": [0, 1]},
    {"kind": "bond_difference", "atoms": [1, 0, 2]},
    {"kind": "angle", "atoms": [1, 0, 2]},
    {"kind": "projection", "atom": 0, "direction": [1., 2., -.5], "reference_atoms": [1, 2]},
    {"kind": "fractional", "atom": 0, "axis": 1, "reference_atoms": [1, 2]},
])
@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
def test_periodic_cv_gradient(spec, estimator):
    s = system()
    cv = make_cv(dict(spec, estimator=estimator), s)
    q = np.repeat(s.geometry[None], 3, axis=0)
    q[1] += np.array([.01, -.015, .02])
    analytical = cv.value_and_jacobian(q).jacobian
    numerical = np.empty_like(analytical)
    for ix in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[ix] += 1e-6
        minus[ix] -= 1e-6
        numerical[(0,) + ix] = (cv.value(plus)[0] - cv.value(minus)[0]) / 2e-6
    np.testing.assert_allclose(analytical, numerical, rtol=2e-6, atol=2e-8)


def test_geometric_cv_image_invariance_but_migration_coordinate_continuous():
    s = system()
    q = np.repeat(s.geometry[None], 4, axis=0)
    image = q.copy()
    image[:, 0] += 3 * s.cell[0] - 2 * s.cell[1]
    for spec in ({"kind": "distance", "atoms": [0, 1]},
                 {"kind": "angle", "atoms": [1, 0, 2]}):
        cv = make_cv(spec, s)
        np.testing.assert_allclose(cv.value(q), cv.value(image), atol=1e-13)
    cv = make_cv({"kind": "fractional", "atom": 0, "axis": 0}, s)
    np.testing.assert_allclose(cv.value(image) - cv.value(q), [3.], atol=1e-13)
    assert cv.periods == (None,)


def job(task="pimd"):
    return {
        "task": task, "beads": 4, "temperature": "300 K", "native": False,
        "system": {"symbols": ["H"], "geometry": [[.1, .1, .1]],
                   "geometry_unit": "angstrom", "cell": np.diag([4., 4., 8.]).tolist(),
                   "cell_unit": "angstrom", "pbc": [True, True, False]},
        "potential": {"provider": "builtin", "name": "periodic_cosine"},
        "dynamics": {"timestep": "0.1 fs"},
        "sampling": {"equilibration_steps": 0, "production_steps": 40,
                     "sample_interval": 2},
    }


def test_public_materials_prepare_and_units():
    config, s = JobSpec.from_mapping(job()).resolve()
    np.testing.assert_allclose(s.cell, np.diag([4., 4., 8.]) * ANGSTROM_TO_AU)
    assert config["sampling"]["thermostat"] == "pile-l"
    assert config["sampling"]["integrator"] == "bcocb"
    assert config["sampling"]["gamma_centroid"] == pytest.approx(.01 / FS_TO_AU)
    assert "reaction" not in config
    report = prepare(job(), numerical=True)
    assert report["ready"]
    assert report["materials_preparation"]["production_method"] == "pimd_sampling"


def test_diffusion_time_fit_and_method_selection():
    raw = job("diffusion")
    raw["diffusion"] = {"axes": [0, 1], "fit_range": ["0.2 fs", "1 fs"]}
    raw["dynamics"]["integrator"] = "normal-mode-bab"
    config, _ = JobSpec.from_mapping(raw).resolve()
    assert config["sampling"]["integrator"] == "baoab"
    assert config["sampling"]["replicas"] == 4
    np.testing.assert_allclose(config["diffusion"]["fit_range"], [.2 * FS_TO_AU, FS_TO_AU])


@pytest.mark.parametrize("task", ["pimd", "diffusion"])
def test_standalone_tasks_reject_unused_analysis_options(task):
    raw = job(task)
    raw["analysis"] = {"bootstrap": 999, "measure": "rate_flux"}
    with pytest.raises(InputError, match=r"do not use \[analysis\]"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("potential", [
    {"provider": "builtin", "name": "harmonic"},
    {"provider": "harmonic"}, {"provider": "morse_pair"},
    {"provider": "cabi", "library": "not_loaded.so"},
    {"provider": "f2py", "module": "not_imported"},
    {"provider": "rpmdrate", "module": "not_imported"},
    {"provider": "torch", "model": "not_loaded.pt"},
])
def test_known_periodic_provider_mismatch_fails_before_output(potential, tmp_path):
    from pyrpmd import run

    raw = job()
    raw["potential"] = potential
    with pytest.raises(InputError, match="cell/PBC contract|nonperiodic"):
        prepare(raw, numerical=False)
    output = tmp_path / "must-not-exist"
    with pytest.raises(InputError, match="cell/PBC contract|nonperiodic"):
        run(raw, output=output)
    assert not output.exists()


@pytest.mark.parametrize("change, message", [
    ({"reaction": {}}, "do not accept"),
    ({"umbrella": {}}, "do not accept"),
    ({"diffusion": {"axes": [0, 0]}}, "distinct"),
    ({"sampling": {"thermostat": "langevin"}}, "sampling"),
    ({"dynamics": {"loop_backend": "native"}}, "PILE-L"),
])
def test_unsupported_or_ambiguous_materials_input_rejected(change, message):
    raw = job("diffusion")
    for key, value in change.items():
        if key in raw and isinstance(raw[key], dict):
            raw[key].update(value)
        else:
            raw[key] = value
    with pytest.raises(InputError, match=message):
        JobSpec.from_mapping(raw).resolve()


def test_periodic_gas_rate_is_rejected():
    with pytest.raises(InputError, match="gas-phase"):
        JobSpec.from_mapping(job("rate")).resolve()
