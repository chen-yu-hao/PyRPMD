"""Method selection, molecular identity and execution-only accelerator choices."""
from copy import deepcopy
import json

import numpy as np
import pytest

from pyrpmd import JobSpec, run, resume_run
from pyrpmd.input import InputError
from pyrpmd.io import fingerprint
from pyrpmd.nonadiabatic_registry import METHODS, make_dynamics, initialize_state
from pyrpmd.propagation import scientific_config
from pyrpmd.units import parse_quantity


def job(method="sm-nrpmd"):
    return {"task": "nonadiabatic", "method": method, "beta": 20., "beads": 2,
            "model": {"name": "tully1"}, "system": {"masses": [2000.]},
            "dynamics": {"native": False, "dt": .02},
            "ensemble": {"trajectories": 2, "steps": 2}}


def molecular(method="sm-nrpmd"):
    raw = job(method)
    raw["model"] = {"name": "nah2_leps_5g_1992"}
    raw["system"] = {"symbols": ["Na", "H", "H"],
                     "positions": [[3.2, .8, 0.], [-.85, 0., 0.], [.85, 0., 0.]]}
    return raw


@pytest.mark.parametrize("method", sorted(METHODS - {"iesh", "multistate-mash"}))
def test_every_method_constructs_real_molecular_driver(method):
    raw = molecular(method)
    before = deepcopy(raw)
    config, system = JobSpec.from_mapping(raw).resolve()
    assert raw == before
    assert system.charge == 0 and system.multiplicity == 2
    assert system.symbols == ("Na", "H", "H")
    driver = make_dynamics(config, system, np.random.default_rng(3))
    try:
        state = initialize_state(config, system, driver, np.random.default_rng(4), np.random.default_rng(5))
        driver.step(state)
        assert state.step == 1 and np.isfinite(driver.energy(state))
    finally:
        driver.potential.close()


def test_molecular_units_atom_permutation_and_default_isotope_masses():
    raw = molecular()
    raw["system"]["symbols"] = ["H", "Na", "H"]
    raw["system"]["positions"] = np.asarray(raw["system"]["positions"])[[1, 0, 2]].tolist()
    raw["system"]["unit"] = "angstrom"
    config, system = JobSpec.from_mapping(raw).resolve()
    assert system.symbols == ("H", "Na", "H")
    assert system.masses[1] > 20*system.masses[0]
    assert system.masses[0] == system.masses[2]
    np.testing.assert_allclose(system.geometry, np.asarray(raw["system"]["positions"])*parse_quantity("1 angstrom", "length"))
    assert config["model"]["parameters"]["symbols"] == ["H", "Na", "H"]


@pytest.mark.parametrize("bad", [None, 2, True, "NaHH", [["Na", "H", "H"]], ["Na", ["H", "H"]]])
def test_malformed_molecular_atom_order_is_input_error(bad):
    raw = molecular()
    raw["model"]["params"] = {"symbols": bad}
    with pytest.raises(InputError, match="symbols"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method,options,match", [
    ("rp-mash", {"native": True}, "requires native=false"),
    ("rp-mash", {"integrator": "cayley-bcb"}, "adaptive-dop853"),
    ("rpsh-dm", {"decoherence": "edc"}, "requires decoherence"),
    ("rpsh-ba", {"rescaling": "ring"}, "rescaling"),
    ("rpsh-ca", {"dboc_step": .001}, "requires dboc"),
    ("soc-rpsh", {"dboc": "yes"}, "boolean"),
    ("rpsh", {"dboc": True}, "Unknown"),
    ("sm-nrpmd", {"electronic_backend": "native", "electronic_device": "cuda"}, "requires electronic_backend"),
    ("sm-nrpmd", {"electronic_backend": "cupy"}, "electronic_backend"),
    ("nrpmd", {"electronic_backend": "torch", "electronic_device": "cuda:-1"}, "electronic_device"),
    ("rp-mash", {"max_events": True}, "integer"),
    ("rpsh-ca", {"dboc": True, "dboc_step": "1 fs"}, "dboc_step"),
])
def test_unused_or_incompatible_method_options_are_rejected(method, options, match):
    raw = job(method)
    raw["dynamics"].update(options)
    with pytest.raises(InputError, match=match):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method", ["sm-nrpmd", "rp-mash"])
def test_two_state_methods_reject_three_states(method):
    raw = job(method)
    raw["model"] = {"name": "lvc", "params": {"h0": np.diag([-.2, 0., .2]),
                                               "linear": [np.diag([.1, .2, .3])]}}
    with pytest.raises(InputError, match="exactly two"):
        JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method", ["sm-nrpmd", "rpsh-ca", "rpsh-dm", "soc-rpsh"])
def test_complex_lvc_is_accepted_only_by_compatible_drivers(method):
    raw = job(method)
    raw["model"] = {"name": "lvc", "params": {
        "h0": {"real": [[-.2, .1], [.1, .2]], "imag": [[0., .02], [-.02, 0.]]},
        "linear": [[[.03, 0.], [0., -.03]]], "harmonic": [.01]}}
    config, _ = JobSpec.from_mapping(raw).resolve()
    assert config["method"] == method
    for incompatible in ("rpsh", "rpsh-ba", "rp-mash"):
        raw["method"] = incompatible
        with pytest.raises(InputError, match="real symmetric"):
            JobSpec.from_mapping(raw).resolve()


@pytest.mark.parametrize("method", ["nrpmd", "sm-nrpmd"])
def test_offline_gpu_resolution_and_physics_identity(method):
    raw = job(method)
    original, system = JobSpec.from_mapping(raw).resolve()
    raw["dynamics"].update(electronic_backend="torch", electronic_device="cuda:2")
    accelerated, other = JobSpec.from_mapping(raw).resolve()
    assert accelerated["dynamics"]["electronic_device"] == "cuda:2"
    assert fingerprint(scientific_config(original), system) == fingerprint(scientific_config(accelerated), other)
    raw["dynamics"]["dt"] *= 2
    changed, other = JobSpec.from_mapping(raw).resolve()
    assert fingerprint(scientific_config(changed), other) != fingerprint(scientific_config(original), system)


@pytest.mark.parametrize("method", ["nrpmd", "sm-nrpmd"])
def test_resume_accepts_explicit_electronic_execution_backend(tmp_path, method):
    pytest.importorskip("pyrpmd._electronic")
    path = tmp_path/method
    result = run(job(method), output=path)
    saved = json.loads((path/"manifest.json").read_text())["physics_fingerprint"]
    resumed = resume_run(path, execution={"electronic_backend": "native", "electronic_device": "cpu"})
    assert resumed["completed"] and resumed["final_populations"] == result["final_populations"]
    assert json.loads((path/"manifest.json").read_text())["physics_fingerprint"] == saved
    with pytest.raises(InputError, match="requires electronic_backend"):
        resume_run(path, execution={"electronic_device": "cuda"})
