"""Physical conventions and strict boundaries of the model trajectory input."""

from copy import deepcopy
import json

import numpy as np
import pytest

from pyrpmd.constants import AMU_TO_AU, FS_TO_AU, KB_AU
from pyrpmd.input import InputError, JobSpec
from pyrpmd.nonadiabatic_input import resolve_nonadiabatic_job
from pyrpmd.potentials.nonadiabatic_models import model_from_dict


def _job(method="nrpmd"):
    return {
        "task": "nonadiabatic", "method": method, "beta": 12.0,
        "model": {"name": "tully1"}, "system": {"masses": [2000.0]},
        "ensemble": {"steps": 0}, "native": False,
    }


def _spin_job():
    raw = _job()
    raw["model"] = {"name": "spin_boson", "params": {
        "frequencies": [0.2, 0.3], "couplings": [0.1, -0.05],
        "masses": [2.0, 5.0], "bias": 0.01, "tunnelling": 0.1,
    }}
    del raw["system"]
    return raw


def _set(raw, path, value):
    *parents, key = path.split(".")
    for name in parents:
        raw = raw.setdefault(name, {})
    raw[key] = value


def test_point_resolves_through_jobspec_without_io_or_mutating_input(tmp_path):
    raw = _job()
    raw["beads"] = 7
    raw["system"] = {"masses": ["2000 me"], "positions": [[-4.0, 0.0, 0.0]]}
    raw["initial"] = {"kind": "point", "momenta": [[15.0, 0.0, 0.0]], "initial_state": 1}
    raw["description"] = "An incident diabatic nuclear phase point"
    before = deepcopy(raw)
    config, system = JobSpec.from_mapping(raw, base_directory=tmp_path).resolve()
    assert raw == before
    assert config["task"] == "nonadiabatic" and config["method"] == "nrpmd"
    assert config["model"] == model_from_dict(config["model"]).to_dict()
    assert config["model"]["parameters"]["a"] == 0.01
    assert system.symbols == ("X",)
    np.testing.assert_array_equal(system.geometry, [[-4.0, 0.0, 0.0]])
    np.testing.assert_array_equal(system.masses, [2000.0])
    assert config["initial"] == before["initial"]
    assert config["ensemble"] == {"trajectories": 32, "steps": 0, "sample_stride": 1}
    assert config["dt_au"] == 0.1 and config["native"] is False
    assert config["dynamics"]["block_steps"] == 128
    assert config["output"]["directory"] == str(tmp_path / "runs" / "nonadiabatic")
    assert "not a thermal" in config["_warnings"][0]
    assert not list(tmp_path.iterdir())
    json.dumps(config, allow_nan=False)
    config["initial"]["momenta"][0][0] = 900.0
    config["system"]["geometry"][0][0] = 900.0
    assert raw == before
    assert system.geometry[0, 0] == -4.0


def test_temperature_mass_time_and_beta_units_are_dimension_checked(tmp_path):
    raw = _job()
    del raw["beta"]
    raw["temperature"] = "300 K"
    raw["system"]["masses"] = ["2 amu"]
    raw["dynamics"] = {"dt": "0.25 fs", "integrator": "normal_mode", "loop_backend": "python"}
    raw["output"] = {"directory": "../relative-results"}
    config, system = resolve_nonadiabatic_job(raw, base_directory=tmp_path, input_path="model.toml")
    assert config["beta"] == pytest.approx(1/(300*KB_AU))
    assert config["temperature_K"] == 300.0
    assert config["dt_au"] == pytest.approx(0.25*FS_TO_AU)
    assert config["dynamics"]["integrator"] == "normal-mode-bab"
    assert config["output"]["directory"] == str(tmp_path.parent / "relative-results")
    assert config["_input_path"] == str(tmp_path / "model.toml")
    np.testing.assert_allclose(system.masses, [2*AMU_TO_AU])
    raw = _job()
    raw["beta"] = "12 hartree^-1"
    config, _ = resolve_nonadiabatic_job(raw)
    assert config["beta"] == 12.0
    assert config["temperature_K"] == pytest.approx(1/(KB_AU*12.0))


def test_spin_boson_model_atom_masses_and_harmonic_reference_are_distinct():
    raw = _spin_job()
    raw["initial"] = {"kind": "harmonic_ring", "frequencies": [0.11, 0.12],
                      "centers": [1.0, -2.0], "centroid_momenta": [0.5, 1.5]}
    config, system = resolve_nonadiabatic_job(raw)
    np.testing.assert_array_equal(system.masses, [2.0, 5.0])
    np.testing.assert_array_equal(system.geometry, [[1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
    assert config["initial"]["coordinates"] == [0, 3]
    assert config["initial"]["frequencies"] == [0.11, 0.12]
    assert config["model"]["parameters"]["frequencies"] == [0.2, 0.3]
    assert "not the coupled" in config["_warnings"][0]
    raw["system"] = {"positions": system.geometry.tolist(), "masses": ["2 me", "5 me"]}
    repeated, _ = resolve_nonadiabatic_job(raw)
    assert repeated["initial"] == config["initial"]
    raw["system"]["masses"] = [2.0, 3.0]
    with pytest.raises(InputError, match="must match.*nuclear_masses"):
        resolve_nonadiabatic_job(raw)


def test_complex_lvc_identity_and_default_cartesian_coordinate_mapping():
    h0 = np.array([[0.2, 0.1j], [-0.1j, -0.2]])
    linear = np.array([np.diag([0.01, -0.01]), np.diag([0.03, 0.07])])
    raw = _job()
    raw["model"] = {"name": "lvc", "params": {
        "h0": {"real": h0.real.tolist(), "imag": h0.imag.tolist()},
        "linear": linear.tolist(), "harmonic": [0.2, 0.4],
    }}
    raw["initial"] = {"kind": "harmonic_ring", "frequencies": [0.1, 0.2], "centers": [1.0, 2.0]}
    config, system = resolve_nonadiabatic_job(raw)
    assert config["initial"]["coordinates"] == [0, 1]
    np.testing.assert_array_equal(system.geometry, [[1.0, 2.0, 0.0]])
    rebuilt = model_from_dict(json.loads(json.dumps(config["model"], allow_nan=False)))
    np.testing.assert_array_equal(rebuilt.h0, h0)
    np.testing.assert_array_equal(rebuilt.linear, linear)


def test_explicit_mult_atom_reference_order_preserves_frequency_correspondence():
    raw = _job()
    raw["model"]["params"] = {"natoms": 2, "coordinate": 4}
    raw["system"]["masses"] = [2.0, 3.0]
    raw["initial"] = {"kind": "harmonic_ring", "coordinates": [4, 0],
                      "frequencies": [0.3, 0.2], "centers": [-2.0, 4.0]}
    config, system = resolve_nonadiabatic_job(raw)
    assert config["initial"]["coordinates"] == [4, 0]
    assert config["initial"]["centroid_momenta"] == [0.0, 0.0]
    np.testing.assert_array_equal(system.geometry, [[4.0, 0.0, 0.0], [0.0, -2.0, 0.0]])


def test_rpsh_adiabatic_state_and_decoherence_options_are_canonical():
    raw = _job("rpsh")
    raw["initial"] = {"active_surface": 1}
    raw["dynamics"] = {"electronic_substeps": 12, "rescaling": "ring", "frustrated_hops": "reverse",
                       "decoherence": "edc", "decoherence_constant": "0.2 hartree"}
    config, _ = resolve_nonadiabatic_job(raw)
    assert config["initial"] == {"kind": "point", "active_surface": 1, "momenta": [[0.0, 0.0, 0.0]]}
    options = config["dynamics"]
    assert options["gap_tolerance"] == 1e-10
    assert options["electronic_substeps"] == 12
    assert options["decoherence_constant"] == 0.2
    defaults, _ = resolve_nonadiabatic_job(_job("rpsh"))
    assert defaults["dynamics"]["decoherence"] == "none"
    assert "decoherence_constant" not in defaults["dynamics"]


def test_rpsh_rejects_complex_hamiltonian_and_complex_gradient_even_at_zero_q():
    raw = _job("rpsh")
    raw["model"] = {"name": "lvc", "params": {
        "h0": np.diag([0.2, -0.2]), "linear": [[[0.0, 0.1j], [-0.1j, 0.0]]],
    }}
    with pytest.raises(InputError, match="real symmetric.*real gradients"):
        resolve_nonadiabatic_job(raw)
    raw = _spin_job()
    raw["method"] = "rpsh"
    raw["model"]["params"]["tunnelling"] = {"real": 0.1, "imag": 0.05}
    with pytest.raises(InputError, match="complex Hermitian"):
        resolve_nonadiabatic_job(raw)


def test_rpsh_reference_degeneracy_is_checked_without_constructing_a_native_driver():
    raw = _job("rpsh")
    raw["native"] = "required"
    raw["model"]["params"] = {"c": 0.0}
    with pytest.raises(InputError, match="degenerate"):
        resolve_nonadiabatic_job(raw)
    raw["method"] = "nrpmd"
    config, _ = resolve_nonadiabatic_job(raw)
    assert config["native"] == "required"


def test_one_state_rpsh_has_no_gap_constraint():
    raw = _job("rpsh")
    raw["model"] = {"name": "lvc", "params": {"h0": [[0.0]], "linear": [[[0.0]]]}}
    config, _ = resolve_nonadiabatic_job(raw)
    assert model_from_dict(config["model"]).nstates == 1


def test_process_and_threads_alias_resolve_to_execution_plan_fields():
    raw = _job()
    raw["parallel"] = {"trajectory_workers": 2, "threads": 3}
    config, _ = resolve_nonadiabatic_job(raw)
    assert config["parallel"] == {"backend": "process", "mode": "serial",
                                  "trajectory_workers": 2, "bead_workers": 1, "threads_per_rank": 3}
    from pyrpmd.parallel import execution_plan
    plan = execution_plan(config["parallel"])
    assert plan.threads == 3 and plan.trajectory_workers == 2


@pytest.mark.parametrize("path,value", [
    ("irrelevant", 1), ("schema_version", True), ("schema_version", 2),
    ("method", "thermal-nrpmd"), ("beads", True), ("beads", 0), ("beads", 1.5),
    ("seed", -1), ("seed", np.bool_(True)), ("beta", False), ("beta", 0),
    ("beta", float("nan")), ("beta", float("inf")), ("beta", "3 fs"),
    ("system", []), ("system.masses", [0.0]), ("system.masses", [True]),
    ("system.masses", None), ("system.masses", ["2000 fs"]),
    ("system.positions", [[True, 0.0, 0.0]]), ("system.positions", [[0.0, 0.0]]),
    ("system.positions", [[0.0, 0.0, float("inf")]]), ("system.symbols", ["H"]),
    ("model.name", "import_custom_factory"), ("model.params.unknown", 1),
    ("model.params.a", True), ("model.params.b", -0.1), ("model.params.a", "0.01 hartree"),
    ("initial.kind", "thermal"), ("initial.active_surface", 0), ("initial.initial_state", 2),
    ("initial.initial_state", True), ("initial.momenta", [[0.0, True, 0.0]]),
    ("initial.momenta", [[1j, 0.0, 0.0]]), ("initial.centers", [0.0]),
    ("ensemble.steps", -1), ("ensemble.steps", 1.0), ("ensemble.steps", False),
    ("ensemble.trajectories", 0), ("ensemble.sample_stride", 0), ("ensemble.bad", 1),
    ("dynamics.dt", -0.1), ("dynamics.dt", True), ("dynamics.dt", "1 me"),
    ("dynamics.integrator", "unknown"), ("dynamics.loop_backend", "native"),
    ("dynamics.block_steps", 0), ("dynamics.electronic_substeps", 8),
    ("dynamics.decoherence", "none"), ("dynamics.threads", 1),
    ("parallel.backend", "mpi"), ("parallel.mode", "gpu"), ("parallel.mode", "bead"),
    ("parallel.bead_workers", 2), ("parallel.threads", False), ("parallel.trajectory_workers", 0),
    ("parallel.devices", [0]), ("output.directory", ""), ("output.checkpoint_every", 10),
    ("native", 1), ("native", "auto"), ("description", 1),
])
def test_invalid_or_unused_input_is_rejected_as_inputerror(path, value):
    raw = _job()
    _set(raw, path, value)
    with pytest.raises(InputError):
        resolve_nonadiabatic_job(raw)


@pytest.mark.parametrize("missing", ["method", "beta", "model", "system", "ensemble"])
def test_required_scientific_input_is_not_guessed(missing):
    raw = _job()
    del raw[missing]
    with pytest.raises(InputError):
        resolve_nonadiabatic_job(raw)


@pytest.mark.parametrize("path,value", [
    ("frequencies", []), ("frequencies", [0.0]), ("frequencies", [True]),
    ("frequencies", ["0.1 au"]), ("frequencies", [[0.1]]),
    ("coordinates", [0.0]), ("coordinates", [True]), ("coordinates", [3]),
    ("coordinates", [0, 0]), ("centers", [0.0, 0.0]),
    ("centroid_momenta", [float("nan")]), ("momenta", [[0.0, 0.0, 0.0]]),
])
def test_harmonic_reference_rejects_shape_unit_and_coordinate_ambiguities(path, value):
    raw = _job()
    raw["initial"] = {"kind": "harmonic_ring", "frequencies": [0.1]}
    raw["initial"][path] = value
    with pytest.raises(InputError):
        resolve_nonadiabatic_job(raw)


def test_harmonic_reference_cannot_silently_ignore_conflicting_positions():
    raw = _job()
    raw["initial"] = {"kind": "harmonic_ring", "frequencies": [0.1], "centers": [1.0]}
    raw["system"]["positions"] = [[1.0, 0.01, 0.0]]
    with pytest.raises(InputError, match="system.positions must equal"):
        resolve_nonadiabatic_job(raw)
    del raw["initial"]["frequencies"]
    with pytest.raises(InputError, match="initial.frequencies is required"):
        resolve_nonadiabatic_job(raw)


@pytest.mark.parametrize("path,value", [
    ("initial.initial_state", 0), ("initial.active_surface", 2),
    ("dynamics.electronic_substeps", True), ("dynamics.electronic_substeps", 0),
    ("dynamics.rescaling", "full"), ("dynamics.frustrated_hops", "ignore"),
    ("dynamics.gap_tolerance", 0.0), ("dynamics.gap_tolerance", 1j),
    ("dynamics.decoherence", True), ("dynamics.decoherence", "other"),
    ("dynamics.decoherence_constant", 0.1),
])
def test_rpsh_method_specific_invalid_input(path, value):
    raw = _job("rpsh")
    _set(raw, path, value)
    with pytest.raises(InputError):
        resolve_nonadiabatic_job(raw)


def test_conflicting_aliases_and_thermal_parameters_are_not_silently_overridden():
    cases = [
        {"temperature": "300 K"},
        {"dynamics": {"native": True}},
        {"parallel": {"threads": 2, "threads_per_rank": 3}},
        {"parallel": {"backend": "serial", "trajectory_workers": 2}},
    ]
    for extra in cases:
        raw = _job()
        raw.update(extra)
        with pytest.raises(InputError):
            resolve_nonadiabatic_job(raw)


@pytest.mark.parametrize("name", ["tully1", "tully2", "tully3"])
def test_three_tully_names_keep_distinct_complete_identities(name):
    raw = _job()
    raw["model"]["name"] = name
    config, _ = resolve_nonadiabatic_job(raw)
    assert config["model"]["model"] == name
    assert config["model"]["parameters"]["coordinate"] == 0
    assert config["model"]["parameters"]["natoms"] == 1


def test_model_overflow_at_reference_is_reported_before_output_creation(tmp_path):
    raw = _job()
    raw["model"] = {"name": "lvc", "params": {"h0": [[0.0]], "linear": [[[1e308]]]}}
    raw["system"]["positions"] = [[10.0, 0.0, 0.0]]
    with pytest.raises(InputError, match="overflowed"):
        resolve_nonadiabatic_job(raw, base_directory=tmp_path)
    assert not list(tmp_path.iterdir())
