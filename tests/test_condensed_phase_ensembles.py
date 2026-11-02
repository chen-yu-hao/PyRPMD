"""SC/PIGLET pressure composition, invalid measures and exact continuation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pyrpmd.barostat import BZPBarostat
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import InputError, JobSpec
from pyrpmd.native_ensemble import NativeEnsemble
from pyrpmd.pimd_methods import SuzukiChinPotential
from pyrpmd.potentials import FreePotential, PESResult, PotentialError, PythonPotential
from pyrpmd.system import System
from pyrpmd.runner import prepare
from pyrpmd.workflows.path_integral import KIND, run_path_integral


def piglet_spec(beads=4, beta=2.0, *, canonical=False):
    """Synthetic test bath; never a substitute for calibrated PIGLET data."""
    a = np.repeat(np.array([[1., .15], [-.15, 1.]])[None], beads, axis=0)
    c = np.repeat(np.eye(2)[None], beads, axis=0)
    if not canonical:
        c[1:, 0, 0] = 1.7
    return {"mode": "piglet", "gle": {
        "A": a.tolist(), "C": c.tolist(), "beta": beta, "beads": beads,
        "provenance": "TEST ONLY: synthetic covariance, not a calibrated PIGLET set",
    }}


def native_piglet(*, thermostat=None, potential=None, seed=81):
    return NativeEnsemble(np.full((4, 2, 3), .2), [2., 3.], 2., .015,
        cell=np.eye(3) * 2, thermostat=piglet_spec() if thermostat is None else thermostat,
        barostat={"mode": "bzp", "pressure": .2, "tau": .8, "gamma": 1.2},
        potential={"kind": "free"} if potential is None else potential, seed=seed)


def test_canonical_piglet_limit_is_identical_to_normal_mode_gle_npt():
    pytest.importorskip("pyrpmd._thermal")
    a = native_piglet(thermostat=piglet_spec(canonical=True))
    spec = piglet_spec(canonical=True)
    spec["mode"] = "nm-gle"
    b = native_piglet(thermostat=spec)
    a.run(40)
    b.run(40)
    for name in ("q", "p", "aux", "cell", "cell_p", "rng"):
        np.testing.assert_array_equal(a.state()[name], b.state()[name])


def test_noncanonical_piglet_npt_restores_auxiliaries_and_piston_bitwise():
    pytest.importorskip("pyrpmd._thermal")
    a = native_piglet(potential={"kind": "periodic_cosine", "heights": [.1, .2, .3]})
    a.run(17)
    checkpoint = a.state()
    a.run(29)
    b = native_piglet(potential={"kind": "periodic_cosine", "heights": [.1, .2, .3]})
    b.restore(checkpoint)
    b.run(29)
    for name in ("q", "p", "aux", "cell", "cell_p", "rng", "energy", "force", "virial"):
        np.testing.assert_array_equal(a.state()[name], b.state()[name])
    assert np.isfinite(a.state()["q"]).all()
    assert not np.array_equal(checkpoint["cell"], a.state()["cell"])


def test_piglet_npt_free_gas_volume_measure_is_gamma_with_n_plus_one():
    """Noncanonical internal modes must not heat the centroid/cell measure."""
    pytest.importorskip("pyrpmd._thermal")
    atoms, beta, pressure = 8, 2., .2
    target = (atoms + 1) / (beta * pressure)
    variance = (atoms + 1) / (beta * pressure)**2
    engine = NativeEnsemble(np.zeros((4, atoms, 3)), np.ones(atoms), beta, .01,
        cell=np.eye(3) * target**(1 / 3), thermostat=piglet_spec(),
        barostat={"mode": "bzp", "pressure": pressure, "tau": .7, "gamma": 2.},
        potential={"kind": "free"}, seed=324)
    volumes = engine.run(160000, stride=10)["samples"][2000:, 6]
    batches = np.array_split(volumes, 20)
    mean_error = np.std([part.mean() for part in batches], ddof=1) / np.sqrt(20)
    variance_error = np.std([np.mean((part - target)**2) for part in batches], ddof=1) / np.sqrt(20)
    assert abs(volumes.mean() - target) < 6 * mean_error + .01 * target
    assert abs(np.mean((volumes - target)**2) - variance) < 6 * variance_error + .02 * variance


@pytest.mark.parametrize("bad", ["cross_covariance", "temperature", "flexible"])
def test_piglet_npt_rejects_noncanonical_centroid_or_unsupported_cell(bad):
    pytest.importorskip("pyrpmd._thermal")
    spec = piglet_spec()
    if bad == "cross_covariance":
        spec["gle"]["C"][0] = [[1., .1], [.1, 1.]]
    elif bad == "temperature":
        spec["gle"]["C"][0][0][0] = 1.2
    barostat = {"mode": "mttk" if bad == "flexible" else "bzp", "pressure": .2}
    with pytest.raises(ValueError, match="canonical centroid|isotropic BZP"):
        NativeEnsemble(np.zeros((4, 1, 3)), [2.], 2., .01,
                       thermostat=spec, barostat=barostat, cell=np.eye(3))


def test_piglet_external_bzp_checkpoint_and_force_failure_restore_every_operator():
    pytest.importorskip("pyrpmd._thermal")
    system = System(("H", "H"), [2., 3.], [[.1, .2, .3], [.5, .2, .1]], cell=np.eye(3) * 2)

    def make():
        potential = FreePotential()
        piston = BZPBarostat(potential, system, 2., 4, .2, .8)
        dynamics = Dynamics(potential, system.masses, 2., 4, .02, seed=17,
                            thermostat=piglet_spec(), barostat=piston, cell=system.cell,
                            pbc=system.pbc, native=False, loop_backend="python")
        state = State(np.repeat(system.geometry[None], 4, axis=0), dynamics.sample_momenta())
        return potential, piston, dynamics, state

    potential, piston, dynamics, state = make()
    for _ in range(5):
        dynamics.step(state)
    saved, saved_ops = state.copy(), dynamics.operator_state_dict()
    original = potential.evaluate
    potential.evaluate = lambda q: (_ for _ in ()).throw(RuntimeError("injected pressure force failure"))
    with pytest.raises(RuntimeError, match="injected"):
        dynamics.step(state)
    np.testing.assert_array_equal(saved.q, state.q)
    np.testing.assert_array_equal(saved.p, state.p)
    assert dynamics.operator_state_dict() == saved_ops
    potential.evaluate = original
    for _ in range(9):
        dynamics.step(state)
    _, other_piston, other_dynamics, other = make()
    other = saved.copy()
    other_dynamics.load_operator_state_dict(saved_ops)
    for _ in range(9):
        other_dynamics.step(other)
    np.testing.assert_array_equal(other.q, state.q)
    np.testing.assert_array_equal(other.p, state.p)
    assert other_piston.snapshot() == piston.snapshot()
    assert other_dynamics.operator_state_dict() == dynamics.operator_state_dict()


class FractionalWell:
    metadata = {"periodic": True}

    def evaluate(self, q, *, cell, pbc):
        volume = np.linalg.det(cell)
        inverse = np.linalg.inv(cell)
        phase = 2 * np.pi * q @ inverse
        energy = .2 * np.sum(1 - np.cos(phase), axis=(1, 2)) + .3 / volume
        force = -.4 * np.pi * np.sin(phase) @ inverse.T
        virial = np.repeat((np.eye(3) * .3 / volume)[None], len(q), axis=0)
        return PESResult(energy, force, virial=virial)


def test_sc_npt_piston_uses_effective_action_cell_derivative():
    cell = np.array([[2.8, 0, 0], [.1, 3.0, 0], [.2, .1, 3.2]])
    system = System(("H", "H"), [2., 3.], [[.1, .3, .2], [.4, .1, .2]], cell=cell)
    action = SuzukiChinPotential(PythonPotential(FractionalWell(), cell=cell), system.masses, 3., 4)
    piston = BZPBarostat(action, system, 3., 4, .02, .8)
    rng = np.random.default_rng(7)
    q = system.geometry[None] + rng.normal(scale=.2, size=(4, 2, 3))
    state = State(q, rng.normal(size=q.shape))
    centroid = q.mean(axis=0)
    energies = []
    epsilon = 2e-6
    for sign in (-1, 1):
        scale = np.exp(sign * epsilon)
        action.set_cell(cell * scale)
        energies.append(action.evaluate(q + centroid[None] * (scale - 1)).energy.sum())
    action.set_cell(cell)
    kinetic = 4 * np.sum(state.p.mean(axis=0)**2 / system.masses[:, None])
    expected = (kinetic - (energies[1] - energies[0]) / (2 * epsilon)) / (12 * np.linalg.det(cell))
    assert piston.pressure(state) == pytest.approx(expected, rel=2e-7, abs=1e-9)


def sc_input():
    return {
        "task": "pimd", "temperature": 1 / (3.1 * KB_AU), "beads": 4,
        "seed": 913, "native": False,
        "system": {"symbols": ["H", "H"], "masses": [2., 3.],
                   "geometry": [[.1, -.2, .3], [-.1, .2, -.3]], "cell": (np.eye(3) * 3).tolist()},
        "potential": {"provider": "free"},
        "dynamics": {"integrator": "normal-mode-bab", "timestep": .01},
        "sampling": {"equilibration_steps": 3, "production_steps": 12, "sample_interval": 2,
                     "thermostat": "pile-l"},
        "path_integral": {"method": "suzuki_chin"},
        "barostat": {"method": "bzp", "pressure": .02, "time_constant": 2., "friction": .5},
        "output": {"checkpoint_steps": 4},
    }


def checkpoint_arrays(path):
    with load_checkpoint(path, require_integrity=True) as saved:
        return {key: saved[key].copy() for key in saved.files}


def test_sc_npt_workflow_restores_piston_after_estimator_failure(tmp_path, monkeypatch):
    config, system = JobSpec.from_mapping(sc_input()).resolve()
    reference = run_path_integral(config, system, tmp_path / "reference")
    original = SuzukiChinPotential.estimators
    calls = 0

    def fail(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(self, *args, **kwargs)
        if calls == 3:
            raise RuntimeError("injected SC estimator failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(SuzukiChinPotential, "estimators", fail)
        with pytest.raises(RuntimeError, match="SC estimator"):
            run_path_integral(config, system, tmp_path / "resumed")
    partial = checkpoint_arrays(tmp_path / "resumed/checkpoints/path_integral_0000.npz")
    assert partial["elapsed"] == 8
    resumed = run_path_integral(config, system, tmp_path / "resumed")
    a = checkpoint_arrays(reference["replicas"][0]["checkpoint"])
    b = checkpoint_arrays(resumed["replicas"][0]["checkpoint"])
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert resumed["ensemble"] == "npt"
    report = resumed["replicas"][0]["thermodynamics"]
    assert report["heat_capacity_kind"] == "Cp"
    assert report["internal_energy_hartree"] == pytest.approx(3 / config["beta"])
    assert "pressure_au" in report
    monkeypatch.setattr(Dynamics, "sample_step", lambda *args, **kwargs: pytest.fail("unexpected propagation"))
    assert run_path_integral(config, system, tmp_path / "resumed") == resumed


def test_fixed_cell_sc_workflow_reports_cv(tmp_path):
    raw = sc_input()
    raw.pop("barostat")
    config, system = JobSpec.from_mapping(raw).resolve()
    result = run_path_integral(config, system, tmp_path)
    report = result["replicas"][0]["thermodynamics"]
    assert report["heat_capacity_kind"] == "Cv"
    assert report["heat_capacity_over_kB"] == pytest.approx(3.)
    assert report["pressure_au"] == pytest.approx(2 / (config["beta"] * np.linalg.det(system.cell)))


@pytest.mark.parametrize("npt", [False, True])
@pytest.mark.parametrize("method, selected, other", [
    ("bcocb", "cayley-bcb", "normal-mode-bab"),
    ("baoab", "normal-mode-bab", "cayley-bcb"),
])
def test_sc_workflow_uses_selected_sampling_map_and_resumes_exactly(
        tmp_path, monkeypatch, npt, method, selected, other):
    raw = sc_input()
    raw["dynamics"]["integrator"] = other
    raw["sampling"]["integrator"] = method
    if not npt:
        raw.pop("barostat")
    config, system = JobSpec.from_mapping(raw).resolve()
    mixed = run_path_integral(config, system, tmp_path / "mixed")

    # An independent dedicated Dynamics kernel must generate the same physical
    # trajectory. Merely accepting or reporting sampling.integrator is not enough.
    dedicated_raw = copy.deepcopy(raw)
    dedicated_raw["dynamics"]["integrator"] = selected
    dedicated_config, dedicated_system = JobSpec.from_mapping(dedicated_raw).resolve()
    dedicated = run_path_integral(dedicated_config, dedicated_system, tmp_path / "dedicated")
    a = checkpoint_arrays(mixed["replicas"][0]["checkpoint"])
    b = checkpoint_arrays(dedicated["replicas"][0]["checkpoint"])
    fields = ["q", "p", "positions", "potential", "sample_steps", "rng", "sc_estimators"]
    if npt:
        fields += ["cells", "barostat"]
    for key in fields:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    original = SuzukiChinPotential.estimators
    calls = 0

    def fail(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(self, *args, **kwargs)
        if calls == 3:
            raise RuntimeError("injected mixed SC observation failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(SuzukiChinPotential, "estimators", fail)
        with pytest.raises(RuntimeError, match="mixed SC observation"):
            run_path_integral(config, system, tmp_path / "resumed")
    resumed = run_path_integral(config, system, tmp_path / "resumed")
    c = checkpoint_arrays(resumed["replicas"][0]["checkpoint"])
    assert set(a) == set(c)
    for key in a:
        np.testing.assert_array_equal(a[key], c[key], err_msg=key)


@pytest.mark.parametrize("bad", [True, 0, -1e-3, 1., 1e-17, np.nan])
def test_sc_beta_displacement_is_explicitly_validated(bad):
    raw = sc_input()
    raw["path_integral"]["beta_fd_step"] = bad
    with pytest.raises(InputError, match="beta_fd_step"):
        JobSpec.from_mapping(raw).resolve()


def test_sc_npt_numerical_preparation_checks_action_pressure_without_writing(tmp_path):
    raw = sc_input()
    raw["path_integral"]["beta_fd_step"] = .002
    raw["output"]["directory"] = str(tmp_path / "absent")
    result = prepare(raw, numerical=True)
    assert result["path_integral_preparation"]["ensemble"] == "npt"
    assert "initial_action_virial_hartree" in result["pes_check"]
    assert result["pes_check"]["initial_sc_estimators"]["finite_difference_relative_beta_step"] == .001
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("bad", ["normalization", "boolean", "missing", "displacement"])
def test_sc_checkpoint_validates_thermodynamic_estimator_schema(tmp_path, bad):
    config, system = JobSpec.from_mapping(sc_input()).resolve()
    result = run_path_integral(config, system, tmp_path)
    path = result["replicas"][0]["checkpoint"]
    data = {key: value for key, value in checkpoint_arrays(path).items() if not key.startswith("_")}
    rows = json.loads(str(data["sc_estimators"]))
    if bad == "normalization":
        rows[0]["normalization_dof"] = 9
    elif bad == "boolean":
        rows[0]["internal_energy_hartree"] = True
    elif bad == "missing":
        for row in rows:
            del row["scaled_action_second_beta_derivative"]
    else:
        rows[0]["finite_difference_relative_beta_step"] *= 2
    data["sc_estimators"] = json.dumps(rows)
    atomic_savez(path, kind=KIND, **data)
    with pytest.raises(ValueError, match="SC estimator checkpoint"):
        run_path_integral(config, system, tmp_path)


def test_sc_and_primitive_piglet_cannot_be_accidentally_combined():
    action = SuzukiChinPotential(FreePotential(), [2.], 2., 4)
    with pytest.raises(ValueError, match="PIGLET.*Suzuki-Chin"):
        Dynamics(action, [2.], 2., 4, .01, thermostat=piglet_spec(), native=False)
    pytest.importorskip("pyrpmd._thermal")
    with pytest.raises(ValueError, match="primitive estimators"):
        NativeEnsemble(np.zeros((4, 1, 3)), [2.], 2., .01,
                       potential=action, allow_python=True, thermostat="pile-l")


def test_periodic_sc_nvt_can_report_energy_without_a_provider_virial(tmp_path):
    raw = sc_input()
    raw.pop("barostat")
    raw["potential"] = {"provider": "periodic_cosine", "params": {"height": .001}}
    config, system = JobSpec.from_mapping(raw).resolve()
    checked = prepare(raw, numerical=True)
    assert "pressure_au" not in checked["pes_check"]["initial_sc_estimators"]
    result = run_path_integral(config, system, tmp_path)
    report = result["replicas"][0]["thermodynamics"]
    assert "pressure_au" not in report
    assert np.isfinite(report["internal_energy_hartree"])
    assert report["heat_capacity_kind"] == "Cv"
    raw["barostat"] = sc_input()["barostat"]
    with pytest.raises(PotentialError, match="virial"):
        prepare(raw, numerical=True)


def test_piglet_npt_native_workflow_preflights_actual_matrices(tmp_path):
    pytest.importorskip("pyrpmd._thermal")
    raw = sc_input()
    raw.pop("path_integral")
    raw["native"] = True
    raw["temperature"] = 1 / (2 * KB_AU)
    raw["sampling"].update(engine="native", thermostat="piglet",
                           thermostat_params={"gle": piglet_spec()["gle"]})
    raw["barostat"] = {"mode": "bzp", "pressure": .02, "tau": 2., "gamma": .5}
    raw["output"]["directory"] = str(tmp_path / "absent")
    checked = prepare(raw, numerical=True)
    assert checked["ready"]
    assert checked["materials_preparation"]["ensemble"] == "npt"
    assert checked["pes_check"]["engine"] == "native"
    assert not (tmp_path / "absent").exists()
    raw["sampling"]["thermostat_params"]["gle"]["C"][0] = [[1., .1], [.1, 1.]]
    with pytest.raises(ValueError, match="canonical centroid"):
        prepare(raw, numerical=True)


@pytest.mark.parametrize("npt", [False, True])
def test_native_pressure_metadata_matches_saved_momenta_not_configured_temperature(tmp_path, npt):
    pytest.importorskip("pyrpmd._thermal")
    from pyrpmd.native_ensemble import _COLUMNS
    from pyrpmd.workflows.native_pimd import run_native_pimd

    raw = sc_input()
    raw.pop("path_integral")
    raw["native"] = True
    raw["temperature"] = 1 / (2 * KB_AU)
    raw["sampling"].update(engine="native", thermostat="piglet",
                           thermostat_params={"gle": piglet_spec()["gle"]})
    if npt:
        raw["barostat"] = {"mode": "bzp", "pressure": .02, "tau": 2., "gamma": .5}
    else:
        raw.pop("barostat")
    config, system = JobSpec.from_mapping(raw).resolve()
    summary = run_native_pimd(config, system, tmp_path)
    journal_path = Path(summary["files"]["journals"][0])
    journal = json.loads(journal_path.read_text())
    chunks = [np.load(journal_path.parent / block["directory"] / "samples.npy", allow_pickle=False)
              for block in journal["blocks"] if block["samples_sha256"] is not None]
    rows = np.concatenate(chunks)
    n, scalar_count = system.natoms, len(_COLUMNS)
    velocities = rows[:, scalar_count+3*n:scalar_count+6*n].reshape(-1, n, 3)
    centroid_twice_kinetic = np.sum(system.masses[None, :, None] * velocities**2, axis=(1, 2))
    volumes, recorded_pressure = rows[:, 6], rows[:, 7]
    # For a free gas R=0. The raw pressure must use the actual momenta,
    # while the independent configured-temperature estimator is N/(beta V).
    np.testing.assert_allclose(recorded_pressure, centroid_twice_kinetic/(3*volumes), rtol=2e-14)
    configured_temperature = recorded_pressure + (3*n/config["beta"]-centroid_twice_kinetic)/(3*volumes)
    np.testing.assert_allclose(configured_temperature, n/(config["beta"]*volumes), rtol=2e-14)
    assert summary["pressure_mean_au"] == pytest.approx(recorded_pressure.mean())
    assert not np.isclose(recorded_pressure.mean(), configured_temperature.mean(), rtol=1e-6)
    metadata = summary["pressure_estimator"]
    assert metadata["name"] == "instantaneous_centroid_momentum_virial"
    assert metadata["uses_configured_temperature_kinetic_term"] is False
    assert metadata["role"] == ("piston_feedback" if npt else "fixed_cell_diagnostic")
