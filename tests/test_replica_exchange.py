"""REUS ensemble measure, transaction boundaries, and execution regressions."""
import itertools
import copy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from pyrpmd.bias import UmbrellaBias
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.cv import Angle, Cartesian, CVStack, PeriodicDistance, RPMDrateCoordinate
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.input import InputError, JobSpec
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.system import System
from pyrpmd.workflows.fes import run_fes, reanalyze_fes, _fingerprint
from pyrpmd.workflows.sampling import checkpoint_cv_config
import pyrpmd.workflows.replica_exchange as rex


def system():
    return System(("H",), np.array([2.3]), np.zeros((1, 3)))


def config(thermostat="andersen", scheme="cayley-bcb"):
    c = {"task": "fes", "beta": 1.7, "beads": 4, "dt_au": .03,
         "seed": 67, "native": False, "dynamics": {"integrator": scheme},
         "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .4}},
         "sampling": {"thermostat": thermostat, "equilibration_steps": 8,
                      "production_steps": 40, "sample_interval": 2, "replicas": 2,
                      "andersen_interval": 7},
         "cvs": [{"name": a, "kind": "cartesian", "atom": 0, "axis": a} for a in ("x", "y")],
         "umbrella": {"centers": [[-.2, -.2], [-.2, .2], [.2, -.2], [.2, .2]],
                      "stiffness": [[1., .2], [.2, .8]]},
         "replica_exchange": {"enabled": True, "interval_steps": 9},
         "analysis": {"bins": [4, 5], "bootstrap": 4}, "output": {"checkpoint_steps": 11}}
    if thermostat == "pile-l":
        c["sampling"]["gamma_centroid"] = .5
    return c


def read(path):
    with load_checkpoint(path, expected_kind=rex.CHECKPOINT_KIND, require_integrity=True) as archive:
        return {k: archive[k] for k in archive.files}


def assert_same_checkpoints(a, b):
    first = sorted(Path(a).glob("checkpoints/exchange_*.npz"))
    assert first
    for file in first:
        x, y = read(file), read(Path(b) / "checkpoints" / file.name)
        assert x.keys() == y.keys()
        for key in x:
            np.testing.assert_array_equal(x[key], y[key], err_msg=key)


@pytest.mark.parametrize("beads", [1, 4, 16])
@pytest.mark.parametrize("periodic", [False, True])
def test_exponent_matches_full_ring_hamiltonians(beads, periodic):
    rng = np.random.default_rng(15)
    cv = CVStack([Cartesian(0, "x"), Cartesian(0, "y")])
    if periodic:
        cv.periods = (2., 3.)
    a = UmbrellaBias(cv, [.1, -.3], [[1., .2], [.2, .8]])
    b = UmbrellaBias(cv, [-.5, .2], [[.6, -.1], [-.1, 1.2]])
    beta = 2.7
    dyn = Dynamics(HarmonicPotential(.4), [2.3], beta, beads, .02, native=False)
    qi, qj = rng.normal(size=(beads, 1, 3)), rng.normal(size=(beads, 1, 3))
    pi, pj = rng.normal(size=qi.shape), rng.normal(size=qj.shape)
    initial = dyn.energy(State(qi, pi), a) + dyn.energy(State(qj, pj), b)
    swapped = dyn.energy(State(qj, pj), a) + dyn.energy(State(qi, pi), b)
    actual = rex.exchange_log_ratio(a, b, cv.value(qi), cv.value(qj), beta)
    assert actual == pytest.approx(-beta / beads * (swapped - initial), abs=4e-12)
    assert rex.exchange_log_ratio(b, a, cv.value(qi), cv.value(qj), beta) == pytest.approx(-actual)


def test_discrete_pair_kernel_obeys_detailed_balance():
    cv = CVStack([Cartesian(0, "x")])
    a, b = UmbrellaBias(cv, [-.4], [[1.2]]), UmbrellaBias(cv, [.6], [[.9]])
    beta = 1.3
    states = list(itertools.product([-1., .1, 1.2], repeat=2))
    def physical(x):
        return .3 * x*x + .2*x**4
    weights = np.array([np.exp(-beta * (physical(x) + physical(y) +
                        a.physical_energy([x]) + b.physical_energy([y]))) for x, y in states])
    weights /= weights.sum()
    transition = np.zeros((len(states), len(states)))
    for i, (x, y) in enumerate(states):
        j = states.index((y, x))
        accept = np.exp(min(0., rex.exchange_log_ratio(a, b, [x], [y], beta)))
        transition[i, j] += accept
        transition[i, i] += 1 - accept
    np.testing.assert_allclose(weights[:, None] * transition,
                               (weights[:, None] * transition).T, rtol=2e-15, atol=2e-17)
    np.testing.assert_allclose(weights @ transition, weights, rtol=2e-15, atol=2e-17)


def test_periodic_graph_connects_across_branch_cut():
    centers = np.array([[-3.], [0.], [3.]])
    pairs = rex._pairs(centers, np.eye(1), 1., (2*np.pi,))
    assert [0, 2] in pairs.tolist() and len(pairs) == 2
    np.testing.assert_array_equal(pairs, rex._pairs(centers, np.eye(1), 1., (2*np.pi,)))


@pytest.mark.parametrize("pairs", [[[0, 0]], [[0, 1], [1, 0]], [[0, 1]],
                                    [[0, 5]], [[0., 1.], [1., 2.]], [], [0, 1]])
def test_invalid_graphs(pairs):
    with pytest.raises(ValueError, match="pairs"):
        rex._pairs(np.array([[-1.], [0.], [1.]]), np.eye(1), 1., (None,), pairs)


@pytest.mark.parametrize("thermostat,scheme", [("andersen", "cayley-bcb"),
    ("andersen", "normal-mode-bab"), ("andersen", "velocity-verlet"),
    ("pile-l", "cayley-bcb"), ("pile-l", "normal-mode-bab")])
def test_2d_ladders_resume_and_offline_analysis(tmp_path, monkeypatch, thermostat, scheme):
    cfg = config(thermostat, scheme)
    result = run_fes(cfg, system(), tmp_path, return_data=True)
    assert result["summary"]["samples"] == 160
    assert result["summary"]["converged"] is False
    assert result["data"]["free_energy"].shape == (4, 5)
    assert result["data"]["bootstrap_block_plan"]["resampling"] == "synchronous_within_group"
    for replica in range(2):
        data = rex._read(cfg, system(), rex.exchange_options(cfg, system()), tmp_path, replica, complete=True)
        assert np.all(data["attempts"] == 48 // 9)
        assert np.all(data["accepts"] <= data["attempts"])
        assert data["samples"].shape == (4, 20, 2)
        np.testing.assert_array_equal(data["walker_history"][(data["sample_steps"] - 1) // 9].T,
                                      data["walker_ids"])
    before = {p.name: p.read_bytes() for p in (tmp_path / "checkpoints").glob("*.npz")}
    def no_pes(*args, **kwargs):
        raise AssertionError("completed resume or offline analysis constructed a PES")
    monkeypatch.setattr(rex, "make_task_potential", no_pes)
    again = run_fes(cfg, system(), tmp_path, return_data=True)
    offline = reanalyze_fes(cfg, system(), tmp_path, return_data=True)
    for key in ("free_energy", "weights", "standard_error", "mask"):
        np.testing.assert_array_equal(result["data"][key], again["data"][key])
        np.testing.assert_array_equal(result["data"][key], offline["data"][key])
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "checkpoints").glob("*.npz")}
    import h5py
    with h5py.File(tmp_path / "fes.h5") as h:
        for group in h["streams"].values():
            assert "exchange_group" in group.attrs and group["walker_ids"].shape == (20,)


@pytest.mark.parametrize("where", ["propagation", "exchange"])
@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
def test_partial_event_rolls_back_entire_ensemble(tmp_path, monkeypatch, where, thermostat):
    cfg = config(thermostat)
    run_fes(cfg, system(), tmp_path / "full")
    seen = 0
    name = "_advance_window" if where == "propagation" else "_exchange_sweep"
    original = getattr(rex, name)
    def failing(*args, **kwargs):
        nonlocal seen
        result = original(*args, **kwargs)
        seen += 1
        if seen == (10 if where == "propagation" else 2):
            raise RuntimeError("injected half-ensemble failure")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(rex, name, failing)
        with pytest.raises(RuntimeError, match="half-ensemble"):
            run_fes(cfg, system(), tmp_path / "resumed")
    assert int(read(tmp_path / "resumed/checkpoints/exchange_0000.npz")["elapsed"]) == 11
    with pytest.raises(RuntimeError, match="complete ladder"):
        reanalyze_fes(cfg, system(), tmp_path / "resumed")
    run_fes(cfg, system(), tmp_path / "resumed")
    assert_same_checkpoints(tmp_path / "full", tmp_path / "resumed")


@pytest.mark.parametrize("published", [False, True])
def test_atomic_write_failure_keeps_old_or_new_complete_ensemble(tmp_path, monkeypatch, published):
    cfg = config()
    run_fes(cfg, system(), tmp_path / "full")
    original = rex.atomic_savez
    seen = 0
    def failing(*args, **kwargs):
        nonlocal seen
        seen += 1
        if seen == 2 and not published:
            raise OSError("write interrupted")
        result = original(*args, **kwargs)
        if seen == 2:
            raise OSError("write interrupted")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(rex, "atomic_savez", failing)
        with pytest.raises(OSError, match="write interrupted"):
            run_fes(cfg, system(), tmp_path / "resumed")
    assert int(read(tmp_path / "resumed/checkpoints/exchange_0000.npz")["elapsed"]) == (11 if published else 0)
    run_fes(cfg, system(), tmp_path / "resumed")
    assert_same_checkpoints(tmp_path / "full", tmp_path / "resumed")


def test_serial_and_two_process_ladders_match(tmp_path):
    cfg = config("pile-l")
    run_fes(cfg, system(), tmp_path / "serial")
    cfg["parallel"] = {"backend": "process", "trajectory_workers": 2}
    run_fes(cfg, system(), tmp_path / "parallel")
    assert_same_checkpoints(tmp_path / "serial", tmp_path / "parallel")


def test_mapping_configs_keep_existing_checkpoint_identity():
    cfg, sys = config(), system()
    canonical = checkpoint_cv_config(cfg)
    assert canonical is not cfg and canonical["cvs"] is cfg["cvs"]
    options = rex.exchange_options(cfg, sys)
    expected = _fingerprint(cfg, sys, {"kind": rex.CHECKPOINT_KIND, "replica": 0,
        "centers": options["centers"], "stiffness": options["stiffness"], "pairs": options["pairs"]})
    assert rex._identity(cfg, sys, options, 0) == expected


def test_cv_object_identity_is_stable_across_copies_and_tracks_parameters():
    cfg, sys = config(), system()
    cfg["cvs"] = [Cartesian(0, "x"), Cartesian(0, "y")]
    options = rex.exchange_options(cfg, sys)
    expected = rex._identity(cfg, sys, options, 0)
    for _ in range(3):
        assert rex._identity(cfg, sys, options, 0) == expected
        copied = copy.deepcopy(cfg)
        assert rex._identity(copied, sys, rex.exchange_options(copied, sys), 0) == expected
    cfg["cvs"][0].axis = 2
    assert rex._identity(cfg, sys, rex.exchange_options(cfg, sys), 0) != expected


def test_cv_object_ladders_resume_after_copy_and_process_execution(tmp_path):
    cfg, sys = config(), system()
    cfg["cvs"] = [Cartesian(0, "x"), Cartesian(0, "y")]
    run_fes(cfg, sys, tmp_path / "serial")
    before = {p.name: p.read_bytes() for p in (tmp_path / "serial/checkpoints").glob("*.npz")}
    run_fes(copy.deepcopy(cfg), sys, tmp_path / "serial")
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "serial/checkpoints").glob("*.npz")}
    parallel = copy.deepcopy(cfg)
    parallel["parallel"] = {"backend": "process", "trajectory_workers": 2}
    run_fes(parallel, sys, tmp_path / "process")
    assert_same_checkpoints(tmp_path / "serial", tmp_path / "process")
    changed = copy.deepcopy(cfg)
    changed["cvs"][0].axis = 2
    with pytest.raises((ValueError, RuntimeError), match="checkpoint physics differs"):
        run_fes(changed, sys, tmp_path / "serial")


@pytest.mark.parametrize("kind", ["instance_override", "class_override", "custom", "periodic"])
def test_unsafe_cv_object_identity_fails_before_output(tmp_path, monkeypatch, kind):
    cfg, sys = config(), system()
    first = Cartesian(0, "x")
    if kind == "instance_override":
        first.value = lambda q: np.array([0.0])
    elif kind == "class_override":
        original = Cartesian.centroid_value_gradient
        monkeypatch.setattr(Cartesian, "centroid_value_gradient", lambda self, q: original(self, q))
    elif kind == "custom":
        class CustomCartesian(Cartesian):
            pass
        first = CustomCartesian(0, "x")
    else:
        first = PeriodicDistance((0, 1), np.eye(3), [True, True, True])
    cfg["cvs"] = [first, Cartesian(0, "y")]
    with pytest.raises(ValueError, match="serializable CV factory configuration"):
        run_fes(cfg, sys, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_nested_cv_object_identities_and_angle_overrides(monkeypatch):
    stack = CVStack([Cartesian(0, "x"), Cartesian(0, "y")])
    cfg = {"cvs": [stack]}
    original = checkpoint_cv_config(cfg)
    assert checkpoint_cv_config(copy.deepcopy(cfg)) == original
    stack.cvs[1].axis = 2
    assert checkpoint_cv_config(cfg) != original
    stack.cvs[0].value = lambda q: np.array([0.0])
    with pytest.raises(ValueError, match="factory configuration"):
        checkpoint_cv_config(cfg)

    coordinate = RPMDrateCoordinate(
        [(0,), (1, 2)], [1, 1, 1], [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
        [(0, 1)], [(1, 2)], 5,
    )
    surface = coordinate.dividing_surface(0.5)
    assert checkpoint_cv_config({"cvs": [surface]}) == checkpoint_cv_config({"cvs": [copy.deepcopy(surface)]})
    coordinate.com.value = lambda q: np.array([0.0])
    with pytest.raises(ValueError, match="factory configuration"):
        checkpoint_cv_config({"cvs": [surface]})

    angle = Angle((0, 1, 2))
    assert checkpoint_cv_config({"cvs": [angle]}) == checkpoint_cv_config({"cvs": [copy.deepcopy(angle)]})
    original = Angle.centroid_value_gradient
    monkeypatch.setattr(Angle, "centroid_value_gradient", lambda self, q: original(self, q))
    with pytest.raises(ValueError, match="factory configuration"):
        checkpoint_cv_config({"cvs": [angle]})


@pytest.mark.parametrize("field", ["q", "sample_steps", "walker_ids", "attempts", "times", "payload_version", "rngs"])
def test_semantically_invalid_checkpoints_rejected_with_valid_hash(tmp_path, field):
    cfg = config()
    run_fes(cfg, system(), tmp_path)
    path = tmp_path / "checkpoints/exchange_0000.npz"
    data = read(path)
    if field == "q":
        data[field] = data[field][:-1]
    elif field == "rngs":
        data[field] = np.asarray("[]")
    else:
        data[field] = data[field] + 1
    atomic_savez(path, kind=rex.CHECKPOINT_KIND, **data)
    with pytest.raises((ValueError, RuntimeError)):
        run_fes(cfg, system(), tmp_path)


@pytest.mark.parametrize("change", [
    {"replica_exchange": {"enabled": 1}}, {"replica_exchange": {"interval_steps": 0}},
    {"replica_exchange": {"unknown": 1}}, {"analysis": {"measure": "rate_flux"}},
    {"adaptive_umbrella": {"enabled": True}}, {"task": "rate"},
    {"umbrella": {"centers": [[0., 0.]]}}, {"sampling": {"production_steps": 1}},
])
def test_invalid_options_rejected_before_output(tmp_path, change):
    cfg = config()
    for key, value in change.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    with pytest.raises(ValueError):
        run_fes(cfg, system(), tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_cli_resolves_exchange_and_rejects_unknown_fields():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "examples/fes/harmonic2d.inp").read_text())
    data["replica_exchange"] = {"enabled": True, "interval_steps": 20}
    resolved, sys = JobSpec.from_mapping(data).resolve()
    assert rex.exchange_options(resolved, sys)["enabled"]
    data["replica_exchange"]["shared_walkers"] = True
    with pytest.raises(InputError, match="Unknown replica_exchange"):
        JobSpec.from_mapping(data).resolve()


@pytest.mark.parametrize("thermostat", ["andersen", "pile-l"])
def test_fixed_periodic_slab_survives_exchanges(tmp_path, thermostat):
    sys = System(("H", "Cu"), np.array([2., 20.]),
                 np.array([[.3, -.4, 1.2], [.3, -.4, .2]]),
                 cell=np.diag([4., 5., 12.]), pbc=[True, True, False], fixed_atoms=[1])
    cfg = config(thermostat)
    root = Path(__file__).resolve().parents[1]
    cfg["potential"] = {"provider": "python", "factory": str(root / "examples/surface/toy_surface.py") + ":build",
                        "params": {"barrier": .2, "height": 1., "vertical_k": 1.}}
    cfg["cvs"] = [{"kind": "surface_lateral", "atoms": [0], "reference_atoms": [1],
                   "direction": [1., 0., 0.]},
                  {"kind": "surface_height", "atoms": [0], "reference_atoms": [1],
                   "normal": [0., 0., 1.]}]
    cfg["umbrella"]["centers"] = [[-.2, .8], [-.2, 1.2], [.2, .8], [.2, 1.2]]
    result = run_fes(cfg, sys, tmp_path, return_data=True)
    for stream in result["streams"]:
        np.testing.assert_array_equal(stream["last_q"][:, 1], np.broadcast_to(sys.geometry[1], (4, 3)))
        np.testing.assert_array_equal(stream["last_p"][:, 1], 0.)
    reanalyze_fes(cfg, sys, tmp_path)
