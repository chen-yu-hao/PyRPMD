"""Preparation comparisons preserve launch invariants and energy conventions."""
from concurrent.futures import Future
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.h3 import state_preparation_study as study
from pyrpmd.initial_states import prepare_initial_ensemble


class HarmonicDiatom:
    def radial(self, distances):
        displacement = distances[0]-study.driver.BOND
        spring = study.driver.MASS/2*study.driver.OMEGA**2
        return .5*spring*displacement**2, np.array([spring*displacement, 0., 0.])


def prepared(config):
    mass, bond = study.driver.MASS, study.driver.BOND
    system = {"symbols":["H"]*3,"masses":[mass]*3,"fragments":[[0],[1,2]],
              "geometry":[[0,0,-12],[0,0,-bond/2],[0,0,bond/2]]}
    section = {"method":config.method,"fragments":[{}, {
        "j":0,"frequencies":[study.driver.OMEGA],"vibrational_quanta":[config.v],
        "modes":[[0,0,-2**-.5,0,0,2**-.5]]}]}
    if config.method=="ne_rpmd":
        section["beta"]=config.beta
    return prepare_initial_ensemble(section,system,config.beads,config.beta)


def install(monkeypatch, config, reference, spec):
    monkeypatch.setattr(study,"_SPEC",spec)
    monkeypatch.setattr(study.driver,"_WORKER",(config,reference,SimpleNamespace(potential=HarmonicDiatom()),None))


@pytest.mark.parametrize("v",[0,1])
def test_full_action_time_phase_pairing_recovers_original_harmonic_initial_state(monkeypatch,v):
    config=study.LaunchConfig(method="qct",beads=1,v=v,total_energy_ev=1.1)
    reference=prepared(config)
    install(monkeypatch,config,reference,study.PreparationSpec(kind="action_qct",conditioning="none"))
    energy=(v+.5)*study.driver.OMEGA
    mu=study.driver.MASS/2
    amplitude=np.sqrt(2*energy/(mu*study.driver.OMEGA**2))

    class ExactHarmonicOrbit:
        energy_hartree=energy

        def sample_phase(self,u):
            return (study.driver.BOND-amplitude*np.cos(2*np.pi*u),
                    mu*study.driver.OMEGA*amplitude*np.sin(2*np.pi*u))

    monkeypatch.setattr(study,"_ORBIT",ExactHarmonicOrbit())
    for identity in range(30):
        expected,launch=study.ORIGINAL_PREPARE(config,reference,identity)
        actual,meta=study.prepare_variant(config,reference,identity)
        np.testing.assert_allclose(actual.q,expected.q,atol=8e-15,rtol=0)
        np.testing.assert_allclose(actual.p,expected.p,atol=2e-13,rtol=0)
        assert meta["b_bohr"]==launch["b_bohr"]
        assert meta["preparation_study"]["actual_asymptotic_total_energy_ev"]==pytest.approx(1.1)


def test_internal_switch_preserves_atom_ring_fragment_com_and_collision_energy(monkeypatch):
    config=study.LaunchConfig(v=1,total_energy_ev=1.1)
    reference=prepared(config)
    install(monkeypatch,config,reference,study.PreparationSpec(kind="switch",switch_time_fs=50.))

    class MockSwitch:
        def run(self,r,p,**kwargs):
            return SimpleNamespace(r=r+.07,p=p*.97,summary={"test":True},trace=[])

    monkeypatch.setattr(study,"_SWITCHER",MockSwitch())
    baseline,_=study.ORIGINAL_PREPARE(config,reference,17)
    actual,meta=study.prepare_variant(config,reference,17)
    np.testing.assert_array_equal(actual.q[:,0],baseline.q[:,0])
    np.testing.assert_array_equal(actual.p[:,0],baseline.p[:,0])
    np.testing.assert_allclose(actual.q[:,1:].sum(axis=1),baseline.q[:,1:].sum(axis=1),atol=2e-14)
    np.testing.assert_allclose(actual.p[:,1:].sum(axis=1),baseline.p[:,1:].sum(axis=1),atol=2e-14)
    radial,_,_=study._radial_components(actual)
    original,_,_=study._radial_components(baseline)
    np.testing.assert_allclose(radial-original,.07,atol=2e-14)
    relative=(actual.p[:,0].mean(axis=0)-actual.p[:,1:].mean(axis=(0,1)))/study.driver.MASS
    assert study.driver.MASS/3*(relative@relative)==pytest.approx(config.collision_energy,rel=1e-12)
    assert meta["preparation_study"]["energy_convention"]=="fixed_collision"


def test_no_filter_and_energy_override_are_explicit(monkeypatch):
    config=study.LaunchConfig(v=1,total_energy_ev=1.1)
    reference=prepared(config)
    install(monkeypatch,config,reference,study.PreparationSpec(conditioning="none"))
    rejected=next(i for i in range(100) if study.ORIGINAL_PREPARE(config,reference,i)[1]["preparation_draws"]>1)
    _,meta=study.prepare_variant(config,reference,rejected)
    assert meta["preparation_draws"]==1
    assert meta["initial_bond_bohr"]<config.dmin_bohr
    with pytest.raises(ValueError,match="no retuning"):
        study.PreparationSpec(energy_convention="fixed_total").validate(config)
    with pytest.raises(ValueError,match="positive"):
        _=study.LaunchConfig(collision_energy_override_au=-1.).collision_energy


def test_failed_preparations_remain_in_audit_attempt_count():
    summary=study.summarize_audit([{"status":"failed","id":5,"reason":"outside PES domain"}])
    assert summary["attempts"]==summary["failed"]==1
    assert summary["prepared"]==0
    assert summary["quantities"]["paired_delta_primitive_full_ev"]=={"n":0,"mean":None,"sem":None}


@pytest.mark.parametrize("changes", [
    {"samples": 0}, {"samples": 1.5}, {"samples": True},
    {"beads": 32.5}, {"beads": True}, {"beads": np.int64(32)},
    {"seed": -1}, {"seed": 1.5}, {"seed": False},
    {"id_offset": -1}, {"id_offset": .5}, {"id_offset": True},
    {"v": True}, {"v": 0.0}, {"v": 2}, {"method": "thermal"},
    {"total_energy_ev": np.nan}, {"total_energy_ev": np.inf},
    {"total_energy_ev": 0}, {"total_energy_ev": "0.9"},
    {"total_energy_ev": .9+0j}, {"total_energy_ev": 10**400},
    {"total_energy_ev": 1e308},
    {"bmin": -.1}, {"bmax": 0.}, {"bmax": np.inf},
    {"bmin": 2., "bmax": 1.}, {"bmax": 1e200},
    {"dt_fs": 0.}, {"dt_fs": True},
    {"time_cap_fs": np.inf, "extended_cap_fs": np.inf},
    {"time_cap_fs": np.nan}, {"time_cap_fs": .03},
    {"extended_cap_fs": 1000.01}, {"extended_cap_fs": 1.},
    {"initial_axial_bohr": 0.}, {"initial_axial_bohr": np.nan},
    {"exit_bohr": np.inf}, {"exit_bohr": 2.5},
    {"bond_cutoff_bohr": 0.}, {"bond_cutoff_bohr": np.nan},
    {"dmin_bohr": -.1}, {"dmin_bohr": np.inf},
    {"persistence_fs": -.1}, {"persistence_fs": .03},
    {"check_fs": 0.}, {"check_fs": .01}, {"check_fs": .03},
    {"beta_override": np.nan}, {"beta_override": 0.},
    {"beta_override": True}, {"beta_override": 1e-300}, {"beta_override": 1e308},
    {"collision_energy_override_au": np.inf},
    {"collision_energy_override_au": -1.},
    {"collision_energy_override_au": True},
    {"integrator": "unknown"}, {"integrator": None},
    {"integrator": "velocity-verlet", "beta_override": 1.},
])
def test_invalid_launch_request_is_rejected_before_pes_or_output(monkeypatch, tmp_path, changes):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid input reached resource initialization")

    monkeypatch.setattr(study.driver, "BKMP2Potential", unexpected)
    monkeypatch.setattr(study, "ProcessPoolExecutor", unexpected)
    monkeypatch.setattr(study, "_source_snapshot", unexpected)
    output = tmp_path/"case"
    with pytest.raises(ValueError):
        study.run(replace(study.LaunchConfig(), **changes), study.PreparationSpec(), output)
    assert not output.exists()


@pytest.mark.parametrize("workers,audit", [(0, False), (1.5, False), (True, False),
                                           (1, 1), (1, None)])
def test_invalid_execution_options_are_rejected_before_pes(monkeypatch, tmp_path, workers, audit):
    def unexpected():
        raise AssertionError("Invalid execution options reached PES initialization")

    monkeypatch.setattr(study.driver, "BKMP2Potential", unexpected)
    with pytest.raises(ValueError):
        study.run(study.LaunchConfig(), study.PreparationSpec(), tmp_path/"case",
                  workers=workers, audit=audit)
    assert not (tmp_path/"case").exists()


@pytest.mark.parametrize("changes", [
    {"switch_time_fs": np.nan}, {"hold_time_fs": True},
    {"preparation_dt_fs": "0.02"}, {"preparation_dt_fs": 0.},
    {"trace_stride_fs": np.inf}, {"trace_stride_fs": .01},
    {"trace_stride_fs": .03}, {"hold_time_fs": .03},
    {"hold_time_fs": 1e-14},
    {"kind": "switch", "switch_time_fs": .03},
    {"kind": "switch", "switch_time_fs": 1e20},
    {"hold_time_fs": 2_000_000.},
])
def test_invalid_preparation_grid_is_rejected_before_pes(monkeypatch, tmp_path, changes):
    def unexpected():
        raise AssertionError("Invalid preparation grid reached PES initialization")

    monkeypatch.setattr(study.driver, "BKMP2Potential", unexpected)
    with pytest.raises(ValueError):
        study.run(study.LaunchConfig(), replace(study.PreparationSpec(), **changes),
                  tmp_path/"case")
    assert not (tmp_path/"case").exists()


def test_valid_time_grids_and_registered_workloads_do_not_require_resources():
    from benchmarks.h3.run_preparation_study import jobs

    # Ratios such as .3/.025 include ordinary floating-point roundoff.
    config = study.LaunchConfig(beads=3, dt_fs=.025, persistence_fs=0.)
    spec = study.PreparationSpec(kind="switch", switch_time_fs=.1, hold_time_fs=.3,
                                 preparation_dt_fs=.025, trace_stride_fs=.2)
    study.validate_request(config, spec, workers=1)
    with pytest.raises(ValueError, match="1000000 beads"):
        study.validate_request(replace(config, beads=1000001), spec)
    cases = list(jobs("all"))
    assert len(cases) == 12
    assert sum(case[2].samples for case in cases) == 30720
    for _, _, config, spec, audit in cases:
        study.validate_request(config, spec, audit=audit)
    with pytest.raises(ValueError, match="workload group"):
        list(jobs("typo"))


@pytest.mark.parametrize("kind,method,beads", [("direct", "ne_rpmd", 32),
                                               ("action_qct", "qct", 1)])
def test_fixed_collision_cannot_relabel_an_energy_override(monkeypatch, tmp_path, kind, method, beads):
    def unexpected():
        raise AssertionError("A mislabeled energy convention reached PES initialization")

    monkeypatch.setattr(study.driver, "BKMP2Potential", unexpected)
    config = study.LaunchConfig(method=method, beads=beads)
    spec = study.PreparationSpec(kind=kind, conditioning="none")
    for collision in (config.collision_energy*1.01, config.collision_energy/2):
        with pytest.raises(ValueError, match="fixed_collision.*cannot retune"):
            study.run(replace(config, collision_energy_override_au=collision), spec, tmp_path/"case")
        assert not (tmp_path/"case").exists()
    for collision in (config.collision_energy, np.nextafter(config.collision_energy, np.inf)):
        study.validate_request(replace(config, collision_energy_override_au=collision), spec)


@pytest.mark.parametrize("method,beads", [("qct", 1), ("ne_rpmd", 32)])
@pytest.mark.parametrize("v,energy", [(0, .55), (1, .90), (1, 1.10)])
def test_published_points_retain_original_beta_rng_and_launch(method, beads, v, energy):
    config = study.LaunchConfig(method=method, beads=beads, v=v, total_energy_ev=energy)
    old_values = asdict(config)
    del old_values["collision_energy_override_au"]
    original = study.driver.Config(**old_values)
    assert config.beta == original.beta
    assert config.beta_override is None
    assert config.auxiliary_beta["convention"] == "paper_2021"
    previous, old_launch = study.ORIGINAL_PREPARE(original, prepared(original), 10017)
    actual, new_launch = study.ORIGINAL_PREPARE(config, prepared(config), 10017)
    np.testing.assert_array_equal(actual.q, previous.q)
    np.testing.assert_array_equal(actual.p, previous.p)
    assert new_launch == old_launch


def test_qct_placeholder_requires_positive_actual_collision_energy():
    config = study.LaunchConfig(method="qct", beads=1, v=1, total_energy_ev=.8)
    with pytest.raises(ValueError, match="actual collision energy"):
        _ = config.beta
    resolved = replace(config, collision_energy_override_au=.01*study.EV_TO_AU)
    assert resolved.beta == 1500.
    assert resolved.auxiliary_beta["convention"] == "qct_single_bead_placeholder_1500_au"
    assert "not a physical temperature" in resolved.auxiliary_beta["physical_role"]
    explicit = replace(resolved, beta_override=731.)
    assert explicit.beta == 731.
    assert explicit.auxiliary_beta["convention"] == "explicit_override"
    with pytest.raises(ValueError, match="harmonic internal energy"):
        _ = replace(resolved, method="ne_rpmd", beads=32).beta


@pytest.mark.parametrize("energy,allowed", [(.8, True), (.78, False)])
def test_fixed_total_resolves_actual_energy_before_worker_initialization(
        monkeypatch, tmp_path, energy, allowed):
    from benchmarks.h3.diatomic_action import DiatomicAction

    orbit_energy = .7851776048192163*study.EV_TO_AU
    orbit = SimpleNamespace(energy_hartree=orbit_energy, metadata={"test_orbit": True})
    monkeypatch.setattr(study.driver, "BKMP2Potential", lambda: object())
    monkeypatch.setattr(DiatomicAction, "from_bkmp2",
                        lambda *args, **kwargs: SimpleNamespace(quantize=lambda v: orbit))
    monkeypatch.setattr(study, "_source_snapshot", lambda: {"files": []})
    monkeypatch.setattr(study, "_verify_snapshot", lambda files: None)
    initialized = []

    class Executor:
        def __init__(self, **kwargs):
            initialized.append(kwargs["initargs"][0])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, identity):
            future = Future()
            future.set_result({"id": identity, "status": "failed", "reason": "synthetic worker"})
            return future

    monkeypatch.setattr(study, "ProcessPoolExecutor", Executor)
    config = study.LaunchConfig(method="qct", beads=1, v=1, total_energy_ev=energy, samples=1)
    spec = study.PreparationSpec(kind="action_qct", conditioning="none", energy_convention="fixed_total")
    output = tmp_path/"case"
    if not allowed:
        with pytest.raises(ValueError, match="actual collision energy"):
            study.run(config, spec, output, workers=1, audit=True)
        assert not output.exists()
        assert initialized == []
        return
    summary = study.run(config, spec, output, workers=1, audit=True)
    assert len(initialized) == 1
    launched = initialized[0]
    assert launched.total_energy_ev == energy
    assert launched.beta_override is None
    assert launched.collision_energy == pytest.approx(energy*study.EV_TO_AU-orbit_energy)
    assert launched.beta == 1500.
    assert summary["auxiliary_beta"] == launched.auxiliary_beta
    provenance = json.loads((output/"provenance.json").read_text())
    assert provenance["auxiliary_beta"] == launched.auxiliary_beta
    assert json.loads((output/"config.json").read_text()) == asdict(launched)


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/"benchmarks/h3/build/libbkmp2.so").is_file(),
                    reason="Build verified BKMP2 for the full-PES near-threshold launch check")
def test_real_bkmp2_near_threshold_action_launch_is_independent_of_auxiliary_beta(monkeypatch):
    from benchmarks.h3.diatomic_action import DiatomicAction

    potential = study.driver.BKMP2Potential()
    model = DiatomicAction.from_bkmp2(potential, reduced_mass=study.driver.MASS/2)
    orbit = model.quantize(1)
    config = study.LaunchConfig(method="qct", beads=1, v=1, total_energy_ev=.8,
        collision_energy_override_au=.8*study.EV_TO_AU-orbit.energy_hartree)
    assert 0 < config.collision_energy < .02*study.EV_TO_AU
    spec = study.PreparationSpec(kind="action_qct", conditioning="none", energy_convention="fixed_total")
    monkeypatch.setattr(study, "_SPEC", spec)
    monkeypatch.setattr(study, "_ORBIT", orbit)
    states = []
    launches = []
    for current in (config, replace(config, beta_override=317.)):
        reference = prepared(current)
        monkeypatch.setattr(study.driver, "_WORKER",
            (current, reference, SimpleNamespace(potential=potential), None))
        state, launch = study.prepare_variant(current, reference, 20017)
        states.append(state)
        launches.append(launch)
        radial, momentum, _ = study._radial_components(state)
        internal = momentum[0]**2/study.driver.MASS + model.values(radial[0])
        relative = (state.p[:, 0].mean(axis=0)-state.p[:, 1:].mean(axis=(0, 1)))/study.driver.MASS
        collision = study.driver.MASS/3*(relative@relative)
        assert internal == pytest.approx(orbit.energy_hartree, abs=2e-14)
        assert collision == pytest.approx(config.collision_energy, abs=2e-14)
        assert (collision+internal)/study.EV_TO_AU == pytest.approx(.8, abs=2e-12)
        assert launch["preparation_study"]["auxiliary_beta"] == current.auxiliary_beta
    np.testing.assert_array_equal(states[0].q, states[1].q)
    np.testing.assert_array_equal(states[0].p, states[1].p)
    assert launches[0]["b_bohr"] == launches[1]["b_bohr"]
    assert (launches[0]["preparation_study"]["orbit_phase"]
            == launches[1]["preparation_study"]["orbit_phase"])
