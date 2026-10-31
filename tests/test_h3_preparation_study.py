"""Preparation comparisons preserve launch invariants and energy conventions."""
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
