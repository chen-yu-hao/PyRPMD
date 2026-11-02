"""Experimental MTS/RPC must match independent physical phase-space maps."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from benchmarks.water.native_optimizations import (
    DelayedSlowKickMTS,NativeSplitRPMD,openmm_frequency_beta,
)
from benchmarks.water.native_reference import sample_momenta
from benchmarks.water.qtip4pf import initial_grid
from benchmarks.water.split_reference import ContractedSlowPotential,GroupEvaluator,SplitMTS
from pyrpmd.dynamics import State


def test_delayed_slow_kick_matches_independent_rrespa_and_reverses():
    pytest.importorskip("openmm")
    q,box=initial_grid(cells_per_axis=2,density_g_cm3=.997,seed=32)
    q[3:6,2]-=.6  # A nonzero-LJ intermolecular pair is part of the map test.
    q=q[None]
    native=DelayedSlowKickMTS(molecules=8,box_angstrom=box,cutoff_angstrom=3.,
                              outer_dt_fs=.5,inner_dt_fs=.1)
    grouped=GroupEvaluator(native.system)
    class Slow:
        def evaluate(self,q):
            energy,force=grouped.evaluate(q,groups=2)
            return float(energy.sum()),force
    independent=SplitMTS(Slow(),native.masses,298,1,.5,.1)
    p=sample_momenta(np.random.default_rng(901),native.masses,298,remove_com=True)
    state=State(q.copy(),p.copy())
    try:
        native.set_state(q,p)
        for _ in range(10):
            native.integrator.step(1)
            independent.step(state)
        nq,np_,energy=native.get_state(energy=True)
        np.testing.assert_allclose(nq,state.q,atol=2e-10,rtol=0)
        np.testing.assert_allclose(np_,state.p,atol=2e-8,rtol=0)
        assert energy==pytest.approx(independent.energy(State(nq.copy(),np_.copy())),abs=2e-12)
        native.set_state(nq,-np_)
        native.integrator.step(10)
        rq,rp,_=native.get_state()
        np.testing.assert_allclose(rq,q,atol=1e-12,rtol=0)
        np.testing.assert_allclose(rp,-p,atol=1e-9,rtol=0)
    finally:
        native.close();grouped.close()


@pytest.mark.parametrize("copies",[1,3,8])
def test_native_rpc_matches_independent_contracted_hamiltonian_bab(copies):
    pytest.importorskip("openmm")
    beads=8
    grid,box=initial_grid(cells_per_axis=2,density_g_cm3=.997,seed=32)
    grid[3:6,2]-=.6
    native=NativeSplitRPMD(molecules=8,box_angstrom=box,beads=beads,long_copies=copies,
                          dt_fs=.1,lj_cutoff_angstrom=3.,electrostatic_cutoff_angstrom=2.)
    grouped=GroupEvaluator(native.system)
    independent=SplitMTS(ContractedSlowPotential(grouped,beads,copies),native.masses,298,beads,.1,.1)
    # Isolate the known OpenMM hbar convention without changing the q/p sample.
    # Rebuild Dynamics so its normal-mode and native caches use that beta too.
    from pyrpmd.dynamics import Dynamics
    from pyrpmd.constants import FS_TO_AU
    independent.fast=Dynamics(independent.fast.potential,native.masses,openmm_frequency_beta(298),
                              beads,.1*FS_TO_AU,native=True,threads=1,integrator="normal-mode-bab")
    rng=np.random.default_rng(2027)
    q=np.repeat(grid[None],beads,axis=0)+rng.normal(0,.025,(beads,24,3))
    p=np.stack([sample_momenta(rng,native.masses,298*beads,remove_com=False)[0] for _ in range(beads)])
    state=State(q.copy(),p.copy())
    try:
        native.set_state(q,p)
        for _ in range(3):
            native.integrator.step(1)
            independent.step(state)
        nq,np_=native.get_state()
        np.testing.assert_allclose(nq,state.q,atol=2e-9,rtol=0)
        np.testing.assert_allclose(np_,state.p,atol=5e-8,rtol=0)
    finally:
        native.close();grouped.close()


@pytest.mark.parametrize("sampling",[False,True])
def test_explicit_ring_copies_plus_context_checkpoint_restore_fresh_context(sampling):
    """Do not assume Context alone serializes a plugin's all-copy phase space."""
    pytest.importorskip("openmm")
    from benchmarks.water.validate_native_optimizations import raw_ring_state,restore_raw_ring
    grid,box=initial_grid(cells_per_axis=2,density_g_cm3=.997,seed=42)
    kwargs=dict(molecules=8,box_angstrom=box,beads=8,long_copies=3,dt_fs=.1,
                lj_cutoff_angstrom=2.5,electrostatic_cutoff_angstrom=2.,sampling=sampling)
    first=NativeSplitRPMD(**kwargs)
    rng=np.random.default_rng(82)
    q=np.repeat(grid[None],8,axis=0)+rng.normal(0,.02,(8,24,3))
    p=np.stack([sample_momenta(rng,first.masses,298*8,remove_com=False)[0] for _ in range(8)])
    first.set_state(q,p);first.integrator.step(3)
    raw_q,raw_v=raw_ring_state(first)
    blob=first.context.createCheckpoint()
    first.integrator.step(3)
    expected=first.get_state()
    first.close()
    restored=NativeSplitRPMD(**kwargs)
    try:
        restored.context.loadCheckpoint(blob)
        restore_raw_ring(restored,raw_q,raw_v)
        for actual,saved in zip(raw_ring_state(restored),(raw_q,raw_v),strict=True):
            np.testing.assert_array_equal(actual,saved)
        restored.integrator.step(3)
        for actual,saved in zip(restored.get_state(),expected,strict=True):
            np.testing.assert_allclose(actual,saved,atol=1e-12,rtol=0)
    finally:restored.close()
