"""Independent analytic forces, split identities, RPC gradients, and MTS maps."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.water.qtip4pf import cluster_energy,initial_grid,make_system,H_MASS_DA,O_MASS_DA
from benchmarks.water.split_reference import (
    ContractedSlowPotential,GroupEvaluator,SplitMTS,contraction_matrix,intramolecular,split_system,
)
from pyrpmd.constants import AMU_TO_AU,FS_TO_AU,KB_AU
from pyrpmd.dynamics import Dynamics,State
from pyrpmd.potentials.base import PESResult


def geometries(beads=1):
    q,box=initial_grid(cells_per_axis=2,density_g_cm3=.997,seed=93)
    rng=np.random.default_rng(2026)
    return np.repeat(q[None],beads,axis=0)+rng.normal(0,.015,(beads,24,3)),box


def evaluator(*,move=True):
    q,box=geometries()
    system,metadata=split_system(molecules=8,box_angstrom=box,lj_cutoff_angstrom=2.5,
                                electrostatic_cutoff_angstrom=2,ewald_tolerance=1e-5,move_exclusions=move)
    return GroupEvaluator(system),metadata


def test_analytic_intramolecular_matches_independent_numpy_energy_and_finite_difference():
    q,_=geometries(3);q=q[:,:3]
    energy,force=intramolecular(q)
    np.testing.assert_allclose(energy,cluster_energy(q),atol=2e-15,rtol=0)
    for atom in range(3):
        for component in range(3):
            plus,minus=q.copy(),q.copy();plus[:,atom,component]+=1e-5;minus[:,atom,component]-=1e-5
            expected=-(cluster_energy(plus)-cluster_energy(minus))/(2e-5)
            np.testing.assert_allclose(force[:,atom,component],expected,atol=2e-10,rtol=0)
    np.testing.assert_allclose(force.sum(axis=1),0,atol=1e-16)


def test_moving_smooth_intra_exclusions_preserves_full_hamiltonian():
    pytest.importorskip("openmm")
    q,box=geometries(2)
    old,old_metadata=evaluator(move=False);new,new_metadata=evaluator(move=True)
    baseline=GroupEvaluator(make_system(molecules=8,box_angstrom=box,cutoff_angstrom=2.5,ewald_tolerance=1e-5))
    try:
        a,b=old.evaluate(q),new.evaluate(q)
        np.testing.assert_allclose(a[0],b[0],atol=3e-12,rtol=0)
        np.testing.assert_allclose(a[1],b[1],atol=3e-12,rtol=0)
        exact=baseline.evaluate(q)
        # Different Ewald real-space cutoffs choose different meshes; both
        # approximate the same Ewald potential at the declared tolerance.
        assert np.max(abs(b[1]-exact[1]))<1e-5
        intra=new.evaluate(q,groups=1)
        independent=intramolecular(q)
        np.testing.assert_allclose(intra[0],independent[0],atol=2e-14,rtol=0)
        np.testing.assert_allclose(intra[1],independent[1],atol=2e-13,rtol=0)
    finally:
        old.close();new.close();baseline.close()


def test_odd_mode_contraction_preserves_centroid_and_force_is_energy_gradient():
    pytest.importorskip("openmm")
    q,_=geometries(5)
    transform=contraction_matrix(5,3)
    np.testing.assert_allclose((transform@q.reshape(5,-1)).mean(axis=0),q.mean(axis=0).ravel(),atol=1e-14)
    np.testing.assert_allclose(transform@np.ones(5),np.ones(3),atol=1e-15)
    grouped,_=evaluator()
    slow=ContractedSlowPotential(grouped,5,3)
    try:
        energy,force=slow.evaluate(q)
        for bead,atom,component in ((0,0,0),(1,1,2),(2,13,1),(4,21,0)):
            plus,minus=q.copy(),q.copy()
            plus[bead,atom,component]+=1e-4;minus[bead,atom,component]-=1e-4
            expected=-(slow.evaluate(plus)[0]-slow.evaluate(minus)[0])/(2e-4)
            assert force[bead,atom,component]==pytest.approx(expected,abs=2e-8)
    finally:
        grouped.close()


def test_native_charge_free_lj_and_custom_oxygen_group_are_identical():
    pytest.importorskip("openmm")
    q,box=geometries(2)
    # Bring one pair within the cutoff; the initial cubic lattice otherwise
    # makes this comparison a vacuous equality between zero LJ forces.
    q[:,3:6,2]-=.6
    results=[]
    for backend in ("nonbonded","custom"):
        system,metadata=split_system(molecules=8,box_angstrom=box,lj_cutoff_angstrom=3.,
                                    electrostatic_cutoff_angstrom=2,ewald_tolerance=1e-5,lj_backend=backend)
        grouped=GroupEvaluator(system)
        try:
            assert np.max(abs(grouped.evaluate(q,groups=1<<3)[1]))>1e-4
            results.append(grouped.evaluate(q))
        finally:grouped.close()
    np.testing.assert_allclose(results[0][0],results[1][0],atol=2e-13,rtol=0)
    np.testing.assert_allclose(results[0][1],results[1][1],atol=2e-13,rtol=0)


def test_equal_inner_and_outer_step_matches_full_pyrpmd_bab_and_is_reversible():
    pytest.importorskip("openmm")
    q,_=geometries(4);grouped,_=evaluator()
    masses=np.tile([O_MASS_DA,H_MASS_DA,H_MASS_DA],8)*AMU_TO_AU
    class Full:
        def evaluate(self,q):
            energy,force=grouped.evaluate(q)
            return PESResult(energy,force)
    full=Dynamics(Full(),masses,1/(KB_AU*298),4,.05*FS_TO_AU,native=True,threads=1,integrator="normal-mode-bab",seed=91)
    split=SplitMTS(ContractedSlowPotential(grouped,4,4),masses,298,4,.05,.05)
    p=full.sample_momenta();a=State(q.copy(),p.copy());b=State(q.copy(),p.copy())
    try:
        for _ in range(6):
            full.step(a);split.step(b)
        np.testing.assert_allclose(a.q,b.q,atol=1e-12,rtol=0)
        np.testing.assert_allclose(a.p,b.p,atol=1e-10,rtol=0)
        assert split.energy(b)==pytest.approx(full.energy(a),abs=1e-11)
        b.p*=-1
        for _ in range(6):split.step(b)
        np.testing.assert_allclose(b.q,q,atol=1e-12,rtol=0)
        np.testing.assert_allclose(b.p,-p,atol=1e-10,rtol=0)
    finally:
        grouped.close()


def test_split_sampling_failure_restores_physical_state_clock_and_rng():
    q,_=geometries(4)
    masses=np.tile([O_MASS_DA,H_MASS_DA,H_MASS_DA],8)*AMU_TO_AU
    class FailingSlow:
        calls=0
        def evaluate(self,q):
            self.calls+=1
            if self.calls>1:raise RuntimeError("injected slow-force failure")
            return 0.,np.zeros_like(q)
    split=SplitMTS(FailingSlow(),masses,298,4,.5,.1)
    p=split.fast.sample_momenta();state=State(q.copy(),p.copy(),7,19.0)
    rng=json.dumps(split.fast.rng.bit_generator.state,sort_keys=True)
    with pytest.raises(RuntimeError,match="injected"):
        split.step(state,sampling=True)
    np.testing.assert_array_equal(state.q,q);np.testing.assert_array_equal(state.p,p)
    assert (state.step,state.time)==(7,19.0)
    assert json.dumps(split.fast.rng.bit_generator.state,sort_keys=True)==rng


def test_even_contraction_has_no_silent_nyquist_convention():
    with pytest.raises(ValueError,match="odd"):
        contraction_matrix(32,6)
    np.testing.assert_array_equal(contraction_matrix(32,32),np.eye(32))
