"""Independent checks of the NRPMD Hamiltonian and its exact MInt subflow."""
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from pyrpmd.dynamics import State, Dynamics, INTEGRATORS
from pyrpmd.potentials import HarmonicPotential
from pyrpmd.potentials.base import PotentialError
from pyrpmd.nonadiabatic import (
    DiabaticResult, DiagonalPotential, NRPMDState, NRPMDDynamics,
    validate_diabatic_result, _mint,
)


class LinearMatrix:
    def __init__(self, matrix, slope=None, curvature=0.0):
        self.matrix = np.array(matrix, dtype=float)
        self.slope = np.zeros_like(self.matrix) if slope is None else np.array(slope, dtype=float)
        self.curvature = curvature

    def evaluate(self, q):
        b, n, _ = q.shape
        s = len(self.matrix)
        matrix = self.matrix[None] + q[:, 0, 0, None, None]*self.slope[None]
        matrix += 0.5*self.curvature*np.sum(q*q, axis=(1, 2))[:, None, None]*np.eye(s)
        gradients = self.curvature*q[:, None, None]*np.eye(s)[None, :, :, None, None]
        gradients = np.broadcast_to(gradients, (b,s,s,n,3)).copy()
        gradients[:, :, :, 0, 0] += self.slope
        return DiabaticResult(matrix, gradients)


def state(beads=3, states=2):
    rng = np.random.default_rng(8271)
    nuclear = State(rng.normal(size=(beads,1,3)), rng.normal(size=(beads,1,3)))
    return NRPMDState.from_nuclear(nuclear, states, 0, rng)


def vector(s):
    return np.concatenate([a.ravel() for a in (s.q,s.p,s.mapping_q,s.mapping_p)])


@pytest.mark.parametrize('states', [1,2,3,7])
def test_focused_state_and_copy(states):
    s = state(states=states)
    np.testing.assert_allclose(s.populations(), np.eye(states)[0], atol=8e-16)
    t = s.copy()
    assert not np.shares_memory(t.mapping_q, s.mapping_q)
    t.mapping_q[:] = t.mapping_p[:] = 0
    assert np.all(t.populations() == -0.5)
    np.testing.assert_allclose(s.populations().sum(), 1, atol=2e-15)


@pytest.mark.parametrize('states', [1,2,3])
def test_constant_matrix_matches_exact_electronic_rotation(states):
    rng = np.random.default_rng(6)
    a = rng.normal(size=(states,states))
    a = a+a.T
    s = state(states=states)
    z0 = s.mapping_q+1j*s.mapping_p
    dyn = NRPMDDynamics(LinearMatrix(a), [1.7], 4, 3, .07, states, native=False)
    for _ in range(11):
        dyn.step(s)
    reference = z0 @ expm(-1j*a*.77).T
    np.testing.assert_allclose(s.mapping_q+1j*s.mapping_p, reference, atol=2e-14)
    np.testing.assert_allclose(np.sum(abs(reference)**2, axis=1), states+2, atol=3e-14)


@pytest.mark.parametrize('gap', [0,1e-15,.8])
def test_mint_matches_independent_ode_including_momentum(gap):
    pot = LinearMatrix([[.2,0],[0,.2+gap]], [[.7,.2],[.2,-.3]])
    s = state(beads=1)
    s.q[:] = 0
    initial = s.copy()
    r = pot.evaluate(s.q)
    def rhs(t, y):
        p,x,z = y[:3], y[3:5],y[5:]
        g = r.gradients[0]
        force = -.5*(np.einsum('i,ijac,j->ac',x,g,x)+np.einsum('i,ijac,j->ac',z,g,z)-np.einsum('iiac->ac',g))
        return np.concatenate((force.ravel(),r.matrix[0]@z,-r.matrix[0]@x))
    y0 = np.concatenate((s.p.ravel(),s.mapping_q.ravel(),s.mapping_p.ravel()))
    exact = solve_ivp(rhs,(0,.41),y0,method='DOP853',rtol=3e-13,atol=3e-14).y[:,-1]
    _mint(s,r,.41)
    actual = np.concatenate((s.p.ravel(),s.mapping_q.ravel(),s.mapping_p.ravel()))
    np.testing.assert_allclose(actual,exact,rtol=2e-12,atol=1e-13)
    np.testing.assert_array_equal(s.q,initial.q)


@pytest.mark.parametrize('scheme', INTEGRATORS)
def test_uncoupled_focused_limit_matches_rpmd(scheme):
    s = state()
    ordinary = State(s.q,s.p)
    a = Dynamics(HarmonicPotential(.7),[1.7],4,3,.01,native=False,integrator=scheme)
    b = NRPMDDynamics(LinearMatrix(np.zeros((2,2)),curvature=.7),[1.7],4,3,.01,2,native=False,integrator=scheme)
    for _ in range(35):
        a.step(ordinary)
        b.step(s)
    np.testing.assert_allclose(s.q,ordinary.q,atol=2e-14)
    np.testing.assert_allclose(s.p,ordinary.p,atol=2e-14)
    assert b.energy(s) == pytest.approx(a.energy(ordinary),rel=2e-14)


@pytest.mark.parametrize('scheme', INTEGRATORS)
def test_reversal_and_norm(scheme):
    s = state()
    original = s.copy()
    pot = LinearMatrix([[.1,.07],[.07,-.1]],[[.2,.03],[.03,-.2]],curvature=.3)
    d = NRPMDDynamics(pot,[1.7],4,3,.006,2,native=False,integrator=scheme)
    e = d.energy(s)
    for _ in range(80): d.step(s)
    assert abs(d.energy(s)-e)<2e-4
    np.testing.assert_allclose(s.populations().sum(),1,atol=2e-13)
    s.p *= -1
    s.mapping_p *= -1
    for _ in range(80): d.step(s)
    s.p *= -1
    s.mapping_p *= -1
    np.testing.assert_allclose(vector(s),vector(original),atol=1e-12)


def test_second_order_convergence_against_independent_full_hamiltonian_ode():
    s = state(beads=1)
    pot = LinearMatrix([[.1,.07],[.07,-.1]],[[.2,.03],[.03,-.2]],curvature=.3)
    def rhs(t,y):
        q,p,x,z = y[:3],y[3:6],y[6:8],y[8:]
        r=pot.evaluate(q.reshape(1,1,3))
        g=r.gradients[0]
        force=-.5*(np.einsum('i,ijac,j->ac',x,g,x)+np.einsum('i,ijac,j->ac',z,g,z)-np.einsum('iiac->ac',g))
        return np.concatenate((p/1.7,force.ravel(),r.matrix[0]@z,-r.matrix[0]@x))
    exact=solve_ivp(rhs,(0,.8),vector(s),method='DOP853',rtol=1e-12,atol=1e-13).y[:,-1]
    errors=[]
    for dt in [.08,.04,.02]:
        d=NRPMDDynamics(pot,[1.7],4,1,dt,2,native=False)
        trial=s.copy()
        for _ in range(round(.8/dt)): d.step(trial)
        errors.append(np.linalg.norm(vector(trial)-exact))
    assert 3.9<errors[0]/errors[1]<4.1
    assert 3.9<errors[1]/errors[2]<4.1


def test_symplectic_jacobian():
    # One bead, one atom: independent finite-difference Jacobian of the full map.
    s=state(beads=1)
    pot=LinearMatrix([[.1,.07],[.07,-.1]],[[.2,.03],[.03,-.2]],curvature=.3)
    d=NRPMDDynamics(pot,[1.7],4,1,.09,2,native=False)
    y=vector(s)
    def advance(y):
        t=NRPMDState(y[:3].reshape(1,1,3),y[3:6].reshape(1,1,3),mapping_q=y[6:8].reshape(1,2),mapping_p=y[8:].reshape(1,2))
        d.step(t)
        return vector(t)
    eps=2e-6
    jac=np.column_stack([(advance(y+eps*e)-advance(y-eps*e))/(2*eps) for e in np.eye(10)])
    omega=np.zeros((10,10))
    for i,j in [(0,3),(1,4),(2,5),(6,8),(7,9)]:
        omega[i,j]=1;omega[j,i]=-1
    np.testing.assert_allclose(jac.T@omega@jac,omega,atol=6e-10)


def test_failed_second_pes_evaluation_rolls_back_electronic_and_nuclear_state():
    s=state()
    before=vector(s)
    class Failing(LinearMatrix):
        count=0
        def evaluate(self,q):
            self.count+=1
            if self.count==2: raise PotentialError('failure')
            return super().evaluate(q)
    d=NRPMDDynamics(Failing([[.1,.2],[.2,-.1]]),[1.7],4,3,.03,2,native=False)
    with pytest.raises(PotentialError): d.step(s)
    np.testing.assert_array_equal(vector(s),before)
    assert (s.step,s.time)==(0,0)


@pytest.mark.parametrize('defect',['complex','asymmetric','gradient_asymmetric','shape','gradient_shape','nan','status','nstates'])
def test_diabatic_contract_rejects_invalid_results(defect):
    q=np.zeros((3,1,3));r=LinearMatrix([[.1,.2],[.2,-.1]]).evaluate(q)
    ns=2
    if defect=='complex': r.matrix=r.matrix.astype(complex)
    if defect=='asymmetric':r.matrix[0,0,1]+=.01
    if defect=='gradient_asymmetric':r.gradients[0,0,1,0,0]=.1
    if defect=='shape':r.matrix=r.matrix[0]
    if defect=='gradient_shape':r.gradients=r.gradients[0]
    if defect=='nan':r.matrix[0,0,0]=np.nan
    if defect=='status':r.status=np.ones(3,dtype=int)
    if defect=='nstates':ns=3
    with pytest.raises(PotentialError):validate_diabatic_result(r,q.shape,ns)


def test_diagonal_adapter_and_all_bead_asymptotic_check():
    pot=LinearMatrix([[1,.2],[.2,2]],[[.7,.1],[.1,-.3]])
    s=state()
    diagonal=DiagonalPotential(pot,1,2).evaluate(s.q)
    np.testing.assert_allclose(diagonal.energy,2-.3*s.q[:,0,0])
    np.testing.assert_allclose(diagonal.forces[:,0,0],.3)
    d=NRPMDDynamics(pot,[1.7],4,3,.03,2,native=False)
    assert d.asymptotic_coupling(s)==pytest.approx(np.max(abs(.2+.1*s.q[:,0,0])))


@pytest.mark.parametrize('states,initial',[(0,0),(2,2),(2,-1),(True,0),(2,True)])
def test_invalid_focused_parameters(states,initial):
    with pytest.raises(ValueError):NRPMDState.from_nuclear(State(np.zeros((1,1,3)),np.zeros((1,1,3))),states,initial,0)


@pytest.mark.parametrize("nstates, selected", [(1,0),(2,0),(2,1)])
@pytest.mark.parametrize("scheme", INTEGRATORS)
def test_distinct_uncoupled_surfaces_follow_selected_diagonal(nstates, selected, scheme):
    pot = LinearMatrix(np.diag([.3,-.6][:nstates]),
                       np.diag([.7,-.2][:nstates]),curvature=.4)
    nuclear = state().copy()
    s = NRPMDState.from_nuclear(nuclear,nstates,selected,981)
    ordinary = State(s.q,s.p)
    a = Dynamics(DiagonalPotential(pot,selected,nstates),[1.7],4,3,.01,
                 native=False,integrator=scheme)
    b = NRPMDDynamics(pot,[1.7],4,3,.01,nstates,native=False,integrator=scheme)
    for _ in range(30):
        a.step(ordinary)
        b.step(s)
    np.testing.assert_allclose(s.q,ordinary.q,atol=3e-14)
    np.testing.assert_allclose(s.p,ordinary.p,atol=3e-14)
    assert b.energy(s) == pytest.approx(a.energy(ordinary),rel=3e-14)
