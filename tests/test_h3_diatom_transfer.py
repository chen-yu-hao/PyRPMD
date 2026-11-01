"""Analytic 3D free/harmonic identities for the finite-P radial trace."""
import numpy as np
import pytest
from benchmarks.h3.diatom_transfer import free_radial_kernel, primitive_partition


def test_zero_angular_momentum_equals_half_line_image_kernel():
    r=np.linspace(.05,4.,67);mass=2.;tau=.7
    expected=np.sqrt(mass/(2*np.pi*tau))*(
        np.exp(-mass*(r[:,None]-r[None])**2/(2*tau))-
        np.exp(-mass*(r[:,None]+r[None])**2/(2*tau)))
    np.testing.assert_allclose(free_radial_kernel(r,tau,mass,0),expected,atol=4e-15,rtol=6e-14)


def test_angular_trace_recovers_cartesian_free_kernel():
    r=np.linspace(.1,3.,20);mass=1.7;tau=.9
    actual=sum((2*j+1)*np.diag(free_radial_kernel(r,tau,mass,j)) for j in range(49))
    expected=4*np.pi*r*r*(mass/(2*np.pi*tau))**1.5
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-14)


@pytest.mark.parametrize('beads',[1,2,8,32])
def test_finite_bead_three_dimensional_harmonic_partition(beads):
    beta=3.;mass=1.3;omega=.8
    actual=primitive_partition(lambda r:.5*mass*omega**2*r*r,
        beta=beta,mass=mass,beads=beads,order=96,domain=(0.,10.),jmax=20)
    exact=3*np.log(beta*omega/(2*np.sinh(beads*np.arcsinh(beta*omega/(2*beads)))))
    assert actual['log_quantum_classical_ratio']==pytest.approx(exact,abs=3e-10)
    assert actual['relative_absolute_negative_trace_bound']<1e-12


def test_finite_bead_partition_energy_zero_cancels():
    options=dict(beta=2.,mass=1.,beads=8,order=64,domain=(0.,10.),jmax=20)
    a=primitive_partition(lambda r:.5*r*r,**options)
    b=primitive_partition(lambda r:.5*r*r-100.,origin=-100.,**options)
    assert a['log_quantum_classical_ratio']==pytest.approx(b['log_quantum_classical_ratio'],abs=1e-13)
