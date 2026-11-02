"""Independent analytic spectrum and partition-function checks."""
import numpy as np
import pytest

from benchmarks.h3.diatom_partition import (
    classical_log_partition, quantum_log_partition, sine_kinetic, spectrum,
)


def test_sine_dvr_free_box_eigenvalues():
    _, kinetic=sine_kinetic(64, .3, 8.3, 2.5)
    expected=(np.pi*np.arange(1,65)/8.)**2/5.
    np.testing.assert_allclose(np.linalg.eigvalsh(kinetic),expected,atol=2e-13,rtol=2e-13)


def test_three_dimensional_harmonic_radial_spectrum():
    mass=2.;omega=.7
    energies=spectrum(lambda r:.5*mass*omega**2*r*r,mass=mass,points=128,
                      domain=(0.,12.),jmax=8,levels=6)
    exact=omega*(2*np.arange(6)[None]+np.arange(9)[:,None]+1.5)
    np.testing.assert_allclose(energies,exact,atol=3e-11,rtol=0)


def test_three_dimensional_harmonic_quantum_and_classical_partition():
    mass=2.;omega=.7;beta=2.
    energies=omega*(2*np.arange(40)[None]+np.arange(61)[:,None]+1.5)
    quantum=quantum_log_partition(energies,beta)
    exact_quantum=-3*np.log(2*np.sinh(beta*omega/2))
    assert quantum['log_partition']==pytest.approx(exact_quantum,abs=2e-14)
    classical=classical_log_partition(lambda r:.5*mass*omega**2*r*r,beta,mass,
                                      domain=(0.,12.))
    assert classical==pytest.approx(-3*np.log(beta*omega),abs=1e-13)
    # This independently checks the 4*pi, thermal-wavelength and 2J+1 factors.
    assert np.exp(quantum['log_partition']-classical)==pytest.approx(
        (beta*omega/(2*np.sinh(beta*omega/2)))**3,rel=1e-13)


def test_energy_origin_cancels_partition_ratio():
    potential=lambda r:.5*r*r
    energies=spectrum(potential,mass=1.,points=96,domain=(0.,12.),jmax=12,levels=8)
    values=[]
    for origin in (0.,-100.,20.):
        values.append(quantum_log_partition(energies,2.,origin=origin)['log_partition']-
                      classical_log_partition(potential,2.,1.,domain=(0.,12.),origin=origin))
    np.testing.assert_allclose(values,values[0],atol=4e-14,rtol=0)
