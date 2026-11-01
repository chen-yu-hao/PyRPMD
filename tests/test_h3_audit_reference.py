"""Reject mismatched nuclear models before assigning a literature target."""
import numpy as np
import pytest

from benchmarks.h3.audit_ui import matched_reference
from benchmarks.h3.classical_tst import MASS_AU
from pyrpmd.system import System


def hydrogen(masses=None):
    return System(('H', 'H', 'H'), np.full(3, MASS_AU) if masses is None else masses,
                  np.array([[-1.4, 0., 0.], [0., 0., 0.], [1.4, 0., 0.]]))


@pytest.mark.parametrize('beads,target', [(1, 1.56e-18), (128, 2.51e-16)])
def test_matched_classical_and_quantum_targets(beads, target):
    assert matched_reference({'temperature_K': 300., 'beads': beads}, hydrogen()) == target


@pytest.mark.parametrize('temperature', [200., 1000., float('nan')])
def test_wrong_temperature_has_no_hydrogen_300k_target(temperature):
    with pytest.raises(ValueError, match='300 K'):
        matched_reference({'temperature_K': temperature, 'beads': 128}, hydrogen())


@pytest.mark.parametrize('mass_factor', [.113, 2.])
def test_muonium_and_deuterium_do_not_inherit_hydrogen_target(mass_factor):
    masses = np.full(3, MASS_AU)
    masses[0] *= mass_factor
    with pytest.raises(ValueError, match='ordinary-1H'):
        matched_reference({'temperature_K': 300., 'beads': 128}, hydrogen(masses))
