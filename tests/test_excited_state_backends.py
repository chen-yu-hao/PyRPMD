from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from pyrpmd.potentials.excited_state import (
    PySCFTDWavefunction,
    auxiliary_wavefunction_overlap,
)
from pyrpmd.potentials.excited_state.pyscf_tddft import _normalized_auxiliary_amplitudes


def test_casida_auxiliary_amplitudes_have_normalized_singlet_determinants():
    x = np.array([[0.3, 0.4]])
    amplitude = _normalized_auxiliary_amplitudes(x, 0, tda=True)
    assert np.isclose(2.0 * np.sum(amplitude**2), 1.0)
    tddft = _normalized_auxiliary_amplitudes(x, np.array([[0.1, -0.1]]), tda=False)
    assert np.isclose(2.0 * np.sum(tddft**2), 1.0)


def test_tda_determinant_overlap_is_identity_at_identical_geometry(monkeypatch):
    fake_gto = types.ModuleType("pyscf.gto")
    fake_gto.loads = lambda value: value
    fake_gto.intor_cross = lambda name, old, new: np.eye(3)
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)

    amplitude = np.array([[1.0 / np.sqrt(2.0), 0.0]])
    payload = PySCFTDWavefunction(
        "molecule", np.zeros((1, 3)), np.eye(3), np.array([2.0, 0.0, 0.0]),
        (amplitude,), True,
    )
    overlap = auxiliary_wavefunction_overlap(payload, payload)
    np.testing.assert_allclose(overlap, np.eye(2), atol=2e-15, rtol=0.0)


def test_determinant_pair_limit_fails_explicitly(monkeypatch):
    fake_gto = types.ModuleType("pyscf.gto")
    fake_gto.loads = lambda value: value
    fake_gto.intor_cross = lambda name, old, new: np.eye(4)
    fake_pyscf = types.ModuleType("pyscf")
    fake_pyscf.gto = fake_gto
    monkeypatch.setitem(sys.modules, "pyscf", fake_pyscf)
    monkeypatch.setitem(sys.modules, "pyscf.gto", fake_gto)
    amplitude = np.ones((2, 2)) / np.sqrt(8.0)
    payload = PySCFTDWavefunction(
        "molecule", np.zeros((2, 3)), np.eye(4), np.array([2.0, 2.0, 0.0, 0.0]),
        (amplitude,), True,
    )
    with pytest.raises(Exception, match="determinant pairs"):
        auxiliary_wavefunction_overlap(payload, payload, max_determinant_pairs=1)
