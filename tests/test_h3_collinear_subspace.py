"""Collinear H + H2 model used by benchmarks/h3_subspace (low-temperature RPMD test)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "benchmarks" / "h3" / "build" / "libbkmp2.so"
sys.path.insert(0, str(ROOT / "benchmarks" / "h3_subspace"))


def test_symmetric_coordinates_are_orthonormal():
    from collinear import CS, CT, MASS

    # mass-weighted Jacobi coordinates: Q = sqrt(2m/3) R, q = sqrt(m/2) r, r1 = r, r2 = R - r/2
    a, b = np.sqrt(2 / MASS), np.sqrt(1.5 / MASS)
    ds = np.array([b, -1.5 * a]) / CS
    dt = np.array([b, 0.5 * a]) / CT
    assert np.allclose([ds @ ds, dt @ dt, ds @ dt], [1.0, 1.0, 0.0], atol=1e-12)


@pytest.mark.skipif(not LIBRARY.exists(), reason="build BKMP2 with benchmarks/h3/build_pes.py")
def test_collinear_saddle_and_crossover_temperature():
    from collinear import Collinear, to_st

    potential = Collinear()
    ts = np.array([0.0, to_st(1.757, 1.757)[1]])
    assert potential(ts) == pytest.approx(0.015347, abs=2e-5)
    h = 0.5
    k = (potential(ts + [h, 0]) - 2 * potential(ts) + potential(ts - [h, 0])) / h**2
    omega = np.sqrt(-k) * 219474.63
    assert omega == pytest.approx(1510.7, abs=2.0)
    assert omega * 1.4387769 / (2 * np.pi) == pytest.approx(346.0, abs=0.5)
    g = potential.gradient(ts[None] + [[3.0, 2.0]])
    e = np.array([1e-4, 0.0])
    fd = (potential(ts + [3.0, 2.0] + e) - potential(ts + [3.0, 2.0] - e)) / 2e-4
    assert g[0, 0] == pytest.approx(fd, rel=1e-5)
