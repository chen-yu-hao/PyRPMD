"""Periodic Lennard-Jones references for forces, virials, lattices and ABI v2."""

import os
import subprocess
import sys

import numpy as np
import pytest

from pyrpmd.potentials import LennardJonesPotential, make_potential
from pyrpmd.constants import ANGSTROM_TO_AU, EV_TO_AU


def _cell():
    return np.array([[12.0, 1.1, 0.4], [0.0, 11.0, 0.7], [0.0, 0.0, 10.0]])


def _potential(backend="numpy", cell=None):
    return LennardJonesPotential(
        _cell() if cell is None else cell,
        epsilon=0.37, sigma=1.15, cutoff=3.4, skin=0.5,
        backend=backend,
    )


def _coordinates():
    return np.array([
        [[1.1, 1.3, 1.0], [2.45, 1.65, 1.2], [8.9, 8.2, 7.4], [10.6, 1.0, 1.5]],
        [[1.3, 1.2, 1.1], [2.70, 1.55, 1.4], [8.5, 8.0, 7.1], [10.7, 1.1, 1.4]],
    ])


def test_lj_force_matches_independent_finite_difference():
    potential = _potential()
    q = _coordinates()
    result = potential.evaluate(q)
    numerical = np.empty_like(q)
    step = 2e-6
    for atom in range(q.shape[1]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += step
            minus[:, atom, axis] -= step
            numerical[:, atom, axis] = -(
                potential.evaluate(plus).energy - potential.evaluate(minus).energy
            ) / (2 * step)
    np.testing.assert_allclose(result.forces, numerical, atol=3e-9, rtol=3e-8)
    np.testing.assert_allclose(result.forces.sum(axis=1), 0.0, atol=2e-14)


def test_lj_virial_matches_finite_homogeneous_strain():
    q = _coordinates()[:1]
    cell = _cell()
    reference = _potential(cell=cell).evaluate(q)
    step = 2e-6
    numerical = np.empty((3, 3))
    for row in range(3):
        for column in range(3):
            plus = np.eye(3)
            minus = np.eye(3)
            plus[row, column] += step
            minus[row, column] -= step
            ep = _potential(cell=cell @ plus).evaluate(q @ plus).energy[0]
            em = _potential(cell=cell @ minus).evaluate(q @ minus).energy[0]
            numerical[row, column] = -(ep - em) / (2 * step)
    np.testing.assert_allclose(reference.virial[0], numerical, atol=4e-9, rtol=4e-8)
    np.testing.assert_allclose(reference.virial[0], reference.virial[0].T, atol=3e-14)


def test_shifted_energy_is_continuous_at_cutoff():
    cell = np.eye(3) * 20.0
    potential = LennardJonesPotential(
        cell, epsilon=0.4, sigma=1.2, cutoff=3.0, skin=0.4,
        backend="numpy",
    )
    at = np.array([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    outside = at.copy()
    outside[0, 1, 0] += 1e-8
    inside = at.copy()
    inside[0, 1, 0] -= 1e-8
    assert potential.evaluate(at).energy[0] == 0.0
    assert potential.evaluate(outside).energy[0] == 0.0
    assert abs(potential.evaluate(inside).energy[0]) < 2e-8


def test_fcc_nearest_neighbor_lattice_sum():
    lattice = 4.0
    repeats = 3
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    positions = []
    for i in range(repeats):
        for j in range(repeats):
            for k in range(repeats):
                positions.extend((basis + [i, j, k]) * lattice)
    q = np.asarray(positions)[None]
    cell = np.eye(3) * repeats * lattice
    epsilon, sigma, cutoff = 0.21, 2.5, 3.0
    potential = LennardJonesPotential(
        cell, epsilon=epsilon, sigma=sigma, cutoff=cutoff, skin=0.3,
        backend="numpy",
    )
    nearest = lattice / np.sqrt(2.0)
    sr6 = (sigma / nearest) ** 6
    cr6 = (sigma / cutoff) ** 6
    pair_energy = 4 * epsilon * (sr6 * sr6 - sr6 - cr6 * cr6 + cr6)
    expected_pairs = len(positions) * 12 // 2
    result = potential.evaluate(q)
    np.testing.assert_allclose(result.energy[0], expected_pairs * pair_energy, rtol=2e-13)
    np.testing.assert_allclose(result.forces, 0.0, atol=2e-13)


def test_factory_parses_units_and_requires_full_periodicity():
    system = {"cell": np.eye(3) * 40.0, "pbc": [True, True, True]}
    potential = make_potential({
        "provider": "lj",
        "params": {
            "epsilon": "0.2 eV", "sigma": "3.0 angstrom",
            "cutoff": "8.0 angstrom", "skin": "0.4 angstrom",
            "backend": "numpy",
        },
    }, system)
    assert isinstance(potential, LennardJonesPotential)
    assert potential.epsilon == pytest.approx(0.2 * EV_TO_AU)
    assert potential.sigma == pytest.approx(3.0 * ANGSTROM_TO_AU)
    assert potential.cutoff == pytest.approx(8.0 * ANGSTROM_TO_AU)
    with pytest.raises(Exception, match="three periodic axes"):
        make_potential(
            {"provider": "lj", "params": {"backend": "numpy"}},
            {"cell": np.eye(3) * 10.0, "pbc": [True, True, False]},
        )
    with pytest.raises(ValueError, match="unknown Lennard-Jones"):
        make_potential(
            {"provider": "lj", "params": {"backend": "numpy", "typo": 1}},
            system,
        )


def test_cutoff_rejects_ambiguous_minimum_image_cell():
    with pytest.raises(ValueError, match="half the shortest cell face height"):
        LennardJonesPotential(np.eye(3) * 5.0, cutoff=2.6, backend="numpy")


def test_native_matches_numpy_and_rebuilds_verlet_list():
    native = pytest.importorskip("pyrpmd._native")
    if not hasattr(native, "builtin_lj"):
        pytest.skip("installed native extension lacks built-in LJ")
    q = _coordinates()
    reference = _potential("numpy")
    compiled = _potential("native")
    for displacement in (0.0, 0.05, 0.31):
        moved = q.copy()
        moved[:, 0, 0] += displacement
        expected, actual = reference.evaluate(moved), compiled.evaluate(moved)
        np.testing.assert_allclose(actual.energy, expected.energy, atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(actual.forces, expected.forces, atol=3e-14, rtol=3e-14)
        np.testing.assert_allclose(actual.virial, expected.virial, atol=3e-14, rtol=3e-14)
        np.testing.assert_array_equal(actual.status, 0)


def test_triclinic_frames_match_ase_lennard_jones():
    ase = pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    cell = _cell()
    q = _coordinates()
    epsilon, sigma, cutoff = 0.37, 1.15, 3.4
    potential = LennardJonesPotential(
        cell, epsilon=epsilon, sigma=sigma, cutoff=cutoff, skin=0.5,
        backend="numpy",
    )
    actual = potential.evaluate(q)
    for frame, positions in enumerate(q):
        atoms = ase.Atoms("Ar4", positions=positions, cell=cell, pbc=True)
        atoms.calc = LennardJones(epsilon=epsilon, sigma=sigma, rc=cutoff, smooth=False)
        np.testing.assert_allclose(actual.energy[frame], atoms.get_potential_energy(), atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(actual.forces[frame], atoms.get_forces(), atol=3e-13, rtol=3e-13)


def test_native_trajectory_is_bitwise_invariant_across_thread_counts():
    native = pytest.importorskip("pyrpmd._native")
    if not native.has_openmp():
        pytest.skip("OpenMP build required")
    code = r'''
import hashlib, os
import numpy as np
from pyrpmd import Dynamics, State
from pyrpmd.potentials import HarmonicPotential
threads = int(os.environ["CASE_THREADS"])
rng = np.random.default_rng(2718)
q = rng.normal(scale=.2, size=(32, 64, 3))
p = rng.normal(scale=.1, size=q.shape)
d = Dynamics(HarmonicPotential(k=.002), np.full(64, 1836.), beta=1000.,
             nbeads=32, dt=.1, threads=threads, native="required",
             loop_backend="auto", integrator="cayley-bcb", seed=17)
s = State(q, p)
d.propagate(s, 20, sample_interval=20, record_centroids=False)
print(hashlib.sha256(s.q.tobytes() + s.p.tobytes()).hexdigest())
'''
    hashes = []
    for threads in (1, 2, 4, 8):
        env = dict(os.environ, CASE_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                   OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1")
        completed = subprocess.run(
            [sys.executable, "-c", code], env=env, check=True,
            text=True, capture_output=True,
        )
        hashes.append(completed.stdout.strip().splitlines()[-1])
    assert len(set(hashes)) == 1


def test_lj_rejects_boolean_and_overflowing_parameters():
    with pytest.raises(ValueError, match="not booleans"):
        LennardJonesPotential(np.eye(3) * 10.0, epsilon=True, backend="numpy")
    with pytest.raises(ValueError, match="overflow"):
        LennardJonesPotential(
            np.eye(3) * 10.0, epsilon=1.0, sigma=1.0,
            cutoff=1e200, skin=0.0, backend="numpy",
        )
