"""Independent finite-P trace sums and harmonic limits for the grid reference."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from pyrpmd.nonadiabatic import DiabaticResult


_SOURCE = Path(__file__).resolve().parents[1]/"examples/electrochemistry/adaptive_nrpmd_reference.py"
_SPEC = importlib.util.spec_from_file_location("adaptive_reference_test_module", _SOURCE)
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)


class HarmonicMatrix:
    def __init__(self, springs=(1.1, 0.8), electronic=None, shift=0.):
        self.springs = np.array(springs, float)
        self.electronic = np.array([[0.]] if electronic is None else electronic, float)
        self.shift = shift

    def evaluate(self, q):
        scalar = 0.5*np.sum(self.springs*q[:, 0, :2]**2, axis=1)+self.shift
        matrix = self.electronic+scalar[:, None, None]*np.eye(len(self.electronic))
        gradients = np.zeros((*matrix.shape, 1, 3))
        index = np.arange(len(self.electronic))
        gradients[:, index, index, 0, :2] = self.springs*q[:, None, 0, :2]
        return DiabaticResult(matrix, gradients)


class NoncommutingMatrix:
    def evaluate(self, q):
        x, y = q[:, 0, 0], q[:, 0, 1]
        scalar = 0.5*(x*x+y*y)
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0] = scalar+0.4*x
        matrix[:, 1, 1] = scalar-0.4*x
        matrix[:, 0, 1] = matrix[:, 1, 0] = 0.2+0.3*y
        return DiabaticResult(matrix, np.zeros((*matrix.shape, 1, 3)))


def explicit_paths(model, *, beta, mass, nbeads):
    # Four nuclear nodes, full enumeration of every nuclear ring. Electronic
    # matrices are multiplied explicitly for each cyclic origin; no transfer
    # matrix or coordinate correlation formula is used here.
    x = np.array([-.7, .9])
    y = np.array([-.4, .8])
    points = np.array(list(itertools.product(x, y)))
    weights = np.full(4, (x[1]-x[0])*(y[1]-y[0])/4)
    q = np.zeros((4, 1, 3))
    q[:, 0, :2] = points
    matrices = model.evaluate(q).matrix
    tau = beta/nbeads
    boltzmann = [expm(-tau*m) for m in matrices]
    total = 0.
    first = np.zeros(2)
    bead_second = np.zeros((2, 2))
    centroid_second = np.zeros((2, 2))
    populations = np.zeros(2)
    for ring in itertools.product(range(4), repeat=nbeads):
        coordinates = points[list(ring)]
        dsq = np.sum((coordinates-np.roll(coordinates, -1, axis=0))**2)
        nuclear = (mass/(2*np.pi*tau))**nbeads*np.exp(-mass*dsq/(2*tau))*np.prod(weights[list(ring)])
        products = []
        for origin in range(nbeads):
            product = np.eye(2)
            for offset in range(nbeads):
                product = product @ boltzmann[ring[(origin+offset) % nbeads]]
            products.append(product)
        electronic = np.trace(products[0])
        weight = nuclear*electronic
        centroid = coordinates.mean(axis=0)
        total += weight
        first += weight*centroid
        bead_second += weight*coordinates.T@coordinates/nbeads
        centroid_second += weight*np.outer(centroid, centroid)
        populations += nuclear*np.mean([np.diag(m) for m in products], axis=0)
    return total, first/total, bead_second/total, centroid_second/total, populations/total


@pytest.mark.parametrize("beads", [1, 2, 3, 4])
def test_transfer_ordering_and_centroid_correlations_match_enumerated_noncommuting_paths(beads):
    model = NoncommutingMatrix()
    kwargs = dict(beta=1.7, mass=1.3, nbeads=beads)
    result = reference.transfer_matrix_reference(
        model, x_bounds=(-.7, .9), y_bounds=(-.4, .8), grid_shape=(2, 2), **kwargs,
    )
    partition, mean, bead_second, centroid_second, populations = explicit_paths(model, **kwargs)
    values = result["observables"]
    assert result["log_partition_xy"] == pytest.approx(np.log(partition), abs=2e-14)
    np.testing.assert_allclose([values["x"], values["y"]], mean, atol=2e-14)
    np.testing.assert_allclose([values["x2_bead"], values["y2_bead"], values["xy_bead"]],
                               [bead_second[0, 0], bead_second[1, 1], bead_second[0, 1]], atol=2e-14)
    np.testing.assert_allclose([values["x2_centroid"], values["y2_centroid"], values["xy_centroid"]],
                               [centroid_second[0, 0], centroid_second[1, 1], centroid_second[0, 1]], atol=2e-14)
    np.testing.assert_allclose([values["population_0"], values["population_1"]], populations, atol=2e-14)


@pytest.mark.parametrize("beads", [1, 2, 4, 8])
def test_finite_bead_harmonic_normal_modes_and_partition(beads):
    beta, mass = 2.3, 1.5
    model = HarmonicMatrix()
    result = reference.transfer_matrix_reference(
        model, beta=beta, mass=mass, nbeads=beads,
        x_bounds=(-5.5, 5.5), y_bounds=(-5.5, 5.5), grid_shape=(31, 31),
    )
    frequency = np.sqrt(model.springs/mass)
    ring_frequency = 2*beads/beta*np.sin(np.pi*np.arange(beads)/beads)
    exact_bead = np.sum(1/(model.springs[:, None]+mass*ring_frequency[None, :]**2), axis=1)/beta
    exact_centroid = 1/(beta*model.springs)
    exact_log_z = -np.log(2*np.sinh(beads*np.arcsinh(beta*frequency/(2*beads)))).sum()
    values = result["observables"]
    np.testing.assert_allclose([values["x"], values["y"]], 0., atol=1e-13)
    np.testing.assert_allclose([values["x2_bead"], values["y2_bead"]], exact_bead, atol=2e-7)
    np.testing.assert_allclose([values["x2_centroid"], values["y2_centroid"]], exact_centroid, atol=2e-7)
    assert result["log_partition_xy"] == pytest.approx(exact_log_z, abs=2e-8)
    assert values["radius_gyration_squared_xy"] == pytest.approx(float(np.sum(exact_bead-exact_centroid)), abs=3e-7)
    assert result["boundary_mass"] < 1e-8


def test_constant_non_diagonal_electronic_matrix_has_exact_boltzmann_populations():
    electronic = np.array([[0.17, .23], [.23, -.31]])
    beta = 1.9
    model = HarmonicMatrix(electronic=electronic)
    result = reference.transfer_matrix_reference(
        model, beta=beta, nbeads=4, x_bounds=(-4.5, 4.5), y_bounds=(-4.5, 4.5),
        grid_shape=(21, 21),
    )
    density = expm(-beta*electronic)
    populations = np.diag(density)/np.trace(density)
    np.testing.assert_allclose([result["observables"][f"population_{i}"] for i in range(2)],
                               populations, atol=2e-14)
    assert result["observables"]["x2_bead"] > result["observables"]["x2_centroid"]


@pytest.mark.parametrize("beads", [1, 2, 4])
def test_uncoupled_unequal_harmonic_surfaces_have_exact_partition_populations(beads):
    springs = np.array([[1.1, .8], [.7, 1.4]])
    offsets = np.array([0., .23])
    beta, mass = 2.3, 1.5

    class Uncoupled:
        def evaluate(self, q):
            matrix = np.zeros((len(q), 2, 2))
            for state in range(2):
                matrix[:, state, state] = offsets[state]+.5*np.sum(
                    springs[state]*q[:, 0, :2]**2, axis=1,
                )
            return matrix

    result = reference.transfer_matrix_reference(
        Uncoupled(), beta=beta, mass=mass, nbeads=beads,
        x_bounds=(-5.5, 5.5), y_bounds=(-5.5, 5.5), grid_shape=(31, 31),
    )
    omega = np.sqrt(springs/mass)
    partitions = np.exp(-beta*offsets)/np.prod(
        2*np.sinh(beads*np.arcsinh(beta*omega/(2*beads))), axis=1,
    )
    populations = partitions/partitions.sum()
    ring_frequency = 2*beads/beta*np.sin(np.pi*np.arange(beads)/beads)
    bead_variances = np.sum(1/(springs[:, :, None]+mass*ring_frequency[None, None, :]**2), axis=2)/beta
    values = result["observables"]
    np.testing.assert_allclose([values["population_0"], values["population_1"]], populations, atol=2e-8)
    np.testing.assert_allclose([values["x2_bead"], values["y2_bead"]], populations@bead_variances, atol=2e-7)
    np.testing.assert_allclose([values["x2_centroid"], values["y2_centroid"]], populations@(1/(beta*springs)), atol=2e-7)
    assert result["log_partition_xy"] == pytest.approx(np.log(partitions.sum()), abs=2e-8)


def test_common_energy_shift_changes_only_log_partition_and_bead_only_option():
    model = HarmonicMatrix(electronic=[[.2, .3], [.3, -.1]])
    base = reference.transfer_matrix_reference(model, beta=1.8, nbeads=3, grid_shape=(11, 13))
    model.shift = 1000.
    shifted = reference.transfer_matrix_reference(model, beta=1.8, nbeads=3, grid_shape=(11, 13))
    np.testing.assert_allclose(list(base["observables"].values()), list(shifted["observables"].values()), atol=2e-13)
    assert shifted["log_partition_xy"]-base["log_partition_xy"] == pytest.approx(-1800., abs=3e-12)
    no_centroid = reference.transfer_matrix_reference(model, beta=1.8, nbeads=3, grid_shape=(11, 13), centroid_moments=False)
    assert no_centroid["centroid_covariance_xy"] is None
    assert "x2_centroid" not in no_centroid["observables"]
    assert no_centroid["observables"]["x2_bead"] == pytest.approx(shifted["observables"]["x2_bead"], abs=1e-14)


@pytest.mark.parametrize("bad", [
    {"beta": 0}, {"mass": np.inf}, {"nbeads": True}, {"nbeads": 1.5},
    {"x_bounds": (1, -1)}, {"y_bounds": (0, np.nan)}, {"grid_shape": (1, 3)},
    {"grid_shape": (3, 3, 3)}, {"centroid_moments": "yes"},
    {"nbeads": 4, "max_memory_mb": .001},
])
def test_invalid_reference_configuration_rejected_before_dense_work(bad):
    with pytest.raises(ValueError):
        reference.transfer_matrix_reference(HarmonicMatrix(), **bad)


def test_nonhermitian_matrix_is_not_silently_symmetrized():
    class Bad:
        def evaluate(self, q):
            return np.broadcast_to([[0., 1.], [0., 0.]], (len(q), 2, 2))
    with pytest.raises(ValueError, match="real symmetric"):
        reference.transfer_matrix_reference(Bad(), nbeads=1, grid_shape=(3, 3))


def test_example_charge_observable_and_cli_json(tmp_path):
    target = tmp_path/"reference.json"
    result = reference.main(["--potential-V", "0.6", "--beads", "2", "--grid", "11", "--output", str(target)])
    values = result["observables"]
    assert values["electron_number_bead"] == pytest.approx(
        .3*values["x"]+.08*values["y"]+1.2*result["chemical_potential_au"], abs=1e-15,
    )
    assert target.is_file()
    assert result["scope"]["physical_converged"] is False
    assert result["scope"]["z_partition_included"] is False
