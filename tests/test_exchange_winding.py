"""Independent lattice sums, derivative checks, and bounded-work contracts."""
import itertools
import math

import numpy as np
import pytest
from scipy.special import logsumexp

from pyrpmd.exchange_winding import GaussianWindingKernel
from pyrpmd.geometry import FixedCellGeometry
from pyrpmd.pimd_methods import BosonicExchangePotential, bosonic_exchange
from pyrpmd.potentials.base import PESResult, PotentialError
from pyrpmd.potentials.builtin import FreePotential


def image_reference(delta, cell, flags, alpha, cutoff=10):
    """Explicit images in the original unreduced basis, without duality."""
    indices = np.asarray(list(itertools.product(*[
        range(-cutoff, cutoff + 1) if flag else (0,) for flag in flags])))
    # Canonicalize by an independent original-cell solve; preserve perpendicular
    # and nonperiodic components, then directly enumerate surrounding images.
    fractional = delta @ np.linalg.inv(cell)
    images = np.zeros_like(fractional)
    images[..., flags] = np.rint(fractional[..., flags])
    vectors = delta[..., None, :] - (images[..., None, :] + indices) @ cell
    exponent = -alpha * np.sum(vectors * vectors, axis=-1)
    logarithm = logsumexp(exponent, axis=-1)
    weights = np.exp(exponent - logarithm[..., None])
    mean = np.einsum("...n,...nc->...c", weights, vectors)
    second = np.einsum("...n,...ni,...nj->...ij", weights, vectors, vectors)
    return logarithm, mean, second


@pytest.mark.parametrize("alpha", [0.35, 1.4, 9.0])
@pytest.mark.parametrize("skew,flags", [
    (False, (True, True, True)), (False, (True, False, True)),
    (True, (True, True, True)), (True, (True, False, True)),
    (True, (False, True, False)),
])
def test_gaussian_link_moments_match_independent_explicit_images(alpha, skew, flags):
    cell = np.array([[2.1, 0, 0], [0.4 if skew else 0, 2.3, 0],
                     [0.3 if skew else 0, 0.2 if skew else 0, 2.6]])
    delta = np.random.default_rng(714).normal(size=(2, 3, 3))
    kernel = GaussianWindingKernel(FixedCellGeometry(cell, flags), alpha)
    result = kernel.evaluate(delta)
    expected = image_reference(delta, cell, np.asarray(flags), alpha)
    for actual, reference in zip((result.log_kernel, result.mean_displacement, result.second_moment), expected):
        np.testing.assert_allclose(actual, reference, rtol=3e-13, atol=3e-13)
    assert max(kernel.error_bounds) <= kernel.tolerance
    assert kernel.terms <= kernel.max_terms


@pytest.mark.parametrize("rotate", [False, True])
def test_orthogonal_partial_cell_and_its_floating_point_rotation(rotate):
    rotation = (np.linalg.qr(np.random.default_rng(18).normal(size=(3, 3)))[0]
                if rotate else np.eye(3))
    cell = np.diag([0.7, 3.7, 5.1]) @ rotation
    flags = np.array([True, False, True])
    kernel = GaussianWindingKernel(FixedCellGeometry(cell, flags), 1.0)
    if not rotate:
        assert kernel.metadata["representations"] == ["reciprocal", "direct"]
    delta = np.random.default_rng(19).normal(size=(8, 3)) * 1.5
    result = kernel.evaluate(delta)
    expected = image_reference(delta, cell, flags, 1.0, cutoff=14)
    np.testing.assert_allclose(result.log_kernel, expected[0], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.mean_displacement, expected[1], rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(result.second_moment, expected[2], rtol=2e-13, atol=2e-13)


def test_near_orthogonal_real_shear_is_not_discarded_by_the_mic_fast_path():
    # This angle passes the existing MIC orthogonality tolerance. Multiplying
    # its omitted cross term by alpha would change log K by 1.14e-8, well above
    # the requested Gaussian tolerance. Every nonzero image is negligible.
    cell = np.array([[2.0, 0, 0], [1.8e-14, 2.0, 0], [0, 0, 2.0]])
    geometry = FixedCellGeometry(cell)
    assert geometry.triangular is None
    kernel = GaussianWindingKernel(geometry, 1e6)
    assert not kernel.separable
    delta = np.array([0.7, 0.9, 0.4])
    result = kernel.evaluate(delta)
    assert float(result.log_kernel) == pytest.approx(-1e6 * np.dot(delta, delta), abs=2e-9, rel=0)
    np.testing.assert_allclose(result.mean_displacement, delta, atol=2e-15, rtol=0)
    np.testing.assert_allclose(result.second_moment, np.outer(delta, delta), atol=3e-15, rtol=0)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("alpha", [0.4, 1.5])
def test_reported_bounds_cover_kernel_and_both_normalized_moments(skew, alpha):
    cell = np.array([[2.2, 0, 0], [0.5 if skew else 0, 2.0, 0],
                     [0.1 if skew else 0, 0.4 if skew else 0, 1.9]])
    geometry = FixedCellGeometry(cell)
    kernel = GaussianWindingKernel(geometry, alpha, tolerance=2e-3)
    delta = np.concatenate((np.random.default_rng(781).uniform(-0.5, 0.5, (40, 3)),
                            np.asarray(list(itertools.product((-0.5, 0.5), repeat=3))))) @ cell
    result = kernel.evaluate(delta)
    ref_log, ref_mean, ref_second = image_reference(delta, cell, np.ones(3, dtype=bool), alpha)
    relative = np.abs(np.expm1(ref_log - result.log_kernel))
    first = math.sqrt(alpha) * np.linalg.norm(ref_mean - result.mean_displacement, axis=1)
    second = alpha * np.linalg.norm(ref_second - result.second_moment, axis=(1, 2)) / (
        1 + alpha * np.linalg.norm(result.second_moment, axis=(1, 2)))
    for errors, bound in zip((relative, first, second), kernel.error_bounds):
        assert np.max(errors) <= bound + 4e-14


@pytest.mark.parametrize("flags", [(True, True, True), (True, False, True)])
def test_full_winding_force_strain_and_independent_bead_image_invariance(flags):
    cell = np.array([[2.5, 0, 0], [0.7, 2.1, 0], [0.2, 0.3, 2.8]])
    q = np.random.default_rng(137).normal(size=(3, 3, 3))
    kwargs = dict(cell=cell, pbc=flags, periodic_mode="full_winding", statistics="fermionic")
    result = bosonic_exchange(q, 1.1, 2.7, **kwargs)
    h = 2e-6
    force = np.empty_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += h
        minus[index] -= h
        force[index] = -(bosonic_exchange(plus, 1.1, 2.7, **kwargs).energy
                         - bosonic_exchange(minus, 1.1, 2.7, **kwargs).energy) / (2 * h)
    np.testing.assert_allclose(result.forces, force, rtol=3e-7, atol=8e-9)
    virial = np.empty((3, 3))
    for a, b in np.ndindex(3, 3):
        strain = np.zeros((3, 3))
        strain[a, b] = h
        plus, minus = np.eye(3) + strain, np.eye(3) - strain
        virial[a, b] = -(bosonic_exchange(q @ plus, 1.1, 2.7, **{**kwargs, "cell": cell @ plus}).energy
            - bosonic_exchange(q @ minus, 1.1, 2.7, **{**kwargs, "cell": cell @ minus}).energy) / (2 * h)
    np.testing.assert_allclose(result.virial, virial, rtol=4e-7, atol=1e-8)
    integer = np.random.default_rng(138).integers(-4, 5, size=q.shape)
    integer[..., np.logical_not(flags)] = 0
    translated = bosonic_exchange(q + integer @ cell, 1.1, 2.7, **kwargs)
    for key in ("energy", "forces", "virial", "spring_energy", "connections",
                "fermionic_sign", "fermionic_spring_energy_numerator"):
        np.testing.assert_allclose(getattr(result, key), getattr(translated, key), rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(np.trace(result.virial), -2 * result.spring_energy)


def test_small_box_has_zero_free_gas_energy_positive_moment_and_finite_negative_action():
    q = np.random.default_rng(161).uniform(0, 0.01, (4, 3, 3))
    cell, mass, beta = np.eye(3) * 0.01, 1.7, 2.3
    result = bosonic_exchange(q, mass, beta, cell=cell, periodic_mode="full_winding",
                              statistics="fermionic", winding_max_terms=3)
    assert result.energy < 0
    assert result.spring_energy == pytest.approx(3 * q.shape[1] * len(q) ** 2 / (2 * beta))
    assert abs(result.fermionic_sign) < 1e-15
    np.testing.assert_array_equal(result.forces, np.zeros_like(q))
    assert result.winding_diagnostics["terms_per_link"] == 3
    assert result.winding_diagnostics["representations"] == ["reciprocal"] * 3
    action = BosonicExchangePotential(FreePotential(), [mass] * 3, beta, 4,
                                      cell=cell, periodic_mode="full_winding", winding_max_terms=3)
    observed = action.estimators(q)
    assert observed["internal_energy_numerator_hartree"] == pytest.approx(0, abs=3e-14)


def test_full_winding_single_bead_single_particle_includes_nonzero_winding():
    cell = np.diag([1.5, 2.1, 3.2])
    q = np.ones((1, 1, 3)) * 0.37
    result = bosonic_exchange(q, 1.0, 2.0, cell=cell, periodic_mode="full_winding")
    assert result.energy < 0
    assert result.spring_energy > 0
    np.testing.assert_array_equal(result.forces, np.zeros_like(q))
    assert np.trace(result.virial) < 0


def test_large_image_count_evaluation_is_chunked_without_changing_moments():
    kernel = GaussianWindingKernel(FixedCellGeometry(
        [[2.1, 0, 0], [0.3, 2.0, 0], [0.2, 0.4, 2.2]]), 1.3)
    delta = np.random.default_rng(808).normal(size=(1200, 3))
    assert len(delta) * kernel.terms > kernel.metadata["working_pair_budget"]
    together = kernel.evaluate(delta)
    selected = np.array([0, 27, 507, 1199])
    separate = kernel.evaluate(delta[selected])
    np.testing.assert_array_equal(together.log_kernel[selected], separate.log_kernel)
    np.testing.assert_array_equal(together.mean_displacement[selected], separate.mean_displacement)
    np.testing.assert_array_equal(together.second_moment[selected], separate.second_moment)


def test_a_single_large_link_reports_its_true_working_pair_bound(monkeypatch):
    # The certified skew cutoff needs 300763 terms, exceeding the default
    # batch target while remaining below the explicitly requested term budget.
    cell = np.array([[1.0, 0, 0], [0.2, 1.0, 0], [0.1, 0.2, 40.0]])
    kernel = GaussianWindingKernel(FixedCellGeometry(cell), 1.0, max_terms=1000000)
    assert kernel.terms > kernel.metadata["link_batch_pair_target"]
    assert kernel.metadata["working_pair_budget"] == kernel.terms
    assert kernel.terms <= kernel.max_terms
    delta = np.array([[0.1, 0.2, 0.3], [-0.3, 0.1, -0.2]])
    calls = []
    original_exp = np.exp

    def record_batch(value, *args, **kwargs):
        if np.ndim(value) == 2:
            calls.append(np.shape(value))
        return original_exp(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np, "exp", record_batch)
        result = kernel.evaluate(delta)
    assert calls == [(1, kernel.terms), (1, kernel.terms)]
    assert max(math.prod(shape) for shape in calls) <= kernel.metadata["working_pair_budget"]
    expected = image_reference(delta, cell, np.ones(3, dtype=bool), 1.0)
    np.testing.assert_allclose(result.log_kernel, expected[0], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result.mean_displacement, expected[1], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(result.second_moment, expected[2], rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("options,match", [
    ({"winding_tolerance": True}, "winding_tolerance"),
    ({"winding_tolerance": "1e-12"}, "winding_tolerance"),
    ({"winding_tolerance": [1e-12]}, "winding_tolerance"),
    ({"winding_tolerance": 0}, "winding_tolerance"),
    ({"winding_tolerance": float("nan")}, "winding_tolerance"),
    ({"winding_tolerance": 1}, "winding_tolerance"),
    ({"winding_max_terms": True}, "winding_max_terms"),
    ({"winding_max_terms": 3.0}, "winding_max_terms"),
    ({"winding_max_terms": 0}, "winding_max_terms"),
    ({"winding_max_terms": 1000001}, "winding_max_terms"),
    ({"winding_max_terms": 2}, "exceeds winding_max_terms"),
])
def test_invalid_full_winding_controls_and_exhausted_budgets_fail(options, match):
    with pytest.raises(ValueError, match=match):
        bosonic_exchange(np.zeros((2, 2, 3)), 1.0, 2.0, cell=np.eye(3),
                         periodic_mode="full_winding", **options)


@pytest.mark.parametrize("mode,cell", [(None, None), ("minimum_image", np.eye(3))])
def test_winding_only_options_cannot_be_silently_ignored(mode, cell):
    with pytest.raises(ValueError, match="require periodic_mode='full_winding'"):
        bosonic_exchange(np.zeros((2, 1, 3)), 1, 2, cell=cell, periodic_mode=mode,
                         winding_tolerance=1e-8)


class CellFree:
    def __init__(self, cell):
        self.cell = np.array(cell, dtype=float)
        self.calls = []
        self.fail_next = False

    def set_cell(self, cell):
        self.calls.append(np.array(cell))
        self.cell = np.array(cell)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected physical cell failure")

    def evaluate(self, q):
        return PESResult(np.zeros(len(q)), np.zeros_like(q), virial=np.zeros((len(q), 3, 3)))


def test_cached_kernel_cell_update_is_atomic_and_centroid_api_stays_rejected():
    old = np.eye(3) * 0.04
    provider = CellFree(old)
    action = BosonicExchangePotential(provider, [1.0], 2.0, 2, cell=old,
        periodic_mode="full_winding", winding_max_terms=3)
    cache = action._winding_kernels
    with pytest.raises(PotentialError, match="centroid-only BZP"):
        action.set_cell(old * 1.1)
    with pytest.raises(ValueError, match="exceeds winding_max_terms"):
        action.set_cell(np.eye(3) * 4, barostat_coupling="all_beads")
    assert not provider.calls
    assert action._winding_kernels is cache
    provider.fail_next = True
    with pytest.raises(RuntimeError, match="injected"):
        action.set_cell(old * 1.1, barostat_coupling="all_beads")
    np.testing.assert_array_equal(action.cell, old)
    np.testing.assert_array_equal(provider.cell, old)
    assert action._winding_kernels is cache
    action.set_cell(old * 1.1, barostat_coupling="all_beads")
    np.testing.assert_array_equal(action.cell, old * 1.1)
    np.testing.assert_array_equal(provider.cell, action.cell)
    assert action._winding_kernels is not cache
    assert action.metadata["winding_kernels"][0]["terms_per_link"] == 3


def test_multiple_groups_and_unlisted_particles_share_full_sampling_contract():
    cell = np.diag([2.5, 2.3, 2.0])
    masses = np.array([1.1, 1.1, 1.7, 1.7, 2.0])
    q = np.random.default_rng(51).normal(size=(3, 5, 3))
    action = BosonicExchangePotential(CellFree(cell), masses, 2.7, 3,
        groups=[[0, 1], [2, 3]], statistics="fermionic", cell=cell,
        periodic_mode="full_winding")
    cache = action._winding_kernels
    full = action.evaluate_sampling(q)
    correction = action.evaluate(q)
    raw_delta = q - np.roll(q, -1, axis=0)
    raw_energy = 0.5 * (len(q) / action.beta) ** 2 * np.sum(masses[None, :, None] * raw_delta ** 2)
    raw_force = -(len(q) / action.beta) ** 2 * masses[None, :, None] * (
        2 * q - np.roll(q, 1, axis=0) - np.roll(q, -1, axis=0))
    raw_virial = -(len(q) / action.beta) ** 2 * np.einsum("n,pni,pnj->ij", masses, raw_delta, raw_delta)
    np.testing.assert_allclose(full.energy.sum(), correction.energy.sum() + raw_energy, atol=1e-13)
    np.testing.assert_allclose(full.forces, correction.forces + raw_force, atol=1e-13)
    np.testing.assert_allclose(full.virial.sum(axis=0), correction.virial.sum(axis=0) + raw_virial, atol=1e-13)
    results = [bosonic_exchange(q[:, ids], masses[ids[0]], action.beta, cell=cell,
        periodic_mode="full_winding", statistics="fermionic") for ids in ([0, 1], [2, 3], [4])]
    sign = math.prod(result.fermionic_sign for result in results)
    spring_numerator = sum(result.fermionic_spring_energy_numerator * math.prod(
        other.fermionic_sign for j, other in enumerate(results) if j != i)
        for i, result in enumerate(results))
    observed = action.estimators(q, volume=np.linalg.det(cell), require_pressure=True)
    assert observed["sign"] == pytest.approx(sign)
    assert observed["kinetic_energy_numerator_hartree"] == pytest.approx(
        3 * len(q) * len(masses) / (2 * action.beta) * sign - spring_numerator / len(q))
    assert action._winding_kernels is cache
