"""Independent checks of the finite-P surface-flux child estimator."""
import json

import numpy as np
import pytest

from benchmarks.h3 import quantum_surface_recrossing as recrossing
from pyrpmd.checkpoint import atomic_savez


def surface_parent(beads=16):
    r, t = 1.76, 1.91
    cosine = 1 - t * t / 2
    q = np.array([[r, 0., 0.], [0., 0., 0.],
                  [r * cosine, r * np.sqrt(1 - cosine * cosine), 0.]])
    q -= q.mean(axis=0)
    return np.repeat(q[None], beads, axis=0)


@pytest.mark.parametrize("beads", [1, 8, 128])
def test_maxwell_flux_has_no_spurious_bead_factor(beads):
    q = surface_parent(beads)
    metric, _ = recrossing.flux_velocity(q, np.zeros_like(q))
    values = []
    for index in range(1024):
        p = recrossing.maxwell_momenta(98271, index, 0, beads)
        fs, velocity = recrossing.flux_velocity(q, p)
        assert fs == metric
        values.append(velocity / fs)
    values = np.array(values)
    # v/fs is N(0, 2*pi), hence E|v/fs|=2 for every bead number.
    assert abs(values.mean()) < .25
    assert abs(np.mean(abs(values)) - 2.) < .16
    assert abs(values.var() / (2 * np.pi) - 1.) < .15


def test_flux_velocity_matches_actual_centroid_motion():
    q = surface_parent(32)
    p = recrossing.maxwell_momenta(98271, 2, 1, 32)
    fs, velocity = recrossing.flux_velocity(q, p)
    surface = recrossing.surface_coordinate()
    dt = 1e-3
    displacement = dt * p / recrossing.MASS[None, :, None]
    finite_difference = float((surface.value(q + displacement) - surface.value(q - displacement))[0] / (2 * dt))
    assert velocity == pytest.approx(finite_difference, rel=2e-8, abs=2e-12)
    assert fs > 0


def test_antithetic_pair_exact_initial_positive_flux_limit():
    for velocity in (-.4, .3):
        values = np.array([[velocity, -velocity], [-velocity, velocity]])
        numerator, denominator = recrossing.paired_flux(velocity, .2, values)
        np.testing.assert_allclose(numerator / denominator, [1., -1.], atol=0)


def test_internal_and_total_com_momenta_do_not_change_centroid_flux():
    q = surface_parent(32)
    p = recrossing.maxwell_momenta(98271, 3, 0, 32)
    internal = np.random.default_rng(17).normal(size=p.shape)
    internal -= internal.mean(axis=0, keepdims=True)
    shifted = p + internal + np.array([3., -7., 2.])[None, None]
    np.testing.assert_allclose(recrossing.flux_velocity(q, shifted), recrossing.flux_velocity(q, p), atol=5e-17)


def test_equivalent_channel_permutation_preserves_metric_and_velocity():
    q = surface_parent(16)
    p = recrossing.maxwell_momenta(17, 6, 0, 16)
    reference = recrossing.flux_velocity(q, p)
    np.testing.assert_allclose(recrossing.flux_velocity(q[:, [0, 2, 1]], p[:, [0, 2, 1]]),
                               reference, rtol=2e-14, atol=1e-16)


def test_parent_keyed_momenta_are_independent_of_job_order():
    original = [recrossing.maxwell_momenta(721301, i, 1, 8) for i in range(8)]
    reversed_run = {i: recrossing.maxwell_momenta(721301, i, 1, 8) for i in reversed(range(8))}
    for i in range(8):
        np.testing.assert_array_equal(original[i], reversed_run[i])
        assert not np.array_equal(original[i], recrossing.maxwell_momenta(721301, i, 0, 8))
    assert not np.array_equal(original[0], recrossing.maxwell_momenta(721303, 0, 1, 8))


def test_chunk_rejects_other_protocol_and_other_parent_inventory(tmp_path):
    path = tmp_path / "children.npz"
    atomic_savez(path, kind="h3.quantum-surface-children", fingerprint="registered",
                 parent_index=np.array([0, 1]), observable=np.array([.3, .4]))
    saved = recrossing.read_chunk(path, "registered", np.array([0, 1]))
    np.testing.assert_array_equal(saved["observable"], [.3, .4])
    with pytest.raises(ValueError, match="identity"):
        recrossing.read_chunk(path, "changed", np.array([0, 1]))
    with pytest.raises(ValueError, match="inventory"):
        recrossing.read_chunk(path, "registered", np.array([1, 0]))


def test_1fs_boundary_minimum_checks_every_bead_and_pair():
    q = surface_parent(8)
    q[3, 2] = q[3, 0] + [.4, 0., 0.]
    assert recrossing.minimum_pair(q) == pytest.approx(.4)


def test_off_surface_parent_is_rejected():
    q = surface_parent(8)
    q[:, 0, 0] += .01
    with pytest.raises(ValueError, match="dividing surface"):
        recrossing.flux_velocity(q, np.zeros_like(q))
