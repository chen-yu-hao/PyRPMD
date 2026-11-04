"""Partitioned IBI targets, conservative forces, portable identity and failures."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez
from pyrpmd.potentials import FreePotential
from pyrpmd.potentials.base import PESResult, Potential, PotentialError, check_potential
from pyrpmd.qcmd import (
    FQCMDModel, FQCMDPotential, HCMDModel, HCMDPotential, TriatomicQuasiCentroid,
    coordinate_histograms, fqcmd_base_identity, hcmd_coordinate_histograms,
    update_hcmd_model,
)
from pyrpmd.system import System


def system_and_model(*, nonzero=True, molecules=2, partitions=None, provenance=None):
    local = np.array([[0., 0., 0.], [1.83, 0., 0.], [-.5, 1.7, 0.]])
    q = np.concatenate([local + [0., 0., 6. * i] for i in range(molecules)])
    system = System(("O", "H", "H") * molecules, np.tile([16., 1., 1.], molecules), q)
    geometry = TriatomicQuasiCentroid(np.arange(3 * molecules).reshape(-1, 3), system.masses)
    groups = [[2 * i, 2 * i, 2 * i + 1] for i in range(molecules)]
    edges, corrections = [], []
    for molecule in range(molecules):
        grids = (np.linspace(.2, 3.8, 37), np.linspace(.2, 2.9, 28))
        centers = [.5 * (edge[1:] + edge[:-1]) for edge in grids]
        factor = molecule + 1
        values = ([factor * .3 * (centers[0] - 1.6) ** 2,
                   factor * .2 * (centers[1] - 1.8) ** 2]
                  if nonzero else [np.zeros_like(center) for center in centers])
        edges.extend(grids)
        corrections.extend(values)
    correction = FQCMDModel(geometry, groups, edges, corrections, beta=2.,
                            base_identity=fqcmd_base_identity({"provider": "free"}, system),
                            provenance=provenance)
    if partitions is None:
        partitions = {"f-qcmd": list(range(molecules // 2)), "f-cmd": list(range(molecules // 2, molecules))}
    return system, HCMDModel(correction, partitions)


def test_hybrid_targets_match_independent_rotating_ring_reference():
    system, model = system_and_model()
    angle = .5
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    ring = np.stack([system.geometry @ rotation, system.geometry @ rotation.T])[None]
    counts = hcmd_coordinate_histograms(ring, model, bead_average=True)
    # Opposite rotations leave each bead bond unchanged. Averaging Cartesian
    # positions contracts planar bonds by cos(angle), a hand-computable check.
    bonds = np.linalg.norm(system.geometry[1:3] - system.geometry[0], axis=1)
    np.testing.assert_array_equal(counts[0], np.histogram(bonds, model.edges[0])[0])
    np.testing.assert_array_equal(counts[2], np.histogram(bonds * np.cos(angle), model.edges[2])[0])
    assert not np.array_equal(counts[0], counts[2])
    assert [count.sum() for count in counts] == [2, 1, 2, 1]
    full_qc = coordinate_histograms(ring, model.geometry, model.groups, model.edges, bead_average=True)
    full_centroid = coordinate_histograms(ring.mean(1), model.geometry, model.groups, model.edges, bead_average=False)
    for i in (0, 1):
        np.testing.assert_array_equal(counts[i], full_qc[i])
    for i in (2, 3):
        np.testing.assert_array_equal(counts[i], full_centroid[i])


def test_only_selected_molecules_require_a_nondegenerate_cartesian_centroid():
    system, model = system_and_model()
    ring = np.broadcast_to(system.geometry, (1, 2, 6, 3)).copy()
    # The f-QCMD molecule has valid bead bond/angle means even when opposing
    # bead orientations collapse its Cartesian centroid; do not evaluate the
    # unused f-CMD mapping for this molecule.
    ring[0, 1, :3, :2] *= -1
    counts = hcmd_coordinate_histograms(ring, model, bead_average=True)
    assert [count.sum() for count in counts] == [2, 1, 2, 1]
    ring[0, 1, 3:, :2] *= -1
    with pytest.raises(ValueError, match="singular bond"):
        hcmd_coordinate_histograms(ring, model, bead_average=True)


def test_single_bead_target_equals_classical_counts_and_same_method_can_pool():
    system, model = system_and_model(molecules=4)
    original = model.correction_model
    pooled = FQCMDModel(original.geometry, [[0, 0, 1], [0, 0, 1], [2, 2, 3], [2, 2, 3]],
                        original.edges[:4], original.corrections[:4], beta=original.beta,
                        base_identity=original.base_identity)
    model = HCMDModel(pooled, {"f-qcmd": [0, 1], "f-cmd": [2, 3]})
    q = np.stack([system.geometry, system.geometry + [1., -2., 3.]])
    a = hcmd_coordinate_histograms(q[:, None], model, bead_average=True)
    b = hcmd_coordinate_histograms(q, model, bead_average=False)
    for actual, expected in zip(a, b):
        np.testing.assert_array_equal(actual, expected)
    assert [count.sum() for count in a] == [8, 4, 8, 4]


@pytest.mark.parametrize("partitions", [
    None, [], {"f-qcmd": [0]}, {"f-qcmd": [0], "f-cmd": [1], "classical": []},
    {"f-qcmd": [], "f-cmd": [0, 1]}, {"f-qcmd": [0, 1], "f-cmd": []},
    {"f-qcmd": [0], "f-cmd": [0]}, {"f-qcmd": [0], "f-cmd": [1, 1]},
    {"f-qcmd": [-1], "f-cmd": [1]}, {"f-qcmd": [2], "f-cmd": [1]},
    {"f-qcmd": [False], "f-cmd": [1]}, {"f-qcmd": [np.bool_(False)], "f-cmd": [1]},
    {"f-qcmd": [0.], "f-cmd": [1]}, {"f-qcmd": ["0"], "f-cmd": [1]},
    {"f-qcmd": [[0]], "f-cmd": [1]}, {"f-qcmd": 0, "f-cmd": [1]},
])
def test_partition_errors_are_explicit(partitions):
    _, model = system_and_model()
    with pytest.raises(ValueError, match="h-CMD partition"):
        HCMDModel(model.correction_model, partitions)


def test_missing_molecule_and_cross_method_coordinate_pooling_fail():
    _, model = system_and_model(molecules=3)
    with pytest.raises(ValueError, match="cover every molecule"):
        HCMDModel(model.correction_model, {"f-qcmd": [0], "f-cmd": [1]})
    original = model.correction_model
    pooled = FQCMDModel(original.geometry, [[0, 0, 1]] * 3, original.edges[:2],
                        original.corrections[:2], beta=original.beta, base_identity=original.base_identity)
    with pytest.raises(ValueError, match="cannot be shared"):
        HCMDModel(pooled, {"f-qcmd": [0], "f-cmd": [1, 2]})


class CoupledPhysicalPotential(Potential):
    """Independent original-PES cross-molecule energy and virial reference."""

    def __init__(self):
        self.calls = 0
        self.closed = False

    def evaluate(self, q):
        self.calls += 1
        displacement = q[:, 3] - q[:, 0]
        energy = .15 * np.sum(displacement ** 2, axis=1)
        forces = np.zeros_like(q)
        forces[:, 0] = .3 * displacement
        forces[:, 3] = -.3 * displacement
        virial = np.einsum("bni,bnj->bij", q, forces)
        return PESResult(energy, forces, messages=("physical",) * len(q), virial=virial)

    def close(self):
        self.closed = True


def test_combined_conservative_force_keeps_physical_coupling_once_and_full_contract():
    system, model = system_and_model()
    q = np.stack([system.geometry, .95 * system.geometry + [.1, .3, 0.]])
    base = CoupledPhysicalPotential()
    wrapper = HCMDPotential(base, model)
    actual = wrapper.evaluate(q)
    assert base.calls == 1
    physical = base.evaluate(q)
    local = q.reshape(2, 2, 3, 3)
    a, b = local[:, :, 1] - local[:, :, 0], local[:, :, 2] - local[:, :, 0]
    r1, r2 = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    angle = np.arccos(np.sum(a * b, axis=-1) / (r1 * r2))
    correction = np.sum(np.array([1., 2.]) * (
        .3 * ((r1 - 1.6) ** 2 + (r2 - 1.6) ** 2) + .2 * (angle - 1.8) ** 2), axis=1)
    np.testing.assert_allclose(actual.energy, physical.energy + correction, atol=1e-14)
    assert check_potential(wrapper, q)["max_absolute_force_error"] < 1e-9
    extra_force = actual.forces - physical.forces
    np.testing.assert_allclose(actual.virial, physical.virial + np.einsum("bni,bnj->bij", q, extra_force))
    np.testing.assert_array_equal(actual.status, physical.status)
    assert actual.messages == physical.messages
    wrapper.close()
    assert not base.closed  # the physical PES is borrowed


def test_zero_correction_exactly_preserves_physical_potential():
    system, model = system_and_model(nonzero=False)
    base = CoupledPhysicalPotential()
    q = np.stack([system.geometry, system.geometry * .95])
    expected = base.evaluate(q)
    actual = HCMDPotential(base, model).evaluate(q)
    for field in ("energy", "forces", "virial", "status"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_hybrid_ibi_matches_independent_boltzmann_inversion_and_records_full_parent():
    _, model = system_and_model(nonzero=False)
    target, sampled, expected = [], [], []
    for i, (x, edge) in enumerate(zip(model.centers, model.edges)):
        measure = x ** 2 if i % 2 == 0 else np.sin(x)
        base = .07 * (x - 1.6) ** 2
        delta = (i + 1) * .06 * (x - 1.5) ** 2 - .03 * x
        sampled.append(measure * np.exp(-model.beta * base) * np.diff(edge))
        target.append(measure * np.exp(-model.beta * (base + delta)) * np.diff(edge))
        expected.append(delta - delta[len(delta) // 2])
    fitted, diagnostics = update_hcmd_model(model, target, sampled, polynomial_degree=2)
    for actual, reference in zip(fitted.corrections, expected):
        np.testing.assert_allclose(actual, reference, atol=3e-15)
    history = fitted.provenance["ibi_history"][-1]
    assert history["hcmd_parent_model_sha256"] == model.fingerprint
    assert history["parent_model_sha256"] == model.correction_model.fingerprint
    assert history["hcmd_partitions"] == model.partitions
    assert diagnostics["sampled_model_sha256"] == model.fingerprint
    assert diagnostics["updated_model_sha256"] == fitted.fingerprint
    assert diagnostics["partitions"] == model.partitions
    assert diagnostics["method"] == "h-cmd"
    assert not diagnostics["converged"] and not diagnostics["physical_converged"]


def test_invalid_update_leaves_model_unchanged():
    _, model = system_and_model()
    original = model.to_dict()
    counts = [np.ones(len(edge) - 1) for edge in model.edges]
    counts[0][0] = 0
    with pytest.raises(ValueError, match="zero-density"):
        update_hcmd_model(model, counts, counts)
    assert model.to_dict() == original


def test_portable_roundtrip_partition_identity_and_input_ownership(tmp_path):
    system, model = system_and_model(molecules=4, provenance={"seed": [17, 19], "beads": 32})
    path = tmp_path / "hcmd.npz"
    model.save(path)
    restored = HCMDModel.load(path)
    assert restored.to_dict() == model.to_dict()
    assert restored.fingerprint == model.fingerprint
    permuted = HCMDModel(model.correction_model, {"f-cmd": [3, 2], "f-qcmd": [1, 0]})
    assert permuted.fingerprint == model.fingerprint
    changed = HCMDModel(model.correction_model, {"f-cmd": [0, 1], "f-qcmd": [2, 3]})
    assert changed.fingerprint != model.fingerprint
    q = system.geometry[None]
    np.testing.assert_array_equal(HCMDPotential(FreePotential(), model).evaluate(q).forces,
                                  HCMDPotential(FreePotential(), restored).evaluate(q).forces)
    model.partitions["f-cmd"].append(0)
    model.provenance["seed"].append(23)
    assert model.to_dict() == restored.to_dict()
    with pytest.raises(AttributeError, match="immutable"):
        model.beta = 8.
    with pytest.raises(ValueError):
        model.groups[0, 0] = 8
    wrapper = HCMDPotential(FreePotential(), model)
    with pytest.raises(AttributeError):
        wrapper.model = changed


def test_hybrid_owns_correction_and_rejects_modified_parameters(tmp_path):
    _, model = system_and_model()
    original = model.correction_model
    owned = HCMDModel(original, model.partitions)
    definition = owned.to_dict()
    original.corrections[0].flags.writeable = True
    original.corrections[0][2] += .01
    assert owned.to_dict() == definition
    for operation in (lambda: model.save(tmp_path / "bad.npz"),
                      lambda: HCMDPotential(FreePotential(), model),
                      lambda: HCMDModel(original, model.partitions)):
        with pytest.raises(ValueError, match="parameters changed"):
            operation()
    assert not (tmp_path / "bad.npz").exists()


@pytest.mark.parametrize("change", ["partition", "table", "mass", "temperature", "provenance"])
def test_embedded_model_changes_cannot_reuse_saved_identity(tmp_path, change):
    _, model = system_and_model()
    path = tmp_path / "hcmd.npz"
    model.save(path)
    definition = model.to_dict()
    correction = definition["correction_model"]
    if change == "partition":
        definition["partitions"] = {"f-qcmd": [1], "f-cmd": [0]}
    elif change == "table":
        correction["corrections"][0][2] += .01
    elif change == "mass":
        correction["masses"][1] *= 2
    elif change == "temperature":
        correction["beta_au"] *= 2
    else:
        correction["provenance"]["new"] = "training"
    # Recalculate storage integrity but retain the original scientific hash.
    atomic_savez(path, kind="PyRPMD.hcmd-model.v1", metadata=json.dumps(definition), fingerprint=model.fingerprint)
    with pytest.raises(ValueError, match="scientific fingerprint"):
        HCMDModel.load(path)


@pytest.mark.parametrize("change", ["mapping", "format", "extra", "inner", "inventory"])
def test_unknown_schemas_and_inventory_fail(tmp_path, change):
    _, model = system_and_model()
    path = tmp_path / "hcmd.npz"
    definition = model.to_dict()
    extra = {}
    if change == "mapping":
        definition["mapping"] = "other"
    elif change == "format":
        definition["format"] = "PyRPMD.hcmd-model.v3"
    elif change == "extra":
        definition["unknown"] = True
    elif change == "inner":
        definition["correction_model"]["interpolation"] = "other"
    else:
        extra["extra_array"] = np.zeros(1)
    atomic_savez(path, kind="PyRPMD.hcmd-model.v1", metadata=json.dumps(definition),
                 fingerprint=model.fingerprint, **extra)
    with pytest.raises(ValueError):
        HCMDModel.load(path)


def test_corrupt_storage_is_rejected_before_model_use(tmp_path):
    _, model = system_and_model()
    path = tmp_path / "hcmd.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["fingerprint"] = np.asarray("a" * 64)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="integrity"):
        HCMDModel.load(path)


def test_same_system_temperature_and_original_pes_are_required():
    system, model = system_and_model()
    model.validate_system(system, model.beta, base_identity=model.base_identity)
    altered = copy.deepcopy(system)
    altered.masses[1] *= 2
    with pytest.raises(ValueError, match="h-CMD.*masses"):
        model.validate_system(altered, model.beta, base_identity=model.base_identity)
    with pytest.raises(ValueError, match="h-CMD.*temperature"):
        model.validate_system(system, 2 * model.beta, base_identity=model.base_identity)
    with pytest.raises(ValueError, match="h-CMD.*potential"):
        model.validate_system(system, model.beta, base_identity="a" * 64)
    for attribute, value in (("cell", np.eye(3)), ("fixed_atoms", (0,)),
                             ("bond_constraints", ((0, 1, 1.83),))):
        altered = copy.deepcopy(system)
        setattr(altered, attribute, value)
        with pytest.raises(ValueError, match="h-CMD"):
            model.validate_system(altered, model.beta, base_identity=model.base_identity)


@pytest.mark.parametrize("metadata", [
    {"execution_device": "cuda:0"}, {"boundary_conditions": "periodic"},
    {"centroid_pmf_beta_au": 2.}, {"quasicentroid_pmf_beta_au": 2.},
    {"hybrid_centroid_pmf_beta_au": 2.},
])
def test_unsupported_base_combinations_fail(metadata):
    _, model = system_and_model()
    base = FreePotential()
    base.metadata = metadata
    with pytest.raises(ValueError, match="h-CMD"):
        HCMDPotential(base, model)


def test_hybrid_cannot_be_quantum_corrected_twice_and_failed_base_remains_failure():
    system, model = system_and_model()
    hybrid = HCMDPotential(FreePotential(), model)
    with pytest.raises(ValueError, match="already fitted PMF"):
        HCMDPotential(hybrid, model)
    with pytest.raises(ValueError, match="already fitted PMF"):
        FQCMDPotential(hybrid, model.correction_model)

    class Failed(Potential):
        def evaluate(self, q):
            return PESResult(np.zeros(len(q)), np.zeros_like(q),
                             status=np.full(len(q), 2, dtype=int), messages=("failed",) * len(q))

    with pytest.raises(PotentialError, match="status=2: failed"):
        HCMDPotential(Failed(), model).evaluate(system.geometry[None])


def test_histogram_and_force_domain_failures_are_explicit():
    system, model = system_and_model()
    q = system.geometry[None].copy()
    q[:, 1, 0] = 6.
    with pytest.raises(ValueError, match="histogram domain"):
        hcmd_coordinate_histograms(q[:, None], model, bead_average=True)
    with pytest.raises(ValueError, match="fitted domain"):
        HCMDPotential(FreePotential(), model).evaluate(q)
    for invalid in (1, "yes", None):
        with pytest.raises(ValueError, match="bead_average"):
            hcmd_coordinate_histograms(q, model, bead_average=invalid)
    for malformed in (system.geometry, q, np.zeros((1, 0, 6, 3)), np.zeros((1, 1, 3, 3))):
        with pytest.raises(ValueError, match="positions"):
            hcmd_coordinate_histograms(malformed, model, bead_average=True)

