"""Reference labels are distinct from predictions and survive failed batches."""

from dataclasses import FrozenInstanceError
import hashlib
import json
import socket
import struct
import threading

import numpy as np
import pytest

from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.potentials.base import PESResult, PotentialError
from pyrpmd.potentials.lifecycle import LabelledDataset, label_candidates


IDENTITY = {"provider": "analytic", "method": "double-well", "version": 1, "barrier": 0.7}


class DoubleWellReference:
    def __init__(self):
        self.calls = []

    def evaluate(self, q):
        self.calls.append(len(q))
        return PESResult(.7 * np.sum((q*q - 1)**2, axis=(1, 2)), -2.8*q*(q*q - 1))


def positions(count=7, atoms=2):
    return np.random.default_rng(817).normal(size=(count, atoms, 3))


def dataset(q=None, **kwargs):
    q = positions() if q is None else q
    return label_candidates(DoubleWellReference(), q, reference_identity=IDENTITY, **kwargs)


def test_batch_labels_match_analytic_energy_and_independent_finite_difference():
    q = positions()
    reference = DoubleWellReference()
    labelled = label_candidates(reference, q, reference_identity=IDENTITY, batch_size=3)
    assert reference.calls == [3, 3, 1]
    np.testing.assert_allclose(labelled.energy, .7*np.sum((q*q - 1)**2, axis=(1, 2)))
    for atom in range(q.shape[1]):
        for axis in range(3):
            plus, minus = q.copy(), q.copy()
            plus[:, atom, axis] += 1e-6
            minus[:, atom, axis] -= 1e-6
            ep = .7*np.sum((plus*plus - 1)**2, axis=(1, 2))
            em = .7*np.sum((minus*minus - 1)**2, axis=(1, 2))
            np.testing.assert_allclose(labelled.forces[:, atom, axis], -(ep-em)/2e-6, rtol=2e-8, atol=1e-8)
    np.testing.assert_array_equal(labelled.status, 0)


def test_batch_and_member_buffer_reuse_cannot_overwrite_previous_labels():
    class Reused:
        def __init__(self):
            self.energy, self.forces = np.empty(2), np.empty((2, 2, 3))

        def evaluate(self, q):
            self.energy[:] = np.sum(q*q, axis=(1, 2)) / 2
            self.forces[:] = -q
            return PESResult(self.energy, self.forces)

    q = positions(6)
    labelled = label_candidates(Reused(), q, reference_identity="harmonic-v1", batch_size=2)
    np.testing.assert_array_equal(labelled.forces, -q)
    np.testing.assert_array_equal(labelled.energy, np.sum(q*q, axis=(1, 2))/2)


def test_partial_statuses_and_failed_batches_are_retained_without_retry():
    class Reference:
        def __init__(self):
            self.calls = []

        def evaluate(self, q):
            self.calls.append(len(q))
            if len(self.calls) == 2:
                raise RuntimeError("SCF did not converge")
            result = DoubleWellReference().evaluate(q)
            result.status[-1] = 9
            result.messages = tuple([""] * (len(q)-1) + ["root not converged"])
            return result

    q, reference = positions(5), Reference()
    labelled = label_candidates(reference, q, reference_identity=IDENTITY, batch_size=3,
                                provenance={"round": 2, "model_hash": "previous"})
    assert reference.calls == [3, 2]
    np.testing.assert_array_equal(labelled.positions, q)
    np.testing.assert_array_equal(labelled.status, [0, 0, 9, 1, 1])
    assert labelled.messages == ("", "", "root not converged", "RuntimeError: SCF did not converge",
                                 "RuntimeError: SCF did not converge")
    assert np.isnan(labelled.energy[2:]).all() and np.isnan(labelled.forces[2:]).all()
    training = labelled.successful()
    assert len(training) == 2 and training.reference_hash == labelled.reference_hash
    assert training.provenance == ({"round": 2, "model_hash": "previous"},) * 2
    np.testing.assert_array_equal(training.positions, q[:2])


def test_process_interruption_is_not_converted_to_failed_labels():
    class Interrupted:
        def evaluate(self, q):
            raise KeyboardInterrupt("stop")

    with pytest.raises(KeyboardInterrupt, match="stop"):
        label_candidates(Interrupted(), positions(), reference_identity=IDENTITY)


@pytest.mark.parametrize("bad", [
    lambda q: PESResult(np.zeros(len(q)+1), np.zeros_like(q)),
    lambda q: PESResult(np.full(len(q), np.nan), np.zeros_like(q)),
    lambda q: PESResult(np.zeros(len(q)), np.zeros_like(q), messages=[12]*len(q)),
    lambda q: "not a PES result",
])
def test_malformed_results_fail_explicitly_instead_of_becoming_reference_labels(bad):
    class Broken:
        evaluate = staticmethod(bad)

    with pytest.raises(PotentialError):
        label_candidates(Broken(), positions(), reference_identity=IDENTITY)


@pytest.mark.parametrize("status", [.5, 2**40, -1])
def test_mutated_status_cannot_be_truncated_into_success(status):
    class Broken:
        def evaluate(self, q):
            result = DoubleWellReference().evaluate(q)
            result.status = np.full(len(q), status)
            return result

    with pytest.raises(PotentialError, match="status"):
        label_candidates(Broken(), positions(), reference_identity=IDENTITY)


def test_dataset_hash_is_stable_but_covers_reference_provenance_and_labels():
    q = positions()
    first = dataset(q, batch_size=1, provenance={"source": "trajectory", "step": 1})
    second = label_candidates(DoubleWellReference(), q, reference_identity=dict(reversed(list(IDENTITY.items()))),
                              batch_size=7, provenance={"step": 1, "source": "trajectory"})
    assert first.sha256 == second.sha256 and first.reference_hash == second.reference_hash
    changed = label_candidates(DoubleWellReference(), q, reference_identity={**IDENTITY, "version": 2},
                               provenance=first.provenance)
    assert changed.sha256 != first.sha256 and changed.reference_hash != first.reference_hash
    assert dataset(q, provenance={"source": "trajectory", "step": 2}).sha256 != first.sha256
    changed_energy = first.energy.copy()
    changed_energy[0] += .1
    changed = LabelledDataset(q, changed_energy, first.forces, reference_identity=IDENTITY,
                              provenance=first.provenance)
    assert changed.sha256 != first.sha256 and changed.reference_hash == first.reference_hash


@pytest.mark.parametrize("identity", [None, "", " ", {}, [], {1: "method"}, {"method": float("nan")}, {"method": object()}])
def test_bad_identity_is_rejected_before_reference_call(identity):
    reference = DoubleWellReference()
    with pytest.raises(ValueError, match="reference_identity"):
        label_candidates(reference, positions(), reference_identity=identity)
    assert not reference.calls


@pytest.mark.parametrize("provenance", ["source", 5, [{"a": 1}], {"step": float("inf")}, {1: "source"}])
def test_bad_provenance_is_rejected_before_reference_call(provenance):
    reference = DoubleWellReference()
    with pytest.raises(ValueError, match="provenance"):
        label_candidates(reference, positions(), reference_identity=IDENTITY, provenance=provenance)
    assert not reference.calls


def test_dataset_owns_immutable_coordinates_labels_and_identity():
    q = positions()
    identity = {"method": {"name": "double well"}}
    provenance = {"round": 0}
    result = DoubleWellReference().evaluate(q)
    labelled = LabelledDataset(q, result.energy, result.forces, reference_identity=identity, provenance=provenance)
    expected_hash = labelled.sha256
    q[:] = 123
    result.energy[:] = 456
    result.forces[:] = 789
    identity["method"]["name"] = "changed"
    provenance["round"] = 7
    labelled.reference_identity["method"]["name"] = "changed again"
    labelled.provenance[0]["round"] = 8
    assert labelled.sha256 == expected_hash
    assert labelled.reference_identity == {"method": {"name": "double well"}}
    assert labelled.provenance == ({"round": 0},) * len(labelled)
    for value in (labelled.positions, labelled.energy, labelled.forces, labelled.status):
        with pytest.raises(ValueError):
            value.flat[0] = 42
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        labelled.sha256 = "changed"


def test_provider_input_mutation_does_not_change_recorded_geometries():
    class Mutation:
        def evaluate(self, q):
            result = DoubleWellReference().evaluate(q)
            q[:] = 999
            return result

    q = positions()
    labelled = label_candidates(Mutation(), q, reference_identity=IDENTITY, batch_size=2)
    np.testing.assert_array_equal(q, labelled.positions)
    np.testing.assert_array_equal(labelled.energy, dataset(q).energy)


def test_save_load_append_and_empty_success_subset(tmp_path):
    q = positions()
    first, second = dataset(q[:3]), dataset(q[3:])
    merged = first.append(second)
    assert merged.sha256 == dataset(q).sha256
    path = tmp_path / "labels.npz"
    merged.save(path)
    restored = LabelledDataset.load(path)
    assert restored.sha256 == merged.sha256
    np.testing.assert_array_equal(restored.positions, q)
    failed = LabelledDataset(q, np.zeros(len(q)), np.zeros_like(q), np.ones(len(q), dtype=int),
                             ["failed"] * len(q), reference_identity=IDENTITY)
    failed.save(path)
    restored = LabelledDataset.load(path)
    assert restored.sha256 == failed.sha256
    assert np.isnan(restored.energy).all()
    empty = restored.successful()
    assert len(empty) == 0 and empty.positions.shape == (0, 2, 3)
    empty.save(path)
    assert LabelledDataset.load(path).sha256 == empty.sha256
    assert empty.append(merged).sha256 == merged.sha256
    with pytest.raises(ValueError, match="atom count"):
        merged.append(dataset(positions(atoms=1)))
    other = label_candidates(DoubleWellReference(), q, reference_identity="different method")
    with pytest.raises(ValueError, match="reference identity"):
        merged.append(other)


@pytest.mark.parametrize("field", ["energy", "reference_identity", "provenance", "sha256"])
def test_semantic_hash_detects_changes_even_if_archive_envelope_is_rebuilt(tmp_path, field):
    path = tmp_path / "labels.npz"
    dataset().save(path)
    with load_checkpoint(path, require_integrity=True) as stored:
        payload = dict(stored.items())
    if field == "energy":
        payload[field][0] += 1
    elif field == "reference_identity":
        payload[field] = json.dumps({**IDENTITY, "version": 99})
    elif field == "provenance":
        payload[field] = np.asarray(['{"source":"changed"}'] * len(payload[field]))
    else:
        payload[field] = "0" * 64
    atomic_savez(path, kind="PyRPMD.labelled-dataset.v1", **payload)
    with pytest.raises(ValueError, match="hash"):
        LabelledDataset.load(path)


def test_legacy_unchecked_dataset_is_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, positions=positions())
    with pytest.raises(ValueError, match="integrity"):
        LabelledDataset.load(path)


def test_load_does_not_silently_replace_corrupted_failure_values(tmp_path):
    path, q = tmp_path / "failed.npz", positions()
    LabelledDataset(q, np.zeros(len(q)), np.zeros_like(q), np.ones(len(q), dtype=int),
                    reference_identity=IDENTITY).save(path)
    with load_checkpoint(path, require_integrity=True) as data:
        payload = dict(data.items())
    payload["energy"][0] = 123
    atomic_savez(path, kind="PyRPMD.labelled-dataset.v1", **payload)
    with pytest.raises(ValueError, match="canonical NaNs"):
        LabelledDataset.load(path)


@pytest.mark.parametrize("metadata", [
    {"stateful": True}, {"warm_start": True}, {"requires_bead_affinity": True},
    {"ring_polymer_coupled": True}, {"coupled_frames": True}, {"batch_independent": False},
    {"capabilities": {"stateful": True}},
])
def test_hidden_state_and_coupled_reference_are_explicitly_rejected(metadata):
    reference = DoubleWellReference()
    reference.metadata = metadata
    with pytest.raises(ValueError, match="independent frames"):
        label_candidates(reference, positions(), reference_identity=IDENTITY)
    assert not reference.calls


@pytest.mark.parametrize("provider", ["pyscf", "tblite"])
def test_optional_electronic_reference_labelling_matches_direct_evaluation(provider):
    from pyrpmd.potentials.adapters import PySCFPotential, TBLitePotential

    pytest.importorskip("pyscf" if provider == "pyscf" else "tblite.interface")
    options = {"basis": "sto-3g", "conv_tol": 1e-12} if provider == "pyscf" else {"accuracy": .1}
    factory = PySCFPotential if provider == "pyscf" else TBLitePotential
    q = np.array([[[0., 0., -.7], [0., 0., .7]], [[0., 0., -.8], [0., 0., .8]]])
    with factory(["H", "H"], **options) as reference:
        labels = label_candidates(reference, q, reference_identity={"provider": provider, "options": options})
        direct = reference.evaluate(q)
    np.testing.assert_allclose(labels.energy, direct.energy, atol=1e-10)
    np.testing.assert_allclose(labels.forces, direct.forces, atol=1e-9)
    assert labels.successful_mask.all()


def test_ipi_socket_reference_labels_match_independent_harmonic_driver():
    from pyrpmd.potentials.socket import IPIPotential

    reference = IPIPotential(port=0, timeout=5)
    errors = []

    def driver():
        try:
            with socket.create_connection(reference.address) as connection:
                state, energy, force = "NEEDINIT", None, None

                def recv(count):
                    out = b""
                    while len(out) < count:
                        chunk = connection.recv(count-len(out))
                        if not chunk:
                            raise EOFError("driver connection ended")
                        out += chunk
                    return out

                while True:
                    header = recv(12).strip()
                    if header == b"STATUS":
                        connection.sendall(state.encode().ljust(12))
                    elif header == b"INIT":
                        _, count = struct.unpack("=ii", recv(8))
                        recv(count)
                        state = "READY"
                    elif header == b"POSDATA":
                        recv(18*8)
                        count = struct.unpack("=i", recv(4))[0]
                        q = np.frombuffer(recv(count*3*8), dtype="=f8").reshape(count, 3)
                        energy, force = np.sum(q*q)/2, -q
                        state = "HAVEDATA"
                    elif header == b"GETFORCE":
                        connection.sendall(b"FORCEREADY".ljust(12) + struct.pack("=di", energy, len(force))
                                           + force.astype("=f8").tobytes() + np.zeros(9, dtype="=f8").tobytes()
                                           + struct.pack("=i", 0))
                        state = "READY"
                    elif header == b"EXIT":
                        break
                    else:
                        raise AssertionError(header)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=driver)
    thread.start()
    try:
        q = positions(5)
        labels = label_candidates(reference, q, reference_identity={"provider": "ipi", "independent_frames": True,
                                  "driver_identity": {"method": "harmonic", "k": 1., "version": 1},
                                  "driver_sha256": hashlib.sha256(b"harmonic-reference-test-driver-v1").hexdigest()},
                                  batch_size=2)
        assert labels.successful_mask.all()
        np.testing.assert_array_equal(labels.energy, np.sum(q*q, axis=(1, 2))/2)
        np.testing.assert_array_equal(labels.forces, -q)
    finally:
        reference.close()
        thread.join(6)
    assert not thread.is_alive() and not errors
