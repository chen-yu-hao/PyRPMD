"""The final benchmark must not silently accept incomplete recovery records."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from pyrpmd.potentials.python import import_object

MODULE = str(Path(__file__).resolve().parents[1] / "benchmarks/h3/rpmd.py")
read_streams = import_object(MODULE + ":_streams")


class BenchmarkIntegrityTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / "checkpoints").mkdir()
        samples = np.arange(12, dtype=float).reshape(6, 2)
        with h5py.File(root / "fes.h5", "w") as h:
            group = h.create_group("streams/0")
            group.attrs["window"] = 3
            group.attrs["replica"] = 1
            group["samples"] = samples
        checkpoint = root / "checkpoints/window_00003_replica_0001.npz"
        return root, checkpoint, samples

    def test_incomplete_prefix_retains_full_samples_but_fails_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root, checkpoint, samples = self.fixture(directory)
            np.savez(checkpoint, values=samples[:4], q=np.ones((2, 3, 3)), elapsed=20)
            values, _, integrity = read_streams(root)
            np.testing.assert_array_equal(values[0], samples)
            self.assertFalse(integrity["passed"])
            self.assertEqual(integrity["mismatches"][0]["hdf5_samples"], 6)
            self.assertEqual(integrity["mismatches"][0]["checkpoint_samples"], 4)
            self.assertTrue(integrity["mismatches"][0]["exact_prefix"])
            np.savez(checkpoint, values=samples, q=np.ones((2, 3, 3)), elapsed=30)
            _, _, complete = read_streams(root)
            self.assertTrue(complete["passed"])
            self.assertEqual(complete["mismatches"], [])

    def test_equal_sample_counts_do_not_hide_changed_sample_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root, checkpoint, samples = self.fixture(directory)
            corrupt = samples.copy()
            corrupt[-1, 0] += 1.0
            np.savez(checkpoint, values=corrupt, q=np.ones((2, 3, 3)), elapsed=30)
            with self.assertRaisesRegex(ValueError, "sample values disagree"):
                read_streams(root)


if __name__ == "__main__":
    unittest.main()
