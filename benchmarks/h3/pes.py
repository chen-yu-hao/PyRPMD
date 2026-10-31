"""Real H + H2 BKMP2 PES: atomic-unit batch energies and analytic Cartesian forces."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyrpmd.potentials import PESResult, Potential, PotentialError, check_potential
from pyrpmd.potentials.base import coordinates, validate_result
from pyrpmd.potentials.cabi import _lock_for

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text())
DEFAULT_LIBRARY = HERE / "build" / "libbkmp2.so"
_DOUBLE_ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_INT_ARRAY = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


class BKMP2Potential(Potential):
    def __init__(self, library=None):
        self.path = Path(library or DEFAULT_LIBRARY).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Build the real BKMP2 PES first: python {HERE / 'build_pes.py'}")
        provenance = self.path.with_suffix(".build.json")
        if not provenance.is_file():
            raise PotentialError(f"Missing BKMP2 build provenance: {provenance}")
        self.provenance = json.loads(provenance.read_text())
        binary_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.provenance["source"]["sha256"] != SOURCE["sha256"] or self.provenance["library_sha256"] != binary_hash:
            raise PotentialError("BKMP2 library/source provenance checksum mismatch")
        self._lock = _lock_for(self.path)
        self.lib = ctypes.CDLL(str(self.path))
        self.batch = self.lib.pyrpmd_bkmp2_batch
        self.batch.argtypes = [ctypes.c_int32, _DOUBLE_ARRAY, _DOUBLE_ARRAY, _DOUBLE_ARRAY, _INT_ARRAY]
        self.batch.restype = None
        self._radial = self.lib.pyrpmd_bkmp2_radial
        self._radial.argtypes = [_DOUBLE_ARRAY, _DOUBLE_ARRAY, _DOUBLE_ARRAY, _INT_ARRAY]
        self._radial.restype = None
        self.metadata = {
            "provider": "BKMP2", "model": SOURCE["model"], "execution_device": "cpu",
            "thread_safety": "process_serial", "native_batch": True, "dtype": "float64",
            "trajectory_batch_safe": True,
            "atomic_numbers": [1, 1, 1], "charge": 0, "multiplicity": 2,
            "energy_zero": SOURCE["energy_zero"], "source_commit": SOURCE["commit"],
            "source_sha256": SOURCE["sha256"], "library_sha256": binary_hash,
        }

    def native_binding(self):
        from pyrpmd.propagation import NativePESBinding, NativeUnavailableError
        if (type(self) is not BKMP2Potential or "evaluate" in self.__dict__
                or BKMP2Potential.evaluate is not _BKMP2_EVALUATE):
            raise NativeUnavailableError("overridden BKMP2 provider uses Python semantics")
        # Canonical-path locking also covers aliases importing this module.
        return NativePESBinding(self, self.batch, self._lock, 3, kind=1)

    def evaluate(self, x):
        x = coordinates(x)
        if x.shape[1:] != (3, 3):
            raise PotentialError("BKMP2 requires exactly three hydrogen nuclei")
        if len(x) > np.iinfo(np.int32).max:
            raise PotentialError("BKMP2 batch is too large")
        energy = np.empty(len(x), dtype=np.float64)
        force = np.empty_like(x)
        status = np.zeros(len(x), dtype=np.int32)
        with self._lock:
            self.batch(len(x), x, energy, force, status)
        messages = ["" if s == 0 else ("BKMP2 geometry outside domain: every pair distance must be >= 0.2 bohr" if s == 1 else "BKMP2 returned nonfinite energy or force") for s in status]
        return validate_result(PESResult(energy, force, status, messages), x.shape, require_success=False)

    def radial(self, distances):
        r = np.ascontiguousarray(distances, dtype=np.float64)
        if r.shape != (3,):
            raise ValueError("radial input must be three interatomic distances")
        energy = np.empty(1); gradient = np.empty(3); status = np.zeros(1, dtype=np.int32)
        with self._lock:
            self._radial(r, energy, gradient, status)
        if status[0]:
            raise PotentialError(f"Invalid BKMP2 radial geometry, status={status[0]}")
        return float(energy[0]), gradient


# The exported native batch must not bypass a later evaluate() replacement.
_BKMP2_EVALUATE = BKMP2Potential.evaluate
# Standalone radial kernels use the same rule for the public radial method.
_BKMP2_RADIAL = BKMP2Potential.radial


def build(library=None):
    """Importable process-local factory used by all H3 benchmark workflows."""
    return BKMP2Potential(library)


def verify(library=None):
    """Independent author oracles, force differences, symmetries and real TS."""
    from scipy.optimize import minimize_scalar
    from pyrpmd.workflows.prepare import numerical_hessian

    potential = build(library)
    # Values below are constants in the authors' bkmp2test.f/test output.
    oracles = [([1.757, 1.757, 3.514], -0.159176000), ([1.401, 99., 99.], -0.174495730)]
    energy_checks = []
    for r, reference in oracles:
        energy, _ = potential.radial(r)
        error = abs(energy-reference)
        if error > 5e-10:
            raise AssertionError(f"Author energy oracle failed for {r}: {energy}")
        energy_checks.append({"r_bohr": r, "reference_hartree": reference, "actual_hartree": energy, "absolute_error_hartree": error})
    compact_e, compact_g = potential.radial([.9, 1.1, 1.5])
    compact_reference = np.array([-.524470263, -.271446000, -.108650526])
    np.testing.assert_allclose(compact_g, compact_reference, atol=6e-10, rtol=0)
    h2 = minimize_scalar(lambda r: potential.radial([r, 99., 99.])[0], bounds=(1.2, 1.6), method="bounded", options={"xatol": 1e-13})
    saddle = minimize_scalar(lambda r: potential.radial([r, r, 2*r])[0], bounds=(1.5, 2.), method="bounded", options={"xatol": 1e-13})
    qts = np.array([[-saddle.x, 0., 0.], [0., 0., 0.], [saddle.x, 0., 0.]])
    eigenvalues = np.linalg.eigvalsh(numerical_hessian(potential, qts))
    negative = int(np.count_nonzero(eigenvalues < -1e-6))
    if negative != 1:
        raise AssertionError(f"BKMP2 collinear stationary point is not a first-order saddle: {eigenvalues}")
    points = np.array([
        [[-1.85, .1, -.2], [0., 0., 0.], [1.6, .2, .1]],
        [[-4., 1., .3], [0., 0., 0.], [h2.x, .1, -.1]],
        [[-1.3, .3, .2], [.1, -.2, 0.], [1.9, .5, -.3]],
        qts,
    ])
    force_error = check_potential(potential, points, epsilon=1e-5)
    if force_error["max_absolute_force_error"] > 2e-8:
        raise AssertionError(f"Cartesian analytic force check failed: {force_error}")
    result = potential.evaluate(points)
    rotation, _ = np.linalg.qr(np.random.default_rng(71).normal(size=(3, 3)))
    moved = potential.evaluate(points @ rotation + np.array([4., -2., 3.]))
    np.testing.assert_allclose(moved.energy, result.energy, atol=2e-13, rtol=0)
    np.testing.assert_allclose(moved.forces, result.forces @ rotation, atol=2e-12, rtol=0)
    perm = [2, 0, 1]
    reordered = potential.evaluate(points[:, perm])
    np.testing.assert_allclose(reordered.energy, result.energy, atol=2e-13, rtol=0)
    np.testing.assert_allclose(reordered.forces, result.forces[:, perm], atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.forces.sum(axis=1), 0., atol=2e-14, rtol=0)
    scalar = [potential.evaluate(q[None]) for q in points]
    np.testing.assert_array_equal(result.energy, np.concatenate([r.energy for r in scalar]))
    np.testing.assert_array_equal(result.forces, np.concatenate([r.forces for r in scalar]))
    invalid = potential.evaluate(np.zeros((1, 3, 3)))
    if not invalid.status[0]:
        raise AssertionError("Invalid geometry was not rejected before upstream Fortran STOP")
    batch = np.repeat(points, 250, axis=0)
    started = time.perf_counter()
    for _ in range(10):
        potential.evaluate(batch)
    elapsed = time.perf_counter()-started
    return {
        "schema": "pyrpmd.h3.bkmp2-verification.v1", "passed": True, "source": SOURCE,
        "build": potential.provenance, "author_energy_oracles": energy_checks,
        "author_compact_derivative": {"r_bohr": [.9, 1.1, 1.5], "energy_hartree": compact_e,
            "reference_hartree_per_bohr": compact_reference.tolist(), "actual_hartree_per_bohr": compact_g.tolist(),
            "maximum_absolute_error": float(np.max(abs(compact_g-compact_reference)))},
        "h2_equilibrium_bohr": float(h2.x), "h2_minimum_hartree": float(h2.fun),
        "ts_neighbor_distance_bohr": float(saddle.x), "ts_energy_hartree": float(saddle.fun),
        "barrier_hartree": float(saddle.fun-h2.fun), "barrier_kcal_mol": float((saddle.fun-h2.fun)*627.5094740631),
        "ts_hessian_eigenvalues_au": eigenvalues.tolist(), "ts_negative_hessian_modes": negative,
        "ts_max_force_au": float(np.max(abs(potential.evaluate(qts[None]).forces))),
        "force_check": force_error, "translation_rotation_permutation_invariance": True,
        "batch_scalar_exact": True, "invalid_geometry_guard": True,
        "benchmark_frames": len(batch)*10, "benchmark_seconds": elapsed,
        "microseconds_per_frame": elapsed*1e6/(len(batch)*10),
        "scope": "Electronic PES and adapter verification, not a converged quantum rate calculation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.library)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
