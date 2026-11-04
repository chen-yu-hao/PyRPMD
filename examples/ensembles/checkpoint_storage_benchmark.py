"""Measure atomic checkpoint storage cost using synthetic high-entropy states.

Run from an installed PyRPMD checkout, with no other CPU or I/O workload:

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
    examples/ensembles/checkpoint_storage_benchmark.py \
    --beads 32 --atoms 128 4096 --repeats 7 --warmup 1 --seed 1 \
    --output checkpoint-storage.json

The reference reproduces the pre-T16b native atomic NPZ writer. The two common
writers additionally copy the input, hash each array, store an integrity
envelope and fsync the containing directory on POSIX. This measures container
publication, not propagation, NativeEnsemble.state(), or physical convergence.
The generated arrays have native checkpoint shapes but are synthetic storage
payloads, not resumable trajectories. Temporary NPZ files are removed on exit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np

from pyrpmd.checkpoint import atomic_savez, atomic_savez_uncompressed, load_checkpoint


FORMAT = "pyrpmd.native-ensemble.v1"


def _payload(rng, beads, atoms):
    shape = (beads, atoms, 3)
    arrays = {name: rng.standard_normal(shape) for name in ("q", "p", "force")}
    arrays.update(
        aux=np.empty((beads, atoms, 3, 0)),
        energy=rng.standard_normal(beads),
        virial=rng.standard_normal((beads, 3, 3)),
        cell=np.eye(3),
        cell_p=np.zeros((3, 3)),
        metadata=json.dumps({
            "format": FORMAT, "fingerprint": "a" * 64, "digest": "b" * 64,
            "step": 10, "force_evaluations": 11, "rng": "1 " * 312,
        }),
    )
    return arrays


def _old_native_save(path, arrays):
    """Original native save lifecycle, given an already detached payload."""
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _common_uncompressed(path, arrays):
    atomic_savez_uncompressed(path, kind=FORMAT, **arrays)


def _common_compressed(path, arrays):
    atomic_savez(path, kind=FORMAT, **arrays)


WRITERS = {
    "old_native_npz": _old_native_save,
    "common_uncompressed": _common_uncompressed,
    "common_compressed": _common_compressed,
}


def benchmark(*, directory, beads, atoms, repeats, warmup, seed):
    rng = np.random.default_rng(seed)
    records = []
    for natoms in atoms:
        arrays = _payload(rng, beads, natoms)
        paths = {name: directory / f"{name}.npz" for name in WRITERS}
        for name, writer in WRITERS.items():
            for _ in range(warmup):
                writer(paths[name], arrays)
        timings = {name: [] for name in WRITERS}
        for repetition in range(repeats):
            # Alternate ordering to reduce systematic cache/scheduling bias.
            names = list(WRITERS)
            if repetition % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                WRITERS[name](paths[name], arrays)
                timings[name].append((time.perf_counter() - start) * 1000)
        for name, values in timings.items():
            # Verify equality outside the measured region, without propagating
            # the deliberately synthetic state or trusting fake native hashes.
            with load_checkpoint(paths[name], expected_kind=FORMAT) as saved:
                if set(saved.files) != set(arrays):
                    raise RuntimeError(f"{name} changed the checkpoint payload inventory")
                for key, expected in arrays.items():
                    np.testing.assert_array_equal(saved[key], expected)
            records.append({
                "atoms": natoms,
                "beads": beads,
                "method": name,
                "median_ms": statistics.median(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "measurements_ms": values,
                "size_bytes": paths[name].stat().st_size,
                "payload_identity_verified": True,
            })
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beads", type=int, default=32)
    parser.add_argument("--atoms", type=int, nargs="+", default=[128, 4096])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("checkpoint-storage.json"))
    parser.add_argument("--temporary-directory", type=Path,
                        help="existing parent directory on the filesystem being measured")
    args = parser.parse_args(argv)
    if args.beads < 1 or any(value < 1 for value in args.atoms) or args.repeats < 1:
        parser.error("beads, atoms and repeats must be positive")
    if args.warmup < 0 or args.seed < 0:
        parser.error("warmup and seed must be nonnegative")
    if args.temporary_directory is not None and not args.temporary_directory.is_dir():
        parser.error("temporary-directory must name an existing directory")
    with tempfile.TemporaryDirectory(prefix="pyrpmd-checkpoint-benchmark-",
                                     dir=args.temporary_directory) as temporary:
        records = benchmark(directory=Path(temporary), beads=args.beads, atoms=args.atoms,
                            repeats=args.repeats, warmup=args.warmup, seed=args.seed)
    report = {
        "purpose": "atomic checkpoint container publication using synthetic arrays; no dynamics",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "threads": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        )},
        "repetitions": args.repeats,
        "warmups": args.warmup,
        "seed": args.seed,
        "beads": args.beads,
        "atom_counts": args.atoms,
        "temporary_parent": str(args.temporary_directory.resolve()
                                if args.temporary_directory else Path(tempfile.gettempdir()).resolve()),
        "notes": [
            "Timings include publication fsyncs, exclude array generation and post-write validation.",
            "Common writers add an owned array copy, SHA256 envelope and POSIX directory fsync.",
            "Storage latency and concurrent workload affect timing; no propagation speedup is implied.",
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
