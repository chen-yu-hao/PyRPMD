"""Measure NumPy free-ring coefficient reuse against the uncached formula.

Both cases use identical FFTs, input checks and arithmetic ordering; only
coefficient preparation differs. This is a kernel microbenchmark, not a
prediction of speedup for a chemistry trajectory with an expensive PES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

from pyrpmd.dynamics import CayleyIntegrator


class UncachedIntegrator(CayleyIntegrator):
    """Pre-audit NumPy map retained solely as the benchmark reference."""

    def _fourier_step(self, q, p, half=False):
        qk = np.fft.rfft(q, axis=0, norm="ortho")
        pk = np.fft.rfft(p, axis=0, norm="ortho")
        z = .5 * self.dt * self._omega
        if self.scheme == "normal-mode-bab":
            h = .5 * self.dt if half else self.dt
            theta = h * self._omega
            a = np.cos(theta)
            b = h * np.sinc(theta / np.pi)
            c = self._omega * np.sin(theta)
        elif half:
            denominator = np.sqrt(1 + z * z)
            a = 1 / denominator
            b = (.5 * self.dt) / denominator
            c = .5 * self.dt * self._omega**2 / denominator
        else:
            denominator = 1 + z * z
            a = (1 - z * z) / denominator
            b = self.dt / denominator
            c = self.dt * self._omega**2 / denominator
        q[:] = np.fft.irfft(
            a[:, None, None] * qk + b[:, None, None] * pk / self.masses[None, :, None],
            n=self.nbeads, axis=0, norm="ortho",
        )
        p[:] = np.fft.irfft(
            a[:, None, None] * pk - c[:, None, None] * self.masses[None, :, None] * qk,
            n=self.nbeads, axis=0, norm="ortho",
        )


def benchmark_case(beads, atoms, scheme, half, steps, repeats):
    rng = np.random.default_rng(1489)
    initial_q = rng.normal(size=(beads, atoms, 3))
    initial_p = rng.normal(size=initial_q.shape)
    masses = np.linspace(1., 3., atoms)
    integrators = {
        "uncached": UncachedIntegrator(masses, 10., beads, .005, scheme=scheme, native=False),
        "cached": CayleyIntegrator(masses, 10., beads, .005, scheme=scheme, native=False),
    }
    seconds = {name: [] for name in integrators}
    states = {}
    for integrator in integrators.values():
        advance = integrator.half_step if half else integrator.free_step
        q, p = initial_q.copy(), initial_p.copy()
        for _ in range(10):
            advance(q, p)
    for repeat in range(repeats):
        for name in (("uncached", "cached") if repeat % 2 == 0 else ("cached", "uncached")):
            integrator = integrators[name]
            advance = integrator.half_step if half else integrator.free_step
            q, p = initial_q.copy(), initial_p.copy()
            start = time.perf_counter()
            for _ in range(steps):
                advance(q, p)
            seconds[name].append(time.perf_counter() - start)
            states[name] = q, p
        for actual, reference in zip(states["cached"], states["uncached"]):
            np.testing.assert_array_equal(actual, reference)
    medians = {name: statistics.median(values) for name, values in seconds.items()}
    return {
        "beads": beads, "atoms": atoms, "scheme": scheme, "half_step": half,
        "steps": steps, "repeats": repeats, "seconds": seconds,
        "median_seconds": medians, "speedup": medians["uncached"] / medians["cached"],
        "positions_and_momenta_exactly_equal": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("free-ring-coefficients.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    cases = [benchmark_case(beads, atoms, scheme, half, args.steps, args.repeats)
             for scheme in ("cayley-bcb", "normal-mode-bab")
             for beads, atoms in ((1, 1), (8, 3), (32, 3), (64, 50))
             for half in (False, True)]
    source = Path(__file__).resolve().parents[2] / "src/pyrpmd/dynamics.py"
    report = {
        "purpose": "NumPy free-ring kernel microbenchmark; not a chemistry speedup",
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "dynamics_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps([{key: case[key] for key in ("beads", "atoms", "scheme", "half_step", "speedup")}
                      for case in cases], indent=2))


if __name__ == "__main__":
    main()
