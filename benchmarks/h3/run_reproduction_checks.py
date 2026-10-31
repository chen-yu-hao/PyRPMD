"""Fixed, recorded sensitivity workload for the 2026-10-31 reproduction.

Run after building BKMP2, with PYTHONPATH=src:. and BLAS/OMP threads=1.
This intentionally refuses an existing case directory through the wrapper.
"""
import argparse
from pathlib import Path

from benchmarks.h3.state_selected_reproduction import Config, ROOT
from benchmarks.h3.state_selected_sensitivity import run_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/state_selected_reproduction/sensitivity")
    args = parser.parse_args()
    jobs = [
        ("beads", Config(v=0, total_energy_ev=.55, beads=64, samples=2048, id_offset=30000)),
        ("beads", Config(v=1, total_energy_ev=1.10, beads=64, samples=512, bmax=4.0, id_offset=30000)),
        ("timestep", Config(v=0, total_energy_ev=.55, samples=256, dt_fs=.01, id_offset=10000)),
        ("timestep", Config(v=1, total_energy_ev=1.10, samples=256, bmax=4.0, dt_fs=.01, id_offset=10000)),
        ("radial", Config(v=0, total_energy_ev=.55, samples=256, id_offset=10000)),
        ("radial", Config(v=1, total_energy_ev=1.10, samples=256, bmax=4.0, id_offset=10000)),
        ("annulus", Config(v=0, total_energy_ev=1.10, samples=1024, bmin=3.5, bmax=4.5, id_offset=40000)),
        ("annulus", Config(v=1, total_energy_ev=.90, samples=1024, bmin=4.0, bmax=5.0, id_offset=40000)),
        ("annulus", Config(v=1, total_energy_ev=1.10, samples=1024, bmin=4.0, bmax=5.0, id_offset=40000)),
        # Added after the P32 low-energy v=1 comparison revealed a discrepancy;
        # its N=1024 was fixed before inspecting this P64 follow-up sample.
        ("beads", Config(v=1, total_energy_ev=.90, beads=64, samples=1024, bmax=4.0, id_offset=30000)),
    ]
    for kind, config in jobs:
        run_sensitivity(kind, config, args.workers, args.output/f"{kind}-v{config.v}-Et{config.total_energy_ev:.3f}")


if __name__ == "__main__":
    main()
