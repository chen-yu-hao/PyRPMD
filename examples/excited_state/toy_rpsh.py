"""Run a deterministic short RPSH trajectory through the provider interface.

The avoided-crossing backend is an analytic interface check, not a physically
converged molecular calculation.  All quantities passed to PyRPMD are in atomic
units.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticRPSHDynamics
from pyrpmd.dynamics import State
from pyrpmd.potentials.excited_state import AvoidedCrossingBackend, ExcitedStateProvider


def run(*, steps: int = 20, dt: float = 0.02, seed: int = 7) -> dict:
    """Run the toy trajectory and return a JSON-serializable diagnostic record."""

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    provider = ExcitedStateProvider(
        AvoidedCrossingBackend(
            1,
            slope=0.02,
            coupling=0.01,
            harmonic=0.002,
        ),
        ["H"],
        2,
        coupling_method="analytic",
        scratch_cleanup="always",
    )
    potential = AdiabaticPythonPotential(provider)
    dynamics = AdiabaticRPSHDynamics(
        potential,
        masses=[1837.1526473652],
        beta=1000.0,
        nbeads=1,
        dt=dt,
        nstates=2,
        native=False,
        seed=seed,
        electronic_substeps=8,
        overlap_tolerance=1e-10,
    )
    nuclear = State(
        q=np.array([[[-0.20, 0.03, 0.0]]]),
        p=np.array([[[1.0, 0.0, 0.0]]]),
    )
    try:
        state = dynamics.initialize(nuclear, active_surface=0)
        initial_energy = dynamics.energy(state)
        for _ in range(steps):
            dynamics.step(state)
        final_energy = dynamics.energy(state)
        drift = final_energy - initial_energy
        return {
            "completed": True,
            
            "model": "analytic two-state avoided crossing",
            "units": "atomic",
            "steps": state.step,
            "time": state.time,
            "active_surface": state.active_surface,
            "populations": dynamics.populations(state).tolist(),
            "initial_energy_hartree": initial_energy,
            "final_energy_hartree": final_energy,
            "energy_drift_hartree": drift,
            "provider_statistics": asdict(provider.statistics()),
        }
    finally:
        potential.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.02, help="atomic time units")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(steps=args.steps, dt=args.dt, seed=args.seed)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
