"""Short on-the-fly LiH SA-CASSCF trajectory using direct adiabatic RPSH.

Install ``pyscf`` and run with, for example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        examples/nonadiabatic/pyscf_lih_rpsh.py --steps 40

All calculations use a CPU, double precision, a STO-3G basis, two equally
weighted singlet CAS(2,2) states, and full analytic derivative couplings.
This is a small infrastructure demonstration, not a converged LiH photochemical
prediction. The default P=1 trajectory takes seconds to minutes on a CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time

import numpy as np

from pyrpmd.adiabatic import AdiabaticPythonPotential, AdiabaticRPSHDynamics
from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, AU_TIME_FS, KB_AU
from pyrpmd.dynamics import State
from pyrpmd.potentials.pyscf_multistate import PySCFMultistatePotential


def run(*, steps=40, dt=1., beads=1, seed=13, initial_state=1):
    """Return complete numerical settings, trajectory diagnostics and timing.

    ``dt`` is in atomic time units. Nuclear initialization is a 1.5 angstrom
    bond with opposite 0.5 atomic-unit momenta and zero center-of-mass momentum.
    For P>1 the beads start collapsed; this is not a thermal ring ensemble.
    No portable quantum-chemistry checkpoint is created by this example.
    """
    for value, name, minimum in ((steps, "steps", 1), (beads, "beads", 1),
                                  (seed, "seed", 0)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (isinstance(dt, (bool, np.bool_)) or not np.isscalar(dt)
            or not np.isreal(dt) or not np.isfinite(dt) or dt <= 0):
        raise ValueError("dt must be positive and finite")
    if (isinstance(initial_state, (bool, np.bool_))
            or not isinstance(initial_state, (int, np.integer))
            or initial_state not in (0, 1)):
        raise ValueError("initial_state must be 0 or 1")
    steps, beads, seed, initial_state = map(int, (steps, beads, seed, initial_state))
    dt = float(dt)
    masses = np.array([7.0160034366, 1.00782503223])*AMU_TO_AU
    q = np.zeros((beads, 2, 3))
    q[:, 1, 0] = 1.5*ANGSTROM_TO_AU
    p = np.zeros_like(q)
    p[:, 0, 0], p[:, 1, 0] = -.5, .5
    beta = 1./(KB_AU*300.)
    bridge = PySCFMultistatePotential(
        ["Li", "H"], basis="sto-3g", ncas=2, nelecas=2, nstates=2,
        casscf_conv_tol=1e-12, casscf_conv_tol_grad=1e-7,
        response_atol=1e-12, response_rtol=1e-9,
    )
    provider = AdiabaticPythonPotential(bridge)
    config = {
        "schema": "PyRPMD.pyscf-lih-rpsh.v1", "provider": bridge.to_dict(),
        "method": "adiabatic-rpsh", "dtype": "float64", "execution_device": "cpu",
        "masses_electron": masses.tolist(), "initial_q_bohr": q.tolist(),
        "initial_p_au": p.tolist(), "beta_hartree_inverse": beta,
        "nbeads": beads, "dt_au": dt, "steps": steps, "seed": seed,
        "initial_state": initial_state, "electronic_substeps": 8,
        "integrator": "cayley-bcb", "native": False, "threads": 1,
        "overlap_tolerance": 1e-6, "frustrated_hops": "keep",
    }
    fingerprint = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    dynamics = AdiabaticRPSHDynamics(
        provider, masses, beta, beads, dt, 2, seed=seed, native=False,
        electronic_substeps=8, overlap_tolerance=1e-6, frustrated_hops="keep",
    )
    started = time.perf_counter()
    try:
        state = dynamics.initialize(State(q, p), active_surface=initial_state)
        energies = [dynamics.energy(state)]
        populations = [dynamics.populations(state).tolist()]
        surfaces = [state.active_surface]
        bond_lengths = [float(np.linalg.norm(state.q.mean(0)[1]-state.q.mean(0)[0]))]
        max_overlap_leakage = 0.
        hops = []
        for _ in range(steps):
            dynamics.step(state)
            energies.append(dynamics.energy(state))
            populations.append(dynamics.populations(state).tolist())
            surfaces.append(state.active_surface)
            bond_lengths.append(float(np.linalg.norm(state.q.mean(0)[1]-state.q.mean(0)[0])))
            for frame in (state.bead_frame, state.centroid_frame):
                singular = np.linalg.svd(frame.result.overlaps, compute_uv=False)
                max_overlap_leakage = max(max_overlap_leakage, float(np.max(abs(singular-1.))))
            hop = dynamics.last_hop
            if hop is not None:
                hops.append({"step": state.step, "accepted": bool(hop.accepted),
                             "frustrated": bool(hop.frustrated),
                             "ring_energy_change_hartree": float(hop.ring_energy_change)})
        elapsed = time.perf_counter()-started
        from pyscf import __version__ as pyscf_version, lib
        from threadpoolctl import threadpool_info

        return {
            "configuration": config, "physics_fingerprint": fingerprint,
            
            "validation_scope": "short molecular electronic-structure and propagation check",
            "time_au": (np.arange(steps+1)*dt).tolist(),
            "duration_fs": state.time*AU_TIME_FS,
            "total_energy_hartree": energies,
            "max_energy_drift_hartree": float(np.max(abs(np.asarray(energies)-energies[0]))),
            "population_history": populations, "active_surface_history": surfaces,
            "bond_length_bohr": bond_lengths, "hops": hops,
            "max_overlap_singular_value_deviation": max_overlap_leakage,
            "cache": bridge.cache_info(), "wall_seconds": elapsed,
            "seconds_per_step_including_initialization": elapsed/steps,
            "runtime": {"python": platform.python_version(), "machine": platform.machine(),
                        "pyscf": pyscf_version, "pyscf_threads": lib.num_threads(),
                        "threadpools": [{"internal_api": item["internal_api"],
                                         "num_threads": item["num_threads"]}
                                        for item in threadpool_info()]},
        }
    finally:
        provider.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--dt", type=float, default=1., help="Atomic time units")
    parser.add_argument("--beads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--initial-state", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    print(json.dumps(run(steps=args.steps, dt=args.dt, beads=args.beads,
                         seed=args.seed, initial_state=args.initial_state), indent=2))
