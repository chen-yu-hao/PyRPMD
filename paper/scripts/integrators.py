"""Energy-conservation order of the three deterministic integrators.

A Morse diatomic (H2-like parameters) ring polymer at 300 K, P = 32, is
propagated for a fixed physical time of 200 fs with each integrator and a
range of time steps.  The maximum deviation of the ring-polymer Hamiltonian
from its initial value is recorded.  All quantities are atomic units.
"""
import json
import sys

import numpy as np

from pyrpmd import Dynamics, State
from pyrpmd.constants import AMU_TO_AU, FS_TO_AU, KB_AU
from pyrpmd.potentials import MorsePairPotential

T = 300.0
P = 32
beta = 1.0 / (KB_AU * T)
masses = np.full(2, 1.00782503223 * AMU_TO_AU)
total = 200.0 * FS_TO_AU


def initial():
    rng = np.random.default_rng(20261111)
    q = np.zeros((P, 2, 3))
    q[:, 1, 0] = 1.401
    q += rng.normal(0.0, 0.03, q.shape)
    return q


def potential():
    return MorsePairPotential(depth=0.1745, alpha=1.0283, equilibrium=1.401)


rows = []
q0 = initial()
with potential() as pes:
    # Thermalise with PILE-L for 2 ps so that the start is an equilibrium ring.
    seed_dyn = Dynamics(pes, masses, beta=beta, nbeads=P, dt=0.1 * FS_TO_AU, seed=11,
                        thermostat={"mode": "pile-l", "gamma_centroid": 0.01 / FS_TO_AU})
    st = State(q0.copy(), seed_dyn.sample_momenta())
    for _ in range(20000):
        seed_dyn.step(st)
    q0, p0 = st.q.copy(), st.p.copy()
    print("thermal start; bond", np.linalg.norm(q0[:, 1] - q0[:, 0], axis=1).mean(), file=sys.stderr)
for name in ("cayley-bcb", "normal-mode-bab", "velocity-verlet"):
    for dt_fs in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125):
        with potential() as pes:
            dyn = Dynamics(pes, masses, beta=beta, nbeads=P, dt=dt_fs * FS_TO_AU,
                           integrator=name, seed=11, loop_backend="auto")
            state = State(q0.copy(), p0.copy())
            e0 = dyn.energy(state)
            steps = int(round(total / (dt_fs * FS_TO_AU)))
            worst = 0.0
            done = 0
            ok = True
            while done < steps:
                n = min(max(1, steps // 250), steps - done)
                try:
                    dyn.propagate(state, n)
                except Exception as error:  # unstable step: record and stop
                    ok = False
                    print(name, dt_fs, "failed:", type(error).__name__, file=sys.stderr)
                    break
                done += n
                e = dyn.energy(state)
                if not np.isfinite(e):
                    ok = False
                    break
                worst = max(worst, abs(e - e0))
            rows.append({"integrator": name, "dt_fs": dt_fs, "steps": steps,
                         "stable": ok, "initial_energy": float(e0),
                         "max_abs_energy_error": float(worst) if ok else None,
                         "max_relative_energy_error": float(worst / abs(e0)) if ok else None,
                         "loop_stats": dyn.loop_stats})
            print(rows[-1], file=sys.stderr)
json.dump(rows, sys.stdout, indent=1)
