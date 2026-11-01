"""Exercise an installed binary distribution without importing the source tree.

Run by cibuildwheel from its isolated test directory and by the conda recipe.
This intentionally uses only runtime dependencies, not pytest or a local compiler.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

import numpy as np
import h5py
import pyrpmd
from pyrpmd import _native, _thermal
from pyrpmd.constants import BOHR_ANGSTROM
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import FreePotential, HarmonicPotential
from pyrpmd.potentials.openmm import OpenMMPotential
from pyrpmd.potentials.lammps import LAMMPSPotential
from pyrpmd.system import System
from pyrpmd.trajectory_export import frames_from_states, write_trajectory


def main():
    source = Path(__file__).resolve().parents[1] / "src"
    assert source not in Path(pyrpmd.__file__).resolve().parents, pyrpmd.__file__
    capabilities = native_capabilities()
    assert capabilities["available"], capabilities
    for module in (_native, _thermal):
        assert Path(module.__file__).suffix in {".so", ".pyd"}, module.__file__
    if sys.platform.startswith("linux"):
        assert capabilities["openmp"], capabilities
    rng = np.random.default_rng(818)
    original = State(rng.normal(size=(4, 2, 3)), rng.normal(size=(4, 2, 3)))
    for scheme in ("cayley-bcb", "normal-mode-bab", "velocity-verlet"):
        reference, native = original.copy(), original.copy()
        engines = [Dynamics(HarmonicPotential(0.3), [1.0, 2.0], 2.5, 4, 0.01,
                            integrator=scheme, native=flag)
                   for flag in (False, "required")]
        for _ in range(20):
            engines[0].step(reference)
            engines[1].step(native)
        np.testing.assert_allclose(native.q, reference.q, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(native.p, reference.p, rtol=3e-12, atol=3e-12)
    with tempfile.TemporaryDirectory(prefix="pyrpmd-wheel-") as directory:
        root = Path(directory)
        # New core exports and constraints must work from the installed wheel.
        # Provider modules must import even when their optional engines are absent.
        assert OpenMMPotential and LAMMPSPotential
        q = np.repeat(np.array([[[0., 0., 0.], [1., 0., 0.]]]), 4, axis=0)
        masses = np.array([1800., 3600.])
        model = System(("H", "D"), masses, q[0], bond_constraints=((0, 1, 1.),))
        constrained = Dynamics(FreePotential(), masses, 5., 4, 0.01,
                               integrator="velocity-verlet", native=False,
                               bond_constraints=model.bond_constraints)
        state = State(q, constrained.sample_momenta(q))
        saved = [state.copy()]
        for _ in range(20):
            constrained.step(state)
        saved.append(state.copy())
        assert max(constrained.constraints.residuals(state.q, state.p)) < 1.1e-12
        for suffix in ("extxyz", "h5md"):
            receipt = write_trajectory(root / f"dimer.{suffix}",
                                       frames_from_states(saved, model), model,
                                       representation="centroid")
            assert receipt["frames"] == 2
        with h5py.File(root / "dimer.h5md") as trajectory:
            np.testing.assert_allclose(trajectory["particles/all/position/value"],
                                       np.array([s.q.mean(axis=0) for s in saved]) * BOHR_ANGSTROM)
        commands = [
            ["--version"], ["info"], ["init", "example"],
            ["validate", "example/quickstart.inp", "--check-pes"],
            ["run", "example/quickstart.inp", "--output", "run"],
            ["analyze", "run"], ["resume", "run"],
        ]
        for args in commands:
            subprocess.run([sys.executable, "-m", "pyrpmd", *args],
                           cwd=root, check=True)
        summary = json.loads((root / "run/results/summary.json").read_text())
        assert summary, "empty quickstart result"
    print(json.dumps({"version": pyrpmd.__version__, "python": sys.version,
                      "native": capabilities, "installed": pyrpmd.__file__}))


if __name__ == "__main__":
    main()
