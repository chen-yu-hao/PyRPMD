"""Evaluate an H2 batch with real tblite or PySCF; no dynamics/convergence claim.

Run: python examples/potentials/on_the_fly.py tblite
     python examples/potentials/on_the_fly.py pyscf
"""
import argparse
import json
import numpy as np

from pyrpmd.potentials import check_potential, make_potential
from pyrpmd.system import System


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("tblite", "pyscf"))
    args = parser.parse_args()
    system = System(("H", "H"), [1837., 1837.], [[0., 0., 0.], [1.4, 0., 0.]])
    params = ({"method": "GFN2-xTB"} if args.provider == "tblite"
              else {"method": "HF", "basis": "sto-3g", "conv_tol": 1e-12})
    q = np.array([system.geometry, system.geometry * 1.05])
    with make_potential({"provider": args.provider, "params": params}, system) as potential:
        result = potential.evaluate(q)
        print(json.dumps({"provider": args.provider,
                          "energy_hartree": result.energy.tolist(),
                          "forces_hartree_per_bohr": result.forces.tolist(),
                          "force_check": check_potential(potential, q, epsilon=1e-4)}, indent=2))


if __name__ == "__main__":
    main()
