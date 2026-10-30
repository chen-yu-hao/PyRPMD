"""Build a real ASE Cu(111)/Cu-adatom FES pilot; requires pyrpmd[ase].

The EMT model supplies a periodic many-atom potential. This is an integration
example; it does not establish a converged quantum diffusion result or the
accuracy of EMT for a catalytic reaction. It writes no geometry files.
"""

import numpy as np


def build():
    from ase.build import add_adsorbate, fcc111
    from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU

    slab = fcc111("Cu", size=(2, 2, 3), vacuum=8.0)
    add_adsorbate(slab, "Cu", height=2.0, position="fcc")
    fixed = np.flatnonzero(slab.get_tags() == 3).tolist()
    top = np.flatnonzero(slab.get_tags() == 1).tolist()
    return {
        "task": "fes", "preset": "pilot", "temperature": "300 K", "beads": 4,
        "system": {
            "symbols": slab.get_chemical_symbols(),
            "masses": (slab.get_masses() * AMU_TO_AU).tolist(),
            "geometry": (slab.positions * ANGSTROM_TO_AU).tolist(),
            "cell": (slab.cell.array * ANGSTROM_TO_AU).tolist(),
            "pbc": slab.pbc.tolist(), "fixed_atoms": fixed,
        },
        "potential": {"provider": "ase", "factory": "ase.calculators.emt:EMT"},
        "dynamics": {"timestep": "0.25 fs"},
        "sampling": {"equilibration_steps": 100, "production_steps": 500, "interval_steps": 5},
        "cvs": [{"kind": "surface_height", "name": "height", "atoms": [len(slab)-1], "reference_atoms": top}],
        "umbrella": {
            "centers": ["1.8 angstrom", "2.0 angstrom", "2.2 angstrom"],
            "stiffness": "1 eV/angstrom^2",
        },
        "analysis": {"bins": 20, "bootstrap": 4},
        "output": {"directory": "runs/ase-cu111"},
    }
