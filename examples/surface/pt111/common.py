"""Shared 39-atom surface, three reaction CVs and the published soft walls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def metadynamics(*, production=False):
    cvs = [{"name": "r1", "kind": "periodic_distance", "atoms": [0, 1]},
           {"name": "r2", "kind": "periodic_distance", "atoms": [0, 2]},
           {"name": "h", "kind": "surface_height", "atoms": [0], "reference_atoms": list(range(3, 12))}]
    walls = [{"cv": r, "upper": "9 angstrom", "kappa": "150 eV/angstrom^2"} for r in ("r1", "r2")]
    walls.append({"cv": "h", "lower": "0.5 angstrom", "upper": "7.2 angstrom", "kappa": "150 eV/angstrom^2"})
    for atom in range(1, 21):
        name = f"z{atom}"
        cvs.append({"name": name, "kind": "cartesian", "atom": atom, "axis": "z"})
        bound = {"lower": "6.8 angstrom"} if atom < 3 else {"upper": "7.9 angstrom" if atom < 12 else "5.8 angstrom"}
        walls.append({"cv": name, **bound, "kappa": "150 eV/angstrom^2"})
    return {
        "task": "metadynamics", "temperature": "600 K", "beads": 1, "seed": 999999999,
        "system": json.loads((ROOT / "system.json").read_text()),
        "cvs": cvs,
        "potential": {"provider": "python", "factory": str(ROOT / "eann.py") + ":build",
                      "params": {"model": str(ROOT / "models" / "REANN_PES_DOUBLE.pt")}},
        "dynamics": {"timestep": "0.2 fs", "integrator": "velocity-verlet"},
        "sampling": {"thermostat": "nose-hoover-chain",
                     "thermostat_params": {"tdamp": "20 fs", "chain_length": 3, "substeps": 1,
                                           "reset_after_equilibration": True},
                     "equilibration_steps": 40000 if production else 100,
                     "production_steps": 200000000 if production else 800,
                     "interval_steps": 400 if production else 20, "replicas": 3 if production else 1},
        "metadynamics": {"cvs": ["r1", "r2", "h"], "mode": "well_tempered",
                         "sigma": ["0.2 angstrom"] * 3, "height": "0.020 eV", "pace": 400,
                         "bias_factor": 39, "walls": walls, "walls_during_equilibration": False},
        "analysis": {"bins": [215, 215, 185] if production else [10, 10, 10], "bootstrap": 0,
                     **({"ranges": [["0.8 angstrom", "9.4 angstrom"], ["0.8 angstrom", "9.4 angstrom"],
                                    ["0.2 angstrom", "7.6 angstrom"]]} if production else {})},
        "output": {"checkpoint_steps": 40000 if production else 100},
    }
