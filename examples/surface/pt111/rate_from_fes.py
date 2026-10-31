"""Pilot reactive flux from a previously sampled 3D surface FES.

PYRPMD_PT111_FES selects the source run (default runs/pt111-pilot).
PYRPMD_PT111_CHANNEL is oxidation (default) or desorption. The latter also
requires PYRPMD_PT111_SELECTION, a Boolean CO-channel domain .npy file.
Insufficient FES support stops the workflow before launching children.
"""
import json
import os
from pathlib import Path
from pyrpmd.system import System


def build():
    source = Path(os.environ.get("PYRPMD_PT111_FES", "runs/pt111-pilot")).resolve()
    channel = os.environ.get("PYRPMD_PT111_CHANNEL", "oxidation")
    if channel not in ("oxidation", "desorption"):
        raise ValueError("PYRPMD_PT111_CHANNEL must be oxidation or desorption")
    saved = json.loads((source / "input/resolved.json").read_text())
    system = System.from_dict(json.loads((source / "input/system.json").read_text()))
    free_energy = {"directory": str(source), "axis": 1 if channel == "oxidation" else 2}
    if channel == "oxidation":
        surface = {"reactant_basin": ["2.05 angstrom", "5.8 angstrom"],
                   "dividing_surface": "1.98 angstrom", "product_side": "lower",
                   "normalization": {"reaction_order": 2, "reference_adsorbates": 1,
                                     "site_count": 9, "coverage": .04}}
    else:
        selection = os.environ.get("PYRPMD_PT111_SELECTION")
        if not selection:
            raise ValueError("Set PYRPMD_PT111_SELECTION to a CO-channel Boolean .npy domain; see README.md")
        free_energy["selection"] = str(Path(selection).resolve())
        surface = {"reactant_basin": ["1.1 angstrom", "2.9 angstrom"],
                   "dividing_surface": "3.04 angstrom", "product_side": "higher"}
    return {"task": "surface_rate", "temperature": saved["temperature_K"], "beads": saved["beads"],
            "seed": 7331, "system": system, "potential": saved["potential"], "dynamics": saved["dynamics"],
            "sampling": {"thermostat": "andersen", "andersen_interval": 100,
                         "equilibration_steps": 100, "production_steps": 100, "interval_steps": 10},
            "surface": {**surface, "free_energy": free_energy}, "analysis": {"method": "source", "bootstrap": 0},
            "recrossing": {"parent_replicas": 2, "parent_samples": 4, "equilibration_steps": 100,
                           "parent_interval_steps": 50, "child_pairs": 2, "child_steps": 500, "bootstrap": 20},
            "output": {"checkpoint_steps": 100}}
