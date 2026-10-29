"""Free Python input using an importable potential factory."""

from pathlib import Path
import tomllib


def build():
    directory = Path(__file__).parent
    config = tomllib.loads((directory / "free_scattering.inp").read_text())
    config["potential"] = {
        "provider": "python",
        "factory": str(directory / "toy_pes.py") + ":build_pes",
        "params": {"height_hartree": 0.002, "width_bohr": 1.5},
    }
    config["output"]["directory"] = "runs/python_pes"
    return config
