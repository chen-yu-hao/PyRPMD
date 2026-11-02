"""Generate and validate a paper-defined H+CH4 pilot input; never launch it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pprint

from pyrpmd.input import JobSpec
from pyrpmd.workflows.prepare import prepare_rate

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "reproduction_protocol.json").read_text())
GEOMETRIES = json.loads((HERE / "geometries.json").read_text())


def pilot_input(library, *, beads=1, seed=None, output="hch4-pilot", workers=1):
    """Return a normal PyRPMD mapping with explicit physics and a small budget.

    The generator deliberately has no production-budget switch: that analysis
    requires a separately reviewed memory-bounded staging plan. A resolved
    input is executable, but no short pilot is a literature reproduction.
    """
    if isinstance(beads, bool) or beads not in (1, 64, 128, 256):
        raise ValueError("Use P=1/128 primary or the preregistered P=64/256 controls")
    if seed is None:
        seed = PROTOCOL["campaign"]["seeds"][0]
    if isinstance(seed, bool) or seed not in PROTOCOL["campaign"]["seeds"]:
        raise ValueError("Use one of the four preregistered pilot seeds")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    physics = PROTOCOL["planned_physics"]
    pilot = PROTOCOL["campaign"]["pilot"]
    return {
        "schema_version": 1, "task": "rate", "preset": "production",
        "description": "H+CH4 CPC-2013 short integration pilot; not a validated rate",
        "temperature": "300 K", "beads": beads, "seed": seed,
        "system": {
            "symbols": ["H", "C", "H", "H", "H", "H"],
            "geometry": GEOMETRIES["official_input"]["coordinates"],
            "geometry_unit": "angstrom", "masses_amu": physics["masses_amu"],
            "charge": 0, "multiplicity": 2,
        },
        "potential": {
            "provider": "python", "factory": "benchmarks.hch4.pes:build",
            "params": {"library": str(Path(library).resolve())},
            "units": {"positions": "bohr", "energy": "hartree"}, "derivative": "force",
        },
        "reaction": {
            "fragments": physics["fragments_zero_based"],
            "forming_bonds": physics["forming_bonds_zero_based"],
            "breaking_bonds": physics["breaking_bonds_zero_based"],
            "equivalent_paths": physics["equivalent_path_permutations"],
            "rinf": "30 bohr", "electronic_weight": 1.0, "symmetry_factor": 1.0,
        },
        "dynamics": {"timestep": "0.1 fs", "integrator": "cayley-bcb",
                     "loop_backend": "native", "block_steps": 256, "native": "required"},
        "sampling": {
            "thermostat": "andersen", "andersen_schedule": "geometric",
            "andersen_interval": 500,
            "replicas": pilot["replicas_per_window_per_seed"],
            "equilibration": f'{pilot["equilibration_fs"]} fs',
            "production": f'{pilot["production_fs"]} fs',
            "interval": f'{pilot["saved_cv_interval_fs"]} fs',
        },
        "umbrella": {"start": -0.05, "stop": 1.05, "count": 111, "kappa": 30.0},
        "analysis": {"method": "mbar", "measure": "rate_flux", "bins": 5000,
                     "ranges": [[-0.02, 1.02]], "bootstrap": 0},
        "rate": {"pmf_method": "mbar", "dividing_surface": 0.999 if beads == 1 else 1.007,
                 "compare_ui": True, "pmf_search_range": [0.9, 1.02]},
        "recrossing": {
            "parent_replicas": pilot["parent_replicas_per_seed"],
            "parent_samples": pilot["parent_samples_per_replica"],
            "equilibration": f'{pilot["parent_equilibration_fs"]} fs',
            "parent_interval": f'{pilot["parent_interval_fs"]} fs',
            "child_pairs": pilot["child_pairs_per_parent"],
            "child_time": f'{pilot["child_duration_fs"]} fs',
            "plateau_fraction": 0.4, "plateau_tolerance": 0.02,
            "bootstrap": 0, "constraint_tolerance": 1e-10,
        },
        "parallel": {"backend": "process" if workers > 1 else "serial",
                     "trajectory_workers": workers, "threads_per_rank": 1},
        "output": {"directory": str(Path(output).resolve()), "checkpoint_steps": 1000},
    }


def resolve_pilot(library, **kwargs):
    raw = pilot_input(library, **kwargs)
    config, system = JobSpec.from_mapping(raw).resolve()
    prepared = prepare_rate(config, system, numerical=False)
    if not prepared["ready"]:
        raise ValueError(prepared["diagnostics"])
    config["reaction"] = prepared["reaction"]
    return raw, config, system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--beads", type=int, choices=[1, 64, 128, 256], default=1)
    parser.add_argument("--seed", type=int, default=PROTOCOL["campaign"]["seeds"][0])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--write-input", type=Path, required=True)
    args = parser.parse_args()
    raw, config, system = resolve_pilot(args.library, beads=args.beads, seed=args.seed,
                                       output=args.output_directory, workers=args.workers)
    rendered = ('"""Generated H+CH4 pilot; budget is not a literature reproduction."""\n\n'
                + "def build():\n    return "
                + pprint.pformat(raw, width=100, sort_dicts=False).replace("\n", "\n    ") + "\n")
    args.write_input.parent.mkdir(parents=True, exist_ok=True)
    args.write_input.write_text(rendered)
    # Check the exact emitted file through the ordinary user input frontend.
    restored, _ = JobSpec.from_file(args.write_input).resolve()
    assert restored["beads"] == config["beads"] and system.natoms == 6
    print(json.dumps({"input": str(args.write_input), "status": "generated_only_not_run",
                      "beads": args.beads, "seed": args.seed, "rate_validated": False}, indent=2))


if __name__ == "__main__":
    main()
