# Quickstart: a restartable free-energy calculation

In a few seconds you will go through the full PyRPMD cycle that every task shares: validate an
input, run umbrella sampling, read the results, restart and reanalyze without rerunning dynamics.
The built-in harmonic potential means no external model is needed.

## 1. Run your first task

After [installation](installation.md), open a terminal in your working directory. The starter is
bundled with PyRPMD, so this works with a wheel or a source installation:

```bash
pyrpmd init my-first-run
pyrpmd validate my-first-run/quickstart.inp --check-pes --output my-first-run/run
pyrpmd run my-first-run/quickstart.inp --output my-first-run/run
pyrpmd analyze my-first-run/run
```

`init` writes `quickstart.inp` into a new or empty directory and prints the next commands. It
refuses a nonempty destination, so use another name when starting a separate example. If your
shell cannot find `pyrpmd`, use `python -m pyrpmd` with the same arguments.

`validate` checks the input, units, task restrictions and resources; `--check-pes` also constructs
the potential and performs the numerical preparation. `run` performs the calculation, prints a JSON
summary and writes the run directory. `analyze` reads the saved summary.
The validation report shows the resolved `input`, `output` and `preset`; `validate --output` previews
the run destination without creating it. The starter's `output.directory="run"` resolves next to
the input, so the explicit `--output my-first-run/run` above selects the same location.

The run directory looks like this:

```text
my-first-run/run/
├── manifest.json          # software, dependencies, native capabilities, physics fingerprint
├── input/                 # original.inp, resolved.json (resolved config), system.json
├── checkpoints/           # window_00000_replica_0000.npz … (one per window and replica)
├── samples/  logs/
├── fes.csv  fes.npz  fes.h5
├── summary.json           # summary written by the workflow
└── results/summary.json   # public summary: adds software version, output path, preset, input notes
```

`analyze` and the Python API read `results/summary.json`. `run` refuses an existing non-empty
directory, so pick a new `--output` each time you start over.

## 2. The input

The example computes the **x-component centroid probability density** of one H nucleus in a 3D
harmonic potential. Three umbrella windows each run two independent replicas: 20 equilibration
steps, then 100 production steps with a sample every 5 steps, using four beads and a 0.1 fs step.

```{literalinclude} ../../src/pyrpmd/templates/quickstart.inp
:language: toml
```

Source file: [bundled quickstart.inp](https://github.com/chen-yu-hao/PyRPMD/blob/main/src/pyrpmd/templates/quickstart.inp).

| Field | Meaning |
| --- | --- |
| `task="fes"` | MBAR reconstruction from the saved umbrella samples |
| `beads=4` | Each atom is a ring of four beads |
| `potential.params.k=0.002` | Force constant of $V=\tfrac12 k\lvert q\rvert^2$ in atomic units |
| `cvs.estimator="centroid"` | The CV is evaluated at the ring centroid |
| `umbrella.stiffness` | Force constant of the physical umbrella; PyRPMD applies the bead scaling |
| `sampling.replicas=2` | Independent random streams per window |
| `analysis.bootstrap=0` | No bootstrap error bars for this run |

Write temperatures, lengths, times and force constants as strings with units. Atom indices start
at 0. The full rules are in [Input](input.md) and [Scientific conventions](science.md).

## 3. Read the results

```python
import json
from pathlib import Path

run_dir = Path("my-first-run/run")
summary = json.loads((run_dir / "results/summary.json").read_text())
for key in ("task", "dimension", "windows", "streams", "samples",
            "overlap_connected", "converged", "warnings", "files"):
    print(key, summary.get(key))
```

The summary reports `task="fes"`, a one-dimensional CV, 3 windows, 6 sampling streams and 120
saved samples; `effective_samples_per_stream` gives the number of statistically independent ones.
`overlap_connected=true` means all windows share one global normalization. `warnings` flags
windows with too few effective samples and unsampled grid cells. `converged` stays `false` for a
single run by design: convergence is established across the series of runs described in
[Scientific conventions](science.md).

`summary["files"]` holds the paths of the data files. In the FES table, `free_energy_hartree` is a
relative free energy, `raw_count` the number of samples per cell, and `masked` marks cells without
sampling support, which hold NaN. See [Output and analysis](results.md).

## 4. Restart and reanalyze

```bash
pyrpmd resume my-first-run/run
pyrpmd analyze my-first-run/run --recompute --bins 8 --bootstrap 10
```

`resume` continues an interrupted run from its last committed state; a finished run stays as it is.
`--recompute` rebuilds the analysis from the saved samples with new bins or bootstrap settings,
without running dynamics. `--bins` and `--bootstrap` imply `--recompute`. The new analysis
overwrites `fes.*` and `results/summary.json`; copy them first if you want to keep the old version.

The saved input defines the run identity. To change the bead number, temperature, potential, time
step or sampling, start a new run directory.

## 5. The same task from Python

Save as `my_first_run.py` in the working directory containing `my-first-run/`:

```python
from pyrpmd import JobSpec, prepare, run, analyze_run, resume_run


def main():
    spec = JobSpec.from_file("my-first-run/quickstart.inp")
    report = prepare(spec, numerical=True)
    print("ready:", report["ready"])
    summary = run(spec, output="my-first-run/python-run")
    print(summary["samples"])
    resume_run("my-first-run/python-run")
    print(analyze_run("my-first-run/python-run")["task"])


if __name__ == "__main__":
    main()
```

The `if __name__ == "__main__"` guard lets PyRPMD use spawn-based worker processes. Python and the
command line share the same parser and workflows; `run`, `prepare`, `resume_run` and `analyze_run`
are the high-level entry points.

## Next steps

For step-by-step control of the integrator, see the low-level example in the [API](api.md). To use a
real model, [connect a potential](potentials.md) and pick a workflow:
[gas phase](workflows.md), [materials and surfaces](periodic.md) or [PCET](pcet.md).
