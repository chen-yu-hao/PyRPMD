# Quickstart: from input to a restartable free-energy calculation

This chapter uses the built-in harmonic potential, so no external model is needed. The goal is to
run one input check, sampling, analysis and restart, and to learn how to judge the output.
**The short trajectories only demonstrate the workflow; they are not a converged free-energy
result.**

## 1. Run your first task

From the root of the installed source repository (`runs/manual-first` must be a new, empty
directory):

```bash
pyrpmd validate examples/manual/quickstart.inp --check-pes
pyrpmd run examples/manual/quickstart.inp --output runs/manual-first
pyrpmd analyze runs/manual-first
```

`validate` checks the input, units, task restrictions and resource settings; `--check-pes` also
constructs the potential and performs the numerical preparation. `run` performs the calculation,
prints a JSON summary in the terminal and stores the configuration, samples and checkpoints in
the run directory. `analyze` reads the saved summary by default and adds no trajectories. The
whole sequence takes a few seconds on an ordinary laptop.

After the run, the directory looks like this:

```text
runs/manual-first/
├── manifest.json          # software, dependencies, native capabilities, physics fingerprint
├── input/                 # original.inp, resolved.json (resolved config), system.json
├── checkpoints/           # window_00000_replica_0000.npz … (one per window and replica)
├── samples/  logs/
├── fes.csv  fes.npz  fes.h5
├── summary.json           # summary written by the workflow itself
└── results/summary.json   # public summary: adds software version, output path, preset, input notes
```

`analyze` and the Python example below read `results/summary.json`. `run` never overwrites an
existing non-empty directory; use a new `--output` when you practice again.

## 2. Read the complete input

The example studies the **x-component centroid probability density** of one H nucleus in a 3D
harmonic potential. Each of three umbrella windows has two independent replicas; each replica
equilibrates for 20 steps, then runs 100 production steps and saves a sample every 5 steps. Four
beads and a 0.1 fs time step are teaching choices.

```{literalinclude} ../../../examples/manual/quickstart.inp
:language: toml
```

Source file: [quickstart.inp](https://github.com/chen-yu-hao/PyRPMD/blob/main/examples/manual/quickstart.inp).

| Field | Meaning in this example |
| --- | --- |
| `task="fes"` | MBAR reconstruction from the saved umbrella samples |
| `beads=4` | Each atom is represented by four ring-polymer beads |
| `potential.params.k=0.002` | Force constant of $V=\tfrac12 k\lvert q\rvert^2$ in atomic units |
| `cvs.estimator="centroid"` | The CV acts on the ring centroid; not interchangeable with a bead-averaged nonlinear CV |
| `umbrella.stiffness` | Force constant of the physical umbrella; the program applies the bead scaling of the ring Hamiltonian |
| `sampling.replicas=2` | Independent random streams per window, not the number of beads |
| `analysis.bootstrap=0` | No bootstrap error bars in this example; a missing error is not a zero error |

Write temperatures, lengths, times and force constants as strings with units whenever possible.
Atom indices start at 0; see [Input](input.md) and [Scientific conventions](science.md) for the
complete rules.

## 3. Check the results, not just the exit code

```python
import json
from pathlib import Path

run_dir = Path("runs/manual-first")
summary = json.loads((run_dir / "results/summary.json").read_text())
for key in ("task", "dimension", "windows", "streams", "samples",
            "overlap_connected", "converged", "warnings", "files"):
    print(key, summary.get(key))
```

This example should report `task="fes"`, a one-dimensional CV, 3 windows, 6 sampling streams and
120 saved samples. The total sample count is not the number of independent samples.
`overlap_connected` tells whether the windows can be joined into one global normalization;
`warnings` may mention too few effective samples and unsampled grid cells. `converged` stays
`false` because no convergence series over bead number, time step, trajectory length and window
coverage was performed.

Use the paths in `summary["files"]` to find data files. In the FES table, `free_energy_hartree`
is a relative free energy, `masked` marks unsupported regions and `raw_count` is the raw count.
A masked region must not be filled in as a zero barrier. See [Output and analysis](results.md).

## 4. Restart and reanalyze

```bash
pyrpmd resume runs/manual-first
pyrpmd analyze runs/manual-first --recompute --bins 8 --bootstrap 10
```

`resume` on a finished run leaves it in its final state and adds no production steps; an
interrupted run continues from its committed state. `--recompute` only reanalyzes the saved
samples and can change the FES binning and number of bootstrap replicates; it does not improve
sampling coverage or extend the dynamics. Passing `--bins` or `--bootstrap` triggers
reanalysis automatically. The new analysis overwrites `fes.*` and `results/summary.json`, so keep
a separate copy of any version you need.

To change the bead number, temperature, potential, time step or sampling scheme, create a new run
directory. Editing a saved `resolved.json` or checkpoint to run longer breaks the run identity;
do not extend a task that way.

## 5. The same task from Python

Save the following as `my_first_run.py` in the repository root, using a new output directory:

```python
from pyrpmd import JobSpec, prepare, run, analyze_run, resume_run


def main():
    spec = JobSpec.from_file("examples/manual/quickstart.inp")
    report = prepare(spec, numerical=True)
    print("ready:", report["ready"])
    summary = run(spec, output="runs/manual-python")
    print(summary["samples"])
    resume_run("runs/manual-python")
    print(analyze_run("runs/manual-python")["task"])


if __name__ == "__main__":
    main()
```

The `if __name__ == "__main__"` guard protects the entry point so that spawn-based process
parallelism can be enabled later. Python and the command line use the same input parser and
workflows. PyRPMD's high-level entry points are `run`, `prepare`, `resume_run` and
`analyze_run`; do not carry PySCF's `kernel()` idiom over to this project.

## Next steps

For full control over integration steps, read the complete low-level example in the
[API](api.md). For a real model, first learn how to [connect a potential](potentials.md), then
choose the [gas-phase workflows](workflows.md), [materials and surfaces](periodic.md) or
[PCET](pcet.md).
