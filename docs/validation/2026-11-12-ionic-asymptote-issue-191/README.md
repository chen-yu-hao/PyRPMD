# Ionic asymptote path reproduction for PyRPMD issue 191

This bundle reproduces the diagnostic behavior reported in [PyRPMD issue #191](https://github.com/chen-yu-hao/PyRPMD/issues/191) for `H2 + NH2- -> NH3 + H-` at 150 K with 64 ring-polymer beads.

The reproduction uses a saved PyRPMD rate-run directory and its PES adapter. Large or site-specific artifacts are kept out of the repository: the PES shared library, umbrella checkpoints, centroid path arrays, and full rate-run directory must be supplied separately.

The run used PyRPMD commit `f9776b7d6f545357bf8decd5c73d3ca387e636a5`.

## Run

Copy the complete saved centroid-path rate run to a local directory. It must contain normal PyRPMD run files, including `manifest.json`, `input/resolved.json`, `centroid_path.npz`, and the referenced PES adapter. Then run:

```bash
python reproduce.py \
  --rate-run /path/to/returned/path/scientific \
  --output /tmp/pyrpmd-ionic-asymptote-191
```

The script creates a temporary input from `subspace.toml`, runs `python -m pyrpmd subspace-correct`, and writes the report to the requested output directory. The command may return exit code 1 when the validity policy reports `status = invalid_model`; that is the expected scientific diagnostic state for this reproduction.

The input uses no finite `path.max_steps`, `asymptote_tolerance = 1e-6`, `asymptote_patience = 50`, a one-dimensional adiabatic subspace, `validity.on_invalid = "diagnose"`, and 64 beads.

## Expected report

The expected top-level status is `invalid_model`. Both path sides should report `asymptote_not_approached`. The bath gate should report unstable transverse curvature, and the grid gate should report an estimator outside fraction of 1.0. The correction record should contain `valid = false`, `factor = null`, and a diagnostic factor around `0.0145105673`.

The captured machine-readable summaries are in `observed/`:

- `pipeline-summary.json` — path termination and validity summary;
- `correction-summary.json` — component rates and correction validity;
- `rate-run-summary.json` — temperature, bead count, sampling coverage, and PES identity.
