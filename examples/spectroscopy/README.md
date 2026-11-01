# Spectroscopy workflow example

```bash
pyrpmd validate examples/spectroscopy/pacmd.inp --check-pes
pyrpmd run examples/spectroscopy/pacmd.inp --output runs/pacmd
pyrpmd resume runs/pacmd
pyrpmd analyze runs/pacmd --recompute
```

This short harmonic calculation checks PA-CMD, two independent replicas,
the total-system property interface, and IR/Raman/complex-SFG output files.
The property tensors are synthetic and the run is intentionally short. Its
intensities and bands are not physical predictions for hydrogen.

See [the spectroscopy guide](../../docs/tier2-spectroscopy.md) for units,
mass/thermostat conventions, TorchScript models, checkpoint semantics,
f-CMD force matching and remaining QCMD/h-CMD work. The Python APIs are
`pyrpmd.cmd.PACMDDynamics`, `pyrpmd.properties.PropertyModel`, and the
spectral routines in `pyrpmd.analysis.spectroscopy`.
