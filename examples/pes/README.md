# Compile and check a native PES

Turn a C object file into a PyRPMD potential, inspect it and verify its forces, in four commands
from the checkout root:

```bash
cc -std=c11 -fPIC -I native/include -c native/examples/harmonic_pes.c -o examples/pes/harmonic.o
pyrpmd pes build examples/pes/harmonic.o --manifest examples/pes/pes.toml --output examples/pes/libharmonic.so
pyrpmd pes inspect examples/pes/libharmonic.so --geometry examples/pes/atom.xyz
pyrpmd pes check examples/pes/libharmonic.so --geometry examples/pes/atom.xyz
```

The model is exactly `V = 0.5*(x*x+y*y+z*z)` hartree for one hydrogen-label test atom, with
`F = -q` in atomic units, so every number `inspect` and `check` print can be verified by hand. It
demonstrates the ABI and numerics, not chemistry.

`pes build` refuses to overwrite an existing explicit output file; omit `--output` to reuse the
manifest-driven cache. The ABI is specified in
[native/ABI.md](../../native/ABI.md).
