# Compile and check a native PES

Run from the checkout root:

```bash
cc -std=c11 -fPIC -I native/include -c native/examples/harmonic_pes.c -o examples/pes/harmonic.o
pyrpmd pes build examples/pes/harmonic.o --manifest examples/pes/pes.toml --output examples/pes/libharmonic.so
pyrpmd pes inspect examples/pes/libharmonic.so --geometry examples/pes/atom.xyz
pyrpmd pes check examples/pes/libharmonic.so --geometry examples/pes/atom.xyz
```

The model is exactly V=0.5*(x*x+y*y+z*z) hartree for one hydrogen-label test
atom, F=-q in atomic units. It is an ABI/numerics example, not a chemical PES.
The build command refuses to overwrite an existing explicit output file.
The manifest-driven cache can be reused without `--output`.
