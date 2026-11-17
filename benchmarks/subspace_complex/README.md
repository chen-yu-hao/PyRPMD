# High-dimensional complex PES validation for subspace & instanton anchoring

Extends [benchmarks/subspace_instanton](../subspace_instanton/README.md) and
[docs/subspace-instanton-anchor.md](../../docs/subspace-instanton-anchor.md) from 2D symmetric
valleys to 6D–10D complex potential energy surfaces with non-planar curvature/torsion,
pre-reaction van der Waals wells, deep pre- and post-reaction ion–dipole complexes, submerged
central barriers ($V_{\mathrm{TS}} < V_{\mathrm{asymptote}}$), and full 9D 3-atom Cartesian
molecular coordinates (`rigid_body = True`, `RPMDrateCoordinate`).

| file | content |
| --- | --- |
| `models.py` | Analytic high-dimensional PES suite: `TwistedCurved` (`twisted_10d`), `PreReactionWell` (`prereaction_well_8d`), `PreReactionComplex` (`prereaction_complex_8d`, `submerged_complex_6d`), dense $SO(D)$ `RotatedEmbedding`, and 9D 3-atom Cartesian `LepsIonComplex9D` (`leps_ion_complex_9d`) |
| `run_shard.py` | 4-shard multi-session benchmark runner (1D MEP, 1D+SCT, 2D RPH, full-dimensional `full_instanton_defect`, skeleton vs naive instanton anchoring, and exact 2D/3D full-core quantum/RPMD references) plus `--merge` -> `results/complex_pes.json` |
| `report.py` | Markdown summary tables and aggregate RMS / max log-rate residuals from `results/complex_pes.json` |

## PES topologies covered

1. **Non-planar curved + slow-torsion valley (`twisted_10d`, $D=10$)**: 3D reaction valley with
   offset planar bend in $y$ and out-of-plane torsion + barrier-top mode softening in $z$
   ($\omega_z \approx 0.70\,\omega_b$), embedded with 7 orthogonal bath modes and rotated across
   all 10 coordinates by a dense orthogonal matrix $Q \in SO(10)$. Tests the strict hierarchy
   $A_H(10\mathrm{D}) > A_{\mathrm{RPH}^0}(2\mathrm{D}) > A_{1\mathrm{D}^0}(1\mathrm{D}) > 0$
   against exact 3D `flux_eigenstates` quantum and 3D RPMD reference rates.
2. **Barrier + pre-reaction van der Waals well (`prereaction_well_8d`, $D=8$)**: Asymptotic free
   reactant channel $\to$ localized pre-reaction well $\to$ curved activation barrier $\to$
   exothermic product channel, rotated in $SO(8)$. Exercises constrained $-\nabla\xi$ well-climb
   path assembly (`accepted == "reactant_reference"`), dome-bounded P-RFO instanton seeding, and
   1D/2D skeleton anchoring at $200\,\mathrm{K}$ and $175\,\mathrm{K}$.
3. **Deep pre- and post-reaction ion–dipole complexes (`prereaction_complex_8d`, $D=8$, and
   `submerged_complex_6d`, $D=6$)**: Curved transfer barrier flanked by deep pre- and
   post-reaction complexes (`accepted == "product_well"` on the product side) with an exothermic
   dissociation channel, covering both positive ($V_{\mathrm{TS}} > 0$) and **submerged**
   ($V_{\mathrm{TS}} < 0 = V_{\mathrm{asymptote}}$) central barriers via topographic-prominence
   saddle detection.
4. **Full 9D 3-atom Cartesian molecular complex (`leps_ion_complex_9d`, $D=9$)**: 3-atom LEPS
   transfer surface in 3D space ($\mathrm{H + TH \to [H\cdots T{-}H] \to [H{-}T\cdots H] \to HT + H}$)
   with damped $-C_{\mathrm{ion}} / (r^4 + R_0^4)$ pre-/post-reaction complexes and collinear
   hydrogen-bond alignment, exact 3D translational/rotational invariance (`rigid_body = True`),
   and `RPMDrateCoordinate`, cross-checking `full_instanton_defect` against
   `pyrpmd.instanton.optimize_instanton` at $220\,\mathrm{K}$ and $180\,\mathrm{K}$.

## Corrections to the first results

- `submerged_complex_6d_200K`: the barrier is 4.7 kT below the reactant asymptote on the valley floor
  (8 kT in centroid free energy), so an RPMD dividing surface at it is not the bottleneck. The first run gave an
  RPMD rate 2300 times the free-particle flux and `ln C_exact = -8.67`. `capture_surface` in
  `run_shard.py` now puts the RPMD surface at the entrance-channel bottleneck
  (`rpmd_surface_offset`, `rpmd_md_time` in the case spec), giving `-0.893 +- 0.021`; the
  pipeline's `surface` validity gate refuses the old construction.
- Both deep-complex cases bend more tightly at the saddle than the transverse mode is wide; the
  RPH model's `ribbon` gate rejects them, and the `plane` model, which holds a rotated 2D core
  exactly, is included (`models` in the case spec). Its RPMD rate agrees with the exact one; its
  quantum rate is still 16-26% off in 8D (see the theory page).
- The 8D exact quantum rate was checked independently with the recursive Green's function of
  `benchmarks/subspace_instanton/greens.py` (valley form `v(x) + w(x)^2 (y - g(x))^2 / 2`,
  6400 energy points, two grid spacings): `5.51e-7` against `5.74e-7` here.

## Reproducing the benchmark

```bash
# Run shards 0..3 across 4 multi-core sessions:
for k in 0 1 2 3; do
  python -m benchmarks.subspace_complex.run_shard \
      --shard-index "$k" --num-shards 4 --out-dir "/tmp/complex_shards/$k"
done

# Merge all shards and print Markdown validation tables:
python -m benchmarks.subspace_complex.run_shard \
    --merge /tmp/complex_shards \
    --out benchmarks/subspace_complex/results/complex_pes.json
python -m benchmarks.subspace_complex.report \
    benchmarks/subspace_complex/results/complex_pes.json
```

Nothing in this benchmark or in `anchor = "skeleton"` marks a rate as physically converged
(`exact_general_rate_claim = False`).
