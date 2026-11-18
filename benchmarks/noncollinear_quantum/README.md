# Noncollinear bimolecular rates and guarded active-space validation

**Status: research validation, not a generally exact selective-quantum method.**
The executable Cl + HBr study propagates both three-component Jacobi vectors,
including impact parameter, rotation and bending. It does not multiply a
collinear rate by an angular factor. No selectively corrected gas-phase rate
is reported: the conservative independent pilot requests all three internal
directions, and the finite-channel quantum audit has not converged.

## What is implemented

- `pyrpmd.analysis.bimolecular_scattering`: 3D incoming-flux energy sampling,
  impact-area normalization, volumetric rate aggregation, retained failure
  bounds, rovibrational partitions, and explicit parity/(2J+1) CRP integration.
- `pyrpmd.analysis.molecular_collision`: arbitrary-atom fragment assembly and
  an executable Cartesian RPMD workflow accepting an ordinary PES callback.
  Prepared canonical fragment ensembles and a product classifier are explicit
  inputs. No thermal preparation or product definition is guessed.
- `pyrpmd.analysis.bimolecular_active_space`: mass-weighted proper molecular
  alignment; removal of rigid motion; local nonlinear-force and observable
  active-space selection; explicit dimensional-budget failure.
- `pyrpmd.analysis.bimolecular_workflow`: reject unmatched units, observables,
  state conventions and angular-momentum coverage before any rate correction.
  Covariance propagation is sampling-only and is not a transfer-error bound.
- Benchmark: original LEPS parameters, independent finite-P reactant HMC,
  unthermostatted noncollinear RPMD, and finite-J three-internal-coordinate
  hyperspherical quantum scattering including Coriolis angular operators.

The arbitrary-atom API is software-tested on 5- and 7-atom controls. A generic
quantum kinetic-operator builder, nonharmonic quantum bath closure, and a real
polyatomic selective-quantum rate calculation are **not** implemented here.

## Physical model and conventions

Reaction: **35Cl + H79Br -> H35Cl + 79Br**, fixed electronic LEPS surface.
`source.json` pins the historical BrHCl_LEPS1 source and records the parameters.
`leps.py` is an independent analytic energy/gradient implementation, not a new
fit and not a claim of modern electronic-structure accuracy. This run checks
its gradients and independent vector/scalar forms; it did not compile or
re-run the upstream Fortran source. No third-party source or binary is bundled.

Cartesian atom order is Br,H,Cl. Atomic units are used throughout the solvers.
`r = H-Br`, `R = Cl-COM(HBr)` are vectors. All six COM-free components and their
ring-polymer modes are propagated with physical reduced masses. The reactant
Morse paths use an explicit bond domain [0.5,7] bohr. Whole-ring random rotations
are sampling moves only; no rotation moves or thermostat act during dynamics.
This incoming-collision RPMD construction is an approximation, not an asserted
reproduction of a converged equilibrium Bennett--Chandler calculation.

For the 3D collision flux, E has density beta^2 E exp(-beta E), b^2 is uniform
on [0,bmax^2], and

    k(T) = sqrt(8/(pi*beta*mu)) * pi*bmax^2 * mean(w * reactive_indicator).

The warmer Gamma proposal has an explicit likelihood ratio. Failed/unfinished
samples are retained as lower/upper bounds **inside the sampled disk**. The
uncomputed tail beyond bmax is not bounded. Rates are cm^3 molecule^-1 s^-1.
Electronic and symmetry weights are both 1; halogen fine-structure averaging
has not been added.

## Run from a PyRPMD checkout containing this PR

Python 3.11+, NumPy/SciPy and optional Numba are required for this benchmark.
The four core analysis modules do not depend on Numba.

```bash
python -m pip install -e '.[test,plots]'
python -m pip install numba
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q tests/test_noncollinear_bimolecular.py tests/test_noncollinear_kernels.py

python benchmarks/noncollinear_quantum/dynamics.py \
  --input benchmarks/noncollinear_quantum/inputs/smoke.json --output runs/noncollinear-smoke

# Main and sensitivity ensembles: all input files are complete, executable JSON.
for name in T300_P1 T300_P16 T300_P32 T150_P32 T600_P16 T300_wide \
            T300_cayley32 T300_cayley32_half T300_cayley64 T300_far; do
  python benchmarks/noncollinear_quantum/dynamics.py \
    --input benchmarks/noncollinear_quantum/inputs/${name}.json \
    --output results/${name}
done

# Independent restrained-PIMD pilot (not selected from reactive trajectories).
python benchmarks/noncollinear_quantum/pilot_selection.py

# Independent bound rovibrational preparation audit.
python benchmarks/noncollinear_quantum/reactant_audit.py

# Both are fixed-J audits, NOT complete quantum thermal rate calculations.
python benchmarks/noncollinear_quantum/hyperspherical.py \
  --input benchmarks/noncollinear_quantum/inputs/quantum_J0_pilot.json \
  --output results/quantum_J0_pilot
python benchmarks/noncollinear_quantum/hyperspherical.py \
  --input benchmarks/noncollinear_quantum/inputs/quantum_J0_check.json \
  --output results/quantum_J0_check

python benchmarks/noncollinear_quantum/summarize.py
python benchmarks/noncollinear_quantum/report.py
```

Each dynamics run saves the scientific input, outcomes, impact parameters,
energy/importance weights, unfinished trajectories, and conservation diagnostics.
Changing a saved input is rejected. Seeds, all lengths, counts and time steps
are in `inputs/`. These are benchmark-driver JSON files, not native `pyrpmd run`
input syntax. Raw output arrays are generated by the commands and provided in
the downloadable validation archive, not committed as large repository blobs.

### Arbitrary molecular systems

```python
from pyrpmd.analysis.molecular_collision import run_bimolecular_scattering

# q_A,p_A and q_B,p_B: independent canonical COM-free fragment paths,
# each [trajectories, beads, atoms, 3]. Supply a physically justified
# internal-state ensemble; zero coordinates are NOT a molecular preparation.
result = run_bimolecular_scattering(
    (q_A, p_A), (q_B, p_B), masses_A, masses_B,
    temperature=300.0, bmax=6.0, launch_R=14.0,
    potential=potential.evaluate, classifier=product_classifier,
    dt=2.0, steps=200000, seed=81021,
    provenance={"pes_identity": "version/hash of your PES",
                "reactant_preparation": "path to canonical preparation evidence"},
)
```

The callback accepts physical Cartesian bead frames and returns energies and
negative-gradient forces. The classifier must distinguish target product,
non-target exit and still-running trajectories. The bundled optimized triatomic
kernel is checked against the generic Cartesian map on identical configurations.
The general driver does not itself supply a quantum active-space Hamiltonian.

## What the measured run establishes

19,456 full noncollinear collision trajectories were propagated across ten
ensembles. See `docs/validation/noncollinear-20261108/summary.json` and the
self-contained report for the measured statistics, including failures.

- At 300 K, P32 Cayley baseline: (1.897 +/- 0.360)e-12 cm^3 molecule^-1 s^-1.
  Halving dt changes it to (1.801 +/- 0.340)e-12; the paired difference is
  (-0.096 +/- 0.130)e-12. These errors are sampling SE only.
- P16/P32/P64, finite launch, impact cutoff and preparation checks do not yet
  certify complete physical convergence. The wider launch has unfinished
  trajectories and correspondingly wide retained bounds.
- 150/300/600 K independent pilots each request all 3 internal directions.
  All 64 pilot resamples at each temperature also select dimension 3. With a
  budget of 2, the selector explicitly refuses or reports residual coupling.
  This is a conservative local heuristic, NOT proof that two-dimensional
  quantum treatment is physically impossible.
- J=0 CRPs change by about 49.1% in relative L2 under the combined refinement.
  Discrete unitarity is ~4e-12, but that is **not** quantum convergence.
  No full J-summed quantum rate, experimental comparison, or corrected 3D rate
  can be inferred from this audit.

## How full quantum rates would be assembled

`partial_wave_thermal_rate` integrates supplied parity-summed N_J(E) with
(2J+1), the relative translation partition density, and an explicitly matched
internal reactant partition. It rejects gaps in J and does not insert a
J-shifting model. Its dimensionality does not depend on the number of internal
molecular coordinates; however, it does **not** solve the required CRPs.

The atom-diatom audit accepts integer J and both allowed parity blocks, with
Coriolis terms. Only J=0 was run here. Its present finite-radius channel
identification and hyperangular bounds require improvement before using even
that block for a physical benchmark. An all-J campaign must establish energy,
J, parity, asymptote, radial, angular and channel convergence; no threshold
alignment, fitted rate correction or forced unitary S matrix is used here.

`assemble_correction` deliberately blocks results without those reference
checks, a consistent active kinetic metric, and a genuine reduced-space
candidate. It accepts paired sampling covariance but cannot certify the
transfer approximation. All measured results retain `physical_converged=False`.

## Source background

- BrHCl source and legacy conventions: `../brhcl/SOURCE.md` and `source.json`.
- Collepardo-Guevara, Suleimanov and Manolopoulos, J. Chem. Phys. 130,
  174713 (2009), DOI 10.1063/1.3127145: bimolecular RPMD methodology. Our independent collision
  ensemble is not asserted to have reproduced that equilibrium rate algorithm.
- Standard atom-diatom body-fixed scattering requires angular momentum/parity
  channels and Coriolis coupling, not a collinear-to-3D scale factor.

This development is a reviewable implementation and validation checkpoint,
not a declaration that a universal low-dimensional quantum reaction method
has been established.
