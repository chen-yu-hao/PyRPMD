# Full-PME quantum water: prepared compute jobs

**These jobs have not been launched.** They are an external OpenMM RPMD reference with the complete q-TIP4P/F force field evaluated on every bead. The existing classical campaign is unchanged. No GPU execution, accuracy or speed is claimed.

The [protocol](protocol.json), [explicit job list](jobs.json), and twenty CPU configuration files specify a paper-sized experiment. The separate [CUDA double example](configs/seed-2101-cuda-double-untested.json) makes a possible future device selection concrete; it is not a tested hardware configuration or a resource request.

## Fixed physical experiment

| Quantity | Template |
|---|---|
| System | 216 flexible H2O molecules, 32 beads |
| State | 298 K, 0.997 g/cm³, fixed cubic cell |
| Parameters | Printed q-TIP4P/F Table I; explicitly declared H-1/O-16 masses |
| Forces | Original full 9 Å PME and oxygen LJ on every bead; no contraction or dispersion tail correction |
| Integrator | Native OpenMM RPMD; single 0.1 fs step; no constraints or MTS |
| Independent chains | Twenty, seeds 2101–2120 |
| New preparation | 100 ps per chain |
| Preparation thermostat | Global geometric Andersen refresh, exact mean 100 fs on the step grid |
| Production | 100 consecutive 4 ps NVE segments per chain, 8000 ps total |
| Momentum initialization | Each bead Maxwell at P × T; only total ring COM momentum removed at each NVE segment boundary |
| Recorded observables | Molecular COM centroid position/velocity and oxygen centroid position every 5 fs; scalar thermodynamics and intramolecular geometry |
| Restart | Nominally each 1 ps, or earlier when a wall budget is reached; every native block is at most 5 fs |

The initial physical geometry is the retained classical 28 ps endpoint, replicated onto all 32 beads. Each chain then receives new momentum draws and a full 100 ps preparation. A common initial geometry still requires liquid-structure, internal-mode, and chain-mixing diagnostics. Independent seeds alone do not certify independent equilibrium samples.

This is a diffusion/structure template. It does not record a dipole time series or implement the paper's separate PA-CMD infrared protocol. A 5 fs record has a Nyquist wavenumber of approximately 3336 cm⁻¹, so these records cannot establish the OH-stretch IR spectrum. Infrared validation needs its own propagator/mass and dipole-recording design.

The target is quantum H2O diffusion **0.221 ± 0.001 Å²/ps**, where the quoted uncertainty is the paper's standard error. The source is Habershon, Markland and Manolopoulos, *J. Chem. Phys.* **131**, 024501 (2009), Table III, [DOI 10.1063/1.3167790](https://doi.org/10.1063/1.3167790), [author manuscript](https://arxiv.org/pdf/1011.1047). The paper uses 32 beads, refined Coulomb contraction with a 5 Å splitting length, and 0.5/0.1 fs intermolecular/intramolecular steps. This template instead supplies the full uncontracted Hamiltonian at a conservative single time step. It does not claim identical original executable inputs, isotope-mass rounding or contraction implementation.

## Geometric collision clock and the canonical ensemble

The new runner deliberately avoids a fixed collision period. With a 0.1 fs integration step, its waiting time is geometric on positive integers with probability `p = 0.001`; its mean is exactly 1000 steps, or 100 fs. Waiting times and Gaussian momenta use the same checkpointed NumPy RNG. A native integration block stops at each scheduled collision deadline.

For a P-bead Hamiltonian sampled at inverse temperature `beta/P`, atomic bead momenta have variance `m × P × kB × T`. Consequently their centroid has the physical-temperature variance `m × kB × T`. The NVE projection subtracts the same mass-weighted translational velocity from all beads, removing only three total translation modes. Removing each bead's COM momentum independently would also suppress internal ring modes; this runner does not do so.

Momentum refresh calls only the per-copy velocity setter. Physical positions are not converted out and back into engine units at a collision. The native PILE thermostat remains disabled in both preparation and production, leaving every stochastic draw under the explicit external RNG/clock contract. This is a declared global geometric heat bath; long-run ensemble accuracy and sufficient preparation still need to be measured.

## What the restart contract actually preserves

[native_quantum.py](../../../../../benchmarks/water/native_quantum.py) writes an initial step-0 checkpoint before propagation and commits subsequent chunks atomically. Each checkpoint contains:

- Raw positions and velocities for **every copy**, including virtual M sites, plus redundant physical canonical q/p and recomputable forces.
- The native Context blob, which supplements the explicit copies and is not assumed to serialize them by itself.
- The absolute next Andersen collision step, the external RNG state, the physical step, active segment, and phase.
- Observation records and collision/segment-refresh event traces since the previous committed checkpoint.

Resume verifies all committed checksums, the source/configuration/environment identities, the initial geometry hash, the model XML, the explicit integrator specification, and the actual platform properties. After loading the Context blob, it explicitly restores every copy and requires exact raw q/v and redundant canonical q/p identity. Independent bounds apply to recomputed forces. The GPU force tolerance is explicitly unvalidated until it has been measured on the selected hardware. A failed restore never silently changes platform, precision, seed, or physical settings.

The fixed preparation clock is retained after the transition to NVE but is no longer active. A checkpoint at the preparation boundary or a segment boundary records that the next NVE momentum refresh is still pending. The saved active segment prevents an extra draw after resume.

On OpenMM 8.6.1, `RPMDIntegrator` has no registered XML serialization proxy. The runner therefore stores and verifies **integrator.json**, containing the actual copy count, temperature, time step, thermostat flag, friction, seed, force-group mask, constraint tolerance and empty contraction map. The system remains in standard `model.xml`. The source/runtime identity and the explicit specification reconstruct the integrator; an unsupported XML serialization is not disguised as a successful engine checkpoint.

The official API describes the per-copy accessors, optional contraction, thermostat switch, and total ring energy in [RPMDIntegrator](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.RPMDIntegrator.html). Platform precision and property meanings are documented in the [OpenMM platform guide](https://docs.openmm.org/latest/userguide/library/04_platform_specifics.html).

## Completed small-system tests and their limits

[Twelve focused tests](../quantum-runner-pytest.xml) passed; the [source and result receipt](../quantum-runner-focused-gate.json) records their identities. They cover:

- Bead-Maxwell and centroid-temperature scaling, with only the total ring COM projected.
- Geometric deadline/RNG continuation across preparation, first production, and later segment boundaries.
- A Reference-platform full-PME native q/p map compared with independent PyRPMD normal-mode BAB, with the previously identified OpenMM spring-frequency convention isolated.
- Native full ring energy compared with the independent potential, kinetic and spring energy after separately accounting for the unit-constant factor.
- Actual Reference and CPU all-copy raw-state restoration, altered-force rejection, and rejection of corrupt completion/model identities before constructing a Context.

The atomic-unit and OpenMM kinetic/spring energy conversions differ by the independently computed factor `VELOCITY_NM_PS² / (AMU_TO_AU × HARTREE_KJ_MOL) = 0.9999999989488311`. Potential energy does not receive this factor. The strict energy test applies that derivable conversion instead of increasing its tolerance. The earlier OpenMM hbar convention difference is documented in the [parent audit](../README.md#native-rpmd-convention).

These are implementation and numerical tests on small systems. They do not establish source-sized liquid equilibration, diffusion accuracy, P32 convergence, GPU compatibility, or future bitwise trajectory replay. No quantum production trajectory was generated by these tests.

## Review and run one bounded job

From the repository root, inspect a concrete job without creating any OpenMM Context:

```bash
.venv/bin/python benchmarks/water/native_quantum.py \
  --config docs/validation/literature-convergence-2026-11-01/water/quantum-template/configs/seed-2101-cpu.json \
  --describe
```

The equivalent `--describe` command also accepts the CUDA example and performs no GPU work. Actual GPU selection is explicit; there is no fallback to another platform.

For a bounded invocation on an available, validated allocation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
.venv/bin/python benchmarks/water/native_quantum.py \
  --config docs/validation/literature-convergence-2026-11-01/water/quantum-template/configs/seed-2101-cpu.json \
  --output ../water-quantum-full-pme/seed-2101 --wall-seconds 1800
```

Use the same command with `--resume` to continue a committed run. `--max-steps` additionally bounds a numerical pilot. The wall budget is checked between native blocks of at most 50 steps; the last block and checkpoint I/O can overrun the nominal budget. The remaining fixed scientific trajectory is not shortened by a compute boundary.

The twenty jobs are listed individually. Nothing in the template allocates cloud resources, launches a scheduler, or submits twenty concurrent simulations. The CPU examples are a reproducible reference specification, not a claim that a paper-sized CPU campaign is affordable in the present shared session.

## Required evidence before a diffusion claim

The primary estimate is the molecular COM centroid VACF integral divided by three to 2 ps. The secondary estimate is the 1–2 ps MSD slope divided by six. Each chain averages all of its fixed NVE segments; confidence intervals use the twenty equally weighted independent chain means. Neither molecules, overlapping time origins, nor neighboring segments are counted as independent replicas.

The fixed protocol asks for a diffusion standard error no larger than 0.001 Å²/ps and for the entire paired-chain 95% confidence interval of each prespecified estimator/window difference to lie inside ±0.01 Å²/ps. These are review targets, not automatic certification. Energy drift, temperature/geometry stationarity, quantum RDFs, internal-mode thermalization, production-half stability, and the actual diffusive time regime must also be checked. Separate P64 and 0.05 fs property controls remain necessary; good short energy behavior is insufficient.

The current classical cadence control cannot automatically certify quantum recording error. A quantum cadence control and float32-storage check must be specified before a high-precision transport comparison. Future GPU work additionally requires identical-configuration force/energy comparison, native-map and stochastic-restart checks on the selected device, and measured source-sized memory/throughput. No speed or accuracy is inferred from the hardware name.

Storage is part of the concrete job: centroid trajectory arrays alone occupy about 12.4 GB before compression over 8000 ps, and the redundant all-copy restart records add roughly 30 GB over 10000 ps at 1 ps intervals. Actual compression and metadata should be measured in a short preflight. Raw output stays outside the Git repository; Git stores the code, fixed configurations, protocol and compact evidence.
