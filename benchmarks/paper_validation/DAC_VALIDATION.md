# Physical validation: Tully double avoided crossing (DAC)

**Status: the published quantum reference is reproducible; one-bead focused
MMST and focused W mapping disagree substantially in the second-crossing and
low-momentum return regimes. This is a documented NEGATIVE physical benchmark,
not a failed numerical integrator test or a claim about converged multibead
NRPMD.**

Reference: Z. Liu, N. Lyu, Z. Hu, H. Zeng, V. S. Batista and X. Sun,
*J. Chem. Phys.* **161**, 024102 (2024),
[DOI 10.1063/5.0208708](https://doi.org/10.1063/5.0208708), Sec. III.D,
Eqs. (19)–(22), (67)–(71), Sec. IV, Fig. 7 and supplementary Fig. S21.
Original data are pinned to
[author repository commit 3951a0a](https://github.com/xiangsunlab/benchmark-data/tree/3951a0a19008fb545ad17f0d2961d934633f7422/Tully).

## Reproduction contract

* Tully model II in a fixed **diabatic** basis:
  `V00=0`, `V11=-0.10*exp(-0.28*x*x)+0.05`,
  `V01=V10=0.015*exp(-0.06*x*x)`, energies in hartree.
* A single nuclear degree of freedom, mass **2000 electron masses**,
  `x0=-9 bohr`, `p0=5` or `15` in atomic units, and fixed diabatic
  electronic state **0**.
* Initial pure Gaussian nuclear wavepacket (its Wigner density is sampled
  independently for every trajectory):
  `Var(x)=Var(p)=1/2`. No fitted or thermostatted nuclear distribution.
* The published population observations are **diabatic state 1**, not
  adiabatic active-state counts. Do not clip signed mapping samples.
* **One nuclear bead** only for this benchmark; `beta=100` in the
  corresponding PyRPMD scalar driver is inactive at `P=1`.
  Cloning the same Wigner sample into several beads would not be a
  valid non-equilibrium ring-polymer preparation.
* The author-provided `Tully_DAC_p15.csv` contains a block of **zero/zero**
  electronic populations beyond t=3000 a.u.; these are not normalized states.
  The pinned `dac_p15_liu2024_reference.csv` stops at **2800 a.u.**
  The p5 benchmark stops at **6000 a.u.** Both datasets are checked to
  contain every 40-a.u. sample with total population one.

## Which approximation is being tested?

For `V=h0*I + hx*sigma_x + hz*sigma_z`, write a two-level mapping spin
`s=(Re(z0*conj(z1)), ..., (|z0|²-|z1|²)/2)`. The exact frozen-coordinate
flow rotates `s` about `h` and integrates the momentum impulse.
This analytical two-state identity is compared **step by step** against the
ordinary `NRPMDDynamics` and `SMNRPMDDynamics` public APIs.

| Label in this benchmark | Electronic spin | Preparation and estimator |
|---|---|---|
| `mmst-focused` / 1-bead NRPMD | `|s|=2` | `s_z(0)=1`, azimuth uniform, `P1(t)=(1-s_z(t))/2`; corresponds to MMST-foc at gamma=1/2, **not** conventional Gaussian LSC |
| `w-focused` / 1-bead SM-NRPMD | `|s|=sqrt(3)` | `s_z(0)=1`, azimuth uniform, same final W projector; corresponds to W-foc |
| `spm-w-fullsphere` | `|s|=sqrt(3)` | uniform **full sphere** and `2*P0_W(s0)*P1_W(st)` with signed initial-projector weights; the paper's **unfocused SPM-W**, Eqs. 19–20. No self-normalization |
| `ehrenfest` | `|s|=1` | pure diabatic initial state without electronic sampling; independent mean-field baseline |

The full-sphere SPM-W and W-focused SM-NRPMD are **not interchangeable**.
The paper's FSSH uses electronic-amplitude populations, whereas the previous
SAC PR #60 used a projected adiabatic active-surface diagnostic. Neither
that SAC projected diagnostic nor an arbitrary active-state count is
substituted for the present diabatic observable.

## Executed large independent replications

Executed locally 2026-11-01 in Python 3.12 / NumPy 2.3.5 with three
independent random-number streams per momentum, **8,192 trajectories per
stream** (`N=24,576`), `dt=4` atomic units. At each momentum the
three batches use different seeds; within a batch focused methods share
the same nuclear parents and electronic azimuth for a paired comparison.
Uncertainties below are **one trajectory SEM**, pooled from the complete
between- and within-batch variance (not SEM across three means only).

| Method | p15: P1 at t=2800 ± SEM | p15 curve RMSE | p5: P1 at t=6000 ± SEM | p5 curve RMSE |
|---|---:|---:|---:|---:|
| Published quantum SOFT | **0.1290114** | — | **0.00069145** | — |
| MMST-focused / P1 NRPMD | **0.224987 ± 0.002641** | 0.06760 | **0.085268 ± 0.001933** | 0.03602 |
| W-focused / P1 SM-NRPMD | **0.210118 ± 0.002284** | 0.05812 | **0.048001 ± 0.001223** | 0.02027 |
| W-full sphere Eq. 20 | **0.206060 ± 0.003911** | 0.05613 | **0.024335 ± 0.003163** | 0.02484 |
| Ehrenfest mean-field | **0.157611 ± 0.000102** | 0.02440 | **0.009470 ± 0.000094** | 0.00531 |

These are non-equilibrium **approximation-vs-quantum** comparisons for the
stated state, not direct reproduction of each original author's plotted
approximate curve. The independent full-sphere and Ehrenfest rows are
validation-only reference implementations.

### Numerical-timestep and exact-source checks

The first 8192 nuclear/electronic initial samples were retained across
all timestep refinements, i.e. **paired deterministic numerical comparisons**.
For the maximum absolute difference between the `dt=4` and `dt=2` mean
curves:

| Case | MMST-focused | W-focused |
|---|---:|---:|
| p15 | 2.35e-5 | 6.47e-6 |
| p5 | 6.95e-5 | 2.59e-5 |

Maximum observed energy deviation scales approximately as `dt²`;
e.g. in p15 MMST-focused the 8192-trajectory maximum went from
`6.56e-5` (dt8), `1.64e-5` (dt4), `4.09e-6` (dt2), to `1.02e-6`
(dt1) hartree. Focused W similarly reduced from `5.73e-5` to
`8.95e-7` hartree. The p5 dt8/dt4/dt2 ratios also agree with this trend.
This does **not** mean classical mapping trajectories have exact quantum
phase coherence.

The analytic vectorized spin-flow diagnostic was **independently compared
against the actual scalar PyRPMD APIs** on seven nuclear parents with
both momentum values and nontrivial initial mapping phases for 1400 steps
at dt=2 a.u. The maximum absolute differences were
`1.66e-12` in the MMST spin variables and `1.28e-13` in W-focused
spin variables; nuclear coordinate and momentum differences were
`<=1.10e-12`. These are numerical implementation checks, not physical
accuracy claims. A shorter version is a permanent pytest regression.

The original published SOFT results have separately been reproduced by
`independent_tully_dac.py`; the original-source comparisons for p5 and
p15 were at the approximately 1e-11 and 1e-12 levels, respectively, in
the validated windows. GitHub Actions checks the pinned data and reruns
the independent quantum and PyRPMD comparison.

## Commands

After installing the project:

```bash
python -m pytest -q tests/test_dac_paper_validation.py

python benchmarks/paper_validation/independent_tully_dac.py \
  --case both --output runs/dac_quantum_reference.json

python benchmarks/paper_validation/dac_spin_statistical.py \
  --momentum 15 --n 8192 --dt 4 --seed 731 --replica 0 \
  --output runs/dac_p15_r0.json --raw

python benchmarks/paper_validation/dac_spin_statistical.py \
  --momentum 5 --n 8192 --dt 2 --seed 731 --replica 0 \
  --output runs/dac_p5_dt2_r0.json
```

Run replica numbers `0,1,2` at the same `n=8192` and `dt=4` for the
three-repeat results. The script saves per-time trajectory mean and SEM,
nuclear Wigner/PES/source identities, and explicit `physical_converged=false`.
With `--raw`, it additionally saves per-trajectory observations in an NPZ
container. RMSE is calculated over *every* 40-a.u. reference time, not just
a selected endpoint. No acceptance threshold is imposed on approximate
semiclassical agreement.

**Decision:** These trajectories demonstrate a statistically resolved and
timestep-stable **negative physical benchmark**. The remaining discrepancy
should be interpreted for the explicitly stated electronic preparation and
single-bead approximation. It does not show that the source code is defective,
that all choices of SQC/SPM fail, or that multi-bead physical convergence has
been established. More molecular and multi-mode paper benchmarks are still
needed.
