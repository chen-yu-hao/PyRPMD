# Quantum umbrella MC: complete 30,000-sweep stage

All eight registered exchange ladders completed 5,000 warmup sweeps and
30,000 production sweeps. Their pooled WHAM QTST estimate is
**2.80072503 × 10⁻¹⁶ cm³ molecule⁻¹ s⁻¹**, with a nominal whole-ladder
95% interval **[2.33507586, 3.35923162] × 10⁻¹⁶**. This is 11.583% above
the Table 2 target 2.51 × 10⁻¹⁶ in
[Suleimanov et al., CPC (2013)](https://doi.org/10.1016/j.cpc.2012.10.017).
The precision and paper-point criteria both fail. A broad interval covering
the paper is not evidence of a precise reproduction.

| Registered check | Observation | Status |
| --- | --- | --- |
| Complete cohort | All eight original seeds | Pass |
| Primary log standard error | 0.0768976; target ≤0.03 | **Fail** |
| Point agreement with paper | +11.583%; target approximately ±5% | **Fail** |
| WHAM convergence and overlap | Residual 5.33 × 10⁻¹⁵; connected graph | Pass |
| Bin-width refinement | Maximum log-rate change 0.00345658 | Pass |
| Local-fit refinement | Maximum log-rate change 0.00824291 | Pass |
| Paired temporal-half consistency | Mean log difference −0.199655; SE 0.144556 | Pass under the registered rule |
| Independent-ladder consistency | Approximate heterogeneity p = 0.0899863 | Pass under the registered rule |

The primary error is the larger of the leave-one-ladder-out jackknife
error (0.0766564) and the between-ladder error (0.0768976). All exchanged
windows stay together within a ladder. The largest split Rhat is only
1.0005765; this does not guarantee precise integrated free-energy differences.
The 95% log half-width is 0.181834. Sweeps are Monte Carlo work units and
must not be reported as physical femtoseconds.

The immutable [continuation decision](continuation-decision.json) selects
the next predeclared prefix, **100,000 production sweeps for every ladder**,
because the complete cohort fails statistical precision. The original
potential, mass, temperature, windows, proposal parameters, random streams,
analysis and acceptance thresholds remain fixed. The 30,000-sweep failures
and raw checkpoint/RNG states are retained in the
[verified first-stage archive](raw-archive/README.md).

The reviewed continuation was launched at the recorded pre-exec time
18:34:50 UTC on 2026-11-01, initially admitting two workers. The
[launch receipt](continuation-launch.json) binds the complete first-stage
archive, unchanged scientific decision, exact scheduler and independent
review. The [scheduler evidence](../umbrella-scheduler/README.md) retains
its initial interrupt/lock defect and successful correction. The older
direct driver remains stopped because it does not participate in the new
coordinator lock. Raising the worker-admission limit to four requires
available CPU capacity; it changes no sample, random stream or criterion.

A simple 1/√N planning extrapolation gives a log standard error of about
0.042 at 100,000 sweeps, so the next stage may still miss 0.03. This is a
cost estimate, not a convergence prediction or a reason to skip the
registered assessment. The 300,000-sweep stage can be selected only after
the complete 100,000-sweep cohort is assessed. The continuation uses the
same data prefixes and does not restart selected seeds.

This independent umbrella campaign remains separate from the independent
surface-partition/native-child reference. Its incomplete precision must
not be replaced by agreement from that other estimator. No transmission
coefficient or full RPMD rate is inferred from this WHAM result.

- [Full numerical comparison](campaign-comparison.json)
- [Execution receipt](execution.json)
- [Registered scientific protocol](../protocol.json)
- [Immutable continuation decision](continuation-decision.json)
- [Raw stage snapshot and archive verification](raw-archive/README.md)
