# Instanton anchor of the subspace correction

Tests for [docs/subspace-instanton-anchor.md](../../docs/subspace-instanton-anchor.md): does
the steepest-descent factor of the full-dimensional instanton, `eps_SD = 2 pi / (beta hbar eta_1)`,
predict the RPMD defect and the transfer error of a subspace model? The results are reported in
[docs/validation/2026-11-14-subspace-instanton-anchor](../../docs/validation/2026-11-14-subspace-instanton-anchor/README.md).

| file | content |
| --- | --- |
| `models.py` | 2D valley models M1-M4 (zero-point change, curved path, both, slow bath), symmetric Eckart barrier |
| `greens.py` | exact rates from N(E): recursive Green's function with exact lead self-energies, no absorbing potential |
| `reductions.py` | minimum-energy path and 1D reductions (`mep`, `fe`, `fe+sct`, `bare`) and their skeletons |
| `instanton.py` | ring-polymer instanton, unstable-mode frequency `eta_1` |
| `run_qm.py` | deterministic: exact 2D and 1D rates, steepest-descent factors -> `results/qm.json` |
| `run_skeleton.py` | deterministic: steepest-descent factors of the skeletons -> `results/skeleton.json` |
| `run_rpmd.py` | RPMD rates of a valley model and its reductions with `pyrpmd.analysis.subspace_rpmd.ring_polymer_rate` -> `results/rpmd_<model>_<T/Tc>.json`; every invocation adds one independent run |
| `run_eckart.py` | RPMD defect of a symmetric Eckart barrier `eps(u, v)` -> `results/eckart_u<u>_v<v>.json` |
| `report.py`, `README.template.md` | the validation report, generated from `results/` |

```bash
python run_qm.py results/qm.json              # about 45 min, mostly the 2D rates of M4
python run_skeleton.py results/qm.json results/skeleton.json
python run_rpmd.py M2 0.8 results/qm.json results/rpmd_M2_0.8.json          # full model and reductions
python run_rpmd.py M2 0.8 results/qm.json results/rpmd_M2_0.8.json full     # one more run of the full model
python run_eckart.py 2pi/0.8 2.1 results/eckart_u7.854_v2.100.json
python report.py                              # writes the validation report
python -m pytest -q ../../tests/test_subspace_instanton.py
```

The tests are campaign tests (`-m campaign`): they import this directory and read `results/`.

## Provenance

The model family (M1-M4 and their parameters), the Green's-function reference and the idea
of a universal Eckart table come from a code package (`subspace_rpmd_code`) supplied for
comparison. Everything here is an independent implementation; nothing from the package is
executed or copied. Its table of `k_QM / k_RPMD` for the symmetric Eckart barrier was used as
independent data to check this repository's RPMD solver and the steepest-descent factor (see
the status table in the theory page); the table itself is not part of this repository.
