# Torch paper-validation measurements

`torch_validation_20261101.json.gz` is losslessly compressed JSON containing all
five completed CPU experiments: stability, sampling, RPC/MTS, resonance, and
joint RPC/MTS/PILE sampling. It includes raw numerical summaries, seeds,
parameters, confidence intervals, environment and the tested backend hash.
The five original JSON objects are retained under their section names. It is
not a trajectory archive and does not contain CUDA or trained-model results.

```bash
python - <<'PY'
import gzip, json
with gzip.open('benchmarks/results/torch_validation_20261101.json.gz', 'rt') as f:
    result = json.load(f)
print(json.dumps(result, indent=2))
PY
```

Tested backend commit: `fdad63de446ca51aed4245823fd1e4a5d2eba4c4`.
No production numerical algorithm changed during this validation.
See [the report](../../docs/torch_validation_report.md) for the unreproduced
published 25% instability statistic, finite-step/RPC bias, CI failure, and scope.
Regenerate measurements with `benchmarks/torch_paper_validation.py`; numerical
samples may change between PyTorch versions/devices even with the same seed.
