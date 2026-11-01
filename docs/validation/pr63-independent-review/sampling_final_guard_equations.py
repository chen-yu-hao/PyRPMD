"""Recheck numerical exchange after the final guard-only changes.

Earlier stochastic experiments retain their actual execution hashes. They are
not rerun or relabeled by this focused check.
"""
import hashlib
import json
from pathlib import Path

from sampling_experiments import exchange_equations


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "src/pyrpmd/replica_exchange.py"
before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
rows = exchange_equations()
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
earlier = json.loads((OUT / "sampling_results_dt0.04.json").read_text())
assert rows == earlier["replica_exchange"]
result = dict(
    final_replica_exchange_sha256=before,
    stochastic_run_replica_exchange_sha256=earlier["source_sha256"]["src/pyrpmd/replica_exchange.py"],
    matches_earlier_deterministic_results_exactly=True,
    cases=rows,
    note="Post-full-suite cache-cycle and advance-preflight guards only; RETIS source/results unchanged",
)
(OUT / "sampling_final_guard_equations.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
