"""Execute the original and repaired exchange guards on identical objects.

The old source is read from the reviewed Git commit and executed in a private
module. No production files are replaced, including during concurrent testing.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.base import PESResult, Potential
from pyrpmd.potentials.builtin import FreePotential
from pyrpmd.potentials.lifecycle import CommitteePotential
from pyrpmd.replica_exchange import ReplicaExchange


HEAD = "3faac3aa42b5424b2711367608e0138c82f62e0a"
ROOT = Path(__file__).resolve().parents[3]
old_source = subprocess.run(["git", "show", HEAD + ":src/pyrpmd/replica_exchange.py"],
    cwd=ROOT, check=True, capture_output=True, text=True).stdout
old = types.ModuleType("pyrpmd._independent_old_exchange")
old.__package__ = "pyrpmd"
sys.modules[old.__name__] = old
exec(compile(old_source, HEAD + "/replica_exchange.py", "exec"), old.__dict__)


class ChangingHarmonic(Potential):
    metadata = {"provider": "changing-harmonic-witness"}

    def __init__(self):
        self.calls = 0
        self.time_dependent = False

    def evaluate(self, q):
        self.calls += 1
        k = 1 + 0.1 * self.calls
        return PESResult(0.5 * k * np.sum(q * q, axis=(1, 2)), -k * q)


def witness(cls):
    member = ChangingHarmonic()
    wrapped = CommitteePotential([member, FreePotential()])
    dyn = [Dynamics(wrapped, [1.0], beta, 1, 0.01, native=False) for beta in (1.0, 2.0)]
    states = [State(np.full((1, 1, 3), x), np.zeros((1, 1, 3))) for x in (0.1, 0.3)]
    sampler = cls(dyn, states)
    member.time_dependent = True
    values, error = [], None
    try:
        values = [sampler.log_acceptance_ratio(0, 1) for _ in range(2)]
    except ValueError as exc:
        error = str(exc)
    return dict(log_ratios_for_unchanged_configurations=values, member_pes_calls=member.calls, error=error)


result = dict(reviewed_commit=HEAD, baseline_source_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
              before=witness(old.ReplicaExchange), after=witness(ReplicaExchange))
assert result["before"]["error"] is None
assert result["before"]["member_pes_calls"] == 8
assert result["before"]["log_ratios_for_unchanged_configurations"][0] != result["before"]["log_ratios_for_unchanged_configurations"][1]
assert result["after"]["error"] is not None and result["after"]["member_pes_calls"] == 0
path = Path(__file__).with_name("sampling_composition_witness.json")
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
