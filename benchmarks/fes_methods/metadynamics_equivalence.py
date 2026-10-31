"""Verify bitwise trajectory equivalence with a saved old bias.py.

The adapter adds only the new history-count/rollback API to the old bias.
Use a fresh --output directory to perform propagation instead of resuming.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pyrpmd.bias import MetadynamicsBias
from pyrpmd.system import System
import pyrpmd.workflows.metadynamics as workflow
from metadynamics_performance import load_before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    args = parser.parse_args()
    before_class = load_before(args.before_source)

    class BeforeAdapter(before_class):
        @property
        def hill_count(self):
            return len(self.hills)

        def truncate_history(self, count):
            del self.hills[count:]

    records = []
    try:
        for mode in ("standard", "well_tempered"):
            config = dict(task="metadynamics", beta=4., beads=4, dt_au=.04, seed=731, native="required",
                          potential=dict(provider="builtin", name="harmonic", params=dict(k=1.)),
                          sampling=dict(equilibration_steps=200, production_steps=args.steps, sample_interval=10,
                                        andersen_interval=40, replicas=1),
                          cvs=[dict(name=a, kind="cartesian", atom=0, axis=a) for a in ("x", "y")],
                          metadynamics=dict(mode=mode, sigma=[.25, .25], height=.025, pace=60,
                                            **(dict(bias_factor=5.) if mode == "well_tempered" else {})),
                          analysis=dict(bins=[12, 12], bootstrap=0), output=dict(checkpoint_steps=100000))
            system = System(("H",), np.array([1.]), np.zeros((1, 3)))
            streams = []
            for label, cls in [("before", BeforeAdapter), ("after", MetadynamicsBias)]:
                workflow.MetadynamicsBias = cls
                streams.append(workflow._sample_replica(dict(config=config, system=system, replica=0,
                    output=str(args.output / f"trajectory-{mode}-{label}"))))
            before, after = streams
            for attr in ("q", "p"):
                np.testing.assert_array_equal(getattr(before["state"], attr), getattr(after["state"], attr))
            for key in ("samples", "sample_steps", "sample_bias", "sample_energies"):
                np.testing.assert_array_equal(before[key], after[key])
            for key in ("rng", "hills", "propagation"):
                assert before[key] == after[key], key
            records.append(dict(mode=mode, beads=4, integration_steps=200+args.steps,
                                hills=len(after["hills"]), samples=len(after["samples"]),
                                final_q_p_exactly_equal=True, samples_energies_bias_exactly_equal=True,
                                rng_hills_propagation_exactly_equal=True))
    finally:
        workflow.MetadynamicsBias = MetadynamicsBias
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "trajectory_equivalence.json").write_text(json.dumps(records, indent=2)+"\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
