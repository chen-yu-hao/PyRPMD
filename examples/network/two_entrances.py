"""Four short teaching calculations: two entrances x two distinct saddles.

The artificial masses and temperature make this a fast numerical example.
These analytic model PESs are not physical H3 or a converged chemical study.
The four rates are independently normalized and are never summed.
"""

from math import log


def build():
    potential = {
        "provider": "python", "factory": "../rate/toy_exchange.py:build",
    }
    transitions = []
    for entrance, channel, forming, breaking, couplings in (
        ("a_bc", "ab_c", [[0, 1]], [[1, 2]], (0.32, 0.40)),
        ("ab_c", "a_bc", [[1, 2]], [[0, 1]], (0.48, 0.56)),
    ):
        for index, coupling in enumerate(couplings):
            radius = 1.2 + log((0.16 + coupling) / 0.16) / 1.2
            transitions.append({
                "id": f"{entrance}_ts{index + 1}", "entrance": entrance,
                "channel": channel,
                "geometry": [[-radius, 0, 0], [0, 0, 0], [radius, 0, 0]],
                "geometry_unit": "bohr", "forming_bonds": forming,
                "breaking_bonds": breaking,
                "potential": {**potential, "params": {"coupling": coupling}},
            })
    return {
        "task": "network", "preset": "pilot", "beads": 2, "seed": 17,
        "description": "Independent model saddle tests, not physical H3 rates",
        "temperature": 63155.004960989, "native": False,
        "system": {"symbols": ["H", "H", "H"], "masses": [2.0, 3.0, 5.0]},
        "dynamics": {"timestep": "0.003 autime"},
        "sampling": {
            "equilibration_steps": 50, "production_steps": 200,
            "sample_interval": 2, "andersen_interval": 20,
        },
        "umbrella": {"start": 0.0, "stop": 1.0, "count": 11, "kappa": 10.0},
        "analysis": {"bins": 30, "bootstrap": 2},
        "recrossing": {
            "dividing_surface": 1.0, "parent_replicas": 1, "parent_samples": 2,
            "equilibration_steps": 20, "parent_interval_steps": 5,
            "child_pairs": 1, "child_steps": 20, "bootstrap": 10,
        },
        "parallel": {"backend": "serial"},
        "output": {"directory": "runs/two-entrances", "checkpoint_steps": 200},
        "network": {
            "aggregation": "separate",
            "entrances": [
                {"id": "a_bc", "fragments": [[0], [1, 2]], "rinf": "4 bohr",
                 "electronic_weight": 0.25},
                {"id": "ab_c", "fragments": [[0, 1], [2]], "rinf": "4.5 bohr",
                 "electronic_weight": 0.75},
            ],
            "channels": [
                {"id": "ab_c", "fragments": [[0, 1], [2]]},
                {"id": "a_bc", "fragments": [[0], [1, 2]]},
            ],
            "transition_states": transitions,
        },
    }
