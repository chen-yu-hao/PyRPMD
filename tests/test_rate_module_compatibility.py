"""The rate workflow keeps its public imports and spawned restart behavior."""

import pickle

import numpy as np

from pyrpmd.parallel import run_tasks
from pyrpmd.system import System
from pyrpmd.workflows.rate import (
    _child_task,
    _child_tasks,
    _parent_task,
    _recrossing_options,
)


def test_rate_workers_keep_legacy_imports_and_spawned_checkpoint_results(tmp_path):
    # Previously serialized global references still resolve through the façade.
    for worker in (_parent_task, _child_task):
        legacy = f"cpyrpmd.workflows.rate\n{worker.__name__}\n.".encode()
        assert pickle.loads(legacy) is worker

    system = System(
        ("H", "H", "H"), np.array([2., 2., 2.]),
        np.array([[-1.5, 0., 0.], [0., 0., 0.], [1.5, 0., 0.]]),
        ((0,), (1, 2)),
    )
    config = {
        "task": "rate", "beta": 5., "beads": 2, "dt_au": .003,
        "seed": 17, "native": False, "potential": {"provider": "free"},
        "sampling": {"andersen_interval": 2},
        "dynamics": {"loop_backend": "python", "block_steps": 3},
        "reaction": {
            "fragments": [[0], [1, 2]], "forming_bonds": [[0, 1]],
            "breaking_bonds": [[1, 2]], "rinf": 4.,
        },
        "output": {"checkpoint_steps": 3},
        "recrossing": {
            "dividing_surface": 1., "parent_replicas": 2, "parent_samples": 2,
            "equilibration_steps": 2, "parent_interval_steps": 2,
            "child_pairs": 1, "child_steps": 5, "bootstrap": 0,
        },
    }
    options = _recrossing_options(config)
    parallel = {"backend": "process", "trajectory_workers": 2}
    tasks = [
        {
            "config": config, "system": system, "options": options,
            "chain": chain, "xi_star": 1.,
            "initial_q": np.repeat(system.geometry[None], config["beads"], axis=0),
            "output": str(tmp_path / "serial"),
        }
        for chain in range(2)
    ]
    serial_parents = [_parent_task(task) for task in tasks]
    process_tasks = [dict(task, output=str(tmp_path / "process")) for task in tasks]
    process_parents = run_tasks(_parent_task, process_tasks, parallel)
    for expected, actual in zip(serial_parents, process_parents):
        np.testing.assert_array_equal(expected["snapshots"], actual["snapshots"])
        assert expected["chain"] == actual["chain"]

    pmf = {"method": "ui", "xi_star": 1.}
    serial_children = _child_tasks(
        serial_parents, config, system, tmp_path / "serial", options, pmf,
    )
    process_children = _child_tasks(
        process_parents, config, system, tmp_path / "process", options, pmf,
    )
    expected = [_child_task(task) for task in serial_children]
    actual = run_tasks(_child_task, process_children, parallel)
    for serial, spawned in zip(expected, actual):
        np.testing.assert_array_equal(serial["numerator"], spawned["numerator"])
        for name in ("denominator", "energy_drift", "chain", "parent", "pair"):
            assert serial[name] == spawned[name]

    resumed_parents = run_tasks(_parent_task, process_tasks, parallel)
    for original, resumed in zip(process_parents, resumed_parents):
        np.testing.assert_array_equal(original["snapshots"], resumed["snapshots"])
        assert resumed["propagation_this_call"]["native_steps"] == 0
        assert resumed["propagation_this_call"]["python_steps"] == 0
    checkpoint_times = {
        path: path.stat().st_mtime_ns
        for path in (tmp_path / "process" / "checkpoints").glob("child_*.npz")
    }
    assert checkpoint_times
    resumed_children = run_tasks(_child_task, process_children, parallel)
    for original, resumed in zip(actual, resumed_children):
        np.testing.assert_array_equal(original["numerator"], resumed["numerator"])
        assert original["denominator"] == resumed["denominator"]
    assert checkpoint_times == {
        path: path.stat().st_mtime_ns for path in checkpoint_times
    }
