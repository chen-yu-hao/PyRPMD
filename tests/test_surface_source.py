"""Analytic 3D density -> conditional PMF -> parents -> paired children."""
import copy
import json
import numpy as np
import pytest

from pyrpmd.constants import KB_AU, ATOMIC_TIME_S
from pyrpmd.input import JobSpec, InputError
from pyrpmd.runner import _initialize_output, run, resume_run, prepare
from pyrpmd.io import write_json
from pyrpmd.workflows.fes import _atomic_npz
from pyrpmd.workflows.surface_source import parent_selector, source_hamiltonian
from pyrpmd.potentials import FreePotential


def source_job():
    return {"task": "metadynamics", "temperature": 1 / (2 * KB_AU), "beads": 1,
            "seed": 33, "native": False,
            "system": {"symbols": ["H", "Cu"], "masses": [2., 30.],
                       "geometry": [[0., .25, .25], [0., 0., 0.]], "fixed_atoms": [1],
                       "cell": np.diag([8., 8., 20.]).tolist(), "pbc": [True, True, False]},
            "potential": {"provider": "free"}, "dynamics": {"timestep": .01},
            "sampling": {"equilibration_steps": 2, "production_steps": 6, "interval_steps": 1},
            "cvs": [{"name": a, "kind": "cartesian", "atom": 0, "axis": a} for a in "xyz"],
            "metadynamics": {"sigma": [.2] * 3, "height": .01, "pace": 2,
                             "walls": [{"cv": a, "lower": -2., "upper": 2., "kappa": 1.} for a in "xyz"]},
            "analysis": {"bins": [4] * 3, "bootstrap": 0}, "output": {"checkpoint_steps": 5}}


def analytic_source(tmp_path, *, masked=False, beads=1):
    # A declared synthetic fixture: the exact free-PES density inside inactive
    # walls is constant. No short simulation is presented as a converged FES.
    original = source_job()
    original["beads"] = beads
    cfg, system = JobSpec.from_mapping(original).resolve()
    directory = tmp_path / "source"
    _initialize_output(cfg, system, directory)
    mask = np.zeros((4, 4, 4), bool)
    if masked:
        mask[1, 3, 2] = True
    _atomic_npz(directory / "fes.npz", free_energy=np.zeros(mask.shape), mask=mask,
                **{f"edges_{i}": np.linspace(-2, 2, 5) for i in range(3)})
    write_json(directory / "summary.json", {"task": "metadynamics", "measure": "cv_density", "converged": False})
    selected = np.ones_like(mask)
    selected[:, :2] = False  # y >= 0 channel, with a finite box in x/y/z.
    np.save(tmp_path / "channel.npy", selected)
    job = copy.deepcopy(original)
    job["task"] = "surface_rate"
    del job["metadynamics"], job["cvs"]
    job["sampling"] = {"equilibration_steps": 1, "production_steps": 10, "interval_steps": 1, "andersen_interval": 2}
    job["analysis"] = {"method": "source", "bootstrap": 0}
    job["surface"] = {"reactant_basin": [-1., -.2], "dividing_surface": 0.,
                      "free_energy": {"directory": str(directory), "axis": 0,
                                      "selection": str(tmp_path / "channel.npy")}}
    job["recrossing"] = {"parent_replicas": 2, "parent_samples": 3, "parent_interval_steps": 4,
                         "equilibration_steps": 4, "child_pairs": 2, "child_steps": 10, "bootstrap": 4}
    return job


def test_saved_3d_source_runs_consistent_conditional_rate_and_exact_resume(tmp_path):
    job = analytic_source(tmp_path)
    config, system = JobSpec.from_mapping(job).resolve()
    assert prepare(job)["source_preparation"]["axis"] == 0
    result = run(job, output=tmp_path / "rate")
    assert result["pmf"]["ready"]
    assert result["pmf"]["k_qtst_s_inv"] == pytest.approx(np.sqrt(.5 / (2 * np.pi * 2)) / .8 / ATOMIC_TIME_S)
    assert result["kappa"] == pytest.approx(1.)
    assert result["rate_estimate_s_inv"] > 0
    assert result["rate_s_inv"] is None
    assert result["rate_standard_error_s_inv"] is None
    assert result["status"] == "projected_pmf_pilot"
    assert [p["accepted"] for p in result["source_parent_sampling"]] == [3, 3]
    selector = parent_selector(config, system)
    for path in (tmp_path / "rate/checkpoints").glob("parent_*.npz"):
        with np.load(path, allow_pickle=False) as data:
            assert all(selector(q) for q in data["snapshots"])
            np.testing.assert_array_equal(data["q"][:, 1], 0.)
    again = resume_run(tmp_path / "rate")
    for key in ("kappa", "rate_estimate_s_inv", "source_parent_sampling"):
        assert result[key] == again[key]
    # The source FES is part of the run's immutable, hash-checked inputs.
    with (tmp_path / "source/fes.npz").open("ab") as out:
        out.write(b"changed source")
    with pytest.raises(RuntimeError, match="changed|missing"):
        resume_run(tmp_path / "rate")


def test_source_channel_support_holes_withhold_rate_and_children(tmp_path):
    job = analytic_source(tmp_path, masked=True)
    result = run(job, output=tmp_path / "rate")
    assert not result["pmf"]["ready"]
    assert result["rate_estimate_s_inv"] is None
    assert not list((tmp_path / "rate/checkpoints").glob("child_*.npz"))


def test_source_trpmd_retains_sensitivity_label_and_withholds_formal_rate(tmp_path):
    job = analytic_source(tmp_path, beads=4)
    job["recrossing"].update(dynamics="trpmd", friction_scale=.5)
    result = run(job, output=tmp_path / "sensitivity")
    assert result["status"] == "trpmd_sensitivity_estimate"
    assert result["dynamics"] == "trpmd"
    assert result["trpmd_rate_estimate_s_inv"] > 0
    assert result["rate_s_inv"] is None
    assert result["normalized_rates"]["per_reactant"]["rate"] is None
    assert not result["rate_quality"]["passed"]
    assert "trpmd_sensitivity_is_not_an_rpmd_rate" in result["rate_quality"]["reasons"]


def test_source_rejects_out_of_channel_parent_snapshots(tmp_path):
    job = analytic_source(tmp_path)
    job["system"]["geometry"][0][1] = -.5
    result = run(job, output=tmp_path / "outside")
    assert result["pmf"]["ready"]
    assert result["rate_estimate_s_inv"] is None
    assert [p["accepted"] for p in result["source_parent_sampling"]] == [0, 0]
    assert not list((tmp_path / "outside/checkpoints").glob("child_*.npz"))


def test_source_parent_channel_and_static_wall_hamiltonian(tmp_path):
    job = analytic_source(tmp_path)
    cfg, system = JobSpec.from_mapping(job).resolve()
    selector = parent_selector(cfg, system)
    q = np.repeat(system.geometry[None], 4, axis=0)
    assert selector(q)
    q[:, 0, 1] = -.2
    assert not selector(q)
    q[:, 0, 1] = 2.5
    assert not selector(q)
    physical = source_hamiltonian(FreePotential(), cfg, system).evaluate(q)
    assert physical.energy.sum() == pytest.approx(4 * .5**2)
    np.testing.assert_allclose(physical.forces[:, 0, 1], -1.)
    np.testing.assert_array_equal(physical.forces[:, 1], 0.)


@pytest.mark.parametrize("beads", [1, 4])
def test_source_batched_children_keep_per_ring_walls_and_scalar_checkpoints(tmp_path, beads):
    from pyrpmd.parallel import run_tasks
    from pyrpmd.workflows.rate import _child_task
    from pyrpmd.workflows.batched_recrossing import _child_batch_task
    from test_batched_recrossing import compare_pair

    job = analytic_source(tmp_path, beads=beads)
    job["dynamics"]["timestep"] = .05
    # Paired momenta drive at least one partner into the y wall. The wall is a
    # centroid bias on each ring, never on a flattened collection of rings.
    job["system"]["geometry"][0][1] = 1.99
    job["recrossing"]["child_batch_pairs"] = 4
    config, system = JobSpec.from_mapping(job).resolve()
    tasks = [{"config": config, "system": system, "chain": 0, "parent": 0,
              "pair": pair, "q": np.repeat(system.geometry[None], beads, axis=0),
              "xi_star": 0., "child_steps": 60, "output": str(tmp_path / "scalar")}
             for pair in range(4)]
    scalar = run_tasks(_child_task, tasks)
    batched_tasks = [dict(task, output=str(tmp_path / "batch")) for task in tasks]
    batched = run_tasks(_child_batch_task, [batched_tasks])[0]
    assert any(row["endpoint_energy_error_hartree_per_bead"] > 1e-9 for row in scalar)
    for actual, expected, task in zip(batched, scalar, batched_tasks):
        compare_pair(actual, expected)
        assert actual["propagation"]["pairs_in_shared_batch"] == 4
        compare_pair(_child_task(task), expected)


def test_source_batched_children_reject_out_of_channel_parents(tmp_path):
    from pyrpmd.workflows.batched_recrossing import _child_batch_task

    job = analytic_source(tmp_path)
    config, system = JobSpec.from_mapping(job).resolve()
    q = system.geometry[None].copy()
    q[:, 0, 1] = -.5
    task = {"config": config, "system": system, "chain": 0, "parent": 0,
            "pair": 0, "q": q, "xi_star": 0., "child_steps": 10,
            "output": str(tmp_path / "rejected")}
    with pytest.raises(ValueError, match="outside the source PMF channel"):
        _child_batch_task([task])
    assert not list((tmp_path / "rejected").glob("checkpoints/child_*.npz"))


def test_channel_file_must_be_a_boolean_array_before_run(tmp_path):
    job = analytic_source(tmp_path)
    np.savez(tmp_path / "wrong.npz", selection=np.ones((4, 4, 4), bool))
    job["surface"]["free_energy"]["selection"] = str(tmp_path / "wrong.npz")
    with pytest.raises(InputError, match="Boolean .npy"):
        JobSpec.from_mapping(job).resolve()


@pytest.mark.parametrize("change, message", [
    ({"temperature": "300 K"}, "temperature"),
    ({"beads": 4}, "bead"),
    ({"potential": {"provider": "periodic_cosine"}}, "PES"),
    ({"analysis": {"method": "source", "bootstrap": 2}}, "bootstrap"),
])
def test_source_rejects_incompatible_physics_before_trajectories(tmp_path, change, message):
    job = analytic_source(tmp_path)
    job.update(change)
    with pytest.raises(InputError, match=message):
        JobSpec.from_mapping(job).resolve()
