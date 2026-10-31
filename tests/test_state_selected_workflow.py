"""Independent state preparation remains independent through launch and restart."""
import csv
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, native_capabilities
from pyrpmd.initial_states import PreparedInitialEnsemble
from pyrpmd.workflows import drpmd


def system():
    return {"symbols": ["H"] * 3, "masses": [1836.0] * 3,
            "geometry": [[0., 0., -10.], [0., 0., -0.7], [0., 0., 0.7]],
            "fragments": [[0], [1, 2]]}


def config(method="ne_rpmd", beads=8):
    section = {"method": method, "fragments": [{}, {
        "frequencies": [np.sqrt(0.4 / 918.0)],
        "modes": [[0., 0., -2**-0.5, 0., 0., 2**-0.5]],
        "vibrational_quanta": [1], "j": 0}]}
    if method == "ne_rpmd":
        section["beta"] = 400.0
    return {
        "task": "drpmd", "beta": 400.0, "temperature_K": 1 / (400 * drpmd.KB),
        "beads": beads, "dt_au": 0.2, "seed": 735, "native": False,
        "initial_ensemble": section, "system": {"separation_au": 10.0},
        "potential": {"provider": "morse_pair", "params": {
            "atoms": [1, 2], "depth": 0.2, "alpha": 1.0, "equilibrium": 1.4}},
        "sampling": {}, "dynamics": {"block_steps": 7},
        "collision": {"trajectories": 3, "bmax_au": 0.5, "max_steps": 17,
                      "sampling": "fixed_energy", "energy_au": 0.02},
        "outcomes": {"bond_cutoffs_au": {"H-H": 2.5}, "channels": {},
                     "exit_distance_au": 10., "check_interval": 3, "persistence_steps": 6},
        "output": {"trajectory_stride": 3, "checkpoint_steps": 5,
                   "diagnostics": {"energy_parts": True, "ring_radii": True}},
        "parallel": {"backend": "serial"},
    }


def assert_same_arrays(first, second, keys=None):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        for key in keys or ("final_q", "final_p", "steps", "time_au", "centroid",
                           "energy_per_bead", "energy_parts", "ring_radii"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@pytest.mark.parametrize("method,beads", [("qct", 1), ("ne_rpmd", 32)])
def test_independent_state_launch_never_thermalizes_and_preserves_vibration(tmp_path, method, beads):
    c = config(method, beads)
    with patch.object(Dynamics, "thermalize", side_effect=AssertionError("thermal parent called")):
        result = drpmd.run_drpmd(c, system(), tmp_path)
    assert result["initialization_mode"] == "independent"
    assert result["temperature_K"] is None
    assert result["initial_ensemble"]["vibrational_quanta"] == [[], [1]]
    assert set(result["parent_thermal_steps"].values()) == {0}
    for record in result["records"]:
        assert record["initial_state"] == {
            "method": method, "vibrational_quanta": [[], [1]], "rotation": "j0"}
        assert record["launch"]["sampling"] == "fixed_energy"
        path = tmp_path / "states" / f"initial-{record['trajectory_id']:08d}.npz"
        meta, arrays = drpmd._load_npz(path)
        assert "thermal_q" not in arrays
        assert meta["initialization_mode"] == "independent"
        q, p, m = arrays["q"], arrays["p"], arrays["masses"]
        before = arrays["prepared_q"][:, 1:] - arrays["prepared_q"][:, 1:].mean(axis=1)[:, None]
        after = q[:, 1:] - q[:, 1:].mean(axis=1)[:, None]
        np.testing.assert_allclose(np.sum(before * before, axis=(1, 2)),
                                   np.sum(after * after, axis=(1, 2)), atol=1e-13)
        relative = (drpmd.fragment_velocity(p, m, [0])
                    - drpmd.fragment_velocity(p, m, [1, 2]))
        mu = m[0] * m[1:].sum() / m.sum()
        assert 0.5 * mu * relative @ relative == pytest.approx(0.02, abs=1e-15)
        with np.load(tmp_path / "trajectories" / f"trajectory-{record['trajectory_id']:08d}.npz") as trace:
            assert trace["steps"].tolist() == [0, 3, 6, 9, 12, 15, 17]
            np.testing.assert_allclose(trace["energy_parts"].sum(axis=1), trace["energy_per_bead"], atol=1e-13)
            if beads == 1:
                np.testing.assert_array_equal(trace["ring_radii"], 0)
    with (tmp_path / "results/channels.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["cross_section_bohr2"] == ""  # unresolved samples are not zero reactions
    assert float(row["possible_upper_bohr2"]) == pytest.approx(np.pi * 0.5**2)


def test_global_rng_and_physics_identity_do_not_depend_on_work_groups(tmp_path):
    c = config()
    first = drpmd.run_drpmd(c, system(), tmp_path / "one")
    c["sampling"]["parents"] = 3
    second = drpmd.run_drpmd(c, system(), tmp_path / "three")
    assert first["records"][0]["physics_hash"] == second["records"][0]["physics_hash"]
    for trajectory in range(3):
        name = f"initial-{trajectory:08d}.npz"
        assert_same_arrays(tmp_path / "one/states" / name, tmp_path / "three/states" / name,
                           ("q", "p", "prepared_q", "prepared_p"))
        name = f"trajectory-{trajectory:08d}.npz"
        assert_same_arrays(tmp_path / "one/trajectories" / name, tmp_path / "three/trajectories" / name)


def test_process_workers_use_the_same_global_initial_conditions(tmp_path):
    c = config("qct", 1)
    drpmd.run_drpmd(c, system(), tmp_path / "serial")
    c["parallel"] = {"backend": "process", "trajectory_workers": 2}
    drpmd.run_drpmd(c, system(), tmp_path / "parallel")
    for trajectory in range(3):
        name = f"trajectory-{trajectory:08d}.npz"
        assert_same_arrays(tmp_path / "serial/trajectories" / name, tmp_path / "parallel/trajectories" / name)


def test_interruption_resumes_full_state_and_diagnostics_without_resampling(tmp_path):
    c = config()
    c["collision"]["trajectories"] = 1
    drpmd.run_drpmd(c, system(), tmp_path / "full")
    save = drpmd._atomic_npz

    def interrupt(path, metadata, **arrays):
        save(path, metadata, **arrays)
        if metadata["format"] == "PyRPMD.drpmd.trajectory-checkpoint.v1" and metadata["done"] == 5:
            raise KeyboardInterrupt("checkpoint persisted")

    with patch.object(drpmd, "_atomic_npz", side_effect=interrupt), pytest.raises(KeyboardInterrupt):
        drpmd.run_drpmd(c, system(), tmp_path / "resume")
    c["resume"] = True
    c["dynamics"]["block_steps"] = 3
    with patch.object(PreparedInitialEnsemble, "sample", side_effect=AssertionError("resampled launch")):
        drpmd.run_drpmd(c, system(), tmp_path / "resume")
        second = drpmd.run_drpmd(c, system(), tmp_path / "resume")
    assert second["trajectories"] == 1
    assert_same_arrays(tmp_path / "full/trajectories/trajectory-00000000.npz",
                       tmp_path / "resume/trajectories/trajectory-00000000.npz")


def test_pes_failure_stays_in_fixed_energy_denominator(tmp_path):
    from pyrpmd.potentials import FreePotential

    c = config("qct", 1)
    with patch.object(FreePotential, "evaluate", side_effect=RuntimeError("outside PES domain")):
        c["potential"] = {"provider": "free"}
        result = drpmd.run_drpmd(c, system(), tmp_path)
    assert result["counts"]["failed"] == 3
    assert result["counts"]["unresolved"] == 0
    assert result["total_reaction"]["cross_section_bohr2"] is None
    assert result["total_reaction"]["possible_upper_bohr2"] == pytest.approx(np.pi * 0.25)
    assert len(result["records"]) == 3


def test_invalid_prepared_reactants_are_recorded_without_rejection_resampling(tmp_path):
    class RejectPreparedReactants:
        def validate_reactants(self, q):
            raise ValueError("prepared molecule has dissociated")

    c = config("qct", 1)
    with patch.object(drpmd, "_make_classifier", return_value=RejectPreparedReactants()):
        result = drpmd.run_drpmd(c, system(), tmp_path)
    assert result["counts"] == {"product": 0, "nonreactive": 0, "unresolved": 0, "failed": 3}
    assert len(list((tmp_path / "states").glob("initial-*.npz"))) == 3
    for record in result["records"]:
        assert record["steps"] == 0
        assert record["reason"].startswith("initial_preparation:")
        assert "dissociated" in record["launch"]["initial_validation_error"]
    assert result["total_reaction"]["possible_upper_bohr2"] == pytest.approx(np.pi * 0.25)


def test_selected_fixed_energy_summary_has_independent_wilson_interval():
    c = config()
    labels = {"method": "ne_rpmd", "vibrational_quanta": [[], [1]], "rotation": "j0"}
    records = [{"trajectory_id": i, "parent_id": 0,
                "status": "product" if i < 2 else "nonreactive",
                "channel": "exchange" if i < 2 else None,
                "launch": {"sampling": "fixed_energy", "collision_energy_au": 0.02},
                "initial_state": labels} for i in range(10)]
    result = drpmd.summarize(records, c, system())
    entry = result["total_reaction"]
    assert entry["cross_section_bohr2"] == pytest.approx(np.pi * 0.25 * 0.2)
    assert entry["iid_wilson95_bohr2"][0] < entry["cross_section_bohr2"] < entry["iid_wilson95_bohr2"][1]
    assert entry["parent_cluster_bootstrap95_bohr2"] is None


@pytest.mark.skipif(not native_capabilities()["available"], reason="native build required")
def test_selected_workflow_uses_existing_native_collision_driver(tmp_path, native_library):
    # Same compiled fixture as the established collision-driver integration test.
    from test_native_propagation import CompiledPotential

    c = config(beads=32)
    c["native"] = "required"
    c["dynamics"]["loop_backend"] = "native"
    potential = CompiledPotential(native_library)
    with patch.object(drpmd, "make_task_potential", return_value=potential):
        result = drpmd.run_drpmd(c, system(), tmp_path)
    for record in result["records"]:
        assert record["propagation"] == {"native_steps": 17, "python_steps": 0}
    with np.load(tmp_path / "trajectories/trajectory-00000000.npz") as trace:
        np.testing.assert_allclose(trace["energy_parts"].sum(axis=1), trace["energy_per_bead"], atol=1e-12)


# Importing a fixture registers it for this module without copying its C code.
from test_native_propagation import native_library  # noqa: E402,F401
