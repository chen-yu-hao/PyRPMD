"""VES public input, shared FES output and exactly resumed optimizer/bath state."""

import copy
import json

import numpy as np
import pytest

from pyrpmd.analysis.fes_result import FESResult
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.input import InputError, JobSpec
from pyrpmd.runner import analyze_run, prepare, resume_run, run
from pyrpmd.ves import VESBias
from pyrpmd.workflows.common import _fingerprint
from pyrpmd.workflows.sampling import FESSampler
from pyrpmd.workflows.ves import _bias_estimate, reanalyze_ves, resolve_ves, run_ves


def job(target="uniform", thermostat="andersen", dimension=1):
    sampling = {"thermostat": thermostat, "equilibration_steps": 2,
                "production_steps": 23, "interval_steps": 2, "andersen_interval": 3}
    if thermostat == "nhc":
        sampling["thermostat_params"] = {"tau": .7, "chain_length": 3, "substeps": 2}
    elif thermostat == "gle":
        sampling["thermostat_params"] = {"gle": {"A": [[.7, .1], [-.1, .9]], "C": np.eye(2).tolist()}}
    return {
        "task": "fes", "temperature": 300, "beads": 4, "seed": 19, "native": False,
        "system": {"symbols": ["H"], "masses": [1.], "geometry": [[.1, .1, 0.]]},
        "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .1}},
        "dynamics": {"timestep": .02}, "output": {"checkpoint_steps": 5},
        "fes": {"method": "ves",
                "cvs": [{"name": axis, "kind": "cartesian", "atom": 0, "axis": axis}
                        for axis in ("x", "y")[:dimension]],
                "ves": {"ranges_au": [[-1., 1.]] * dimension, "orders": 3, "pace": 3,
                        "target": target, "target_update_interval": 2, "bias_factor": 4.,
                        "wall_stiffness_au": 1.},
                "sampling": sampling, "analysis": {"bins": 8, "min_count": 1}},
    }


def payload(path):
    with load_checkpoint(path, expected_kind="ves_stream", require_integrity=True) as saved:
        return {key: saved[key].copy() for key in saved.files}


def assert_same_checkpoint(first, second):
    left, right = payload(first / "checkpoints/ves_0000.npz"), payload(second / "checkpoints/ves_0000.npz")
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


@pytest.mark.parametrize("target", ["uniform", "well_tempered"])
@pytest.mark.parametrize("dimension", [1, 2])
def test_public_run_prepare_reanalyze_and_resume(tmp_path, target, dimension):
    raw = job(target, dimension=dimension)
    assert prepare(raw)["ves_preparation"]["target"] == target
    summary = run(raw, output=tmp_path)
    assert summary["task"] == "fes" and summary["method"] == "ves"
    assert summary["samples"] == 11 and summary["updates_per_replica"] == [7]
    assert summary["observations_per_replica"] == [23]
    assert summary["pending_observations_per_replica"] == [2]
    assert summary["target_updates_per_replica"] == ([3] if target == "well_tempered" else [0])
    assert not summary["converged"] and not summary["physical_converged"]
    result = FESResult.read(tmp_path)
    assert result.method == "ves" and result.free_energy.shape == (8,) * dimension
    assert result.uncertainty_kind == "replica_spread"
    assert np.isnan(result.uncertainty).all()
    for filename in ("fes.npz", "fes.csv", "fes.json", "summary.json"):
        assert (tmp_path / filename).is_file()
    assert analyze_run(tmp_path)["method"] == "ves"
    assert resume_run(tmp_path)["method"] == "ves"
    np.testing.assert_array_equal(FESResult.read(tmp_path).free_energy, result.free_energy)


@pytest.mark.parametrize("target", ["uniform", "well_tempered"])
@pytest.mark.parametrize("thermostat", ["andersen", "pile-l", "nhc", "gle"])
def test_interruption_replays_partial_blocks_rng_and_baths_exactly(tmp_path, monkeypatch, target, thermostat):
    if thermostat in ("nhc", "gle"):
        pytest.importorskip("pyrpmd._thermal")
    config, system = JobSpec.from_mapping(job(target, thermostat)).resolve()
    full, split = tmp_path / "full", tmp_path / "split"
    run_ves(config, system, full)
    original = FESSampler.step

    def fail(self, dynamics, state, bias):
        if state.step == 7:
            original(self, dynamics, state, bias)
            raise KeyboardInterrupt("injected interruption after stochastic step")
        return original(self, dynamics, state, bias)

    with monkeypatch.context() as patch:
        patch.setattr(FESSampler, "step", fail)
        with pytest.raises(KeyboardInterrupt, match="stochastic step"):
            run_ves(config, system, split)
    saved = payload(split / "checkpoints/ves_0000.npz")
    assert int(saved["step"]) == 7
    bias = json.loads(str(saved["bias_state"]))
    assert bias["observations"] == 5 and bias["pending_count"] == 2
    run_ves(config, system, split)
    assert_same_checkpoint(full, split)


def test_failed_coefficient_and_target_update_rolls_back_transactionally(tmp_path, monkeypatch):
    config, system = JobSpec.from_mapping(job("well_tempered")).resolve()
    full, split = tmp_path / "full", tmp_path / "split"
    run_ves(config, system, full)
    original = VESBias.observe

    def fail(self, value):
        result = original(self, value)
        if self.observations == 6:
            assert self.target_updates == 1
            raise RuntimeError("injected failure after target update")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(VESBias, "observe", fail)
        with pytest.raises(RuntimeError, match="target update"):
            run_ves(config, system, split)
    run_ves(config, system, split)
    assert_same_checkpoint(full, split)


@pytest.mark.parametrize("phase", ["before_statistics", "after_statistics", "partial_trace"])
def test_interrupt_during_bookkeeping_preserves_durable_prefix_and_exact_resume(tmp_path, monkeypatch, phase):
    from pyrpmd.dynamics import Dynamics
    from pyrpmd.workflows import ves

    config, system = JobSpec.from_mapping(job("well_tempered", "pile-l")).resolve()
    full, split = tmp_path / "full", tmp_path / "split"
    run_ves(config, system, full)
    record = Dynamics._record_python_steps
    runtime_type = ves.SamplingRuntime

    def interrupted_statistics(self, steps):
        # Step 7 is not sampled: before/after the counter increment isolate
        # both propagation-count and fallback-diagnostic commit boundaries.
        if phase == "before_statistics" and self.propagation_stats["python_steps"] == 6:
            raise KeyboardInterrupt("injected bookkeeping interruption")
        record(self, steps)
        if phase == "after_statistics" and self.propagation_stats["python_steps"] == 7:
            raise KeyboardInterrupt("injected bookkeeping interruption")

    class InterruptedValues(list):
        def append(self, value):
            super().append(value)
            # The third production sample is at absolute step 8. Its CV is
            # appended, but its clock/bias/energy have not yet been appended.
            if len(self) == 3:
                raise KeyboardInterrupt("injected bookkeeping interruption")

    def interrupted_runtime(*args, **kwargs):
        current = runtime_type(*args, **kwargs)
        current.values = InterruptedValues(current.values)
        return current

    with monkeypatch.context() as patch:
        if phase == "partial_trace":
            patch.setattr(ves, "SamplingRuntime", interrupted_runtime)
        else:
            patch.setattr(Dynamics, "_record_python_steps", interrupted_statistics)
        with pytest.raises(KeyboardInterrupt, match="injected bookkeeping interruption"):
            run_ves(config, system, split)
    saved = payload(split / "checkpoints/ves_0000.npz")
    assert int(saved["elapsed"]) == int(saved["step"]) == 5
    assert json.loads(str(saved["bias_state"]))["observations"] == 3
    run_ves(config, system, split)
    assert_same_checkpoint(full, split)


def test_inconsistent_normal_checkpoint_candidate_raises_without_replacing_prefix(tmp_path, monkeypatch):
    from pyrpmd.workflows import ves

    config, system = JobSpec.from_mapping(job()).resolve()
    sample = ves.run_biased_sampling

    def inconsistent_save(runtime, *args, **kwargs):
        save = kwargs["save"]

        def broken(current):
            if current.elapsed == 5:
                current.clocks.append(999)
            save(current)

        kwargs["save"] = broken
        return sample(runtime, *args, **kwargs)

    monkeypatch.setattr(ves, "run_biased_sampling", inconsistent_save)
    with pytest.raises(ValueError, match="inconsistent VES checkpoint candidate: sample trace lengths"):
        run_ves(config, system, tmp_path)
    assert int(payload(tmp_path / "checkpoints/ves_0000.npz")["step"]) == 0


@pytest.mark.parametrize("target", ["uniform", "well_tempered"])
def test_frozen_histogram_excludes_update_step_and_relaxation(tmp_path, target):
    raw = job(target)
    raw["fes"]["ves"].update(freeze_after_steps=12, freeze_equilibration_steps=3)
    raw["fes"]["analysis"]["method"] = "reweighted_histogram"
    config, system = JobSpec.from_mapping(raw).resolve()
    result = run_ves(config, system, tmp_path, return_data=True)
    summary = result["summary"]
    assert summary["reweighted"]
    assert summary["updates_per_replica"] == [4]
    assert summary["observations_per_replica"] == [12]
    assert summary["pending_observations_per_replica"] == [0]
    # Production observations at 16,18,20,22 (absolute steps 18,20,22,24).
    assert summary["retained_samples_per_replica"] == [4] and summary["samples"] == 4
    assert np.sum(result["data"]["raw_count"]) == 4
    stream = result["streams"][0]
    np.testing.assert_array_equal(stream["sample_steps"], np.arange(4, 25, 2))
    rebuilt = reanalyze_ves(config, system, tmp_path, return_data=True)
    np.testing.assert_array_equal(rebuilt["data"]["free_energy"], result["data"]["free_energy"])


@pytest.mark.parametrize("name,value", [
    ("ranges_au", [[-1.2, 1.]]), ("orders", 4), ("target", "well_tempered"),
    ("bias_factor", 5.), ("learning_rate_au", .0002), ("pace", 4),
    ("target_update_interval", 3), ("quadrature_order", 24), ("wall_stiffness_au", 2.),
    ("freeze_after_steps", 12),
])
def test_every_scientific_control_enters_identity(name, value):
    raw = job()
    original, system = JobSpec.from_mapping(raw).resolve()
    changed = copy.deepcopy(raw)
    changed["fes"]["ves"][name] = value
    resolved, _ = JobSpec.from_mapping(changed).resolve()
    assert _fingerprint(original, system, {}) != _fingerprint(resolved, system, {})


def test_freeze_relaxation_enters_identity_but_analysis_bins_do_not():
    raw = job()
    raw["fes"]["ves"]["freeze_after_steps"] = 12
    first, system = JobSpec.from_mapping(raw).resolve()
    raw["fes"]["ves"]["freeze_equilibration_steps"] = 1
    second, _ = JobSpec.from_mapping(raw).resolve()
    assert _fingerprint(first, system, {}) != _fingerprint(second, system, {})
    raw["fes"]["analysis"]["bins"] = 6
    third, _ = JobSpec.from_mapping(raw).resolve()
    assert _fingerprint(second, system, {}) == _fingerprint(third, system, {})


def test_incompatible_checkpoint_fails_before_pes_or_writes(tmp_path, monkeypatch):
    raw = job()
    config, system = JobSpec.from_mapping(raw).resolve()
    run_ves(config, system, tmp_path)
    path = tmp_path / "checkpoints/ves_0000.npz"
    before = path.read_bytes()
    raw["fes"]["ves"]["pace"] = 4
    changed, _ = JobSpec.from_mapping(raw).resolve()

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible checkpoint reached provider or write")

    monkeypatch.setattr("pyrpmd.workflows.ves.make_task_potential", forbidden)
    monkeypatch.setattr("pyrpmd.workflows.ves.atomic_savez", forbidden)
    with pytest.raises(ValueError, match="physics differs"):
        run_ves(changed, system, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["clock", "time", "samples", "rng", "operators", "bias_clock"])
def test_semantically_damaged_checkpoint_rejected_before_pes(tmp_path, monkeypatch, corruption):
    config, system = JobSpec.from_mapping(job()).resolve()
    run_ves(config, system, tmp_path)
    path = tmp_path / "checkpoints/ves_0000.npz"
    arrays = payload(path)
    if corruption == "clock":
        arrays["elapsed"] = np.asarray(20)
    elif corruption == "time":
        arrays["time"] = np.asarray([float(arrays["time"])])
    elif corruption == "samples":
        arrays["sample_steps"][-1] -= 1
    elif corruption == "rng":
        arrays["rng"] = json.dumps(np.random.default_rng(1234).bit_generator.state)
    elif corruption == "operators":
        operators = json.loads(str(arrays["operator_state"]))
        operators["version"] = True
        arrays["operator_state"] = json.dumps(operators)
    else:
        state = json.loads(str(arrays["bias_state"]))
        state["updates"] += 1
        arrays["bias_state"] = json.dumps(state)
    atomic_savez(path, kind="ves_stream", **arrays)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid checkpoint caused a PES request or write")

    monkeypatch.setattr("pyrpmd.workflows.ves.make_task_potential", forbidden)
    monkeypatch.setattr("pyrpmd.workflows.ves.atomic_savez", forbidden)
    with pytest.raises(ValueError):
        run_ves(config, system, tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", ["nhc", "gle"])
def test_missing_bath_is_rejected_even_for_completed_stream(tmp_path, monkeypatch, mode):
    pytest.importorskip("pyrpmd._thermal")
    config, system = JobSpec.from_mapping(job(thermostat=mode)).resolve()
    run_ves(config, system, tmp_path)
    path = tmp_path / "checkpoints/ves_0000.npz"
    arrays = payload(path)
    operators = json.loads(str(arrays["operator_state"]))
    assert operators["sampling"]
    operators["sampling"] = {}
    arrays["operator_state"] = json.dumps(operators)
    atomic_savez(path, kind="ves_stream", **arrays)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid bath caused provider construction")

    monkeypatch.setattr("pyrpmd.workflows.ves.make_task_potential", forbidden)
    with pytest.raises(ValueError, match="missing thermostat auxiliary state"):
        run_ves(config, system, tmp_path)


def test_reanalysis_never_constructs_potential(tmp_path, monkeypatch):
    config, system = JobSpec.from_mapping(job()).resolve()
    run_ves(config, system, tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("reanalysis constructed a potential")

    monkeypatch.setattr("pyrpmd.workflows.ves.make_task_potential", forbidden)
    assert reanalyze_ves(config, system, tmp_path)["method"] == "ves"


def test_redundant_cvs_fail_before_pes_or_output(tmp_path, monkeypatch):
    raw = job(dimension=2)
    config, system = JobSpec.from_mapping(raw).resolve()
    raw["fes"]["cvs"][1]["axis"] = "x"
    config["cvs"][1]["axis"] = "x"

    def forbidden(*args, **kwargs):
        pytest.fail("redundant CVs reached potential construction")

    monkeypatch.setattr("pyrpmd.workflows.ves.make_task_potential", forbidden)
    with pytest.raises(InputError, match="full rank"):
        prepare(raw)
    with pytest.raises(ValueError, match="full rank"):
        run_ves(config, system, tmp_path / "not_created")
    assert not (tmp_path / "not_created").exists()


def test_unitful_ranges_and_stiffness_resolve_in_atomic_units():
    raw = job()
    raw["fes"]["cvs"][0]["unit"] = "angstrom"
    table = raw["fes"]["ves"]
    table.pop("ranges_au")
    table.pop("wall_stiffness_au")
    table.update(ranges=[[-1., 1.]], wall_stiffness="2 eV/angstrom^2", learning_rate="0.01 eV")
    config, system = JobSpec.from_mapping(raw).resolve()
    from pyrpmd.units import parse_quantity
    _, _, options = resolve_ves(config, system)
    np.testing.assert_allclose(options["ranges"], [[-parse_quantity("1 angstrom", "length"), parse_quantity("1 angstrom", "length")]])
    assert options["bias_kwargs"]["learning_rate"] == pytest.approx(parse_quantity("0.01 eV", "energy"))
    assert options["bias_kwargs"]["wall_stiffness"][0] == pytest.approx(parse_quantity("2 eV/angstrom^2", (-2, 1, 0, 0, 0, 0)))


@pytest.mark.parametrize("mutation", [
    lambda j: j["fes"]["ves"].update(pace=True),
    lambda j: j["fes"]["ves"].update(orders=True),
    lambda j: j["fes"]["ves"].update(quadrature_order=True),
    lambda j: j["fes"]["ves"].update(learning_rate=True),
    lambda j: j["fes"]["ves"].update(wall_stiffness_au=True),
    lambda j: j["fes"]["ves"].update(ranges_au=[[False, 1.]]),
    lambda j: j["fes"]["ves"].update(ranges_au=[[1., -1.]]),
    lambda j: j["fes"]["ves"].update(ranges=[[-1., 1.]]),
    lambda j: j["fes"]["ves"].update(learning_rate=.1, learning_rate_au=.1),
    lambda j: j["fes"]["ves"].update(target="other"),
    lambda j: j["fes"]["ves"].update(unknown_parameter=3),
    lambda j: j["fes"]["ves"].update(freeze_after_steps=4),
    lambda j: j["fes"]["ves"].update(freeze_equilibration_steps=2),
    lambda j: j["fes"]["analysis"].update(method="reweighted_histogram"),
    lambda j: j["fes"]["analysis"].update(method="mbar"),
    lambda j: j["fes"]["analysis"].update(measure="rate_flux"),
    lambda j: j["fes"]["analysis"].update(bootstrap=2),
    lambda j: j["fes"]["analysis"].update(ranges=[[-2., 1.]]),
    lambda j: j["fes"]["analysis"].update(seed=3),
    lambda j: j["fes"]["analysis"].update(bins=True),
    lambda j: j["fes"]["analysis"].update(bins=[4, 4]),
    lambda j: j["fes"]["analysis"].update(max_grid_points=1),
    lambda j: j["fes"]["sampling"].update(andersen_schedule="geometric"),
    lambda j: j["fes"]["sampling"].update(thermostat_params={"gamma": .1}),
    lambda j: j["fes"]["sampling"].update(parents=11),
    lambda j: j["fes"]["sampling"].update(engine="legacy"),
    lambda j: j["output"].update(trajectory_stride=13),
    lambda j: j["dynamics"].update(loop_backend="native"),
    lambda j: j["fes"]["cvs"][0].update(estimator="bead_average"),
    lambda j: j["fes"].update(opes={"barrier": 1.}),
    lambda j: j.update(reaction={"electronic_weight": .5}),
    lambda j: j.update(task="pimd"),
])
def test_unsupported_or_malformed_options_fail_preflight(mutation):
    raw = job()
    mutation(raw)
    with pytest.raises((InputError, ValueError)):
        JobSpec.from_mapping(raw).resolve()


def test_ves_table_is_not_silently_accepted_for_another_task():
    raw = job()
    raw["ves"] = raw["fes"].pop("ves")
    raw.pop("fes")
    raw["task"] = "pimd"
    with pytest.raises(InputError, match="requires fes.method"):
        JobSpec.from_mapping(raw).resolve()


def test_large_analysis_grid_uses_bounded_chunks_and_preserves_polynomial_surface(monkeypatch):
    from pyrpmd.cv import Cartesian, CVStack

    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    bias = VESBias(cv, 1., [[-1., 1.], [-1., 1.]], orders=[3, 3])
    terms = {(1, 1): .3, (2, 0): .2, (0, 2): -.1}
    for index, degree in enumerate(bias.basis.indices):
        bias.coefficients[index] = terms.get(tuple(degree), 0.)
    bins = [83, 77]
    edges = [np.linspace(-1., 1., count + 1) for count in bins]
    grid = np.stack(np.meshgrid(*[(edge[1:] + edge[:-1]) * .5 for edge in edges], indexing="ij"), axis=-1)
    observed = []
    evaluate = bias.free_energy

    def checked(values):
        assert values.ndim == 2 and values.shape[1] == 2
        observed.append(len(values))
        assert len(values) * bias.basis.size <= 65536
        return evaluate(values)

    monkeypatch.setattr(bias, "free_energy", checked)
    result = _bias_estimate([{"samples": grid.reshape(-1, 2)}], [bias], cv,
                           {"analysis_ranges": [[-1., 1.], [-1., 1.]], "bins": bins, "min_count": 1})
    x, y = grid[..., 0], grid[..., 1]
    expected = -(.3 * x * y + .2 * (3*x*x - 1) / 2 - .1 * (3*y*y - 1) / 2)
    expected -= expected.min()
    assert len(observed) > 1 and sum(observed) == np.prod(bins)
    assert not result["mask"].any()
    np.testing.assert_array_equal(result["raw_count"], np.ones(bins, dtype=int))
    np.testing.assert_allclose(result["free_energy"], expected, rtol=1e-13, atol=1e-14)
