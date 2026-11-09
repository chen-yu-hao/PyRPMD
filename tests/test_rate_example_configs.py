"""The runnable rate examples document the sampler/rate-estimator boundary."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pyrpmd.input import JobSpec, rate_pmf_options, rate_pmf_stage
from pyrpmd.workflows.rate import run_rate


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "rate"
CASES = [
    ("minimal.inp", "umbrella", "ui"),
    ("research.inp", "umbrella", "ui"),
    ("pmf_umbrella.inp", "umbrella", "ui"),
    ("pmf_umbrella_mbar.inp", "umbrella", "mbar"),
    ("pmf_metadynamics.inp", "metadynamics", "reweighted_histogram"),
    ("pmf_opes.inp", "opes", "reweighted_histogram"),
]


@pytest.mark.parametrize("filename,sampler,estimator", CASES)
def test_rate_examples_validate_with_pes_and_select_one_estimator(filename, sampler, estimator):
    path = EXAMPLES / filename
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    check = subprocess.run(
        [sys.executable, "-m", "pyrpmd", "validate", str(path), "--check-pes"],
        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
    )
    assert check.returncode == 0, check.stderr or check.stdout
    report = json.loads(check.stdout)
    assert report["valid"] and report["task"] == "rate"
    assert report["pes_check"]["status"] == [0]

    job = JobSpec.from_file(path)
    config, system = job.resolve()
    options = rate_pmf_options(config)
    assert options["sampler"] == sampler
    assert options["method"] == estimator
    assert report["pmf_plan"]["sampler"] == sampler
    assert report["pmf_plan"]["estimator"] == estimator
    if filename == "minimal.inp":
        # The intentionally minimal, legacy-schema input still defaults to UI.
        assert "pmf" not in job.data.get("rate", {})
        assert rate_pmf_stage(config, system) is None
    else:
        assert job.data["rate"]["pmf"]["estimator"] == estimator
        stage = rate_pmf_stage(config, system)
        assert stage["analysis"]["measure"] == "rate_flux"
        if sampler == "umbrella":
            assert "method" not in stage["analysis"]
            assert "method" not in job.data["rate"]["pmf"]["analysis"]
        else:
            assert stage["analysis"]["method"] == "reweighted_histogram"


@pytest.mark.parametrize("filename,estimator", [
    ("pmf_umbrella.inp", "ui"),
    ("pmf_umbrella_mbar.inp", "mbar"),
])
def test_short_example_runs_export_selected_rate_estimator(tmp_path, filename, estimator):
    # Run the example's real analytic PES and chosen PMF path, but reduce only
    # the teaching budgets. The resulting rates are NOT physical benchmarks.
    job = JobSpec.from_file(EXAMPLES / filename)
    raw = deepcopy(job.data)
    pmf = raw["rate"]["pmf"]
    pmf["umbrella"].update(start=0.0, stop=1.05, count=6)
    for name in ("equilibration", "production", "interval"):
        pmf["sampling"].pop(name, None)
    pmf["sampling"].update(
        equilibration_steps=2, production_steps=20, interval_steps=2, replicas=1,
    )
    pmf["analysis"].update(bins=20, bootstrap=0)
    raw["recrossing"].update(
        parent_replicas=1, parent_samples=1, child_pairs=1,
        child_time="1 fs", equilibration="0.2 fs", parent_interval="0.2 fs",
        bootstrap=0,
    )
    config, system = JobSpec.from_mapping(raw, base_directory=job.base_directory).resolve()
    run_rate(config, system, tmp_path)
    diagnostics = json.loads((tmp_path / "rate_pmf.json").read_text())
    assert diagnostics["sampler"] == "umbrella"
    assert diagnostics["estimator"] == diagnostics["method"] == estimator
    assert diagnostics["intermediate_fes_bootstrap"] == 0
    stage_input = json.loads((tmp_path / "input" / "pmf-resolved.json").read_text())
    assert "method" not in stage_input["analysis"]
    fes_summary = json.loads((tmp_path / "umbrella" / "summary.json").read_text())
    assert fes_summary["analysis_owner"] == "rate"
    assert fes_summary["rate_estimator"] == estimator
    assert fes_summary["bootstrap_requested"] == 0
    assert (tmp_path / "rate_pmf.npz").is_file()
    assert len(list((tmp_path / "umbrella" / "checkpoints").glob("*.npz"))) == 6


@pytest.mark.parametrize("filename,sampler", [
    ("pmf_metadynamics.inp", "metadynamics"),
    ("pmf_opes.inp", "opes"),
])
def test_short_enhanced_examples_keep_frozen_bias_reweighting(tmp_path, filename, sampler):
    job = JobSpec.from_file(EXAMPLES / filename)
    raw = deepcopy(job.data)
    pmf = raw["rate"]["pmf"]
    pmf["sampling"].update(
        equilibration_steps=2, production_steps=80, interval_steps=5, replicas=1,
    )
    pmf[sampler].update(pace=10, freeze_after_steps=20, freeze_equilibration_steps=10)
    raw["recrossing"].update(
        parent_replicas=1, parent_samples=1, child_pairs=1,
        child_time="1 fs", equilibration="0.2 fs", parent_interval="0.2 fs",
        bootstrap=0,
    )
    config, system = JobSpec.from_mapping(raw, base_directory=job.base_directory).resolve()
    run_rate(config, system, tmp_path)
    diagnostics = json.loads((tmp_path / "rate_pmf.json").read_text())
    assert diagnostics["sampler"] == sampler
    assert diagnostics["estimator"] == "reweighted_histogram"
    # This short smoke intentionally does not require calibrated barrier support.
    assert diagnostics["status"] in ("ready", "unsupported_pmf")
    assert (tmp_path / "pmf" / "summary.json").is_file()
    assert (tmp_path / "rate_pmf.npz").is_file()
