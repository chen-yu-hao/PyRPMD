"""End-to-end: a rate run with output.centroid_path writes a readable mean path."""

import shutil
from pathlib import Path

import numpy as np

from pyrpmd.analysis.centroid_path import mean_path_from_rate_run

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "rate"

INPUT = """task = "rate"
preset = "pilot"
temperature = "1000 K"
beads = 4
seed = 612

[system]
transition_state = "ts.xyz"

[potential]
provider = "python"
factory = "toy_exchange.py:build"

[dynamics]
integrator = "cayley-bcb"
timestep = "0.1 fs"

[sampling]
thermostat = "andersen"
andersen_interval = 20

[reaction]
rinf = "5 angstrom"

[rate]
dividing_surface = 1.0

[rate.pmf]
method = "umbrella"
estimator = "ui"

[rate.pmf.dynamics]
integrator = "normal-mode-bab"
timestep = "0.2 fs"

[rate.pmf.sampling]
thermostat = "pile-l"
gamma_centroid = "0.02 fs^-1"
equilibration = "2 fs"
production = "8 fs"
interval = "0.4 fs"
replicas = 2

[rate.pmf.umbrella]
start = -0.05
stop = 1.10
count = 10
kappa = "0.16 hartree"

[rate.pmf.analysis]
bins = 20
bootstrap = 0

[recrossing]
parent_replicas = 1
parent_samples = 2
equilibration = "1 fs"
parent_interval = "1 fs"
child_pairs = 1
child_time = "2 fs"
bootstrap = 0

[output]
directory = "run"
checkpoint_interval = "2 fs"
centroid_path = true
"""


def test_rate_run_accumulates_window_moments(tmp_path):
    from pyrpmd.input import JobSpec
    from pyrpmd.runner import run

    for name in ("toy_exchange.py", "ts.xyz"):
        shutil.copy(EXAMPLES / name, tmp_path / name)
    (tmp_path / "rate.inp").write_text(INPUT)
    run(JobSpec.from_file(tmp_path / "rate.inp"))
    path = mean_path_from_rate_run(tmp_path / "run")
    assert len(path.xi) == 10
    # Two replicas of 20 samples per window, all accumulated.
    assert np.all(path.count == 40) and path.complete.all()
    assert path.bead_covariance.shape == (10, 9, 9)
    assert np.allclose(path.bead_covariance, path.bead_covariance.transpose(0, 2, 1))
    assert path.mean_distance.shape == (10, 3)
    assert path.linear_axis is not None  # the toy TS is collinear
    assert np.isfinite(path.orientation_spread).all()
