"""Bennett--Chandler bimolecular RPMD rate with explicit sampling measures.

The RPMDrate/Caracal flux-metric PMF and paired-child estimator are used,
with a newly implemented centroid Cayley-RATTLE constrained parent sampler.
The rate is k_RPMD = k_QTST * kappa; kappa is always measured, never guessed.
Reference implementations are MIT licensed; see THIRD_PARTY_NOTICES.md.
The implementation is divided into geometry, options, recrossing, PMF and
analysis modules. Explicit re-exports preserve the established Python API.
"""

from __future__ import annotations

import copy
from pathlib import Path

from ..checkpoint import atomic_savez as _atomic_npz
from ..input import rate_pmf_options
from ..io import fingerprint as _fingerprint
from ..parallel import root_call, run_tasks
from ..validation import checked_integer
from .fes import run_fes
from .rate_analysis import _aggregate
from .rate_constraints import (
    _normal,
    constrained_step,
    project_momenta,
    project_positions,
    rate_coordinate,
)
from .rate_options import (
    _pmf_method,
    _recrossing_dynamics,
    _recrossing_options,
    _recrossing_spec,
)
from .rate_pmf import (
    FLUX_MEASURE,
    _check_pmf_domain,
    _histogram_endpoint_support,
    _pmf_and_parent_tasks,
    _prepare,
    _save_pmf_arrays,
    _ui_rate_pmf,
)
from .rate_recrossing import _child_task, _child_tasks, _dispatch_children, _parent_task


def run_rate(config, system, output, executor=None):
    if any(system.pbc) or system.fixed_atoms:
        raise ValueError("Gas-phase run_rate does not support periodic/fixed-atom systems; use surface_rate")
    config = root_call(_prepare, config, system, str(output))
    options = _recrossing_options(config)
    sampling_config = config
    if rate_pmf_options(config)["method"] == "mbar":
        sampling_config = copy.deepcopy(config)
        # The rate backend performs the one paired endpoint bootstrap. The
        # intermediate FES only saves samples, overlap and a preliminary curve.
        sampling_config["analysis"]["bootstrap"] = 0
    sampled = run_fes(
        sampling_config,
        system,
        Path(output) / "umbrella",
        executor=executor,
        return_data=True,
    )
    pmf, parent_tasks = root_call(
        _pmf_and_parent_tasks, sampled, config, system, str(output), options
    )

    def dispatch(function, tasks):
        return (
            executor.map(function, tasks)
            if executor is not None
            else run_tasks(function, tasks, config.get("parallel", {}))
        )

    if not parent_tasks:
        return root_call(_aggregate, [], pmf, config, str(output), options)
    parents = dispatch(_parent_task, parent_tasks)
    tasks = root_call(_child_tasks, parents, config, system, str(output), options, pmf)
    children = _dispatch_children(tasks, config, dispatch)
    return root_call(_aggregate, children, pmf, config, str(output), options)
