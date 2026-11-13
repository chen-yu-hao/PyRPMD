"""Exact full-dimensional references and the pipeline's subspace models on a model run."""

from __future__ import annotations

import numpy as np

from pyrpmd.analysis import subspace_pipeline as pipeline
from pyrpmd.analysis.subspace_model import SubspaceModel
from pyrpmd.analysis.subspace_paths import mep_core

QUANTUM = {**pipeline.DEFAULTS["quantum"], "truncation_min": 0.0}


def build_path(run, step=4.0):
    core, record = mep_core(run, {"step": step})
    path, info = pipeline._assemble(run, core, {"step": step, "product_drop_kt": 15.0}, {"mep": record})
    return path, info


def subspace_models(run, path, path_info, which=("1d", "1d_sct", "plane", "rph"), environment="adiabatic"):
    gating = {"path_info": path_info, "validity": {}, "hessian_displacement": 5e-3}
    out = {}
    for name in which:
        options = {**QUANTUM, "environment": environment, **gating}
        if name in ("1d", "1d_sct"):
            model, quantum, info = pipeline.path_model(run, path, {**options, "curvature": "small" if name == "1d_sct" else "none"})
            out[name] = (model, quantum, info, model)
        elif name == "plane":
            direction = np.asarray(pipeline._static_curvature_width(run, path, 0.5)["candidate_direction"])
            out[name] = pipeline.plane_model(run, path, direction, options)
        else:
            out[name] = pipeline.rph_model(run, path, options)
    return out


def full_model(potential, beta, *, spacing, half_width, reference, cap, refinement=1, truncation_kt=30.0):
    """The whole model potential on a tensor grid: axis 0 = x, dividing surface x = 0."""
    length = reference + cap + 0.05 * (2 * (reference + cap))
    axes = [np.arange(-length, length + 1e-9, spacing[0])]
    for i, width in enumerate(half_width):
        lo, hi = width
        axes.append(np.arange(lo, hi + 1e-9, spacing[i + 1]))
    grids = np.meshgrid(*axes, indexing="ij")
    points = np.stack(grids, axis=-1)
    values = potential.energy(points)
    model = SubspaceModel(tuple(axes), values, np.ones(len(axes)), 0.0, "asymptotic", -reference,
                          {"model": "full"}, function=potential.energy, gradient=potential.gradient)
    return model, {"cap_width": cap}
