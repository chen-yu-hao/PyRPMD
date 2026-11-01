"""Independent finite-bead equilibrium reference for the two-dimensional model.

The electronic mapping variables are integrated out analytically. A coordinate
quadrature evaluates the remaining primitive path-integral matrix trace with
the exact free-particle kernel. It therefore targets the same *finite-P* signed
SEO equilibrium marginal as the thermal sampler, without sampling its signs or
using its Metropolis/MBAR machinery. It is not an exact real-time benchmark.

For tau=beta/P, B_i=exp(-tau*V(r_i)), and symmetric quadrature weights,

    T[i,a,j,b] = sqrt(w_i*w_j) K_free(r_i,r_j;tau) B_i[a,b].

Then Z_P=Tr(T**P). Nuclear coordinate moments insert diagonal coordinate
operators; electronic populations insert the diabatic projector *before* the
matrix product, as in the SEO projector estimator. Different B_i need not
commute. A centroid second moment includes all imaginary-time lags,

    <xbar*ybar> = (1/P) sum_k Tr(X T**k Y T**(P-k)) / Z_P.

This distinction matters: bead x squared is not centroid x squared.
The separately confined z coordinate in the example is proportional to the
electronic identity and cancels from all xy/projector expectations. Reported
partition functions include the xy free-particle kinetic normalization and
electronic trace. They exclude the z partition and auxiliary sampling momenta.

Theory: Richardson and Thoss, JCP 139,031102 (2013), doi:10.1063/1.4816124.
Grid spacing, coordinate domain, and imaginary-time bead count have separate
errors; the convergence CLI reports each rather than labeling a grid exact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


def _positive_float(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_int(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _grid(bounds, count, name):
    bounds = np.asarray(bounds)
    if (bounds.shape != (2,) or bounds.dtype.kind not in "fiu"
            or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]):
        raise ValueError(f"{name} must be finite increasing bounds")
    count = _positive_int(count, "grid_shape", minimum=2)
    nodes = np.linspace(float(bounds[0]), float(bounds[1]), count)
    weights = np.full(count, (nodes[-1]-nodes[0])/(count-1))
    weights[[0, -1]] *= 0.5
    return nodes, weights


def _matrix_values(potential, coordinates):
    q = np.zeros((len(coordinates), 1, 3))
    q[:, 0, :2] = coordinates
    result = potential.evaluate(q)
    matrix = np.asarray(getattr(result, "matrix", result))
    if (matrix.dtype.kind not in "fiu" or matrix.ndim != 3
            or matrix.shape[0] != len(q) or matrix.shape[1] < 1
            or matrix.shape[1] != matrix.shape[2]
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.swapaxes(1, 2), atol=1e-12, rtol=0)):
        raise ValueError("reference requires finite real symmetric matrices [grid,S,S]")
    return np.array(matrix, dtype=float, copy=True)


def transfer_matrix_reference(
    matrix_potential, *, beta=3.0, mass=1.5, nbeads=2,
    x_bounds=(-2.5, 2.5), y_bounds=(-2.0, 2.5), grid_shape=(31, 31),
    centroid_moments=True, max_memory_mb=2048.,
):
    """Return quadrature expectations of the primitive finite-P matrix trace.

    The x/y masses are the same ``mass``; hbar=1 and all inputs use atomic
    units. The potential must describe a single particle and either be truly
    two-dimensional or have separable, state-independent z dependence. The
    function evaluates it at z=0 and cannot infer separability from callbacks.

    Trapezoidal quadrature restricts every bead to the supplied rectangle. It
    does not impose periodic boundary conditions. ``boundary_mass`` measures
    the one-bead marginal on the outermost grid layer, not an error bound.
    For P>=3 the dense matrix dimension is grid_x*grid_y*nstates; the memory
    guard estimates working arrays and fails before dense allocation.
    """
    start = time.perf_counter()
    beta, mass = _positive_float(beta, "beta"), _positive_float(mass, "mass")
    nbeads = _positive_int(nbeads, "nbeads")
    max_memory_mb = _positive_float(max_memory_mb, "max_memory_mb")
    if not isinstance(centroid_moments, (bool, np.bool_)):
        raise ValueError("centroid_moments must be boolean")
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must contain two integers")
    x, wx = _grid(x_bounds, grid_shape[0], "x_bounds")
    y, wy = _grid(y_bounds, grid_shape[1], "y_bounds")
    xx, yy = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    weights = (wx[:, None]*wy[None, :]).ravel()
    matrices = _matrix_values(matrix_potential, points)
    states, count = matrices.shape[1], len(points)
    dimension = count*states
    if nbeads == 1:
        estimated_bytes = 8*count*(states**2+12)
    elif nbeads == 2:
        estimated_bytes = 8*(4*count**2+count*states**2+12*count)
    else:
        # All lag powers may be retained. This is deliberately conservative,
        # including matrix multiplication/lag contraction temporary arrays.
        arrays = nbeads+2 if centroid_moments else 4+2*int(np.ceil(np.log2(nbeads)))
        estimated_bytes = 8*(arrays*dimension**2+3*count**2)
    if estimated_bytes > max_memory_mb*1024**2:
        raise ValueError(
            f"reference estimated dense working memory {estimated_bytes/1024**2:.1f} MiB "
            f"exceeds max_memory_mb={max_memory_mb:g}; reduce grid/beads or raise the explicit limit"
        )
    tau = beta/nbeads
    energies, vectors = np.linalg.eigh(matrices)
    energy_shift = float(energies.min())
    exponentials = np.exp(-tau*(energies-energy_shift))
    boltzmann = (vectors*exponentials[:, None, :]) @ vectors.swapaxes(1, 2)
    prefactor = mass/(2*np.pi*tau)  # two nuclear dimensions, hbar=1
    centroid_second = None

    if nbeads == 1:
        joint_diagonal = prefactor*weights[:, None]*np.diagonal(boltzmann, axis1=1, axis2=2)
        scale_log = 0.
        algorithm = "one_bead_direct_matrix_boltzmann_quadrature"
    else:
        distance_squared = (points[:, None, 0]-points[None, :, 0])**2
        distance_squared += (points[:, None, 1]-points[None, :, 1])**2
        kernel = prefactor*np.exp(-mass*distance_squared/(2*tau))
        kernel *= np.sqrt(weights[:, None]*weights[None, :])
        del distance_squared
        if nbeads == 2:
            # Exact contraction of T**2, avoiding a dense electronic-space
            # matrix multiply. This also independently checks the general path.
            kernel *= kernel
            averaged = (kernel @ boltzmann.reshape(count, states*states)).reshape(count, states, states)
            joint_diagonal = np.einsum("iab,iba->ia", boltzmann, averaged)
            scale_log = 0.
            algorithm = "two_bead_pair_contraction"
            if centroid_moments:
                pair = kernel*np.einsum("iab,jba->ij", boltzmann, boltzmann)
                lag_one = points.T @ pair @ points
        else:
            transfer = np.einsum("ij,iab->iajb", kernel, boltzmann).reshape(dimension, dimension)
            del kernel
            scale = float(np.max(np.sum(np.abs(transfer), axis=1)))
            if not np.isfinite(scale) or scale <= 0:
                raise FloatingPointError("invalid transfer matrix scale")
            transfer /= scale
            powers = {1: transfer}

            def power(exponent):
                if exponent not in powers:
                    if exponent % 2:
                        powers[exponent] = power(exponent-1) @ transfer
                    else:
                        half = power(exponent//2)
                        powers[exponent] = half @ half
                return powers[exponent]

            # Only the diagonal of T**P is needed. Contract two powers
            # directly instead of allocating/multiplying a full T**P.
            left, right = power(nbeads//2), power(nbeads-nbeads//2)
            joint_diagonal = np.einsum("ij,ji->i", left, right).reshape(count, states)
            scale_log = nbeads*np.log(scale)
            algorithm = "dense_coordinate_electronic_transfer_powers"

    partition_scaled = float(np.sum(joint_diagonal))
    if partition_scaled <= 0 or not np.isfinite(partition_scaled):
        raise FloatingPointError("nonpositive/nonfinite finite-bead partition quadrature")
    marginal = joint_diagonal.sum(axis=1)/partition_scaled
    electronic_populations = joint_diagonal.sum(axis=0)/partition_scaled
    means = marginal @ points
    bead_second = points.T @ (marginal[:, None]*points)
    if centroid_moments:
        if nbeads == 1:
            centroid_second = bead_second.copy()
        elif nbeads == 2:
            centroid_second = 0.5*(bead_second+lag_one/partition_scaled)
        else:
            lag_sum = bead_second.copy()
            for lag in range(1, nbeads//2+1):
                a = power(lag).reshape(count, states, count, states)
                b = power(nbeads-lag).reshape(count, states, count, states)
                pair = np.einsum("iajb,jbia->ij", a, b)
                correlation = points.T @ pair @ points / partition_scaled
                # The complementary lag is the transpose by cyclicity of
                # the trace, irrespective of noncommuting electronic matrices.
                lag_sum += correlation if lag*2 == nbeads else correlation+correlation.T
            centroid_second = lag_sum/nbeads
        centroid_second = 0.5*(centroid_second+centroid_second.T)

    observables = {
        "x": float(means[0]), "y": float(means[1]),
        "x2_bead": float(bead_second[0, 0]), "y2_bead": float(bead_second[1, 1]),
        "xy_bead": float(bead_second[0, 1]),
        **{f"population_{i}": float(value) for i, value in enumerate(electronic_populations)},
    }
    if centroid_second is not None:
        observables.update(
            x2_centroid=float(centroid_second[0, 0]),
            y2_centroid=float(centroid_second[1, 1]),
            xy_centroid=float(centroid_second[0, 1]),
            radius_gyration_squared_xy=float(np.trace(bead_second-centroid_second)),
        )
    outer = np.zeros((len(x), len(y)), dtype=bool)
    outer[[0, -1], :] = True
    outer[:, [0, -1]] = True
    outer_mass = float(marginal[outer.ravel()].sum())
    return {
        "method": "finite_bead_matrix_trace_coordinate_quadrature",
        "algorithm": algorithm, "beta_au": beta, "mass_au": mass,
        "nbeads": nbeads, "nstates": states,
        "observables": observables,
        "bead_covariance_xy": (bead_second-np.outer(means, means)).tolist(),
        "centroid_covariance_xy": (None if centroid_second is None else
                                   (centroid_second-np.outer(means, means)).tolist()),
        "log_partition_xy": float(np.log(partition_scaled)+scale_log-beta*energy_shift),
        "boundary_mass": outer_mass,
        "boundary_mass_definition": "one-bead marginal on outermost coordinate-grid layer; not an error bound",
        "grid": {"shape": [len(x), len(y)], "x_bounds": [float(x[0]), float(x[-1])],
                 "y_bounds": [float(y[0]), float(y[-1])],
                 "spacing": [float(x[1]-x[0]), float(y[1]-y[0])],
                 "quadrature": "tensor_product_trapezoidal",
                 "periodic": False, "coordinate_nodes": count,
                 "coordinate_electronic_dimension": dimension},
        "estimated_working_memory_MiB": estimated_bytes/1024**2,
        "elapsed_seconds": time.perf_counter()-start,
        "scope": {
            "finite_bead_equilibrium_reference": True,
            "mapping_variables_integrated_analytically": True,
            "signed_SEO_projector_ordering": True,
            "xy_free_particle_kinetic_normalization_included": True,
            "z_partition_included": False,
            "centroid_moments_include_imaginary_time_correlations": bool(centroid_moments),
            "grid_and_domain_converged": False,
            "bead_count_converged": False,
            "real_time_dynamics_validated": False,
            "physical_converged": False,
        },
    }


def _example_model_source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = (
        "examples/electrochemistry/adaptive_constant_potential.py",
        "src/pyrpmd/electrochemistry/nonadiabatic.py",
        "src/pyrpmd/electrochemistry/potentials.py",
        "src/pyrpmd/electrochemistry/reservoir.py",
    )
    return {"files": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}}


def example_reference(*, potential_V=0.0, nbeads=2, beta=3.0, mass=1.5, **grid_options):
    """Build the existing analytic example at one fixed model electrode potential."""
    from pyrpmd.electrochemistry import (
        AdiabaticElectrodePotential, ConstantPotentialDiabaticPotential, ElectronReservoir,
    )
    source = Path(__file__).with_name("adaptive_constant_potential.py")
    spec = importlib.util.spec_from_file_location("_adaptive_nrpmd_reference_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reservoir = ElectronReservoir(
        potential_V=potential_V, reference_potential_V=0.,
        chemical_potential_at_reference_au=0.1,
        potential_reference="analytic-model-reference; no SHE/RHE calibration",
    )
    electrode = AdiabaticElectrodePotential(
        module.CommonChargingModel(), reservoir, electron_bounds=(-5., 5.),
        chemical_potential_tolerance_au=1e-10,
    )
    potential = ConstantPotentialDiabaticPotential(module.CurvedTwoStateModel(), electrode)
    result = transfer_matrix_reference(potential, beta=beta, mass=mass, nbeads=nbeads, **grid_options)
    result["potential_V"] = float(potential_V)
    result["chemical_potential_au"] = float(reservoir.chemical_potential_au)
    result["potential_reference"] = reservoir.potential_reference
    result["model_scope"] = "analytic_two_state_curved_valley_with_common_BO_charging"
    result["model_source_identity"] = _example_model_source_identity()
    values = result["observables"]
    values["electron_number_bead"] = (0.3*values["x"]+0.08*values["y"]
                                       +1.2*reservoir.chemical_potential_au)
    return result


def convergence_report(*, potentials=(-0.6, 0., 0.6), beads=(1, 2, 4), grid_shape=(31, 31)):
    """Model reference grid with separate spacing, box, and bead comparisons.

    The finer grid doubles the number of intervals, capped at 41 nodes per
    axis. Every voltage/bead case is also evaluated on two successively enlarged
    rectangles with exactly the finer spacing. The second extension moves the
    y bounds farther out along the model's curved valley. ``records`` are the
    initial grid and ``preferred_records`` are the largest-box calculations. Each retains its
    own discretization metadata; the comparisons are evidence, not a claim
    of rigorous error bounds. Higher-P references at U=0 retain the coarse
    spacing and original box so they isolate the bead-count dependence.
    """
    records = [example_reference(potential_V=u, nbeads=p, grid_shape=grid_shape)
               for u in potentials for p in beads]
    fine_shape = tuple(2*(n-1)+1 for n in grid_shape)
    # At the default 31x31 setting, 61x61 would exceed the dense memory guard.
    # A 41x41 convergence comparison is practical and explicitly reports its
    # actual spacing; user-selected smaller grids can double their resolution.
    fine_shape = tuple(min(n, 41) for n in fine_shape)
    def differences(a, b):
        return {key: float(b["observables"][key]-value) for key, value in a["observables"].items()}

    comparisons = []
    for coarse in records:
        u, p = coarse["potential_V"], coarse["nbeads"]
        fine = example_reference(potential_V=u, nbeads=p, grid_shape=fine_shape)
        dx, dy = fine["grid"]["spacing"]
        extra_x, extra_y = int(np.ceil(.5/dx)), int(np.ceil(.5/dy))
        wide_shape = (fine_shape[0]+2*extra_x, fine_shape[1]+2*extra_y)
        wide = example_reference(
            potential_V=u, nbeads=p, grid_shape=wide_shape,
            x_bounds=(-2.5-extra_x*dx, 2.5+extra_x*dx),
            y_bounds=(-2.-extra_y*dy, 2.5+extra_y*dy),
        )
        farther = example_reference(
            potential_V=u, nbeads=p,
            grid_shape=(wide_shape[0], fine_shape[1]+4*extra_y),
            x_bounds=(-2.5-extra_x*dx, 2.5+extra_x*dx),
            y_bounds=(-2.-2*extra_y*dy, 2.5+2*extra_y*dy),
        )
        comparisons.append({
            "potential_V": u, "nbeads": p, "coarse": coarse, "fine": fine,
            "wider_domain": wide, "farther_y_domain": farther,
            "fine_minus_coarse": differences(coarse, fine),
            "wider_minus_fine": differences(fine, wide),
            "farther_minus_wider": differences(wide, farther),
            "log_partition_fine_minus_coarse": fine["log_partition_xy"]-coarse["log_partition_xy"],
            "log_partition_wider_minus_fine": wide["log_partition_xy"]-fine["log_partition_xy"],
            "log_partition_farther_minus_wider": farther["log_partition_xy"]-wide["log_partition_xy"],
        })
    selected = next((r for r in comparisons if r["potential_V"] == 0. and r["nbeads"] == 4), None)
    if selected is None:
        baseline = example_reference(potential_V=0., nbeads=4, grid_shape=grid_shape)
    else:
        baseline = selected["coarse"]
    higher = [example_reference(potential_V=0., nbeads=p, grid_shape=grid_shape)
              for p in (8, 16)]
    return {
        "schema_version": 1, "records": records,
        "preferred_records": [r["farther_y_domain"] for r in comparisons],
        "convergence": {
            "cases": comparisons,
            "at_potential_V": 0., "grid_comparison_nbeads": 4,
            "coarse": baseline,
            "higher_bead_references": higher,
            "eight_minus_four": differences(baseline, higher[0]),
            "sixteen_minus_eight": differences(higher[0], higher[1]),
        },
        "scope": "Deterministic finite-P equilibrium check for the analytic model; no real-time/chemical validation.",
        "physical_converged": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--potential-V", type=float, default=0.)
    parser.add_argument("--beads", type=int, default=2)
    parser.add_argument("--grid", type=int, default=31)
    parser.add_argument("--convergence", action="store_true", help="run voltage/bead grid plus discretization diagnostics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = (convergence_report(grid_shape=(args.grid, args.grid)) if args.convergence else
              example_reference(potential_V=args.potential_V, nbeads=args.beads,
                                grid_shape=(args.grid, args.grid)))
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
