"""Reproduce Lambert et al., PRR 5, 013181 (2023), Figure 1 population curves.

Run from the repository root after installing PyRPMD and its NumPy/SciPy
dependencies (the default check reads only the two CSVs beside this script):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
        docs/validation/t13-nonadiabatic-2026-11-02/validate_lambert_figure1.py

Use ``--output-dir /tmp/lambert-revalidation`` to save regenerated JSON, CSVs,
and a PNG plot; Matplotlib is needed only when writing the plot.  Without that
option, the script prints a summary and never overwrites recorded evidence.

Optional ``--pdf /path/to/2010.10806v3.pdf`` repeats the original hash-checked
vector extraction instead of reading the CSVs.  This mode alone needs PyMuPDF.
The script never downloads files or executes the author's notebook.

Reference populations are extracted vertices from the published PDF, not a
fresh QuTiP calculation.  Only CSV columns ``time,published_population`` enter
the comparison; the recorded HEOM columns are not used as reference values.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from pyrpmd.analysis.heom import SpinBosonHEOM


ROOT = Path(__file__).resolve().parent
PDF_SHA256 = "f93671a5ad76b2ca4fab72a936ffed32653579833e927e57a6c6d1f307b47c6f"
NOTEBOOK_COMMIT = "8c3c58a4e84a331a7aac69c4a2ecaa7f2a1369bc"
NOTEBOOK_SHA256 = "5e8af32e0ac2727005b13e595e6f4788ae9faa8dd8ca8aca80c3ff69287c1e1f"
DEPTHS = (3, 5, 7, 9)
PUBLICATION_TOLERANCE = 1e-6
CURVES = (
    ("without_terminator", False, 24, (0., 0., 1.)),
    ("terminator", True, 25, (0., .5, 0.)),
)
AXIS_CALIBRATION = {
    "x_at_t_0": 83.64217376708984,
    "x_at_t_40": 254.34829711914062,
    "y_at_population_1": 60.286529541015625,
    "y_at_population_0_6": 128.53404235839844,
}


def validate_reference(reference):
    if (reference.ndim != 2 or reference.shape[1] != 2
            or len(reference) < 2 or not np.isfinite(reference).all()
            or reference[0, 0] != 0
            or not np.all(np.diff(reference[:, 0]) > 0)
            or not np.all((reference[:, 1] >= 0) & (reference[:, 1] <= 1))):
        raise ValueError("Reference must contain increasing times from zero and finite populations in [0,1]")
    return reference


def read_references(pdf_path=None):
    """Read portable CSV reference columns, or explicitly re-extract the PDF."""
    if pdf_path is None:
        references = {}
        for label, *_ in CURVES:
            path = ROOT / f"lambert-figure1-{label}.csv"
            with path.open() as handle:
                if handle.readline().strip().split(",")[:2] != ["time", "published_population"]:
                    raise ValueError(f"Unexpected reference columns in {path.name}")
            reference = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1))
            references[label] = validate_reference(reference)
        return references, "bundled_csv_reference_columns"

    pdf_path = Path(pdf_path)
    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError("The PDF hash differs: re-inspect the figure and vector-path indices")
    import fitz  # Optional PyMuPDF dependency; default CSV validation does not import it.

    with fitz.open(pdf_path) as document:
        drawings = document[6].get_drawings()
    # Tick locations independently read from the rendered figure: t=0,40 and
    # P=1,.6.  The coordinate conversion is not fitted to our solver output.
    x0 = drawings[1]["items"][0][1].x
    x40 = drawings[6]["items"][0][1].x
    y1 = drawings[18]["items"][0][1].y
    y06 = drawings[14]["items"][0][1].y
    if not np.allclose([x0, x40, y1, y06], list(AXIS_CALIBRATION.values()), rtol=0, atol=1e-5):
        raise ValueError("Unexpected PDF axis geometry; inspect before using the extracted values")
    references = {}
    for label, _, index, color in CURVES:
        drawing = drawings[index]
        if (drawing["color"] != color
                or not all(item[0] == "l" for item in drawing["items"])):
            raise ValueError("Unexpected PDF curve geometry; inspect before extracting values")
        points = np.array([
            (drawing["items"][0][1].x, drawing["items"][0][1].y),
            *((item[2].x, item[2].y) for item in drawing["items"]),
        ])
        times = (points[:, 0] - x0) * 40 / (x40 - x0)
        published = 1 - .4 * (points[:, 1] - y1) / (y06 - y1)
        references[label] = validate_reference(np.column_stack((times, published)))
    return references, "hash_checked_pdf_vector_extraction"


def make_report(reference_input):
    """Keep source identities fixed and record this run's numerical environment."""
    return {
        "paper": {
            "title": "QuTiP-BoFiN: A bosonic and fermionic numerical hierarchical-equations-of-motion library with applications in light-harvesting, quantum control, and single-molecule electronics",
            "authors": "N. Lambert et al.",
            "journal": "Physical Review Research 5, 013181 (2023)",
            "doi": "10.1103/PhysRevResearch.5.013181",
            "html": "https://arxiv.org/html/2010.10806v3",
            "pdf": "https://arxiv.org/pdf/2010.10806v3",
            "pdf_sha256": PDF_SHA256,
            "figure": "Figure 1, upper panel, PDF page 7",
            "notebook": f"https://raw.githubusercontent.com/qutip/qutip-notebooks/{NOTEBOOK_COMMIT}/examples/heom/heom-1a-spin-bath-model-basic.ipynb",
            "notebook_commit": NOTEBOOK_COMMIT,
            "notebook_sha256": NOTEBOOK_SHA256,
            "source_parameters": "Figure caption: epsilon=.5 Delta, lambda=.1 Delta, gamma=.5 Delta, T=.5 Delta. Eqs.13-15: H=(epsilon sigma_z+Delta sigma_x)/2, Q=sigma_z, J=2 lambda gamma omega/(omega^2+gamma^2). Author notebook: NC=5, Nk=2, rho0=|0><0|, tlist=linspace(0,50,1000).",
        },
        "parameters": {
            "epsilon": 0.5, "delta": 1.0, "reorganization_energy": 0.1,
            "cutoff": 0.5, "beta": 2.0, "coupling_scale": 1.0,
            "matsubara_terms": 2,
        },
        "initial_density": [[1, 0], [0, 0]],
        "initial_bath": "factorized free thermal bath; all higher ADOs zero",
        "published_hierarchy_depth": 5,
        "reference_input": reference_input,
        "publication_absolute_tolerance": PUBLICATION_TOLERANCE,
        "extraction": {
            "method": "PDF vector-line vertices, affine axis calibration from labelled tick marks",
            "page_zero_based": 6,
            "pixel_digitization": False,
            **AXIS_CALIBRATION,
            "precision_note": "PDF plotting coordinates are rounded and paths are simplified; comparison is at retained vertices, not all 1000 original time samples. Differences below about 1e-6 cannot be interpreted as solver accuracy.",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "machine": platform.machine(),
            "processor": platform.processor() or "not reported by platform.processor",
            "logical_cpus_visible": os.cpu_count(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "timing_note": "Wall times for this invocation; shared-host load is not controlled.",
        },
        "curves": [],
        "physical_converged": False,
        "interpretation": "Reproduction of the paper's specified finite K=2,L=5 approximation, with and without its Matsubara-tail terminator. Agreement is not a claim that either curve is the fully converged continuum-bath answer; the paper itself demonstrates sensitivity to the tail approximation.",
    }


def plot_comparison(tables, destination):
    """Plot either regenerated tables or the recorded CSV tables without solving."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
        for column, ((label, terminator, _, _), color) in enumerate(zip(CURVES, ("#2359aa", "#087f64"))):
            data = tables[label]
            times, published = data[:, 0], data[:, 1]
            prediction = data[:, 2 + DEPTHS.index(5)]
            residual = prediction - published
            population_axis, residual_axis = axes[:, column]
            population_axis.plot(times, prediction, color=color, lw=1.6, label="PyRPMD HEOM")
            population_axis.plot(times[::4], published[::4], "o", ms=3.2, mfc="white",
                                 mec="#303944", mew=.8, label="Published PDF vertices (every fourth)")
            population_axis.set_title("With Matsubara-tail terminator" if terminator else "Without terminator", fontsize=11, pad=10)
            population_axis.set_ylim(.22, 1.04)
            population_axis.set_ylabel("Initial-state population $P_0$")
            population_axis.legend(loc="upper right", fontsize=8, frameon=False)
            population_axis.grid(axis="y", color="#dfe3e8", lw=.6)
            residual_axis.axhline(0, color="#8a919b", lw=.7)
            residual_axis.plot(times, residual * 1e6, ".-", color=color, ms=2.2, lw=.65)
            residual_axis.set_ylim(-.65, .65)
            residual_axis.set_yticks([-.5, 0, .5])
            residual_axis.set_xlim(0, 50)
            residual_axis.set_xlabel(r"Time $t\Delta$")
            residual_axis.set_ylabel(r"HEOM $-$ PDF ($10^{-6}$)")
            residual_axis.text(.98, .90, f"Max absolute difference: {np.max(np.abs(residual)):.2e}",
                               ha="right", va="top", fontsize=8, transform=residual_axis.transAxes)
        fig.suptitle("Published spin–boson population curves reproduced", fontsize=14, y=.98)
        fig.text(.5, .926, r"Lambert et al., Phys. Rev. Research 5, 013181 (2023), Fig. 1  |  $L=5$, $K=2$, 56 ADOs", ha="center", fontsize=10)
        fig.text(.06, .03, "Reference: published PDF vector vertices; coordinate precision approximately $10^{-6}$.\n"
                 "These are finite-truncation curves. Residuals measure agreement with the figure, not integrator error.", fontsize=8.5, color="#4c5664")
        fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.16, hspace=.30, wspace=.27)
        fig.savefig(destination, dpi=200, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, help="Optional versioned PDF to re-extract (requires PyMuPDF)")
    parser.add_argument("--output-dir", type=Path, help="Write regenerated JSON, CSVs, and PNG here (requires Matplotlib)")
    args = parser.parse_args()
    references, source = read_references(args.pdf)
    report = make_report(source)
    tables = {}
    rho0 = np.diag([1.0, 0.0])
    for label, terminator, drawing_index, _ in CURVES:
        times, published = references[label].T
        curve = {
            "name": "green Matsubara K=2 with terminator" if terminator else "blue Matsubara K=2 without terminator",
            "pdf_drawing_index": drawing_index,
            "terminator": terminator,
            "reference_columns_sha256": hashlib.sha256(references[label].astype("<f8").tobytes()).hexdigest(),
            "published_vertex_count": len(times),
            "time_range": [float(times[0]), float(times[-1])],
            "runs": [],
        }
        populations = []
        for depth in DEPTHS:
            begin = time.perf_counter()
            model = SpinBosonHEOM(**report["parameters"], max_depth=depth, terminator=terminator)
            result = model.propagate(rho0, times)
            elapsed = time.perf_counter() - begin
            prediction = result.populations[:, 0]
            populations.append(prediction)
            difference = prediction - published
            curve["runs"].append({
                "max_depth": depth,
                "ado_count": model.ado_count,
                "run_identity": result.final_state.run_identity,
                "elapsed_seconds": elapsed,
                "max_population_error_vs_published": float(np.max(np.abs(difference))),
                "rms_population_error_vs_published": float(np.sqrt(np.mean(difference ** 2))),
                "final_population": float(prediction[-1]),
                "max_trace_error": result.max_trace_error,
                "max_hermiticity_error": result.max_hermiticity_error,
                "min_density_eigenvalue": result.min_density_eigenvalue,
            })
        for row, population in zip(curve["runs"], populations):
            row["max_population_difference_vs_depth9"] = float(np.max(np.abs(population - populations[-1])))
        curve["publication_reproduced"] = curve["runs"][DEPTHS.index(5)]["max_population_error_vs_published"] <= PUBLICATION_TOLERANCE
        tables[label] = np.column_stack((times, published, *populations))
        report["curves"].append(curve)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_comparison(tables, args.output_dir / "lambert-figure1-comparison.png")
        for label, table in tables.items():
            np.savetxt(args.output_dir / f"lambert-figure1-{label}.csv", table, delimiter=",",
                       header="time,published_population,heom_L3,heom_L5,heom_L7,heom_L9", comments="")
        (args.output_dir / "lambert-figure1-validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    passed = all(curve["publication_reproduced"] for curve in report["curves"])
    summary = {
        "reference_input": source,
        "publication_absolute_tolerance": PUBLICATION_TOLERANCE,
        "passed": passed,
        "output_directory": str(args.output_dir) if args.output_dir is not None else None,
        "recorded_evidence_unchanged": args.output_dir is None,
        "curves": [{
            "name": curve["name"],
            "vertices": curve["published_vertex_count"],
            "at_published_depth5": curve["runs"][DEPTHS.index(5)],
        } for curve in report["curves"]],
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
