"""Export the recorded T07 monomer runs without storing large trajectory files.

From an installed checkout with the plots extra, pass the metrics JSON whose
cases name the actual stage output directories:

    python docs/validation/t07-qcmd/collect_results.py --metrics ../t07-monomer-metrics.json

The original stage reports, complete fitted-model records, band spectra CSV
and a two-panel figure are retained beside this script. Numerical values are
rechecked against the saved, integrity-verified spectra before export.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyrpmd.checkpoint import load_checkpoint
from pyrpmd.qcmd import QCMDModel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path(__file__).resolve().parent
    metrics = json.loads(args.metrics.read_text())
    records, curves = [], {}
    for case in metrics["cases"]:
        temperature, seed = case["temperature_K"], case["master_seed"]
        for method, directory_key, report_key, value_key in (
            ("pa-cmd", "cmd_output_directory", "cmd_report", "pa_cmd_peak_cm1"),
            ("f-qcmd", "fqcmd_output_directory", "fqcmd_report", "f_qcmd_peak_cm1"),
        ):
            report = json.loads(Path(case[report_key]).read_text())
            (output / f"{temperature}K-seed{seed}-{method}.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n")
            spectrum = Path(case[directory_key]) / f"{method}-spectrum.npz"
            with load_checkpoint(spectrum, require_integrity=True) as saved:
                frequency = saved["wavenumber_cm1"].copy()
                intensity = saved["relative_ir"].copy()
            band = (frequency >= 2800.) & (frequency <= 4400.)
            x, y = frequency[band], intensity[band]
            assert float(x[np.argmax(y)]) == case[value_key]
            curves.setdefault((temperature, method), []).append((x, y))
            records.extend((temperature, seed, method, float(a), float(b)) for a, b in zip(x, y))
    for fit in metrics["fit"]:
        temperature = fit["temperature_K"]
        report_file = Path(fit["report"])
        (output / f"{temperature}K-fit.json").write_text(report_file.read_text())
        model = QCMDModel.load(report_file.parent / "qcmd-model.npz")
        (output / f"{temperature}K-model.json").write_text(
            json.dumps(model.to_dict(), indent=2, allow_nan=False) + "\n")
    (output / "monomer-metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    with (output / "monomer-spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["temperature_K", "master_seed", "method", "wavenumber_cm1", "relative_ir"])
        writer.writerows(records)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    colors = {"pa-cmd": "#ba532f", "f-qcmd": "#186fa0"}
    for axis, temperature in zip(axes, (150, 300)):
        peak = {}
        for method in ("pa-cmd", "f-qcmd"):
            samples = curves[(temperature, method)]
            x = samples[0][0]
            assert all(np.array_equal(x, item[0]) for item in samples)
            mean = np.mean([item[1] for item in samples], axis=0)
            peak[method] = float(x[np.argmax(mean)])
            if len(samples) > 1:
                for _, y in samples:
                    axis.plot(x, y / y.max(), color=colors[method], alpha=.22, lw=1.)
            axis.plot(x, mean / mean.max(), color=colors[method], lw=2.2,
                      label="PA-CMD" if method == "pa-cmd" else "f-QCMD")
        label = "16 monomers, 2 production seeds" if temperature == 150 else "8 monomers, 1 production seed"
        axis.set_title(f"{temperature} K | {label}", fontsize=12, pad=15)
        axis.text(.04, .93, f"Peak shift: +{peak['f-qcmd'] - peak['pa-cmd']:.0f} cm$^{{-1}}$",
                  transform=axis.transAxes, ha="left", va="top")
        axis.set_xlim(2800, 4400)
        axis.set_ylim(0, 1.25)
        axis.set_xlabel("Wavenumber (cm$^{-1}$)")
        axis.grid(alpha=.15)
        axis.legend(loc="upper right", frameon=False)
    axes[0].set_ylabel("Relative IR, normalized within OH band")
    fig.suptitle("Isolated q-TIP4P/F water monomers", y=.99, fontsize=15)
    fig.text(.5, .015, "P = 32 for target PIMD / PA-CMD; 1 ps production per trajectory. Thin lines: individual seeds.\n"
             "Different PES from the published gas-phase QCMD comparison; physical convergence is not established.",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, bottom=.21, top=.82, wspace=.1)
    fig.savefig(output / "monomer-spectra.png", dpi=180)
    plt.close(fig)
    print(f"Exported {len(metrics['cases'])} measured cases and {len(records)} spectral rows.")


if __name__ == "__main__":
    main()
