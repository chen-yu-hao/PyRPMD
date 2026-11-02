#!/usr/bin/env python3
"""Plot recorded preparation diagnostics, without a diffusion-convergence claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from benchmarks.water.campaign import atomic_json,sha256
from pyrpmd.constants import KCAL_MOL_TO_AU


def plot(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory=Path(directory)
    report=json.loads((directory/"report.json").read_text())
    with np.load(directory/"curves.npz",allow_pickle=False) as data:
        curves={name:data[name].copy() for name in data.files}
    rows=(("temperature_momenta_K","Kinetic temperature (K)",1),
          ("potential_per_molecule_hartree","Potential energy (kcal/mol/water)",1/KCAL_MOL_TO_AU),
          ("mean_OH_angstrom","Mean OH distance (Å)",1),
          ("mean_HOH_degrees","Mean HOH angle (degrees)",1))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.labelsize":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    colours=plt.get_cmap("tab10")
    for colour,chain in enumerate(report["chains"]):
        seed=chain["seed"];prefix=f"seed_{seed}_"
        times=curves[prefix+"sampling_time_ps"]
        stride=float(np.median(np.diff(times))) if len(times)>1 else .1
        window=max(1,round(5/stride))
        for axis,(name,label,scale) in zip(axes.flat,rows,strict=True):
            values=curves[prefix+"sampling_"+name]*scale
            axis.plot(times,values,color=colours(colour),alpha=.18,lw=.65)
            if len(times)>=window:
                kernel=np.ones(window)/window
                axis.plot(np.convolve(times,kernel,mode="valid"),np.convolve(values,kernel,mode="valid"),
                          color=colours(colour),lw=2,label=f"Seed {seed} · {times[-1]:g} ps")
            axis.set_ylabel(label)
    axes[0,0].axhline(298,color="#475569",lw=1,ls="--",label="298 K target")
    for axis in axes.flat:
        axis.axvspan(50,75,color="#94a3b8",alpha=.07,zorder=-2)
        axis.axvspan(75,100,color="#94a3b8",alpha=.14,zorder=-2)
        axis.axvline(50,color="#94a3b8",lw=.8,ls=":")
        axis.axvline(75,color="#94a3b8",lw=.8,ls=":")
        axis.set_xlim(0,100);axis.set_xticks([0,25,50,75,100])
        axis.grid(axis="y",color="#e2e8f0",lw=.65)
        axis.tick_params(labelsize=9)
    axes[0,0].legend(loc="best",frameon=False,fontsize=8)
    for axis in axes[1]:axis.set_xlabel("New preparation time (ps)")
    title="q-TIP4P/F classical preparation"
    if not report["all_predeclared_chains_complete"]:title+=" — campaign in progress"
    fig.suptitle(title,x=.075,y=.98,ha="left",fontsize=18,fontweight="bold")
    fig.text(.075,.936,"216 molecules · 298 K · full PME · 0.25 fs native Verlet · external OpenMM reference",fontsize=11,color="#475569")
    fig.text(.075,.018,"Thin lines: 0.1 ps observations. Thick lines: 5 ps rolling means. Shading: registered final 25 ps windows.\nPreparation diagnostics alone do not establish transport convergence.",fontsize=9,color="#475569")
    fig.tight_layout(rect=(.02,.065,.99,.92))
    fig.savefig(directory/"preparation-progress.png",dpi=170,facecolor="white")
    fig.savefig(directory/"preparation-progress.svg",facecolor="white")
    plt.close(fig)
    atomic_json(directory/"figure-provenance.json",{
        "report_sha256":sha256(directory/"report.json"),"curves_sha256":sha256(directory/"curves.npz"),
        "plot_source_sha256":sha256(Path(__file__)),
        "artifacts":{name:sha256(directory/name) for name in ("preparation-progress.png","preparation-progress.svg")}})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    plot(parser.parse_args().directory)
