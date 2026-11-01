#!/usr/bin/env python3
"""Independent map, virtual-site, and throughput checks for native references."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import Config as PyConfig, atomic_json, atomic_npz, create_dynamics, sha256
from benchmarks.water.native_reference import (
    BOHR_NM, HARTREE_KJ_MOL, VELOCITY_NM_PS, Config, create_engine, get_state,
    sample_momenta, set_state,
)
from benchmarks.water.qtip4pf import initial_grid, make_system, PARAMETERS
from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU, KCAL_MOL_TO_AU
from pyrpmd.dynamics import State


def native_rpmd_map(beads=8, steps=20, *, match_openmm_hbar=False):
    import openmm as mm
    from openmm import unit
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=617)
    config = PyConfig(cells_per_axis=2, beads=beads, cutoff_angstrom=2.5,
                      platform="Reference", threads=1, dt_fs=.1, ewald_tolerance=1e-5)
    provider, dynamics = create_dynamics(config, box)
    rng = np.random.default_rng(812)
    state = State(np.repeat(q[None], beads, axis=0) + rng.normal(0, .01, (beads, 24, 3)),
                  dynamics.sample_momenta())
    frequency_ratio = 1.0
    if match_openmm_hbar:
        # This is a deterministic-map control only. Keep the same physical q,p;
        # vary the PyRPMD free-ring frequency convention to isolate OpenMM's
        # source-code value of hbar. Do not use this effective beta for sampling.
        from scipy.constants import Avogadro
        from pyrpmd.dynamics import Dynamics
        native_hbar_kj_mol_ps = 1.054571628e-34 * Avogadro / (1000 * 1e-12)
        native_boltz_kj_mol_K = float(unit.MOLAR_GAS_CONSTANT_R / (unit.kilojoule_per_mole / unit.kelvin))
        native_beta = native_hbar_kj_mol_ps * (1000 * FS_TO_AU) / (native_boltz_kj_mol_K * 298)
        frequency_ratio = dynamics.beta / native_beta
        dynamics = Dynamics(provider, dynamics.masses, native_beta, beads, .1 * FS_TO_AU,
                            native=True, threads=1, integrator="normal-mode-bab")
    system = make_system(molecules=8, box_angstrom=box, cutoff_angstrom=2.5, ewald_tolerance=1e-5)
    integrator = mm.RPMDIntegrator(beads, 298, 10, .0001)
    integrator.setApplyThermostat(False)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    try:
        for b in range(beads):
            positions, velocities = np.zeros((8, 4, 3)), np.zeros((8, 4, 3))
            positions[:, :3] = state.q[b].reshape(8, 3, 3) * BOHR_NM
            velocities[:, :3] = (state.p[b] / dynamics.masses[:, None]).reshape(8, 3, 3) * VELOCITY_NM_PS
            positions[:, 3] = PARAMETERS["gamma"] * positions[:, 0] + (1-PARAMETERS["gamma"]) * (positions[:, 1]+positions[:, 2])/2
            integrator.setPositions(b, positions.reshape(-1, 3))
            integrator.setVelocities(b, velocities.reshape(-1, 3))
        integrator.step(steps)
        for _ in range(steps):
            dynamics.step(state)
        qq, pp = [], []
        for b in range(beads):
            native = integrator.getState(b, getPositions=True, getVelocities=True)
            qq.append(np.asarray(native.getPositions(asNumpy=True) / unit.nanometer).reshape(8, 4, 3)[:, :3].reshape(24, 3) / BOHR_NM)
            vv = np.asarray(native.getVelocities(asNumpy=True) / (unit.nanometer/unit.picosecond)).reshape(8, 4, 3)[:, :3].reshape(24, 3) / VELOCITY_NM_PS
            pp.append(vv * dynamics.masses[:, None])
        return {"beads": beads, "steps": steps, "dt_fs": .1, "platform": "Reference",
                "maximum_position_difference_bohr": float(np.max(abs(np.asarray(qq)-state.q))),
                "maximum_momentum_difference_au": float(np.max(abs(np.asarray(pp)-state.p))),
                "thermostat": False, "contraction": None,
                "match_openmm_hbar_frequency_control": match_openmm_hbar,
                "openmm_to_modern_free_ring_frequency_ratio": frequency_ratio if match_openmm_hbar else None,
                "comparison": "native OpenMM RPMDIntegrator vs PyRPMD normal-mode-bab at identical physical q,p"}
    finally:
        provider.close()
        del context, integrator


def custom_virtual_site_probe():
    """Record actual last-kick force, separately from endpoint force re-query."""
    import openmm as mm
    q, box = initial_grid(cells_per_axis=2, density_g_cm3=.997, seed=5)
    provider, dynamics = create_dynamics(PyConfig(cells_per_axis=2, beads=1, cutoff_angstrom=2.5,
                                                  platform="Reference", threads=1, dt_fs=.1,
                                                  ewald_tolerance=1e-5), box)
    system = make_system(molecules=8, box_angstrom=box, cutoff_angstrom=2.5, ewald_tolerance=1e-5)
    integrator = mm.CustomIntegrator(.0001)
    integrator.addPerDofVariable("first_force", 0)
    integrator.addPerDofVariable("last_force", 0)
    integrator.addComputePerDof("first_force", "f")
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("last_force", "f")
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    try:
        from openmm import unit
        p = sample_momenta(np.random.default_rng(51), dynamics.masses, 298, remove_com=True)
        positions, velocities = np.zeros((8, 4, 3)), np.zeros((8, 4, 3))
        positions[:, :3] = q.reshape(8, 3, 3) * BOHR_NM
        velocities[:, :3] = (p[0] / dynamics.masses[:, None]).reshape(8, 3, 3) * VELOCITY_NM_PS
        context.setPositions(positions.reshape(-1, 3)); context.computeVirtualSites()
        context.setVelocities(velocities.reshape(-1, 3))
        initial_force = provider.evaluate(q[None]).forces
        integrator.step(1)
        native = context.getState(getPositions=True, getVelocities=True)
        final_q = np.asarray(native.getPositions(asNumpy=True)/unit.nanometer).reshape(8,4,3)[:,:3].reshape(1,24,3)/BOHR_NM
        final_p = np.asarray(native.getVelocities(asNumpy=True)/(unit.nanometer/unit.picosecond)).reshape(8,4,3)[:,:3].reshape(1,24,3)/VELOCITY_NM_PS*dynamics.masses[None,:,None]
        actual_force = provider.evaluate(final_q).forces
        force = lambda name: np.asarray(integrator.getPerDofVariableByName(name)).reshape(8,4,3)[:,:3].reshape(1,24,3)*BOHR_NM/HARTREE_KJ_MOL
        return {"platform":"Reference", "dt_fs":.1, "steps":1,
                "initial_force_max_error_hartree_bohr":float(np.max(abs(force("first_force")-initial_force))),
                "last_kick_force_max_error_hartree_bohr":float(np.max(abs(force("last_force")-actual_force))),
                "canonical_momentum_map_error_au":float(np.max(abs(final_p-p-.5*.1*FS_TO_AU*(initial_force+actual_force)))),
                "interpretation":"On this runtime the CustomIntegrator last kick uses intermediate stale M-site positions. Force captured inside the integrator differs from a fresh independent endpoint evaluation. This path is not used for production.",
                "virtual_site_update_source":"https://github.com/openmm/openmm/blob/8.6.0/platforms/reference/src/SimTKReference/ReferenceCustomDynamics.cpp"}
    finally:
        provider.close()
        del context, integrator


def cpu_thread_timing(snapshot, steps=500):
    with np.load(snapshot, allow_pickle=False) as data:
        q, p, box = data["q"], data["p"], float(data["box_angstrom"])
    rows, endpoints = [], []
    for threads in (1, 2):
        config = Config(threads=threads)
        system, integrator, context, masses = create_engine(config, box)
        try:
            set_state(context, q, p, masses)
            started = time.perf_counter()
            integrator.step(steps)
            elapsed = time.perf_counter()-started
            q1, p1, u = get_state(context, masses, energy=True)
            endpoints.append((q1, p1))
            rows.append({"threads":threads,"steps":steps,"dt_fs":config.dt_fs,
                         "seconds":elapsed,"seconds_per_step":elapsed/steps,
                         "potential_per_water_kcal_mol":u/(216*KCAL_MOL_TO_AU)})
        finally:
            del context, integrator
    return {"cases":rows,"maximum_thread_count_position_difference_bohr":float(np.max(abs(endpoints[0][0]-endpoints[1][0]))),
            "maximum_thread_count_momentum_difference_au":float(np.max(abs(endpoints[0][1]-endpoints[1][1]))),
            "note":"Parallel force summation is not bitwise invariant; no long-trajectory identity is assumed."}


def quantization_probe(raw_run):
    """Bound float32 storage effects using an already saved 4 ps trajectory."""
    from pyrpmd.analysis.diffusion import analyze_diffusion
    raw_run=Path(raw_run)
    manifest=json.loads((raw_run/"manifest.json").read_text())
    config=manifest["config"]
    positions, velocities, indices=[],[],[]
    source_chunks=[]
    for entry in manifest["chunks"]:
        with np.load(raw_run/entry["file"],allow_pickle=False) as data:
            select=data["segment"]==0
            if select.any():
                positions.append(data["molecular_com_bohr"][select]);velocities.append(data["molecular_velocity_au"][select]);indices.append(data["segment_step"][select])
                source_chunks.append({"file":entry["file"],"sha256":sha256(raw_run/entry["file"])})
    positions=np.concatenate(positions);velocities=np.concatenate(velocities);indices=np.concatenate(indices)
    outcomes=[]
    for dtype in (np.float64,np.float32):
        result=analyze_diffusion(positions.astype(dtype),indices*config["dt_fs"]*FS_TO_AU,
                                velocities=velocities.astype(dtype),max_lag=400,
                                fit_range=(1000*FS_TO_AU,2000*FS_TO_AU),bootstrap=0)
        outcomes.append({"dtype":np.dtype(dtype).name,"D_MSD_angstrom2_ps":result["raw_fit"]["diffusion_m2_s"]*1e8,
                         "D_VACF_angstrom2_ps":result["vacf_integral_m2_s"][-1]*1e8})
    return {"source_chunks":source_chunks,"cases":outcomes,
            "absolute_D_MSD_difference_angstrom2_ps":abs(outcomes[0]["D_MSD_angstrom2_ps"]-outcomes[1]["D_MSD_angstrom2_ps"]),
            "absolute_D_VACF_difference_angstrom2_ps":abs(outcomes[0]["D_VACF_angstrom2_ps"]-outcomes[1]["D_VACF_angstrom2_ps"])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--raw-quantization-source",type=Path)
    parser.add_argument("--timing",action="store_true")
    args=parser.parse_args()
    import openmm
    report={"schema":"pyrpmd.water-native-validation.v1","openmm":openmm.__version__,
            "source_sha256":{name:sha256(ROOT/name) for name in (
                "benchmarks/water/native_validation.py","benchmarks/water/native_reference.py",
                "benchmarks/water/qtip4pf.py","src/pyrpmd/dynamics.py")},
            "external_reference":True,"custom_virtual_site_probe":custom_virtual_site_probe(),
            "native_rpmd_maps":[native_rpmd_map(1),native_rpmd_map(8),
                                native_rpmd_map(8, match_openmm_hbar=True)]}
    if args.timing:
        report["cpu_threads"]=cpu_thread_timing(ROOT/"docs/validation/literature-audit-2026-11-01/water/classical-pilot/endpoint.npz")
    if args.raw_quantization_source:
        report["float32_storage"]=quantization_probe(args.raw_quantization_source)
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/"native-validation.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
