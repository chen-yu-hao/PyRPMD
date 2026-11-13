#!/usr/bin/env python3
"""Stream fixed-stage classical data; estimate uncertainty across seed chains.

All complete NVE segments are retained. The independent uncertainty unit is a
newly equilibrated seed chain, never a molecule, overlapping time origin, or
adjacent NVE segment. Reference proximity is not an eligibility criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.water.campaign import atomic_json, atomic_npz, sha256
from benchmarks.water.native_reference import verify_manifest
from benchmarks.water.analyze_campaign import rdf_oxygen
from pyrpmd.analysis.diffusion import analyze_diffusion
from pyrpmd.constants import BOHR_ANGSTROM, FS_TO_AU, KCAL_MOL_TO_AU

OBSERVABLES = ("step", "segment", "segment_step", "molecular_com_bohr", "molecular_velocity_au",
               "oxygen_centroid_bohr", "temperature_momenta_K", "temperature_centroid_K",
               "potential_per_molecule_hartree", "hamiltonian_per_bead_hartree",
               "mean_OH_angstrom", "mean_HOH_degrees")


def chain_uncertainty(values):
    """Student-t interval across independent chain means; singleton has no SEM."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("chain means must be a nonempty finite vector")
    result = {"chains": len(values), "mean": float(values.mean()), "degrees_of_freedom": len(values)-1,
              "standard_deviation": None, "standard_error": None, "CI95_halfwidth": None, "CI95": None}
    if len(values) > 1:
        sd = float(values.std(ddof=1))
        sem = sd / np.sqrt(len(values))
        half = float(student_t.ppf(.975, len(values)-1) * sem)
        result.update(standard_deviation=sd, standard_error=float(sem), CI95_halfwidth=half,
                      CI95=[float(values.mean()-half), float(values.mean()+half)])
    return result


def paired_check(first, second, tolerance=.01, equivalence_margin=.01):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("paired checks require the same seed chains")
    summary = chain_uncertainty(first-second)
    threshold = max(tolerance, summary["CI95_halfwidth"] or 0)
    return {"paired_difference": summary, "threshold_angstrom2_ps": threshold,
            "compatible": len(first) > 1 and abs(summary["mean"]) <= threshold,
            "compatibility_interpretation": "Failure-to-detect-difference screen; a wide CI is not evidence of equivalence.",
            "equivalence_margin_angstrom2_ps": equivalence_margin,
            "equivalent_within_margin": summary["CI95"] is not None and
                summary["CI95"][0] >= -equivalence_margin and summary["CI95"][1] <= equivalence_margin,
            "equivalence_interpretation": "The complete paired-chain 95% CI must lie inside the prespecified symmetric absolute margin."}


def basic_rhat(chain_blocks):
    values = np.asarray(chain_blocks, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        return None
    chains, blocks = values.shape
    within = np.mean(np.var(values, axis=1, ddof=1))
    if within == 0:
        return None
    between = blocks * np.var(values.mean(axis=1), ddof=1)
    return float(np.sqrt(((blocks-1)*within/blocks+between/blocks)/within))


def observation_blocks(directory, manifest):
    molecules = manifest["config"]["cells_per_axis"]**3
    for record in manifest["chunks"]:
        with np.load(directory/record["file"], allow_pickle=False) as data:
            if "segment" in data:
                result = {name: data[name].copy() for name in OBSERVABLES}
                frames = len(result["segment"])
                for name, values in result.items():
                    expected = (frames, molecules, 3) if name in (
                        "molecular_com_bohr", "molecular_velocity_au", "oxygen_centroid_bohr") else (frames,)
                    if values.shape != expected or not np.isfinite(values).all():
                        raise ValueError(f"invalid/nonfinite trajectory observation: {name}")
                yield result


def validate_source_contract(identity, protocol, checkpoint_contract=None):
    """Accept the registered source or the documented checkpoint-only fix."""
    expected = protocol["source_sha256"]
    if identity == expected:
        return
    path = "benchmarks/water/native_reference.py"
    revised = dict(expected)
    if checkpoint_contract is not None and checkpoint_contract["from_source_sha256"] == expected[path]:
        revised[path] = checkpoint_contract["to_source_sha256"]
        if identity == revised:
            return
    raise ValueError("run source differs from preregistration or documented checkpoint-only revision")


def read_sampling_and_segments(directory, manifest):
    """Yield one NVE segment at a time; retain only small sampling histories."""
    sampling, current, active = [], [], None
    for block in observation_blocks(directory, manifest):
        for segment in np.unique(block["segment"]):
            select = block["segment"] == segment
            selected = {name: value[select] for name, value in block.items()}
            if segment == -1:
                sampling.append(selected)
                continue
            if active is not None and segment != active:
                yield "NVE", active, {name: np.concatenate([part[name] for part in current]) for name in OBSERVABLES}
                current = []
            active = int(segment)
            current.append(selected)
    if current:
        yield "NVE", active, {name: np.concatenate([part[name] for part in current]) for name in OBSERVABLES}
    if sampling:
        yield "sampling", -1, {name: np.concatenate([part[name] for part in sampling]) for name in OBSERVABLES}


def analyze_chain(directory, protocol):
    directory = Path(directory)
    manifest_bytes = (directory/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not {"model.xml", "integrator.xml"}.issubset(manifest.get("artifacts", {})):
        raise ValueError("serialized model/integrator identity records are required")
    verify_manifest(directory, manifest)
    if manifest.get("schema") != "pyrpmd.water-openmm-external-reference.v1" or not manifest.get("external_reference"):
        raise ValueError("expected an explicitly labeled native external-reference run")
    config = manifest["config"]
    conditions = protocol["conditions"]
    required = {"cells_per_axis": round(conditions["molecules"]**(1/3)),
                "beads":conditions["beads"],"isotope":conditions["isotope"],
                "temperature_K":conditions["temperature_K"],"density_g_cm3":conditions["density_g_cm3"],
                "dt_fs":conditions["timestep_fs"],"cutoff_angstrom":conditions["short_range_cutoff_angstrom"],
                "dispersion_correction":conditions["tail_correction"],"ewald_tolerance":1e-5,
                "integration":"velocity-verlet","sampling_refresh_fs":100.0,"threads":1,"platform":"CPU"}
    if any(config[name] != value for name,value in required.items()):
        raise ValueError("run model, numerical method, or thermodynamic conditions differ from registration")
    n = config["cells_per_axis"]**3
    dt_ps = config["record_stride"]*config["dt_fs"]/1000
    reference = protocol["fixed_stage"]
    if config["seed"] not in reference["seeds"]:
        raise ValueError("run seed is outside the predeclared stage")
    if (config["segments"] != reference["NVE_segments_per_chain"]
            or not np.isclose(config["segment_steps"]*config["dt_fs"]/1000, reference["NVE_ps_per_segment"])
            or not np.isclose(config["equilibration_steps"]*config["dt_fs"]/1000, reference["new_sampling_ps_per_chain"])):
        raise ValueError("run duration differs from the predeclared fixed stage")
    report = {"seed":config["seed"], "manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
              "config":config, "complete":manifest["complete"], "committed_steps":manifest["committed_steps"],
              "elapsed_seconds":manifest["elapsed_seconds"], "segments":[], "sampling":{},
              "incomplete_segment":None, "raw_chunks_verified":True,
              "raw_directory_name":directory.name, "source_sha256":manifest["source_sha256"],
              "environment":manifest["environment"],
              "source_migrations":manifest.get("source_migrations", []),
              "restart_validations":manifest.get("restart_validations", [])}
    curves, sums = {}, {}
    for phase, index, data in read_sampling_and_segments(directory, manifest):
        if phase == "sampling":
            expected_steps = np.arange(config["equilibration_record_stride"],
                min(manifest["committed_steps"], config["equilibration_steps"])+1,
                config["equilibration_record_stride"])
            if not np.array_equal(data["step"], expected_steps):
                raise ValueError("missing, duplicate, or nonuniform sampling observations")
            times = data["step"]*config["dt_fs"]/1000
            final_time = config["equilibration_steps"]*config["dt_fs"]/1000
            for name in ("temperature_momenta_K", "potential_per_molecule_hartree", "mean_OH_angstrom", "mean_HOH_degrees"):
                windows, counts = [], []
                for lower, upper in ((final_time-50, final_time-25), (final_time-25, final_time)):
                    mask = (times > lower) & (times <= upper)
                    windows.append(float(data[name][mask].mean()) if mask.any() else None)
                    counts.append(int(np.count_nonzero(mask)))
                blocks = []
                for lower in np.arange(final_time-50, final_time, 5):
                    mask = (times > lower) & (times <= lower+5)
                    blocks.append(float(data[name][mask].mean()) if mask.any() else None)
                report["sampling"][name] = {"last_50ps_half_means":windows, "last_50ps_5ps_block_means":blocks,
                                          "half_window_observation_counts":counts,
                                          "expected_observations_per_half_window":round(25000/(config["dt_fs"]*config["equilibration_record_stride"])),
                                          "half_mean_difference":windows[1]-windows[0] if None not in windows else None}
                curves["sampling_"+name] = data[name]
            curves["sampling_time_ps"] = times
            for label, lower, upper in (("early_late",final_time-50,final_time-25),("late",final_time-25,final_time)):
                select = np.flatnonzero((times > lower) & (times <= upper))
                if len(select):
                    stride = max(1, round(.5/(config["equilibration_record_stride"]*config["dt_fs"]/1000)))
                    r, g = rdf_oxygen(data["oxygen_centroid_bohr"][select[::stride]], manifest["box_angstrom"])
                    curves["rdf_r_angstrom"], curves["rdf_"+label] = r,g
            continue
        steps = data["segment_step"]
        expected = np.arange(0, config["segment_steps"]+1, config["record_stride"])
        if not np.array_equal(steps, expected):
            if not manifest["complete"] and len(steps) and np.array_equal(steps, expected[:len(steps)]):
                report["incomplete_segment"] = {"segment":index,"frames":len(steps)}
                continue
            raise ValueError("missing, duplicate, or nonuniform observations in a completed NVE segment")
        times = steps*config["dt_fs"]*FS_TO_AU
        maximum_lag = round(2/dt_ps)
        result = analyze_diffusion(data["molecular_com_bohr"],times,velocities=data["molecular_velocity_au"],
                                   max_lag=maximum_lag,fit_range=(1000*FS_TO_AU,2000*FS_TO_AU),bootstrap=0)
        lag = np.asarray(result["lag_time_fs"])/1000
        msd = np.asarray(result["msd_au"])*BOHR_ANGSTROM**2
        vacf = np.asarray(result["vacf_au"])*(BOHR_ANGSTROM*1000*FS_TO_AU)**2
        integral = np.asarray(result["vacf_integral_m2_s"])*1e8
        energies = data["hamiltonian_per_bead_hartree"]
        delta = (energies-energies[0])/(n*KCAL_MOL_TO_AU)
        row = {"segment":index,"frames":len(steps),"D_VACF_2ps":float(integral[-1]),
               "D_MSD_1_2ps":float(result["raw_fit"]["diffusion_m2_s"]*1e8),
               "log_msd_slope":float(result["raw_fit"]["log_msd_slope"]),
               "individual_window_status":result["status"],"included_in_mean":True,
               "mean_temperature_K":float(data["temperature_centroid_K"].mean()),
               "max_abs_sampled_energy_change_kcal_mol_water":float(np.max(abs(delta))),
               "energy_drift_kcal_mol_water_ps":float(np.polyfit(steps*config["dt_fs"]/1000,delta,1)[0]),
               "VACF_cutoffs":{},"MSD_windows":{}}
        for cutoff in protocol["estimators"]["diagnostics_VACF_cutoffs_ps"]:
            row["VACF_cutoffs"][str(cutoff)] = float(integral[round(cutoff/dt_ps)])
        for low, high in protocol["estimators"]["diagnostics_MSD_windows_ps"]:
            mask = (lag >= low-1e-12) & (lag <= high+1e-12)
            row["MSD_windows"][f"{low:g}-{high:g}"] = float(np.polyfit(lag[mask],msd[mask],1)[0]/6)
        report["segments"].append(row)
        for name, values in (("msd_angstrom2",msd),("vacf_angstrom2_ps2",vacf),("D_VACF_angstrom2_ps",integral)):
            sums[name] = sums.get(name, np.zeros_like(values))+values
            curves[f"segment_{index}_"+name] = values
        curves["lag_time_ps"] = lag
    count = len(report["segments"])
    if manifest["complete"] and count != config["segments"]:
        raise ValueError("completed manifest does not contain every predeclared NVE segment")
    for key, values in sums.items():
        curves["mean_"+key] = values/count
    if count:
        report["mean_D_VACF_2ps"] = float(np.mean([row["D_VACF_2ps"] for row in report["segments"]]))
        report["mean_D_MSD_1_2ps"] = float(np.mean([row["D_MSD_1_2ps"] for row in report["segments"]]))
        report["conditional_segment_SD_VACF"] = float(np.std([row["D_VACF_2ps"] for row in report["segments"]],ddof=1)) if count>1 else None
    return report,curves


def analyze(runs, protocol_path, output):
    protocol_path,output = Path(protocol_path),Path(output)
    analysis_sources = {name:sha256(ROOT/name) for name in (
        "benchmarks/water/analyze_native.py", "benchmarks/water/analyze_campaign.py",
        "benchmarks/water/campaign.py", "benchmarks/water/native_reference.py",
        "src/pyrpmd/analysis/diffusion.py", "src/pyrpmd/constants.py")}
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = sha256(protocol_path)
    supplement_path = protocol_path.parent/"paired-equivalence-preregistration.json"
    supplement = json.loads(supplement_path.read_text())
    supplement_sha = sha256(supplement_path)
    if supplement["protocol_sha256"] != protocol_sha:
        raise ValueError("equivalence preregistration does not identify this protocol")
    margin = supplement["equivalence_margin_angstrom2_ps"]
    contract_path = protocol_path.parent/"checkpoint-contract-v2.json"
    checkpoint_contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
    if checkpoint_contract is not None and sha256(protocol_path.parent/"checkpoint-contract-v2.patch") != checkpoint_contract["patch_sha256"]:
        raise ValueError("checkpoint-only revision patch checksum mismatch")
    output.mkdir(parents=True,exist_ok=True)
    chains, curves, seen = [],{},set()
    for directory in runs:
        report, values = analyze_chain(directory,protocol)
        validate_source_contract(report["source_sha256"], protocol, checkpoint_contract)
        config_path = protocol_path.parent/"configs"/f"classical-seed-{report['seed']}.json"
        if json.loads(config_path.read_text()) != report["config"]:
            raise ValueError("run configuration differs from its registered seed configuration")
        report["registered_config_sha256"] = sha256(config_path)
        if report["seed"] in seen:
            raise ValueError("duplicate seed chains cannot be counted as independent")
        seen.add(report["seed"]); chains.append(report)
        for key,value in values.items():
            curves[f"seed_{report['seed']}_"+key] = value
    chains.sort(key=lambda row:row["seed"])
    completed = [row for row in chains if row["complete"]]
    all_complete = seen == set(protocol["fixed_stage"]["seeds"]) and len(completed)==len(chains)
    result = {"schema":"pyrpmd.water-native-chain-analysis.v1","external_reference":True,
              "pyrpmd_production_trajectory":False,"protocol_sha256":protocol_sha,
              "paired_equivalence_preregistration_sha256":supplement_sha,
              "checkpoint_contract_revision_sha256":sha256(contract_path) if checkpoint_contract is not None else None,
              "all_predeclared_chains_complete":all_complete,"chains":chains,
              "literature_reproduced":False,
              "eligibility_checks":{},"literature_comparison":None,
              "source_sha256":analysis_sources,
              "analysis_environment":{"python":platform.python_version(),
                  "dependencies":{name:importlib.metadata.version(name) for name in ("numpy","scipy")}}}
    # A partly completed stage is a progress report, not an optional stopping
    # opportunity based on target agreement. Do not expose a reference test.
    if all_complete:
        limits = protocol["predeclared_eligibility_checks"]
        gk = [row["mean_D_VACF_2ps"] for row in chains]
        msd = [row["mean_D_MSD_1_2ps"] for row in chains]
        result["diffusion_VACF_angstrom2_ps"] = chain_uncertainty(gk)
        result["diffusion_MSD_angstrom2_ps"] = chain_uncertainty(msd)
        result["estimator_comparison"] = paired_check(gk,msd,equivalence_margin=margin)
        for cutoff in protocol["estimators"]["diagnostics_VACF_cutoffs_ps"]:
            values = [np.mean([segment["VACF_cutoffs"][str(cutoff)] for segment in row["segments"]]) for row in chains]
            result.setdefault("VACF_cutoff_comparison",{})[str(cutoff)] = paired_check(values,gk,equivalence_margin=margin)
        for low,high in protocol["estimators"]["diagnostics_MSD_windows_ps"]:
            name = f"{low:g}-{high:g}"
            values = [np.mean([segment["MSD_windows"][name] for segment in row["segments"]]) for row in chains]
            result.setdefault("MSD_window_comparison",{})[name] = paired_check(values,msd,equivalence_margin=margin)
        checks = result["eligibility_checks"]
        checks["fixed_stage_complete"] = True
        checks["energy_envelope"] = all(segment["max_abs_sampled_energy_change_kcal_mol_water"]<=limits["max_abs_energy_change_kcal_mol_water_per_4ps_segment"] for row in chains for segment in row["segments"])
        checks["energy_linear_drift"] = all(abs(segment["energy_drift_kcal_mol_water_ps"])<=limits["max_abs_energy_linear_drift_kcal_mol_water_ps"] for row in chains for segment in row["segments"])
        checks["precision"] = result["diffusion_VACF_angstrom2_ps"]["CI95_halfwidth"]<=limits["diffusion_95pct_CI_halfwidth_max_angstrom2_ps"]
        checks["MSD_VACF_compatibility"] = result["estimator_comparison"]["compatible"]
        checks["VACF_cutoff_compatibility"] = all(value["compatible"] for value in result["VACF_cutoff_comparison"].values())
        checks["MSD_window_compatibility"] = all(value["compatible"] for value in result["MSD_window_comparison"].values())
        checks["paired_equivalence"] = result["estimator_comparison"]["equivalent_within_margin"] and all(
            value["equivalent_within_margin"] for comparisons in (
                result["VACF_cutoff_comparison"], result["MSD_window_comparison"]) for value in comparisons.values())
        for name,threshold,scale in (("temperature_momenta_K",limits["max_split_mean_temperature_difference_K"],1),
                                     ("potential_per_molecule_hartree",limits["max_split_mean_potential_difference_kcal_mol_water"],1/KCAL_MOL_TO_AU),
                                     ("mean_OH_angstrom",limits["max_split_mean_OH_difference_angstrom"],1),
                                     ("mean_HOH_degrees",limits["max_split_mean_angle_difference_degrees"],1)):
            checks["sampling_"+name] = all(row["sampling"][name]["half_mean_difference"] is not None and abs(row["sampling"][name]["half_mean_difference"]*scale)<=threshold for row in chains)
            result.setdefault("sampling_5ps_block_basic_Rhat",{})[name] = basic_rhat([row["sampling"][name]["last_50ps_5ps_block_means"] for row in chains])
        result["numeric_predeclared_checks_passed"] = all(checks.values())
        result["remaining_reviews"] = limits["additional_reviews"]
        if all(checks.values()):
            comparison = protocol["literature_comparison"]
            estimate = result["diffusion_VACF_angstrom2_ps"]
            combined = float(np.hypot(estimate["standard_error"],comparison["standard_error"]))
            difference = estimate["mean"]-comparison["reference"]
            result["literature_comparison"] = {"difference_angstrom2_ps":difference,"combined_standard_error":combined,
                "difference_over_combined_SE":difference/combined,"reference":comparison,
                "interpretation":"Fixed-stage numerical/statistical screen only. Remaining transport/timestep reviews and external-reference status preclude automatic reproduction certification."}
    if (analysis_sources != {name:sha256(ROOT/name) for name in analysis_sources}
            or sha256(protocol_path)!=protocol_sha or sha256(supplement_path)!=supplement_sha):
        raise RuntimeError("analysis source or preregistration changed during analysis")
    atomic_npz(output/"curves.npz",**curves)
    result["artifacts"] = {"curves.npz":{"sha256":sha256(output/"curves.npz")}}
    atomic_json(output/"report.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",nargs="+",type=Path)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.runs,args.protocol,args.output)
    print(json.dumps({key:result[key] for key in ("all_predeclared_chains_complete","literature_comparison")},indent=2))


if __name__=="__main__":
    main()
