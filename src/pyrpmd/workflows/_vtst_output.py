"""Atomic VTST result serialization and scientific-identity helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import csv
import io
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .. import __version__
from ..input import InputError
from ..io import atomic_write_text, referenced_files, write_json
from ..vtst import MEPOptions

_SCHEMA = "pyrpmd.vtst-run"
_SCHEMA_VERSION = 1

def _scientific_identity(
    config: Mapping[str, Any],
    system: Any,
    temperatures_kelvin: Sequence[float],
    mep: MEPOptions,
    *,
    reactant_symmetry_numbers: Sequence[int],
    reactant_electronic_degeneracies: Sequence[int],
    transition_state_symmetry_number: int,
    transition_state_electronic_degeneracy: int,
    reaction_degeneracy: int,
    tunnelling_quadrature_points: int,
    bond_scale: float,
    source_sha256: str | None,
) -> dict[str, Any]:
    sections = {
        key: copy.deepcopy(config[key])
        for key in ("schema_version", "system", "potential", "reaction", "preparation")
        if key in config
    }
    return {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "input": sections,
        "system": system.to_dict(),
        "source_sha256": source_sha256,
        "referenced_files": referenced_files(config),
        "temperatures_kelvin": list(temperatures_kelvin),
        "mep": {
            "integrator": "normalized_mass_scaled_heun_with_monotonic_line_search",
            **mep.__dict__,
        },
        "partition_functions": {
            "reactant_symmetry_numbers": list(reactant_symmetry_numbers),
            "reactant_electronic_degeneracies": list(
                reactant_electronic_degeneracies
            ),
            "transition_state_symmetry_number": transition_state_symmetry_number,
            "transition_state_electronic_degeneracy": (
                transition_state_electronic_degeneracy
            ),
            "reaction_degeneracy": reaction_degeneracy,
            "translation": "ideal_gas_number_density_per_bohr3",
            "rotation": "classical_rigid_rotor",
            "vibration": "quantum_harmonic",
        },
        "tunnelling": {
            "methods": ["zct", "sct"],
            "model": "uniform_semiclassical_ground_state_wkb_with_reflection",
            "quadrature_points": tunnelling_quadrature_points,
        },
        "endpoint_connectivity": {"bond_scale": bond_scale},
    }


def _path_csv(result: Mapping[str, Any]) -> str:
    path = result["path"]
    modes = result["mode_analysis"]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "index",
            "s_bohr",
            "energy_hartree",
            "gradient_norm_hartree_per_bohr",
            "zero_point_energy_hartree",
            "adiabatic_energy_hartree",
            "curvature_bohr_inverse",
            "effective_frequency_au",
            "transverse_turning_point_bohr",
            "effective_mass_ratio_sct",
            "frequencies_au",
        ]
    )
    for index, (coordinate, energy, gradient, zpe, curvature, omega, turning, ratio, frequencies) in enumerate(
        zip(
            path["s_bohr"],
            path["energies_hartree"],
            path["gradients_hartree_per_bohr"],
            modes["zero_point_energy_hartree"],
            modes["curvature_bohr_inverse"],
            modes["effective_frequency_au"],
            modes["transverse_turning_point_bohr"],
            modes["effective_mass_ratio_sct"],
            modes["frequencies_au"],
        )
    ):
        writer.writerow(
            [
                index,
                format(float(coordinate), ".17g"),
                format(float(energy), ".17g"),
                format(float(np.linalg.norm(gradient)), ".17g"),
                format(float(zpe), ".17g"),
                format(float(energy + zpe), ".17g"),
                format(float(curvature), ".17g"),
                format(float(omega), ".17g"),
                format(float(turning), ".17g"),
                format(float(ratio), ".17g"),
                ";".join(format(float(value), ".17g") for value in frequencies),
            ]
        )
    return output.getvalue()


def _format_optional_float(value: Any) -> str:
    return "" if value is None else format(float(value), ".17g")


def _rates_csv(result: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "temperature_K",
            "rate_unit",
            "tst_rate",
            "cvt_rate",
            "cvt_zct_rate",
            "cvt_sct_rate",
            "tst_s_bohr",
            "cvt_s_bohr",
            "cvt_variational_transmission_coefficient",
            "zct_kappa",
            "sct_kappa",
            "zct_log_kappa",
            "sct_log_kappa",
        ]
    )
    for row in result["temperatures"]:
        writer.writerow(
            [
                format(float(row["temperature_kelvin"]), ".17g"),
                row["cvt"]["rate_unit"],
                row["tst"]["rate"],
                row["cvt"]["rate"],
                row["cvt_zct"]["rate"],
                row["cvt_sct"]["rate"],
                format(float(row["tst"]["s_bohr"]), ".17g"),
                format(float(row["cvt"]["s_bohr"]), ".17g"),
                format(
                    float(row["cvt"]["variational_transmission_coefficient"]),
                    ".17g",
                ),
                _format_optional_float(row["zct"]["kappa"]),
                _format_optional_float(row["sct"]["kappa"]),
                format(float(row["zct"]["log_kappa"]), ".17g"),
                format(float(row["sct"]["log_kappa"]), ".17g"),
            ]
        )
    return output.getvalue()


def _summary(result: Mapping[str, Any], output: Path, fingerprint: str) -> dict[str, Any]:
    rows = []
    for row in result["temperatures"]:
        rows.append(
            {
                "temperature_kelvin": row["temperature_kelvin"],
                "rate_unit": row["cvt"]["rate_unit"],
                "tst_rate": row["tst"]["rate"],
                "cvt_rate": row["cvt"]["rate"],
                "cvt_zct_rate": row["cvt_zct"]["rate"],
                "cvt_sct_rate": row["cvt_sct"]["rate"],
                "cvt_s_bohr": row["cvt"]["s_bohr"],
                "zct_kappa": row["zct"]["kappa"],
                "sct_kappa": row["sct"]["kappa"],
            }
        )
    return {
        "schema": "pyrpmd.vtst-summary",
        "schema_version": 1,
        "task": "vtst",
        "output": str(output),
        "physics_fingerprint": fingerprint,
        "physical_converged": False,
        "numerical_converged": False,
        "mep_side_force_converged": result["path"]["side_converged"],
        "endpoint_connected": result["endpoint_validation"].get("connected"),
        "rates": rows,
        "warnings": result["warnings"],
    }


def _publish(
    target: Path,
    *,
    config: Mapping[str, Any],
    system: Any,
    source_path: Path | None,
    identity: Mapping[str, Any],
    fingerprint: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
        )
    )
    try:
        for child in ("input", "results"):
            (stage / child).mkdir()
        write_json(stage / "input" / "resolved.json", config)
        write_json(stage / "input" / "system.json", system.to_dict())
        if source_path is not None:
            shutil.copy2(
                source_path,
                stage / "input" / ("original" + source_path.suffix.lower()),
            )
        manifest = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "software": "PyRPMD",
            "version": __version__,
            "physics_fingerprint": fingerprint,
            "physics_identity": identity,
            "physical_converged": False,
            "restartable": False,
            "rng": "not_used",
            "units": {
                "length": "bohr",
                "energy": "hartree",
                "mass": "electron_mass",
                "time": "atomic_time",
            },
        }
        write_json(stage / "manifest.json", manifest, sort_keys=True)
        write_json(stage / "results" / "vtst.json", result)
        atomic_write_text(stage / "results" / "vtst_path.csv", _path_csv(result))
        atomic_write_text(stage / "results" / "vtst_rates.csv", _rates_csv(result))
        summary = _summary(result, target, fingerprint)
        write_json(stage / "results" / "summary.json", summary)
        if target.exists():
            raise InputError(
                f"VTST output appeared while publishing: {target}; no files were replaced"
            )
        os.replace(stage, target)
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)

