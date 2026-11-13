"""Generate and import auditable GLE/PIGLET parameters, in atomic units.

The public functions return detached, JSON-serializable ``gle`` mappings for
``Dynamics``, ``NativeEnsemble`` and ``PropagationPlan``.  No downloaded matrix
library is bundled.  A is a drift rate, C is the free-particle covariance
divided by the ring-polymer temperature P/beta, and the continuous diffusion
is A C + C A.T.  In particular, C is *not* the noise amplitude B.

The canonical generator includes the analytic white-noise OPT(V) limit;
colored fits and PIGLET fits use the harmonic Lyapunov equations of Ceriotti,
Bussi and Parrinello, JCTC 6, 1170 (2010), and the centroid constraint of
Ceriotti and Manolopoulos, PRL 109, 100604 (2012).  Fit convergence is never a
claim of anharmonic, timestep, statistical or physical convergence.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import numbers
from pathlib import Path
import re

import numpy as np

from .constants import KB_AU
from .units import parse_quantity, unit_factor

_SCHEMA = "pyrpmd.gle-parameters.v1"
_MODES = {"gle", "nm-gle", "nm-gle-g", "piglet"}
# Historical serialization factors in gle4md/ingen/ingen.js.  These are unit
# definitions of that exporter, not changes to PyRPMD's physical constants.
_GLE4MD_KELVIN = {"k": 1., "ev": 11604.506, "hartree": 315774.67,
                  "ha": 315774.67, "kcalmol": 503.55573, "kjmol": 120.27222}
_GLE4MD_AUTIME_PS = 2.4188843e-5
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_MATRIX_HEADER = re.compile(r"^([AC])\s+MATRIX\s*:\s*(?:\(([^()]*)\))?\s*$", re.I)
_MODE_HEADER = re.compile(r"^Matrix\s+for\s+normal\s+mode\s*:\s*(\d+)\s*$", re.I)


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive real number")
    return float(value)


def _count(value, name, minimum=1, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        bound = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{bound}")
    return int(value)


def _mode(mode):
    if not isinstance(mode, str) or mode.lower().replace("_", "-") not in _MODES:
        raise ValueError("GLE parameter mode must be gle, nm-gle, nm-gle-g or piglet")
    return mode.lower().replace("_", "-")


def _unit_name(unit):
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("an explicit, nonempty matrix unit is required")
    result = unit.strip().lower().replace("−", "-")
    for source, target in (
        ("atomic time units", "autime"), ("atomic time unit", "autime"),
        ("atomic energy units", "hartree"), ("atomic energy unit", "hartree"),
        ("picoseconds", "ps"), ("picosecond", "ps"),
        ("femtoseconds", "fs"), ("femtosecond", "fs"),
        ("seconds", "s"), ("second", "s"), ("kelvin", "k"),
        ("electronvolts", "ev"), ("electronvolt", "ev"),
    ):
        result = result.replace(source, target)
    return result


def _frequency_factor(unit):
    """Physical angular frequency; spectroscopic cm^-1 means E/hbar."""
    normalized = _unit_name(unit)
    if normalized in ("cm^-1", "cm-1", "wavenumber"):
        return unit_factor(normalized, "energy")  # hbar = 1 in atomic units
    if normalized in ("hz", "thz"):
        return 2 * math.pi * (1e12 if normalized == "thz" else 1) * unit_factor("s^-1", "frequency")
    return unit_factor(normalized, "frequency")


def _drift_factor(unit):
    normalized = _unit_name(unit)
    # Matrix A is a rate, not a cyclic frequency.  Do not silently add 2*pi.
    if normalized in ("hz", "thz"):
        raise ValueError("GLE drift units must be inverse time, not ambiguous Hz/THz")
    return _frequency_factor(normalized)


def _source_drift_factor(unit, convention):
    normalized = _unit_name(unit).replace(" ", "").replace("**", "^")
    if convention == "gle4md" and normalized in (
            "autime^-1", "1/autime", "au", "a.u.", "atomic", "atomic_units"):
        return unit_factor("ps^-1", "frequency") / _GLE4MD_AUTIME_PS
    return _drift_factor(unit)


def _source_energy_factor(unit, convention):
    normalized = _unit_name(unit).replace("kcal/mol", "kcalmol").replace("kj/mol", "kjmol")
    if convention == "gle4md":
        if normalized in ("au", "a.u.", "atomic", "atomic_units"):
            normalized = "hartree"
        if normalized not in _GLE4MD_KELVIN:
            raise ValueError(f"unsupported gle4md historical energy unit {unit!r}")
        return _GLE4MD_KELVIN[normalized] * KB_AU
    return (KB_AU * unit_factor(normalized, "temperature") if normalized == "k"
            else unit_factor(normalized, "energy"))


def _frequency_range(value, frequency_unit):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (2,):
        raise ValueError("frequency_range must contain two positive finite frequencies")
    band = np.array([_positive(item, "frequency_range") for item in raw]) * _frequency_factor(frequency_unit)
    if not np.isfinite(band).all() or np.any(band <= 0) or band[0] > band[1]:
        raise ValueError("frequency_range requires 0 < minimum <= maximum")
    return band.tolist()


def _json_copy(value):
    """Reject nonportable identity values, rather than stringify them."""
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("parameter_metadata keys must be strings")
            for val in item.values():
                check(val)
        elif isinstance(item, (list, tuple)):
            for val in item:
                check(val)
    check(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("parameter_metadata must contain finite JSON values") from exc


def _real_array(value, name):
    try:
        raw = np.asarray(value)
        original = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real matrix") from exc
    if raw.dtype.kind not in "fiu" or any(isinstance(item, (bool, np.bool_)) for item in original.flat):
        raise ValueError(f"{name} must contain real numerical values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def validate_gle_parameters(spec, *, mode="gle"):
    """Check stable drift, positive covariance and factorizable diffusion.

    Return numerical diagnostics without changing any input matrix.  The
    generalized fluctuation--dissipation relation is D = A C + C A.T; only
    canonical GLE requires unit physical variance and zero p/s covariance.
    A nonsymmetric drift need not itself be positive definite.
    """
    mode = _mode(mode)
    if not isinstance(spec, dict) or "A" not in spec:
        raise ValueError("GLE parameters require an A matrix")
    a = _real_array(spec["A"], "GLE A")
    if (a.ndim not in (2, 3) or a.shape[-2] != a.shape[-1]
            or not 1 <= a.shape[-1] <= 64 or (a.ndim == 3 and not len(a))):
        raise ValueError("GLE A must have shape [s,s] or [P,s,s], 1 <= s <= 64")
    c = _real_array(spec.get("C", np.eye(a.shape[-1])), "GLE C")
    if c.ndim == 2 and a.ndim == 3 and c.shape == a.shape[-2:]:
        c = np.repeat(c[None], len(a), axis=0)
    if c.shape != a.shape:
        raise ValueError("GLE covariance dimensions differ from A")
    if mode == "gle" and a.ndim != 2:
        raise ValueError("bead-space gle requires one A matrix; use nm-gle for per-mode matrices")
    if mode == "piglet":
        if a.ndim != 3 or _count(spec.get("beads"), "beads", 2) != len(a):
            raise ValueError("PIGLET requires one matrix per bead and beads >= 2")
        _positive(spec.get("beta"), "beta")
        if not isinstance(spec.get("provenance"), str) or not spec["provenance"].strip():
            raise ValueError("PIGLET requires nonempty parameter provenance")
    aa, cc = (a[None], c[None]) if a.ndim == 2 else (a, c)
    drift_min, covariance_min, diffusion_min = [], [], []
    for index, (drift, covariance) in enumerate(zip(aa, cc)):
        eigen = np.linalg.eigvals(drift)
        if np.min(eigen.real) <= 0:
            raise ValueError("GLE drift must have eigenvalues with positive real parts")
        cscale = float(np.max(np.abs(covariance)))
        if not np.allclose(covariance, covariance.T, atol=1e-12 * cscale, rtol=1e-12):
            raise ValueError("GLE covariance must be symmetric")
        ceigen = np.linalg.eigvalsh((covariance + covariance.T) / 2)
        if ceigen.min() <= 0:
            raise ValueError("imported/generated GLE covariance must be positive definite")
        canonical = mode != "piglet" or index == 0
        if canonical and (not math.isclose(covariance[0, 0], 1, rel_tol=1e-10, abs_tol=1e-12)
                          or not np.allclose(covariance[0, 1:], 0, atol=1e-12, rtol=0)
                          or not np.allclose(covariance[1:, 0], 0, atol=1e-12, rtol=0)):
            raise ValueError("canonical GLE/centroid needs unit physical variance and zero physical/auxiliary covariance")
        diffusion = drift @ covariance + covariance @ drift.T
        deigen = np.linalg.eigvalsh((diffusion + diffusion.T) / 2)
        dscale = float(np.linalg.norm(diffusion, ord=2))
        if deigen.min() < -1e-11 * dscale:
            raise ValueError("continuous GLE diffusion A C + C A.T is not positive semidefinite")
        drift_min.append(float(eigen.real.min()))
        covariance_min.append(float(ceigen.min()))
        diffusion_min.append(float(deigen.min()))
    return {"minimum_drift_real_eigenvalue_au": min(drift_min),
            "minimum_covariance_eigenvalue": min(covariance_min),
            "minimum_diffusion_eigenvalue_au": min(diffusion_min),
            "matrix_size": a.shape[-1], "matrix_count": len(aa),
            "diffusion_convention": "D = A C + C A.T; B B.T = D"}


def generate_optimal_gle(frequency_range, *, frequency_unit="autime^-1", naux=0,
                         grid_size=32, max_nfev=600):
    """Generate a canonical OPT(V) bath for the stated physical frequency band.

    With naux=0 the analytic minimax white-noise subset has
    gamma=sqrt(omega_min*omega_max), C=1, and
    tau_V=1/(2*gamma)+gamma/(2*omega**2).  For naux>0 a deterministic colored
    fit minimizes the log dimensionless potential autocorrelation time on a
    logarithmic grid.  It does not claim a global optimum.
    """
    band = _frequency_range(frequency_range, frequency_unit)
    naux = _count(naux, "naux", 0, 3)
    grid_size = _count(grid_size, "grid_size", 8, 256)
    max_nfev = _count(max_nfev, "max_nfev", 1, 10000)
    if naux:
        from ._gle_fitting import fit_optimal_gle
        result = fit_optimal_gle(band, naux=naux, grid_size=grid_size, max_nfev=max_nfev)
    else:
        gamma = math.sqrt(band[0]) * math.sqrt(band[1])
        frequencies = np.geomspace(*band, grid_size)
        tau = .5 / gamma + .5 * gamma / frequencies**2
        efficiency = 1 / (frequencies * tau)
        result = {"A": [[gamma]], "C": [[1.]],
                  "provenance": "PyRPMD analytic white-noise OPT(V) minimax subset; no external matrix data",
                  "parameter_metadata": {
                      "schema": _SCHEMA, "generator": "optimal-sampling", "algorithm": "white-noise-minimax-v1",
                      "frequency_range_au": band, "frequency_grid_au": frequencies.tolist(),
                      "objective": "maximize minimum 1/(omega*tau_V) on the frequency interval within scalar Langevin baths",
                      "naux": 0, "grid_size": grid_size, "max_nfev": max_nfev,
                      "residuals": {"log_omega_tau_V": np.log(frequencies * tau).tolist(),
                                    "minimum_efficiency": float(min(efficiency)),
                                    "tau_V_au": tau.tolist()},
                      "fit_converged": True,
                      "references": ["https://doi.org/10.1021/ct900563s"],
                  }}
    result["parameter_metadata"]["input_frequency_unit"] = frequency_unit
    result["parameter_metadata"]["validation"] = validate_gle_parameters(result)
    return _json_copy(result)


def generate_piglet(frequency_range, *, beta, beads, frequency_unit="autime^-1",
                    naux=2, grid_size=32, max_nfev=600, tolerance=.01):
    """Fit quantum harmonic potential and centroid-virial kinetic energies.

    ``beta`` is in inverse hartree; frequencies are angular frequencies unless
    an explicit spectroscopic/cyclic unit is selected.  A canonical centroid
    and one shared noncanonical internal-mode bath enforce the PIGLET
    centroid constraint.  ``tolerance`` bounds the maximum relative energy
    residual on the independent validation grid, not physical convergence.
    Failure to converge or meet it raises ValueError; no preset is substituted.
    """
    from ._gle_fitting import fit_piglet
    band = _frequency_range(frequency_range, frequency_unit)
    result = fit_piglet(band, beta=_positive(beta, "beta"), beads=_count(beads, "beads", 2),
                        naux=naux, grid_size=grid_size, max_nfev=max_nfev, tolerance=tolerance)
    result["parameter_metadata"]["input_frequency_unit"] = frequency_unit
    result["parameter_metadata"]["validation"] = validate_gle_parameters(result, mode="piglet")
    return _json_copy(result)


def _parse_matrix_text(text, *, default_kind="A"):
    """Read the public gle4md raw-output grammar, never execute file text."""
    sections, units, comments = {}, {}, []
    kind, index, rows, explicit_block = default_kind, None, [], False

    def flush():
        nonlocal rows, explicit_block
        if not rows:
            if explicit_block:
                raise ValueError("empty GLE normal-mode matrix block")
            return
        if len(rows) != len(rows[0]) or any(len(row) != len(rows) for row in rows):
            raise ValueError("GLE matrix blocks must be square with equal row lengths")
        group = sections.setdefault(kind, {})
        if index in group:
            raise ValueError(f"duplicate {kind} matrix/mode block")
        group[index] = rows
        rows, explicit_block = [], False

    for lineno, original in enumerate(text.splitlines(), 1):
        data, separator, comment = original.partition("#")
        if separator:
            comment = html.unescape(comment.strip()).replace("\xa0", " ")
            comments.append(comment)
            if not data.strip():
                matrix_header = _MATRIX_HEADER.fullmatch(comment)
                mode_header = _MODE_HEADER.fullmatch(comment)
                if matrix_header:
                    flush()
                    kind, index = matrix_header[1].upper(), None
                    if kind in units or kind in sections:
                        raise ValueError(f"duplicate {kind} MATRIX header")
                    units[kind] = matrix_header[2]
                elif mode_header:
                    flush()
                    index, explicit_block = int(mode_header[1]), True
                elif re.match(r"^(?:[AC]\s+MATRIX|Matrix\s+for\s+normal\s+mode)\b", comment, re.I):
                    raise ValueError(f"invalid matrix header at line {lineno}")
                continue
        if not data.strip():
            continue
        words = data.split()
        if any(re.fullmatch(_NUMBER, word) is None for word in words):
            raise ValueError(f"invalid numerical matrix entry at line {lineno}")
        row = [float(word.replace("D", "e").replace("d", "e")) for word in words]
        if not all(math.isfinite(number) for number in row):
            raise ValueError(f"nonfinite matrix entry at line {lineno}")
        rows.append(row)
    flush()
    if set(units) - set(sections):
        raise ValueError("empty GLE matrix section")
    if not sections:
        raise ValueError("GLE matrix file is empty")
    packed = {}
    for label, blocks in sections.items():
        if set(blocks) == {None}:
            packed[label] = _real_array(blocks[None], label)
        else:
            if None in blocks or set(blocks) != set(range(len(blocks))):
                raise ValueError("normal-mode labels must be contiguous, starting at zero")
            packed[label] = _real_array([blocks[i] for i in range(len(blocks))], label)
    return packed, units, comments


def _header_metadata(comments, unit_convention="physical"):
    result = {}
    for comment in comments:
        bead = re.search(r"\bUse for simulations with\s+(\d+)\s+beads\b", comment, re.I)
        bead = bead or re.fullmatch(r"beads\s*[:=]\s*(\d+)", comment, re.I)
        temperature = re.fullmatch(r"T\s*=\s*(" + _NUMBER + r")\s+(.+)", comment, re.I)
        inverse = re.fullmatch(r"beta\s*[:=]\s*(" + _NUMBER + r")\s*(.*)", comment, re.I)
        frequency = re.fullmatch(r"(?:omega|ω)[ _]*(min|max)\s*=\s*(" + _NUMBER + r")\s+(.+)", comment, re.I)
        values = {}
        if bead:
            values["beads"] = _count(int(bead[1]), "header beads")
        if temperature:
            number = _positive(float(temperature[1].replace("D", "e").replace("d", "e")), "header temperature")
            unit = _unit_name(temperature[2])
            energy = number * _source_energy_factor(unit, unit_convention)
            values["beta"] = 1 / energy
        if inverse:
            number = float(inverse[1].replace("D", "e").replace("d", "e"))
            values["beta"] = _positive(parse_quantity((number, inverse[2] or "hartree^-1"), (0, -1, 0, 0, 0, 0)), "header beta")
        if frequency:
            number = float(frequency[2].replace("D", "e").replace("d", "e"))
            values[f"omega_{frequency[1].lower()}_au"] = _positive(number * _frequency_factor(frequency[3]), "header frequency")
        for key, value in values.items():
            if key in result and not math.isclose(result[key], value, rel_tol=1e-10, abs_tol=0):
                raise ValueError(f"conflicting {key} metadata in GLE file")
            result[key] = value
    if ("omega_min_au" in result and "omega_max_au" in result
            and result["omega_min_au"] > result["omega_max_au"]):
        raise ValueError("file frequency range requires omega_min <= omega_max")
    return result


def _choose_unit(header, explicit, factor, label):
    if header is None and explicit is None:
        raise ValueError(f"{label} unit is missing; supply it explicitly")
    if header is not None and explicit is not None:
        if not math.isclose(factor(header), factor(explicit), rel_tol=1e-12, abs_tol=0):
            raise ValueError(f"explicit {label} unit conflicts with the file header")
    chosen = header if explicit is None else explicit
    return chosen, factor(chosen)


def load_gle4md(path, *, covariance_path=None, drift_unit=None,
                covariance_unit=None, beta=None, beads=None, mode="gle", mode_order="real",
                unit_convention="physical"):
    """Import a gle4md raw text file or separate A/C files.

    Recognized headers include ``# A MATRIX: (picoseconds^-1)``,
    ``# C MATRIX: (K)`` and ``# Matrix for normal mode: 0``.  Headerless files
    require explicit units.  C in energy units is normalized by P/beta;
    use ``covariance_unit='dimensionless'`` only for an already normalized C.
    An absent C means identity for canonical GLE, and is an error for PIGLET.

    ``mode_order='real'`` means centroid, cos1, sin1, ..., Nyquist;
    ``'fourier'`` explicitly reorders a file indexed 0,1,...,P-1.  File bytes,
    comments, conversions and calibration are recorded in parameter_metadata.
    File paths are not required after this call.

    ``unit_convention='physical'`` uses PyRPMD's current physical constants.
    Explicitly select ``'gle4md'`` to decode the website's historical rounded
    energy/time conversion constants (notably eV/Hartree exports).  That choice
    and its conversion factors are recorded; no covariance is silently rescaled
    to make a mismatched centroid pass validation.
    """
    mode = _mode(mode)
    if mode_order not in ("real", "fourier"):
        raise ValueError("mode_order must be 'real' or 'fourier'")
    if unit_convention not in ("physical", "gle4md"):
        raise ValueError("unit_convention must be 'physical' or 'gle4md'")
    sources, all_comments = [], []

    def read(source, default_kind):
        payload = Path(source).read_bytes()
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("GLE files must be UTF-8 text") from exc
        matrices, units, comments = _parse_matrix_text(text, default_kind=default_kind)
        sources.append({"role": default_kind, "sha256": hashlib.sha256(payload).hexdigest(), "comments": comments})
        all_comments.extend(comments)
        return matrices, units

    matrices, units = read(path, "A")
    if "A" not in matrices:
        raise ValueError("the drift file must contain an A matrix")
    if covariance_path is not None:
        if "C" in matrices:
            raise ValueError("C is present in both the combined file and covariance_path")
        extra, extra_units = read(covariance_path, "C")
        if set(extra) != {"C"}:
            raise ValueError("covariance_path must contain only a C matrix")
        matrices.update(extra)
        units.update(extra_units)
    headers = _header_metadata(all_comments, unit_convention)
    beta = None if beta is None else _positive(beta, "beta")
    beads = None if beads is None else _count(beads, "beads")
    for key, supplied in (("beta", beta), ("beads", beads)):
        if supplied is not None and key in headers and not math.isclose(supplied, headers[key], rel_tol=1e-10, abs_tol=0):
            raise ValueError(f"explicit {key} conflicts with file metadata")
    beta = headers.get("beta") if beta is None else beta
    beads = headers.get("beads") if beads is None else beads
    a = matrices["A"]
    if a.ndim == 3:
        if beads is not None and beads != len(a):
            raise ValueError("GLE normal-mode matrix count disagrees with beads metadata")
        beads = len(a)
    if mode == "piglet" and (beta is None or beads is None):
        raise ValueError("PIGLET import requires explicit beta and beads or matching file metadata")
    au, afactor = _choose_unit(units.get("A"), drift_unit,
                               lambda unit: _source_drift_factor(unit, unit_convention), "drift")
    a = a * afactor
    cfactor, cu = 1., "canonical identity"
    if "C" in matrices:
        def convert_c(unit):
            normalized = _unit_name(unit)
            if normalized in ("1", "dimensionless"):
                return 1.
            if beta is None or beads is None:
                raise ValueError("dimensional C requires beta and beads to normalize by P/beta")
            energy_factor = _source_energy_factor(unit, unit_convention)
            return energy_factor * beta / beads
        cu, cfactor = _choose_unit(units.get("C"), covariance_unit, convert_c, "covariance")
        c = matrices["C"] * cfactor
    else:
        if mode == "piglet":
            raise ValueError("PIGLET requires an explicit C covariance matrix")
        if covariance_unit is not None:
            raise ValueError("covariance_unit supplied but no C matrix is present")
        c = np.eye(a.shape[-1])
    if mode_order == "fourier":
        if a.ndim != 3:
            raise ValueError("fourier mode_order requires per-mode matrices")
        order = [0] + [j for k in range(1, (len(a) + 1) // 2) for j in (k, len(a) - k)]
        if len(a) % 2 == 0:
            order.append(len(a) // 2)
        a = a[order]
        if c.ndim == 3:
            if len(c) != len(a):
                raise ValueError("GLE covariance dimensions differ from A")
            c = c[order]
    result = {"A": a.tolist(), "C": c.tolist(),
              "provenance": "Imported gle4md-style raw matrices; source SHA-256 and original comments in parameter_metadata"}
    if beta is not None:
        result["beta"] = beta
    if beads is not None:
        result["beads"] = beads
    try:
        validation = validate_gle_parameters(result, mode=mode)
    except ValueError as exc:
        if (unit_convention == "physical" and "unit physical variance" in str(exc)
                and any("gle4md" in comment.lower() for comment in all_comments)):
            raise ValueError(str(exc) + "; for original gle4md exports using historical unit constants, "
                             "select unit_convention='gle4md' explicitly") from exc
        raise
    result["parameter_metadata"] = {
        "schema": _SCHEMA, "source": "gle4md-raw-import", "importer_version": 1,
        "source_files": sources, "source_metadata": headers, "mode": mode,
        "source_mode_order": mode_order, "mode_order": "real",
        "units": {"convention": unit_convention, "drift": au, "covariance": cu, "drift_to_au": afactor,
                  "covariance_to_dimensionless": cfactor, "target_covariance": "C / (P/beta)"},
        "frequency_range_au": [headers.get("omega_min_au"), headers.get("omega_max_au")],
        "objective": "external calibration; see preserved source comments",
        "residuals": None, "fit_converged": None,
        "validation": validation,
    }
    if unit_convention == "gle4md":
        result["parameter_metadata"]["units"]["source_converters"] = {
            "energy_to_kelvin": dict(_GLE4MD_KELVIN), "atomic_time_to_picoseconds": _GLE4MD_AUTIME_PS,
            "reference": "https://github.com/lab-cosmo/gle4md/blob/gh-pages/ingen/ingen.js"}
    return _json_copy(result)


def resolve_gle_parameters(spec, *, mode=None, base_directory=None):
    """Materialize new file/generator declarations; preserve legacy mappings.

    Existing A/C mappings without parameter_metadata retain their exact
    representation and validation path, so their run identities and restart
    compatibility are unchanged.  New declarations are resolved once before
    the propagation plan is serialized.
    """
    if not isinstance(spec, dict):
        raise ValueError("GLE requires a parameter mapping")
    mode = _mode("gle" if mode is None else mode)
    result = copy.deepcopy(spec)
    declarations = {key for key in ("generator", "file") if key in result}
    if len(declarations) > 1:
        raise ValueError("specify only one GLE file or generator")
    if "generator" in result:
        generator = result.pop("generator")
        if generator == "optimal-sampling":
            if mode == "piglet":
                raise ValueError("optimal-sampling is canonical; PIGLET requires the piglet generator")
            allowed = {"frequency_range", "frequency_unit", "naux", "grid_size", "max_nfev"}
            function = generate_optimal_gle
        elif generator == "piglet":
            if mode != "piglet":
                raise ValueError("the piglet generator requires thermostat mode='piglet'")
            allowed = {"frequency_range", "frequency_unit", "beta", "beads", "naux", "grid_size", "max_nfev", "tolerance"}
            function = generate_piglet
        else:
            raise ValueError("unknown GLE generator; use optimal-sampling or piglet")
        if set(result) - allowed:
            raise ValueError("unknown GLE generator parameters: " + ", ".join(sorted(map(str, set(result) - allowed))))
        required = {"frequency_range"} | ({"beta", "beads"} if generator == "piglet" else set())
        if required - set(result):
            raise ValueError("missing GLE generator parameters: " + ", ".join(sorted(required - set(result))))
        return function(**result)
    if "file" in result:
        allowed = {"file", "covariance_file", "drift_unit", "covariance_unit", "beta", "beads", "mode_order", "unit_convention"}
        if set(result) - allowed:
            raise ValueError("unknown GLE file parameters: " + ", ".join(sorted(map(str, set(result) - allowed))))
        root = Path.cwd() if base_directory is None else Path(base_directory)
        path = root / result.pop("file")
        if "covariance_file" in result:
            result["covariance_path"] = root / result.pop("covariance_file")
        return load_gle4md(path, mode=mode, **result)
    if "parameter_metadata" in result:
        metadata = result["parameter_metadata"]
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("parameter_metadata must be a nonempty JSON mapping")
        result["parameter_metadata"] = _json_copy(metadata)
        validate_gle_parameters(result, mode=mode)
    return result


__all__ = ["generate_optimal_gle", "generate_piglet", "load_gle4md",
           "resolve_gle_parameters", "validate_gle_parameters"]
