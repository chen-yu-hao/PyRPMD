"""Audit and summarize all four prespecified H3 PMF study cases.

Missing, duplicate, smoke, failed, or mismatched cases remain explicit report
failures. This report never declares physical convergence. It does not run MD
or refit MBAR; all plotted estimates and uncertainties come from saved data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARTREE_KCAL_MOL = 627.5094740631
SCHEMA = "pyrpmd.h3.pmf-convergence-report.v1"
SOLVER_SCHEMA = "pyrpmd.h3.mbar-solver-comparison.v1"


def _plan_module():
    spec = importlib.util.spec_from_file_location(
        "h3_convergence_plan", HERE / "convergence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _number(value):
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
        else None
    )


def _scaled(value, factor=1.0):
    value = _number(value)
    return None if value is None else value * factor


def _interval(value, factor=1.0):
    if not isinstance(value, list) or len(value) != 2:
        return None
    values = [_scaled(x, factor) for x in value]
    return (
        values
        if all(x is not None for x in values) and values[0] <= values[1]
        else None
    )


def _metrics(data):
    pmf = data.get("pmf") or {}
    ui = pmf.get("ui_comparison") or {}
    rec = data.get("recrossing") or {}
    diag = data.get("diagnostics") or {}
    paired = ui.get("paired_difference_statistics") or {}
    return {
        "mbar_barrier_kcal_mol": _scaled(
            pmf.get("free_energy_barrier_hartree"), HARTREE_KCAL_MOL
        ),
        "mbar_barrier_se_kcal_mol": _scaled(
            pmf.get("barrier_standard_error_hartree"), HARTREE_KCAL_MOL
        ),
        "ui_barrier_kcal_mol": _scaled(
            ui.get("free_energy_barrier_hartree"), HARTREE_KCAL_MOL
        ),
        "ui_barrier_se_kcal_mol": _scaled(
            (ui.get("ui_barrier_statistics") or {}).get("standard_error_hartree"),
            HARTREE_KCAL_MOL,
        ),
        "mbar_minus_ui_kcal_mol": _scaled(
            ui.get("mbar_minus_ui_hartree"), HARTREE_KCAL_MOL
        ),
        "paired_difference_se_kcal_mol": _scaled(
            paired.get("standard_error_hartree"), HARTREE_KCAL_MOL
        ),
        "paired_difference_95_ci_kcal_mol": _interval(
            paired.get("percentile_95_ci_hartree"), HARTREE_KCAL_MOL
        ),
        "kappa": _number(rec.get("kappa")),
        "kappa_se": _number(rec.get("kappa_standard_error")),
        "rate_cm3_molecule_s": _number(data.get("rate_estimate_cm3_molecule_s")),
        "rate_full_se_cm3_molecule_s": _number(
            data.get("rate_standard_error_cm3_molecule_s")
        ),
        "rate_full_95_ci_cm3_molecule_s": _interval(
            data.get("rate_percentile_95_ci_cm3_molecule_s")
        ),
        "rate_conditional_se_cm3_molecule_s": _number(
            data.get("conditional_rate_standard_error_cm3_molecule_s")
        ),
        "effective_samples_minimum": _number(diag.get("effective_samples_minimum")),
        "effective_samples_median": _number(diag.get("effective_samples_median")),
        "effective_samples_sum": _number(diag.get("effective_samples_sum")),
        "maximum_endpoint_energy_error_kbt": _number(
            diag.get("maximum_endpoint_energy_error_kbt")
        ),
        "bootstrap_valid": (pmf.get("bootstrap") or {}).get("valid"),
        "paired_bootstrap_valid": paired.get("valid"),
        "samples": diag.get("samples"),
        "streams": diag.get("streams"),
    }


def aggregate(directory):
    """Return a complete audit, including failures, without writing files."""
    plan = _plan_module()
    directory = Path(directory).resolve()
    expected_cases = list(plan.CASES)
    errors, found = [], {case: [] for case in expected_cases}

    def error(case, code, detail):
        item = {"case": case, "code": code, "detail": str(detail)}
        errors.append(item)
        return item

    for path in sorted(directory.rglob("benchmark.json")):
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("benchmark must be an object")
        except (OSError, ValueError) as exc:
            error(None, "unreadable_benchmark", f"{path}: {exc}")
            continue
        case = data.get("case")
        if data.get("schema") != plan.SCHEMA or case not in found:
            error(
                case, "unexpected_benchmark", f"{path}: schema={data.get('schema')!r}"
            )
            continue
        found[case].append((path, data))
    cases = {}
    for case in expected_cases:
        before = len(errors)
        paths = found[case]
        item = {
            "present": bool(paths),
            "eligible": False,
            "errors": [],
            "metrics": _metrics({}),
        }
        cases[case] = item
        if len(paths) != 1:
            error(
                case,
                "missing_case" if not paths else "duplicate_case",
                [str(p) for p, _ in paths],
            )
            item["errors"] = errors[before:]
            continue
        path, data = paths[0]
        settings, budget = data.get("settings") or {}, data.get("budget") or {}
        expected = plan.case_settings(case, smoke=False)
        expected_budget = plan.compute_budget(expected)
        item.update(
            source=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            seed=data.get("seed"),
            settings=settings,
            budget=budget,
            metrics=_metrics(data),
            production_gates=data.get("production_gates"),
            rate_uncertainty=data.get("rate_uncertainty"),
            fingerprints=data.get("fingerprints"),
            warnings=data.get("warnings", []),
            execution_error=data.get("error"),
        )
        if data.get("smoke") is not False or settings.get("smoke") is not False:
            error(
                case,
                "smoke_ineligible",
                "Only explicit smoke=false production cases are eligible",
            )
        if data.get("passed") is not True:
            error(case, "benchmark_failed", data.get("error", "passed is not true"))
        if data.get("converged") is not False:
            error(case, "invalid_convergence_claim", "converged must remain false")
        for key, value in expected.items():
            if settings.get(key) != value:
                error(
                    case,
                    "settings_mismatch",
                    f"{key}: expected {value!r}, received {settings.get(key)!r}",
                )
        for key, value in expected_budget.items():
            if budget.get(key) != value:
                error(
                    case,
                    "budget_mismatch",
                    f"{key}: expected {value!r}, received {budget.get(key)!r}",
                )
        if data.get("seed") != expected["seed"]:
            error(
                case,
                "seed_mismatch",
                f"expected {expected['seed']}, received {data.get('seed')!r}",
            )
        controls = {
            k: v
            for k, v in expected.items()
            if k not in ("case", "seed", "workers", "cpu_count")
        }
        fingerprints = data.get("fingerprints") or {}
        if fingerprints.get("comparison_controls") != _digest(controls):
            error(
                case,
                "comparison_controls_mismatch",
                "Fingerprint differs from the prespecified scientific controls",
            )
        item["comparison_controls"] = fingerprints.get("comparison_controls")
        gates = data.get("gates") or {}
        if (
            not gates
            or any(value is not True for value in gates.values())
            or gates.get("sources_unchanged_during_run") is not True
        ):
            error(
                case,
                "execution_gates_failed",
                {k: v for k, v in gates.items() if v is not True},
            )
        diag, pmf, rec = (
            data.get("diagnostics") or {},
            data.get("pmf") or {},
            data.get("recrossing") or {},
        )
        try:
            recomputed = plan.quality_gates(expected, {"pmf": pmf, **rec}, diag)
        except (KeyError, TypeError, ValueError) as exc:
            recomputed = {}
            error(case, "unreadable_production_gates", exc)
        item["recomputed_production_gates"] = recomputed
        recorded = data.get("production_gates") or {}
        for key, value in recomputed.items():
            if value is not True or recorded.get(key) is not True:
                error(
                    case,
                    "production_gate_failed",
                    f"{key}: recorded={recorded.get(key)!r}, recomputed={value!r}",
                )
        if not recomputed or set(recorded) != set(recomputed):
            error(
                case,
                "production_gate_schema",
                "Missing or unexpected production gate fields",
            )
        for key in ("samples", "streams"):
            wanted = expected_budget["saved_samples" if key == "samples" else key]
            if diag.get(key) != wanted:
                error(
                    case,
                    "actual_work_mismatch",
                    f"{key}: expected {wanted}, received {diag.get(key)!r}",
                )
        if (
            pmf.get("sampling_measure") != "rate_flux"
            or pmf.get("reference") != expected["reference_xi"]
        ):
            error(
                case,
                "measure_or_reference_mismatch",
                "Requires rate_flux and the prespecified reference",
            )
        for key in (
            "mbar_barrier_kcal_mol",
            "mbar_barrier_se_kcal_mol",
            "ui_barrier_kcal_mol",
            "ui_barrier_se_kcal_mol",
            "mbar_minus_ui_kcal_mol",
            "paired_difference_se_kcal_mol",
            "paired_difference_95_ci_kcal_mol",
            "kappa",
            "kappa_se",
            "rate_cm3_molecule_s",
            "rate_full_se_cm3_molecule_s",
            "rate_full_95_ci_cm3_molecule_s",
        ):
            if item["metrics"].get(key) is None:
                error(case, "missing_estimate_or_uncertainty", key)
        uncertainty = data.get("rate_uncertainty") or {}
        if (
            uncertainty.get("available") is not True
            or uncertainty.get("pmf_bootstrap_valid") != expected["bootstrap"]
            or (_number(uncertainty.get("recrossing_bootstrap_valid")) or 0) < 2
        ):
            error(case, "full_rate_uncertainty_unavailable", uncertainty)
        boot = pmf.get("bootstrap") or {}
        records = boot.get("draws") or []
        if sorted(record.get("draw", -1) for record in records) != list(
            range(expected["bootstrap"])
        ) or any(record.get("valid") is not True for record in records):
            error(
                case,
                "bootstrap_ledger_invalid",
                "Require exactly one valid record per prespecified draw",
            )
        ui = pmf.get("ui_comparison") or {}
        if (
            ui.get("xi_star") != expected["xi_star"]
            or ui.get("reference") != expected["reference_xi"]
        ):
            error(
                case,
                "ui_surface_mismatch",
                "UI must use exactly the same fixed surface and reference",
            )
        try:
            dif = np.asarray(ui["paired_difference_bootstrap_hartree"], float)
            mb = np.asarray(ui["paired_mbar_bootstrap_barriers"], float)
            ub = np.asarray(ui["paired_ui_bootstrap_barriers"], float)
            stats = ui["paired_difference_statistics"]
            if (
                dif.shape != (expected["bootstrap"],)
                or mb.shape != dif.shape
                or ub.shape != dif.shape
                or not np.isfinite(dif).all()
                or not np.allclose(dif, mb - ub, rtol=1e-10, atol=1e-12)
            ):
                raise ValueError("paired differences or array lengths disagree")
            if not np.isclose(
                stats["standard_error_hartree"], dif.std(ddof=1), rtol=1e-8, atol=1e-12
            ) or not np.allclose(
                stats["percentile_95_ci_hartree"],
                np.quantile(dif, [0.025, 0.975]),
                rtol=1e-8,
                atol=1e-12,
            ):
                raise ValueError("paired CI/SE disagrees with the retained draws")
        except (KeyError, TypeError, ValueError) as exc:
            error(case, "paired_estimator_audit_failed", exc)
        sensitivity = data.get("bin_sensitivity") or []
        item["bin_sensitivity"] = {
            str(expected["bins"]): item["metrics"].get("mbar_barrier_kcal_mol")
        }
        if sorted(entry.get("bins", -1) for entry in sensitivity) != sorted(
            expected["sensitivity_bins"]
        ):
            error(
                case,
                "bin_sensitivity_schema",
                "Require exactly the prespecified 85/340-bin checks",
            )
        for entry in sensitivity:
            number = entry.get("bins")
            value = _scaled(entry.get("free_energy_barrier_hartree"), HARTREE_KCAL_MOL)
            item["bin_sensitivity"][str(number)] = value
            if (
                entry.get("ready") is not True
                or value is None
                or (entry.get("bootstrap") or {}).get("requested") != 0
                or entry.get("uncertainty_available") is not False
            ):
                error(
                    case,
                    "bin_sensitivity_failed",
                    f"bins={number}: missing support/value or unexpected bootstrap",
                )
        solver_path = path.parent / "solver-check.json"
        try:
            solver = json.loads(solver_path.read_text())
            if not isinstance(solver, dict):
                raise ValueError("solver-check must be an object")
            item["solver_check"] = solver
            sg = solver.get("gates") or {}
            required = [
                "auto_residual_below_1e_9",
                "legacy_residual_below_1e_9",
                "target_weights_match_legacy",
                "gauge_free_energies_match_legacy",
                "overlap_components_match_legacy",
                "target_weights_match_independent_pymbar",
                "gauge_free_energies_match_independent_pymbar",
            ]
            if (
                solver.get("schema") != SOLVER_SCHEMA
                or solver.get("passed") is not True
                or any(sg.get(key) is not True for key in required)
                or any(v is not True for v in sg.values())
            ):
                error(
                    case, "solver_check_failed", {key: sg.get(key) for key in required}
                )
            independent = solver.get("independent_pymbar") or {}
            if (
                not independent.get("version")
                or independent.get("asymptotic_uncertainty_used") is not False
            ):
                error(
                    case,
                    "independent_pymbar_missing",
                    "Official PyMBAR comparison is required; its asymptotic uncertainty is not used",
                )
            for key in (
                "target_weight_l1_difference",
                "maximum_gauge_free_energy_difference",
            ):
                value = _number(independent.get(key))
                if value is None or not 0 <= value < 1e-6:
                    error(case, "independent_pymbar_disagrees", key)
            if not fingerprints.get("final_hdf5_sha256") or solver.get(
                "source_hdf5_sha256"
            ) != fingerprints.get("final_hdf5_sha256"):
                error(
                    case,
                    "solver_source_mismatch",
                    "Solver comparison must refer to this case's final HDF5",
                )
            if solver.get("streams") != expected_budget["streams"]:
                error(case, "solver_stream_mismatch", solver.get("streams"))
            stride = solver.get("stride")
            if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
                error(case, "invalid_solver_stride", stride)
            else:
                per_stream = round(
                    expected["production_fs"] / expected["sample_interval_fs"]
                )
                wanted = expected_budget["streams"] * (
                    (per_stream + stride - 1) // stride
                )
                if solver.get("samples") != wanted:
                    error(
                        case,
                        "solver_sample_mismatch",
                        f"expected {wanted}, received {solver.get('samples')!r}",
                    )
        except (OSError, ValueError, TypeError) as exc:
            error(case, "solver_check_unavailable", f"{solver_path}: {exc}")
        item["errors"] = errors[before:]
        item["eligible"] = not item["errors"]
    if all(cases[case].get("seed") is not None for case in expected_cases):
        if len({cases[case]["seed"] for case in expected_cases}) != 4:
            error(
                None, "nonindependent_seeds", "All four global seeds must be distinct"
            )
    comparisons = {}
    quantities = {
        "mbar_barrier_kcal_mol": "mbar_barrier_se_kcal_mol",
        "ui_barrier_kcal_mol": "ui_barrier_se_kcal_mol",
        "mbar_minus_ui_kcal_mol": "paired_difference_se_kcal_mol",
        "kappa": "kappa_se",
        "rate_cm3_molecule_s": "rate_full_se_cm3_molecule_s",
    }
    for group in ("k400", "k800"):
        a, b = cases[group + "-a"], cases[group + "-b"]
        matching = bool(a.get("comparison_controls")) and a.get(
            "comparison_controls"
        ) == b.get("comparison_controls")
        if a["present"] and b["present"] and not matching:
            error(
                group,
                "replicate_controls_mismatch",
                "Independent seeds must share the same scientific controls within each group",
            )
        item = {
            "available": bool(a["eligible"] and b["eligible"] and matching),
            "difference_order": "b minus a",
            "descriptive_only": True,
            "independent_runs": 2,
            "quantities": {},
        }
        if item["available"]:
            for value_key, error_key in quantities.items():
                av, bv = a["metrics"].get(value_key), b["metrics"].get(value_key)
                ae, be = a["metrics"].get(error_key), b["metrics"].get(error_key)
                difference = bv - av if av is not None and bv is not None else None
                se = (
                    float(np.hypot(ae, be))
                    if ae is not None and be is not None
                    else None
                )
                item["quantities"][value_key] = {
                    "difference": difference,
                    "combined_standard_error": se,
                    "difference_over_combined_se": (
                        difference / se
                        if difference is not None and se is not None and se > 0
                        else None
                    ),
                }
        comparisons[group] = item
    return {
        "schema": SCHEMA,
        "input_directory": str(directory),
        "passed": not errors,
        "converged": False,
        "expected_cases": expected_cases,
        "cases": cases,
        "errors": errors,
        "seed_comparisons": comparisons,
        "fixed_xi_star": 1.00925,
        "reference_xi": 0.0,
        "background_only": {
            "previous_mbar_minus_ui_kcal_mol": 3.00533727,
            "note": "Previous weaker/coarser umbrella run; not matched sampling and not an independent statistical comparison.",
        },
        "limitations": [
            "Two independent global seeds per group provide descriptive repeatability only, not established statistical convergence.",
            "K, center spacing, and the entrance grid edge change together; differences cannot be attributed to K alone.",
            "UI and MBAR reuse the same samples; paired bootstrap differences are not independent trajectories.",
            "Rate SE combines PMF and constrained-parent recrossing bootstrap with the surface fixed; it excludes timestep, bead, Rinf, and equilibration systematic errors.",
            "85/170/340-bin checks reuse the same trajectories; secondary bin checks have no additional uncertainty estimate.",
            "Independent PyMBAR verifies finite-data nonlinear equations only; striding is not claimed to decorrelate the data.",
        ],
    }


def _format(value, precision=5):
    return "缺失" if value is None else f"{value:.{precision}g}"


def _plusminus(value, error):
    return _format(value) + " ± " + _format(error)


def markdown(report):
    status = (
        "所有正式输入与验收项目通过"
        if report["passed"]
        else "未通过：存在缺失、失败或不匹配输入"
    )
    lines = [
        "# H3 / BKMP2：PMF 估计方法与独立重复研究",
        "",
        f"**{status}；physical convergence = false。**",
        "",
        "共同分界面 ξ*=1.00925，参考 ξ=0；1500 K、16 珠子、dt=0.1 fs、R∞=10 bohr。",
        "每条时间流热化300 fs、生产1 ps；每窗两条replica，两个不同全局seed分别重算parent和child。",
        "βK=400采用47窗/间隔0.025；βK=800采用59窗/间隔0.02。K、间隔及入口网格边缘共同改变，不能声称孤立的K效应。",
        "",
        "## 势垒与配对方法差值",
        "",
        "势垒单位kcal/mol，±为1 SE；差值区间为同draw配对bootstrap的95%百分位区间。",
        "",
        "| Case | 验收 | Seed | MBAR ΔW ± SE | Gaussian UI ΔW ± SE | MBAR−UI | 配对95% CI |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for case in report["expected_cases"]:
        item = report["cases"][case]
        m = item["metrics"]
        ci = m.get("paired_difference_95_ci_kcal_mol")
        lines.append(
            f"| {case} | {'通过' if item['eligible'] else '拒绝/缺失'} | {item.get('seed', '缺失')} | {_plusminus(m.get('mbar_barrier_kcal_mol'),m.get('mbar_barrier_se_kcal_mol'))} | {_plusminus(m.get('ui_barrier_kcal_mol'),m.get('ui_barrier_se_kcal_mol'))} | {_format(m.get('mbar_minus_ui_kcal_mol'))} | {'['+', '.join(_format(v) for v in ci)+']' if ci else '缺失'} |"
        )
    lines += [
        "",
        "## Recrossing、速率与采样证据",
        "",
        "速率单位cm³ molecule⁻¹ s⁻¹；完整SE同时计入PMF和recrossing抽样误差，分界面保持固定。缺失误差不补零。",
        "",
        "| Case | κ ± SE | Rate ± 完整SE | ESS最小/中位 | PMF/配对有效draw | 最大端点能量误差/kBT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in report["expected_cases"]:
        m = report["cases"][case]["metrics"]
        lines.append(
            f"| {case} | {_plusminus(m.get('kappa'),m.get('kappa_se'))} | {_plusminus(m.get('rate_cm3_molecule_s'),m.get('rate_full_se_cm3_molecule_s'))} | {_format(m.get('effective_samples_minimum'))}/{_format(m.get('effective_samples_median'))} | {m.get('bootstrap_valid','缺失')}/{m.get('paired_bootstrap_valid','缺失')}（各需40） | {_format(m.get('maximum_endpoint_energy_error_kbt'))}（需<0.1） |"
        )
    lines += [
        "",
        "## Bin宽敏感性",
        "",
        "同一数据及同一分界面；85/340 bins不新增bootstrap，不作为独立重复。",
        "",
        "| Case | 85 bins | 170 bins（主结果） | 340 bins |",
        "|---|---:|---:|---:|",
    ]
    for case in report["expected_cases"]:
        values = report["cases"][case].get("bin_sensitivity", {})
        lines.append(
            f"| {case} | {_format(values.get('85'))} | {_format(values.get('170'))} | {_format(values.get('340'))} |"
        )
    lines += [
        "",
        "## 不同seed的描述性比较",
        "",
        "差值为b−a；合并SE=√(SEₐ²+SEᵦ²)，假设两个seed的估计独立。每组仅两个重复，不作收敛检验或总体方差估计。",
        "",
        "| 组 | 数量 | b−a | 合并SE | 差值/合并SE |",
        "|---|---|---:|---:|---:|",
    ]
    for group, item in report["seed_comparisons"].items():
        if not item["available"]:
            lines.append(f"| {group} | 输入未全部验收，不作比较 | 缺失 | 缺失 | 缺失 |")
        for key, value in item["quantities"].items():
            lines.append(
                f"| {group} | {key} | {_format(value['difference'])} | {_format(value['combined_standard_error'])} | {_format(value['difference_over_combined_se'])} |"
            )
    lines += ["", "## 验收与独立求解器对照", ""]
    if report["errors"]:
        for issue in report["errors"]:
            detail = issue["detail"].replace("\n", " ")
            lines.append(f"- **{issue['case'] or '全局'} / {issue['code']}**：{detail}")
    else:
        lines.append(
            "四个case各自的预定设置、预算、完整production gates、不同seed、同组controls和官方PyMBAR对照均通过；源HDF5校验值匹配。"
        )
    lines += [
        "",
        "| Case | PyMBAR版本 | Target weights L1差 | Gauge自由能最大差 |",
        "|---|---|---:|---:|",
    ]
    for case in report["expected_cases"]:
        independent = (report["cases"][case].get("solver_check") or {}).get(
            "independent_pymbar"
        ) or {}
        lines.append(
            f"| {case} | {independent.get('version','缺失')} | {_format(_number(independent.get('target_weight_l1_difference')))} | {_format(_number(independent.get('maximum_gauge_free_energy_difference')))} |"
        )
    lines += [
        "",
        "![四case比较](figures/pmf-comparison.png)",
        "",
        "## 解释边界",
        "",
        "上一轮MBAR−UI约3.00533727 kcal/mol仅作背景，其umbrella与当前方案不同，不混为匹配采样或独立统计检验。",
        "本研究没有改变温度、珠子数、时间步长或R∞，也没有验证这些参数的收敛；通过全部门槛仍保持converged=false。",
        "PyMBAR只验证同一有限数据上的方程解，没有使用其渐近误差，也不把stride抽样声称为去相关。",
        "",
    ]
    return "\n".join(lines)


def plot_report(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
        fig.subplots_adjust(top=0.84, bottom=0.14, hspace=0.37, wspace=0.30)
        for index, case in enumerate(report["expected_cases"]):
            item = report["cases"][case]
            m = item["metrics"]
            if not item["eligible"]:
                for ax in axes.flat:
                    ax.text(
                        index,
                        0.04,
                        "missing / rejected",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        color="0.5",
                        transform=ax.get_xaxis_transform(),
                        fontsize=8,
                    )
                continue
            for offset, key, color, label in [
                (-0.09, "mbar", "#1364a5", "MBAR"),
                (0.09, "ui", "#b95609", "Gaussian UI"),
            ]:
                value, error = m.get(key + "_barrier_kcal_mol"), m.get(
                    key + "_barrier_se_kcal_mol"
                )
                if value is not None:
                    label_once = (
                        label
                        if label not in axes[0, 0].get_legend_handles_labels()[1]
                        else None
                    )
                    axes[0, 0].errorbar(
                        index + offset,
                        value,
                        yerr=error,
                        fmt="o",
                        capsize=3,
                        color=color,
                        label=label_once,
                    )
            difference, ci = m.get("mbar_minus_ui_kcal_mol"), m.get(
                "paired_difference_95_ci_kcal_mol"
            )
            if difference is not None:
                axes[0, 1].plot(index, difference, "o", color="#733e91")
            if ci is not None:
                axes[0, 1].vlines(index, *ci, color="#733e91")
                axes[0, 1].hlines(ci, index - 0.055, index + 0.055, color="#733e91")
            for ax, key, error_key, scale in [
                (axes[1, 0], "kappa", "kappa_se", 1.0),
                (
                    axes[1, 1],
                    "rate_cm3_molecule_s",
                    "rate_full_se_cm3_molecule_s",
                    1e11,
                ),
            ]:
                value, error = m.get(key), m.get(error_key)
                if value is not None:
                    ax.errorbar(
                        index,
                        value * scale,
                        yerr=error * scale if error is not None else None,
                        fmt="o",
                        capsize=3,
                        color="#167a69",
                    )
        axes[0, 0].set_ylabel("Barrier relative to xi=0 (kcal/mol)")
        if axes[0, 0].get_legend_handles_labels()[0]:
            axes[0, 0].legend(frameon=False)
        axes[0, 1].set_ylabel("MBAR - UI (kcal/mol), paired 95% CI")
        axes[0, 1].axhline(0, color="0.6", lw=0.8, ls=":")
        axes[1, 0].set_ylabel("Transmission coefficient kappa (1 SE)")
        axes[1, 1].set_ylabel(
            "Rate (10$^{-11}$ cm$^3$ molecule$^{-1}$ s$^{-1}$), full SE"
        )
        for ax in axes.flat:
            ax.set_xticks(range(4), report["expected_cases"])
            ax.set_xlim(-0.5, 3.5)
            ax.grid(axis="y", alpha=0.18)
        validation = (
            "Input and production checks passed"
            if report["passed"]
            else "INCOMPLETE / REJECTED INPUT SET"
        )
        fig.suptitle(
            "H + H$_2$ / BKMP2: fixed xi = 1.00925\n"
            + validation
            + "; physical convergence not established",
            fontsize=13,
            y=0.96,
        )
        fig.text(
            0.5,
            0.045,
            "1500 K, 16 beads, 1 ps production per stream. K and center spacing change together.\nTwo independent seeds per group; bars are 1 SE except the paired 95% interval.",
            ha="center",
            fontsize=9,
        )
        paths = []
        for suffix in ("png", "pdf"):
            path = directory / f"pmf-comparison.{suffix}"
            fig.savefig(path, dpi=180)
            paths.append(str(path))
        plt.close(fig)
        return paths


def write_report(directory, output):
    report = aggregate(directory)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["figures"] = plot_report(report, output.parent / "figures")
    output.write_text(markdown(report), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = write_report(args.directory, args.output)
    print(
        "PYRPMD_H3_CONVERGENCE_REPORT "
        + json.dumps(
            {
                "passed": report["passed"],
                "converged": False,
                "errors": len(report["errors"]),
                "output": str(args.output),
                "fixed_xi_star": report["fixed_xi_star"],
                "cases": {
                    name: {"eligible": item["eligible"], **item["metrics"]}
                    for name, item in report["cases"].items()
                },
            },
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
