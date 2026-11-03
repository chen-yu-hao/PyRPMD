---
orphan: true
---

# T42a 统一核心 II-a：映射类非绝热方法移植与 SU(N) 推广

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 2 |
| 前置 | T26 合并（辅助自由度算符协议与 NRPMD 试点） |
| 可并行 | T42b（两张卡的文件不重叠） |
| 建议分支 | `agent/t42a-mapping-methods` |

## 背景

T26 定义了辅助自由度协议，并把 NRPMD 迁移到统一传播核心上。以下映射类和连续电子变量的方法
仍在各自的 `step` 循环中，并且有态数限制：

- `SMNRPMDDynamics`（`spin_mapping.py:150`：只支持两个电子态）；
- `RPMASHDynamics`（`rp_mash.py:96`：只支持两个电子态）；
- `MASHDynamics`（`mash.py:38`：两态）与 `MultistateMASHDynamics`（P=1）；
- `ThermalNRPMDDynamics`（`thermal_nrpmd.py`）；
- `IsomorphicEhrenfestDynamics`（`isomorphic.py`：两态）；
- `QDNRPMDDynamics`（`adiabatic.py`）。

## 目标

把上述方法移植到 T26 的协议上，从而获得统一的组合能力（周期晶胞、RPC/MTS、偏置）、
统一的 checkpoint 和统一的输出。在已有发表公式的前提下，把 SM-NRPMD 推广到 N 个电子态。

## 范围

**要做：**

1. **逐个移植。** 每个方法都实现为一个辅助自由度，保留原类名作为适配层；
   NVE 下与旧实现逐位一致或误差 ≤ 1e-12。每移植一个方法，就更新 T26 的组合白名单表。
2. **SU(N) SM-NRPMD。** 依据 Bossion、Ying、Chowdhury、Huo（JCP 157, 084105, 2022，
   在 SU(N) 李群相空间中的非绝热 RPMD）以及 Runeson 与 Richardson 的广义自旋映射
   （JCP 152, 084110, 2020）实现 N 态推广。实现前核实公式与估计量；
   两态情形必须与现有 W-spin 实现一致。
3. **工作流。** `task="nonadiabatic"` 中对应的 `method` 改为调用统一核心；
   旧 checkpoint 可以读取并续算，新格式带版本号。

**不做：**

- 不实现环聚合物多态 MASH，也不实现复数 SOC 的 MASH（仍属研究方向，见原 T25）。
- 不修改跃迁类方法（归 T42b）。

## 文件所有权

- **独占：** `src/pyrpmd/spin_mapping.py`、`rp_mash.py`、`mash.py`、`thermal_nrpmd.py`、`isomorphic.py`、
  `adiabatic.py` 中的 `QDNRPMDDynamics` 部分、`workflows/nonadiabatic_extensions.py`、
  `workflows/nonadiabatic_multistate.py` 中的映射类分支、相关测试。
- **共享注册点：** `engine.py`/`auxiliary.py` 的白名单表（追加条目）、`nonadiabatic_input.py`。

## 交付物

- 移植后的方法、SU(N) SM-NRPMD、更新后的指南（`docs/nonadiabatic-spin-methods.md`、
  `docs/auxiliary-operators.md`）。
- 验证记录 `docs/validation/mapping-port-<日期>/README.md`。

## 验收标准

- 每个移植方法在 NVE 下与旧实现一致：逐位一致，或相对误差 ≤ 1e-12。
  `examples/nonadiabatic/extensions/` 的所有示例结果不变。
- SU(N) SM-NRPMD 在 N=2 时与现有实现一致；在 N=3 的模型上（例如三态 LVC 或 Tully 型模型），
  布居与独立的精确量子参考（`analysis/quantum_reference.py`）作比较，如实报告近似误差。
- 新开放的组合（例如周期晶胞中的 SM-NRPMD）有数值测试；白名单之外的组合会被拒绝。
- 满足通用规则 G5–G9。

## 验证命令

```bash
python -m pytest -q tests -k "nonadiabatic or spin or mash or isomorphic or thermal_nrpmd"
python -m pytest -q -m "not campaign"
```
