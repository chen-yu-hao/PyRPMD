---
orphan: true
---

# T40 在线激发态电子结构接口

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 1 |
| 前置 | 无 |
| 取代 | 原 T25 中的“TDDFT NAC”一项 |
| 建议分支 | `agent/t40-excited-state-interface` |

## 背景

`src/pyrpmd/adiabatic.py` 中的 `AdiabaticPythonPotential` 定义了在线电子数据契约：
绝热能量、正核梯度、非绝热耦合（NAC）和重叠矩阵，单位都是原子单位；回调可以拿到上一帧几何与上下文。
目前随附的提供者只有 `PySCFMultistatePotential`（SA-CASSCF，见 `potentials/pyscf_multistate.py`）。

与 SHARC 4 和 Newton-X 相比，缺少：

- TDDFT/TDA、ADC(2)、CASPT2/MS-CASPT2 等常用激发态方法；
- 外部程序接口（OpenMolcas、ORCA、Q-Chem、TeraChem、Molpro、Turbomole）；
- 基于波函数重叠的时间导数耦合、相位追踪与态追踪；
- 失败处理与 scratch 管理。

## 目标

建立通用的在线激发态提供者框架，至少交付一个可以在 CI 中测试的真实后端（PySCF TDDFT/TDA）
和一组基于文件接口的外部程序后端。所有后端都满足 `AdiabaticPythonPotential` 契约，
无需修改即可被现有非绝热方法使用。

## 范围

**要做：**

1. **框架。** 新建 `src/pyrpmd/potentials/excited_state/` 包，提供：
   - 后端基类；
   - 每个珠子、每个进程独立的 scratch 目录与清理策略；
   - 上一帧波函数和轨道的缓存，用于重叠计算与 SCF 初猜；
   - 程序版本、输入模板、基组和赝势文件哈希进入 run identity；
   - SCF 或激发态求解失败时映射为提供者状态码，不悄悄返回错误数值；
   - 每次计算的耗时与调用计数。
2. **耦合与相位。**
   - 后端提供解析 NAC 时直接使用；
   - 否则由相邻两帧的波函数重叠计算时间导数耦合（HST 或 NPI 方案，实现前核实出处）；
   - 统一的相位对齐和态追踪，在近简并时给出诊断；
   - 重叠矩阵的约定与 `adiabatic.py` 一致：O_ij = <old_i|new_j>。
3. **PySCF TDDFT/TDA 后端。** 提供激发能、激发态梯度（PySCF 支持时）和基于重叠的耦合；
   可选给出振子强度与跃迁偶极，供 T39 的激发选择使用。
4. **外部程序后端。** 为 ORCA、OpenMolcas、Q-Chem 实现基于文件接口的后端：输入生成器、输出解析器，
   并支持外部程序的重叠计算或 wfoverlap 类工具。用录制的输出文件做解析测试；
   没有实机运行过的后端在文档中标注为“未实测”。
5. **契约扩展。** 如需在 `AdiabaticResult` 中追加可选字段（例如跃迁偶极），只允许向后兼容的追加；
   不修改 `adiabatic.py` 第 254 行之后的动力学类。

**不做：**

- 不实现 QM/MM 嵌入（单独立卡）。
- 不修改非绝热动力学类（归 T26、T42a、T42b）。
- 不分发任何外部程序或其许可文件。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/potentials/excited_state/`、`potentials/pyscf_multistate.py`
  （只允许为共享框架所做的重构，保持行为不变）、相关测试和测试夹具、
  新建 `docs/excited-state-interface.md`（`orphan: true`）、`examples/excited_state/`。
- **受限修改：** `src/pyrpmd/adiabatic.py` 中的提供者契约部分（第 1–253 行），只允许向后兼容的追加。
- **共享注册点：** `potentials/__init__.py`、`pyproject.toml`（可选依赖）。

## 交付物

- 框架、PySCF TDDFT/TDA 后端、外部程序后端与解析测试、英文指南和示例。
- 验证记录 `docs/validation/excited-state-interface-<日期>/README.md`。

## 验收标准

- **PySCF TDDFT/TDA：**
  - 激发态梯度与能量的有限差分对照；
  - 基于重叠的时间导数耦合与解析 NAC（若 PySCF 提供）或有限差分重叠导数对照；
  - 沿一条小分子轨迹，耦合与重叠的相位连续，符号不会随机翻转。
- **已有方法兼容：** 现有 SA-CASSCF 提供者的结果不变；用 TDDFT 后端运行
  `AdiabaticRPSHDynamics` 的短轨迹，能量守恒（按跃迁规则）在报告的容差之内。
- **外部程序解析：** 每个后端都有录制输出的解析测试，覆盖正常、SCF 未收敛和态数不足三种情形。
- 后端失败时，非绝热工作流按照已有失败与未判定的统计规则处理，不写入半个步长。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pip install -e '.[pyscf]'
python -m pytest -q tests -k "excited_state or adiabatic or pyscf"
```
