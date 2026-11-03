---
orphan: true
---

# T35 绝对量子自由能热力学积分

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t35-quantum-free-energy` |

## 背景

PyRPMD 有大量相对自由能工具（FES、MBAR、伞形积分），也有一个质量热力学积分估计函数
（`src/pyrpmd/quantum_statistics.py` 中的 `trapezoidal_mass_thermodynamic_integration`，
用于同位素分馏），但没有绝对自由能工作流。

量子固体与液体的相稳定性、熔点和同位素效应研究，通常组合以下几种方法：

- Einstein 晶体参考态与 Frenkel–Ladd λ 积分；
- 经典到量子的质量热力学积分（Ceriotti 与 Markland，JCP 138, 014112, 2013）；
- 两个 PES 之间的 λ 积分（例如从 ML 势到参考势）。

i-PI 社区通常要手工搭建这些步骤。

## 目标

新增 `task="free_energy"` 工作流。由多个独立的 λ 或质量窗口组成，可以并行执行、可以续算；
用 MBAR 或高斯求积组合各窗口的结果，给出带不确定度的绝对量子自由能。

## 范围

**要做：**

1. **Einstein 晶体参考。** 每个原子拴在晶格位点上的谐振势：
   - 有限 P 下的离散路径积分配分函数解析可得，P→∞ 的极限也解析可得；
   - 处理质心固定约束（Frenkel–Ladd 的质心修正）；
   - 弹簧常数可以由均方位移自动选择。
2. **λ 积分。** 在 V_λ = (1−λ)V_E + λV 上做 PIMD 采样，λ 取 Gauss–Legendre 节点，
   用 MBAR 组合；把每个 λ 窗口当作独立任务，复用已有的并行执行器。
3. **质量积分。** 从经典（或大质量）参考到物理质量，对动能估计量积分；
   复用并推广 `quantum_statistics.py` 中的估计函数，支持多个质量节点、自动选择积分变量
   （例如 1/√m）和收敛诊断。
4. **两个 PES 之间的 λ 积分。** 用于 ML 势与参考势之间的修正，复用 `potentials/contraction.py`
   中的差分或分裂势概念。
5. **不确定度与诊断。** 每个窗口的自相关和有效样本数、相邻窗口的重叠、
   求积误差估计（节点加倍对照），以及总误差分解。

**不做：**

- 不修改 `workflows/materials.py`（归 T16b）。每个窗口通过公开的 `Dynamics` 和传播栈 API
  或已有采样函数运行。
- 不实现相共存或界面法求熔点（可以另行立卡）。

## 文件所有权

- **独占：** 新建 `src/pyrpmd/free_energy_ti.py`、`workflows/free_energy.py`、
  新建 `potentials/einstein.py`、相关测试、新建 `docs/quantum-free-energy.md`（`orphan: true`）、
  `examples/free_energy/`。
- **共享注册点：** `input.py`、`input_tasks.py`、`input_schema.py`、`runner.py`（新任务 `free_energy`）、`cli.py`。
- **只读调用：** `quantum_statistics.py`、`analysis/mbar.py`。需要修改时写进“需要协调”。

## 交付物

- 工作流、Einstein 参考势、英文指南和示例输入。
- 验证记录 `docs/validation/quantum-free-energy-<日期>/README.md`。

## 验收标准

- **谐振晶体：** 把真实 PES 设为另一组弹簧常数的谐振势；在有限 P 下，自由能差与解析离散路径积分
  结果一致（在统计误差之内），P→∞ 外推与解析量子结果一致。
- **一维非谐势（Morse 或四次势）：** 绝对量子自由能与网格对角化得到的精确值对照，
  误差在统计误差与有限 P 误差之内，并给出 P 收敛表。
- **经典极限：** 质量积分在大质量端与经典自由能一致。
- 中断并续算的工作流结果与不中断的结果一致；已完成的窗口不会重复运行。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "free_energy or einstein or quantum_statistics or mbar"
python -m pyrpmd run examples/free_energy/harmonic_crystal.inp --output runs/t35
```
