---
orphan: true
---

# T23（扩充）路径采样工作流

| 字段 | 内容 |
| --- | --- |
| 规模 | L |
| 波次 | 1 |
| 前置 | 无（T10 已合并） |
| 建议分支 | `agent/t23-path-sampling` |

原路线图中的 T23 只有一行：“∞RETIS 与路径采样 TOML 工作流”。本卡把它扩充为完整的工作流，
并加入反应坐标优化。

## 背景

以下 Python API 已经可用（见 `docs/transition-interface-sampling.md`、`docs/committor.md`）：

- `ReplicaExchangeTIS`，含 [0⁻] minus 系综，以及 `estimate_retis_flux`（`retis.py`）；
- `PPTISSampler`、`ReplicaExchangePPTIS`（`pptis.py`）；
- `ForwardFluxSampler`（`ffs.py`）；
- 两路射击的 TIS 系综（`transition_interface.py`）；
- 固定环的提交子（committor）分析（`committor.py`）。

但这些都没有 TOML 或 CLI 工作流，也没有持久的采样器状态。实现评审
（`docs/implementation-review-2026-11-03.md`）建议：先选一种方法做成持久的公开工作流，
再做并行调度。

## 目标

新增 `task="path_sampling"` 工作流，覆盖 TIS、RETIS、REPPTIS、FFS，带持久状态、续算和
带不确定度的速率；在此基础上实现 ∞RETIS 式的异步并行调度，并实现 aimless shooting 加
似然最大化，用于反应坐标优化。

## 范围

**要做：**

1. **第一阶段：持久工作流。**
   - 严格的输入校验，包括界面序列、序参量（复用 `[[cvs]]`）、射击参数和系综定义；
   - 持久保存路径系综（完整路径或可重放的种子与端点）、RNG、界面链状态和接受统计；
   - `run`、`analyze`、`resume` 命令；
   - 交叉概率、通量与速率，以及考虑路径相关性的不确定度。
   从 RETIS 开始，然后加入 FFS 和 REPPTIS。
2. **第二阶段：∞RETIS 式调度。** 多个 worker 异步执行路径移动，用无限交换（infinite swapping）
   的权重组合系综（以 Zhang、Lervik、van Erp 等关于 infretis 的论文为准，实现前核实出处）；
   在 worker 数变化时仍可续算。
3. **第三阶段：反应坐标优化。** aimless shooting（Peters 与 Trout，JCP 125, 054108, 2006）
   与似然最大化（Peters、Beckham、Trout，JCP 127, 034109, 2007）：在候选 CV 的线性组合中
   优化提交子模型，并给出 BIC 等模型选择准则；可以把结果反馈为 Bennett–Chandler 速率的分隔面。
4. **环聚合物约定。** 序参量默认使用质心；射击时的动量重采样遵循环聚合物的正确分布；
   文档写明这一约定与 RPMD 速率理论的关系。

**不做：**

- 不改变现有 Python API 的行为（工作流在其上封装；如需修改，必须保持兼容）。
- 不修改 Bennett–Chandler 速率工作流本身。

## 文件所有权

- **独占：** `src/pyrpmd/retis.py`、`pptis.py`、`ffs.py`、`transition_interface.py`、`committor.py`、
  新建 `workflows/path_sampling.py`、新建 `aimless_shooting.py`、`examples/path_sampling/`、相关测试、
  `docs/transition-interface-sampling.md`。
- **共享注册点：** `input.py`、`input_tasks.py`、`input_schema.py`、`runner.py`、`cli.py`。

## 交付物

- 三个阶段的实现与英文指南。三个阶段可以分开提 PR，但必须按顺序合并。
- 验证记录 `docs/validation/path-sampling-<日期>/README.md`。

## 验收标准

- **一维和二维解析双势阱（经典，P=1）：** TIS、RETIS、FFS 的速率与数值精确速率一致，
  在统计误差之内；精确速率可以用长时间直接模拟的首达时间，或在过阻尼极限下的 Kramers/数值积分得到。
- **P>1 的谐振加势垒模型：** RETIS 速率与同一 PES 上的 RPMD Bennett–Chandler 速率
  在统计误差内一致。
- **续算：** 中断后续算与不中断一致（CPU 确定性路径逐位一致）；∞RETIS 在 worker 数改变后
  的续算结果在统计上等价。
- **似然最大化：** 在已知真实反应坐标的二维模型上，优化得到的线性组合方向与真实方向的夹角
  在给定样本数下收敛，并给出收敛曲线。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "retis or pptis or ffs or transition_interface or committor or path_sampling"
python -m pyrpmd run examples/path_sampling/double_well_retis.inp --output runs/t23
python -m pyrpmd analyze runs/t23
```
