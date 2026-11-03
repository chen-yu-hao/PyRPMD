---
orphan: true
---

# T37 并行温度与哈密顿副本交换工作流

| 字段 | 内容 |
| --- | --- |
| 规模 | M |
| 波次 | 1 |
| 前置 | 无 |
| 建议分支 | `agent/t37-tempering` |

## 背景

`src/pyrpmd/replica_exchange.py` 中的 `ReplicaExchange` 已经正确实现了环聚合物的温度与哈密顿交换：
接受概率包含与温度相关的弹簧作用量，交换后对动量做重标度。但它是串行的，只能通过 Python API
使用，没有独立的 CLI 或续算格式（见 `docs/tier2-methods.md`）。

已有的 `[replica_exchange]` 表只用于固定窗口的伞形副本交换（REUS），要求 `task="fes"`。

## 目标

提供可以从 TOML 运行、MPI 并行、可以续算的温度副本交换（PT）与哈密顿副本交换（HREMD）工作流，
用于 PIMD 平衡采样和 FES 采样；并提供跨温度的重加权分析。

## 范围

**要做：**

1. **输入设计。** 新增独立的 `[tempering]` 表，不改变已有 `[replica_exchange]`（REUS）的语义。
   该表支持：
   - 温度阶梯（几何或显式列表）；
   - 哈密顿阶梯（λ 缩放的 PES、偏置或 `SplitPotential`）；
   - 交换间隔与配对方案（相邻、奇偶交替）。
   可以与 `task="pimd"` 和 `task="fes"`（umbrella、metadynamics）组合；组合白名单写在指南中。
2. **执行。** 每个副本对应一个 MPI 组（复用 `src/pyrpmd/parallel/` 的 G×B 划分），
   交换只传递能量与标量；在没有 MPI 时串行执行，结果逐位一致。
3. **持久化。** 原子的系综 checkpoint 包含：所有副本的状态和 RNG、交换 RNG、副本与温度的置换、
   接受统计。部分副本缺失时拒绝续算。
4. **分析。**
   - 相邻副本的接受率、往返时间、副本在温度空间的扩散；
   - 按温度（而不是按副本）重新组织的轨迹与估计量；
   - 用 MBAR 跨温度重加权 PIMD 估计量，并说明估计量对温度的显式依赖。

**不做：**

- 不修改 REUS 工作流（`workflows/replica_exchange.py`）。
- 不实现 REST 或溶质回火（可以作为哈密顿阶梯的一种用户自定义情形）。

## 文件所有权

- **独占：** `src/pyrpmd/replica_exchange.py`、新建 `workflows/tempering.py`、
  `examples/tempering/`、相关测试、新建 `docs/tempering.md`（`orphan: true`）。
- **共享注册点：** `input.py`、`input_tasks.py`、`input_schema.py`（`[tempering]`）、
  `runner.py`、`cli.py`。

## 交付物

- 工作流、MPI 执行、分析、英文指南和示例。
- 验证记录 `docs/validation/tempering-<日期>/README.md`。

## 验收标准

- **谐振子 PIMD：** 各温度的动能与势能估计量与解析有限 P 值一致；
  交换接受率与解析分布的预测一致。
- **细致平衡检验：** 在离散的小模型上，副本–温度置换的平稳分布与理论值一致（卡方检验）。
- **双势阱：** PT 采样的势阱占据比收敛到精确的 Boltzmann 比值，而单温度采样明显卡在一个势阱中。
- 中断后续算与不中断逐位一致；4 进程 MPI 结果与串行执行逐位一致。
- 满足通用规则 G5–G10。

## 验证命令

```bash
python -m pytest -q tests -k "replica_exchange or tempering"
mpiexec -n 4 python -m pyrpmd run examples/tempering/double_well.inp --output runs/t37
```
